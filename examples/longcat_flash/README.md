# LongCat Flash 独立接入和验证

本目录的脚本只依赖 HyperMulticore、PyTorch、torch-npu、CANN 和驱动；不导入
HyperParallel 或 Transformers。生产接口在 `hyper_multicore.longcat`，由调用方保留
路由器、专家参数、优化器和 EP 进程组。

## 接入已有模型

调用方负责按模型原有逻辑计算 TopK 专家 ID 和路由权重，并在训练结束、最后一次反向执行后关闭资源：

```python
from hyper_multicore.longcat import LongcatFlashExperts

# 在 EP 进程组创建后构造一次；所有 EP rank 的配置和调用次序须一致。
experts = LongcatFlashExperts(
    hidden_size=6144,
    intermediate_size=2048,
    num_experts=32,          # EP4 示例：每卡 8 个真实专家
    zero_experts=16,         # EP4 示例：每卡对应 4 个恒等专家
    top_k=12,
    ep_size=4,
    ep_group=ep_group,
    gate_up_padded=True,     # 如果 gate_up 包含每卡 4 行零专家占位
    swiglu_precision="fused",
    routing_backend="torch",
    replica_slots_per_rank=0,
    initial_token_capacity=2048 * 12,  # 每卡最大 token 数乘 TopK；在首次执行前预留
)

# hidden 可以有多个 token 维；ID/权重先展平为 [token数, 12]。
# 原生模型持有并更新两个 EOI 参数，不能在 forward 里重新创建。
output = experts(
    hidden,
    expert_ids.reshape(-1, 12),
    router_weights.reshape(-1, 12),
    expert_weights=(gate_up, down),
)

# 训练结束且所有相关 backward 已完成后，在各 rank 调用：
experts.close()
```

参数采用 EOI，即 `gate_up` 为
`[本卡真实专家数 + 可选零专家占位数, 2 * intermediate_size, hidden_size]`，
`down` 为 `[本卡真实专家数, hidden_size, intermediate_size]`。将全局真实专家 ID
排在 `[0, num_experts)`，零专家 ID 排在
`[num_experts, num_experts + zero_experts)`。路由权重是 FP32，输入和专家参数为
BF16；权重应与相应 expert ID 配对，不能丢弃或重新归一化零专家的权重。
`gate_up_padded=False` 表示不含占位行；有占位时设为 `True`。
开启副本时会保留外部参数所有权并只对真实专家启用副本，使用
`replica_slots_per_rank=2` 可设置每卡两个副本槽位。

默认 `swiglu_precision="fused"` 与原生参考路径的
`torch_npu.npu_swiglu` 对齐；MegaMoE 在融合 kernel 中调用 CANN 对应的
SwiGLU 实现，并非通过 Python 再调用一次 `torch_npu.npu_swiglu`。
`swiglu_precision="hf"` 保留 HF 的 BF16 中间舍入语义，诊断客户原模型精度时应与
模型当前语义一致。二者数值可能存在 BF16 尾数差异，不能混用作为严格对照。

默认 `routing_backend="torch"` 不要求 Triton。可用的 Ascend Triton 环境可选
`"triton"`，须在目标设备和版本上单独验证。EP 实际规模及每卡权重应与传入配置一致。

## MTE 不互通的 EP 组与 910B2C 16 卡

SHMEM 是通信接口，具体能否跨设备访问取决于传输引擎和实际拓扑。
默认 MTE 模式同时用 MTE 搬运数据、更新远端 Add 计数器；PCIe peer access
或非空的对称虚拟地址不代表这些操作可达。上游 SHMEM 的
[使用限制](https://github.com/hicann/shmem/blob/master/docs/debug/Troubleshooting_FAQs.md)
说明 A2 的两个 8P HCCS 域之间不能直接使用 MTE。判断应以实际探测为准，不能只根据 CPU 架构或卡数。

默认 `communication_backend="shmem"`、MTE 引擎会在构建首个融合计划前读取 SDK 的 MTE
可达性，并把本 rank 的失败汇合到 EP 组。缺少任何 peer 会报出不可达的 Root PE，
避免提交必定无法完成的远端 MTE 操作。这个检查需要重新编译当前源码的原生绑定；
旧二进制缺少探测接口时会明确要求重编译。

先按照下方构建章节，激活客户自己的 CANN 环境并 source 当前编译产物的
`build/native/payload/hyper_multicore/lib/set_env.bash`。随后可检查真实拓扑；
此命令只初始化、读取可达性并关闭资源，不提交远端搬运：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=16 \
  examples/longcat_flash/check_transport.py --output ./transport-probe
```

检查每个 `rank*.json` 的 `mte_reachable` 和 `all_peers_mte_reachable`。
`guard_injection_pass` 只说明模拟缺失 MTE peer 时的保护生效，不能作为 PCIe 实测通过。
`--exercise both` 可在全部 peer 确认可达后检查真实 MTE/Host SDMA Put 和 Add 完成通知；
脚本不会在缺失 MTE 链路时强行提交这些操作。

### 融合 SHMEM RoCE 路径

在创建模型、初始化 HPM SHMEM 之前选择进程内统一的引擎：

```bash
export HYPER_MULTICORE_SHMEM_DATA_ENGINE=mte_roce
```

保留 `communication_backend="shmem"`。`mte_roce` 根据 SDK 的真实能力选择：
可达 peer 使用 MTE，其余 peer 使用设备侧 RoCE；不会写死逻辑 rank 的前后各 8 卡。
`roce` 则强制所有远端 peer 使用 RoCE，便于在同时具备 MTE 的机器上验证真实 RDMA 路径；
本卡仍使用 MTE。所需 peer 缺少可用能力时，在融合内核启动前汇合报错。

dispatch、combine、反向 dX 均保留在 MegaMoE 设备任务图内。每个 peer 固定一个 AIV
提交 RDMA，使用每源独立的累积完成通知替代远端 Add；GMM1 按专家就绪依赖启动，
SwiGLU、GMM2 和结果回传继续沿任务图推进。完成与源缓冲区复用仍包含最终 quiet 和跨 rank fence。
前向只暂存原始 `[T,H]` 激活和行索引，不生成发送端 `[T*K,H]` 展开激活。
反向 assignment 梯度、接收区、回传区和对称 heap 预留需分别核算。

LongCat 启用热专家副本时，RDMA 模式默认使用 `replica_transport="p2p"`
交换副本权重和归还 dW；激活与 dX 保持融合 SHMEM 通信。底层 `MegaMoeExperts`
需显式指定该副本 transport，当前不支持将 MTE 内核中的副本梯度协议直接切到 RDMA。
EOI 权重与默认 fused SwiGLU 均保持原配置。

完整原宽命令、协议边界及验收结果见 [融合 RDMA 验证](RDMA_VALIDATION.md)。
生产代码不依赖 HyperParallel。客户 x86_64、910B2C 需要在本机编译 `ascend910b`，
并验证自己的 RNIC、网络和驱动；本机 910C 强制 RoCE 的通过结果不能替代客户拓扑验收。

### 外层 HCCL 兼容路径

如需单独验证跨 MTE 域 EP16 的通信兼容性，可在构造 `LongcatFlashExperts` 或
`MegaMoeExperts` 时显式加入：

```python
communication_backend="hccl",
replica_slots_per_rank=0,
```

此时全局 EP 仍为 16，专家所有权、每卡 8 个真实专家、模型宽度、TopK 和外部 EOI
权重均保持原配置。计数交换、dispatch/combine 和对应梯度走 HCCL；每卡使用一个
单 PE 的 SHMEM runtime 执行本地 MegaMoE GMM/SwiGLU/dW 融合内核，内核里的搬运和
信号仅访问自身。原生 GMM 参考路径仍独立用于验收。

HCCL 路径目前不支持热专家副本，启用副本会在构造时明确报错。一个进程内的
MegaMoE 层应统一选择通信 backend；不能同时保留需要跨 PE SHMEM 的执行器和
HCCL 路径的单 PE runtime。该路径拆开了跨卡通信与本地计算，其性能需在目标拓扑
重新测量，不能套用全 MTE 路径的通信计算重叠收益。当前实现先完成外层 dispatch
All-to-All，随后进入本地融合计算，再执行 combine All-to-All，因此没有保留按专家
就绪推进的跨卡通信计算流水。这里的开关用于兼容性验证，不作为目标拓扑的性能优化结论。

面向跨 MTE 域的融合性能路径使用上述 SHMEM RoCE 模式。
[最初的单边 RDMA 原型](one_sided_probe/README.md)保留传输机制实验，结果与集成后的模型验收分开记录。

在客户已有 PyTorch/torch-npu/CANN 环境中从本分支重编译后，运行完整 EP16 原宽验收：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=16 \
  examples/longcat_flash/validate.py --check accuracy --backend push \
  --communication-backend hccl --ep 16 --tokens 2048 \
  --hidden 6144 --intermediate 2048 --real 128 --zero 64 --top-k 12 \
  --swiglu-precision fused --routing-backend torch --updates 10 \
  --output ./longcat-hccl-accuracy
```

需要额外覆盖热点 owner、空源和全恒等专家时加入 `--exercise-edge-routes`。
构造完整模型后，HCCL 路径会在第一次计算前登记当前 EP 组所有已创建层的本地执行器，
使 Heap 预留包含后续层，避免前一层仍持有反向状态时才为后一层追加预留。
串行层仍可在首次执行前显式共享执行资源；`initial_token_capacity` 在 HCCL 模式下
也传递给本地计算执行器，用于预留本卡接收的真实 assignment 行数。

两个专家层前向保留计算图后一起反传的检查，可用以下命令执行：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=16 \
  examples/longcat_flash/check_hccl_lifecycle.py --communication-backend hccl \
  --ep 16 --tokens 2048 --hidden 6144 --intermediate 2048 \
  --real 128 --zero 64 --top-k 12 --output ./longcat-hccl-two-layer
```

报告中的 `last_execution_plan.compute_ep_size=1` 描述每卡本地计算 kernel 的坐标，
不表示将 EP16 工作负载缩成单卡模型；报告 `shape.ep` 应为 16。

当前已完成的硬件、数值与资源释放检查见 [通信验证记录](TRANSPORT_VALIDATION.md)。

## 单机原宽精度验证

先使用当前机器已有的 PyTorch/torch-npu 和 CANN 编译 HPM，参考
[构建说明](../../docs/build.md)。例如 910B2C 选 `--soc-list ascend910b`；编译时
不要替换或安装客户自编译的 torch-npu。激活 CANN 和编译产物后，在仓库根目录运行：

```bash
source /path/to/cann/set_env.sh
source build/native/payload/hyper_multicore/lib/set_env.bash
export OMP_NUM_THREADS=1
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  examples/longcat_flash/validate.py --check accuracy --backend push \
  --ep 4 --tokens 2048 --hidden 6144 --intermediate 2048 \
  --real 32 --zero 16 --top-k 12 --swiglu-precision fused \
  --routing-backend torch --output ./longcat-accuracy-b0
```

默认裁剪保留每卡 8 个真实和 4 个零专家，隐藏层和专家中间层宽度不变。
EP16 使用 `--ep 16 --real 128 --zero 64` 和 16 个本机进程。验证脚本目前限定
单机，跨机训练需由客户项目建立 EP 进程组。

脚本对照原生 NPU `permute → All-to-All → GMM → npu_swiglu → GMM →
All-to-All → unpermute`，检查输出、dX、路由权重梯度及两组专家权重梯度。
检查每个 rank 的 `rank*.json` 中 `result.strict_pass`、每项误差、
`environment` 和 `shmem_after_close`；所有进程退出为零才表示通过。
测试范围是专家组件，不代表完整模型、完整训练或客户真实权重的精度已通过。

热专家副本持续更新检查使用相同尺寸，并覆盖热点、空源和全零专家路由：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  examples/longcat_flash/validate.py --check accuracy --backend push \
  --ep 4 --tokens 2048 --real 32 --zero 16 --top-k 12 \
  --replica-slots 2 --updates 10 --output ./longcat-accuracy-b2
```

此检查使用独立 FP32 SGD master 更新，除了逐项精度还验证两边更新后参数一致，
并要求至少一次真实副本迁移。每次更新的详细记录写在各 rank 报告中。

## 组件性能验证

在相同环境和完全空闲的设备上分别启动原生和 MegaMoE 的新进程：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  examples/longcat_flash/validate.py --check benchmark --backend native \
  --routing-backend torch --warmup 10 --samples 30 --output ./longcat-bench-native

python -m torch.distributed.run --standalone --nproc-per-node=4 \
  examples/longcat_flash/validate.py --check benchmark --backend push \
  --routing-backend torch --warmup 10 --samples 30 --output ./longcat-bench-push
```

两个命令的尺寸、SwiGLU 精度和路由后端须一致；先对照 `identities` 哈希，再比较
`result.rank_max_ms`。建议在新进程中交错重复 native/push/push/native 和
push/native/native/push，保留全部样本。报告中包含本机路径、软件版本和设备名称，
向外发送前由用户检查。计时仅包含专家组件前向加反向，不含 router projection、
优化器或全模型其他部分。

Push 的可选通信任务裁剪可通过所有 rank 一致设置
`HYPER_MULTICORE_MOE_COMPACT_COMMUNICATION=1` 进行实验。对照实验应使用同一批
输入并分别运行开关开启和关闭的进程；查看报告中的
`compact_communication_requested`、`last_execution_plan` 和对应任务数量，
确认实际执行计划是否发生变化。该开关的性能收益需在目标设备及实际路由上测量。

使用 `--route-fixture hot` 可以构造热点负载：所有 rank 的每个 token 都选择第一个
owner 的 8 个真实专家及 4 个零专家，权重沿用确定性生成的数据。这是合成压力场景，
应同时保留默认的 `router` 场景；比较 B0/B2 时保持其余参数和种子一致，并检查
`result.replica_plans` 中是否实际出现迁移。对 B2 增加 `--replica-slots 2`。

可单独检查当前安装制品的副本计数 kernel 是否覆盖边界和非法路由：

```bash
python examples/longcat_flash/check_count_kernel.py --output ./longcat-count-check.json
```

该命令只测试整数计数算子，改变专家数和行数以覆盖计数边界，不用于替代上述原宽
LongCat 专家组件精度或性能验证。

## HF 原始实现数值诊断（可选）

如果项目已安装包含 `LongcatFlashExperts` 的 Transformers，可用实际安装的 HF 类分别对照
原生 GMM 和 MegaMoE 的两种 SwiGLU 模式；它不依赖 HyperParallel，也不会安装或替换
Torch、torch-npu 或 Transformers：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=1 \
  examples/longcat_flash/diagnose_hf.py --ep 1 --tokens 257 \
  --real 8 --zero 4 --top-k 12 --routing-backend torch \
  --output ./longcat-hf-diagnosis
```

该诊断维持原宽 H6144/I2048、每卡 8 个真实专家及 4 个零专家。
报告分别保存 `native_GMM_hf_vs_HF`、`native_GMM_fused_vs_HF` 及对应的
`Mega_vs_native_GMM` 输出和梯度误差。HF 逐专家与 GMM 的差异作为诊断项保留；
重复 HF 的一致性以及相同 SwiGLU 语义下的 MegaMoE/GMM 对照是退出状态的主验收项。
报告中的 `primary_status` 和 `hf_strict_status` 分别表示主验收及 HF 严格对照结果，
前者通过并不意味着后者通过。
这些合成输入不能取代客户真实权重、路由和完整训练的排查。
