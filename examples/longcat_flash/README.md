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
