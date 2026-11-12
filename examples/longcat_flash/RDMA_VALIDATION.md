# LongCat 融合 SHMEM RDMA

本路径在独立 HyperMulticore 中实现设备侧单边通信，支持 LongCat 真实专家与恒等专家混合路由。
激活 dispatch、结果 combine 和反向 dX 由 MegaMoE 内核中的 AIV 提交，保留按专家就绪推进的任务图。
原生参考使用 NPU permute/unpermute、HCCL All-to-All、GMM 与 fused SwiGLU。

## 配置与复现

在客户已经安装的 PyTorch/torch-npu 和 CANN 环境中，从本分支源码重编译；
不需要 HyperParallel，也不需要安装其他版本的 torch-npu。x86_64 二进制必须在客户环境构建，
不能使用 aarch64 wheel。910B2C 的构建目标为 `ascend910b`：

```bash
source /path/to/cann/set_env.sh
python -m pip install --no-deps --no-build-isolation -e .
python hyper_multicore/_build/build_native.py --soc-list ascend910b --jobs 8
source build/native/payload/hyper_multicore/lib/set_env.bash
export OMP_NUM_THREADS=1
```

数据引擎必须在创建模型、首次获取 HPM SHMEM 之前统一确定；同一 EP 组全部 rank 使用相同配置。
保持 `communication_backend="shmem"`，通过环境变量选择：

| 引擎 | 设备侧数据与通知 |
| --- | --- |
| `mte`，默认 | 全 peer 必须支持 MTE；远端 Add 沿用原协议 |
| `mte_roce` | 按实际 peer 能力选 MTE；缺少 MTE 的 peer 使用 RoCE |
| `roce` | 所有远端 peer 强制使用 RoCE；自身使用 MTE，适合验证 RDMA 实际执行 |

需要 NPU RNIC、RoCE 网络、驱动和锁定 SHMEM SDK 的支持。CPU 架构或 PCIe 字样本身不能说明能否运行。
缺少所需 peer 能力会在融合计划构建时汇合失败。引擎选择不改变 EP 大小与专家所有权。

以下配置保留每卡 8 个真实专家、4 个恒等专家；不缩 tokens、H、I 或 TopK，仅使用单个专家层验收。
外部 EOI Gate/Up 包含恒等专家占位，为 `[12,4096,6144]`；Down 为 `[8,6144,2048]`。

```bash
python -m torch.distributed.run --standalone --nproc-per-node=16 \
  examples/longcat_flash/validate.py --check accuracy --backend pull \
  --communication-backend shmem --shmem-data-engine mte_roce \
  --ep 16 --tokens 2048 --hidden 6144 --intermediate 2048 \
  --real 128 --zero 64 --top-k 12 --swiglu-precision fused \
  --updates 10 --exercise-edge-routes --output ./longcat-rdma-pull
```

将 `pull` 改为 `push` 检查另一分发方向；将 `mte_roce` 改为 `roce` 强制远端经过 RDMA。
每项都应检查完整的 16 个 `rank*.json`，包括输出、dX、路由梯度、两组 dW、更新后参数、
`runtime_ok` 和关闭后的 SHMEM 状态。精度门限与原有原生 GMM 验收相同。
`--operator-timeout` 仅设置诊断进程内算子超时，默认 60 秒。

热专家副本的完整前反向验收可在上述命令加入 `--replica-slots 2 --replica-transport p2p`。
LongCat 在 RDMA 模式下会默认选 p2p 副本 transport；此处显式写出以便核对。
权重预取和 dW 归还经过 p2p，激活、combine 和 dX 保持融合 SHMEM。
MTE 内核副本梯度协议没有移植为 RDMA 原子协议，底层 API 会拒绝该组合。

两个专家层保留前向计算图后联合反传：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=16 \
  examples/longcat_flash/check_hccl_lifecycle.py --communication-backend shmem \
  --shmem-data-engine roce --backend pull \
  --ep 16 --tokens 2048 --hidden 6144 --intermediate 2048 \
  --real 128 --zero 64 --top-k 12 --output ./longcat-rdma-two-layer
```

脚本名保留历史入口，现支持 SHMEM 与 HCCL；SHMEM 案例的活动 Root 大小应为 16。
模型在第一次执行前创建所有专家层，以便对称 heap 统一预留。
两层使用不同权重；加入 `--replica-slots 2 --replica-transport p2p --route-fixture hot`
还会让两层使用不同热点，检查副本权重重新预取、旧层计算图和联合反向。

组件性能采样使用同一固定 fixture 和权重，在独立进程交替执行：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=16 \
  examples/longcat_flash/validate.py --check benchmark --backend pull \
  --shmem-data-engine roce --ep 16 --tokens 2048 --hidden 6144 --intermediate 2048 \
  --real 128 --zero 64 --top-k 12 --swiglu-precision fused \
  --warmup 10 --samples 30 --output ./longcat-rdma-benchmark
```

以 `--backend native` 采样原生参考；每次保留所有 rank 的本地时间与同次 rank-max 时间，
再对 rank-max 样本取中位数。该时间边界是专家组件前向与反向，不含 attention、优化器或完整 Trainer。
当原生 HCCL 与强制 RoCE 使用不同物理链路时，必须在报告中注明，不能当作同拓扑客户加速比。

内部通信/Cube/Vector 区间可将命令改为 `--check profile --warmup 3 --samples 2`。
每个 rank 输出 `trace-rank*.json`；脚本拒绝丢记录或包含解码警告的 profile。
正式组件耗时仍使用 `--check benchmark` 的无 profiler 采样。

## 设备协议与存储

- 每个 peer/QP 固定一个 AIV 提交普通 NBI 与 quiet，不使用不受该后端支持的多 AIV 同 QP 并发提交。
- RoCE 以对端 RNIC 注册区域的基址加对称 heap 内偏移形成远端地址，并检查区域边界。
  P2P 映射属于当前进程，可能与对端注册的虚拟地址不同；不能直接用 MTE 的映射地址提交 RDMA。
  MTE 仍使用 SDK 的 P2P 映射，不改写全局映射表。
- 本地协调 AIV 仅更新本次 generation 和本地启动事件。每个 peer 的 ready、数据、credit、quiet
  和 done 都由其固定 owner 提交，不在中央握手 AIV 与数据 AIV 之间接力同一个 QP。
  一个 owner 覆盖多个 peer 时，先发布所有 ready，再等待对端；退出时同样先发布所有 done，
  避免等待某个 peer 时堵住另一个 peer 所需的通知。
- 每源独立的累计完成计数由不可变的对称常量记录发出，数据与通知在同一 QP 上保持顺序；
  不依赖远端原子 Add。零行任务仍发布任务图所需完成通知。
- dispatch 提交优先于 combine 等待，避免固定通信 AIV 被尚未就绪的 combine 占住而堵塞后续 dispatch。
  计算核按原有事件依赖执行 GMM1/SwiGLU/GMM2。
- 原始 `[T,H]` 激活只暂存一次；行索引描述实际发送顺序。发送端不物化 `[T*K,H]` 前向激活。
  对称 source allocation 同时服务反向 assignment 梯度，物理预留可以大于前向实际写入的原始源。
  接收区、GMM2/dX 回传源、路由结果区、SDK heap 与队列元数据分别占用 HBM。
- pull 读取索引后，MTE 路径显式建立 MTE3→标量依赖，再读取索引；只等待 MTE3→MTE2
  无法保证标量读取已得到新索引。RDMA Get 则在读取索引前完成对应 peer 的 quiet。
- 每次调用保留 source-ready generation；末尾收敛计算核与通信 owner，完成最终 quiet
  和跨 rank 源复用 fence。多层反向所需激活单独保存，不复用仍在计算图中的状态。
- RDMA pull 的专家接收区和回传源属于对称 heap，按全组最大接收负载共同扩容。
  原始源 token 数不变时，集中路由仍可能要求扩大这两个缓冲。计划缓存同时区分源容量与接收容量，
  并保留旧计划供尚未结束的反向使用。内核启动前检查已约定的接收负载，拒绝越界提交。
- Python 侧要求原生适配器提供 `mega_moe_rdma_version() == 2`。此前的单边通信原型不满足这一检查，
  需要从当前源码完整重编译，不能只替换 Python 文件。

自动增长的回归用例可在精度命令中加入 `--initial-token-capacity 128`。
该选项只改变初始预留，不改变每卡 tokens、模型宽度、专家数或 TopK。
多层训练应在首次执行前构造所有层，并用每卡最大 token 数乘 TopK 预留源容量，
以免后续层执行时因其他层仍保留反向计算图而无法重建对称 heap。

## 本地验收记录

2026-11-12 使用独立 HPM、16 个 Ascend910_9382 逻辑 NPU、aarch64、Python 3.11、
PyTorch/torch-npu 2.9.1、CANN 9.1.0 完成以下检查。模型维度保持为上方配置。
原生库编译目标包含 `ascend910_93` 和 `ascend910b`；910B2C/x86_64 尚无实机验收。

| 检查 | 案例 | 结果 |
| --- | --- | --- |
| 普通路由 | 强制 RoCE push/pull，各三步独立 FP32 master 更新 | 16/16 rank 全值一致，更新后的 BF16 参数 hash 一致 |
| 热点与空路由 | push/pull × B0/B2，各十步；热点迁移、空源、纯恒等专家 | 四组均通过，B2 确实执行副本传输 |
| 兼容与扩容 | MTE、mte_roce、初始容量 128 的 RoCE、RoCE 环境中的 HCCL，各三步 | 四组均通过 |
| 两层联合反向 | push/pull × B0/B2；B2 使用不同层权重和不同热点 | 四组均全值一致，活动 Root 为 16，关闭后为 Uninitialized |
| CPU 回归 | 独立仓库 UT | 495 passed，2634 subtests passed |

单层检查合计 928 个 rank-step，输出、dX、路由梯度、两组 dW 的 `max_abs` 与相对 L2 均为 0。
多步更新独立维护两份 FP32 master，再比较更新后 BF16 参数；没有放宽原生 GMM 精度门限。
所有硬件案例都检查完整 rank 报告与关闭状态，未加载 HyperParallel。

本地定位的阻塞包含两项：P2P 映射与对端 RNIC 注册地址相差 1 GiB，导致 ready 在不同地址组间
无法到达；热点 pull 所需接收容量也可能超过旧的对称缓冲。当前分别使用注册地址换算、
按全组最大接收负载扩容，并在启动前检查容量和原生接口版本。
这是本地复现与修复证据，尚不能替代客户机器日志的根因确认。

## 组件性能与重叠

以下为同一节点、固定输入/权重/路由、默认 fused SwiGLU、无副本的专家组件前向加反向。
每个独立进程 warmup 10 次，再保留 30 次同步采样；每次先取 16 个 rank 的最大值，再取中位数。
进程顺序为 native、RoCE pull、RoCE push、MTE pull，再以相反顺序复测。保留全部慢样本。

| 路径 | 第一轮中位数 ms | 反向顺序复测中位数 ms |
| --- | ---: | ---: |
| 原生 NPU GMM + fused SwiGLU + HCCL | 85.489 | 83.408 |
| MegaMoE，SHMEM MTE pull | 75.064 | 75.922 |
| MegaMoE，强制全远端 RoCE pull | 104.371 | 107.146 |
| MegaMoE，强制全远端 RoCE push | 105.791 | 106.083 |

MTE 路径在这两轮分别约为原生的 1.14×、1.10×。
**强制全 RoCE 当前慢于原生，尚未达到性能目标。** 原生 HCCL 使用默认选路，
没有把其物理链路强制为与 RoCE 案例相同，因此不能把此表当作客户 910B2C 的同拓扑加速比。
`mte_roce` 的独立补测中位数为 82.866、87.259 ms；本机所有 peer 都支持 MTE，
该补测实际走 MTE，不能作为真实混合 MTE/RoCE 域的性能验收。

16 个 rank、64 次融合内核调用的内部 profile 没有丢记录。
通信 AIV 的提交/Get-quiet 区间与 Cube 计算区间有交集：各调用交集比例的中位数，
前向约 29.4%，反向约 34.6%；SwiGLU 与 Cube 也有交集。
这证明设备任务图仍在流水执行。上述区间包含提交和等待，不是 RNIC 独占 DMA 时间，
也不能直接当作节省的组件耗时。

另以 baseline/candidate/candidate/baseline 顺序测试了按专家交错 GMM1/GMM2 的前向队列。
普通/B2 三步精度仍全值一致，但 RoCE 与 mte_roce 都没有显示稳定性能收益，交付代码保留原队列。

完整维度、逐 rank 验收摘要、逐进程采样、库 SHA256 和实验边界保存在
[结构化结果](RDMA_RESULTS_20261112.json)。原始运行路径、主机名与网络地址不写入该交付文件。
