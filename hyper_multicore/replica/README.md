# 容量受限的专家热副本

热点专家可以临时复制到其他 EP rank 的 B 个执行槽位。原 rank 继续拥有参数和优化器状态，
副本仅借用权重，反向将梯度以 FP32 归还 owner。每次调用使用当前路由生成计划，
不缓存跨调用的专家权重。B=0 使用原执行路径。

## 使用入口

Native GroupedExperts 的 `ExpertParallel` 接入仍由独立的 Hyper-Parallel 仓库提供。
本仓库提供下述 MegaMoE 副本接口和共享规划器。

MegaMoe push/pull share one planner and support three replica transports:

| `replica_transport` | 权重与梯度执行 |
| --- | --- |
| `p2p` | HCCL P2P 权重复制，按 peer 顺序归还 FP32 梯度 |
| `shmem_signal_kernel_gradient` | 按 peer 并行 SDMA 预取，分投影 ready，反向内核归还 FP32 梯度 |
| `shmem_signal_kernel_gradient_bands` (default) | Cube publishes completed gradient bands for independent AIV return |

```python
from hyper_multicore import MegaMoeExperts

module = MegaMoeExperts(
    hidden_size=5120, intermediate_size=1792,
    num_experts=24, top_k=8, ep_size=4, ep_group=ep_group,
    dispatch_mode="push", replica_slots_per_rank=1,
    replica_transport="shmem_signal_kernel_gradient_bands",
    replica_planner="cpu",
)
output = module(values, topk_ids, topk_weights)
output.backward(gradient)
module.close()
```

由调用者提供参数时，设置 `create_parameters=False`，每次 forward 传入
`expert_weights=(gate_up_weight, down_weight)`。本地矩阵布局分别为 `[H, hidden, 2*intermediate]`
和 `[H, intermediate, hidden]`，H 是每个 rank 拥有的逻辑专家数。副本不注册为参数。
Multicore 的构建、载荷激活和 SHMEM 生命周期见
[构建说明](../../docs/build.md)和 [MegaMoe 入口](../../README.md)。

## 规划、容量与收益准入

逻辑专家数 E 必须能被 EP rank 数 P 整除，H=E/P。每个 rank 固定 H 个 home 槽位和 B 个 guest
槽位；物理专家数为 P*(H+B)，空槽位标记为 -1。合法路由要求每个 token 的 TopK 专家互不重复，
且满足 1<=K<=E。路由错误在稀疏传输之前由所有 rank 一致拒绝。

规划器汇总各 source 的整数计数，生成确定性的放置和配额。稳定重映射保留每个 TopK 位置、
源顺序和总行数，既不丢 token，也不执行 multi-wave。`replica_target_load` 是期望接收负载，
独立于缓冲区容量；push 未设置时使用当前容量，pull 未设置时使用平均负载。

`ExpertReplicaConfig.maximum_receive_rows(S, K)` 给出有界接收容量。B=0 的未对齐上界为
P*S*min(K,H)，B>=H 时为 S*K；其余预算按 home/guest 混合界和整数截断余量计算。
push 按实际路由需求增长并受此界约束，pull 继续使用路由 source 布局。
`initial_capacity_factor`、`capacity_growth_factor` 仅用于 push，不能作为丢弃阈值。

常用选项：

| 参数 | 约束与行为 |
| --- | --- |
| `replica_slots_per_rank` | 非负整数 B；每个 EP rank 使用相同预算 |
| `replica_planner` | `cpu` 或 `device`；默认 `cpu` |
| `replica_min_rows` | 小副本软阈值；Native 默认 0，MegaMoe 默认 1024；容量必需副本继续保留 |
| `replica_target_load` | 可选正整数；所有 EP rank 一致 |
| `replica_minimum_load_reduction` | `None` 要求当前路由峰值下降至少 40%；`0.0` 关闭可选副本收益门槛 |

收益门槛只控制可选副本。若理想平均负载也不能达到门槛，规划器直接使用 home 计划；
求解后再次检查最终峰值，容量必需的复制继续保留。低层规划 API 的
`minimum_load_reduction` 和 `minimum_saved_rows` 默认均为零，分别表示相对与绝对收益门槛。
空副本计划使用紧凑 home 拓扑，下一次调用可以重新启用副本。

## 传输与调用生命周期

`p2p` 在使用 guest 权重之前等待发送/接收完成。guest 梯度按 peer 顺序加到 FP32 owner
缓冲区，通信存储受 B 槽位限制；没有 remote 工作的 rank 仍参与一致的调用顺序。

`shmem_signal_kernel_gradient` 在一个独占 provider lease 内执行以下协议：

1. receiver 在上一轮消费者完成后发放槽位 credit；owner 完成 credit 等待后启动 SDMA。
2. forward 先复制 W13 再复制 W2，backward 反序复制；每个投影分别发布 ready。
3. GMM 仅等待所消费投影的 ready，允许 home 计算与 guest 复制重叠。
4. 反向等待完整投影生产 fence，归还 W2/W13 FP32 梯度，并等待所有 remote reader 的 ACK。
5. 所有复制流 join 到调用流后才释放 lease。epoch 回卷前由所有 rank 同步清空信号。

缺少完整投影 fence 的调度使用内部 SDMA 梯度回退；它不额外暴露传输模式。
同一调用的 backward 可以复用 forward 的权重代际，但必须先通过 autograd 参数版本校验，
且该代际仍在原槽位中。其他 forward、provider/heap 重建和 epoch 回卷都会使复用失效。
延迟 backward 持有自己的路由、参数版本和中间张量，不能读取已被下一层覆盖的 workspace。

共享 execution resources 需要相同拓扑、传输、shape、dtype 和 tile splits。串行 lease 与事件
约束跨流重用；同一个 provider 不允许并发 host lease。关闭模块之前应完成或释放所有保留的图。
底层 `return_gradients(..., consume=True)` 仅接受调用独占的 detached FP32 owner 缓冲区，
并要求调用者停止使用原值；普通借用输入通过复制保留。

## 设备规划器与融合计数

设置 `replica_planner="device"` 可使用可选 Ascend 910B 求解器。它保留与 CPU 相同的整数策略、
准入条件、容量约束和配额守恒；路由配额留在设备，host 仅复制一次紧凑控制摘要。
因此不应将该入口解释为全路径零 readback。设备规划器不支持离线成本微调。

激活匹配的 CANN 环境后，在仓库根目录构建可选库：

```bash
cmake -S hyper_multicore/replica/_device_kernel -B build/replica-planner
cmake --build build/replica-planner -j 4
cmake --install build/replica-planner \
  --prefix "$PWD/build/native/payload/hyper_multicore/lib/replica"
```

规划器 ABI 为 4；ABI 不匹配明确报错。未安装规划器时，显式 device 入口报错。
融合计数库为可选加速：不支持的 shape/device 或未安装时使用 Torch 整数计数，
重复专家和越界标记随计数一起参加 collective。Native 接收已有专家计数，不重建 TopK。

Count ABI 2 supports 1 to 512 logical experts, including E384. Rebuild and install the optional count library
to enable this path. Count ABI 1 remains supported for up to 256 experts; larger shapes retain the Torch fallback.
This extends the count kernel independently of the device quota planner.

910B/910C 标准构建会将可选设备库安装到 native payload 的 `lib/replica`。950 构建不会使用这些 910 设备库。

## 离线成本模型

`ExpertReplicaCostModel` 在容量规划之后微调已有副本配额。模型记录 shape、EP、backend、
transport、schedule 和 FP32 partial dtype，拒绝配置不匹配或旧 schema。校准范围之外保留
原策略；没有实测依据时不扣除假设的 home overlap。两轮确定性搜索保留 B 预算和 source 守恒。
模型的 rank score 缓存只存在于一次求解内，不代表权重驻留缓存。

离线实测数据由调用方提供；所有 EP rank 应加载同一个文件，再传入
`ExpertReplicaCostModel(**payload["model"])`。模型的预测需要用完整步骤基准验证。

## 验证与基准

CPU 回归验证计划守恒、准入、路由、transport 协议、生命周期及调度/ABI：

```bash
python3 -m pytest -q tests/ut/a5 tests/ut/modules/mega_moe
```

NPU 组件和完整训练验收需要在目标设备运行；迁移后的 PanGu 提供独立仓路径预检与性能脚本。
对照应保持 route、seed、shape、checkpoint 和资源所有权一致，使用新进程 A/B/B/A；
诊断或单算子耗时不能替代完整步骤收益，CPU 回归和 build/load 也不能代替 NPU 数值验收。

## 按行带重叠梯度返回

`shmem_signal_kernel_gradient_bands` 使用 Cube 直接发布行带就绪信号，
使 W2/W13 梯度可以在整专家计算结束之前返回。启用此模式前需要重新构建 multicore 原生载荷。
行带按 FP32 矩阵形状和 4 MiB 目标大小划分，不根据路由 skew、seed 或副本预算选择策略。
每个 Cube 只在该行带全部 K 维累加与最终 GM 写回完成后发布信号；
没有分配到输出 tile 的 Cube 和零行专家也必须发布完成信号。

返回侧按实际 worker 数和行带数划分最多四个独立 AIV 组。
一组等待某个 peer 时，其他组可以推进自己的行带。
每个元素仍按原目标 rank 顺序累加 FP32 梯度，不改变路由、副本配额和数值精度。
两个投影均完成所有读取后，在最终 ACK 汇合处释放 guest 租约。

对称的 guest 信号表直接放入已有 replica inbox，纳入构造阶段的 SHMEM 堆预算，
随 inbox 一起释放，在 epoch 回绕时同步清零；普通 owner 信号表保持调用内独占并记录 stream 生命周期。
无法证明两个投影的生产调度、行带数超过 64 或 Cube 数超过预留的 32 时，保留完整专家返回。
B=0 和没有实际副本搬运的路由沿用原有执行路径。

正的槽位上限默认启用自适应，不需要传成本模型或 adaptive 开关。成本模型需匹配当前执行路径和校准来源。

## Optional native host primitives

The CPU planner can use a small CPython extension for integer-matrix validation
and ordered calibrated compute accumulation. It retains the existing placement,
capacity, topology, copy/gradient costs, candidate order and admission policy.
Unsupported Python integer ranges, dictionary subclasses and nonzero integer
time knots keep the reference path. The extension has no
Torch, CANN, SHMEM or NPU link dependency. Editable installs and wheel builds compile this extension. A source checkout without
the extension retains the Python reference implementation.

For critical-path scoring, the native primitive prepares home compute sums and
ordered guest durations from immutable knots and quotas. Token row accounting
remains in Python. The reference retains all dependency scheduling, communication
resource ordering, gradient releases and ACK semantics. ABI version 3 is
required for this additional primitive; model subclasses retain their custom
Python interpolation contract.

To build it separately, use the same Python interpreter used for training:

```bash
cmake -S hyper_multicore/replica/_host_kernel \
  -B build/replica-host -DPython3_EXECUTABLE="$(command -v python)" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/replica-host --parallel 4
cmake --install build/replica-host \
  --prefix "$PWD/hyper_multicore/replica/_host_kernel"
```

The optional binary is detected once per process. Missing binaries preserve the
Python implementation. An installed binary with an incompatible ABI fails at
load time. The internal regression switch
`HYPER_PARALLEL_REPLICA_HOST_NATIVE=0` selects the reference in a fresh process;
no model or adaptive constructor argument is added.

The wheel builds the CPython extension from the same source as part of packaging. Float contraction and fast-math are disabled to
preserve the reference interpolation and accumulation order. CPU differential
tests compile the actual source and require exact plans and FP64 score bits:

```bash
python -m pytest -q tests/ut/a5/test_replica_host.py
```

## Experimental recompute metadata reuse

The internal fresh-process switch `HYPER_PARALLEL_REPLICA_METADATA_REUSE=1`
retains invocation-owned integer dispatch and source-run descriptors after the
existing exact CPU plan replay guard succeeds. Current token IDs still rebuild
their mapping and permutation. Counts readback, validation, candidate selection
and the count/header wire protocol remain unchanged.

Compact home-only decisions retain an owned logical count matrix instead of
creating guest dispatch metadata. Upload-ready events and consumer stream
records protect cross-stream use. Clearing the invocation registry drops only
cache references; saved tensors and allocator stream tracking retain active
consumers. B0 and non-checkpoint execution do not create this cache.

The switch defaults to disabled pending accelerator numerical and lifecycle
validation. CPU tests cover exact hits, changed counts/policy, current-ID remap,
early-stop cleanup, later retained-graph replay and mocked stream ordering:

```bash
python -m pytest -q tests/ut/a5/test_replica_metadata_replay.py
```
