# 有界专家副本

专家副本将热点专家的一部分 token 交给其他 EP rank 执行。owner 保有参数、梯度和优化器状态；
guest 只使用有界的临时执行槽位，计算完成后将梯度归还 owner。
规划使用本次精确整数计数和实际节点拓扑，不依赖离线成本表或拟合模型。

## 使用

`MegaMoeTuningOptions.replica_slots_per_rank` 默认 `0`，保留 owner 路由。
`MegaMoeExperts` 不持有参数；模型传入当前本地专家权重。
正值表示每个 rank 的 guest 槽位上限，不保证每次调用都启用该数量的副本。

下面假设应用已创建32-rank的 `ep_group`，并提供当前输入和 Router 输出：

```python
import torch
from hyper_multicore import MegaMoeExperts
from hyper_multicore.experimental import MegaMoeTuningOptions
from hyper_multicore.replica.residency import mark_replica_weights_materialized

experts = MegaMoeExperts(
    hidden_size=5120,
    intermediate_size=1792,
    num_experts=192,
    top_k=8,
    ep_group=ep_group,
    tuning_options=MegaMoeTuningOptions(replica_slots_per_rank=6, dispatch_mode="push"),
).to(device="npu", dtype=torch.bfloat16)

# 模型已初始化 gate_up_weight [6, 5120, 3584] 和 down_weight [6, 1792, 5120]。
# 在初次物化和后续框架 all-gather 后调用；普通优化器更新由 autograd 版本计数检查。
mark_replica_weights_materialized(gate_up_weight, down_weight)

try:
    output = experts(
        hidden_states, topk_ids, topk_weights,
        expert_weights=(gate_up_weight, down_weight),
    )
    output.float().square().mean().backward()
finally:
    experts.close()
```

`num_experts` 和 `topk_ids` 始终使用逻辑专家编号，参数形状仍只包含本rank的home专家。
应用应在完成使用后关闭执行器，再销毁 process group；初始化与native路径激活见[构建指南](build.md)。

可在 `MegaMoeTuningOptions` 中改为 `dispatch_mode="pull"`；Pull沿用已有暂存限制，不接受Push容量因子。
`tokens_per_expert` 是当前IDs的可信精确histogram，提供后不再逐调用重算比较，不能传入旧路由的计数。

## 准入与容量

- 常规副本使用路由峰值下降门槛，并保留软小副本门槛。训练且两个权重投影均需要梯度时使用25%门槛，
  同时计入梯度搬运代价；其余情况使用40%门槛。这些是当前启发式而非步时保证。
- 为满足实际接收容量所必需的副本走独立规则；不能用普通收益门槛拒绝必要容量修复。
- 所有source/expert配额守恒，实际token重新映射；不丢token、不通过多轮执行越过槽位上限。
- 均衡、收益不足或无法证明安全时保留owner-only执行；合法路由仍须满足硬容量。
- 同计数可以复用计划，不能复用上次token permutation；本次IDs仍独立处理。

B是容量上限，不是任意负载下比B0更快的保证。短序列和部分Pull负载可能没有收益。

## 权重、重算与梯度

正预算副本支持 `EIO` 和连续的 `EOI` 布局，使用调用方的权重布局分配guest矩阵及梯度分带。
布局在第一次绑定时确定，后续调用保持一致。
当前完整设备性能证据来自Ascend910_93；设备支持范围沿用所基于接口分支的构建合同。
可选融合计数仅在支持的A2/A3设备启用，不支持的设备或形状回退到精确Torch计数。

原参数版本由Torch autograd检查；重算按调用ordinal匹配计划，每次仍进行新的全组agreement。
多个deferred forward各自保存反向输入，可逆序backward，不能借用workspace最近一次调用的数据。

外部FSDP等框架在all-gather或替换底层存储后，应使驻留证据失效：

```python
from hyper_multicore.replica.residency import mark_replica_weights_materialized

mark_replica_weights_materialized(gate_up_weight, down_weight)
```

该标记不替代autograd版本检查。没有物化证据的外部权重按保守miss处理。
两路投影的ready、band、completion和ACK都必须完成后才能复用guest存储。

## 默认加速与回退

当前默认保留精确count agreement、host整数原语、路由metadata复用、Push通信队列界、
正常ready queue、梯度band归并和原生能力协商。具备完整证明时，由kernel直接生成最终BF16 owner梯度；
guest贡献和peer累加保持FP32。旧原生库按能力回退，显式要求缺失能力时直接报错。

私有研究开关不是业务构造参数，详见[默认路径与研究控制](design/expert_replica_default_paths.zh-CN.md)。
[执行设计](design/expert_replica_runtime.zh-CN.md)说明协议及生命周期；
[最终梯度设计](design/expert_replica_final_gradient_output.zh-CN.md)说明FP32归并与BF16写出。

## 测试与性能边界

```bash
python scripts/check_repository.py
python -m pytest tests/ut/replica tests/ut/modules/mega_moe tests/ut/ops -q
pytest -q tests/torch/test_expert_replica.py
```

CPU用例验证规划、重算、容量、协议布局和执行选择；设备ST覆盖实际副本传输、空source、
两步更新、checkpoint与逆序deferred backward。设备用例实际占用多卡，按[测试指南](../tests/README.md)运行。
CPU结果不能代替跨节点设备数值、性能或物理HBM验收。
基准形状、统计方法和已有证据范围见[性能验证协议](design/expert_replica_performance.zh-CN.md)。
