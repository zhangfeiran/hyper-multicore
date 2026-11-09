# RFC：独立无表专家副本执行闭环

状态：第一版组件及 EP32 扫描已完成并观察到时延回退；本轮恢复非成本默认路径，
CPU434项及2433个子例、EIO设备72条rank/case及新EP32完整复测已完成；
按 HyperParallel `003-RFC.yml` 的必填字段组织。本文对应独立仓
`feat/expert-replica-heuristic`，与[迁移方案](expert_replica_migration.zh-CN.md)共同评审。

## 动机（Motivation）

原特性分支把启发式、成本拟合、校准、实验搜索、框架 hooks 和设备协议放在同一条差分中。
仅关闭成本开关仍然依赖其类型、文件和运行签名，也难以审阅训练正确性。
本次将可训练的有界副本收敛到独立组件；框架只提供原参数、调用上下文与物化证据。

## 目标设计

owner 保有参数、梯度及优化器。guest 只借用每个 EP group 共享的有界执行存储，
不注册新参数。规划器只使用精确整数计数、真实节点拓扑和固定无表准入规则。
实际路由保留逐 source/expert 配额并重映射本次 token；重复计数不能充当 token permutation 缓存。

```mermaid
sequenceDiagram
    participant F as 模型和 checkpoint
    participant R as 路由与无表 planner
    participant P as 共享副本池
    participant K as MegaMoE 或 Native
    F->>R: 当前 IDs / histogram / group 控制信息
    R->>R: 全组校验、容量重检、精确计划复用或求解
    R-->>F: 不可变计划和当前 physical IDs
    F->>P: 申请 lease、当前权重代次
    P->>K: owner/guest 分离指针、projection ready
    K->>K: forward、owned backward state、FP32 partials
    K->>P: W2/W13 bands 或整投影完成
    P->>P: 固定 peer 顺序归还、完整 quorum、ACK
    P-->>F: 原 owner 梯度、释放 lease
```

### 模块边界

| 模块 | 职责 |
| --- | --- |
| `replica/routing.py` | 一次精确 count agreement、错误协调、硬容量和当前 ID remap |
| `replica/recompute.py` | invocation registry；原 forward 与重算共享；包括 home bypass ordinal |
| `replica/native.py`、`communication.py` | Native P2P dispatch/combine、grouped experts、可微 all-to-all |
| `replica/pool.py`、`transport.py` | 有界普通池、互斥 lease、stream event、FP32 peer 顺序归还 |
| `replica/residency.py` | owner 授权的驻留证据、物化失效、slot 全局写入历史 |
| `replica/signal_transport.py` | credit、projection ready、epoch、ACK、saved generation |
| `modules/mega_moe/` | B0/owner/guest 执行、owned backward state、Heap 事务、静态任务裁剪 |
| `shmem/_api.py`、`shmem/ccsrc/` | 映射 peer 的 SDMA put/get；默认 RMA 签名保持兼容 |
| `ops/runtime/` 与锁定 GMM patch | complete-K band 发布、FIX fence、归还和释放 reader 的完成边界 |

初版业务构造器只新增 `replica_slots_per_rank`。启发式沿用默认 40% receive-peak 降幅门槛和
1024 行软小副本门槛；容量必需副本保留其独立规则。实际 Heap 上限由当前分配器给出。
B 是上限，均衡或无足够收益的调用可执行 owner-only 计划。

## 需要设计评审 DFX 建议

### 正确性与故障语义

- 任何 rank 的非法 ID、重复 TopK、错误计数、超大尺寸或 autograd 参与不一致，必须在稀疏传输前拒绝。
  Router 提供的 histogram 是可信输入合同；其 shape、类型、总量和逐 expert 上界仍参与检查。
- 重算每次仍执行新的全组 agreement。只有逐 source/expert 完全相同、group/rank/policy 相同且符合
  当前硬容量的计划才能复用；F/R ordinal 记录也包括均衡旁路。
- forward/backward 保存 owner 参数，让 Torch 检查版本。多个 deferred forward 的激活与 route 各自拥有，
  允许逆序 backward；不把 graph 绑定到 workspace 的最近一次调用。
- 权重 copy stream 只发布当前 projection ready。驻留命中仍走当前 credit/ready/ACK；
  receiver 不独立决定跳过 payload。失败、provider/Heap 重建、epoch rollover 都使缓存失效。
- 两路投影分别预留 epoch 与 completion storage。bands 必须等完整 K 的最终 GM store fence；
  空 producer 也发布贡献。归还维持固定 peer 顺序，并在所有消费者结束后发 ACK。
- 不能证明 bands 可用时，两路投影一致回退到整投影协议；不能只让某一投影进入不匹配的协议。
- 外部参数没有物化证据时普通驻留保守 miss。框架在 all-gather 或 `.data` 更新后调用
  `mark_replica_weights_materialized`；该标记不替代 Torch autograd version。

### 兼容性与影响范围

保留目标基线的 EIO/EOI 参数布局及 B0 缓存；正预算 EOI 仍明确拒绝。
恢复默认auto调度时，跨节点保持来源的auto_w1选择。EOI owner保留原native group-list路径；
EOI累积检查在本轮前冻结包也失败，单独记录，不宣称本轮EOI设备验收通过。
Pull 的暂存字节限制、B0 route/header cache、CPU 可导入及原独立构建/loader 边界继续保留。
事件尾部统一增加完整 expert producer fence 的预留；较大 expert 配置的事件数组会增加少量槽位，
测试需按新的容量计算，不能把序列化字节与旧版完全一致作为兼容条件。

独立仓不导入 HyperParallel。下游桥接把 `ExpertParallel`、GroupedExperts、checkpoint registry 和
完整参数物化 hooks 指向独立实现；框架 mesh、并行切分与参数所有权继续由下游维护。
原框架中的私有历史 multicore 模块不属于新的独立 API。

SHMEM 的低层 `put/get(..., use_sdma=True)` 在当前 NPU stream 上复制到直接映射的 peer，
由 CANN 地址映射失败时返回错误。默认 `put/get` 保持原有 RMA 调用签名。
权重及 ready publication 使用这条 SDMA 路径，梯度的内部回退也需要对应的 get 支持；
必须同时迁移 Python API、pybind 参数和 CANN host stream copy，不能只迁移调用方。

### 最小性与替代方案

| 选择 | 取舍 |
| --- | --- |
| 直接合并来源分支 | 携带成本/校准、device planner、研究开关及非副本调优，差分难以审阅 |
| 只保留 planner | 可做 CPU 差分，但缺少可训练执行、参数代次和远端 reader 完成语义 |
| 全部改成普通 P2P | 保留为 Native 基线，不能覆盖 MegaMoE 的投影重叠和 native bands |
| 当前选择 | CPU 精确无表 planner + Native P2P + MegaMoE 默认 bands；共用 route 和生命周期合同 |

不引入成本表、自动校准或独立 device planner。恢复非成本 host/count 加速、默认 auto
调度、可选 D2 descriptor cache、研究梯度队列及必要 observer，详见
[默认路径 RFC](expert_replica_default_paths.zh-CN.md)。
首版默认矩阵存储为共享的 B 上限；按需 prefix 的存储布局和增长接口保留边界测试，
默认仍共享 B 上限存储；B>1 且未显式设置 Heap 时，可用研究开关启用按需 inbox。
该控制需独立验证预算重规划、全组增长事务和活动 lease 拒绝增长。

### 性能影响和验收边界

无表准入是负载启发式，不能保证任意形状或任意 B 都比 B0 快。
CPU 精确差分验证策略保持；mock/host harness 验证控制流；当前 native payload 的两卡 ST 验证
实际数值和完成协议。迁移性能对照固定为EP32、E192、H5120、I1792、TopK8、BF16、4096 token/rank，
比较旧分支成本关闭的B1与精简B1，扫描均衡至热点。该矩阵已完成，见
[EP32性能RFC](expert_replica_performance.zh-CN.md)：49点F+B相对变化中位数Push +1.23%、Pull +1.08%，
首版所有计划一致但存在回退；恢复默认路径后，Push/Pull逐点变化中位数为+0.034%/-0.167%，
实际调度及98点计划一致，本矩阵未再观察到首版的整体回退。
完整Trainer及相同后端B0收益仍需各自验收，数值ST不代替性能结论。

## 相关的 RFCs 和 API

```python
from hyper_multicore import MegaMoeExperts
from hyper_multicore.replica.recompute import replica_checkpoint
from hyper_multicore.replica.residency import mark_replica_weights_materialized

experts = MegaMoeExperts(
    hidden_size=512, intermediate_size=512, num_experts=8, top_k=2,
    ep_group=ep_group, replica_slots_per_rank=1,
)
output = replica_checkpoint(experts, hidden, topk_ids, probabilities)
# 外部框架完成一次完整参数物化后：
mark_replica_weights_materialized(gate_up_weight, down_weight)
```

`replica_checkpoint` 默认非重入，也支持显式 `use_reentrant=True`。
已有框架 wrapper 可通过 `create_recompute_contexts` 组合自己的 context，而不重复持有另一份 registry。
Native EP 通过 `dispatch_native_replicas`、`native_replica_experts` 和 `combine_native_replicas` 组成闭环。

验收入口：

```bash
pytest -q tests/ut
pytest -q tests/torch/test_expert_replica.py
python scripts/check_repository.py
```

两 rank 的 Push 热副本用例显式使用已有的 `initial_capacity_factor=1.0`，保持 40% 准入门槛，
确保测试输入确实可准入。ragged 长度为 1024/769；空 source 的 Push 允许 owner-only 回退。
设备 ST 覆盖 Push/Pull、Native P2P、B0、热/均衡、ragged/空 source、两种 checkpoint、
两次更新及逆序 deferred backward。各 rank 分别保存终态、实际 transfer、梯度投影和最大误差。
完整实验状态记录在交付 REPORT；没有终态证据的项目不能按通过处理。

2026-11-09 的匹配构建验收：374 项 CPU UT、2098 个子例，桥接 105 项 UT，native host ctest 7/7；
双卡 Push 7 项、Pull 7 项、Native 2 项，每项两 rank，共32条通过记录。最大 owner dW 绝对误差
`2.9802322387695312e-8`，两步更新后的权重一致。所有 rank 正常终止，节点退出0，任务进程和设备锁
已释放，所选两卡健康且无进程。详细来源、失败修复和边界见交付目录 `implementation/REPORT-implementation.md`。
reentrant 原始 no-grad forward 无法从外层图观察 route；其副本完成证据采用实际 kernel 两投影归还。
后续桥接组合验收已完成单机四卡及双机各两卡的 EP2/DP2 FSDP：
两种 reshard 策略、GAS2、真实框架 checkpoint、跨两步同热专家、owner梯度与SGD动量对照均通过。
每种拓扑16条rank/case记录通过，真实all-gather在Torch version保持0时更新物化标记。
运行时代码和native沿用上述冻结版本，新增内容位于下游ST及基准入口。
这些结果不包含完整Trainer、其他FSDP拓扑或峰值HBM验收；具体ABBA结果见`fsdp-integration/REPORT-fsdp-integration.md`。
此前单机四卡H512/I512的受控FSDP训练步ABBA/BAAB中，B1的Push/Pull中位数分别增加43.15%/49.91%；
这是该小形状的组合测试记录，不作为当前指定EP32场景的迁移性能验收。
当前采用[EP32跨分支对照](expert_replica_performance.zh-CN.md)，保持准入规则和全部样本，不把功能通过等同于性能无回退。

## 完整的反馈期限

待维护者安排；本文不自动发布 issue 或设置评审截止日期。

## CC List

待维护者补充。

## 其他补充说明

构建使用独立仓原 dependency lock、隔离 source assembly 和发布前 source hash 校验。
新增 replica GMM patch 也进入 SHA 校验；构建期间源码变化时拒绝发布混合 payload。
所有迁移与桥接均使用独立工作树，原始文档和来源特性分支保留。
