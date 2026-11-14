# 专家副本优化代码合入与默认选择

本文记录原 `feat/expert-replica-pr` 至 `3dc373a` 的历史采用选择。当前接口整合分支已按通用能力与副本协议拆分为两个单提交 PR；
下面的设备资格仍归属于原始源码，当前 CPU 与原生构建资格单独记录。

## 原分支决策

本轮按“收益成立则默认启用；仍有具体可检验改进路径则默认关闭保留；当前机制缺少关键路径收益或剩余上限很小则不合入”筛选。
新增实现没有获得足够的跨路由性能证据，因此本轮不新增默认开启的性能选项。
此前已经保留的默认优化不变；成本模型继续排除在运行依赖之外。

合入基线为 `ea8607a`。保存融合和普通图 trace 的原始提交链至 `283403c` 直接 fast-forward 保留，
不 squash、不改写已有 PR 提交。空 FC2 原型 `fc2c011` 不合入，原研究分支和冻结实验仍保留。

| 能力 | 处理 | 默认 | 证据和后续空间 |
| --- | --- | --- | --- |
| host 整数原语、融合计数、精确 metadata/plan 复用 | 保留既有实现 | 既有支持条件下开启 | 本轮不改变其已采用选择及回退 |
| owner 免清零、自动梯度队列、原生 BF16 输出协商、summary 准入 | 保留既有实现 | 按既有证明和原生能力选择 | 不把本轮 off/old 波动记为新收益 |
| dispatch-ready 保存与 GMM1 并行，v3 | 合入 | 关闭 | 存储/生命周期已通过；存在重复读取、reader/save join、搬运争用和时机的可检验改进路径 |
| 保存搬运分区控制 | 随 v3 合入 | 不启用保存；请求值 0 为完整 scheduled lanes | 保留争用实验控制；已测变慢的 8 lane 不是推荐默认值 |
| 普通图有界任务 trace、checkpoint owner 和异步准备修复 | 合入 | 无显式 scope 时关闭 | 诊断能力已验证，后续优化需要保持真实普通图的局部时序证据 |
| 旧 Combine 尾部保存 v1 | 不保留为独立可选实现 | 不可选 | 最终保存实现已被 v3 替代；原提交和历史文档保留 |
| 仅解除空 home FC2 依赖 | 不合入实现 | 无当前环境选项 | 本轮移除约 4.6 ms 局部等待，却未显著提前真实 Y；空尾缩短不构成端到端收益 |
| 单独压缩保存事件发布 | 不增加专用实现 | 无选项 | 已测发布约 0.03 ms，相对 48–50 ms F+B 的可省绝对量很小 |
| 梯度缓冲复用、未通过的 producer/W13 研究分支 | 不恢复已淘汰实现 | 不可选或保留已有独立控制 | 保留原证伪记录；不把旧研究分支整体 merge 进 PR |
| 稀疏 forward 完成信号协议 | 仅合入 RFC | 尚未实现 | 先审计已有 empty Dispatch/Combine 和实际贡献者，再决定协议实现 |

“仍有空间”要求指出可改变的工作量或依赖，而不是只因某个环境参数尚未扫描就保留任意实现。
对 FC2-only 的排除限于当前机制和已验证工作负载，并不宣称所有其他形状都无收益；若有新的关键路径证据，可重新评估研究原型。

## 默认关闭路径的约束

`HYPER_MULTICORE_MOE_FUSED_DISPATCH_SAVE=1` 才请求保存融合。
`HYPER_MULTICORE_MOE_SAVE_DISPATCH_LANES` 默认为 0，只有保存实际启用时才有意义。
两项均进入资源键和 EP manifest；不同 rank 的不一致请求必须在分配与执行之前拒绝。

普通图 trace 只由 `capture_forward_tasks()` 的显式 scope 启用，正常执行不分配 trace buffer，也不修改任务和完成 credit。
不使用 PROFILE 图替代真实融合图；trace 时长不计入采用性能。
保存图需 native capability v3，普通 trace 需其独立 capability v1；旧库分别按现有协议回退或显式拒绝观测请求。

保留保存研究路径的原因是原独立 copy/保存阶段仍有可测成本，且当前融合增加了 GMM1 body 区间。
后续可以检验复用 GMM1 已读数据、改变保存粒度和 join 时机等具体路径；这些是待验证方向，不是本轮取得的收益。
不继续把只有约 0.03 ms 的事件发布计为足以解释整段损失的优化目标。

## 证据与验证边界

保存与普通图 trace 的匹配设备代码为 `c042325`，产品代码与 `283403c` 相同：
EP32/E192/H5120/I1792/TopK8/BF16/4096 token/rank、B6、Push、真实 AdamW，四点 fresh-process 正反顺序。
该 cohort 核对 288 个 worker terminal、1024 个性能 rank-case、20480 个 wall rank-sample、
1664 次 saved-X 检查、1536 份性能 trace 和 768 份生命周期 trace，设备与锁释放完整。
两种 checkpoint、逆序多图和跨 stream 存储生命周期通过，保存仍没有可采用的跨路由收益。

空 FC2 的研究代码 `fc2c011` 使用相同固定形状，另一个完整 cohort 核对 320 个终态、256 个生命周期 rank-case、
2304 次 saved-X 检查和 3072 份正常图 trace。F+B 在 alpha=0.007/0.001 仅改善 0.042%/0.215%，
真实 Y 最大本地产出时长分别为 14.354→14.380 ms、12.107→12.076 ms。
这些本地时长没有跨 worker/rank 的绝对时钟对齐；不能将删除的空尾直接解释为全局关键路径收益。

完整历史报告位于交付实验目录：`replica-normal-task-trace-context-20261114` 与 `replica-empty-fc2-wait-20261114`。
其冻结输入、构建来源、全部慢样本和完整节点归档保持不变。

本轮合入新增的运行时修复只有保存选项的构造期 EP 协商字段；普通执行器和 native 内核代码保持既有已测实现。
合入后在目标 worktree 使用独立 editable 环境运行完整 CPU、wire ABI、默认关闭、旧库回退及不一致 rank 请求验证，
并单独记录该 worktree 的 native 构建结果。此前同码设备证据不冒充一次新的 NPU 性能验收。
Native 浮点精度矩阵按用户要求继续暂停；原八点 Native/PR7 采用门槛不因合入默认关闭研究路径而改变。

## 相关设计

- [默认路径与研究控制](expert_replica_default_paths.zh-CN.md)
- [保存融合及并发控制](expert_replica_dispatch_save_overlap.zh-CN.md)
- [普通图任务观测](expert_replica_normal_task_trace.zh-CN.md)
- [空 FC2 原型研究记录](expert_replica_empty_fc2_wait.zh-CN.md)
- [稀疏 forward 信号下一版计划](expert_replica_sparse_forward_signals.zh-CN.md)
- [既有性能采用协议](expert_replica_performance.zh-CN.md)
