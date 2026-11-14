# RFC：普通任务图内的有界 forward 时序观测

## 动机（Motivation）

[提前保存 X](expert_replica_dispatch_save_overlap.zh-CN.md) 在逻辑图中可与 GMM1 并行，
实际四点实验却未取得可采用的跨路由净收益，8 lane 还比 48 lane 更慢。
已有 PROFILE 模式使用原 descriptor image、静态队列和原缓冲角色，不能用于解释此保存图的真实调度。
本版实现普通图的独立观测协议，保留 NORMAL 模式、原 ready queue、任务与事件 credits。
该能力默认关闭，仅用于显式诊断上下文，不构成性能优化或新的准入结论。

## 目标设计

复用 forward ABI 的独立 diagnostic_buffer 端口，不占用业务张量、saved X 或对称通信缓冲。
增加 native capability `mega_moe_normal_trace_version() == 1`；不匹配时显式失败，不能返回空 trace 当作成功。
host 在副本后缀和 guest 裁剪完成后复制整份 runtime，只改 header 的两个预留 uint32：tag 与每 worker 容量。
NPU 使用设备上的 clone 和异步八字节更新，准备阶段不把 runtime 读回 CPU，也不等待 producer stream。
worker 数来自已经准备好的 plan；实际长度和 worker ID 由 native 再次验证。
runtime 快照在 kernel 完成后才读回，包含正常 GMM 写入的 scratch，不把 scratch 内容变化视为图变化。
诊断模式、profiling 容量、所有 descriptor、队列、event 阈值与副本后缀逐字节保持。
禁止与 PROFILE/TENSOR_DUMP 并用，也禁止嵌套上下文。

内部使用入口为 `hyper_multicore.modules.mega_moe.normal_trace.capture_forward_tasks(capacity=1024)`。
上下文仅影响显式包围的 forward；退出时恢复 ContextVar，已完成捕获的张量由 session 继续持有。
NPU checkpoint 重算可能切换 Python 执行上下文，因此不能单靠 ContextVar 继承。
原 forward 将会话登记到该 checkpoint invocation 的资源注册表，重算从同一 invocation 恢复仍活跃的会话。
退出 trace 上下文后会话失活，不再捕获更晚的重算；其他 invocation 不能取得它。
单个 checkpoint invocation 禁止归属多个 trace scope，避免重算时将不同原调用混入同一会话。
调用 capture.decode() 时先等待捕获的 producer stream，再读回张量，允许从另一个当前 stream 发起解码。
调用方必须将整个诊断阶段放在性能计时窗口之外。

### 有界存储

每物理 worker 有独占的 64-byte header 和定长 64-byte record 数组，容量为 1–16384。
Cube 按既有最小 24 槽布局，AIV 从 Cube 区后开始；未使用的 Cube 槽保留为零。
实际每个 Cube/AIV worker 都初始化自己的 header，包括无调度任务的 AIV。
无共享写指针、无跨核或跨 rank 抢占，超限只累计 droppedCount，不能越界覆盖下一 worker。
host 分配独立的零初始化本地 GM；native 从 tiling 获取实际输入字节数，写入前检查容量、长度和地址对齐。
forward native tiling 增加 diagnosticBufferBytes，原有拓扑/runtime/event 标量的偏移不变。

### 时刻与身份

记录以下本地系统 cycle，不在每次读时刻前加入额外 PIPE_ALL：

| 字段 | 语义 |
| --- | --- |
| selectedCycle | ExecuteTaskFast 入口，开始读取 descriptor 前 |
| dependencyObservedCycle | 原 WaitForDependency 返回后 |
| computeReturnedCycle | 原 ExecuteComputeKernel 返回后 |
| eventPublishedCycle | 原 producer/trigger 发布路径返回后 |

每条同时记录 task ID/type、task_index、dependent/trigger event、input position 和 split_num/value。
动态有效行范围和空保存任务可结合捕获的 group_list 与完整 runtime descriptor 离线还原。
保留原有通信任务在 compute 内发布事件的行为。
worker 退出时发布最终 count、dropped、exitCycle 并清理自己已写记录的 cache lines。

## 需要设计评审 DFX 建议

### 生命周期与完整性

每个 invocation 拥有独立 tagged runtime、trace buffer 和 group-list 快照，并记录使用 stream。
原缓存 runtime 不被写入，不复用另一个调用的 trace buffer；跨图和后续 workspace 复用不能修改旧记录。
decode 检查所有实际 worker 的 magic、版本、容量、身份和退出时刻。
拒绝 dropped、缺失/重复任务、与实际 descriptor 不符或非单调的时刻。
所有 scheduled Cube/Vector task 的多重集合必须完全一致，空任务也计入完成证据。

### 时间解释边界

ready queue 在 IsQueuedTaskReady 已观察到 ready 后才选择任务，所以 selected→observed 不是任务从未就绪开始的完整等待。
同 worker 前一任务 published→下一任务 selected 的 gap 包含记录写入、ready scan 和扫描等待，不能全部归因于下一任务依赖。
computeReturned 表示任务函数返回；published 区间还包括原事件发布与同步操作。
这些记录不能直接当作硬件带宽计数或 MTE/Cube 的精确忙碌时间。

跨 worker/rank 的 cycle 域未校准前，只报告各自的局部区间与顺序，不拼接跨核绝对关键路径。
在同 worker 上观察保存是否延长后续任务、保存/计算自身耗时和依赖等待，再结合整体 kernel 事件定位下一候选。
带 trace 的开销会改变并发竞争，trace 结果只用于诊断；端到端性能继续采用独立无 trace 的固定协议。

### 验证方案

CPU 检查普通/保存图、B0/B1/B6、不同容量及 replica suffix 的字节恒等，只有两个 header 字段允许不同。
C++ 原生布局通过 CPU 编译，校验与 Python struct 的字节一致及实际存储长度边界。
模拟全部 worker 的记录，验证完整任务覆盖、空 worker、时刻顺序、错误 descriptor、缺失和 overflow 拒绝。
验证 scoped 开关、异常退出恢复、独立分配及旧 native capability 拒绝。

独立 native 全量构建后，在 EP32/E192/H5120/I1792/TopK8/BF16/4096 token/rank、B6 Push 上采集。
四个冻结点为均衡、alpha=10/0.007/0.001；对 off/48/8 保留原图并核对实际选择。
先运行无 trace 的旧版/关闭/48/8 正反顺序，之后在单独诊断窗口采集两步普通图 trace。
各性能臂计时窗口外继续核对 saved-X SHA256，最后验证全部 rank/node 终态和资源释放。
不以诊断四点替代原八点 Native/PR7 准入，不恢复已暂停的 Native 浮点精度矩阵。

## 相关的 RFCs 和 API

本实现接续保存图的[实验结论与后续观测方案](expert_replica_dispatch_save_overlap.zh-CN.md#后续观测方案历史计划现已实现)。
不增加业务 API；显式上下文与二进制协议仅用于研究诊断。
若观测不足以区分下一机制，就保留当前默认路径，不扩大参数扫描挑选有利点。

## 实现与验证结果

匹配运行包提交为 `c042325711565b2e5768393e963ad6020749a7f4`。
584 项 CPU 测试、1705 个子用例通过；独立 native 构建完成，后续 Python 修正由校验 native 输入的 vendor snapshot 复用并重新打包。
普通图 trace capability 为 1；所有性能诊断记录的 NORMAL 模式和 vector_ready_queue=1 都通过检查。

首轮 NPU 暴露 checkpoint 重算切换线程后丢失外层 ContextVar，导致 6 次 forward 只捕获 4 次。
该轮在资格阶段停止，没有性能样本。单卡上下文探针确认 invocation 身份仍保持，修复后按 invocation 恢复活跃会话。
原 6/5 捕获门禁未降低；关闭 scope、复制上下文及 invocation 隔离均有独立回归。

修复轮两节点退出 0，288 个 worker terminal 完整，1024 个性能 rank-case、20480 个 wall rank 样本保留。
1536 份性能 trace、768 份完整性 trace 均覆盖每次实际的 72 个 worker，无溢出或缺失任务。
另有 32 次容量为 1 的原生溢出探针，所有记录槽有界，解码全部按预期拒绝。
1664 次 saved-X 行数/SHA256 检查通过，未执行 Native 浮点误差矩阵。

| alpha | 最忙 AIV 保存本体 ms，48 lane | 保存本体 ms，8 lane | 最忙 AIV 保存事件发布 ms，48 lane | Cube GMM1 本体 ms，关闭 | Cube GMM1 本体 ms，48 lane |
| --- | ---: | ---: | ---: | ---: | ---: |
| 均衡 | 0.4462 | 0.6951 | 0.0283 | 4.0960 | 4.4153 |
| 10 | 0.4089 | 0.8737 | 0.0283 | 4.0754 | 4.3315 |
| 0.007 | 0.5427 | 1.3916 | 0.0262 | 3.6457 | 3.7290 |
| 0.001 | 0.5401 | 1.3994 | 0.0270 | 3.6412 | 3.6514 |

表中每份 trace 对同类阶段取单 worker 累计区间的最大值，再在每点每臂的 128 份 trace 中取中位数。
GMM1 本体包含任务函数内的权重就绪等待，不能等同纯 Cube 数学时间；这些局部区间不能跨阶段相加。
8 lane 的热点保存本体约为 48 lane 的 2.6 倍，事件发布仅约 0.03 ms，继续减此处完成事件的优先级较低。
均衡和中等偏斜下，保存开启后的 GMM1 本体也增长；资源争用仍是待验证解释，不据此宣称具体带宽因果。

更明显的局部阻塞是空专家 FC2：热点路由下，它在 Cube 静态队列里等待依赖，延后同 worker 后面的有效 FC1。
关闭保存时，首个有效 FC1 前的空 FC2 等待累计中位数为 4.0621/4.4020 ms（alpha=0.007/0.001）。
已据此收敛下一版[解除空 FC2 等待的方案](expert_replica_empty_fc2_wait.zh-CN.md)，该原型后续已完成实验，本 PR 只保留研究记录。

本轮未插桩 wall 的 48 lane 相对关闭在四点分别为 -1.693%、+1.736%、+0.897%、+0.453%。
没有跨路由、跨 cohort 的稳定收益，保存候选在该 cohort 结束时默认关闭且未合入 PR；当前已作为默认关闭研究路径合入，原八点五臂准入保留。
双节点退出、健康、空闲、PID、锁与 UID relay 收尾均完成。

## PR 中的使用与边界

普通图 trace 及 checkpoint ownership、失活 scope、worker 边界、无预启动回读和 producer stream 修复已纳入 PR 分支。
默认仍需显式 scope 才启用；这项诊断能力不作为性能收益计算。
后续空 FC2 原型已完成实验，但未获得可采用净收益，PR 只保留[研究记录](expert_replica_empty_fc2_wait.zh-CN.md)。
筛选标准和保留的保存研究路径见[合入决策](expert_replica_integration_decisions.zh-CN.md)。
