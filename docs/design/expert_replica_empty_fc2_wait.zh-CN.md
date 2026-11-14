# 研究记录：空专家 FC2 依赖解除（未纳入 PR 实现）

本文记录独立研究分支的已完成实验。PR 分支只保留证据与淘汰理由，不包含 `empty_fc2.py`、
`empty_forward_experts`、runtime bitmap 缓存或对应环境开关。原型提交 `fc2c011` 保留在
`perf/replica-empty-fc2-wait-20261114`，便于复现。本文件中的实现细节均指该原型。

## 动机（Motivation）

[普通图 trace](expert_replica_normal_task_trace.zh-CN.md) 已完成 EP32 全形状验证。
当前热点路由的 Cube 程序顺序包含 GMM1[e]、GMM2[e-1]，而 ExecuteTaskFast 总在判定动态行数前等待 dependent_event。
当 GMM2 对应专家没有 token 时，其计算本体只执行空行返回，仍可能等待上游 SwiGLU 数毫秒。
同一 Cube worker 在此期间不能执行后面的有效 GMM1；这是本地程序顺序内可直接确认的阻塞，不依赖跨核时钟对齐。

固定四点、每点每臂 128 份 trace，对每份 trace 取单 Cube worker 的本地累计等待最大值，再取中位数：

| alpha | 关闭保存：空 FC2 总等待 ms | 关闭保存：首个有效 FC1 前的空 FC2 等待 ms | 48 lane：首个有效 FC1 前 ms | 8 lane：首个有效 FC1 前 ms |
| --- | ---: | ---: | ---: | ---: |
| 均衡 | 0 | 0 | 0 | 0 |
| 10 | 0 | 0 | 0 | 0 |
| 0.007 | 4.6606 | 4.0621 | 4.6632 | 5.0620 |
| 0.001 | 4.5770 | 4.4020 | 4.7072 | 5.4910 |

这不是预计可节省的 wall 时间：依赖链调整会改变通信与计算重叠，必须实现后进行无 trace 验收。
相比之下，本轮保存任务的最忙 AIV 事件发布累计中位数约 0.011–0.028 ms，优先级低于上述空计算等待。

## 目标设计

第一版只处理可证明为零行的 forward GMM2，先在关闭提前保存的基线实现，避免同时改变多个机制。
复用路由阶段已有的整数计数，不引入 NPU→CPU 读回。`empty_owner_experts` 只表达全非空证明，不能用作空专家列表。
新增独立的 `RouteMetadata.empty_forward_experts`，由 host descriptor 的累计 group-list 相邻差为零生成，随元素单位视图一起保存。
设备端 descriptor 路径缺少 host 证明时保留原等待，不为此增加 readback。
只将这些专家的 GMM2 dependent_event 改成 EVENT_INVALID_ID，保持任务 ID、数量、worker residue 和 trigger_event 不变。
保留原计算调用的空行返回，以及原完成事件的一个 credit。
如果现有 host 信息只证明 home 专家为空，第一版就只覆盖 home，不从未映射 guest 推断任意专家的行数。

不能简单删除 GMM2 任务或减少事件阈值，否则会破坏 Combine 的完成计数和既有 guest prefix。
不修改通信任务的依赖或信号；不修改 backward，空专家权重梯度的清零语义保留。
后续仅在该最小变体通过后，再评估零行 GMM1、SwiGLU、保存任务或与提前保存的组合。

## 需要设计评审 DFX 建议

### 空行证明

必须同时证明目标是标准 per-expert GMM2、group-list 行区间为零、所有输入和输出均按同一行区间索引。
验证实际 native 实现的零行返回位于数据读取、权重等待和输出写入之前；不以阶段名字推断语义。
如果某研究变体在零行时仍有副作用，或者 host 计数不完整，保留原依赖。
首先限制在已验证的普通 BF16 Push、标准 FC1/SwiGLU/FC2 及现有 saved-dispatch 可证明布局。

### 提前完成的安全性

零目标行意味着该专家所有接收源的整数 token 计数均为零。
提前声明 GMM2 完成只允许 Combine 的零长度分段更早进入队列，不能让非零 Y 提前被读取。
原 dispatch/Combine 任务、各 peer 信号和 kernel 全部 worker 收尾仍执行，保留跨调用对称缓冲握手。
多图逆序 backward、跨 stream 和 checkpoint 的 owned X、参数与优化器语义不变。

### Runtime 与缓存

构建计划时证明原图与 saved-dispatch 图的 FC2 描述符，保存各 home 专家的 dependency 字节偏移及原事件。
在副本后缀及 guest prefix 应用完成后，修改独占 runtime 的这些字。所有 prefix 变体必须通过偏移不重叠检查。
避免每步 clone 整份大 runtime；固定路由应命中缓存。
每个 workspace 最多保留八组基础 image 与 layout 身份，只保存最近 bitmap。已有副本 split image 原地修改；
没有独占 image 的 B0 路径只在首次建立缓存时 clone。固定路由无 metadata copy。
bitmap 改变时同时设置新空组和恢复旧空组；证据未知时全部恢复。提交失败标记 dirty，下次重写全部目标字。
淘汰 split image 的记录前恢复原依赖，避免遗留无法跟踪的无等待状态；清理 owned clone 时释放底层 storage。
原图与保存图使用独立布局；prefix 的队列和计数字段保持不变，因此无需按每一种 prefix 再分配整图。
私有开关默认关闭；PROFILE/TENSOR_DUMP 保留原图。

### 验证与准入

CPU 独立模拟实际 token 行数与延迟完成，证明非空专家的全部数据依赖仍成立。
加入错误 bitmap 将非空专家当作空的负例，必须拒绝或使独立内存顺序断言失败。
检查所有任务、完成 credit、通信分段和终止条件均未减少。
覆盖 EP32/E192/B0/B1/B6、原图/保存图、guest 前缀、零 token rank、多图和两类 checkpoint。

NPU 使用均衡与 alpha=10/0.007/0.001，EP32/H5120/I1792/TopK8/BF16/4096 token/rank、B6、真实 AdamW。
先用同一完整 payload 的开关正反顺序，分别记录无 trace F+B/step 与独立正常图 trace。
Trace 应证实零行 FC2 依赖等待消失，并检查它是否仅转移到后续的有效任务。
存储与生命周期继续使用冻结 saved-X 参考，不恢复已暂停的 Native 浮点精度矩阵。
有可采用的跨路由净收益后，才进入原八点 Native/PR7 五臂验收。

## 相关的 RFCs 和 API

研究原型位于其分支的 `modules/mega_moe/empty_fc2.py`，原型私有控制为 `HYPER_MULTICORE_MOE_EMPTY_FORWARD_FC2=1`，默认关闭。
CPU 验证包括 EP32/E192/B6、B0/B1、原图/保存图、prefix 偏移、动态 bitmap 恢复、缓存淘汰、提交失败及非空负例。
NPU 性能结论见本文件的 v1 实施结果，不把数毫秒的本地等待直接当作收益。
源码和实验依据来自普通图 trace 的完整记录，未改变原四点 wall 聚合，也未挑选慢样本。
该候选不增加业务 API，不恢复默认不可达的成本模型。

## v1 实施结果

测量代码为 `fc2c011`，EP32/E192/H5120/I1792/TopK8/BF16/4096 token/rank、B6、真实 AdamW。
四臂按 old/off/saved48+empty/empty/empty/saved48+empty/off/old 跑 fresh-process 正反顺序；
每点每臂 40 个无 trace rank-max 样本，中位数聚合，保留全部慢样本。

| alpha | off F+B ms | empty F+B ms | 变化 | 完整 step 变化 |
| --- | ---: | ---: | ---: | ---: |
| 均衡 | 25.49953 | 25.64497 | +0.570% | -0.384% |
| 10 | 28.95297 | 29.04110 | +0.304% | -0.314% |
| 0.007 | 48.38005 | 48.35965 | -0.042% | -0.011% |
| 0.001 | 49.57862 | 49.47180 | -0.215% | -0.077% |

空 FC2 等待在热点点约从 4.6 ms 降到 0.01 ms，但 FC1 的前置依赖等待相应增加。
以各 Cube worker 自身首任务选择为起点，取最后非空 FC2 发布的本地时长，再取全 rank 最大值，
对四个诊断组取中位数：两点分别为 14.354→14.380 ms、12.107→12.076 ms，几乎不变。
这不是跨 rank 对齐后的完成时间，只用于说明没有观测到显著的真实产出提前。
不能把 Cube 空尾缩短当作关键路径收益；也不能把 ready-queue gap 一概叫作通信 body。

实现已通过完整 CPU 590 tests / 1709 subtests、两种布局的 256 个生命周期 rank-case、
2304 次 saved-X 字节检查、3072 份正常图 trace，以及全部 320 个 worker 终态与两节点资源审计。
按用户要求没有恢复 Native 浮点精度矩阵。

本候选没有可采用的跨路由端到端净收益。按合入筛选，仅 FC2 解除依赖主要移除非关键等待与空尾，
没有减少真实产出的关键工作；后续稀疏信号协议也不需要该专用缓存，因此本 PR 不合入原型实现。
此判断限于当前已验证机制和工作负载，不宣称所有其他形状都没有收益。原八点 Native/PR7 准入不变。
下一版改为先审计实际非零消息与完成信号，见 [稀疏 forward 信号计划](expert_replica_sparse_forward_signals.zh-CN.md)。
