# RFC：按实际非零消息发布 forward 通信完成信号（待实现）

## 动机（Motivation）

[空 home FC2 v1](expert_replica_empty_fc2_wait.zh-CN.md) 已实现并完成 EP32 四点验证，但没有可采用的端到端收益。
alpha=0.007/0.001 的空 FC2 本地等待约 4.6 ms，解除后约 0.01 ms；同包 F+B 只改善约 0.04%/0.22%。
真实非空 Y 的最大本地产出时长在两点分别为 14.354→14.380 ms、12.107→12.076 ms，几乎不变。
后一个指标以每个 worker 自身的首任务选择为起点，取全 rank 最大值，再对四个诊断组取中位数，
不能当作跨 rank 对齐的绝对完成时间。

同一 rank0 / Cube0，空 FC2[0] 等待从 2.405 ms 降为 0.001 ms，而随后非空 FC1[2] 等待从 0.002 ms 增为 2.253 ms。
部分 Cube span 缩短来自空任务尾部；最慢 Vector 的本地 span 仍约 16–17 ms。
长 Vector gap 分别出现在非空 SwiGLU 就绪和最终接收完成之前，不能一概称为通信 body。
下一版需要减少真实输入到达与输出返回链上的工作，不能继续把局部空等待的缩短当作预测收益。

## 现有机制与范围

| 机制 | 当前状态 | 本方案的关系 |
| --- | --- | --- |
| guest prefix | 已实现，按映射裁剪未使用 guest 并匹配信号 | 保留，证明与新阈值兼容 |
| empty Dispatch 批 credit | 已实现私有选项，当前限制 B0 | 先审计已有路径，不重复宣称为新优化 |
| empty home Combine 批 credit | 已实现私有选项，可覆盖 B>0 home | 已有机制；批量发布仍不等于取消零消息 credit |
| 空 home FC2 dependency 解除 | 研究原型已验证，未纳入 PR 实现 | 保留研究记录；新协议不依赖原型代码 |
| 按非零消息改写 forward 阈值 | 本 RFC，尚未实现 | 本轮提出的后续研究协议 |

第一版只考虑 canonical BF16 Push forward、完整物理 count table 已可用的 active-replica 路径。
先关闭 saved-dispatch 和已有 empty-credit batching，避免同时改变多个协议。
不修改 backward、参数/梯度精度、优化器、源端 permute 存储、对称缓冲握手或默认业务 API。
不新增 TopK 展开 activation 的 HBM 副本，也不引入额外 NPU→CPU readback。

## 目标设计

设 `C[s,p]` 是全局已验证的物理 token count table，`s` 为源 rank，`p` 为目标物理 expert slot。
把每个 peer/expert 的完整消息作为是否为零的判据。令每个正消息的 dispatch/combine 原 tile 数为 `Td` / `Tc`：

```text
DispatchReady[p] = Td * sum_s(C[s,p] > 0)
OutputReady[s]   = Tc * sum_p(C[s,p] > 0)
```

正消息保留原 tile 数、数据 fence 和全部 tile credit，哪怕某个 tile 的数据长度恰好为零。
整条消息为零时，保持 task ID、队列长度和 worker residue，只省略它的远端消息完成 credit。
不能把“整条消息为零”误写成“当前 tile 为零”，否则上式与实际 credit 不再一致。

需要协调修改 dispatch 的接收阈值和 Combine 的返回阈值，不能只在发送端跳过零信号。
对全零 expert，dispatch 阈值为零应使空 FC1 更早返回，再沿原 FC1→SwiGLU→FC2 事件链完成空计算。
必须确认 native 对已使用零阈值事件的行为及终止语义；若 ABI 不支持，设计等价且明确的 invalid-dependency 表达，
不能靠未验证的零阈值假设上线。

### 第一阶段：信号和可达性审计

先在当前源码逐项确认以下内容，完成前不开始新 NPU 性能实验：

1. 把 `plan.py`、`empty_dispatch.py`、`inactive_tasks.py` 和 native `EmptyDispatchCredit` 的现有行为列成表，
   明确每个 scheduled task、每个远端目标和每个消费者事件的 credit。先确认已有选项是否已经解决相同问题。
2. 用冻结物理路由量化零完整消息、正消息内零 tile、未映射 guest 三类数量，三者分别处理。
3. 证明非空 GMM1 的 dependency 是输入完成事件，并区分它与 GMM body 中的 guest 权重等待。
   本轮 rank21 的非空 guest FC1 仍有约 6 ms 的前置依赖等待，body 约 3.63 ms；不能归因于 body 中的权重等待。
4. 按真实非空 Y、零尾部和最终接收完成分开检查；保留 ready-queue gap 与显式等待的区别。
   若要判断哪个 peer 最晚到达，增加诊断专用的目标端观察或贡献者记录，不对齐现有跨 rank 原始时钟来猜测。

### 第二阶段：最小协议实现

- Native 只增加一个有版本能力检查的零完整消息信号策略；正消息走原逻辑。
  新策略与现有 batch-empty-credit 的优先级必须明确，第一版直接禁止组合，防止一个分支仍替零消息发布 `T` 个 credit。
- Host 复用已存在的完整物理 count table，生成接收和返回的实际阈值；未知证据保持原协议。
  所有 EP rank 必须选择一致协议；不能有的 rank 降阈值、有的 rank 继续发送旧 credit。
- 验证每个消息的大小 descriptor、目标 rank / slot、消费者事件和 tile 数，不能仅凭阶段名或边际总量判断。
- 在独占 workspace 最终 runtime image 上应用动态阈值，原图保持不变。
  补丁在 guest-prefix 之后应用；缓存有界，固定 route 命中不增加整图 clone 或每步 Host 上传。
- PROFILE / TENSOR_DUMP 仍走原协议；所有原全局 ready、跨调用缓冲复用和 worker 收尾任务保留。
- 只有证明零阈值沿原链已消除空依赖，且新协议取得收益后，才考虑去掉 v1 的私有依赖缓存以收敛生产 diff。

## 需要设计评审 DFX 建议

### 事件、内存与生命周期

严格按实际消息证明 `实际发布 credit == 消费者阈值`。不允许计数跨步残留，也不能拿缩短的阈值接收尚未完成的数据。
正消息沿用原数据 copy→fence→signal 顺序。零消息没有输入读取和输出写入的前提必须来自实际 native 分支。
不减少 worker 遍历任务或保留 source / guest buffer 的跨调用握手。

零 token source 仍可能接收别的 source 的 expert 输入；自身输出阈值为零不能让它跳过其他 rank 需要的计算。
多个 autograd graph、跨 stream、反向图执行顺序和 checkpoint 的 owned X 语义均保持。

### 验证与准入

CPU 模拟使用真实行数，独立检查每个源、目标、tile、事件及全部完成条件：

- EP32/E192，B0/B1/B6，全非零、稀疏、零 token rank、仅远端输入、动态 route 和 guest-prefix 各变体。
- 正消息中部分零 tile 的用例，防止按 tile 长度误降 credit。
- 丢 credit、重复 credit、错误阈值、错误物理矩阵但相同边际总量的负例，必须拒绝或触发独立断言。
- 两图逆序、显式跨 stream、两类 checkpoint、缓存恢复/淘汰/失败提交。

NPU 仍先用 EP32/E192/H5120/I1792/TopK8/BF16/4096 token/rank、B6、真实 AdamW 的均衡/10/0.007/0.001。
新 cohort 冻结输入与聚合，采用同包开关 fresh-process 正反顺序；与本轮 v1 的数据不合并。
诊断检查实际 credit、任务覆盖和整数布局；原始浮点精度矩阵按用户要求继续暂停。

主要成功条件是非空 FC1 的真实输入依赖、真实 Y 产出与最终接收出现一致改善，同时无 trace F+B/step 取得可采用收益。
若只缩短零任务尾部或局部等待，停止采用该候选，不用局部时长替换吞吐结论。
通过后再新跑原八点 Native/PR7 对照，达到采用条件后才跑至少 30 alpha 的性能和 HBM 曲线。

## 相关的 RFCs 和 API

[空 home FC2 v1](expert_replica_empty_fc2_wait.zh-CN.md) 保留实现与本轮证伪结果。
[普通任务 trace](expert_replica_normal_task_trace.zh-CN.md) 定义正常图记录与时钟适用范围。
本文件是下一版实施计划，没有实现、精度等价或性能收益结论；不恢复默认不可达的成本模型。
