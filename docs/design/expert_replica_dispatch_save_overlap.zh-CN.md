# RFC：dispatch 就绪后保存 X，与 GMM1 并行

## 动机（Motivation）

上一版 Combine 后保存没有取得跨路由净收益。
强热点下多数 rank 只有一个活跃专家，尾部保存缺少其他专家计算可供覆盖。
本版让接收侧保存与同一专家的 GMM1 并行，保持原完整激活存储数量。
候选由私有 `HYPER_MULTICORE_MOE_FUSED_DISPATCH_SAVE=1` 请求，默认关闭。

## 目标设计

| 缓冲 | 原路径 | 新路径 |
| --- | --- | --- |
| pos0，对称 dispatch | X | GMM1 阶段为 X，GMM2 之后为 Y |
| pos11，本图 owned down_proj | Y，随后 Python copy 保存 X | 接收侧保存 X，直到 backward |
| pos8 | activation | activation |
| pos12，对称 Combine | 返回 Y | 返回 Y |

增加 `TASK_SAVE_DISPATCH=142`；RuntimeHeader/TaskDesc 布局不变。
每专家的 scheduled 保存任务数等于 AIV worker 数，普通静态队列也插入完整 worker 轮次。
每个 lane 根据 group-list 划分实际有效行，空段直接完成，其他段用已有 MTE pipeline 从本 PE pos0 复制到 pos11。
最终 MTE3 写入完成后才触发原 GMM1 完成事件。原 GMM1 生产者数与新增保存任务数共同构成该事件阈值。
SwiGLU 因此同时等待所有 GMM1 X reader 与所有保存任务；GMM2 随后才允许在对应 pos0 段覆写 Y。
Combine 从 pos0 发送 Y，backward 继续从 owned X 读取 dW1 输入。

## 需要设计评审 DFX 建议

### 覆盖与 reader 完成证明

host 要求每专家 GMM1/SwiGLU/GMM2 lane 完整唯一、端口和动态行布局符合标准图。
完整 GMM1→SwiGLU→GMM2 fence 与原阈值必须匹配；额外 X reader、Y writer 或未证明模式被拒绝。
所有 peer/expert Combine 分段必须完整，且等待对应 GMM2 完成。
只有证明后才构建独立新 image，原 config 不被修改。
非终止任务 ID 保持；新增任务使用连续 ID，termination 仍位于 task_num，保留原完成事件。

CPU 模拟器按全 EP 发送目标推进事件，模拟不同 Cube/AIV 完成延迟、静态队列和 256 项就绪扫描窗口。
在 SwiGLU 与 GMM2 开始时独立检查全部 X reader/save 已完成；移除 join credits 的负例必须失败。
测试覆盖完整 EP32/E192/B6/H5120/I1792/TopK8/4096、24 Cube/48 AIV，以及普通/compact/nominal、B0/B1/B6。

### guest 前缀与诊断

保存任务按专家参加 guest suffix 裁剪，删除时同步减少原 GMM1 join credits。
原图和新图各有独立的 prefix patches，使用各自精确序列化容量，避免 nominal 图裁剪后的布局偏移错误。
全局和 rank-local 通信前缀都通过带完成延迟的分布式调度模拟。

正常训练使用新 image；动态 PROFILE/TENSOR_DUMP 使用原图、原端口和原 copy。
no_grad 与重入 checkpoint 初始 forward 也显式使用原 image，不执行无用保存或缓冲角色切换。
两种 descriptor image 都按使用 stream 记录生命周期。
已有跨调用全 EP ready handshake 继续保护对称缓冲，不增加全局 barrier。

### 能力与回退

独立 native capability 必须恰为 v3，支持本版缓冲角色与并发上限。v1 只有尾部保存，v2 不解释并发上限，均回退原路径。
缺能力、Pull、batched、概率提前乘入 FC2、clamp 或非对齐 H 保留原路径。
旧 v1 的 flag 16 保留其 wire 常量，本版使用独立任务类型，不再给 Combine 设置该位。

### 验证与性能准入

pos0 结束后已是 Y，不能拿它作为 saved X 的参考。
使用上一轮已验证的 512 个 saved-X 行数/SHA256，严格匹配输入字节、路由、placement 和 initial/scaled/recompute ordinal。
额外检查重入 checkpoint 的 recompute X、多图逆序 backward 及事件有序跨 stream。
这是通信与存储资格，不扩展已暂停的 Native 浮点精度矩阵。

仍使用固定 EP32 全形状、B6、真实 AdamW、四个冻结路由点。
v2 使用旧二进制/新构建关闭/新构建开启三臂正反顺序；v3 使用旧二进制/关闭/48 lane/8 lane 四臂正反顺序。
完整保留全部 rank 样本，分别报告未插桩 F+B/step、独立阶段事件、allocator、SHMEM 与物理 HBM。
有可采用净收益后才进入原八点五臂准入；未取得稳定收益时保持默认关闭；是否保留为研究路径见[合入决策](expert_replica_integration_decisions.zh-CN.md)。

## 相关的 RFCs 和 API

本候选接续 [Combine 保存实验](expert_replica_saved_dispatch.zh-CN.md)。
不改变业务接口、参数布局或优化器语义。
新增任务、能力版本、两份 runtime image 和私有环境选项是匹配 payload 的内部协议。

## 搬运并发上限

`HYPER_MULTICORE_MOE_SAVE_DISPATCH_LANES` 接受 0–128 的整数，0 使用全部 scheduled lanes。
非零请求截到实际 AIV worker 数，在这些 worker 中均匀分布 active copy lanes。
完整 worker 轮次、任务 ID、空 lane 的 join credits 保持不变；仅部分任务执行实际 MTE 搬运。
该值进入资源键与 EP manifest，保存任务的 task_split_value 记录有效 lane 数。
不能用 v2 payload 默默忽略这一请求，故能力版本升为 v3。
这用于检验突发搬运与计算/通信的资源争用，不能据逻辑并行就宣称有性能收益。

这里限制每专家实际执行保存的任务分区数，不保证整个 kernel 同时只有指定数量的 AIV 搬运。
不同专家可能同时保存；静态队列的硬件并发也受到活跃 worker 数约束。

## 实现与实验结论

v2 实现提交为 `847c389cc0504551cb7482eda7f0acd063f61706`。
v3 并发控制提交为 `be80ff96eeda7d489970c4479b03d0ab3b203491`，能力版本为 3。
两版均完成独立 native 全量构建；v3 的 570 项 CPU 测试、1675 个子用例通过。
事件模拟覆盖逻辑顺序，实际 NPU 测试另行核对 owned X 的字节和跨图生命周期。

v2 的独立阶段事件窗口显示 F→B 间隙缩短约 0.55–0.66 ms，同时 forward 增长约 0.43–0.74 ms。
四点未插桩 F+B 相对同构建关闭分别为 +0.214%、+0.725%、+0.643%、-0.307%，未得到跨路由净收益。
阶段事件与未插桩 wall 属于不同窗口，不能把阶段中位数相加来解释 wall 差值。

v3 保持相同二进制、缓冲角色、任务数及事件 credits，只比较 48 与 8 个实际搬运分区。
固定顺序为 old/off/wide/on/on/wide/off/old，后半反转路由点。
每点每臂两次 fresh process、5 次 warmup、20 次测量，共 40 个全 rank 最大值样本，不裁剪慢样本。

| alpha | 关闭 F+B ms | 48 lane F+B ms | 8 lane F+B ms | 48/关闭 | 8/48 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 均衡 | 25.4181 | 25.4742 | 26.4432 | +0.221% | +3.804% |
| 10 | 29.4770 | 29.2359 | 29.4082 | -0.818% | +0.589% |
| 0.007 | 48.5206 | 48.2291 | 49.1326 | -0.601% | +1.873% |
| 0.001 | 49.7469 | 49.7093 | 50.0381 | -0.076% | +0.661% |

正值表示变慢。48 lane 的小幅收益未跨 cohort 稳定复现；8 lane 在这四点均比 48 lane 慢。
当前结论只否定此处的 8 lane 配置，不能推广为所有并发设置均无收益，也未证明具体的带宽或 worker 争用因果。
候选仍默认关闭，未替代原八点五臂准入，当时也未合入准备提 PR 的分支；当前保留策略见末尾的 PR 合入决策。

v3 共核对 288 个 worker terminal、1024 个性能 rank-case、20480 个 wall rank 样本。
640 次独立完整性检查和 1024 次性能臂计时窗口外检查均匹配冻结的 saved-X 行数/SHA256。
普通/缩放输入、多图逆序 backward、事件有序跨 stream、两类 checkpoint 均通过；未重启 Native 浮点精度矩阵。
全部输入与 placement 一致，两节点退出 0，无遗留自有 PID，32 卡健康、空闲、锁释放，UID relay 关闭。

没有增加完整激活副本；启用后的 allocator allocated 峰值增加约 2.49–2.81 MiB，包含额外 descriptor image。
SHMEM heap 的容量与大小在各臂相同。物理板卡 HBM 是包含完整性阶段的整次采样，不能作为每臂显存收益。

## 后续观测方案（历史计划，现已实现）

本节保留当时计划；普通图观测现已[实现并完成验证](expert_replica_normal_task_trace.zh-CN.md#实现与验证结果)。

先测普通图内的保存/GMM1/GMM2 执行区间，再决定新的调度改动。
已有 PROFILE 模式改用原图和原端口，不能用于证明此候选的实际任务重叠。
需要为普通任务图增加独立、默认关闭的有界 trace：记录任务类型、专家、worker、开始/结束 cycle 和本地 event 阶段。
使用独立预分配记录区与每 worker 写入范围，写满时显式报告溢出；不复用 saved X 或对称通信缓冲。
trace 开关不能改变任务图或完成 credits；只分析本地 worker 的间隔，未校准时不拼接跨 rank 绝对时间。

先比较关闭、48 lane、8 lane 的普通图与 trace 图是否同构，再在同一配置下取得局部时序。
用这些记录区分保存本身耗时、保存等待、GMM1 reader 收尾和保存对其他工作造成的延后。
带 trace 的结果只用于定位；候选性能仍以独立无 trace 的固定 wall 协议判断。
只有定位到具体关键路径后，才考虑 GMM1-start 门控或改变保存分区，避免继续仅凭逻辑并行推断收益。

## PR 合入决策

上述“未合入”描述对应原实验完成时的历史状态。现在保留 dispatch-ready 并行保存 v3 及其有界 lane 控制，
作为默认关闭的研究实现；没有重新启用旧的 Combine 尾部保存算法，也没有默认选用已测变慢的 8 lane。
潜在改进来自减少保存与 GMM1 的重复读、收紧 reader/save join、缓解搬运竞争和调整保存时机，
不是将约 0.03 ms 的事件发布单独作为值得采用的优化。这里保留的是可检验方向，尚未获得新的收益资格。
启用与 lane 请求已显式进入组内 manifest，CPU 负例拒绝不同 rank 的不一致选择。
详细选择和验证边界见[合入决策](expert_replica_integration_decisions.zh-CN.md)。
