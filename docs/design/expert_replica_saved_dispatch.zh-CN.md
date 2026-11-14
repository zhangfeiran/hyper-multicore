# RFC：在前向 Combine 中保存 dispatch X

本文记录已淘汰的 v1 实验，PR 中没有独立的 v1 执行选项；当前实现见 [dispatch-ready 保存 v2](expert_replica_dispatch_save_overlap.zh-CN.md)。

## 动机与状态

前向 kernel 结束后，autograd 需要把对称 dispatch 缓冲中的 X 保存到本图拥有的存储。
当前实现复用 `down_proj`，通过独立 `copy_` 保存，后续 forward 和 backward 可以重用通信缓冲。
本候选将同样的数据搬运放入 Combine 分段任务，争取与其他专家的计算和返回重叠。
搬运字节数保持不变，不新增完整激活缓冲，也不取消输出 clone。

候选默认关闭；设置 `HYPER_MULTICORE_MOE_FUSED_DISPATCH_SAVE=1` 请求启用。
该选项参与资源键与 EP 配置协商。实现完成不代表取得性能收益，是否默认启用由完整实测决定。

## 目标设计

1. 仅在前向 Push Combine 描述符增加 `VectorTaskFlags.SAVE_DISPATCH=16`。
   不改变 RuntimeHeader、TaskDesc 布局、队列、原事件阈值或 backward 配置。
2. host 验证每个 peer/expert group 的 tile 完整且唯一，端口和偏移符合标准布局。
   同一专家的各 peer 任务共享 FC2 完成依赖；普通分段与 compact 分段均覆盖实际有效接收行。
   零行任务保留原 credit 行为，不执行保存。
3. 原 `PutMemSignalKernel.Process` 完成两个 MTE3 ping/pong 写入等待后发布远端原信号。
   返回后该段 FC2 源数据已无未完成读取，才允许覆盖对应 `down_proj` 段。
4. 使用已有 `data_plane::get_pipelined`，从本 PE 的对称 dispatch 读取相同行偏移，
   写入本图拥有的普通 GM `down_proj`。参数 `max_block_bytes=0` 使用完整 UB 半区。
   两个最终 MTE3 写入完成后返回，不增加原完成事件的 credits。
5. Python 仅在已验证能力且本次正常执行时跳过独立 `copy_`。
   后续图仍不能复用本图保存的 `down_proj`；此生命周期由原 autograd 存储与 stream 记录保持。
6. 远端信号先于 X 保存发出，但下一次 F/B 的 dispatch 受已有全 EP ready handshake 保护。
   每个 rank 完成本次 kernel 和本地 stream 前序工作后才能参加下一次握手；
   全 EP 就绪之前不能覆写 dispatch。此约束与原 Python 保存路径一致，不新增全局 barrier。

## 需要设计评审 DFX 建议：兼容与回退

通过独立 native capability `mega_moe_saved_dispatch_version()==1` 确认支持，旧 payload 保留原拷贝。
初始资格限定 BF16、H 为 16 的倍数、普通非 batched Push 数值路径。
Pull、概率提前乘入 FC2、clamp 变体及未支持的 native 版本保留原拷贝。

PROFILE 与 TENSOR_DUMP_L1 模式中 device 不覆盖 FC2 输出，Python 同步选择原拷贝路径。
这使诊断仍能观察原中间结果；外部 CANN profiler 对普通 kernel 的观测不改变此选择。
guest 前缀仅裁剪无实际接收行的任务，CPU 测试要求两种前缀策略的 patches 与原配置一致。

## 验证与准入

- CPU 验证 descriptor 完整性、实际路由偏移与分段覆盖、B0/B1/B6、两种 guest 前缀策略，
  以及多 forward 延后逆序 backward、checkpoint 和诊断回退的图内存所有权。
- NPU 先比较 saved X 有效行的原始字节，并验证多图、重计算和有事件依赖的跨 stream 生命周期。
  不扩大到已暂停的 Native 浮点精度矩阵。
- 固定 EP32/E192/H5120/I1792/TopK8/BF16、4096 token/rank、B6 Push、真实 AdamW。
  比较旧二进制、新二进制关闭、新二进制开启；均衡与 alpha 10/0.007/0.001，双向 fresh process。
- 全部样本与 rank 保留。独立保存拷贝消失且 F 增量小于拷贝节省，并取得 F+B/step 净收益后，
  再进入原八点五臂性能准入。原有未通过的八点 1% 门槛不能由本轮筛选覆盖。

## 后续决策

若保存搬运使尾部 AIV、HBM 或 Combine 返回更慢，则保留默认关闭并记录负结果。
只有完成上述收益与生命周期验证，才考虑默认启用和将同类设计扩展到其他数值变体。

## 相关的 RFCs 和 API

本候选关联[专家副本运行时](expert_replica_runtime.zh-CN.md)与
[默认路径和研究控制](expert_replica_default_paths.zh-CN.md)。
不增加业务 API；私有环境开关、native capability 和 descriptor 位由匹配构建共同解释。
替代方案保留原独立 copy，优点是实现直接、诊断中间结果稳定，代价是可能形成前后向之间的串行搬运。
本候选增加小范围 device/host 分支及覆盖证明，是否值得保留取决于三臂同形状性能数据。

## 2026-11-14 实测结论

候选代码 `52e66dd` 已完成完整 native 构建、565 项 CPU 测试及 EP32 检查。
32 rank、4 个冻结路由点、512 次 saved X 有效行字节核对通过；多图逆序 backward、跨 stream、checkpoint 通过。
每臂每点两次 fresh process、40 个全 rank 最大值样本，真实 AdamW：

| 路由 | 开启相对旧 PR 的 F+B | 开启相对同新二进制关闭的 F+B |
| --- | ---: | ---: |
| 均衡 | +0.698% | +1.352% |
| alpha=10 | -1.157% | -2.063% |
| alpha=0.007 | +1.993% | +0.560% |
| alpha=0.001 | +0.228% | +0.129% |

正值表示更慢。本轮未取得可采用的跨路由净收益，在 v1 实验结束时保持默认关闭且未合入 PR 分支；当前仅保留后续 v3 研究实现。
alpha=0.007/0.001 分别有 22/30 个 rank 只有一个活跃 slot；Combine 后保存缺少后续专家计算可覆盖。
另采事件计时显示部分 F→B 开销进入 F 内，但不能把事件中位数直接相加解释未插桩 wall 差异。
各点三个臂的 allocator 峰值一致；SHMEM 与物理 HBM 分开保留证据。两节点退出 0、设备和锁释放。
原八点五臂准入保持原结论，本轮未触发扩展验收，也未重跑 Native 浮点精度矩阵。
