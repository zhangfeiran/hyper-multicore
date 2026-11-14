# MoE 通用执行优化

本设计描述可在 owner-only MoE 中独立使用的优化。参数由 `MegaMoeExperts` 固定到资源规格，
沿用无参数专家接口、EIO/EOI 布局和模块拥有的概率执行；数值策略仍由 `MegaMoeNumericsOptions` 管理。

## 调度与初始化

- 保留当前基线的紧凑 task 与 ready queue。新增 private row descriptor，避免每个非空 GMM 反复写完整 group-list scratch。
- 在图构造时选择 owner 梯度队列、Cube lane 轮转和可验证的 ActGrad 优先顺序。每个变体必须保持任务集合、每个 Cube 的归属和完成信用。
- 自动模式要求对应 native 能力标记；显式选择不支持的 native 变体会在 launch 前拒绝。
- 复用已验证的 Host 路由元数据来证明 owner 非空，从而跳过全量清零及额外的 empty-row zero kernel；缺少证明时继续原有初始化。
- 事件刷新、空 Dispatch/Combine 信用合并和按最大边行数裁剪通信，作为显式研究控制保留。空 Combine 当前限定 Ascend910C。
- 跨节点自动队列保留 W2 分工，只调整 W1；节点信息并入已有资源一致性 collective。

## 研究控制

这些环境变量在实例创建或 plan 构造时读取，并进入资源兼容性校验；实例内不改变选择。

| 控制 | 用途 |
| --- | --- |
| `HYPER_MULTICORE_MOE_GRADIENT_QUEUE_ORDER` | 自动或显式 owner 梯度顺序 |
| `HYPER_MULTICORE_MOE_EVENT_REFRESH_US` | Cube:Vector 依赖刷新间隔 |
| `HYPER_MULTICORE_MOE_OWNER_GRADIENT_INIT` | `full` 恢复完整初始化 |
| `HYPER_MULTICORE_MOE_BATCH_EMPTY_DISPATCH` | 启用空 Dispatch 信用合并 |
| `HYPER_MULTICORE_MOE_BATCH_EMPTY_COMBINE` | 启用空 Combine 信用合并 |
| `HYPER_MULTICORE_MOE_COMPACT_COMMUNICATION` | 使用已验证的路由边界降低 task 覆盖范围 |

PTA 当前上下文回调只在 adapter 中按已有初始化契约缓存；SHMEM 的显式 `use_sdma` 对直接映射 peer 使用流内复制，
无法直接映射时拒绝该显式路径。默认 SHMEM 传输接口继续由调用方选择。

## 验证边界

CPU 回归检查任务覆盖、事件计数、重排归属、缓存隔离和错误拒绝；native 构建检查实际模板实例化、
框架 ABI、CTest 和加载。历史性能研究不构成当前新基线的 NPU 精度或性能验收。

## 保存重叠与普通任务观测

`HYPER_MULTICORE_MOE_FUSED_DISPATCH_SAVE=1` 显式请求 v3 保存研究路径；
`HYPER_MULTICORE_MOE_SAVE_DISPATCH_LANES` 控制参与拷贝的 lanes，两项都进入资源键和 EP 协商。
默认关闭，旧 native、clamp、pre-FC2 或不兼容几何继续独立保存。复制与 GMM1 读 X 并行，
它们共同完成原 GMM1 fence 后才允许后续覆盖 X；未执行数据搬运的 scheduled lane 仍贡献 join credit。

`capture_forward_tasks()` 只在显式 scope 中开启有界普通图观测；不替换成 PROFILE 图，
不修改 task、队列、完成信用或副本 suffix。checkpoint 调用共享通用 invocation owner，
异步上传和读回保留原流依赖。Native 在使用任何观测指针之前校验 worker、容量和实际 tiling 字节数。
当前没有新增默认启用选项，已有历史观测不充当此整合分支的性能资格。
