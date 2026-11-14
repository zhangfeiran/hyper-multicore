# RFC：专家副本默认路径与研究控制

## 动机（Motivation）

无表版本需要保留默认执行加速及同链路研究变体，同时将成本拟合与校准彻底移出运行依赖。
本文件记录实际入口及回退，不按“默认关闭”删除有意义的非成本变体。

## 目标设计

| 能力 | 默认选择 | 证明与回退 |
| --- | --- | --- |
| summary home admission | 精确summary未达收益门槛时旁路 | 容量和当前输入仍先校验，记录F/R ordinal |
| host整数原语 | 匹配扩展可用时启用 | 非法类型、溢出或缺扩展沿Python严格路径 |
| 融合计数 | 支持的A2/A3、int64、E≤512、TopK≤8 | 缺库或不支持时回退精确Torch计数 |
| metadata/plan复用 | 当前计数及控制精确相同 | 本次token仍重新映射，存储和stream归调用持有 |
| owner免清零 | 完整producer覆盖时启用 | 冷专家和缺覆盖证明时保留清零 |
| 梯度队列 | 业务入口 `auto` | 跨节点自动限定 `auto_w1`，显式模式保持显式选择 |
| BF16最终输出 | 按实际原生能力协商 | 缺能力按旧协议回退，显式要求缺失能力时报错 |

直接构造低层spec保留原 `original` 队列默认值；业务入口解析后的选项进入资源键和组内manifest。
跨节点依据实际节点身份，不能由EP大小推测拓扑。

### 私有环境控制

下表供研究和故障定位使用，必须在构造执行器前一致设置。它们不构成稳定的业务参数接口。

| 环境变量 | 默认 | 语义 |
| --- | --- | --- |
| `HYPER_MULTICORE_MOE_GRADIENT_QUEUE_ORDER` | `auto` | 梯度producer队列模式 |
| `HYPER_MULTICORE_MOE_EVENT_REFRESH_US` | `0:0` | Cube:Vector轮询间隔，0保留原设置 |
| `HYPER_MULTICORE_MOE_SUMMARY_HOME_ADMISSION` | `1` | 启用精确summary收益旁路 |
| `HYPER_MULTICORE_REPLICA_HOST_NATIVE` | `1` | 优先host整数扩展，0强制Python |
| `HYPER_MULTICORE_REPLICA_METADATA_REUSE` | `0` | 可选metadata replay，仍精确验证当前计数 |
| `HYPER_MULTICORE_MOE_HOT_REPLAY_READBACK` | `0` | 已使用副本的匹配重算可选host回读路径 |
| `HYPER_MULTICORE_MOE_ROUTE_METADATA` | host | `device`选择设备metadata路径 |
| `HYPER_MULTICORE_MOE_OWNER_GRADIENT_INIT` | 按证明选择 | `full`恢复完整初始化 |
| `HYPER_MULTICORE_MOE_BATCH_EMPTY_DISPATCH` | 关闭 | Push且B0时合并空dispatch |
| `HYPER_MULTICORE_MOE_BATCH_EMPTY_COMBINE` | 关闭 | Push空combine研究路径 |
| `HYPER_MULTICORE_MOE_COMPACT_COMMUNICATION` | 关闭 | Push紧凑通信任务 |
| `HYPER_MULTICORE_MOE_ADAPTIVE_REPLICA_INBOX` | 关闭 | B>1且未固定SHMEM Heap时按需增长inbox |
| `HYPER_MULTICORE_REPLICA_LOCAL_BF16` | 原生能力协商 | 0关闭，1要求匹配的local能力 |
| `HYPER_MULTICORE_MOE_FUSED_DISPATCH_SAVE` | `0` | 已验证存储语义的并行保存研究路径，需 native v3 |
| `HYPER_MULTICORE_MOE_SAVE_DISPATCH_LANES` | `0` | 保存启用时每专家的搬运分区控制，0 使用全部 scheduled lanes |
| `HYPER_MULTICORE_REPLICA_FINAL_BF16` | 原生能力协商 | 0保留旧归并输出，1要求final能力 |

队列模式包括 `original`、`auto`、`auto_w1`、`w1_reverse`、`w1_join_once`、
`device_w1_asc`、`device_w1_desc`、`device_w1_asc_join`、`device_late_auto`、
`device_lane_rotate`、`device_w1_asc_join_once`、`device_w1_asc_join_rotate`，
以及 `gradient_queue.py` 定义的自适应与active-projection模式。
不能证明producer覆盖、join-once或band一一映射时走保守调度。

正常图任务观测通过 `capture_forward_tasks()` 显式 scope 启用，默认无 scope、无 trace buffer。
它保留普通图、ready queue 和 credit，用于诊断；插桩时长不计入正式性能采用依据。

本轮保留与排除项见[合入决策](expert_replica_integration_decisions.zh-CN.md)。

## 需要设计评审DFX建议

- 研究选项必须进入资源签名及组内agreement，避免不同rank执行不一致的协议。
- replay只能复用精确匹配的计划/metadata；不能以相同histogram复用旧token排列。
- 普通驻留和saved-generation命中仍受当前lease、credit、ready、epoch和ACK保护。
- 自适应inbox需完成所有rank准备后再提交；活动lease时禁止无保护增长和provider更换。
- `FINAL_BF16=1`与 `LOCAL_BF16=0`冲突时拒绝；两投影必须共同证明协议可用。
- B0或手工低层spec的保守默认值不因缺能力而被强制开启。

测试覆盖默认选择、显式拒绝、fallback、路由精确性、研究入口可达性和生命周期。
显式研究路径可达不代表已经获得默认路径同等的性能验收。

## 相关RFCs和API

[执行设计](expert_replica_runtime.zh-CN.md)、[最终梯度](expert_replica_final_gradient_output.zh-CN.md)
和[性能协议](expert_replica_performance.zh-CN.md)。公共业务API只新增有界槽位上限。
