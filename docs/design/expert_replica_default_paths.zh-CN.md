# RFC：专家副本默认执行路径与非成本研究变体

状态：2026-11-09 已接入代码；CPU 回归412项、2399个子例通过；6000组规划差分保持一致。
新 native 构建及 NPU 验收进行中，不以第一版设备结果代替本轮结果。
按来源 `.github/ISSUE_TEMPLATE/003-RFC.yml` 的必填字段组织。

## 动机（Motivation）

第一版迁移把默认可达的执行加速与成本研究一并删掉。指定 EP32 扫描中，计划和数值相同，
Push/Pull 逐点变化中位数仍分别为+1.23%/+1.08%。CPU 单变量确认 compact-home 短路、
整数快校验及 host 整数原语的缺失会增加控制开销。默认 auto 梯度调度和有证明的免清零
也属于原执行语义的性能部分，需要迁移。

本轮只从运行依赖移除成本预测、校准表及其专用浮点评分。默认路径及与其共享实现的
非成本研究变体保留明确入口、保守回退和独立验收边界。

## 目标设计

### 控制路径

| 能力 | 默认行为 | 实现与回退 |
| --- | --- | --- |
| compact-home | 精确 summary 未达到40%收益时提前返回 | 缓存记录含 F/R ordinal；非法输入和容量仍先校验 |
| 整数校验 | 精确非负 Python int 走内联快路径 | bool、子类、越界与不支持类型沿严格验证 |
| host 原语 | 可加载时使用 ABI3 整数扩展 | 列和、局部性、最终物化；溢出或不可用回退 Python |
| fused count | 支持的 NPU、int64、E≤512、TopK≤8 尝试 ABI2 | 缺库或不支持形状走精确 Torch 计数 |
| compact candidate | 复用已验证的无表配额候选 | 精确核对 counts、容量、拓扑及策略后再物化 |
| owner 免清零 | 默认启用完整覆盖证明 | 正 K、无 guest 且生产者覆盖完整；证据不足全量清零 |
| owned metadata | 默认 host descriptor 上传 | 固定 stride、pinned source、stream 与调用所有权 |

host 扩展只包含四类整数原语，不携带曲线插值、rank compute、critical-path cost 或拟合。
融合 count 库由正式构建编译并随 payload manifest 一起校验，不能仅迁移 Python loader。

### 设备执行

`MegaMoeExperts` 默认解析 `auto`。实际调度根据 ABI、图形状、owner/guest、通信和
梯度 producer 证明选择 active lanes、ActGrad 优先或保守原序。直接构造低层 spec 的
既有默认值保持 original。解析结果进入资源键和组内 manifest，避免 rank 间不一致。

梯度 lanes、band 发布、observer、GMM local tiling 和 explicit split 必须同时构建。
归还继续遵守 complete-K、投影 ready、固定 peer 顺序、quorum 和 ACK。
不能证明生产者一一覆盖时不启用 join-once；有真实 band producer 时禁止破坏映射的 lane rotation。

目标仓已有 EOI 的直接梯度输出、B0 路由缓存和 Pull 暂存上限继续使用原实现。
正预算 EOI 仍明确拒绝，不通过搬回来源旧 transpose 分配逻辑改变该合同。

### 显式研究控制

| 控制 | 默认值 | 限制与回退 |
| --- | --- | --- |
| `HYPER_MULTICORE_MOE_GRADIENT_QUEUE_ORDER` | auto | original、W1逆序、join-once、device顺序、late-auto、lane与active投影；需 producer 资格证书 |
| event refresh cube/vector 间隔 | 0 | 精确量化并验证配置；写入 native task 配置 |
| `HYPER_MULTICORE_MOE_BATCH_EMPTY_DISPATCH` | 关闭 | Push且B0；空消息用 marker 保持 credit 语义 |
| `HYPER_MULTICORE_MOE_BATCH_EMPTY_COMBINE` | 关闭 | Push；保留 incoming combine 的精确等待配额 |
| rank-local task prefix | 保守模式 | 通信前缀需要全局 mapping，不能按本地空任务猜测 |
| metadata D2 replay | 关闭 | 精确设备、counts、布局、容量及 stream 生命周期匹配 |
| binary / hot readback replay | 关闭 | 精确整数 payload；当前 token IDs 始终重新映射 |
| `HYPER_MULTICORE_MOE_ADAPTIVE_REPLICA_INBOX` | 关闭 | B>1且无显式Heap；保留预算重规划和全组增长事务 |
| `HYPER_MULTICORE_MOE_OWNER_GRADIENT_INIT=full` | 未设置 | 提供保守全量初始化对照 |
| `HYPER_MULTICORE_MOE_ROUTE_METADATA=device` | 未设置 | 提供 device descriptor 对照 |

完整环境名以各模块解析实现为准，研究参数不新增到业务模型构造器。
独立 device planner 与 complete/projected guest search 没有默认入口，尚未迁移；
成本模型、自动校准、性能表安装和成本专用评分明确排除。

## 需要设计评审 DFX 建议

### 正确性与资源安全

计数和计划缓存不能替代本次 collective agreement、硬容量复查或当前 token permutation。
F/R 旁路同样记录 invocation ordinal；metadata 的 pinned source 和目标 buffer 均有明确 owner。
免清零只接受完整覆盖证据，guest partial 仍使用正确累加存储。共享 Heap 有活动 lease 时不增长。
所有新配置参与资源/组一致性检查，旧 ABI 或资格不足时使用完整协议回退。

### 复杂度和兼容性

大型前缀编译、band shape 和 producer 判断拆成单一职责函数；不关闭复杂度门。
目标仓要求 CCN≤15、有效行数≤100，本轮 Pylint 和 Lizard 均通过。
新增 host 扩展可独立导入失败后回退；成本模块不再成为 import 依赖。
低层 native 的调度 ABI3、adaptive gradient ABI4、empty-dispatch ABI1 必须匹配本包。

### 验收

CPU 用例覆盖整数溢出、子类、精确配额、compact 拒绝/接受、不重复搜索、producer 证明、
队列 oracle 和控制字段。6000组冻结规划差分包含4363个计划、1637个一致拒绝，
摘要仍为 `a2fa515c2b3b4a8ab7ebd0e1802158408afad43d35ea57e8a62da8519b24b6a4`。

NPU 先验证当前源码和匹配 payload 的数值、重算、空源、逆序 backward 及研究控制；
性能再按 EP32/E192/H5120/I1792/TopK8/BF16/T4096 per-rank、B1，
从均衡到热点的49点分别扫描 Push/Pull。输入、权重、路由、fresh process、rank-max、
ABBA/BAAB、慢样本及所有rank终态必须保留。默认计数能力的基线差异单独记录。

本轮交付目录：
`/home/ma-user/work/z00961864/expert-replica-minimal-migration-20261109/default-path-migration/`。
CPU 证据不能作为新 native、NPU 性能、Trainer、FSDP大形状或峰值HBM的证明。

## 相关的 RFCs 和 API

业务 API 保持只增加 `replica_slots_per_rank`。关联
[迁移方案](expert_replica_migration.zh-CN.md)、
[执行闭环](expert_replica_runtime.zh-CN.md)与
[指定形状性能协议](expert_replica_performance.zh-CN.md)。

## 完整的反馈期限

待评审排期确定。

## CC List

待项目维护者指定；本地文档不发送通知。

## 其他补充说明

本轮代码与实验独立于第一版冻结包，保留旧结果及失败样本。变更仅本地提交，未 push。
