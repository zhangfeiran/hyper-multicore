# A5 MXFP8 对齐专家片段副本

研究分支基于 `perf/a5@e198c862`，独立于 A3 BF16 副本路径。选择
`MXFP8MegaMoeExperts(..., replica_slots_per_rank=2, replica_partition="aligned64")`；
默认 `whole` 保留完整专家训练策略。本功能暂为显式研究能力。

## 路由合同

每个逻辑专家的行序为 source rank 顺序、source 内原始 TopK 展平稳定顺序。
只允许 home 前缀与一个 guest 后缀，内部切点是 64 行的整数倍。允许整个专家迁移；
最后一个片段保留原专家的尾块。副本预算仍是每个目标 rank 的 B 个槽。
每个 source 的派发配额为其全局前缀区间与片段区间的交集；不能在 source 内各自对齐。

例如 source 行数 50、90、116，在 128 行切开，配额是 home=(50,78,0)、
guest=(0,12,116)。后端用最终 physical IDs 重新排列 token，再按 source rank 拼接。
因此 guest 前 32 行正是原专家 128..159，即 source1 的12行及 source2的20行。

`AlignedExpertExecutionPlan` 保存物理槽对应的全局专家区间；remap 顺序按逻辑专家与
片段起点排序。guest 的物理 rank 可以小于 owner，不能按物理槽编号隐式排序。
计划校验独立检查区间连续、64行边界、各source交集和逻辑计数守恒。

## 数值与权重合同

保留当前 FP32 roots、TN 输出及公共 dW。片段产生真实 FP32 局部梯度，先 home
再唯一 guest 做 FP32 累加。没有先 BF16 量化再转 FP32。全局固定输入的 MX 分组可以
保持，但 GMM 形状和 dW 的浮点归约分段变化仍可能改变完整输出、dX、dProbs、dW；
不承诺整层逐位一致，不把 CPU 行序检查当硬件精度验收。

权重 owner 身份不变；forward 消费当前 col_data/col_scale，backward 重新解析当前
row_data/row_scale。继续使用原有权重租约、投影 ready、梯度 inbox 与 ACK 协议。
不缓存没有可信 FSDP generation 的权重，不占用梯度 UDMA scratch 作为驻留空间。

## 决策与生命周期

纯 CPU 计划使用已有精确 counts，选择降低 `(最大rank负载, rank负载平方和)` 的
整专家或64行切分动作。每专家最多一个guest，每个片段至少 replica_min_rows（最小64）；
满足已有准入阈值后才迁移。近平衡继续走真实 home-only 路由，不建立活跃guest图。
完整计划保留在 checkpoint 上下文中；重算验证原逻辑counts及策略后复用，不能在F与R
重新切分。跨rank构造清单包含 partition，资源生命周期也包含该选择。

该启发式仅估算行数，不代表通信成本或实际加速。设备验收需比较相同 T6、FP32 dW、
QP与数据引擎、相同逻辑输入和每rank终态的 B0 / whole / aligned64；B0-only peer-QP0
路径须单独标注，不能擅自绕过当前 active 限制。组件结果不外推 Trainer 或训练收敛。


## 活跃副本与 peer token 传输的组合

第一阶段研究提交 `16f0b46` 为 P2P 路径增加独立开关
`PANGU_MX_PEER_QP0=1 PANGU_MX_ACTIVE_PEER_QP0=1`。按远端peer统一其 token
Put+quiet 的奇数 AIV owner；物理专家数包含 B，不改变数学任务与事件阈值。
原有 B0 开关不自动启用 active；source-row FP8 继续拒绝。

融合梯度及投影预取扩展单独使用 launch v16（梯度）、v17（权重ready）、v18（两者）。
v15 仍只表示普通 peer token。新路径须同时通过 adapter版本2、原生对象 SHA 和
`mx_active_peer_qp0` capability；旧对象继续拒绝新协议。

token 使用每peer QP0，四个梯度 helper 使用明确保留的非零QP。owner校验读取任务的
真实peer标记，不能按物理AIV编号猜测QP。现用SHMEM1.8源码中32位ADD信号通过mapped MTE，
SET在MTE拓扑走mapped store；不与token UDMA队列混用。旧固定QP bank协议仍在device拒绝。
发布ready仍须全部Cube投影完成，owner归并结束后ACK；不改变FP32 TN、partial/public dW。
原来没有同时表达这些合同的launch ABI，不能只放开Python参数就声称融合路径已支持。

第二版计划同时删去只改善平方和的副本边，并以既有全局 row credit 比较完整专家候选。
当整专家最大接收负载只多出允许的行数、却能减少完整权重/dW传输时，优先完整专家。
该规则是无表启发式，不引入成本模型；准入阈值和共同容量界限仍须满足。


## 固定 MX 图的逻辑 counts 复用候选

`PANGU_MX_SHARED_COUNTS=1` 只选择现有逻辑路由 agreement 作为 MX 的 count 控制通道。
控制spec使用逻辑专家数和已有INT32 wire，执行spec仍固定token数；不切换实际执行图为动态token。
复用已有ID/容量/grad mask校验及小表阈值，保留原有大表readback策略。planner收到同一次新鲜
agreement的CPU counts和source summary；home-only把已拥有的小表交给offset打包，避免再次用
设备算子重建相同metadata。F和R每次仍读取当前counts，不缓存未经本次校验的路由；新开关加入
EP构造清单和资源生命周期。该候选尚须匹配设备screen，默认关闭。


## 准入与粗切分必须使用相同目标

旧 compact 准入按平均负载目标寻找可填充 guest。当冷 rank 到平均值只剩少于最小片段的
空间时，它会拒绝一个实际能显著降低双边峰值的大块切分。aligned64 因此单独用相同
minimax/64行规则预览；只使用 global totals 的合成source表用于CPU决策，绝不上传执行。
真正派发仍依据本次完整source counts重新生成区间交集。拒绝memo包含partition标记。

节点身份作为计划与预览的输入；峰值及平方和相同时优先同节点。与whole候选比较时
继续传递同一rank_nodes。必须满足的容量迁移优先于可选的最小guest行数/收益阈值，
但内部切点仍须64对齐。新增回归用例检查“平均目标拒绝、实际单guest方案可行”的边界，
以及小矩阵的必要整专家迁移。
