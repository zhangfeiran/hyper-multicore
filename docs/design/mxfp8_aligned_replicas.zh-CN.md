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
