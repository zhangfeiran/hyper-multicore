# RFC：专家副本迁移的 EP32 性能对照

状态：2026-11-09 首版扫描发现回退；恢复默认非成本路径后完成全矩阵复测，
Push/Pull变化中位数+0.034%/-0.167%，未再观察到首版的整体回退。
下面先保留首版冻结结果，恢复后结果见[完整复测](#默认路径恢复后的完整复测)。
本文按 HyperParallel `003-RFC.yml` 组织，与[执行闭环](expert_replica_runtime.zh-CN.md)和
[迁移方案](expert_replica_migration.zh-CN.md)共同评审。

## 动机（Motivation）

迁移的性能验收固定为 EP32、E192、H5120、I1792、TopK8、BF16、4096 token/rank，
扫描均衡到热点的路由，直接比较来源 `expert-hot-replica` 与独立精简分支。
此前 H512/I512 的 FSDP 数据只用于该小形状的组合验证，不能承担本轮迁移性能验收。

## 目标设计

| 对象 | 冻结身份与设置 |
| --- | --- |
| 原分支 | `expert-hot-replica`，`22ad8d4b3308efba597ca670d2f16b8e4e750863`；成本关闭、CPU host 扩展开启 |
| 精简分支 | `feat/expert-replica-heuristic`，`4cc7e0c935556acdc5f7d33d7f0cff170383e7d6`；纯 Python 无表策略 |
| 精简实际包 | runtime `4b281054507c837595cf4daee3edb38c5d6d7494`、native `2c671b6bcaeac64becfb0d1f04c57b6b5f80c4ec`；与上述分支运行时代码相同 |
| 资源 | 同一物理 SuperPod 的两节点各16卡，共32rank，CANN9.1.0、PTA2.9.1、ascend910_93 |
| 共同策略 | B=1、默认 gradient bands、40% 准入、1024行软门槛，Push容量因子1.25、Pull默认值 |
| 比较范围 | 单层 MoE F+B 与 F+B+AdamW，不含 FSDP/checkpoint，不预先提供 histogram |

两个实现使用各自源码与匹配 native payload，分别重新启动独立进程，不混用运行库。
原分支保留当前 host 加速，避免通过关闭原有加速人为提高精简实现的相对表现。

路由为精确均衡锚点，以及 alpha10000、3000、1000、300、100、30、10、3、1、0.5、0.2、0.1、
0.05、0.02、0.01、0.005，每档 seed7/42/83，共49点。沿用来源的 Dirichlet 配额和无重复 TopK 构造，
所有 source 共享热点 profile；保留实际 owner-only max/mean 倾斜、source counts 和每rank IDs指纹。
有限 seed 的 owner 倾斜不保证随 alpha 严格单调。

每点恢复相同初始 owner 参数，清空 AdamW 状态，固定输入、上游梯度及 router 概率。
warmup5次、计时20次；每模式按 ABBA 再 BAAB 运行8次完整扫描，后4次反向遍历点序。
每实现每点4次复测，共80个全rank最大样本，全部保留。
每点的路由在warmup和计时中固定，保留组件正常的稳态复用行为；每step改变路由的重新规划开销不在本轮结论范围内。

## 需要设计评审 DFX 建议

### 计时和数值合同

主指标为设备同步包围的 CPU wall-clock F+B，全组取 MAX；包含计数、校验、规划、通信和反向。
副指标增加真实 AdamW 更新。计时前 barrier、zero_grad 和时延归约位于计时之外。
Heap epoch、capacity 和 inbox 在计时窗口保持不变，冷启动与扩容不进入稳态指标。

Push/Pull各对均衡、alpha1/seed42、alpha0.01/seed42做全张量跨进程资格检查，
覆盖 output、dx、router gradient、gate/up dW、down dW。六组×32rank全部通过，最大绝对误差和relative-L2均为0。
alpha0.01两模式实际23个transfer，两路kernel梯度归还均到达。
所有98个mode/route点的执行计划在两实现及八次复测之间一致。

### 实测性能

下表固定 seed42，单位ms；正变化表示精简实现更慢。每个时延为四次复测80个样本的中位数。

| Alpha | Push原分支 | Push精简 | Push变化 | Pull原分支 | Pull精简 | Pull变化 |
| --- | --- | --- | --- | --- | --- | --- |
| 精确均衡 | 28.050 | 28.210 | +0.57% | 28.743 | 28.696 | -0.16% |
| 10 | 33.209 | 33.630 | +1.27% | 34.487 | 34.569 | +0.24% |
| 1 | 55.376 | 57.950 | +4.65% | 56.150 | 57.774 | +2.89% |
| 0.1 | 63.327 | 64.106 | +1.23% | 57.690 | 58.605 | +1.59% |
| 0.01 | 66.014 | 66.761 | +1.13% | 61.094 | 62.176 | +1.77% |
| 0.005 | 66.508 | 67.528 | +1.53% | 59.067 | 60.163 | +1.86% |

49点F+B相对变化中位数为Push +1.23%、Pull +1.08%，点范围分别为-0.19%～+5.54%、-1.57%～+5.96%。
这只是测试集合各点等权统计，不代表生产路由加权的整网变化。
实际启用副本的21个Push点、23个Pull点全部出现点中位数回退，变化中位数为+1.26%、+1.57%。
alpha1/seed42的四组配对复测在两模式下全部回退；均衡点的配对变化跨越正负。

最大点Push为alpha1/seed7的+5.54%，Pull为alpha3/seed83的+5.96%；后者四组配对为
+8.41%、+1.22%、-2.20%、+10.70%，存在明显复测波动。报告保留全部慢样本和配对结果。
未预设等价容忍区间，也不事后设置门槛；当前不能将迁移状态标为“性能无回退”。

### 兼容性、解释与后续边界

本轮不修改算法、准入门槛和产品运行时代码。B1两边inbox配置均为None，
全部98点的allocator allocated峰值和SHMEM Heap字节一致，本轮没有按需inbox存储差异。
计划相同只能排除决策变化；host扩展、CPU准备或native执行等因素的因果归属需要独立单变量诊断。
不依据本轮总时延直接认定某个已移除模块是原因。

### 回退归因进展

后续诊断沿用本轮冻结counts，分别对旧版host开/关、精简版，以及两个单变量控制做两个fresh process镜像复测。
CPU控制路径使用相同核心、单线程、每点80样本；全部16个mode/route输入在各配置之间的完整计划一致。

- 无表compact-home准入失败短路被删除。alpha1/seed42实际不用副本时，CPU route准备原版Push/Pull为
  0.381/0.440ms，精简版为1.659/1.720ms。仅禁用原版短路就增加约0.57ms；此路径原版host开关基本不影响时延。
- Python整数校验由普通int快判断变为每count调用`_integer`，一次增加约6144次Python函数调用。
  仅恢复inline校验使精简版上述准备下降到1.121/1.186ms，无需成本模块或C++。
- 无表host整数原语有实际加速作用。alpha0.01/seed42仅关闭旧版host，使CPU planner从
  Push1.316→1.799ms、Pull1.334→1.841ms；精简版为2.354/2.398ms，恢复inline后为1.786/1.811ms。
- 来源默认`auto`在受测ABI4 payload上支持active gradient lanes，owner-only还选择ActGrad优先。
  精简版未保留这些默认执行优化；它们不应与默认关闭的研究队列混为一类。
- 原默认owner免清零证明没有接回。在已证明所有专家正K的home路径，额外全量清零对应315MiB/rank的BF16 dW写入；
  这是源码层面的流量差异，独立时延尚未做NPU单变量验证。

CPU诊断不包含真实D2H、NPU kernel或通信；不能把上述差值机械相加为全EP32回退的完整分解。
可选融合count库在受测旧包中未安装，所以不是本轮差异来源。
完整样本、可达函数、ABI/构图证据及解释见交付目录
`ep32-regression-diagnosis/REPORT-EP32-regression-diagnosis.md`。
本轮已恢复这些与成本模型无关的默认加速，见
[默认路径及研究变体 RFC](expert_replica_default_paths.zh-CN.md)。CPU 回归和6000组差分通过；
新 native 和 NPU 扫描结果另记，以上旧性能数据仍仅对应第一版精简实现。

这项跨分支测试回答迁移损失，不承担B0收益、成本模型收益、完整Trainer、FSDP大形状或峰值HBM验收。
allocator包含相同的初始权重快照，外部SHMEM Heap另记，physical HBM仅为采样值。

## 相关的 RFCs 和 API

不新增业务API。性能对照协议与运行程序保留于交付目录：

`/home/ma-user/work/z00961864/expert-replica-minimal-migration-20261109/ep32-alpha-comparison/`

- `REPORT-EP32-alpha-comparison.md`：完整结论、图、来源、复测与限制。
- `protocol.json`、`prepare_cases.py`、`route_definition.py`：冻结输入和采样定义。
- `worker.py`、`accuracy.py`、`run-comparison.sh`：双实现资格检查及完整扫描。
- `summary.json`、`comparison.csv`、`alpha-summary.md`：逐点、逐seed和复测统计。
- `ep32-alpha-comparison.svg/.png`：全部alpha的时延、相对变化和实际负载倾斜；阴影表示seed范围。
- `completion-audit.json`、`post-inventory.json`：全部rank终态、节点退出和32卡释放。

已有结果重新审核不需要设备：

```bash
cd /home/ma-user/work/z00961864/expert-replica-minimal-migration-20261109/ep32-alpha-comparison
python3 summarize.py
/home/ma-user/zfr/miniforge3/envs/pta/bin/python plot_results.py
```

## 完整的反馈期限

待维护者安排。

## CC List

待维护者补充。

## 其他补充说明

两节点均退出0，896份rank/stage记录完成；设备监控无foreign PID或健康异常。
逐卡终态确认两机各16卡健康、无NPU进程、锁空闲，任务拥有的bootstrap转发已关闭。
完整accuracy张量比较通过后删除，保留数值误差和指纹。全部native、协议和worker指纹结束后重新核对一致。
原来源工作树及用户的已有文档修改保留，分支未push。

## 默认路径恢复后的完整复测

运行时`613b6c5`，参考来源仍为`22ad8d4`；两边匹配各自native包，成本关闭。
恢复 compact-home、整数快校验、host原语、融合count、auto梯度及owner覆盖证明，
并保留跨节点将auto限定为W1的来源规则。配置、可达入口及研究控制见
[默认执行路径RFC](expert_replica_default_paths.zh-CN.md)。

EP32/E192/H5120/I1792/TopK8/BF16/T4096 per-rank、B1；每模式49点，
每点5 warmup/20 measured、ABBA+BAAB共8轮fresh process、每实现每点80样本。
统计先取各点80个rank-max F+B样本的中位数，再求49点相对变化的分布；正值表示新版更慢。

| 模式 | 变化中位数 | 变化均值 | 最快相对点 | 最慢相对点 | F+B+AdamW变化中位数 |
| --- | --- | --- | --- | --- | --- |
| Push | +0.034% | +0.045% | -0.94% | +1.03% | +0.012% |
| Pull | -0.167% | -0.204% | -2.17% | +1.16% | -0.210% |

98点计划指纹完全匹配。25,088条benchmark rank/case的实际梯度模式均为active W1，
逐rank逐点old/new一致；每case均执行26次融合count。六个全形状数值锚点×32rank的
output、dx、router及两投影dW逐元素误差0。初始操作数均核对完整字节SHA。
AdamW在计时扫描实际执行并检查version更新，未将锚点声明为AdamW状态逐元素验收。

首轮预扫描曾发现跨节点资格限制遗漏，主动终止、确认32卡释放后修正，并从头运行本表全矩阵。
停止实验不计入性能；本轮两边均启用各自host ABI3和融合count ABI2，
与首版两边没有融合count的包配置不同，不能把跨轮时延相减当作单变量归因。
未预设等价容差，未剔除慢样本；结论仅为当前矩阵未再观察到首版整体回退。

两节点31470、30468于20:32:21 +0800退出0，所属PID为空，post-inventory确认32卡健康、idle、
lock可获取。全程物理监测均无外来NPU PID，所属UID SSH转发已按PID代次/argv核对并关闭。
本矩阵的allocator allocated峰值及SHMEM heap字节逐点一致，未据此声称整网峰值HBM验收。

本地证据目录：
`/home/ma-user/work/z00961864/expert-replica-minimal-migration-20261109/default-path-migration/ep32-restored/`。
其中`REPORT-EP32-restored.md`、`comparison.csv`、`summary.json`、`completion-audit.json`及
`ep32-alpha-comparison.png/.svg`保留完整协议、98点、各rank终态和导出图。
EOI B0累积检查在本轮前冻结包也失败，另记，未作为本矩阵EIO验收通过。

## 路由准备性能调优

在上述固定 EP32 形状上，对均衡、alpha1/seed42 和 alpha0.01/seed42 分别做 Push/Pull
逐rank诊断。热点路径的 host wall span 显示统计读取、compact candidate 和后续计划构造均可见；
例如 Push alpha0.01 的32个rank中位数再取中位数，分别约为 0.858、0.539、1.004 ms。
这些是有嵌套关系的主机计时，可能包含等待设备的时间，不能相加当作精确耗时分解。

本轮针对不改变准入和路由结果的部分实施了以下优化：

- 节点流量计算复用已有 host 扩展的列求和；扩展不可用或整数超出原生范围时仍使用 Python 精确求和。
  6000 组含大整数和交错节点的独立差分通过。固定 EP32 输入的 CPU planner 对照中，
  强热点 case-44 的 Push/Pull 中位数约由 1.272/1.286 ms 降至 1.224/1.239 ms；
  这是 CPU 规划测试，不是完整层时延。
- 有 source controls 时不再提前计算未使用的完整 counts 表行和。固定输入的孤立 CPU 表达式
  中位数由 36.42 降至 1.87 微秒；该数值不包含设备、通信或模型计算。
- 恢复 B≥2 严重倾斜时共享已读主机表的路径，以及 opt-in checkpoint 热点重算读取路径。
  重算仍读取当前 counts、核对静态策略和精确计划，并按当前 token IDs 重映射；
  它们不依赖默认关闭的成本模型。EP8、B2 的 Push/Pull 已检查输出和全部相应梯度，
  但该小形状功能验证不能替代 EP32 性能验收。

对无条件提前读全表的独立原型，在 EP32 相同形状、Push/Pull 各7个路由点做8次
fresh-process ABBA+BAAB 对照。六组数值锚点的32个rank逐元素一致，全部计划一致；
F+B 相对变化范围 Push 为 -0.93%～+0.47%，Pull 为 -1.04%～+0.48%，配对方向并不稳定。
因此不将该原型设为默认策略，也不把它与上面的已实施优化混为一项。

进一步的独立原型将设备侧 summary 和每专家总数合并后一次读回。单卡固定 count 表的
组件测试中，原先 summary 加第二次总数读取的中位数为 0.300 ms，融合原型为 0.149 ms；
120 组设备整数和3000组共享算法的 CPU 独立校验通过。EP8、B2 功能检查中，Push/Pull
各8个rank实际调用融合统计并完成 checkpoint 热点重算，输出及对应梯度均与非重算对照逐元素一致。
组件测试不包含多rank collective 或 MoE F+B，不能据此推断端到端加速。

随后在指定 EP32 形状进行七点筛查，比较同一个干净构建中的融合读取和强制回退：356 个
package 文件只有控制该选择的函数不同，native payload 相同。每模式8轮 fresh process，
ABBA+BAAB，含反向点序，每点每实现保留80个 rank-max F+B 样本。下表为 seed42 的时延中位数，
正变化表示融合读取更慢；这是独立原型的筛查，不是来源分支与交付分支的49点验收。

| Alpha | Push回退 ms | Push融合 ms | Push变化 | Pull回退 ms | Pull融合 ms | Pull变化 |
| --- | --- | --- | --- | --- | --- | --- |
| 精确均衡 | 27.671 | 27.871 | +0.72% | 27.843 | 28.122 | +1.00% |
| 10 | 32.809 | 32.665 | -0.44% | 34.044 | 34.093 | +0.14% |
| 1 | 55.860 | 54.922 | -1.68% | 54.995 | 55.059 | +0.12% |
| 0.1 | 62.979 | 63.232 | +0.40% | 57.329 | 57.405 | +0.13% |
| 0.02 | 67.467 | 67.298 | -0.25% | 57.746 | 57.732 | -0.02% |
| 0.01 | 65.434 | 65.609 | +0.27% | 60.852 | 60.711 | -0.23% |
| 0.005 | 66.093 | 66.372 | +0.42% | 58.551 | 58.939 | +0.66% |

七点变化的中位数为 Push +0.27%、Pull +0.13%；这是测试点等权统计，不是生产路由加权收益。
各点的四组配对变化有正有负；例如均衡点 Push 为 +1.21%、+0.73%、+1.27%、-0.35%，
Pull 为 -0.50%、+3.94%、+2.07%、+1.32%。没有观察到稳定的整体收益，均衡点还出现回退，
因此不将该原型合入默认实现。

六组数值锚点 × 32 rank 的输出、dx、router及两投影 dW 逐元素一致；融合统计在每个锚点
均与当前完整 counts 的每专家总数一致，所有14个模式/路由点的计划指纹一致。
两节点正常退出、任务所属 PID 为空，全部监测采样无外来进程，结束后32卡健康空闲。
完整样本和审计见交付目录下 `screen-fused/`。

诊断、差分、全部7点样本及实验限制保存在：
`/home/ma-user/work/z00961864/expert-replica-performance-tuning-20261109/`。
本轮已实施代码的49点 EP32 全量对照在 Pull 资格阶段发生 HCCL Broadcast 超时，
未进入性能计时。已完成的 Push 三组和 Pull 均衡组均通过32 rank 数值检查；
失败阶段同节点还检测到任务外 NPU 进程，不能据此断定超时原因。
因此本轮不能评价已实施代码的完整 EP32 性能。
