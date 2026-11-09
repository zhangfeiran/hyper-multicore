# RFC：专家副本迁移的 EP32 性能对照

状态：2026-11-09 完成指定形状的跨分支扫描；数值和计划一致，精简实现仍有时延回退。
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

本轮不修改算法、准入门槛和产品运行时代码。旧分支按需inbox与精简分支共享B上限存储保留各自默认。
计划相同只能排除决策变化；host扩展、CPU准备或native执行等因素的因果归属需要独立单变量诊断。
不依据本轮总时延直接认定某个已移除模块是原因。

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
