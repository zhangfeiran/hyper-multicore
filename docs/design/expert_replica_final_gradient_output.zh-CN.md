# RFC：固定FP32归并顺序与最终BF16梯度

## 动机（Motivation）

FP32 guest梯度归并后再由autograd转换整个home输出，会额外物化完整FP32矩阵。
方案在保持peer加法顺序的前提下，由最后一个peer直接生成最终BF16块，
并只为实际owned专家保留紧凑FP32 scratch。

## 目标设计

| 专家角色 | producer写出 | 类型 | 消费者 |
| --- | --- | --- | --- |
| home、无远端贡献 | 最终home切片 | BF16 | 原autograd叶子 |
| home、有远端贡献 | 紧凑scratch切片 | FP32 | 原归还worker |
| guest | 对称guest梯度矩阵 | FP32 | 远端owner |

每个投影持有独立矩阵与epoch。CPU根据实际transfer表生成 `home_to_scratch` 和 `last_edges`，
不依据alpha或测量表选策略。零表示直接home输出，正值为一基scratch索引。

owner继续按稳定peer顺序执行FP32 Add。非最后边写回FP32 scratch；最后边在同一UB中完成加法后
转换BF16并写最终home，不再物化无后续消费者的FP32结果。
同专家的边可能不相邻，最后标记必须来自完整边表，不能只比较相邻owner。

原band分组和worker条带保证输出不重叠。最终MTE3写出后才能发布completion，
全worker quorum后发ACK。guest lease在ACK前保持有效。

### 原生能力和回退

- protocol7保留原FP32 band归还。
- protocol11提供无owned任务rank的直接BF16输出。
- protocol13编码最终home指针、紧凑scratch、home映射和每边最后标记。
- `mega_moe_local_gradient_version()==1`与 `mega_moe_final_gradient_version()==1`分别协商两种能力。

不能从输出指针猜协议，也不能将旧local能力当作final能力。
默认根据实际库协商，旧库保留已有回退；显式强制缺失能力时失败。
缺少双投影完整producer/band证明时共同保留FP32输出合同。

## 需要设计评审DFX建议

归并之前所有guest贡献保持FP32，每个输出的peer顺序不变，不以提前BF16累加换取速度。
紧凑scratch是容量合同的一部分；提前新增完整FP32 scratch与完整BF16输出会改变峰值，必须独立测量。
owner本地为空时显式清零对应scratch；缺少真实覆盖证明时不能省略初始化。

主要实现为 `final_gradients.py` 的映射、`kernel_gradients.py` 的protocol13记录、
反向worker的home/guest指针选择，以及 `replica_gradient.h` 的最后peer写出。
原NORMAL队列、每块事件依赖、投影完成、quorum和ACK保持。
研究中的projection缓冲复用、取消逐peer排空和其他失败候选不属于此实现。

CPU检查最后peer、映射合法性、冷专家、双投影选择和旧库拒绝；设备检查整数/有限值、实际协议和生命周期。
数值验收以对应源码及库的设备结果为准，CPU夹具不是跨设备数值证明。

## 相关RFCs和API

关联[执行闭环](expert_replica_runtime.zh-CN.md)及[默认执行控制](expert_replica_default_paths.zh-CN.md)。
公共返回仍为原owner参数形状和dtype，不增加guest参数；性能统计边界见[性能协议](expert_replica_performance.zh-CN.md)。
