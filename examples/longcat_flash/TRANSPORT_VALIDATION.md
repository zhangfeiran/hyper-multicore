# LongCat 通信兼容验证

验证日期：2026-11-12。以下结果对应独立 HPM 源码，不要求 HyperParallel。

## 配置与验收边界

- NPU：Ascend910_9382（910C），16 个逻辑设备；CPU：aarch64。
- 设备间拓扑：HCCS/SIO。已检查的可用机器池没有 PCIe 跨域设备组合。
- PyTorch：2.9.1+cpu；torch-npu：2.9.1；CANN：9.1.0。
- 全局 EP16，每卡 2048 tokens，H=6144，I=2048，TopK=12。
- 全局 128 个真实专家、64 个恒等专家，即每卡 8 个真实专家、4 个恒等专家。
- BF16，外部 EOI 权重；Gate/Up 包括恒等专家占位，形状为 `[12,4096,6144]`，
  Down 为 `[8,6144,2048]`。保留原宽度，只调整测试层数。
- 使用 fused SwiGLU；主参考为同语义原生 NPU permute/All-to-All/GMM/SwiGLU 路径。
  本次没有将 literal HF 逐专家结果改作主门限，也没有修改既有精度门限。
- 热专家副本关闭。HCCL 兼容路径目前明确拒绝非零副本数。

原生构建记录的源码摘要为
`2369b6e1ea602efda3aea63eda8b1bd70c173203e5c908ea27dd2e221d06e71e`。
该摘要覆盖 HPM Python/C++、构建脚本和锁定依赖配置。机器路径、模型权重和编译二进制不属于此次 Git 交付。

## 终态结果

每项都检查了全部 16 个 rank 的终态报告和每步结果；下表误差为所有 rank、所有可比较张量的最大值。

| 案例 | 执行范围 | max_abs | relative_l2 | 结果 |
| --- | --- | --- | --- | --- |
| 默认 SHMEM 通信 | 原宽 EP16，1 次前向和反向 | 0 | 0 | 全 rank 通过 |
| HCCL 通信与两个专家层 | 两次连续前向保留计算图，再联合反传 | 0 | 0 | 全 rank 通过 |
| HCCL push，本地 MegaMoE 计算 | 10 次独立 FP32 master 更新，含热点、空源、全恒等路由 | 0 | 0 | 全 rank 通过，更新后参数一致 |
| HCCL pull，本地 MegaMoE 计算 | 原宽 EP16，1 次前向和反向 | 0 | 0 | 全 rank 通过 |

两层检查额外确认每卡本地 SHMEM Root 大小为 1，而全局 EP 仍为 16。
主误差项覆盖输出、输入梯度、路由梯度、Gate/Up 梯度和 Down 梯度。
所有进程退出为 0；所有 rank 的 SHMEM 状态恢复为 `Uninitialized`、引用计数为 0。
节点退出后确认全部 16 个设备健康、无设备进程且设备锁已释放。

另外，`check_transport.py --exercise both` 在全部 rank 检查了真实 SDK MTE 可达性、
逐 peer 的 MTE Put、Host SDMA Put，以及各次搬运后的远端 Add 计数器。
两类数据和计数器都正确；每条传输为 16 KiB，这是通信功能检查，不是模型性能测量。
模拟缺失 MTE peer 的保护检查也通过。这些结果属于 HCCS/SIO 拓扑，不能作为 PCIe 跨域验收。

CPU 回归为 171 项测试通过，另有 701 个子用例；新增传输契约覆盖不可达 peer、
配置校验、梯度契约不一致、共享关闭与多层预登记。原生 SHMEM Host 测试 7 项通过，
新绑定已编译、加载并在实际 NPU 上读取可达性。
本地共享文件系统在更新已发布目录时返回 `renameat2(RENAME_EXCHANGE)` 的 `EINVAL`；
恢复时先检查构建源码摘要、完整产物校验值和原生 manifest，再通过已有发布函数发布到已清空的目标目录。
原始构建失败记录和恢复校验回执均保留。

## 复现与目标机验收

构建、探测、单层及两层命令见 [使用说明](README.md#mte-不互通的-ep-组与-910b2c-16-卡)。
`--exercise-edge-routes --updates 10` 可重现上述路由边界与参数更新检查。
客户需在自己的 x86_64、910B2C、PyTorch/torch-npu/CANN 环境中从当前源码重新编译，
不能复用 aarch64 原生二进制。

该记录验证专家组件与两层专家计算图，不代表完整 LongCat Trainer 或客户真实权重验收。
当前没有真实 910B2C PCIe 跨域实验，也没有测量该兼容路径的性能或通信计算重叠收益。
已修复的是无 MTE 可达性检查的问题，并提供显式 HCCL 通信路径；客户首步超时的实际根因仍需其拓扑与首个错误日志确认。
