# RFC：按依赖闭包复用原生编译组件

状态：已迁移到新的多架构构建流程。缓存按架构、后端和 ABI 隔离；历史研究分支的耗时仅作为动机，
不代表当前接口基线的构建或设备验收。当前分支的验证记录随 PR 提供。

## 动机（Motivation）

先fetch并核查最新上游master a11aaff3a97b7bd986c9bbcc434f462c93f6df22。
上游已有整包vendor快照、多PTA适配器并行，ec1e445新增共享jobs预算下的多设备架构并行。
这些机制不重复实现；当前单设备架构的GMM改动仍会使整包vendor失效，SDK/ops和适配器仍重建。
缓存键仍记录源码和SDK绝对路径、完整PATH，不能直接跨隔离worktree复用。

本实现保留接口分支的 `prepare/vendor/merge` 阶段、多架构并行、共享 jobs 预算和多 SHMEM 后端。
目标是只编译变化的依赖闭包，保持注册、加载、ELF、ABI 与完整 payload 校验。

## 实现方案

1. 保留官方CANN注册、host编译、打包与校验流程，用CMake自定义命令launcher缓存六个算子的二进制输出。
   键来自组装后的实际算子源码、公共host合同、生成配置、工具链及SDK；GMM改动只影响MoE前后向。
   命中时用官方工具重新生成当前构建的动态入口，不复制含旧绝对include路径的动态Python文件。
2. 分别复用SHMEM SDK、SHMEM ops和Torch/SHMEM适配器。
   适配器键包含真实框架头文件、链接库、Python/框架ABI和公开vendor头；不绑定无关kernel实现或全仓commit。
   命中仍进行当前配置、SHMEM host CTest、实际加载和ELF检查。
3. 缓存身份使用内容和逻辑相对名；原始源码路径、commit、工具路径及各输出SHA另存来源记录。
   默认同一用户的共享缓存，也允许HYPER_MULTICORE_NATIVE_CACHE指定位置；不复制或放宽已有worktree所有权标记。
   外部工具和ABI仍核查，未知或未覆盖环境采用保守重建。
4. 命中前检查成员集合、SHA和目录所有权，损坏为miss；并发使用逐键锁和原子发布。
   成功编译只暂存组件，整体payload提交后核查实际输入再发布共享缓存；失败不发布本次暂存。
   --clean绕过读取已有组件，不清空其他worktree共用的缓存。原有发布、失败和取消语义保持。
5. build_info、阶段记录和缓存回执逐组件列出命中、编译、键、输出SHA及实际生产来源。
   当前源码身份与复用组件生产身份分别报告，不把复用记为本次重新编译。

## 需要设计评审DFX建议

CPU覆盖源码/SDK/工具/ABI失效、GMM局部失效、跨路径同内容命中、损坏/不完整/重定向拒绝、
并发发布、未成功整体构建不发布及clean绕过。按仓库add-unit-test流程补行为测试。
真实构建验证冷构建、完全命中、GMM变化和第二worktree复用，记录每个组件的实际编译动作和全payload来源。
保留所有当前ELF、CTest、加载、wheel及源码守恒检查；本项不申请NPU、不做运行时性能或浮点精度验收。

单个大kernel的阶段计时及模板去重属于后续编译研究；本轮先解决重复工作，不调整优化等级或提高jobs。
耗时结论只使用本轮同条件构建，不用旧469.7秒与新构建直接推算改动的独立收益。

## 相关RFC和API

- [MoE 通用执行优化](moe_execution_optimizations.zh-CN.md)
- [当前原生构建说明](../build.md)
- 上游master的vendor_cache.py、build_controller.py及ec1e445已逐项核查。

## 完整的反馈期限

内部构建改进，完成验证后供维护者评审；不自动发布或push。

## CC List

建议由native构建、CANN kernel及框架适配器维护者评审依赖闭包、缓存失效和产物来源。

## 历史研究验证记录

本节记录原研究分支的验收，涉及的源码和环境与当前接口整合分支不同，不作为当前分支验收。

核心源码eaecbe9387575c77ce77796a11b8868afc30f9d4，独立研究worktree完成以下同条件构建。
CANN9.1.0、Torch2.9.1+cpu、PTA2.9.1、cp311、aarch64、ascend910_93，全部jobs12。
每场景一个完整进程，耗时包含配置、编译/复用、加载、ELF、wheel和提交后维护；不把单次观测当作稳定均值。

| 场景 | 总耗时/秒 | 组件命中 | 实际OPC任务 |
| --- | ---: | ---: | ---: |
| 强制冷构建 | 495.5 | 0/11 | 10 |
| 同目录命中 | 68.7 | 11/11 | 0 |
| 新worktree命中 | 108.8 | 11/11 | 0 |
| 仅GMM补丁变化 | 409.5 | 9/11 | 6 |

同目录耗时减少86.14%，跨worktree减少78.04%，GMM局部改动减少17.36%，均相对此表同轮冷构建。
没有将旧469.7秒与新构建直接相减作为因果收益，也不与master历史多架构数据比较。
GMM夹具42aae42只修改锁定补丁的一行注释并更新SHA，真实组装/编译/打包重新执行，不改变计算token。
它是失效验证夹具，不是运行时候选，不迁入主分支。

新worktree使用独立editable环境；构建期间临时移开生产者common/variants编译目录，11组件仍全部命中并完成加载/打包，之后目录原样恢复。
命中组件key、输出字节与实际生产来源完整保持；GMM修改只有MoE前后向新key/新生产来源，其他9组件保持。
完整wheel含13共享组件、10 kernel对象；原生payload含12共享组件，wheel另生成host计数扩展。
独立审计核对各封存包、wheel与payload的完整成员和SHA，不仅检查成功摘要。

后续3b771f0补齐准确日志、来源时长与当前时长分开，以及缺CANN版本记录/不支持编译器身份探测的保守回退。
不能确认可复用身份时关闭组件缓存，继续正常编译；不新增CANN版本准入。
当前命中的compile_seconds为0，快照历史放snapshot_measurements，整包查找成本由vendor阶段记录。
最终源码强制冷构建473.0秒、同源码命中81.1秒，耗时减少82.85%，11组件命中、OPC任务0。
两轮22个native二进制SHA一致，命中当前compile_seconds均零，快照历史单列；
该交付版数据不与上表不同源码轮次混算，也不将冷构建差别解释成独立优化收益。
构建专项131项/49子例，全量CPU503项/2602子例通过；命中仍执行配置、CTest、实际加载、ELF及wheel验证。

- [完整报告与全部阶段](/home/ma-user/work/z00961864/replica-build-cache-20261114/REPORT.md)
- [最终交付源码与冷/热独立核验](/home/ma-user/work/z00961864/replica-build-cache-20261114/delivery-audit.json)
- [四场景独立审计](/home/ma-user/work/z00961864/replica-build-cache-20261114/build-case-audit.json)
- [跨worktree不依赖旧编译目录证明](/home/ma-user/work/z00961864/replica-build-cache-20261114/cross-origin-unavailable-proof.json)

## 后续编译方向与停止边界

全部命中仍需输入扫描、vendor host注册/编译、配置探针、加载和wheel；不宣称整个进程零编译。
仅改GMM仍要编译6个大MoE对象，因此任务数减少不按比例等同耗时减少。
下一项先按工具链/ABI在每次构建内共享一次内容扫描清单，再由组件引用，并在提交后对输入并集复核。
不以跨构建mtime近似替代SHA。随后补单大kernel内部阶段计时，再决定模板去重，不先提高jobs或降低优化等级。
这些后续编译项尚未实施；运行时调优继续pending，没有新增NPU性能或浮点精度验收。
