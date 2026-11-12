# LongCat 跨 MTE 域的 SHMEM 单边通信研究

这是独立的传输协议原型，使用本仓库锁定的 SHMEM 源码构建 RDMA SDK，不依赖 HyperParallel。
原型验证 AIV 在内核中提交 RoCE 数据与完成通知，以及接收端按专家提前观察数据就绪。
当前生产 MegaMoE 仍使用 MTE 数据搬运和远端 Add 计数器，尚未接入此协议。

## 为什么继续使用 SHMEM

当前 `communication_backend="hccl"` 在外层 dispatch All-to-All 完成后进入单 PE
MegaMoE，再执行外层 combine All-to-All。本地 GMM/SwiGLU/dW 融合仍在，但按专家就绪
启动计算的跨卡流水被拆开了。该兼容实现没有通过目标机器的性能验收。

PCIe 域间无法直接使用 MTE，并不等于 SHMEM 无法进行单边通信。SHMEM 还有 RoCE
RDMA 引擎；设备 RNIC 和网络需要支持并连通，CPU 的 x86_64/aarch64 架构不能单独决定结果。
上游 [SHMEM 使用限制](https://github.com/hicann/shmem/blob/master/docs/debug/Troubleshooting_FAQs.md)
说明 A2 16 卡机器的两个 8P Fullmesh 组之间不能直接使用 MTE。
实际客户拓扑应由探测确认，不能只根据 16 卡或 910B2C 名称推断。

当前锁定版本为 `d30702bb55d0c9ce028a241ea608ae38dbaebc8e`，生产 SDK 构建没有开启
`ACLSHMEM_RDMA_SUPPORT`，运行时也只接受 MTE。原型在独立构建目录开启 RDMA，不替换已交付的运行库。

## 协议约束与实现

锁定源码中的 `include/device/gm2gm/shmem_device_rma.h` 明确限制 RDMA 对同一 PE
的并发提交；`src/device/gm2gm/engine/rdma_backends/rdma_device_backend_in_die.hpp`
明确不支持 IN_DIE 后端的原子操作。因此不能把多 AIV 的 MTE Put/Add 原样切到 RDMA。

原型采用以下协议：

1. 每个目标 peer/QP 固定一个 AIV 提交，数据与完成通知使用同一个 QP。
2. 源数据保持为对称内存中的原始 `[2048,6144]` BF16 等字节布局，合计 24 MiB。
   按路由行索引直接发送，每行 12 KiB；发送端不生成 `[TopK*T,H]` 展开张量。
3. 每完成一个专家的数据提交，就在同一 QP 上写入一个 4 字节的前缀完成位图。
   通知记录包含 generation 和 8 个专家位，不使用远端 Add。
   每个源 rank 写接收端自己的独立槽位，避免多源覆盖。
4. 每个专家的通知源记录独立且不可变，所有未完成 WQE 的源地址都保持有效。
   接收端看到当前 generation 下所有源的专家位后，立即读取该专家的数据；后续专家继续传输。
5. 每个目标 QP 在整个批次末尾执行 quiet。零行仍发布通知。
   主机只在所有内核完成且完成跨 rank 同步后复用源数据、通知记录与接收区。

`kernel.cpp` 使用显式 `aclshmemx_roce_put_nbi` 和 `aclshmemx_roce_quiet`，不会因
物理机器同时具备 MTE 能力而走 MTE 数据捷径。RoCE 模式下 SDK 的 capability mask
仍可能同时含 MTE 与 RoCE；mask 表示可用能力，不能用它独立证明实际数据通道。

## 已完成的 16 卡实验

2026-11-12 使用 Ascend Cluster Run 预留完整空闲节点，实际硬件为 aarch64、
910C（Ascend910_9382）、HCCS/SIO，环境为 Python 3.11、Torch/torch-npu 2.9.1、CANN 9.1.0。
这验证了显式 RoCE 内核协议，尚无 x86_64、910B2C 的真实 PCIe 跨域验收。

| 检查 | 配置 | 全 rank 结果 |
| --- | --- | --- |
| MTE 对照原语 | 0、16 KiB、1.5 MiB、24 MiB，含 warmup | 768 个 rank 案例通过 |
| RoCE 数据和通知原语 | 同上，分别检查两次 quiet 与仅末尾 quiet | 1536 个 rank 案例通过 |
| 8 个 QP owner、8 专家完成位图 | 零行与每源每专家 128 行 | 256 个 rank 案例通过 |

每次检查都包含不同 generation 的全量数据逐位比对，并在通知就绪时立即采样，防止只在
内核整体结束后读取数据而漏掉通知过早的问题。传输载荷用 uint32 编码校验原始位，
没有进行 BF16 数学计算。上述计数包含 warmup，以下时间统计排除 warmup。

在按索引发送实验中，每个 rank 的 8 个 AIV 向另一组 8 个 peer 发送数据，
总接收量为 96 MiB，覆盖 8 个专家，每专家每源 128 行。原始源仍为 24 MiB，
接收缓冲区与发送端展开不是同一项；另预留了 256 MiB 对称 heap，SDK 队列/MR 元数据单独占用资源。
每个目标 QP 发送 1024 个行 WQE 和 8 个通知 WQE，未使用仅适用于其他后端的批量接口。

6 个正式迭代的同次 rank-max event 时间中位数为 **4.425 ms**。
96 个正式 rank 样本中，首专家就绪时全部仍有后续专家未就绪；首末专家就绪间隔为
**3.300–3.734 ms，中位数 3.557 ms**。这说明协议可以向计算侧提供提前就绪窗口。
观察者是 AIV，尚未接入 Cube GMM，因此该窗口不能直接作为节省的训练时间。
也不能用这个单边原型与单 AIV MTE 原语的时间差推导 MegaMoE 加速比。

所有 rank 完成并关闭 SHMEM；节点退出为 0，随后检查 16 个设备健康、无设备进程、锁已释放。
压缩后的全 rank 审计记录、产物与源码摘要见 [实验结果](RESULTS_20261112.json)。
`audit.py` 从各 rank 的 event 时间重新计算同次 rank 最大值，再对正式迭代取中位数，
同时拒绝缺失 rank、失败终态、未关闭 runtime、重复 case 或产物 SHA 不一致。

## 构建和运行

在机器本地激活自己的 CANN、PyTorch/torch-npu 环境。x86_64 必须本地编译；
仓库不交付上述 aarch64 二进制。当前实测 CANN 为 9.1.0，其他环境需要独立验证。
进程只加载原型自己的私有 SDK，不需要激活生产 MegaMoE 自定义算子环境。

先通过仓库既有依赖准备流程得到锁定 SHMEM 源码。`build.py` 对源码和适配补丁摘要
进行核对，通过 `git archive` 在新的目录展开副本，不会修改原始 SDK 源码。

```bash
probe_root="$PWD/examples/longcat_flash/one_sided_probe"
probe_build="$PWD/build/one-sided-rdma"
python "$probe_root/build.py" \
  --shmem-source "$PWD/build/native/deps/shmem/src" \
  --build-root "$probe_build" --jobs 8

export LD_LIBRARY_PATH="$probe_build/payload/lib:${LD_LIBRARY_PATH:-}"
python -m torch.distributed.run --standalone --nproc-per-node=16 \
  "$probe_root/run.py" --engine roce \
  --library "$probe_build/payload/lib/liblongcat_one_sided_probe.so" \
  --output ./one-sided-pair --warmup 2 --samples 10

python -m torch.distributed.run --standalone --nproc-per-node=16 \
  "$probe_root/pipeline_run.py" \
  --library "$probe_build/payload/lib/liblongcat_one_sided_probe.so" \
  --output ./one-sided-pipeline --warmup 2 --samples 6

python "$probe_root/audit.py" --kind pipeline \
  --reports ./one-sided-pipeline --payload "$probe_build/payload" \
  --output ./one-sided-pipeline-audit.json
```

这些脚本用 HCCL 做 UID 分发、迭代前同步、结果汇合；被测内核内部的数据、通知和等待
都由 SHMEM 完成。MTE 对照使用 `run.py --engine mte`，只能在已确认对应 peer
MTE 可达的拓扑执行。每次 SDK 初始化方式或引擎变化都使用新进程。

## 接入 MegaMoE 的具体边界

目标是在同一个融合内核中按 peer 选择：可达域内保留 MTE，跨域使用 AIV 直驱 RDMA。
选择必须依据实际 SDK 拓扑，不能写死逻辑 rank 的前后各 8 卡；原型里的两组划分仅用于可控验证。
通信和 GMM 都仍属于 MegaMoE 的设备任务图，专家一就绪即可启动 GMM1，
随后推进 SwiGLU、GMM2 和按专家返回结果。

需要调整的实现入口包括：

- `hyper_multicore/shmem/_build/shmem_sdk.sh` 与 `shmem/ccsrc/runtime/`：
  RDMA SDK 构建、运行时引擎、peer 能力与队列生命周期。
- `hyper_multicore/tasks/alltoall.py` 与 `ops/runtime/worker_kernel.h`：
  每 peer 的 AIV 所有权、任务事件和跨源完成位图；保留域内 MTE 完成依赖。
- dispatch、combine、反向 dX 通信：分别维护源数据 ready、数据消费完成、源区复用和最终 quiet。
  combine 还要等待该专家 GMM2 的生产完成，不能在某个 AIV 阻塞等待时占住所有后续任务的执行资源。
- 源激活放入对称区，保留原始行和路由索引；如果来自普通 Torch 内存，单独核算原始源 staging。
  不引入发送端 TopK 展开；接收缓存、原始源 staging、heap 和 SDK 元数据分别报告。

本原型尚未覆盖 MegaMoE 的完整前反向、dW、优化器或热专家副本路径。
后续模型验收继续保持 EP16、每卡 8 个真实专家和 4 个恒等专家、H6144/I2048、
T2048/TopK12、外部 EOI、fused SwiGLU，以原生 NPU GMM 路径验收输出和全部梯度及更新。
性能比较必须在目标拓扑对同语义、同权重和路由的完整工作负载进行，提前就绪只作为机制证据。
