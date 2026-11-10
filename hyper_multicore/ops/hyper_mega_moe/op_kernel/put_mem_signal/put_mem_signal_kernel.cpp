/**
 * Copyright 2024-2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "data_plane/rma.h"
#include "data_plane/sync.h"
#include "kernel_operator.h"
#include "shmem.h"

using namespace AscendC;

namespace shmem_data_plane = hyper_multicore::shmem::data_plane;

namespace hyper_multicore {

constexpr uint32_t TOKEN_TRANSPORT_PROBE_MAGIC = 0x4D545031;
enum class TokenTransportProbeSlot : uint32_t {
  MAGIC = 0,
  REMOTE_MTE = 1,
  REMOTE_UDMA = 2,
  LOCAL_MTE = 3,
  EMPTY = 4,
  ROOT = 5,
  UB_BYTES = 6,
  CAP_MAP = 7,
};

__aicore__ inline void RecordTokenTransport(GM_ADDR probe, TokenTransportProbeSlot slot) {
  if (probe == nullptr) return;
  auto counters = reinterpret_cast<__gm__ uint32_t *>(probe);
  const uint32_t index = static_cast<uint32_t>(slot);
  counters[index] = counters[index] + 1;
}

__aicore__ inline void RecordMteScratchBytes(GM_ADDR probe, uint32_t bytes) {
  if (probe == nullptr) return;
  auto counters = reinterpret_cast<__gm__ uint32_t *>(probe);
  const uint32_t index = static_cast<uint32_t>(TokenTransportProbeSlot::UB_BYTES);
  if (counters[index] < bytes) counters[index] = bytes;
}

template <typename T>
__aicore__ inline void CopyGmSingleValueToUb(GM_ADDR gm_addr, T *result) {
  __ubuf__ T *ubAddr = (__ubuf__ T *)(32);
  smem_shm_copy_gm2ub(ubAddr, (__gm__ T *)gm_addr, sizeof(T));
  AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  *result = *ubAddr;
}

template <typename T>
class PutMemSignalKernel {
 public:
  __aicore__ inline PutMemSignalKernel() {}
  __aicore__ inline void Init(GM_ADDR target, int64_t target_offset, GM_ADDR src, int64_t src_offset, int64_t size,
                              GM_ADDR signal, int64_t signal_offset, int32_t signal_value, int64_t signal_op,
                              int64_t target_pe, bool non_blocking, GM_ADDR workspace, uint32_t udma_qp_limit = 0,
                              bool prefer_mapped_mte = false, GM_ADDR transport_probe = nullptr,
                              uint32_t mapped_mte_ub_bytes = 0) {
    // Initialize pointers and parameters
    target_ = reinterpret_cast<__gm__ T *>(target);
    src_ = reinterpret_cast<__gm__ T *>(src);
    signal_ = reinterpret_cast<__gm__ int32_t *>(signal);
    target_offset_ = target_offset;
    src_offset_ = src_offset;
    size_ = size;
    signal_offset_ = signal_offset;
    signal_value_ = signal_value;
    signal_op_ = signal_op;
    target_pe_ = target_pe;
    non_blocking_ = non_blocking;
    udma_qp_limit_ = udma_qp_limit;
    prefer_mapped_mte_ = prefer_mapped_mte;
    transport_probe_ = transport_probe;
    mapped_mte_ub_bytes_ = mapped_mte_ub_bytes;

    // UB tiling set
    aiv_idx_ = 0;
    aiv_num_ = 1;

    syncGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace), 256);
    pipe.InitBuffer(localwork, 1, sizeof(int32_t) * aiv_num_);
  }

  __aicore__ inline void Process() {
    // Empty tiles still satisfy remote dependencies without resolving data addresses.
    if (size_ == 0) {
      aclshmemx_mte_quiet();
      shmem_data_plane::signal(signal_ + signal_offset_, signal_value_,
                               signal_op_ == 1 ? shmem_data_plane::SignalOp::Add : shmem_data_plane::SignalOp::Set,
                               target_pe_);
      RecordTokenTransport(transport_probe_, TokenTransportProbeSlot::EMPTY);
      return;
    }
    // Perform data transfer
    auto size_per_core = size_ / aiv_num_;
    auto target_ptr = target_ + target_offset_ + aiv_idx_ * size_per_core;
    auto src_ptr = src_ + src_offset_ + aiv_idx_ * size_per_core;
    if (aiv_idx_ == aiv_num_ - 1) {
      size_per_core = size_ - size_per_core * aiv_idx_;
    }
    __gm__ aclshmem_device_host_state_t *device_state = aclshmemi_get_state();
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
    // Token traffic may use mapped MTE while replica transfers retain their UDMA queues.
    // Missing MTE capability or a peer mapping must preserve the existing UDMA path.
    const bool mapped_mte = prefer_mapped_mte_ &&
                            (device_state->topo_list[target_pe_] & ACLSHMEM_TRANSPORT_MTE) != 0 &&
                            device_state->p2p_device_heap_base != nullptr &&
                            reinterpret_cast<uintptr_t>(device_state->p2p_device_heap_base[target_pe_]) != 0;
    if (!mapped_mte && target_pe_ != aclshmem_my_pe() &&
        (device_state->topo_list[target_pe_] & ACLSHMEM_TRANSPORT_UDMA) != 0) {
      // Host validation gives each peer/QP pair one odd-worker owner.
      const uint32_t worker = AscendC::GetBlockIdx();
      if (worker % 2 == 0 || worker / 2 >= 32) {
        AscendC::Trap();
      }
      uint32_t qp_count = aclshmemi_udma_qp_info_fetch()->qp_num;
      if (qp_count != 1 && qp_count != 8 && qp_count != 16 && qp_count != 32) {
        AscendC::Trap();
      }
      if (udma_qp_limit_ != 0) {
        if (udma_qp_limit_ != 16 || qp_count != 32) {
          AscendC::Trap();
        }
        qp_count = udma_qp_limit_;
      }
      const uint32_t qp = worker / (64 / qp_count);
      AscendC::TBuf<AscendC::TPosition::VECOUT> staging;
      pipe.InitBuffer(staging, ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE);
      auto scratch = reinterpret_cast<__ubuf__ uint8_t *>(staging.Get<uint8_t>().GetPhyAddr());
      uint64_t remaining = size_per_core * sizeof(T);
      auto destination = reinterpret_cast<__gm__ uint8_t *>(target_ptr);
      auto source = reinterpret_cast<__gm__ uint8_t *>(src_ptr);
      while (remaining > 0) {
        const uint32_t bytes = remaining < (1ULL << 28) ? remaining : (1ULL << 28);
        aclshmemx_udma_qp_put_nbi<uint8_t>(destination, source, scratch, bytes, target_pe_, qp, 0);
        aclshmemx_udma_qp_quiet(target_pe_, qp);
        destination += bytes;
        source += bytes;
        remaining -= bytes;
      }
      shmem_data_plane::signal(signal_ + signal_offset_, signal_value_,
                               signal_op_ == 1 ? shmem_data_plane::SignalOp::Add : shmem_data_plane::SignalOp::Set,
                               target_pe_);
      RecordTokenTransport(transport_probe_, TokenTransportProbeSlot::REMOTE_UDMA);
      return;
    }
#endif
    /* CopyUB Config Set */
    uint64_t copy_ub = device_state->mte_config.aclshmem_ub;
    uint64_t copy_ub_size = device_state->mte_config.ub_size;
    PrepareMteScratch(copy_ub, copy_ub_size);
    __ubuf__ T *ping_buff = reinterpret_cast<__ubuf__ T *>(copy_ub);
    __ubuf__ T *pong_buff = reinterpret_cast<__ubuf__ T *>(copy_ub + copy_ub_size / 2);
    auto ptr = shmem_data_plane::remote_ptr(target_ptr, target_pe_);
    __gm__ T *remote_ptr = reinterpret_cast<__gm__ T *>(ptr);
    uint64_t block_size = copy_ub_size / 2 / sizeof(T) * sizeof(T);
    uint64_t remain = (size_per_core * sizeof(T)) % block_size;

    uint64_t repeat_times = (size_per_core * sizeof(T)) / block_size;
    uint64_t repeat_elem = block_size / sizeof(T);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint64_t i = 0; i < repeat_times; i++) {
      AscendC::TEventID EVENT_ID = (i & 1) ? EVENT_ID0 : EVENT_ID1;
      __ubuf__ T *buf = (i & 1) ? ping_buff : pong_buff;
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
      aclshmemi_copy_gm2ub(buf, src_ptr + i * repeat_elem, block_size);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
      aclshmemi_copy_ub2gm(remote_ptr + i * repeat_elem, buf, block_size);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
    }
    if (remain > 0) {
      AscendC::TEventID EVENT_ID = (repeat_times & 1) ? EVENT_ID0 : EVENT_ID1;
      __ubuf__ T *buf = (repeat_times & 1) ? ping_buff : pong_buff;
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
      aclshmemi_copy_gm2ub(buf, src_ptr + repeat_times * repeat_elem, remain);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
      aclshmemi_copy_ub2gm(remote_ptr + repeat_times * repeat_elem, buf, remain);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

    // Write signal after data transfer
    auto signal_ptr = signal_ + signal_offset_;
    if (non_blocking_) {
      aclshmemx_mte_quiet();
    } else {
      aclshmemx_mte_quiet();
    }
    // Sync ensure corresponding tasks are done
    if (aiv_num_ > 1) {
      AscendC::LocalTensor<int32_t> workLocal = localwork.AllocTensor<int32_t>();
      AscendC::SyncAll(syncGlobal, workLocal, aiv_num_);
      localwork.FreeTensor(workLocal);
    }

    if (aiv_idx_ == 0) {
      shmem_data_plane::signal(signal_ptr, signal_value_,
                               signal_op_ == 1 ? shmem_data_plane::SignalOp::Add : shmem_data_plane::SignalOp::Set,
                               target_pe_);
      RecordTokenTransport(transport_probe_, target_pe_ == aclshmem_my_pe() ? TokenTransportProbeSlot::LOCAL_MTE
                                                                          : TokenTransportProbeSlot::REMOTE_MTE);
      RecordMteScratchBytes(transport_probe_, static_cast<uint32_t>(copy_ub_size));
    }
  }

 private:
  __aicore__ inline void PrepareMteScratch(uint64_t &address, uint64_t &bytes) {
    if (mapped_mte_ub_bytes_ == 0) return;
    if (mapped_mte_ub_bytes_ != 32768 && mapped_mte_ub_bytes_ != 65536) AscendC::Trap();
    // Keep the SDK scratch unchanged for indexed ingress and other communication helpers.
    pipe.InitBuffer(mapped_mte_staging_, mapped_mte_ub_bytes_);
    address = reinterpret_cast<uint64_t>(mapped_mte_staging_.Get<uint8_t>().GetPhyAddr());
    bytes = mapped_mte_ub_bytes_;
  }

  __gm__ T *target_;
  __gm__ T *src_;
  __gm__ int32_t *signal_;
  AscendC::TPipe pipe;

  int64_t target_offset_ = 0;
  int64_t src_offset_ = 0;
  int64_t size_ = 0;
  int64_t signal_offset_ = 0;
  int32_t signal_value_ = 0;
  int64_t signal_op_ = 0;
  int64_t target_pe_ = 0;
  bool non_blocking_ = true;
  uint32_t udma_qp_limit_ = 0;
  bool prefer_mapped_mte_ = false;
  GM_ADDR transport_probe_ = nullptr;
  uint32_t mapped_mte_ub_bytes_ = 0;
  AscendC::TBuf<AscendC::TPosition::VECOUT> mapped_mte_staging_;
  AscendC::GlobalTensor<int32_t> syncGlobal;
  AscendC::TQue<AscendC::TPosition::VECIN, 1> localwork;
  // ub
  uint32_t aiv_idx_ = 0;
  uint32_t aiv_num_ = 0;
};

}  // namespace hyper_multicore

extern "C" inline __aicore__ void put_mem_signal_kernel(GM_ADDR target, int64_t target_offset, GM_ADDR src,
                                                        int64_t src_offset, int64_t size, GM_ADDR signal,
                                                        int64_t signal_offset, int64_t signal_value, GM_ADDR workspace,
                                                        int64_t signal_op, int64_t target_pe, bool non_blocking,
                                                        uint32_t udma_qp_limit = 0,
                                                        bool prefer_mapped_mte = false,
                                                        GM_ADDR transport_probe = nullptr,
                                                        uint32_t mapped_mte_ub_bytes = 0) {
  hyper_multicore::PutMemSignalKernel<DTYPE_DISPATCH_TARGET> op;
  op.Init(target, target_offset, src, src_offset, size, signal, signal_offset, signal_value, signal_op, target_pe,
          non_blocking, workspace, udma_qp_limit, prefer_mapped_mte, transport_probe, mapped_mte_ub_bytes);
  op.Process();
}
