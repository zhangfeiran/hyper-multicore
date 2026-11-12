/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HYPER_MULTICORE_OPS_RUNTIME_RDMA_TRANSPORT_H_
#define HYPER_MULTICORE_OPS_RUNTIME_RDMA_TRANSPORT_H_

#include "kernel_operator.h"
#include "shmem.h"

namespace MulticoreRuntime {

constexpr uint64_t RDMA_RUNTIME_MAGIC = 0x3152444D4153484DULL;
constexpr uint64_t RDMA_RUNTIME_BYTES = 64;

class RdmaTransport {
 public:
  __aicore__ inline void Init(GM_ADDR runtime, uint64_t &bytes, uint32_t ep, uint32_t events) {
    if (bytes < RDMA_RUNTIME_BYTES) return;
    auto trailer = reinterpret_cast<__gm__ uint64_t *>(runtime + bytes - RDMA_RUNTIME_BYTES);
    if (trailer[0] != RDMA_RUNTIME_MAGIC) return;
    bytes -= RDMA_RUNTIME_BYTES;
    enabled = true;
    ep_ = ep;
    events_ = events;
    credits_ = reinterpret_cast<__gm__ int32_t *>(trailer[1]);
    progress_ = reinterpret_cast<__gm__ int32_t *>(trailer[2]);
    values_ = reinterpret_cast<__gm__ int32_t *>(trailer[3]);
    value_count_ = trailer[4];
    force_roce_ = (trailer[5] & 1) != 0;
    publish_control_ = (trailer[5] & 2) != 0;
    row_indices = reinterpret_cast<__gm__ int32_t *>(trailer[6]);
    source_rows = trailer[7];
    auto state = aclshmemi_get_state();
    rank_ = state->mype;
    if (ep_ != state->npes || !credits_ || !progress_ || !values_ || value_count_ < 2 || !source_rows) {
      AscendC::Trap();
    }
#ifndef __DAV_C220_CUBE__
    AscendC::SetAtomicNone();
#endif
  }

  __aicore__ inline bool UsesRdma(uint32_t peer) const {
    return peer != rank_ && (force_roce_ ||
      (aclshmemi_get_state()->topo_list[peer] & ACLSHMEM_TRANSPORT_MTE) == 0);
  }

  __aicore__ inline int32_t Received(uint32_t event) const {
    int32_t total = 0;
    for (uint32_t peer = 0; peer < ep_; ++peer) {
      if (!UsesRdma(peer)) continue;
      auto counter = credits_ + static_cast<uint64_t>(peer) * events_ + event;
      dcci_cachelines(reinterpret_cast<GM_ADDR>(counter), sizeof(int32_t));
      total += *reinterpret_cast<__gm__ volatile int32_t *>(counter);
    }
    return total;
  }

  __aicore__ inline void Put(GM_ADDR destination, GM_ADDR source, uint32_t bytes, uint32_t peer) {
#ifndef __DAV_C220_CUBE__
    if (bytes) Transfer<false>(destination, source, bytes, peer);
#endif
  }

  __aicore__ inline void Get(GM_ADDR destination, GM_ADDR source, uint32_t bytes, uint32_t peer) {
#ifndef __DAV_C220_CUBE__
    if (bytes) Transfer<true>(destination, source, bytes, peer);
#endif
  }

  __aicore__ inline void Quiet(uint32_t peer) {
#ifndef __DAV_C220_CUBE__
    aclshmemx_roce_quiet(peer, Scratch(), 0);
#endif
  }

  __aicore__ inline void Credit(uint32_t event, int32_t amount, uint32_t peer) {
#ifndef __DAV_C220_CUBE__
    auto progress = progress_ + static_cast<uint64_t>(peer) * events_ + event;
    dcci_cachelines(reinterpret_cast<GM_ADDR>(progress), sizeof(int32_t));
    int32_t value = *progress + amount;
    if (value <= 0 || static_cast<uint64_t>(value) >= value_count_) AscendC::Trap();
    *progress = value;
    dcci_cachelines(reinterpret_cast<GM_ADDR>(progress), sizeof(int32_t));
    // Values are initialized once, remain symmetric and never change while a WQE can read them.
    Put(reinterpret_cast<GM_ADDR>(credits_ + static_cast<uint64_t>(rank_) * events_ + event),
        reinterpret_cast<GM_ADDR>(values_ + value), sizeof(int32_t), peer);
#endif
  }

  __aicore__ inline void SignalSet(__gm__ int32_t *destination, __gm__ int32_t *source, uint32_t peer) {
#ifndef __DAV_C220_CUBE__
    if (UsesRdma(peer)) {
      Put(reinterpret_cast<GM_ADDR>(destination), reinterpret_cast<GM_ADDR>(source), sizeof(int32_t), peer);
      Quiet(peer);
    } else {
      aclshmemx_signal_op(destination, *source, ACLSHMEM_SIGNAL_SET, peer);
    }
#endif
  }

  __aicore__ inline void PublishSignalSource(__gm__ int32_t *source) {
    if (publish_control_) PublishControlSource(source);
  }

  __aicore__ inline __gm__ int32_t *IndexScratch(uint32_t peer) const {
    auto base = progress_ + static_cast<uint64_t>(ep_) * events_;
    return base + peer * ((values_ - base) / ep_);
  }

  bool enabled = false;
  __gm__ int32_t *row_indices = nullptr;
  uint64_t source_rows = 1;

 private:
  __aicore__ inline GM_ADDR RegisteredRemoteAddress(GM_ADDR symmetric, uint32_t bytes, uint32_t peer) const {
    auto state = aclshmemi_get_state();
    const uint64_t base = reinterpret_cast<uint64_t>(state->heap_base);
    const uint64_t address = reinterpret_cast<uint64_t>(symmetric);
    if (peer >= ep_ || address < base || bytes > state->heap_size) AscendC::Trap();
    const uint64_t offset = address - base;
    if (offset > state->heap_size - bytes) AscendC::Trap();
    auto info = aclshmemi_qp_info_fetch();
    auto region = reinterpret_cast<__gm__ aclshmemi_rdma_mem_info *>(info->mem_ptr) + peer;
    if (bytes > region->size || offset > region->size - bytes) AscendC::Trap();
    // P2P mappings belong to this process and can differ from the peer's RNIC-registered VA.
    // The SDK's generic RoCE NBI wrappers translate through aclshmem_ptr(), which assumes equal VAs.
    return reinterpret_cast<GM_ADDR>(region->addr + offset);
  }

  template <bool Read>
  __aicore__ inline void Transfer(GM_ADDR destination, GM_ADDR source, uint32_t bytes, uint32_t peer) {
#ifndef __DAV_C220_CUBE__
    auto scratch = Scratch();
    AscendC::LocalTensor<uint32_t> words;
    words.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    words.address_.bufferAddr = reinterpret_cast<uint64_t>(scratch);
    words.address_.dataLen = UB_ALIGN_SIZE;
    AscendC::LocalTensor<uint64_t> double_words;
    double_words.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    double_words.address_.bufferAddr = reinterpret_cast<uint64_t>(scratch) + UB_ALIGN_SIZE;
    double_words.address_.dataLen = UB_ALIGN_SIZE;
    if constexpr (Read) {
      aclshmemi_roce_read(destination, RegisteredRemoteAddress(source, bytes, peer), peer, 0,
                          bytes, double_words, words, 0);
    } else {
      aclshmemi_roce_write(RegisteredRemoteAddress(destination, bytes, peer), source, peer, 0,
                           bytes, double_words, words, 0);
    }
#endif
  }

  __aicore__ inline void PublishControlSource(__gm__ int32_t *source) {
#ifndef __DAV_C220_CUBE__
    auto state = aclshmemi_get_state();
    if (state->mte_config.ub_size < 32) AscendC::Trap();
    AscendC::LocalTensor<int32_t> value;
    value.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    value.address_.bufferAddr = state->mte_config.aclshmem_ub;
    value.address_.dataLen = 32;
    value.SetValue(0, *reinterpret_cast<__gm__ volatile int32_t *>(source));
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID2);
    // Publish the scalar generation to GM before the RNIC can read this immutable source word.
    aclshmemi_copy_ub2gm(source, reinterpret_cast<__ubuf__ int32_t *>(value.GetPhyAddr()), sizeof(int32_t), false);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
#endif
  }

  __aicore__ inline __ubuf__ uint8_t *Scratch() const {
    // Dedicated owners execute no compute tasks; the ready coordinator hands off only after quiet.
    // Borrow the configured SDK area rather than the compiler's reserved high UB addresses.
    auto state = aclshmemi_get_state();
    if (state->mte_config.ub_size < 128) AscendC::Trap();
    return reinterpret_cast<__ubuf__ uint8_t *>(state->mte_config.aclshmem_ub);
  }
  __gm__ int32_t *credits_ = nullptr;
  __gm__ int32_t *progress_ = nullptr;
  __gm__ int32_t *values_ = nullptr;
  uint64_t value_count_ = 0;
  uint32_t rank_ = 0;
  uint32_t ep_ = 0;
  uint32_t events_ = 0;
  bool force_roce_ = false;
  bool publish_control_ = false;
};

}  // namespace MulticoreRuntime
#endif  // HYPER_MULTICORE_OPS_RUNTIME_RDMA_TRANSPORT_H_
