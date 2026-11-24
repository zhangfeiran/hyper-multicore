// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "data_plane/rma.h"
#include "data_plane/sync.h"
#include "kernel_operator.h"
#include "shmem.h"

using namespace AscendC;
namespace shmem_data_plane = hyper_multicore::shmem::data_plane;
namespace hyper_multicore {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
template <typename T>
__aicore__ inline void PutIndexedRows(
    GM_ADDR target, int64_t target_offset, GM_ADDR source, __gm__ const int32_t *rows,
    int64_t first, int64_t count, int64_t source_rows, int64_t peer, __gm__ int32_t *signal,
    bool publish = true) {
#if !defined(__DAV_C220_CUBE__) && !defined(__DAV_CUBE__)
  constexpr int64_t hidden = 4096;
  constexpr int64_t row_bytes = hidden * sizeof(T);
  constexpr int64_t kPackets = 4;
  __gm__ aclshmem_device_host_state_t *state = aclshmemi_get_state();
  if (state->mte_config.aclshmem_ub != 0 || state->mte_config.ub_size != 2 * row_bytes) {
    AscendC::Trap();
  }
  // This is an isolated AIV task. Its two 32 KiB ping-pong packets and
  // 128 index words are disjoint, and no other AIV task shares its UB.
  constexpr int64_t kGroupBytes = kPackets * row_bytes;
  constexpr int64_t kBankBytes = 2 * kGroupBytes;
  constexpr int64_t kGroupIndexRows = 128;
  static_assert(kBankBytes + kGroupIndexRows * sizeof(int32_t) <= 192 * 1024);
  auto target_row = reinterpret_cast<__gm__ T *>(target) + target_offset;
  auto receive = reinterpret_cast<__gm__ T *>(shmem_data_plane::remote_ptr(target_row, peer));
  if (receive == nullptr) AscendC::Trap();
  auto input = reinterpret_cast<__gm__ T *>(source);
  auto ids = reinterpret_cast<__ubuf__ int32_t *>(kBankBytes);
  SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
  SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
  for (int64_t offset = 0; offset < count; offset += kGroupIndexRows) {
    const int64_t left = count - offset < kGroupIndexRows ? count - offset : kGroupIndexRows;
    aclshmemi_copy_gm2ub(ids, const_cast<__gm__ int32_t *>(rows) + first + offset, left * sizeof(int32_t));
    SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
    for (int64_t local = 0; local < left; local += kPackets) {
      const int64_t n = left - local < kPackets ? left - local : kPackets;
      const int64_t base = offset + local;
      const bool odd = (base / kPackets) % 2 != 0;
      const TEventID event = odd ? EVENT_ID1 : EVENT_ID0;
      auto packet = reinterpret_cast<__ubuf__ T *>(odd * kGroupBytes);
      WaitFlag<HardEvent::MTE3_MTE2>(event);
      for (int64_t i = 0; i < n; ++i) {
        const int64_t source_row = ids[local + i];
        if (source_row < 0 || source_row >= source_rows) AscendC::Trap();
        aclshmemi_copy_gm2ub(packet + i * hidden, input + source_row * hidden, row_bytes);
      }
      SetFlag<HardEvent::MTE2_MTE3>(event);
      WaitFlag<HardEvent::MTE2_MTE3>(event);
      aclshmemi_copy_ub2gm(receive + base * hidden, packet, n * row_bytes);
      SetFlag<HardEvent::MTE3_MTE2>(event);
    }
  }
  WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
  WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
  shmem_data_plane::fence();
  if (publish) shmem_data_plane::signal(signal, 1, shmem_data_plane::SignalOp::Add, peer);
#endif
}
#endif
}  // namespace hyper_multicore
