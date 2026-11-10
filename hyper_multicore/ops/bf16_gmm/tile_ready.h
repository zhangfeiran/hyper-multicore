/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under CANN Open Software License Agreement Version 1.0.
 * See LICENSE in the root of the software repository for the full text.
 */
#ifndef HYPER_MULTICORE_BF16_GMM_TILE_READY_H_
#define HYPER_MULTICORE_BF16_GMM_TILE_READY_H_

namespace MulticoreBf16 {
constexpr uint32_t INPUT_TILE_ROWS = 128;
constexpr uint32_t TILE_READY_TAG = 0x4D54494C;

__aicore__ inline uint64_t InputTileStride(uint64_t tokens, uint64_t top_k, uint64_t ep) {
  return (ep * tokens * top_k + INPUT_TILE_ROWS - 1) / INPUT_TILE_ROWS;
}

__aicore__ inline uint64_t InputTileTableOffset(uint64_t tokens, uint64_t top_k, uint64_t experts) {
  uint64_t slots = tokens * top_k;
  uint64_t statistics_rows = 8 * experts * ((slots + 127) / 128);
  return (slots + statistics_rows) * 4 * sizeof(int32_t);
}

__aicore__ inline void WaitForInputRows(__gm__ int32_t *table, uint64_t stride, uint32_t expert,
                                      int64_t first_row, int64_t rows, int64_t expert_rows) {
  if (table == nullptr || stride == 0 || first_row < 0 || rows <= 0 || first_row + rows > expert_rows) {
    AscendC::Trap();
  }
  uint64_t first = first_row / INPUT_TILE_ROWS;
  uint64_t last = (first_row + rows - 1) / INPUT_TILE_ROWS;
  if (last >= stride) { AscendC::Trap(); }
  AscendC::GlobalTensor<int32_t> counters;
  counters.SetGlobalBuffer(table + expert * stride, stride);
  for (uint64_t tile = first; tile <= last; ++tile) {
    int64_t remaining = expert_rows - tile * INPUT_TILE_ROWS;
    int32_t needed = remaining < INPUT_TILE_ROWS ? remaining : INPUT_TILE_ROWS;
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                     AscendC::DcciDst::CACHELINE_OUT>(counters[tile]);
    int32_t current = counters.GetValue(tile);
    int64_t previous = AscendC::GetSystemCycle();
    while (current < needed) {
      // A5 system cycles are fixed at 1000 per microsecond. Read-only cache
      // entries must refresh while independent AIV producers publish rows.
      if (AscendC::GetSystemCycle() - previous >= 10000) {
        AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                         AscendC::DcciDst::CACHELINE_OUT>(counters[tile]);
        current = counters.GetValue(tile);
        previous = AscendC::GetSystemCycle();
      }
    }
  }
}
}  // namespace MulticoreBf16
#endif
