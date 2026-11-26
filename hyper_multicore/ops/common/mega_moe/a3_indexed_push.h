// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef HMC_MOE_A3_INDEXED_PUSH_H_
#define HMC_MOE_A3_INDEXED_PUSH_H_
#include <stdint.h>

namespace MulticoreMoE {
constexpr uint32_t A3_INDEXED_MAX_TOP_K = 8;
constexpr uint32_t A3_INDEXED_MAX_HIDDEN = 8192;
constexpr uint32_t A3_INDEXED_HIDDEN_ALIGNMENT = 16;
constexpr int64_t A5_INDEXED_TOP_K = 8;
constexpr int64_t A3_INDEXED_TOP_K_SHIFT = 17;
constexpr int64_t A3_INDEXED_TOP_K_MASK = 15;
constexpr uint32_t A3_INDEXED_MAX_TOKENS = 16384;
constexpr int64_t A3_INDEXED_SOURCE_BITS = 21;
constexpr int64_t A3_INDEXED_SOURCE_MASK = (int64_t{1} << 16) - 1;
constexpr int64_t A3_INDEXED_ROUTE_FENCED = int64_t{1} << 16;
constexpr int32_t A3_INDEXED_TILING_SOURCE_ROWS = 8;
constexpr int32_t A3_INDEXED_TILING_ROUTE_FENCED = 9;
constexpr int32_t A3_INDEXED_TILING_SOURCE_SLOTS = 10;
}  // namespace MulticoreMoE

#endif  // HMC_MOE_A3_INDEXED_PUSH_H_
