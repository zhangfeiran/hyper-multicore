/** Copyright 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
 */
#ifndef HYPER_MULTICORE_RUNTIME_GRADIENT_ORDER_POLICY_H_
#define HYPER_MULTICORE_RUNTIME_GRADIENT_ORDER_POLICY_H_

#include <cstdint>

#if defined(__CCE_AICORE__)
#define HP_GRADIENT_POLICY_INLINE __aicore__ inline
#else
#define HP_GRADIENT_POLICY_INLINE inline
#endif

namespace MulticoreRuntime {
constexpr uint32_t GRADIENT_ORDER_MAX_EXPERTS = 64;
constexpr uint32_t GRADIENT_ORDER_MAGIC = 0x5A000000;
constexpr uint32_t GRADIENT_ORDER_MAGIC_MASK = 0xFF000000;
constexpr uint32_t GRADIENT_ORDER_EVENT_MASK = 0x000FFFFF;
constexpr uint32_t GRADIENT_LANE_MARKER = GRADIENT_ORDER_MAGIC | (5U << 20);
constexpr uint32_t GRADIENT_SHAPE_POLICY = 8;
constexpr uint32_t GRADIENT_SHAPE_MARKER = GRADIENT_ORDER_MAGIC | (GRADIENT_SHAPE_POLICY << 20);
constexpr uint32_t GRADIENT_ACTIVE_LANE_MARKER = GRADIENT_ORDER_MAGIC | (9U << 20);
constexpr uint32_t GRADIENT_LANE_DISABLED = UINT32_MAX;

// Rotate the remainder tiles across experts while preserving a one-to-one
// assignment for every individual GEMM. Workspace ownership stays physical.
HP_GRADIENT_POLICY_INLINE uint32_t RotateGradientLane(
    uint32_t physical, uint32_t cores, uint32_t tiles, uint32_t expert) {
  return (physical + expert * (tiles % cores)) % cores;
}

// The policy uses immutable row counts. Every Cube worker therefore makes the
// same choice without a host readback, a new collective, or a shared queue lock.
HP_GRADIENT_POLICY_INLINE bool BuildGradientExpertOrder(
    const int64_t *rows, uint32_t experts, uint32_t policy, uint32_t *order) {
  for (uint32_t i = 0; i < experts; ++i) {
    order[i] = i;
  }
  if (policy == 4) {
    int64_t total = 0, maximum = 0;
    for (uint32_t i = 0; i < experts; ++i) {
      total += rows[i];
      if (rows[i] > maximum) maximum = rows[i];
    }
    return experts > 0 && rows[0] * experts > 2 * total && rows[0] == maximum;
  }
  if (policy == GRADIENT_SHAPE_POLICY) {
    int64_t total = 0, maximum = 0, second = 0, minimum = INT64_MAX;
    for (uint32_t i = 0; i < experts; ++i) {
      const int64_t count = rows[i];
      total += count;
      if (count >= maximum) {
        second = maximum;
        maximum = count;
      } else if (count > second) {
        second = count;
      }
      if (count < minimum) minimum = count;
    }
    // The host selects the measured 4096-token, top-8 geometry. Require
    // 1.5x its nominal owner load and at least two substantial GEMMs.
    if (total < 49152 || second < 4096 || maximum > 8 * second || maximum - minimum < 4096) return false;
    // Keep the ordinary schedule for moderately skewed owners with no
    // dominant GEMM. Admit either a deep individual GEMM or a heavy queue.
    if (total < 98304 && (maximum < 28672 || 2 * maximum < 5 * second)) return false;
  }
  for (uint32_t i = 1; i < experts; ++i) {
    const uint32_t expert = order[i];
    uint32_t j = i;
    while (j > 0 && (policy == 2 ? rows[expert] > rows[order[j - 1]] : rows[expert] < rows[order[j - 1]])) {
      order[j] = order[j - 1];
      --j;
    }
    order[j] = expert;
  }
  return policy == 3 || policy == GRADIENT_SHAPE_POLICY;
}
}  // namespace MulticoreRuntime
#undef HP_GRADIENT_POLICY_INLINE
#endif  // HYPER_MULTICORE_RUNTIME_GRADIENT_ORDER_POLICY_H_
