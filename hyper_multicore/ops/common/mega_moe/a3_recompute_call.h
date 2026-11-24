// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>

namespace MulticoreMoE {
constexpr int64_t kA3RecomputeSequence = int64_t{1} << 40;
constexpr uint64_t kA3RecomputeCall = 0x41335245434f4d52ULL;
constexpr uint32_t kA3ForwardInputs = 26;
constexpr uint32_t kRecomputeReadyWordsPerExpert = 16;
constexpr uint32_t kMaxEarlyExpertsPerRank = 32;

// All addresses are valid for the entire compound device launch.
struct A3RecomputeCall {
  uint64_t magic;
  uint64_t forward_inputs[kA3ForwardInputs];
  uint64_t local_barrier;
  uint64_t source_g;
  uint64_t received_g;
  uint64_t source_p;
  uint64_t received_p;
  uint64_t local_dp;
  uint64_t returned_dp;
  uint64_t trace;
  uint64_t pg_ready;
  uint64_t gradient_rows_per_batch;
};
static_assert(sizeof(A3RecomputeCall) == 37 * sizeof(uint64_t));
}  // namespace MulticoreMoE
