// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../../runtime/cycle_trace_recorder.h"

namespace MulticoreMoE {
constexpr uint32_t kRecomputeTraceRecordsPerWorker = 1024;

enum class RecomputeStage : uint32_t {
  FORWARD = 1,
  DISPATCH_X_G_P,
  W2,
  DP,
  PG,
  DP_AND_PG,
  RETURN_DP,
  FORWARD_BARRIER,
  BACKWARD_TASKS,
  ACT_GRAD,
  RETURN_DX,
  MERGE_GRAD,
  WAIT_GRAD_ACK,
  EARLY_ACT_GRAD,
};

__aicore__ inline uint64_t RecomputeTraceStart(MulticoreRuntime::CycleTraceRecorder *trace) {
  return trace == nullptr ? 0 : trace->Now();
}

__aicore__ inline void RecordRecomputeStage(MulticoreRuntime::CycleTraceRecorder *trace,
                                           RecomputeStage stage, uint64_t start,
                                           uint32_t index = 0, uint32_t rows = 0, uint32_t peer = 0) {
  if (trace != nullptr) {
    trace->Record(static_cast<uint32_t>(stage), index, rows, peer, start, trace->Now());
  }
}
}  // namespace MulticoreMoE
