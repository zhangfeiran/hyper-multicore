/**
 * Copyright 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied.
 */
#ifndef HYPER_MULTICORE_REPLICA_TRACE_H
#define HYPER_MULTICORE_REPLICA_TRACE_H

#include "cycle_trace_recorder.h"

namespace MulticoreRuntime {

enum class ReplicaTraceEvent : uint32_t {
  PRODUCER_READY = 1,
  HEAD_PENDING = 2,
  READY_OBSERVED = 3,
  RETURN_WORK = 4,
  INDEPENDENT_READY = 5,
  OWNER_MERGED = 6,
  ACK_ISSUED = 7,
  ACK_OBSERVED = 8,
  EMPTY_READY = 9,
  LOCAL_NOT_READY = 10,
  GUEST_NOT_READY = 11,
};

// The existing 32-byte record ABI carries protocol identities, never remote timestamps.
__aicore__ inline void RecordReplica(CycleTraceRecorder &trace, ReplicaTraceEvent event,
                                    uint32_t projection, uint32_t band, uint32_t peer, uint32_t epoch,
                                    uint32_t rank, uint32_t expert, uint64_t start, uint64_t end,
                                    uint32_t producer_core = 63) {
  trace.Record(0x11000 + projection * 0x100 + static_cast<uint32_t>(event),
               (peer << 14) | (producer_core << 8) | band, epoch, (rank << 16) | expert, start, end);
}

}  // namespace MulticoreRuntime
#endif  // HYPER_MULTICORE_REPLICA_TRACE_H
