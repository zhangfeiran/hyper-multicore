/*
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef HYPER_MULTICORE_OPS_RUNTIME_NORMAL_TASK_TRACE_LAYOUT_H_
#define HYPER_MULTICORE_OPS_RUNTIME_NORMAL_TASK_TRACE_LAYOUT_H_

#include <cstdint>

#if defined(__CCE_AICORE__)
#define HP_NORMAL_TRACE_INLINE __aicore__ inline
#else
#define HP_NORMAL_TRACE_INLINE inline
#endif

namespace MulticoreRuntime {

constexpr uint32_t NORMAL_TASK_TRACE_TAG = 0x4E545231;
constexpr uint64_t NORMAL_TASK_TRACE_MAGIC = 0x3152434152544EULL;
constexpr uint32_t NORMAL_TASK_TRACE_VERSION = 1;
constexpr uint32_t NORMAL_TASK_TRACE_MAX_CAPACITY = 16384;

struct NormalTaskTraceHeader {
  uint64_t magic;
  uint64_t entryCycle;
  uint64_t exitCycle;
  uint32_t recordCount;
  uint32_t droppedCount;
  uint32_t recordCapacity;
  uint32_t coreType;
  uint32_t workerId;
  uint32_t version;
  uint64_t reserved[2];
};

struct NormalTaskTraceRecord {
  uint64_t selectedCycle;
  uint64_t dependencyObservedCycle;
  uint64_t computeReturnedCycle;
  uint64_t eventPublishedCycle;
  uint32_t taskId;
  uint32_t taskType;
  uint32_t stageTaskIndex;
  uint32_t dependentEvent;
  uint32_t triggerEvent;
  uint32_t inputPosition;
  uint32_t splitNum;
  uint32_t splitValue;
};

static_assert(sizeof(NormalTaskTraceHeader) == 64);
static_assert(sizeof(NormalTaskTraceRecord) == 64);

HP_NORMAL_TRACE_INLINE uint64_t NormalTaskTraceSlotBytes(uint32_t capacity) {
  return sizeof(NormalTaskTraceHeader) + static_cast<uint64_t>(capacity) * sizeof(NormalTaskTraceRecord);
}

HP_NORMAL_TRACE_INLINE bool NormalTaskTraceWorkerValid(uint32_t worker, uint32_t workers, bool cube) {
  return worker < (cube ? workers / 2 : workers);
}

HP_NORMAL_TRACE_INLINE bool NormalTaskTraceStorageValid(uint32_t capacity, uint32_t workers,
                                                       uint32_t cube_slots, uint64_t buffer_bytes) {
  return capacity > 0 && capacity <= NORMAL_TASK_TRACE_MAX_CAPACITY && workers > 0 && workers % 2 == 0 &&
         cube_slots >= workers / 2 &&
         buffer_bytes >= (static_cast<uint64_t>(cube_slots) + workers) * NormalTaskTraceSlotBytes(capacity);
}

}  // namespace MulticoreRuntime

#undef HP_NORMAL_TRACE_INLINE

#endif  // HYPER_MULTICORE_OPS_RUNTIME_NORMAL_TASK_TRACE_LAYOUT_H_
