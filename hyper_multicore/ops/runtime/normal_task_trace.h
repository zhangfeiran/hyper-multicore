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

#ifndef HYPER_MULTICORE_OPS_RUNTIME_NORMAL_TASK_TRACE_H_
#define HYPER_MULTICORE_OPS_RUNTIME_NORMAL_TASK_TRACE_H_

#include "kernel_operator.h"
#include "core_type.h"
#include "runtime_config.hpp"
#include "normal_task_trace_layout.h"

namespace MulticoreRuntime {

class NormalTaskTrace {
 public:
  __aicore__ inline void Init(GM_ADDR buffer, uint64_t bytes, __gm__ RuntimeHeader *runtime, uint32_t worker) {
    const uint32_t capacity = runtime->padding[1];
    const uint32_t cube_slots = getCubeWorkerCapacity(reinterpret_cast<GM_ADDR>(runtime));
#ifdef HMC_RUNTIME_CUBE
    constexpr bool cube = true;
#else
    constexpr bool cube = false;
#endif
    if (!NormalTaskTraceWorkerValid(worker, runtime->num_workers, cube) ||
        runtime->diagnostic_mode != static_cast<uint32_t>(DiagnosticMode::NORMAL) || buffer == nullptr ||
        reinterpret_cast<uint64_t>(buffer) % 64 != 0 || runtime->padding[2] != 0 || runtime->padding[3] != 0 ||
        !NormalTaskTraceStorageValid(capacity, runtime->num_workers, cube_slots, bytes)) {
      AscendC::Trap();
      return;
    }
    const uint32_t type = cube ? 1 : 2;
    const uint32_t slot = cube ? worker : cube_slots + worker;
    header_ = reinterpret_cast<__gm__ NormalTaskTraceHeader *>(buffer + slot * NormalTaskTraceSlotBytes(capacity));
    header_->magic = NORMAL_TASK_TRACE_MAGIC;
    header_->entryCycle = Now();
    header_->exitCycle = 0;
    header_->recordCount = 0;
    header_->droppedCount = 0;
    header_->recordCapacity = capacity;
    header_->coreType = type;
    header_->workerId = worker;
    header_->version = NORMAL_TASK_TRACE_VERSION;
    header_->reserved[0] = 0;
    header_->reserved[1] = 0;
    capacity_ = capacity;
  }

  __aicore__ inline bool Enabled() const { return header_ != nullptr; }

  __aicore__ inline uint64_t Now() const {
    return static_cast<uint64_t>(AscendC::GetSystemCycle());
  }

  __aicore__ inline void Record(TaskId task_id, const TaskDesc &task, uint64_t selected, uint64_t observed,
                               uint64_t computed, uint64_t published) {
    if (count_ == capacity_) {
      ++dropped_;
      return;
    }
    auto record = reinterpret_cast<__gm__ NormalTaskTraceRecord *>(header_ + 1) + count_;
    record->selectedCycle = selected;
    record->dependencyObservedCycle = observed;
    record->computeReturnedCycle = computed;
    record->eventPublishedCycle = published;
    record->taskId = task_id;
    record->taskType = static_cast<uint32_t>(task.task_type);
    record->stageTaskIndex = task.task_index;
    record->dependentEvent = task.dependent_event;
    record->triggerEvent = task.trigger_event;
    record->inputPosition = task.num_inputs > 0 ? task.inputs[0].input_position : 0xFFFFFFFF;
    record->splitNum = task.task_split_num;
    record->splitValue = task.task_split_value;
    ++count_;
  }

  __aicore__ inline void Finish() {
    if (!Enabled()) {
      return;
    }
    header_->exitCycle = Now();
    header_->recordCount = count_;
    header_->droppedCount = dropped_;
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::GlobalTensor<uint64_t> storage;
    storage.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(header_));
    for (uint32_t line = 0; line <= count_; ++line) {
      AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                       AscendC::DcciDst::CACHELINE_OUT>(storage[line * 8]);
    }
  }

 private:
  __gm__ NormalTaskTraceHeader *header_ = nullptr;
  uint32_t capacity_ = 0;
  uint32_t count_ = 0;
  uint32_t dropped_ = 0;
};

}  // namespace MulticoreRuntime

#endif  // HYPER_MULTICORE_OPS_RUNTIME_NORMAL_TASK_TRACE_H_
