// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef PANGU_MX_PLANNED_WORKER_H_
#define PANGU_MX_PLANNED_WORKER_H_
#include "mx_task_flags.h"
namespace MulticoreRuntime {
class MxPlannedWorker : public MxTaskWorker<VectorWorkerPolicy::ALL> {
 public:
  __aicore__ inline void ProcessPlanned(bool gradients = false) {
    if (gradients != (this->replica_gradient_config_ != nullptr)) AscendC::Trap();
    ReadyHandshakeMeta meta;
    const bool has_ready = this->LoadReadyHandshakeMeta(&meta);
    if (meta.completion_event != 0 || this->core_num != 32) AscendC::Trap();
    if (has_ready) this->WaitForDeviceReady(meta);
#ifndef __DAV_CUBE__
    if (this->replica_gradient_config_ != nullptr && this->worker_id_ % 2 == 0 && this->worker_id_ < 8) {
      this->core_num = 4;
      this->ProcessReplicaGradients();
      return;
    }
#endif
    const auto mode = getDiagnosticMode(this->runtimeConfigPtr);
    if (mode == DiagnosticMode::PROFILE)
      Run<true>();
    else if (mode == DiagnosticMode::NORMAL)
      Run<false>();
    else
      AscendC::Trap();
  }

 private:
  template <bool Profiled>
  __aicore__ inline void Run() {
    uint32_t begin, end, stride;
#ifdef __DAV_CUBE__
    begin = this->worker_id_;
    end = this->cube_task_num;
    stride = 32;
#else
    auto offsets =
      reinterpret_cast<__gm__ int32_t *>(this->runtimeConfigPtr + getMixTaskIndexsOffset(this->runtimeConfigPtr));
    begin = offsets[this->worker_id_];
    end = offsets[this->worker_id_ + 1];
    stride = 1;
    if (begin > end || end > this->vector_task_num) AscendC::Trap();
#endif
    CycleTraceRecorder recorder;
    if constexpr (Profiled)
      recorder.Init(this->worker_id_, this->input_list[this->DIAGNOSTIC_IDX],
                    getAicProfileRecordCapacity(this->runtimeConfigPtr),
                    getAivProfileRecordCapacity(this->runtimeConfigPtr), getCubeWorkerCapacity(this->runtimeConfigPtr));
    if (begin == end) return;
    const uint32_t depth = ReadyDepth(PlannedTask(begin));
    const uint32_t total = (end - begin + stride - 1) / stride;
    if (depth && total > 64) AscendC::Trap();
    uint64_t finished = 0;
    uint32_t head = 0;
    while (head < total) {
      const uint32_t selected = depth ? SelectReady(begin, total, stride, depth, finished, head) : head;
      if (selected == total) continue;
      const TaskId id = PlannedTask(begin + selected * stride);
      if constexpr (Profiled)
        this->ExecuteTaskProfiled(id, recorder);
      else
        this->ExecuteTaskFast(id);
      if (depth) {
        finished |= uint64_t(1) << selected;
        while (head < total && (finished & (uint64_t(1) << head))) ++head;
      } else {
        ++head;
      }
    }
  }

  __aicore__ inline uint32_t ReadyDepth(TaskId id) {
    TaskDesc first;
    getTaskDesc(this->runtimeConfigPtr, &first, id);
    const uint32_t flags = first.extra_value_1;
    const bool depth8 = flags & MulticoreMx::MX_READY_DEPTH8;
    const bool depth1 = flags & MulticoreMx::MX_READY_DEPTH1;
    if (depth8 && depth1) AscendC::Trap();
    return depth8 ? 8 : depth1 ? 1 : 0;
  }

  __aicore__ inline TaskId PlannedTask(uint32_t position) {
#ifdef __DAV_CUBE__
    return this->cube_task_indexs.GetValue(position);
#else
    return this->vector_task_indexs.GetValue(position);
#endif
  }

  __aicore__ inline uint32_t SelectReady(uint32_t begin, uint32_t total, uint32_t stride,
                                      uint32_t depth, uint64_t finished, uint32_t head) {
    for (uint32_t lane = head; lane < total && lane < head + depth; ++lane) {
      if (finished & (uint64_t(1) << lane)) continue;
      TaskDesc task;
      getTaskDesc(this->runtimeConfigPtr, &task, PlannedTask(begin + lane * stride));
      if (task.dependent_event == EVENT_INVALID_ID) return lane;
      if (task.dependent_event >= this->runtime_event_capacity) AscendC::Trap();
      DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
          this->all_event_counters[task.dependent_event]);
      if (this->all_event_counters.GetValue(task.dependent_event) >=
          this->all_event_num_triggers.GetValue(task.dependent_event)) return lane;
    }
    const uint64_t until = AscendC::GetSystemCycle() + 10 * SYSTEM_CYCLES_PER_US;
    while (AscendC::GetSystemCycle() < until) {}
    return total;
  }
};
}  // namespace MulticoreRuntime
#endif
