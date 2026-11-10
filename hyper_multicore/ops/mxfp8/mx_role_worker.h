// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef PANGU_MX_ROLE_WORKER_H_
#define PANGU_MX_ROLE_WORKER_H_
namespace MulticoreRuntime {

template <bool Ready, bool Rule = false>
class MxRoleWorker : public MxTaskWorker<VectorWorkerPolicy::ALL> {
 public:
  __aicore__ inline void ProcessRoles() {
    ReadyHandshakeMeta meta;
    const bool has_ready = this->LoadReadyHandshakeMeta(&meta);
    if (meta.completion_event != 0 || this->core_num != 32) AscendC::Trap();
    if (has_ready) this->WaitForDeviceReady(meta);
    const auto mode = getDiagnosticMode(this->runtimeConfigPtr);
    if (mode == DiagnosticMode::PROFILE) {
      RunPool<true>();
    } else if (mode == DiagnosticMode::TENSOR_DUMP_L1) {
      AscendC::Trap();
    } else {
      RunPool<false>();
    }
  }

 private:
  uint32_t pool_begin_ = 0;
  uint32_t pool_count_ = 0;
  uint32_t lane_ = 0;

  __aicore__ inline TaskId PoolTask(uint32_t position) {
#ifdef __DAV_CUBE__
    return this->cube_task_indexs.GetValue(position);
#else
    return this->vector_task_indexs.GetValue(pool_begin_ + position);
#endif
  }

  __aicore__ inline bool IsReady(TaskId id) {
    if (id >= this->runtime_task_capacity) AscendC::Trap();
    const uint32_t dependency =
      *reinterpret_cast<__gm__ uint32_t *>(this->runtimeConfigPtr + getAllTasksOffset(this->runtimeConfigPtr) +
                                           id * sizeof(TaskDesc) + offsetof(TaskDesc, dependent_event));
    if (dependency == EVENT_INVALID_ID) return true;
    if (dependency >= this->runtime_event_capacity) AscendC::Trap();
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      this->all_event_counters[dependency]);
    return this->all_event_counters.GetValue(dependency) >= this->all_event_num_triggers.GetValue(dependency);
  }

  template <bool Profiled>
  __aicore__ inline void Execute(TaskId id, CycleTraceRecorder &recorder) {
    if constexpr (Profiled) {
      this->ExecuteTaskProfiled(id, recorder);
    } else {
      this->ExecuteTaskFast(id);
    }
  }

  __aicore__ inline uint32_t LaneStart(uint32_t begin) { return begin + (lane_ + 32 - begin % 32) % 32; }

  template <bool Profiled>
  __aicore__ inline void RunPool() {
    const uint32_t comm_count = *reinterpret_cast<__gm__ int32_t *>(
      this->runtimeConfigPtr + getTaskIndexNumOffset(this->runtimeConfigPtr) + 3 * sizeof(int32_t));
    if (comm_count == 0 || comm_count > this->vector_task_num || comm_count % 2 != 1) AscendC::Trap();
#ifdef __DAV_CUBE__
    lane_ = this->worker_id_;
    pool_count_ = this->cube_task_num;
#else
    lane_ = this->worker_id_ / 2;
    const bool communication = this->worker_id_ % 2 != 0;
    pool_begin_ = communication ? 0 : comm_count;
    pool_count_ = communication ? comm_count : this->vector_task_num - comm_count;
#endif
    CycleTraceRecorder recorder;
    if constexpr (Profiled) {
      recorder.Init(this->worker_id_, this->input_list[this->DIAGNOSTIC_IDX],
                    getAicProfileRecordCapacity(this->runtimeConfigPtr),
                    getAivProfileRecordCapacity(this->runtimeConfigPtr), getCubeWorkerCapacity(this->runtimeConfigPtr));
    }
    if constexpr (!Ready) {
      for (uint32_t position = lane_; position < pool_count_; position += 32)
        Execute<Profiled>(PoolTask(position), recorder);
      return;
    }
    uint32_t begin[4] = {}, end[4] = {}, cursor[4] = {};
    uint32_t head_count = 2;
    uint32_t terminal = UINT32_MAX;
    bool fair_communication = false;
    if constexpr (Rule) {
#ifdef __DAV_CUBE__
      const uint32_t pool = 0;
#else
      const uint32_t pool = communication ? 1 : 2;
      fair_communication = communication;
#endif
      // Only launch v7 interprets the otherwise inactive mix queue as ranges.
      auto ranges =
        reinterpret_cast<__gm__ int32_t *>(this->runtimeConfigPtr + getMixTaskIndexsOffset(this->runtimeConfigPtr)) +
        pool * 10;
      head_count = ranges[0];
      terminal = static_cast<uint32_t>(ranges[1]);
      if (head_count == 0 || head_count > 4) AscendC::Trap();
      for (uint32_t index = 0; index < head_count; ++index) {
        begin[index] = ranges[2 + 2 * index];
        end[index] = ranges[3 + 2 * index];
        if (begin[index] >= end[index] || end[index] > pool_count_) AscendC::Trap();
      }
      if (terminal != UINT32_MAX && terminal >= pool_count_) AscendC::Trap();
    } else {
#ifdef __DAV_CUBE__
      auto launch = reinterpret_cast<__gm__ MulticoreMx::MxLaunchDescriptor *>(this->input_list[this->TILING_IDX]);
      const bool backward = launch->buffer_count == 33;
      head_count = backward ? 4 : 2;
      if (pool_count_ % head_count != 0) AscendC::Trap();
      const uint32_t stage = pool_count_ / head_count;
      const uint32_t priorities[4] = {3, 1, 2, 0};
      for (uint32_t index = 0; index < head_count; ++index) {
        const uint32_t group = backward ? priorities[index] : 1 - index;
        begin[index] = group * stage;
        end[index] = begin[index] + stage;
      }
#else
      const uint32_t scheduled = communication ? pool_count_ - 1 : pool_count_;
      if (scheduled % 2 != 0) AscendC::Trap();
      const uint32_t stage = scheduled / 2;
      begin[0] = stage;
      end[0] = scheduled;
      end[1] = stage;
      if (communication) terminal = scheduled;
#endif
    }
    for (uint32_t index = 0; index < head_count; ++index) cursor[index] = LaneStart(begin[index]);
    uint32_t preferred = 0;
    while (true) {
      bool pending = false, executed = false;
      for (uint32_t choice = 0; choice < head_count; ++choice) {
        const uint32_t index = fair_communication ? (preferred + choice) % 2 : choice;
        if (cursor[index] >= end[index]) continue;
        pending = true;
        const TaskId id = PoolTask(cursor[index]);
        if (IsReady(id)) {
          Execute<Profiled>(id, recorder);
          cursor[index] += 32;
          if (fair_communication) preferred = 1 - index;
          executed = true;
          break;
        }
      }
      if (!pending) break;
      if (!executed) {
        // Only the bounded ready selector uses10us refresh; original task
        // execution and the FIFO control retain their declared polling policy.
        const int64_t start = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - start < 10 * SYSTEM_CYCLES_PER_US) {
        }
      }
    }
    if (terminal != UINT32_MAX && terminal % 32 == lane_) Execute<Profiled>(PoolTask(terminal), recorder);
  }
};
}  // namespace MulticoreRuntime
#endif
