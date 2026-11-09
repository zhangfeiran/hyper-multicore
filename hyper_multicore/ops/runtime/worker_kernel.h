/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file worker_kernel.h
 * @brief Shared CRTP base class for AIC/AIV scheduling loop and event synchronization.
 *
 * Each op derives KernelWorker : KernelWorkerBase<KernelWorker> and supplies:
 *   - static constexpr uint32_t TILING_IDX  — index into input_list for tiling params
 *   - static constexpr uint32_t EVENT_IDX   — index into input_list for all_event_counters
 *   - void ExecuteComputeKernel(TaskDesc)    — op-specific task dispatch switch
 *   - static constexpr uint32_t DIAGNOSTIC_IDX — index into input_list for the shared profiling/dump buffer
 *
 * Compute kernels (ExecuteMatmul, ExecuteShmemPutMem, etc.) are NOT part of this class;
 * they belong in each op's worker_kernel.cpp.
 */

#ifndef HYPER_MULTICORE_CORE_MULTICORE_OPS_RUNTIME_WORKER_KERNEL_H_
#define HYPER_MULTICORE_CORE_MULTICORE_OPS_RUNTIME_WORKER_KERNEL_H_

#include "kernel_operator.h"
#include "shmem.h"
#include "runtime_config.hpp"
#include "runtime_storage_bounds.h"
#include "cycle_trace_recorder.h"
#include "l1_tensor_dump.h"

#include "get_mem.h"
#include "replica_gradient.h"
#include "gradient_band.h"
#include "gradient_band_observer.h"

using namespace AscendC;  // NOLINT(build/namespaces)

namespace MulticoreRuntime {

constexpr uint32_t VECTOR_WORKER_STRIDE = 2;
constexpr int64_t EVENT_REFRESH_TIME_UNIT_CYCLES = 50;
constexpr int64_t CUBE_EVENT_REFRESH_INTERVAL_UNITS = 50;     // 2,500 system cycles.
constexpr int64_t VECTOR_EVENT_REFRESH_INTERVAL_UNITS = 150;  // 7,500 system cycles.
// The selected SwiGLU tiling uses baseRowLen=19; smaller dynamic tails must lower it.
constexpr int64_t SWIGLU_DYNAMIC_TAIL_BASE_ROW_LIMIT = 19;
constexpr int64_t READY_SIGNAL_RADIX = 2;

enum class VectorWorkerPolicy : uint32_t {
  LEGACY_ODD = 0,
  ALL = 1,
};

enum class RuntimeStorageBoundsPolicy : uint32_t {
  NATIVE_TILING = 0,
  GRAPH_OWNED = 1,
};

template <typename Derived, VectorWorkerPolicy WorkerPolicy = VectorWorkerPolicy::LEGACY_ODD,
          RuntimeStorageBoundsPolicy StorageBoundsPolicy = RuntimeStorageBoundsPolicy::NATIVE_TILING>
class KernelWorkerBase {
 public:
  __aicore__ inline KernelWorkerBase() {}

  static constexpr uint32_t PROFILE_DESC_WAIT_DEPENDENCY = 0x10000;
  static constexpr uint32_t PROFILE_DESC_TRIGGER_EVENT = 0x10007;
  static constexpr uint32_t PROFILE_DESC_TASK_TYPE_BASE = 0x20000;

  __aicore__ inline void Init(uint32_t worker_id, __gm__ uint8_t *runtimeConfigPtr, GM_ADDR *input_list) {
    this->worker_id_ = worker_id;
    this->runtimeConfigPtr = runtimeConfigPtr;

    uint64_t runtime_bytes = 0;
    uint64_t event_bytes = 0;
    uint32_t ep_size = 1;
    uint64_t local_experts = 0;
    if constexpr (StorageBoundsPolicy == RuntimeStorageBoundsPolicy::NATIVE_TILING) {
      runtime_bytes = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 6);
      event_bytes = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 7);
      ep_size = static_cast<uint32_t>(getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1));
      local_experts = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 2) / ep_size;
    } else {
      auto bounds = reinterpret_cast<__gm__ RuntimeStorageBounds *>(
        input_list[Derived::TILING_IDX] + Derived::RUNTIME_BOUNDS_OFFSET);
      runtime_bytes = bounds->runtime_config_bytes;
      event_bytes = bounds->event_counter_bytes;
    }
    if (!isRuntimeStorageValid(runtimeConfigPtr, runtime_bytes, event_bytes, ep_size, local_experts)) {
      AscendC::Trap();
    }
    InitReplicaExtension(runtime_bytes, local_experts);
    this->runtime_task_capacity = getRuntimeTaskCapacity(runtimeConfigPtr);
    this->runtime_event_capacity = getRuntimeEventCapacity(runtimeConfigPtr);
#ifdef __DAV_C220_CUBE__
    // The worker's scratch region is fixed for the lifetime of this task graph.
    this->grouped_matmul_group_list_offset_ = getGroupedMatmulGroupListOffsetById(runtimeConfigPtr, worker_id);
#endif
    all_event_counters.SetGlobalBuffer((__gm__ int32_t *)(input_list[Derived::EVENT_IDX]), runtime_event_capacity);

    all_event_num_triggers.SetGlobalBuffer((__gm__ int32_t *)(runtimeConfigPtr + getAllEventNumTriggersOffset()),
                                           runtime_event_capacity);
    vector_task_indexs.SetGlobalBuffer(
      (__gm__ int32_t *)(runtimeConfigPtr + getVectorTaskIndexsOffset(runtimeConfigPtr)), runtime_task_capacity);
    cube_task_indexs.SetGlobalBuffer((__gm__ int32_t *)(runtimeConfigPtr + getCubeTaskIndexsOffset(runtimeConfigPtr)),
                                     runtime_task_capacity);
    atomic_add_values.SetGlobalBuffer((__gm__ int32_t *)(runtimeConfigPtr + getAtomicAddValuesOffset(runtimeConfigPtr)),
                                      ATOMIC_ADD_VALUE_LEN);

    this->input_list = input_list;
    this->task_num = getTaskNum(this->runtimeConfigPtr);
    this->vector_task_num = getTaskIndexNumByTaskType(this->runtimeConfigPtr, TaskAiCoreType::TASK_AICORE_VECTOR);
    this->cube_task_num = getTaskIndexNumByTaskType(this->runtimeConfigPtr, TaskAiCoreType::TASK_AICORE_CUBE);
    // RuntimeConfig is operator-independent; native tiling buffers are not.
    // Reading a fixed native-tiling offset here corrupts the round-robin stride
    // for operators whose sixth field is not the AIC count (for example mHC).
    this->vector_num = getNumWorkers(this->runtimeConfigPtr);
    this->core_num = this->vector_num / 2;
    this->active_vector_num = WorkerPolicy == VectorWorkerPolicy::ALL ? this->vector_num : this->vector_num / 2;
  }

  __aicore__ inline void Process() {
#ifndef __DAV_C220_CUBE__
    if (this->worker_id_ % VECTOR_WORKER_STRIDE == 0 && replica_gradient_config_ != nullptr) {
      ProcessReplicaGradients();
      return;
    }
#endif
    ReadyHandshakeMeta meta;
    bool has_ready = LoadReadyHandshakeMeta(&meta);
    pull_protocol_ = meta.completion_event != 0;
    if (has_ready) {
      WaitForDeviceReady(meta);
    }
    switch (getDiagnosticMode(this->runtimeConfigPtr)) {
      case DiagnosticMode::PROFILE:
        ProcessProfiled();
        return;
      case DiagnosticMode::TENSOR_DUMP_L1:
        ProcessL1Dump();
        break;
      default:
        ProcessFast();
    }
    if (meta.completion_event != 0) {
      CompleteDeviceReads(meta);
    }
  }

 protected:
  __aicore__ inline void InitReplicaReady(__gm__ uint64_t *extension) {
    const uint64_t epoch = extension[8];
    if (epoch == 0 || epoch > INT32_MAX) {
      AscendC::Trap();
    }
    for (uint32_t index = 0; index < 2; ++index) {
      const uint64_t address = extension[6 + index];
      if (address == 0 || address % DATA_CACHE_LINE_SIZE != 0) {
        AscendC::Trap();
      }
      replica_ready_[index] = reinterpret_cast<GM_ADDR>(address);
    }
    replica_epoch_ = static_cast<int32_t>(epoch);
  }

  __aicore__ inline void InitReplicaExtension(uint64_t runtime_bytes, uint32_t local_experts) {
    const uint64_t base_bytes = getAtomicAddValuesOffset(runtimeConfigPtr) + ATOMIC_ADD_VALUE_LEN * INT32_T_SIZE;
    if (runtime_bytes != base_bytes) {
      if (runtime_bytes != base_bytes + 48 && runtime_bytes != base_bytes + 72 &&
          runtime_bytes != base_bytes + 80) {
        AscendC::Trap();
      }
      __gm__ uint64_t *extension = reinterpret_cast<__gm__ uint64_t *>(runtimeConfigPtr + base_bytes);
      const bool kernel_gradients = runtime_bytes == base_bytes + 80;
      const bool projection_ready = runtime_bytes != base_bytes + 48;
      const uint64_t magic = kernel_gradients ? 0x0000000553505754ULL :
                             projection_ready ? 0x0000000453505754ULL : 0x0000000253505754ULL;
      if (extension[0] != magic || extension[1] == 0 || extension[1] >= local_experts) {
        AscendC::Trap();
      }
      home_experts_ = extension[1];
      for (uint32_t index = 0; index < 4; ++index) {
        replica_matrix_bases_[index] = reinterpret_cast<GM_ADDR>(extension[index + 2]);
      }
      if (projection_ready) {
        InitReplicaReady(extension);
      }
      if (kernel_gradients) {
        if (extension[9] == 0 || extension[9] % DATA_CACHE_LINE_SIZE != 0) {
          AscendC::Trap();
        }
        replica_gradient_config_ = reinterpret_cast<GM_ADDR>(extension[9]);
      }
    }
  }

  __aicore__ inline GM_ADDR GetExpertMatrix(
      uint32_t position, int64_t expert, int64_t matrix_bytes, uint32_t replica_index) const {
    if (expert >= home_experts_) {
      if (replica_index < 2 && replica_ready_[replica_index] != nullptr) {
        WaitForReplicaWeights(expert - home_experts_, replica_index);
      }
      return replica_matrix_bases_[replica_index] + (expert - home_experts_) * matrix_bytes;
    }
    return input_list[position] + expert * matrix_bytes;
  }

  int64_t home_experts_ = 0;
  GM_ADDR replica_matrix_bases_[4] = {};
  GM_ADDR replica_ready_[2] = {};
  int32_t replica_epoch_ = 0;
  CycleTraceRecorder *replica_trace_ = nullptr;
  uint32_t replica_trace_task_ = 0;
  GM_ADDR replica_gradient_config_ = nullptr;

  __aicore__ inline void ProcessReplicaGradients() {
#ifndef __DAV_C220_CUBE__
    __gm__ uint64_t *config = reinterpret_cast<__gm__ uint64_t *>(replica_gradient_config_);
    CycleTraceRecorder trace;
    const bool profiling = isCycleProfileEnabled(runtimeConfigPtr);
    if (profiling) {
      trace.Init(worker_id_, input_list[Derived::DIAGNOSTIC_IDX], getAicProfileRecordCapacity(runtimeConfigPtr),
                  getAivProfileRecordCapacity(runtimeConfigPtr));
    }
    if (config[0] == 2) {
      ProcessReplicaGradient(config + config[1], 0x10008, trace, profiling);
      ProcessReplicaGradient(config + config[2], 0x10009, trace, profiling);
      if (worker_id_ == 0) {
        WaitBandIncomingAcks(config + config[1], profiling ? &trace : nullptr, 0);
        WaitBandIncomingAcks(config + config[2], profiling ? &trace : nullptr, 1);
      }
    } else {
      ProcessReplicaGradient(config, 0x10008, trace, profiling);
      if (worker_id_ == 0) {
        WaitBandIncomingAcks(config, profiling ? &trace : nullptr, 0);
      }
    }
    if (profiling) {
      const uint64_t now = trace.Now();
      trace.Record(0x1001F, 0, 0, 0xFFFFFFFF, now, now);
    }
#endif
  }

  __aicore__ inline void ValidateReplicaGradient(__gm__ uint64_t *config, int32_t ep) {
    const int32_t rank = config[1], slots = config[2], epoch = config[3];
    const int64_t elements = config[4];
    const int32_t incoming_count = config[10], owned_count = config[11];
    if ((config[0] != 1 && config[0] != 7) || rank < 0 || rank >= ep || slots <= 0 || epoch <= 0 || elements <= 0 ||
        incoming_count < 0 || incoming_count > slots || owned_count < 0 || owned_count > ep * slots) {
      AscendC::Trap();
    }
  }

  __aicore__ inline void ProcessReplicaGradient(__gm__ uint64_t *config, uint32_t profile_desc,
                                                CycleTraceRecorder &trace, bool profiling) {
#ifndef __DAV_C220_CUBE__
    const int32_t rank = config[1];
    const int32_t slots = config[2];
    const int32_t epoch = config[3];
    const int64_t elements = config[4];
    const int32_t incoming_count = config[10];
    const int32_t owned_count = config[11];
    const int32_t ep = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1);
    ValidateReplicaGradient(config, ep);
    const int32_t worker = worker_id_ / VECTOR_WORKER_STRIDE;
    GM_ADDR output = reinterpret_cast<GM_ADDR>(config[5]);
    GM_ADDR guest = reinterpret_cast<GM_ADDR>(config[6]);
    GM_ADDR ready = reinterpret_cast<GM_ADDR>(config[7]);
    __gm__ uint64_t *incoming = config + (config[0] == 7 ? 18 : 12);
    __gm__ uint64_t *owned = incoming + incoming_count * 3;
    // Publishing every local producer before remote waits breaks ring dependencies.
    if (worker == 0 && config[0] != 7) {
      for (int32_t index = 0; index < incoming_count; ++index) {
        const int32_t peer = incoming[index * 3];
        const int32_t slot = incoming[index * 3 + 1];
        WaitForDependency(incoming[index * 3 + 2]);
        aclshmemx_signal_op(reinterpret_cast<__gm__ int32_t *>(ready +
            (rank * slots + slot) * DATA_CACHE_LINE_SIZE), epoch, ACLSHMEM_SIGNAL_SET, peer);
      }
    }
    if (config[0] == 7) {
      ProcessGradientBands(config, owned, profile_desc, trace, profiling);
    } else {
    for (int32_t index = 0; index < owned_count; ++index) {
      const int32_t peer = owned[index * 4];
      const int32_t slot = owned[index * 4 + 1];
      const int32_t owner = owned[index * 4 + 2];
      WaitForDependency(owned[index * 4 + 3]);
      WaitReplicaCompletion(ready + (peer * slots + slot) * DATA_CACHE_LINE_SIZE, epoch);
      const uint64_t start = profiling ? trace.Now() : 0;
      AddReplicaGradient(output + owner * elements * sizeof(float), guest + slot * elements * sizeof(float),
                          elements, peer, worker, core_num);
      if (profiling) {
        trace.Record(profile_desc, 0, peer, rank * (home_experts_ + slots) + owner, start, trace.Now());
      }
    }
    }
    CompleteReplicaGradient(config, owned, incoming, worker, trace, profiling, profile_desc);
#endif
  }

  __aicore__ inline void CompleteReplicaGradient(__gm__ uint64_t *config, __gm__ uint64_t *owned,
                                                __gm__ uint64_t *incoming, int32_t worker,
                                                CycleTraceRecorder &trace, bool profiling, uint32_t profile_desc) {
#ifndef __DAV_C220_CUBE__
    const int32_t rank = config[1], slots = config[2], epoch = config[3];
    const int32_t owned_count = config[11], incoming_count = config[10];
    GM_ADDR done = reinterpret_cast<GM_ADDR>(config[9]);
    GM_ADDR ack = reinterpret_cast<GM_ADDR>(config[8]);
    StoreReplicaCompletion(done + worker * DATA_CACHE_LINE_SIZE, epoch);
    if (worker == 0) {
      for (int32_t index = 0; index < core_num; ++index) {
        WaitReplicaCompletion(done + index * DATA_CACHE_LINE_SIZE, epoch);
      }
      if (profiling) {
        const uint64_t now = trace.Now();
        RecordReplica(trace, ReplicaTraceEvent::OWNER_MERGED, profile_desc == 0x10009, 255, rank, epoch,
                      rank, 65535, now, now);
      }
      for (int32_t index = 0; index < owned_count; ++index) {
        const int32_t peer = owned[index * 4];
        const int32_t slot = owned[index * 4 + 1];
        aclshmemx_signal_op(reinterpret_cast<__gm__ int32_t *>(ack +
            (rank * slots + slot) * DATA_CACHE_LINE_SIZE), epoch, ACLSHMEM_SIGNAL_SET, peer);
        if (profiling) {
          const uint64_t now = trace.Now();
          RecordReplica(trace, ReplicaTraceEvent::ACK_ISSUED, profile_desc == 0x10009, 255, peer, epoch,
                        rank, owned[index * 4 + 2], now, now);
        }
      }
      if (config[0] != 7) {
        for (int32_t index = 0; index < incoming_count; ++index) {
          WaitReplicaCompletion(ack + (incoming[index * 3] * slots + incoming[index * 3 + 1]) *
                                      DATA_CACHE_LINE_SIZE, epoch);
        }
      }
    }
#endif
  }

  __aicore__ inline void WaitBandIncomingAcks(__gm__ uint64_t *config, CycleTraceRecorder *trace,
                                               uint32_t projection) {
#ifndef __DAV_C220_CUBE__
    if (config[0] != 7) {
      return;
    }
    __gm__ uint64_t *incoming = config + 18;
    GM_ADDR ack = reinterpret_cast<GM_ADDR>(config[8]);
    for (int32_t index = 0; index < config[10]; ++index) {
      const uint64_t start = trace == nullptr ? 0 : trace->Now();
      WaitReplicaCompletion(ack + (incoming[index * 3] * config[2] + incoming[index * 3 + 1]) *
                                  DATA_CACHE_LINE_SIZE, config[3]);
      if (trace != nullptr) {
        RecordReplica(*trace, ReplicaTraceEvent::ACK_OBSERVED, projection, 255, incoming[index * 3], config[3],
                      config[1], home_experts_ + incoming[index * 3 + 1], start, trace->Now());
      }
    }
#endif
  }

  __aicore__ inline bool HasBandGradientReturn() const {
    if (replica_gradient_config_ == nullptr) {
      return false;
    }
    __gm__ uint64_t *config = reinterpret_cast<__gm__ uint64_t *>(replica_gradient_config_);
    return config[0] == 7 || (config[0] == 2 && config[config[1]] == 7);
  }

  __aicore__ inline GM_ADDR GetGradientBandProduction(uint32_t position, uint32_t expert,
                                                        uint32_t *rows, uint32_t *band_rows, int32_t *epoch) {
    if (replica_gradient_config_ == nullptr || (position != 6 && position != 18)) {
      return nullptr;
    }
    __gm__ uint64_t *config = reinterpret_cast<__gm__ uint64_t *>(replica_gradient_config_);
    if (config[0] == 2) {
      config += config[position == 6 ? 1 : 2];
    }
    if (config[0] != 7) {
      return nullptr;
    }
    __gm__ uint64_t *incoming = config + 18;
    __gm__ uint64_t *owned = incoming + config[10] * 3;
    bool needed = false;
    if (expert >= home_experts_) {
      for (int32_t index = 0; index < config[10]; ++index) {
        needed = needed || incoming[index * 3 + 1] == expert - home_experts_;
      }
    } else {
      for (int32_t index = 0; index < config[11]; ++index) {
        needed = needed || owned[index * 4 + 2] == expert;
      }
    }
    if (!needed) {
      return nullptr;
    }
    *band_rows = config[14];
    *rows = config[15];
    *epoch = config[3];
    const uint64_t expert_stride = config[17] * core_num * DATA_CACHE_LINE_SIZE;
    if (expert >= home_experts_) {
      return reinterpret_cast<GM_ADDR>(config[13]) + (expert - home_experts_) * expert_stride;
    }
    return reinterpret_cast<GM_ADDR>(config[12]) + expert * expert_stride;
  }

  __aicore__ inline void ValidateGradientBandShape(__gm__ uint64_t *config, int64_t rows, int64_t columns,
                                                    int64_t elements, int32_t band_rows, int32_t bands) {
    if (band_rows <= 0 || rows <= 0 || columns <= 0 || rows * columns != elements ||
        bands != (rows + band_rows - 1) / band_rows || bands > GradientBandProducer::kMaxBands ||
        config[12] == 0 || config[13] == 0) {
      AscendC::Trap();
    }
  }

  __aicore__ inline void ProcessGradientBands(__gm__ uint64_t *config, __gm__ uint64_t *owned,
                                               uint32_t profile_desc, CycleTraceRecorder &trace, bool profiling) {
#ifndef __DAV_C220_CUBE__
    if (profiling) {
      ProcessGradientBandsProfiled(config, owned, profile_desc, trace);
      return;
    }
    const int32_t epoch = config[3];
    const int64_t elements = config[4];
    const int64_t rows = config[15];
    const int64_t columns = config[16];
    const int32_t band_rows = config[14];
    const int32_t bands = config[17];
    ValidateGradientBandShape(config, rows, columns, elements, band_rows, bands);
    const int32_t worker = worker_id_ / VECTOR_WORKER_STRIDE;
    const uint64_t expert_stride = bands * core_num * DATA_CACHE_LINE_SIZE;
    const auto partition = PartitionGradientBands(worker, core_num, bands);
    // Independent groups can consume later ready bands while another group
    // waits for a peer. Their output ranges are disjoint; peer order stays fixed.
    for (int32_t band = partition.group; band < bands; band += partition.groups) {
      const int64_t offset = band * static_cast<int64_t>(band_rows) * columns;
      const int64_t count = elements - offset < band_rows * columns ? elements - offset : band_rows * columns;
      for (int32_t index = 0; index < config[11]; ++index) {
        const int32_t peer = owned[index * 4];
        const int32_t slot = owned[index * 4 + 1];
        const int32_t owner = owned[index * 4 + 2];
        GM_ADDR owner_flags = reinterpret_cast<GM_ADDR>(config[12]) + owner * expert_stride;
        GM_ADDR guest_flags = reinterpret_cast<GM_ADDR>(config[13]) + slot * expert_stride;
        auto remote = hyper_multicore::shmem::data_plane::remote_ptr(
            reinterpret_cast<__gm__ void *>(guest_flags), peer);
        for (int32_t producer = 0; producer < core_num; ++producer) {
          const uint64_t signal_offset = (band * core_num + producer) * DATA_CACHE_LINE_SIZE;
          WaitBandProducer(owner_flags + signal_offset, epoch);
          WaitBandProducer(reinterpret_cast<GM_ADDR>(remote) + signal_offset, epoch);
        }
        const uint64_t start = profiling ? trace.Now() : 0;
        AddReplicaGradient(reinterpret_cast<GM_ADDR>(config[5]) + (owner * elements + offset) * sizeof(float),
                           reinterpret_cast<GM_ADDR>(config[6]) + (slot * elements + offset) * sizeof(float),
                           count, peer, partition.worker, partition.workers);
        if (profiling) {
          trace.Record(profile_desc, band, peer, config[1] * (home_experts_ + config[2]) + owner, start, trace.Now());
        }
      }
    }
#endif
  }

  __aicore__ inline void ProcessGradientBandsProfiled(__gm__ uint64_t *config, __gm__ uint64_t *owned,
                                                     uint32_t profile_desc, CycleTraceRecorder &trace) {
#ifndef __DAV_C220_CUBE__
    const int32_t bands = config[17], band_rows = config[14];
    const int64_t elements = config[4], columns = config[16];
    if (band_rows <= 0 || config[15] <= 0 || columns <= 0 || config[15] * columns != elements ||
        bands != (config[15] + band_rows - 1) / band_rows || bands > GradientBandProducer::kMaxBands ||
        config[12] == 0 || config[13] == 0) {
      AscendC::Trap();
    }
    const int32_t worker = worker_id_ / VECTOR_WORKER_STRIDE;
    const auto partition = PartitionGradientBands(worker, core_num, bands);
    GradientBandObserver observer;
    observer.Init(config, owned, core_num, profile_desc == 0x10009, &trace, partition.worker == 0);
    // Match the frozen static executor: finish every edge of a band before the next band.
    for (int32_t band = partition.group; band < bands; band += partition.groups) {
      const int64_t offset = band * static_cast<int64_t>(band_rows) * columns;
      const int64_t count = elements - offset < band_rows * columns ? elements - offset : band_rows * columns;
      for (uint32_t index = 0; index < config[11]; ++index) {
        uint32_t prefix = 0;
        uint64_t refreshed = 0;
        while (true) {
          const uint64_t now = AscendC::GetSystemCycle();
          if (now - refreshed >= 50) {
            refreshed = now;
            if (observer.Poll(index, band, prefix)) {
              break;
            }
          }
        }
        const uint64_t start = trace.Now();
        AddReplicaGradient(
            reinterpret_cast<GM_ADDR>(config[5]) + (owned[index * 4 + 2] * elements + offset) * sizeof(float),
            reinterpret_cast<GM_ADDR>(config[6]) + (owned[index * 4 + 1] * elements + offset) * sizeof(float),
            count, owned[index * 4], partition.worker, partition.workers);
        observer.Record(index, band, ReplicaTraceEvent::RETURN_WORK, start, trace.Now());
      }
    }
#endif
  }

  __aicore__ inline void WaitForReplicaWeights(int64_t slot, uint32_t projection) const {
    uint64_t start = replica_trace_ == nullptr ? 0 : replica_trace_->Now();
    GM_ADDR address = replica_ready_[projection] + slot * DATA_CACHE_LINE_SIZE;
    __gm__ volatile int32_t *ready_value =
      reinterpret_cast<__gm__ volatile int32_t *>(address);
    GlobalTensor<int32_t> ready;
    ready.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(address), 1);
    // SDMA publishes without an AIV helper; each poll must reload the remotely written value.
    do {
      DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(ready);
    } while (*ready_value < replica_epoch_);
    if (replica_trace_ != nullptr) {
      replica_trace_->Record(0x1000A + projection, replica_trace_task_, replica_epoch_,
                             static_cast<uint32_t>(slot), start, replica_trace_->Now());
    }
  }

  __aicore__ inline void ProcessFast() {
#ifdef __DAV_C220_CUBE__
    uint32_t block_idx = this->worker_id_;
    do {
      if (block_idx >= this->cube_task_num) {
        return;
      }
      TaskId task_index = GetTaskIndex(block_idx);
      ExecuteTaskFast(task_index);
      block_idx = block_idx + this->core_num;
    } while (1);
#else
    uint32_t block_idx = GetVectorScheduleIndex();
    if (block_idx == INVALID_VECTOR_SCHEDULE_INDEX) {
      return;
    }
    do {
      if (block_idx >= this->vector_task_num) {
        return;
      }
      TaskId task_index = GetTaskIndex(block_idx);
      ExecuteTaskFast(task_index);
      block_idx = block_idx + this->active_vector_num;
    } while (1);
#endif
  }

  __aicore__ inline void ProcessL1Dump() {
#ifdef __DAV_C220_CUBE__
    uint32_t block_idx = this->worker_id_;
    do {
      if (block_idx >= this->cube_task_num) {
        return;
      }
      TaskId task_index = GetTaskIndex(block_idx);
      ExecuteTaskL1Dump(task_index);
      block_idx = block_idx + this->core_num;
    } while (1);
#else
    uint32_t block_idx = GetVectorScheduleIndex();
    if (block_idx == INVALID_VECTOR_SCHEDULE_INDEX) {
      return;
    }
    do {
      if (block_idx >= this->vector_task_num) {
        return;
      }
      TaskId task_index = GetTaskIndex(block_idx);
      ExecuteTaskL1Dump(task_index);
      block_idx = block_idx + this->active_vector_num;
    } while (1);
#endif
  }

  __aicore__ inline void ProcessProfiled() {
    CycleTraceRecorder recorder;
    recorder.Init(worker_id_, input_list[Derived::DIAGNOSTIC_IDX], getAicProfileRecordCapacity(runtimeConfigPtr),
                  getAivProfileRecordCapacity(runtimeConfigPtr));
    ProcessProfiledQueue(recorder);
    const uint64_t done = recorder.Now();
    recorder.Record(0x1001E, 0, 0, 0xFFFFFFFF, done, done);
    ReadyHandshakeMeta meta;
    LoadReadyHandshakeMeta(&meta);
    if (meta.completion_event != 0) {
      CompleteDeviceReads(meta);
    }
    const uint64_t complete = recorder.Now();
    recorder.Record(0x1001F, 0, 0, 0xFFFFFFFF, complete, complete);
  }

  __aicore__ inline void ProcessProfiledQueue(CycleTraceRecorder &cycle_trace_recorder) {
#ifdef __DAV_C220_CUBE__
    uint32_t block_idx = this->worker_id_;
    do {
      if (block_idx >= this->cube_task_num) {
        return;
      }
      TaskId task_index = GetTaskIndex(block_idx);
      ExecuteTaskProfiled(task_index, cycle_trace_recorder);
      block_idx = block_idx + this->core_num;
    } while (1);
#else
    uint32_t block_idx = GetVectorScheduleIndex();
    if (block_idx == INVALID_VECTOR_SCHEDULE_INDEX) {
      return;
    }
    do {
      if (block_idx >= this->vector_task_num) {
        return;
      }
      TaskId task_index = GetTaskIndex(block_idx);
      ExecuteTaskProfiled(task_index, cycle_trace_recorder);
      block_idx = block_idx + this->active_vector_num;
    } while (1);
#endif
  }

  __aicore__ inline bool LoadReadyHandshakeMeta(ReadyHandshakeMeta *meta) {
    getReadyHandshakeMeta(this->runtimeConfigPtr, meta);
    return meta->ready_event != 0;
  }

  __aicore__ inline void PublishDeviceReady(const ReadyHandshakeMeta &meta, bool completion = false) {
#ifndef __DAV_C220_CUBE__
    int64_t ep = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1);
    int64_t rank = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 0) % ep;
    constexpr uint32_t ready_stride = DATA_CACHE_LINE_SIZE / INT32_T_SIZE;
    __gm__ int32_t *ready = (__gm__ int32_t *)(input_list[Derived::EVENT_IDX]) + runtime_event_capacity;
    if (completion) {
      ready += (ep + 1) * ready_stride;
    }

    GlobalTensor<int32_t> ready_state;
    ready_state.SetGlobalBuffer(ready, static_cast<uint32_t>((ep + 1) * ready_stride));
    uint32_t generation_index = static_cast<uint32_t>(ep) * ready_stride;
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      ready_state[generation_index]);
    int32_t previous_generation = ready_state.GetValue(generation_index);
    if (previous_generation == INT32_MAX) {
      AscendC::Trap();
    }
    int32_t generation = previous_generation + 1;
    ready_state.SetValue(generation_index, generation);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      ready_state[generation_index]);
    PipeBarrier<PIPE_ALL>();

    uint32_t round = 0;
    for (int64_t distance = 1; distance < ep; distance *= READY_SIGNAL_RADIX, ++round) {
      int64_t target = (rank + distance) % ep;
      __gm__ int32_t *round_ready = ready + round * ready_stride;
      aclshmemx_signal_op(round_ready, generation, ACLSHMEM_SIGNAL_SET, static_cast<int>(target));
      aclshmem_signal_wait_until(round_ready, ACLSHMEM_CMP_GE, generation);
    }
    if (!completion) {
      TriggerEvent(meta.ready_event);
    }
#endif
  }

  __aicore__ inline void CompleteDeviceReads(const ReadyHandshakeMeta &meta) {
#ifndef __DAV_C220_CUBE__
    if (this->worker_id_ % 2 == 0) {
      return;
    }
#endif
    TriggerEvent(meta.completion_event);
#ifndef __DAV_C220_CUBE__
    if (this->worker_id_ == 1) {
      WaitForDependency(meta.completion_event);
      if (meta.ready_event != 0) {
        PublishDeviceReady(meta, true);
      }
    }
#endif
  }

  template <typename T>
  __aicore__ inline void ExecuteShmemGetMem(const TaskDesc &task) {
    int64_t ep = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1);
    int64_t experts = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 2);
    int64_t hidden = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 3);
    int64_t tiles = task.task_split_num / experts;
    int64_t source_pe = task.task_index / ((experts / ep) * tiles);
    int64_t size =
      reinterpret_cast<__gm__ int32_t *>(input_list[task.inputs[3].input_position])[task.inputs[3].base_ptr_offset];
    int64_t start = (task.task_index % tiles) * static_cast<int64_t>(task.task_split_value) * hidden;
    if (start >= size) {
      return;
    }
    int64_t tile_elements = static_cast<int64_t>(task.task_split_value) * hidden;
    int64_t elements = size - start < tile_elements ? size - start : tile_elements;
    int64_t destination_offset =
      reinterpret_cast<__gm__ int64_t *>(input_list[task.inputs[0].input_position])[task.inputs[0].base_ptr_offset];
    int64_t source_offset =
      reinterpret_cast<__gm__ int64_t *>(input_list[task.inputs[2].input_position])[task.inputs[2].base_ptr_offset];
    PullToLocal<T>(input_list[task.outputs[0].input_position] + (destination_offset + start) * sizeof(T),
                   input_list[task.inputs[1].input_position] + (source_offset + start) * sizeof(T), elements,
                   source_pe);
  }

  __aicore__ inline void WaitForDeviceReady(const ReadyHandshakeMeta &meta) {
#ifdef __DAV_C220_CUBE__
    WaitForDependency(meta.ready_event);
#else
    if (this->worker_id_ % VECTOR_WORKER_STRIDE == 0) {
      return;
    }
    if (this->worker_id_ == 1) {
      PublishDeviceReady(meta);
    }
    WaitForDependency(meta.ready_event);
#endif
  }

  __aicore__ inline TaskId GetTaskIndex(uint32_t task_id) {
#ifdef __DAV_C220_CUBE__
    return cube_task_indexs.GetValue(task_id);
#else
    return vector_task_indexs.GetValue(task_id);
#endif
  }

  __aicore__ inline uint32_t GetVectorScheduleIndex() {
    if constexpr (WorkerPolicy == VectorWorkerPolicy::ALL) {
      return this->worker_id_ < this->active_vector_num ? this->worker_id_ : INVALID_VECTOR_SCHEDULE_INDEX;
    }
    return this->worker_id_ % 2 == 0 ? INVALID_VECTOR_SCHEDULE_INDEX : this->worker_id_ / 2;
  }

  __aicore__ inline void AtomicAddForAllEventCounters(uint32_t event_index) {
    TPipe eventPipe;
#ifdef __DAV_C220_CUBE__
    TBuf<AscendC::TPosition::A1> eventBuffer;
#else
    TBuf<AscendC::TPosition::VECOUT> eventBuffer;
#endif
    eventPipe.InitBuffer(eventBuffer, DISPATCH_TOKEN_UB_SIZE);
    LocalTensor<int32_t> localSet = eventBuffer.GetWithOffset<int32_t>(EXP_TOKEN_COUNT_FLAG_CNT, 0);
#ifdef __DAV_C220_CUBE__
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    DataCopy(localSet, this->atomic_add_values, EXP_TOKEN_COUNT_FLAG_CNT);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    AscendC::SetAtomicAdd<int32_t>();
    SyncFunc<AscendC::HardEvent::S_MTE2>();
    DataCopy(this->all_event_counters[event_index], localSet, EXP_TOKEN_COUNT_FLAG_CNT);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    AscendC::SetAtomicNone();
#else
    localSet.SetValue(0, 1);
    for (int32_t i = 1; i < EXP_TOKEN_COUNT_FLAG_CNT; ++i) {
      localSet.SetValue(i, 0);
    }
    AscendC::SetAtomicAdd<int32_t>();
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(this->all_event_counters[event_index], localSet, EXP_TOKEN_COUNT_FLAG_CNT);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    AscendC::SetAtomicNone();
#endif
    eventBuffer.FreeTensor(localSet);
    eventPipe.Destroy();
  }

  __aicore__ inline void ResolveTaskDescriptor(TaskId &, TaskDesc &) {}

  __aicore__ inline void ExecuteTaskFast(TaskId task_id) {
    if (task_id >= runtime_task_capacity) {
      AscendC::Trap();
    }
    TaskDesc task_desc;
    getTaskDesc(this->runtimeConfigPtr, &(task_desc), task_id);
    static_cast<Derived *>(this)->ResolveTaskDescriptor(task_id, task_desc);
    if ((task_desc.dependent_event != EVENT_INVALID_ID && task_desc.dependent_event >= runtime_event_capacity) ||
        static_cast<uint64_t>(task_desc.trigger_event) + ATOMIC_ADD_VALUE_LEN > runtime_event_capacity) {
      AscendC::Trap();
    }
    if (task_desc.dependent_event != EVENT_INVALID_ID) {
      WaitForDependency(task_desc.dependent_event, task_desc.extra_value_2);
    }
    static_cast<Derived *>(this)->ExecuteComputeKernel(task_desc);
    if (task_desc.task_type != TaskType::TASK_SHMEM_PUT_MEM_SIGNAL) {
      TriggerProducerEvent(task_desc);
      TriggerEvent(task_desc.trigger_event);
    }
  }

  __aicore__ inline void ExecuteTaskL1Dump(TaskId task_id) {
    if (task_id >= runtime_task_capacity) {
      AscendC::Trap();
    }
    TaskDesc task_desc;
    getTaskDesc(this->runtimeConfigPtr, &(task_desc), task_id);
    static_cast<Derived *>(this)->ResolveTaskDescriptor(task_id, task_desc);
    if ((task_desc.dependent_event != EVENT_INVALID_ID && task_desc.dependent_event >= runtime_event_capacity) ||
        static_cast<uint64_t>(task_desc.trigger_event) + ATOMIC_ADD_VALUE_LEN > runtime_event_capacity) {
      AscendC::Trap();
    }
    if (task_desc.dependent_event != EVENT_INVALID_ID) {
      WaitForDependency(task_desc.dependent_event, task_desc.extra_value_2);
    }
    CaptureL1TaskTensors(task_id, L1_DUMP_PHASE_BEFORE);
    static_cast<Derived *>(this)->ExecuteComputeKernel(task_desc);
    PipeBarrier<PIPE_ALL>();
    CaptureL1TaskTensors(task_id, L1_DUMP_PHASE_AFTER);
    if (task_desc.task_type != TaskType::TASK_SHMEM_PUT_MEM_SIGNAL) {
      TriggerEvent(task_desc.trigger_event);
    }
  }

  __aicore__ inline void CaptureL1TaskTensors(TaskId task_id, uint32_t phase) {
    MulticoreRuntime::CaptureL1TaskTensors(input_list[Derived::DIAGNOSTIC_IDX], input_list, task_id, phase, worker_id_);
  }

  __aicore__ inline void ExecuteTaskProfiled(TaskId task_id, CycleTraceRecorder &cycle_trace_recorder) {
    if (task_id >= runtime_task_capacity) {
      AscendC::Trap();
    }
    TaskDesc task_desc;
    getTaskDesc(this->runtimeConfigPtr, &(task_desc), task_id);
    static_cast<Derived *>(this)->ResolveTaskDescriptor(task_id, task_desc);
    if ((task_desc.dependent_event != EVENT_INVALID_ID && task_desc.dependent_event >= runtime_event_capacity) ||
        static_cast<uint64_t>(task_desc.trigger_event) + ATOMIC_ADD_VALUE_LEN > runtime_event_capacity) {
      AscendC::Trap();
    }
    uint32_t owner_id = getTaskProfileOwnerId(this->runtimeConfigPtr, task_id);
    if (task_desc.dependent_event != EVENT_INVALID_ID) {
      uint64_t wait_start_cycle = cycle_trace_recorder.Now();
      WaitForDependency(task_desc.dependent_event, task_desc.extra_value_2);
      uint64_t wait_end_cycle = cycle_trace_recorder.Now();
      cycle_trace_recorder.Record(Derived::PROFILE_DESC_WAIT_DEPENDENCY, task_id, task_desc.task_index, owner_id,
                                  wait_start_cycle, wait_end_cycle);
    }
    uint32_t profile_desc_id = getTaskProfileDescId(this->runtimeConfigPtr, task_id);
    if (profile_desc_id == PROFILE_DESC_INVALID_ID) {
      profile_desc_id = PROFILE_DESC_TASK_TYPE_BASE + static_cast<uint32_t>(task_desc.task_type);
    }
    replica_trace_ = &cycle_trace_recorder;
    replica_trace_task_ = task_id;
    uint64_t compute_start_cycle = cycle_trace_recorder.Now();
    static_cast<Derived *>(this)->ExecuteComputeKernel(task_desc);
    uint64_t compute_end_cycle = cycle_trace_recorder.Now();
    replica_trace_ = nullptr;
    cycle_trace_recorder.Record(profile_desc_id, task_id, task_desc.task_index, owner_id, compute_start_cycle,
                                compute_end_cycle);
    if (task_desc.task_type != TaskType::TASK_SHMEM_PUT_MEM_SIGNAL) {
      uint64_t trigger_start_cycle = cycle_trace_recorder.Now();
      TriggerProducerEvent(task_desc);
      TriggerEvent(task_desc.trigger_event);
      uint64_t trigger_end_cycle = cycle_trace_recorder.Now();
      cycle_trace_recorder.Record(Derived::PROFILE_DESC_TRIGGER_EVENT, task_id, task_desc.task_index, owner_id,
                                  trigger_start_cycle, trigger_end_cycle);
    }
  }

  __aicore__ inline void WaitForDependency(uint32_t event_index, uint32_t poll_interval_us = 0) {
    // Pull producers signal locally after MTE3 completion. Observe short-lived
    // GMM/SwiGLU dependencies promptly while retaining a bounded refresh rate.
#ifdef __DAV_C220_CUBE__
    const uint32_t default_poll_interval_us = pull_protocol_ ? 10 : CUBE_EVENT_REFRESH_INTERVAL_UNITS;
#else
    const uint32_t default_poll_interval_us = pull_protocol_ ? 30 : VECTOR_EVENT_REFRESH_INTERVAL_UNITS;
#endif
    if (poll_interval_us == 0) {
      poll_interval_us = default_poll_interval_us;
    }
    constexpr int64_t cycles_per_us = EVENT_REFRESH_TIME_UNIT_CYCLES;
    int32_t needed = all_event_num_triggers.GetValue(event_index);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      all_event_counters[event_index]);
    int32_t current = all_event_counters.GetValue(event_index);
    PipeBarrier<PIPE_ALL>();
    int64_t previous_cycle = AscendC::GetSystemCycle();
    while (current < needed) {
      int64_t elapsed_cycles = AscendC::GetSystemCycle() - previous_cycle;
      if (elapsed_cycles / cycles_per_us > poll_interval_us) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
          all_event_counters[event_index]);
        current = all_event_counters.GetValue(event_index);
        previous_cycle = AscendC::GetSystemCycle();
      }
    }
  }

  __aicore__ inline void TriggerEvent(uint32_t event_index) { AtomicAddForAllEventCounters(event_index); }

  __aicore__ inline void TriggerProducerEvent(const TaskDesc &task) {
    // GMM producers can publish a separate full-expert fence while preserving
    // their ordinary graph completion counter and the Cube computation order.
    if (task.task_type == TaskType::TASK_GROUPED_MATMUL && task.extra_value_1 != 0) {
      // Row-band consumers use the direct Cube table and never read this whole
      // expert fence. Ordinary task graph triggers remain unchanged.
      if (HasBandGradientReturn()) {
        return;
      }
      if (static_cast<uint64_t>(task.extra_value_1) + ATOMIC_ADD_VALUE_LEN > runtime_event_capacity) {
        AscendC::Trap();
      }
      PipeBarrier<PIPE_ALL>();
      TriggerEvent(task.extra_value_1);
    }
  }

  bool pull_protocol_ = false;
  uint32_t worker_id_ = 0;
  uint32_t grouped_matmul_group_list_offset_ = 0;
  uint32_t task_num = 0;
  uint32_t runtime_task_capacity = 0;
  uint32_t runtime_event_capacity = 0;
  int32_t vector_task_num = 0;
  int32_t cube_task_num = 0;

  GlobalTensor<int32_t> all_event_counters;
  GlobalTensor<int32_t> all_event_num_triggers;
  GlobalTensor<int32_t> atomic_add_values;
  GlobalTensor<int32_t> vector_task_indexs;
  GlobalTensor<int32_t> cube_task_indexs;

  __gm__ uint8_t *runtimeConfigPtr;
  GM_ADDR *input_list = nullptr;
  int64_t core_num = 0;
  int64_t vector_num = 0;
  uint32_t active_vector_num = 0;
  static constexpr uint32_t INVALID_VECTOR_SCHEDULE_INDEX = 0xFFFFFFFF;
};

}  // namespace MulticoreRuntime

#endif  // HYPER_MULTICORE_CORE_MULTICORE_OPS_RUNTIME_WORKER_KERNEL_H_
