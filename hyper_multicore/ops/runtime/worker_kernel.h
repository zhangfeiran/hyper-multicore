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
#include "core_type.h"
#include "shmem.h"
#include "runtime_config.hpp"
#include "runtime_storage_bounds.h"
#include "cycle_trace_recorder.h"
#include "normal_task_trace.h"
#include "l1_tensor_dump.h"
#include "vector_ready_queue.h"


#include "get_mem.h"
#include "empty_dispatch.h"
#include "rdma_transport.h"
#include "replica_gradient.h"
#include "gradient_band.h"
#include "gradient_band_observer.h"

using namespace AscendC;  // NOLINT(build/namespaces)


namespace MulticoreRuntime {

constexpr uint32_t VECTOR_WORKER_STRIDE = 2;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
// GetSystemCycle uses the fixed system counter, independent of AI Core clocks.
constexpr int64_t SYSTEM_CYCLES_PER_US = 1000;
constexpr int64_t CUBE_EVENT_REFRESH_INTERVAL_US = 50;
constexpr int64_t VECTOR_EVENT_REFRESH_INTERVAL_US = 150;
#else
constexpr int64_t EVENT_REFRESH_TIME_UNIT_CYCLES = 50;
constexpr int64_t CUBE_EVENT_REFRESH_INTERVAL_UNITS = 50;     // 2,500 system cycles.
constexpr int64_t VECTOR_EVENT_REFRESH_INTERVAL_UNITS = 150;  // 7,500 system cycles.
#endif
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
    rdma_.Init(runtimeConfigPtr, runtime_bytes, ep_size, getRuntimeEventCapacity(runtimeConfigPtr));
    InitReplicaExtension(runtime_bytes, local_experts);
    this->runtime_task_capacity = getRuntimeTaskCapacity(runtimeConfigPtr);
    this->runtime_event_capacity = getRuntimeEventCapacity(runtimeConfigPtr);
#ifdef HMC_RUNTIME_CUBE
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
    const auto header = reinterpret_cast<__gm__ RuntimeHeader *>(runtimeConfigPtr);
    ready_queue_ = header->vector_ready_queue == 1 &&
                   getDiagnosticMode(runtimeConfigPtr) == DiagnosticMode::NORMAL;
    // Guest bands are published by Cube producers. Ranks without owner merges
    // can use every AIV, then retain their guest buffers until the owner ACKs.
    ready_queue_all_ = ready_queue_ && (replica_gradient_config_ == nullptr ||
      HasOnlyGuestBandGradients(reinterpret_cast<__gm__ const uint64_t *>(replica_gradient_config_)));
    this->active_vector_num = (WorkerPolicy == VectorWorkerPolicy::ALL || ready_queue_all_) ?
                              this->vector_num : this->vector_num / 2;
  }

  __aicore__ inline void Process() {
    const auto header = reinterpret_cast<__gm__ RuntimeHeader *>(runtimeConfigPtr);
    if (header->padding[0] == NORMAL_TASK_TRACE_TAG) {
      normal_trace_.Init(input_list[Derived::DIAGNOSTIC_IDX],
                         static_cast<Derived *>(this)->NormalTraceStorageBytes(), header, worker_id_);
    }
#ifndef HMC_RUNTIME_CUBE
    if (this->worker_id_ % VECTOR_WORKER_STRIDE == 0 && replica_gradient_config_ != nullptr && !ready_queue_all_) {
      ProcessReplicaGradients();
      normal_trace_.Finish();
      return;
    }
#endif
    ReadyHandshakeMeta meta;
    bool has_ready = LoadReadyHandshakeMeta(&meta);
    pull_protocol_ = meta.completion_event != 0;
    if (has_ready) {
      WaitForDeviceReady(meta);
    }
    const auto mode = getDiagnosticMode(this->runtimeConfigPtr);
    CycleTraceRecorder recorder;
    if (mode == DiagnosticMode::PROFILE) {
      recorder.Init(this->worker_id_, this->input_list[Derived::DIAGNOSTIC_IDX],
                    getAicProfileRecordCapacity(this->runtimeConfigPtr),
                    getAivProfileRecordCapacity(this->runtimeConfigPtr),
                    getCubeWorkerCapacity(this->runtimeConfigPtr));
    }
    ProcessTasks(mode, recorder);
    if (meta.completion_event != 0) {
      CompleteDeviceReads(meta);
    }
#ifndef HMC_RUNTIME_CUBE
    if (this->worker_id_ == 0 && ready_queue_all_ && replica_gradient_config_ != nullptr) {
      CompleteGuestBandGradients();
    }
#endif
    normal_trace_.Finish();
  }

 protected:
  __aicore__ inline uint64_t NormalTraceStorageBytes() const { return 0; }

  NormalTaskTrace normal_trace_;

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
      ValidateReplicaExtension(extension, magic, local_experts, projection_ready);
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

  __aicore__ inline void ValidateReplicaExtension(__gm__ uint64_t *extension, uint64_t magic,
                                                  uint32_t local_experts, bool projection_ready) {
    if (extension[0] != magic || extension[1] == 0 || extension[1] > local_experts) {
      AscendC::Trap();
    }
    if (extension[1] == local_experts &&
        (projection_ready || extension[2] || extension[3] || extension[4] || extension[5])) {
      AscendC::Trap();
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

  __aicore__ inline void CompleteGuestBandGradients() {
#ifndef HMC_RUNTIME_CUBE
    __gm__ uint64_t *config = reinterpret_cast<__gm__ uint64_t *>(replica_gradient_config_);
    const int32_t ep = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1);
    if (config[0] == 2) {
      ValidateReplicaGradient(config + config[1], ep);
      ValidateReplicaGradient(config + config[2], ep);
      WaitBandIncomingAcks(config + config[1], nullptr, 0);
      WaitBandIncomingAcks(config + config[2], nullptr, 1);
    } else {
      ValidateReplicaGradient(config, ep);
      WaitBandIncomingAcks(config, nullptr, 0);
    }
#endif
  }

  __aicore__ inline void ProcessReplicaGradients() {
#ifndef HMC_RUNTIME_CUBE
    __gm__ uint64_t *config = reinterpret_cast<__gm__ uint64_t *>(replica_gradient_config_);
    CycleTraceRecorder trace;
    const bool profiling = isCycleProfileEnabled(runtimeConfigPtr);
    if (profiling) {
      trace.Init(worker_id_, input_list[Derived::DIAGNOSTIC_IDX], getAicProfileRecordCapacity(runtimeConfigPtr),
                 getAivProfileRecordCapacity(runtimeConfigPtr), getCubeWorkerCapacity(runtimeConfigPtr));
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
    if ((config[0] != 1 && !IsGradientBandProtocol(config[0])) || rank < 0 || rank >= ep || slots <= 0 ||
        epoch <= 0 || elements <= 0 || incoming_count < 0 || incoming_count > slots ||
        owned_count < 0 || owned_count > ep * slots || (config[0] == 11 && owned_count != 0)) {
      AscendC::Trap();
    }
    if (config[0] == 13) {
      ValidateFinalGradient(config);
    }
  }

  __aicore__ inline void ValidateFinalGradient(__gm__ uint64_t *config) {
    const int32_t incoming_count = config[10], owned_count = config[11];
    if (config[19] != home_experts_ || config[20] > config[19] ||
        (owned_count != 0 && config[18] == 0)) {
      AscendC::Trap();
    }
    auto owned = config + GradientRecordHeaderWords(config) + incoming_count * 3;
    for (int32_t index = 0; index < owned_count; ++index) {
      const uint32_t owner = owned[index * 4 + 2];
      const uint32_t scratch = GradientScratchIndex(config, owner);
      if (scratch == 0 || config[config[22] + index] > 1) {
        AscendC::Trap();
      }
      bool last = true;
      for (int32_t next = index + 1; next < owned_count; ++next) {
        last = last && owned[next * 4 + 2] != owner;
      }
      if (config[config[22] + index] != static_cast<uint64_t>(last)) {
        AscendC::Trap();
      }
    }
  }

  __aicore__ inline void ProcessReplicaGradient(__gm__ uint64_t *config, uint32_t profile_desc,
                                                CycleTraceRecorder &trace, bool profiling) {
#ifndef HMC_RUNTIME_CUBE
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
    __gm__ uint64_t *incoming = config + GradientRecordHeaderWords(config);
    __gm__ uint64_t *owned = incoming + incoming_count * 3;
    // Publishing every local producer before remote waits breaks ring dependencies.
    if (worker == 0 && !IsGradientBandProtocol(config[0])) {
      for (int32_t index = 0; index < incoming_count; ++index) {
        const int32_t peer = incoming[index * 3];
        const int32_t slot = incoming[index * 3 + 1];
        WaitForDependency(incoming[index * 3 + 2]);
        aclshmemx_signal_op(reinterpret_cast<__gm__ int32_t *>(ready +
            (rank * slots + slot) * DATA_CACHE_LINE_SIZE), epoch, ACLSHMEM_SIGNAL_SET, peer);
      }
    }
    if (IsGradientBandProtocol(config[0])) {
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
#ifndef HMC_RUNTIME_CUBE
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
      if (!IsGradientBandProtocol(config[0])) {
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
#ifndef HMC_RUNTIME_CUBE
    if (!IsGradientBandProtocol(config[0])) {
      return;
    }
    __gm__ uint64_t *incoming = config + GradientRecordHeaderWords(config);
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
    return IsGradientBandProtocol(config[0]) || (config[0] == 2 && IsGradientBandProtocol(config[config[1]]));
  }

  __aicore__ inline uint32_t ResolveGradientOutputBytes(uint32_t position, uint32_t expert,
                                                        uint32_t descriptor_bytes) const {
    if (replica_gradient_config_ == nullptr || (position != 6 && position != 18)) {
      return descriptor_bytes;
    }
    auto config = reinterpret_cast<__gm__ const uint64_t *>(replica_gradient_config_);
    if (config[0] == 2) {
      config += config[position == 6 ? 1 : 2];
    }
    return GradientOutputElementBytes(config, expert, home_experts_, descriptor_bytes);
  }

  __aicore__ inline GM_ADDR ResolveGradientOutputAddress(uint32_t position, uint32_t expert,
                                                         int64_t matrix_bytes, uint32_t replica_index) const {
    if (replica_gradient_config_ != nullptr && expert < home_experts_) {
      auto config = reinterpret_cast<__gm__ const uint64_t *>(replica_gradient_config_);
      if (config[0] == 2) {
        config += config[position == 6 ? 1 : 2];
      }
      if (config[0] == 13) {
        const uint32_t scratch = GradientScratchIndex(config, expert);
        if (scratch != 0) {
          return reinterpret_cast<GM_ADDR>(config[18]) + (scratch - 1) * matrix_bytes;
        }
      }
    }
    return GetExpertMatrix(position, expert, matrix_bytes, replica_index);
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
    if (!IsGradientBandProtocol(config[0])) {
      return nullptr;
    }
    __gm__ uint64_t *incoming = config + GradientRecordHeaderWords(config);
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
#ifndef HMC_RUNTIME_CUBE
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
        MergeGradientBand(config, owned, index, offset, count, partition);
        if (profiling) {
          trace.Record(profile_desc, band, peer, config[1] * (home_experts_ + config[2]) + owner, start, trace.Now());
        }
      }
    }
#endif
  }

  __aicore__ inline void ProcessGradientBandsProfiled(__gm__ uint64_t *config, __gm__ uint64_t *owned,
                                                     uint32_t profile_desc, CycleTraceRecorder &trace) {
#ifndef HMC_RUNTIME_CUBE
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
        MergeGradientBand(config, owned, index, offset, count, partition);
        observer.Record(index, band, ReplicaTraceEvent::RETURN_WORK, start, trace.Now());
      }
    }
#endif
  }

  __aicore__ inline void MergeGradientBand(__gm__ uint64_t *config, __gm__ uint64_t *owned,
                                           uint32_t index, int64_t offset, int64_t count,
                                           const GradientBandPartition &partition) {
#ifndef HMC_RUNTIME_CUBE
    const int64_t elements = config[4], owner = owned[index * 4 + 2];
    const int32_t peer = owned[index * 4], slot = owned[index * 4 + 1];
    GM_ADDR source = reinterpret_cast<GM_ADDR>(config[6]) + (slot * elements + offset) * sizeof(float);
    GM_ADDR output = reinterpret_cast<GM_ADDR>(config[5]) + (owner * elements + offset) * sizeof(float);
    if (config[0] == 13) {
      const uint32_t scratch = GradientScratchIndex(config, owner);
      if (scratch == 0) {
        AscendC::Trap();
      }
      output = reinterpret_cast<GM_ADDR>(config[18]) + ((scratch - 1) * elements + offset) * sizeof(float);
      if (config[config[22] + index] != 0) {
        GM_ADDR final_output = reinterpret_cast<GM_ADDR>(config[5]) +
                               (owner * elements + offset) * sizeof(bfloat16_t);
        AddReplicaGradient<true>(output, source, count, peer, partition.worker, partition.workers, final_output);
        return;
      }
    }
    AddReplicaGradient(output, source, count, peer, partition.worker, partition.workers);
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

  struct RdmaQueueCursor {
    uint32_t owner;
    uint32_t owners;
    uint32_t phase;
    uint32_t slot;
  };

  // Keep one task-dispatch call site for normal, profiled, and RDMA work.
  __aicore__ inline void ProcessTasks(DiagnosticMode mode, CycleTraceRecorder &recorder) {
#ifdef HMC_RUNTIME_CUBE
    uint32_t block_idx = this->worker_id_;
    const uint32_t count = this->cube_task_num;
    const uint32_t stride = this->core_num;
#else
    const bool rdma_owner = rdma_.enabled && this->worker_id_ % 2 == 0;
    RdmaQueueCursor cursor{0, 0, 0, 0};
    if (rdma_owner) {
      cursor.owner = this->worker_id_ / 2;
      cursor.owners = RdmaOwnerCount();
      if (cursor.owner >= cursor.owners) return;
      ExchangeRdmaEpoch(false);
      executing_communication_ = true;
    } else if (ready_queue_) {
      ProcessVectorReady(mode, recorder);
      return;
    }
    uint32_t block_idx = rdma_owner ? 0 : GetVectorScheduleIndex();
    if (block_idx == INVALID_VECTOR_SCHEDULE_INDEX) return;
    const uint32_t count = this->vector_task_num;
    const uint32_t stride = this->active_vector_num;
#endif
    for (;;) {
      TaskId task_id;
#ifndef HMC_RUNTIME_CUBE
      if (rdma_owner) {
        if (!NextRdmaCommunicationTask(cursor, task_id)) break;
      } else
#endif
      {
        if (block_idx >= count) break;
        task_id = GetTaskIndex(block_idx);
        block_idx += stride;
      }
      ExecuteTask(task_id, mode, recorder);
    }
#ifndef HMC_RUNTIME_CUBE
    if (rdma_owner) {
      ExchangeRdmaEpoch(true);
      executing_communication_ = false;
    }
#endif
  }

  __aicore__ inline bool IsQueuedTaskReady(TaskId task_id) {
    const uint32_t dependent = *reinterpret_cast<__gm__ uint32_t *>(runtimeConfigPtr +
      getAllTasksOffset(runtimeConfigPtr) + task_id * sizeof(TaskDesc) + offsetof(TaskDesc, dependent_event));
    if (dependent == EVENT_INVALID_ID) {
      return true;
    }
    if (dependent >= runtime_event_capacity) {
      AscendC::Trap();
    }
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      all_event_counters[dependent]);
    return all_event_counters.GetValue(dependent) >= all_event_num_triggers.GetValue(dependent);
  }

  struct ReadyExecution {
    KernelWorkerBase &worker;
    DiagnosticMode mode;
    CycleTraceRecorder &recorder;
    __aicore__ inline TaskId TaskAt(uint32_t index) { return worker.GetTaskIndex(index); }
    __aicore__ inline bool IsReady(TaskId task) { return worker.IsQueuedTaskReady(task); }
    __aicore__ inline void Execute(TaskId task) {
      worker.ExecuteTask(task, mode, recorder);
    }
    __aicore__ inline void Wait(TaskId task) {
      const uint32_t dependent = *reinterpret_cast<__gm__ uint32_t *>(worker.runtimeConfigPtr +
        getAllTasksOffset(worker.runtimeConfigPtr) + task * sizeof(TaskDesc) + offsetof(TaskDesc, dependent_event));
      worker.WaitForDependency(dependent, 5);
    }
  };

  __aicore__ inline void ProcessVectorReady(DiagnosticMode mode, CycleTraceRecorder &recorder) {
    const uint32_t first = GetVectorScheduleIndex();
    if (first == INVALID_VECTOR_SCHEDULE_INDEX) return;
    ReadyExecution execution{*this, mode, recorder};
    RunVectorReadyQueue(execution, first, this->vector_task_num, this->active_vector_num);
  }

  __aicore__ inline bool LoadReadyHandshakeMeta(ReadyHandshakeMeta *meta) {
    getReadyHandshakeMeta(this->runtimeConfigPtr, meta);
    return meta->ready_event != 0;
  }

  __aicore__ inline void PublishDeviceReady(const ReadyHandshakeMeta &meta, bool completion = false) {
#ifndef HMC_RUNTIME_CUBE
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

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
  __aicore__ inline void CompleteDeviceReads(const ReadyHandshakeMeta &meta) {
    if (rdma_.enabled) {
#ifndef HMC_RUNTIME_CUBE
      if (this->worker_id_ % 2 == 0 && this->worker_id_ / 2 >= RdmaOwnerCount()) return;
#endif
      TriggerEvent(meta.completion_event);
#ifndef HMC_RUNTIME_CUBE
      if (this->worker_id_ == 1) WaitForDependency(meta.completion_event);
#endif
      return;
    }
#ifndef HMC_RUNTIME_CUBE
    if (!IsActiveVectorWorker()) {
      return;
    }
#endif
    TriggerEvent(meta.completion_event);
#ifndef HMC_RUNTIME_CUBE
    if (this->worker_id_ == 1) {
      WaitForDependency(meta.completion_event);
      if (meta.ready_event != 0 && !rdma_.enabled) {
        PublishDeviceReady(meta, true);
      }
    }
#endif
  }
#else
  __aicore__ inline void CompleteDeviceReads(const ReadyHandshakeMeta &meta) {
    if (rdma_.enabled) {
#ifndef HMC_RUNTIME_CUBE
      if (this->worker_id_ % 2 == 0 && this->worker_id_ / 2 >= RdmaOwnerCount()) return;
#endif
      TriggerEvent(meta.completion_event);
#ifndef HMC_RUNTIME_CUBE
      if (this->worker_id_ == 1) WaitForDependency(meta.completion_event);
#endif
      return;
    }
#ifndef HMC_RUNTIME_CUBE
    if (this->worker_id_ % 2 == 0) {
      return;
    }
#endif
    TriggerEvent(meta.completion_event);
#ifndef HMC_RUNTIME_CUBE
    if (this->worker_id_ == 1) {
      WaitForDependency(meta.completion_event);
      if (meta.ready_event != 0) {
        PublishDeviceReady(meta, true);
      }
    }
#endif
  }
#endif


#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
  __aicore__ inline void WaitForDeviceReady(const ReadyHandshakeMeta &meta) {
    if (rdma_.enabled) {
#ifndef HMC_RUNTIME_CUBE
      if (this->worker_id_ == 1) PrepareRdmaEpoch(meta);
#endif
      WaitForDependency(meta.ready_event);
      return;
    }
#ifdef HMC_RUNTIME_CUBE
    WaitForDependency(meta.ready_event);
#else
    // Every active sender must observe peer counter resets before issuing PUTs.
    if (GetVectorScheduleIndex() == INVALID_VECTOR_SCHEDULE_INDEX) {
      return;
    }
    if (this->worker_id_ == 1) {
      PublishDeviceReady(meta);
    }
    WaitForDependency(meta.ready_event);
#endif
  }
#else
  __aicore__ inline void WaitForDeviceReady(const ReadyHandshakeMeta &meta) {
    if (rdma_.enabled) {
#ifndef HMC_RUNTIME_CUBE
      if (this->worker_id_ == 1) PrepareRdmaEpoch(meta);
#endif
      WaitForDependency(meta.ready_event);
      return;
    }
#ifdef HMC_RUNTIME_CUBE
    WaitForDependency(meta.ready_event);
#else
    if (this->active_vector_num != this->vector_num && this->worker_id_ % VECTOR_WORKER_STRIDE == 0) {
      return;
    }
    const uint32_t leader = this->active_vector_num == this->vector_num ? 0 : 1;
    if (this->worker_id_ == leader) {
      PublishDeviceReady(meta);
    }
    WaitForDependency(meta.ready_event);
#endif
  }
#endif

  __aicore__ inline TaskId GetTaskIndex(uint32_t task_id) {
#ifdef HMC_RUNTIME_CUBE
    return cube_task_indexs.GetValue(task_id);
#else
    return vector_task_indexs.GetValue(task_id);
#endif
  }

  __aicore__ inline uint32_t GetVectorScheduleIndex() {
    if (ready_queue_all_) {
      return this->worker_id_ < this->active_vector_num ? this->worker_id_ : INVALID_VECTOR_SCHEDULE_INDEX;
    }
    if constexpr (WorkerPolicy == VectorWorkerPolicy::ALL) {
      return this->worker_id_ < this->active_vector_num ? this->worker_id_ : INVALID_VECTOR_SCHEDULE_INDEX;
    }
    return this->worker_id_ % 2 == 0 ? INVALID_VECTOR_SCHEDULE_INDEX : this->worker_id_ / 2;
  }

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
  __aicore__ inline bool IsActiveVectorWorker() const {
    if (WorkerPolicy == VectorWorkerPolicy::ALL || ready_queue_) {
      return this->worker_id_ < this->active_vector_num;
    }
    return this->worker_id_ % VECTOR_WORKER_STRIDE != 0;
  }
#endif

  __aicore__ inline void AtomicAddForAllEventCounters(uint32_t event_index) {
#if defined(HMC_RUNTIME_CUBE) && defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
    // A5 cannot publish an L1 buffer to GM; preserve the original scalar atomic completion path.
    PipeBarrier<PIPE_ALL>();
    (void)AscendC::AtomicAdd<int32_t>(this->all_event_counters.GetPhyAddr(event_index), 1);
    PipeBarrier<PIPE_ALL>();
#else
    TPipe eventPipe;
#ifdef HMC_RUNTIME_CUBE
    TBuf<AscendC::TPosition::A1> eventBuffer;
#else
    TBuf<AscendC::TPosition::VECOUT> eventBuffer;
#endif
    eventPipe.InitBuffer(eventBuffer, DISPATCH_TOKEN_UB_SIZE);
    LocalTensor<int32_t> localSet = eventBuffer.GetWithOffset<int32_t>(EXP_TOKEN_COUNT_FLAG_CNT, 0);
#ifdef HMC_RUNTIME_CUBE
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
#endif
  }

  __aicore__ inline void ResolveTaskDescriptor(TaskId &, TaskDesc &) {}

  __aicore__ inline void CaptureL1TaskTensors(TaskId task_id, uint32_t phase) {
    MulticoreRuntime::CaptureL1TaskTensors(input_list[Derived::DIAGNOSTIC_IDX], input_list, task_id, phase, worker_id_);
  }

  __aicore__ inline void LoadTaskDescriptor(TaskId task_id, TaskDesc &task_desc) {
    if (task_id >= runtime_task_capacity) {
      AscendC::Trap();
    }
    getTaskDesc(this->runtimeConfigPtr, &task_desc, task_id);
    static_cast<Derived *>(this)->ResolveTaskDescriptor(task_id, task_desc);
    if ((task_desc.dependent_event != EVENT_INVALID_ID && task_desc.dependent_event >= runtime_event_capacity) ||
        static_cast<uint64_t>(task_desc.trigger_event) + ATOMIC_ADD_VALUE_LEN > runtime_event_capacity) {
      AscendC::Trap();
    }
  }

  __aicore__ inline uint64_t NormalTraceNow() const {
    return normal_trace_.Enabled() ? normal_trace_.Now() : 0;
  }

  __aicore__ inline void RecordNormalTrace(TaskId task_id, const TaskDesc &task, uint64_t selected,
                                          uint64_t observed, uint64_t computed) {
    if (normal_trace_.Enabled()) {
      normal_trace_.Record(task_id, task, selected, observed, computed, normal_trace_.Now());
    }
  }

  __aicore__ inline void ExecuteTask(TaskId task_id, DiagnosticMode mode, CycleTraceRecorder &recorder) {
    const uint64_t selected = NormalTraceNow();
    TaskDesc task_desc;
    LoadTaskDescriptor(task_id, task_desc);
    if (rdma_.enabled && IsCommunication(task_desc) && !executing_communication_) return;
    const bool profile = mode == DiagnosticMode::PROFILE;
    const bool dump = mode == DiagnosticMode::TENSOR_DUMP_L1;
    const uint32_t owner_id = profile ? getTaskProfileOwnerId(this->runtimeConfigPtr, task_id) : 0;
    if (task_desc.dependent_event != EVENT_INVALID_ID) {
      const uint64_t start = profile ? recorder.Now() : 0;
      WaitForDependency(task_desc.dependent_event, task_desc.extra_value_2);
      if (profile) {
        recorder.Record(Derived::PROFILE_DESC_WAIT_DEPENDENCY, task_id, task_desc.task_index, owner_id,
                        start, recorder.Now());
      }
    }
    const uint64_t observed = NormalTraceNow();
    if (dump) CaptureL1TaskTensors(task_id, L1_DUMP_PHASE_BEFORE);
    const uint64_t compute_start = profile ? recorder.Now() : 0;
    replica_trace_ = profile ? &recorder : nullptr;
    replica_trace_task_ = task_id;
    if (UsesRdmaTransfer(task_desc)) {
      ExecuteRdmaTransfer(task_desc);
    } else {
      static_cast<Derived *>(this)->ExecuteComputeKernel(task_desc);
    }
    replica_trace_ = nullptr;
    const uint64_t computed = NormalTraceNow();
    if (profile) {
      const uint64_t compute_end = recorder.Now();
      uint32_t desc_id = getTaskProfileDescId(this->runtimeConfigPtr, task_id);
      if (desc_id == PROFILE_DESC_INVALID_ID) {
        desc_id = PROFILE_DESC_TASK_TYPE_BASE + static_cast<uint32_t>(task_desc.task_type);
      }
      recorder.Record(desc_id, task_id, task_desc.task_index, owner_id, compute_start, compute_end);
    }
    if (dump) {
      PipeBarrier<PIPE_ALL>();
      CaptureL1TaskTensors(task_id, L1_DUMP_PHASE_AFTER);
    }
    if (task_desc.task_type != TaskType::TASK_SHMEM_PUT_MEM_SIGNAL) {
      const uint64_t start = profile ? recorder.Now() : 0;
      TriggerProducerEvent(task_desc);
      TriggerEvent(task_desc.trigger_event);
      if (profile) {
        recorder.Record(Derived::PROFILE_DESC_TRIGGER_EVENT, task_id, task_desc.task_index, owner_id,
                        start, recorder.Now());
      }
    }
    RecordNormalTrace(task_id, task_desc, selected, observed, computed);
  }

  __aicore__ inline void WaitForDependency(uint32_t event_index, uint32_t poll_interval_us = 0) {
    // Pull producers signal locally after MTE3 completion. Observe short-lived
    // GMM/SwiGLU dependencies promptly while retaining a bounded refresh rate.
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
#ifdef HMC_RUNTIME_CUBE
    const uint32_t default_poll_interval_us = pull_protocol_ ? 10 : CUBE_EVENT_REFRESH_INTERVAL_US;
#else
    const uint32_t default_poll_interval_us = pull_protocol_ ? 30 : VECTOR_EVENT_REFRESH_INTERVAL_US;
#endif
#else
#ifdef HMC_RUNTIME_CUBE
    const uint32_t default_poll_interval_us = pull_protocol_ ? 10 : CUBE_EVENT_REFRESH_INTERVAL_UNITS;
#else
    const uint32_t default_poll_interval_us = pull_protocol_ ? 30 : VECTOR_EVENT_REFRESH_INTERVAL_UNITS;
#endif
#endif
    if (poll_interval_us == 0) {
      poll_interval_us = default_poll_interval_us;
    }
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
    constexpr int64_t cycles_per_us = SYSTEM_CYCLES_PER_US;
#else
    constexpr int64_t cycles_per_us = EVENT_REFRESH_TIME_UNIT_CYCLES;
#endif
    int32_t needed = all_event_num_triggers.GetValue(event_index);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
      all_event_counters[event_index]);
    int32_t current = all_event_counters.GetValue(event_index);
    if (rdma_.enabled && current < needed) current += rdma_.Received(event_index);
    PipeBarrier<PIPE_ALL>();
    int64_t previous_cycle = AscendC::GetSystemCycle();
    while (current < needed) {
      int64_t elapsed_cycles = AscendC::GetSystemCycle() - previous_cycle;
      if (elapsed_cycles / cycles_per_us > poll_interval_us) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
          all_event_counters[event_index]);
        current = all_event_counters.GetValue(event_index);
        if (rdma_.enabled && current < needed) current += rdma_.Received(event_index);
        previous_cycle = AscendC::GetSystemCycle();
      }
    }
  }

  __aicore__ inline void TriggerEvent(uint32_t event_index) { AtomicAddForAllEventCounters(event_index); }

  __aicore__ inline void TriggerProducerEvent(const TaskDesc &task) {
    // GMM producers can publish a separate full-expert fence while preserving
    // their ordinary graph completion counter and the Cube computation order.
    if (this->home_experts_ > 0 && task.task_type == TaskType::TASK_GROUPED_MATMUL && task.extra_value_1 > 3) {
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

  __aicore__ inline bool IsCommunication(const TaskDesc &task) const {
    return task.task_type == TaskType::TASK_SHMEM_PUT_MEM_SIGNAL || task.task_type == TaskType::TASK_SHMEM_GET_MEM;
  }

  __aicore__ inline uint32_t CommunicationPeer(const TaskDesc &task) const {
    const uint32_t ep = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1);
    return task.task_index / (task.task_split_num / ep);
  }

  __aicore__ inline uint32_t RdmaOwnerCount() const {
    const uint32_t ep = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1);
    return ep < core_num ? ep : core_num;
  }

  __aicore__ inline __gm__ int32_t *RdmaEpochStorage(bool completion) const {
    const uint32_t ep = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1);
    constexpr uint32_t stride = DATA_CACHE_LINE_SIZE / INT32_T_SIZE;
    auto ready = reinterpret_cast<__gm__ int32_t *>(input_list[Derived::EVENT_IDX]) + runtime_event_capacity;
    return completion ? ready + (ep + 1) * stride : ready;
  }

  __aicore__ inline void PrepareRdmaEpoch(const ReadyHandshakeMeta &meta) {
#ifndef HMC_RUNTIME_CUBE
    const uint32_t ep = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1);
    constexpr uint32_t stride = DATA_CACHE_LINE_SIZE / INT32_T_SIZE;
    auto source = RdmaEpochStorage(false) + ep * stride;
    dcci_cachelines(reinterpret_cast<GM_ADDR>(source), sizeof(int32_t));
    int32_t generation = *reinterpret_cast<__gm__ volatile int32_t *>(source);
    if (generation == INT32_MAX) AscendC::Trap();
    *source = generation + 1;
    dcci_cachelines(reinterpret_cast<GM_ADDR>(source), sizeof(int32_t));
    rdma_.PublishSignalSource(source);
    TriggerEvent(meta.ready_event);
#endif
  }

  __aicore__ inline void ExchangeRdmaEpoch(bool completion) {
#ifndef HMC_RUNTIME_CUBE
    ReadyHandshakeMeta meta;
    if (!LoadReadyHandshakeMeta(&meta)) return;
    const uint32_t ep = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1);
    const uint32_t rank = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 0) % ep;
    constexpr uint32_t stride = DATA_CACHE_LINE_SIZE / INT32_T_SIZE;
    auto source = RdmaEpochStorage(false) + ep * stride;
    auto flags = RdmaEpochStorage(completion);
    dcci_cachelines(reinterpret_cast<GM_ADDR>(source), sizeof(int32_t));
    const int32_t generation = *reinterpret_cast<__gm__ volatile int32_t *>(source);
    const uint32_t owner = worker_id_ / 2;
    // Publish every owned peer before waiting: one owner can cover several peers at larger EP sizes.
    for (uint32_t peer = owner; peer < ep; peer += RdmaOwnerCount()) {
      if (peer != rank) rdma_.SignalSet(flags + rank * stride, source, peer);
    }
    for (uint32_t peer = owner; peer < ep; peer += RdmaOwnerCount()) {
      if (peer != rank) aclshmem_signal_wait_until(flags + peer * stride, ACLSHMEM_CMP_GE, generation);
    }
#endif
  }

  __aicore__ inline bool NextRdmaCommunicationTask(RdmaQueueCursor &cursor, TaskId &selected) {
#ifndef HMC_RUNTIME_CUBE
    const uint32_t ep = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 1);
    while (cursor.phase < 2) {
      while (cursor.slot < vector_task_num) {
        const TaskId id = GetTaskIndex(cursor.slot++);
        auto task = reinterpret_cast<__gm__ const TaskDesc *>(
            runtimeConfigPtr + getAllTasksOffset(runtimeConfigPtr) + id * sizeof(TaskDesc));
        const auto kind = task->task_type;
        if (kind != TaskType::TASK_SHMEM_PUT_MEM_SIGNAL && kind != TaskType::TASK_SHMEM_GET_MEM) continue;
        const uint32_t peer = task->task_index / (task->task_split_num / ep);
        if (peer % cursor.owners != cursor.owner) continue;
        if ((task->inputs[1].input_position == 2) != (cursor.phase == 0)) continue;
        selected = id;
        return true;
      }
      for (uint32_t peer = cursor.owner; peer < ep; peer += cursor.owners) {
        if (rdma_.UsesRdma(peer)) rdma_.Quiet(peer);
      }
      ++cursor.phase;
      cursor.slot = 0;
    }
#endif
    return false;
  }

  __aicore__ inline bool UsesRdmaTransfer(const TaskDesc &task) const {
    return rdma_.enabled && IsCommunication(task) &&
      (rdma_.UsesRdma(CommunicationPeer(task)) ||
       (rdma_.row_indices != nullptr && task.inputs[1].input_position == 2));
  }

  __aicore__ inline void ExecuteRdmaTransfer(const TaskDesc &task) {
#ifndef HMC_RUNTIME_CUBE
    const int64_t experts = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 2);
    const int64_t hidden = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 3);
    const int64_t tiles = task.task_split_num / experts;
    const int64_t tile = task.task_index % tiles;
    const int64_t size = reinterpret_cast<__gm__ int32_t *>(
      input_list[task.inputs[3].input_position])[task.inputs[3].base_ptr_offset];
    const int64_t chunk = static_cast<int64_t>(task.task_split_value) * hidden;
    const int64_t start = tile * chunk;
    const uint32_t peer = CommunicationPeer(task);
    const bool pull = task.task_type == TaskType::TASK_SHMEM_GET_MEM;
    const bool network = rdma_.UsesRdma(peer);
    if (start < size) {
      const int64_t destination_offset = reinterpret_cast<__gm__ int64_t *>(
        input_list[task.inputs[0].input_position])[task.inputs[0].base_ptr_offset] + start;
      const int64_t source_offset = reinterpret_cast<__gm__ int64_t *>(
        input_list[task.inputs[2].input_position])[task.inputs[2].base_ptr_offset] + start;
      const uint32_t elements = static_cast<uint32_t>(size - start < chunk ? size - start : chunk);
      GM_ADDR destination = input_list[task.outputs[0].input_position] +
                            destination_offset * task.outputs[0].data_type;
      GM_ADDR source = input_list[task.inputs[1].input_position];
      if (rdma_.row_indices != nullptr && task.inputs[1].input_position == 2) {
        ExecuteIndexedRows(task, destination, source, source_offset / hidden, elements / hidden, peer);
      } else {
        source += source_offset * task.inputs[1].data_type;
        const uint32_t bytes = elements * task.inputs[1].data_type;
        if (pull) rdma_.Get(destination, source, bytes, peer);
        else rdma_.Put(destination, source, bytes, peer);
      }
      if (pull && network) rdma_.Quiet(peer);
    }
    if (!pull) {
      int32_t credit = start < size ? 1 : EmptyDispatchCredit(task.extra_value_0, tile, tiles, start, size, chunk);
      if (credit) {
        if (network) rdma_.Credit(task.trigger_event, credit, peer);
        else aclshmemx_signal_op(reinterpret_cast<__gm__ int32_t *>(input_list[Derived::EVENT_IDX]) +
                                  task.trigger_event, credit, ACLSHMEM_SIGNAL_ADD, peer);
      }
    }
#endif
  }

  __aicore__ inline void ExecuteIndexedRows(const TaskDesc &task, GM_ADDR destination, GM_ADDR source,
                                            uint32_t first, uint32_t rows, uint32_t peer) {
#ifndef HMC_RUNTIME_CUBE
    const uint32_t hidden = getExtraValueFromTiling(input_list[Derived::TILING_IDX], 3);
    const uint32_t row_bytes = hidden * task.inputs[1].data_type;
    const bool pull = task.task_type == TaskType::TASK_SHMEM_GET_MEM;
    const bool network = rdma_.UsesRdma(peer);
    auto indices = rdma_.row_indices + first;
    if (pull) {
      auto local_indices = rdma_.IndexScratch(peer);
      if (network) {
        rdma_.Get(reinterpret_cast<GM_ADDR>(local_indices), reinterpret_cast<GM_ADDR>(indices),
                  rows * sizeof(int32_t), peer);
        rdma_.Quiet(peer);
      } else {
        PullToLocal<int32_t>(reinterpret_cast<GM_ADDR>(local_indices), reinterpret_cast<GM_ADDR>(indices), rows, peer);
        // Communication owners have no TPipe. Keep this event separate from get_pipelined's EVENT_ID0/1.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID2);
      }
      indices = local_indices;
    }
    dcci_cachelines(reinterpret_cast<GM_ADDR>(indices), rows * sizeof(int32_t));
    for (uint32_t row = 0; row < rows; ++row) {
      int32_t source_row = *reinterpret_cast<__gm__ volatile int32_t *>(indices + row);
      if (source_row < 0 || static_cast<uint64_t>(source_row) >= rdma_.source_rows) AscendC::Trap();
      GM_ADDR original = source + static_cast<uint64_t>(source_row) * row_bytes;
      GM_ADDR target = destination + static_cast<uint64_t>(row) * row_bytes;
      if (network) {
        if (pull) rdma_.Get(target, original, row_bytes, peer);
        else rdma_.Put(target, original, row_bytes, peer);
      } else {
        if (pull) PullToLocal<uint8_t>(target, original, row_bytes, peer);
        else hyper_multicore::shmem::data_plane::put(target, original, row_bytes, peer);
      }
    }
#endif
  }

  RdmaTransport rdma_;
  bool executing_communication_ = false;

  bool pull_protocol_ = false;
  bool ready_queue_ = false;
  bool ready_queue_all_ = false;
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
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
  bool ready_queue_ = false;
#endif
  static constexpr uint32_t INVALID_VECTOR_SCHEDULE_INDEX = 0xFFFFFFFF;
};

}  // namespace MulticoreRuntime

#undef HMC_RUNTIME_CUBE
#endif  // HYPER_MULTICORE_CORE_MULTICORE_OPS_RUNTIME_WORKER_KERNEL_H_
