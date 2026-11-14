/**
 * Copyright 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software distributed
 * under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied.
 */
#ifndef HYPER_MULTICORE_MULTICORE_GRADIENT_BAND_OBSERVER_H
#define HYPER_MULTICORE_MULTICORE_GRADIENT_BAND_OBSERVER_H

#include "gradient_band.h"
#include "replica_gradient.h"

namespace MulticoreRuntime {

// Allocated only by the profiled traversal. Probes never consume or reorder work.
class GradientBandObserver {
 public:
  __aicore__ inline void Init(__gm__ uint64_t *config, __gm__ uint64_t *owned, int32_t cores,
                              uint32_t projection, CycleTraceRecorder *trace, bool probe) {
    config_ = config;
    owned_ = owned;
    cores_ = cores;
    projection_ = projection;
    trace_ = trace;
    probe_ = probe;
  }

  __aicore__ inline bool Ready(uint32_t index, uint32_t band, uint32_t &prefix) {
    const uint64_t stride = config_[17] * cores_ * DATA_CACHE_LINE_SIZE;
    GM_ADDR owner = reinterpret_cast<GM_ADDR>(config_[12]) + owned_[index * 4 + 2] * stride;
    GM_ADDR guest = reinterpret_cast<GM_ADDR>(config_[13]) + owned_[index * 4 + 1] * stride;
    auto remote = hyper_multicore::shmem::data_plane::remote_ptr(
        reinterpret_cast<__gm__ void *>(guest), owned_[index * 4]);
    return AreBandProducersReady(owner, reinterpret_cast<GM_ADDR>(remote), band, cores_, config_[3], prefix);
  }

  __aicore__ inline bool Poll(uint32_t index, uint32_t band, uint32_t &prefix) {
    const bool first = pending_[band] == 0;
    if (first) {
      pending_[band] = trace_->Now();
    }
    if (!Ready(index, band, prefix)) {
      if (first) {
        const uint64_t failed = trace_->Now();
        Record(index, band, prefix % 2 ? ReplicaTraceEvent::GUEST_NOT_READY : ReplicaTraceEvent::LOCAL_NOT_READY,
               failed, failed, prefix / 2);
      }
      Probe(index, band);
      return false;
    }
    const uint64_t now = trace_->Now();
    Record(index, band, ReplicaTraceEvent::HEAD_PENDING, pending_[band], now);
    Record(index, band, ReplicaTraceEvent::READY_OBSERVED, now, now);
    pending_[band] = 0;
    attempts_[band] = 0;
    last_probe_[band] = 0;
    return true;
  }

  __aicore__ inline void Record(uint32_t index, uint32_t band, ReplicaTraceEvent event,
                                uint64_t start, uint64_t end, uint32_t producer_core = 63) {
    RecordReplica(*trace_, event, projection_, band, owned_[index * 4], config_[3],
                  config_[1], owned_[index * 4 + 2], start, end, producer_core);
  }

 private:
  __aicore__ inline void Probe(uint32_t head, uint32_t band) {
    constexpr uint32_t max_attempts = 4;
    constexpr uint32_t lookahead = 4;
    constexpr uint64_t interval_cycles = 5000;
    const uint64_t now = AscendC::GetSystemCycle();
    if (!probe_ || attempts_[band] >= max_attempts ||
        (last_probe_[band] != 0 && now - last_probe_[band] < interval_cycles)) {
      return;
    }
    ++attempts_[band];
    last_probe_[band] = now;
    for (uint32_t index = head + 1; index < config_[11] && index <= head + lookahead; ++index) {
      if (owned_[index * 4 + 2] == owned_[head * 4 + 2]) {
        continue;
      }
      uint32_t prefix = 0;
      if (Ready(index, band, prefix)) {
        const uint64_t observed = trace_->Now();
        uint32_t head_prefix = 0;
        if (!Ready(head, band, head_prefix)) {
          Record(index, band, ReplicaTraceEvent::INDEPENDENT_READY, observed, trace_->Now());
          attempts_[band] = max_attempts;
        }
        break;
      }
    }
  }

  __gm__ uint64_t *config_ = nullptr;
  __gm__ uint64_t *owned_ = nullptr;
  CycleTraceRecorder *trace_ = nullptr;
  int32_t cores_ = 0;
  uint32_t projection_ = 0;
  bool probe_ = false;
  uint64_t pending_[64] = {}, last_probe_[64] = {};
  uint32_t attempts_[64] = {};
};

}  // namespace MulticoreRuntime
#endif  // HYPER_MULTICORE_MULTICORE_GRADIENT_BAND_OBSERVER_H
