/**
 * Copyright 2026 Huawei Technologies Co., Ltd.
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
#ifndef HYPER_MULTICORE_MULTICORE_GRADIENT_BAND_H
#define HYPER_MULTICORE_MULTICORE_GRADIENT_BAND_H

#include "kernel_operator.h"
#include "replica_trace.h"

namespace MulticoreRuntime {

__aicore__ inline bool IsGradientBandProtocol(uint64_t protocol) {
  return protocol == 7 || protocol == 11 || protocol == 13;
}

__aicore__ inline uint32_t GradientRecordHeaderWords(__gm__ const uint64_t *record) {
  return record[0] == 13 ? 24 : (IsGradientBandProtocol(record[0]) ? 18 :
                                  ((record[0] == 8 || record[0] == 9) ? 16 : 12));
}

__aicore__ inline uint32_t GradientScratchIndex(__gm__ const uint64_t *record, uint32_t expert) {
  if (record[0] != 13 || expert >= record[19] || record[21] != 24 + record[10] * 3 + record[11] * 4 ||
      record[22] != record[21] + record[19] || record[23] != record[22] + record[11]) {
    AscendC::Trap();
  }
  const uint64_t encoded = record[record[21] + expert];
  if (encoded > record[20] || (encoded != 0 && record[18] == 0)) {
    AscendC::Trap();
  }
  return encoded;
}

__aicore__ inline uint32_t GradientOutputElementBytes(__gm__ const uint64_t *record, uint32_t expert,
                                                      uint32_t home_experts,
                                                      uint32_t descriptor_bytes) {
  if (record == nullptr || (record[0] != 11 && record[0] != 13)) {
    return descriptor_bytes;
  }
  if (descriptor_bytes != sizeof(float) || (record[0] == 11 && record[11] != 0)) {
    AscendC::Trap();
  }
  if (expert >= home_experts) {
    return sizeof(float);
  }
  return record[0] == 13 && GradientScratchIndex(record, expert) ? sizeof(float) : sizeof(uint16_t);
}

__aicore__ inline bool HasOnlyGuestBandGradients(__gm__ const uint64_t *config) {
  if (config == nullptr) {
    return false;
  }
  if (config[0] == 2) {
    const auto first = config + config[1];
    const auto second = config + config[2];
    return IsGradientBandProtocol(first[0]) && IsGradientBandProtocol(second[0]) &&
           first[11] == 0 && second[11] == 0;
  }
  return IsGradientBandProtocol(config[0]) && config[11] == 0;
}

struct GradientBandPartition {
  int32_t groups;
  int32_t group;
  int32_t worker;
  int32_t workers;
};

__aicore__ inline GradientBandPartition PartitionGradientBands(int32_t worker, int32_t workers, int32_t bands) {
  if (worker < 0 || workers <= 0 || worker >= workers || bands <= 0) {
    AscendC::Trap();
  }
  int32_t groups = bands < 4 ? bands : 4;
  groups = workers < groups ? workers : groups;
  const int32_t group = worker % groups;
  return {groups, group, worker / groups, (workers - 1 - group) / groups + 1};
}

// Only final complete-K output writes contribute. Each Cube owns its padded
// flag; no remote atomic counter or AIV coordinator participates in production.
class GradientBandProducer {
 public:
  static constexpr uint32_t kMaxBands = 64;

  __aicore__ inline void Set(GM_ADDR flags, int32_t epoch, uint32_t band_rows) {
    flags_ = flags;
    epoch_ = epoch;
    band_rows_ = band_rows;
  }

  __aicore__ inline bool Enabled() const { return flags_ != nullptr; }

  __aicore__ inline void SetProfile(CycleTraceRecorder *trace, uint32_t projection, uint32_t rank,
                                   uint32_t expert) {
    trace_ = trace;
    projection_ = projection;
    rank_ = rank;
    expert_ = expert;
  }

  __aicore__ inline void Begin(uint32_t rows, uint32_t core, uint32_t cores) {
    if (band_rows_ == 0) {
      AscendC::Trap();
    }
    bands_ = (rows + band_rows_ - 1) / band_rows_;
    core_ = core;
    cores_ = cores;
    if (band_rows_ == 0 || bands_ > kMaxBands || epoch_ <= 0 || core >= cores) {
      AscendC::Trap();
    }
    for (uint32_t band = 0; band < bands_; ++band) {
      pending_[band] = 0;
    }
  }

  __aicore__ inline void Expect(uint32_t row, uint32_t rows) {
    for (uint32_t band = row / band_rows_; band <= (row + rows - 1) / band_rows_; ++band) {
      ++pending_[band];
    }
  }

  __aicore__ inline void PublishEmpty() {
    for (uint32_t band = 0; band < bands_; ++band) {
      if (pending_[band] == 0) {
        Publish(band, true);
      }
    }
  }

  __aicore__ inline void Retire(uint32_t row, uint32_t rows) {
    const uint32_t first = row / band_rows_;
    const uint32_t last = (row + rows - 1) / band_rows_;
    bool publish = false;
    for (uint32_t band = first; band <= last; ++band) {
      if (pending_[band] == 0) {
        AscendC::Trap();
      }
      publish = --pending_[band] == 0 || publish;
    }
    if (!publish) {
      return;
    }
    // IterateAll<false> can return before the final GM output. This is the
    // installed MatmulClient::WaitIterateAll FIX_MTE2 completion boundary.
    auto event = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::FIX_MTE2));
    AscendC::SetFlag<AscendC::HardEvent::FIX_MTE2>(event);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE2>(event);
    AscendC::PipeBarrier<PIPE_ALL>();
    const uint64_t readable = trace_ == nullptr ? 0 : trace_->Now();
    for (uint32_t band = first; band <= last; ++band) {
      if (pending_[band] == 0) {
        Publish(band, false, readable);
      }
    }
  }

 private:
  __aicore__ inline void Publish(uint32_t band, bool empty, uint64_t readable = 0) {
    AscendC::GlobalTensor<int32_t> flag;
    flag.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(flags_ + (band * cores_ + core_) * 64), 1);
    AscendC::PipeBarrier<PIPE_ALL>();
    const uint64_t start = trace_ == nullptr ? 0 : (readable ? readable : trace_->Now());
    flag.SetValue(0, epoch_);
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                      AscendC::DcciDst::CACHELINE_OUT>(flag);
    AscendC::PipeBarrier<PIPE_ALL>();
    if (trace_ != nullptr) {
      RecordReplica(*trace_, empty ? ReplicaTraceEvent::EMPTY_READY : ReplicaTraceEvent::PRODUCER_READY,
                    projection_, band, rank_, epoch_, rank_, expert_, start, trace_->Now(), core_);
    }
  }

  GM_ADDR flags_ = nullptr;
  int32_t epoch_ = 0;
  uint32_t band_rows_ = 0;
  uint32_t bands_ = 0;
  uint32_t core_ = 0;
  uint32_t cores_ = 0;
  uint32_t pending_[kMaxBands];
  CycleTraceRecorder *trace_ = nullptr;
  uint32_t projection_ = 0, rank_ = 0, expert_ = 0;
};

}  // namespace MulticoreRuntime
#endif  // HYPER_MULTICORE_MULTICORE_GRADIENT_BAND_H
