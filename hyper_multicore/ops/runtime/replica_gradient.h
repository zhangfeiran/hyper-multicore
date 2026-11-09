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
#ifndef HYPER_MULTICORE_MULTICORE_REPLICA_GRADIENT_H
#define HYPER_MULTICORE_MULTICORE_REPLICA_GRADIENT_H

#include "kernel_operator.h"
#include "data_plane/rma.h"

namespace MulticoreRuntime {

// Every worker owns disjoint blocks; peers visit each block in the same FP32 order.
__aicore__ inline void AddReplicaGradient(GM_ADDR output, GM_ADDR source, int64_t elements,
                                         int32_t peer, int32_t worker, int32_t workers) {
#ifndef __DAV_C220_CUBE__
  constexpr int32_t block_elements = 4096;
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
  pipe.InitBuffer(buffer, 2 * block_elements * sizeof(float));
  auto local = buffer.Get<float>();
  auto remote = local[block_elements];
  auto mapped = hyper_multicore::shmem::data_plane::remote_ptr(
      reinterpret_cast<__gm__ void *>(source), peer);
  AscendC::GlobalTensor<float> home, guest;
  home.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output), elements);
  guest.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(mapped), elements);
  for (int64_t offset = static_cast<int64_t>(worker) * block_elements; offset < elements;
       offset += static_cast<int64_t>(workers) * block_elements) {
    uint32_t count = static_cast<uint32_t>(elements - offset < block_elements ? elements - offset : block_elements);
    AscendC::DataCopyExtParams params{1, count * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> padding{false, 0, 0, 0};
    AscendC::DataCopyPad(local, home[offset], params, padding);
    AscendC::DataCopyPad(remote, guest[offset], params, padding);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::Add(local, local, remote, count);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::SetAtomicNone();
    AscendC::DataCopyPad(home[offset], local, params);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
  }
  pipe.Destroy();
#endif
}

__aicore__ inline void StoreReplicaCompletion(GM_ADDR address, int32_t epoch) {
  AscendC::GlobalTensor<int32_t> flag;
  flag.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(address), 1);
  flag.SetValue(0, epoch);
  AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                    AscendC::DcciDst::CACHELINE_OUT>(flag);
  AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void WaitReplicaCompletion(GM_ADDR address, int32_t epoch) {
  AscendC::GlobalTensor<int32_t> flag;
  flag.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(address), 1);
  __gm__ volatile int32_t *value = reinterpret_cast<__gm__ volatile int32_t *>(address);
  do {
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                      AscendC::DcciDst::CACHELINE_OUT>(flag);
  } while (*value < epoch);
}

// A band can remain in production for many tiles. Limit global cache refreshes
// while polling so its readers do not continually invalidate the producer line.
__aicore__ inline void WaitBandProducer(GM_ADDR address, int32_t epoch) {
  AscendC::GlobalTensor<int32_t> flag;
  flag.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(address), 1);
  __gm__ volatile int32_t *value = reinterpret_cast<__gm__ volatile int32_t *>(address);
  uint64_t refreshed = 0;
  constexpr uint64_t refresh_cycles = 50;
  do {
    const uint64_t now = AscendC::GetSystemCycle();
    if (now - refreshed >= refresh_cycles) {
      AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                        AscendC::DcciDst::CACHELINE_OUT>(flag);
      refreshed = now;
    }
  } while (*value < epoch);
}

__aicore__ inline bool IsBandProducerReady(GM_ADDR address, int32_t epoch) {
  AscendC::GlobalTensor<int32_t> flag;
  flag.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(address), 1);
  __gm__ volatile int32_t *value = reinterpret_cast<__gm__ volatile int32_t *>(address);
  AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                  AscendC::DcciDst::CACHELINE_OUT>(flag);
  return *value >= epoch;
}

__aicore__ inline bool AreBandProducersReady(GM_ADDR owner_flags, GM_ADDR guest_flags, int32_t band,
                                           int32_t cores, int32_t epoch, uint32_t &prefix) {
  while (prefix < 2 * cores) {
    const uint64_t offset = (band * cores + prefix / 2) * DATA_CACHE_LINE_SIZE;
    GM_ADDR address = (prefix % 2 ? guest_flags : owner_flags) + offset;
    if (!IsBandProducerReady(address, epoch)) {
      return false;
    }
    ++prefix;
  }
  return prefix == 2 * cores;
}

}  // namespace MulticoreRuntime
#endif  // HYPER_MULTICORE_MULTICORE_REPLICA_GRADIENT_H
