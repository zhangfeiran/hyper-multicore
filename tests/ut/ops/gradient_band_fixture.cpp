/** Copyright 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
 */
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#define __aicore__
#define __gm__
using GM_ADDR = unsigned char *;
using event_t = int;
constexpr int PIPE_ALL = 0;
uint64_t cycle = 0, fence = 0, published = 0;
namespace AscendC {
inline uint64_t GetSystemCycle() { return 0; }

enum class HardEvent { FIX_MTE2 };
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
inline void Trap() { throw std::runtime_error("invalid producer state"); }
template <int> void PipeBarrier() { ++cycle; }
template <HardEvent> void SetFlag(event_t) { ++cycle; }
template <HardEvent> void WaitFlag(event_t) { fence = ++cycle; }
template <class T> struct GlobalTensor {
  T *data;
  void SetGlobalBuffer(T *value, uint32_t) { data = value; }
  void SetValue(uint32_t index, T value) { data[index] = value; published = ++cycle; }
};
template <class T, CacheLine, DcciDst> void DataCacheCleanAndInvalid(GlobalTensor<T>) { ++cycle; }
}  // namespace AscendC
struct Pipe { int FetchEventID(AscendC::HardEvent) { return 1; } } pipe;
Pipe *GetTPipePtr() { return &pipe; }
#include "gradient_band.h"

int main() {
  std::array<int32_t, 32> flags = {};
  MulticoreRuntime::GradientBandProducer producer;
  producer.Set(reinterpret_cast<GM_ADDR>(flags.data()), 7, 4);
  producer.Begin(8, 0, 1);
  producer.Expect(0, 2);
  producer.Expect(2, 2);
  producer.PublishEmpty();
  if (flags[0] != 0 || flags[16] != 7) throw std::runtime_error("empty contribution corrupted pending output");
  producer.Retire(0, 2);
  if (flags[0] != 0) throw std::runtime_error("partial band published before its last tile");
  producer.Retire(2, 2);
  if (flags[0] != 7 || !fence || published <= fence) throw std::runtime_error("band published without output fence");
  bool trapped = false;
  try { producer.Retire(2, 2); } catch (const std::runtime_error &) { trapped = true; }
  if (!trapped) throw std::runtime_error("duplicate retirement accepted");
  std::array<int, 9> visits = {};
  for (int worker = 0; worker < 20; ++worker) {
    auto part = MulticoreRuntime::PartitionGradientBands(worker, 20, 9);
    if (part.worker == 0) {
      for (int band = part.group; band < 9; band += part.groups) ++visits[band];
    }
  }
  for (int count : visits) if (count != 1) throw std::runtime_error("independent band groups overlap");
  std::cout << "Verified final-tile fence, empty bands, duplicate rejection and disjoint groups\n";
}
