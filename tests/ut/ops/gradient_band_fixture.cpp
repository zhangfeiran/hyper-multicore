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

void CheckLocalOutputProtocol(uint64_t *metadata) {
  auto first = metadata + 4;
  auto second = metadata + 32;
  for (auto record : {first, second}) record[0] = 11;
  if (!MulticoreRuntime::HasOnlyGuestBandGradients(metadata)) {
    throw std::runtime_error("local BF16 bands removed ordinary vector workers");
  }
  for (bool guest : {false, true}) {
    const auto bytes = MulticoreRuntime::GradientOutputElementBytes(first, guest ? 1 : 0, 1, sizeof(float));
    if (bytes != (guest ? sizeof(float) : sizeof(uint16_t))) {
      throw std::runtime_error("home and guest matrix strides share the wrong dtype");
    }
  }
  first[11] = 1;
  bool rejected_owner = false;
  try { MulticoreRuntime::GradientOutputElementBytes(first, 0, 1, sizeof(float)); }
  catch (const std::runtime_error &) { rejected_owner = true; }
  if (!rejected_owner) throw std::runtime_error("BF16 merge destination accepted a remote contribution");
  first[11] = 0;
  first[0] = 7;
  if (MulticoreRuntime::GradientOutputElementBytes(first, 0, 1, sizeof(float)) != sizeof(float)) {
    throw std::runtime_error("legacy FP32 output changed");
  }
}

void CheckGuestWorkerReservation() {
  std::array<uint64_t, 64> metadata = {};
  auto first = metadata.data() + 4;
  auto second = metadata.data() + 32;
  metadata[0] = 2;
  metadata[1] = 4;
  metadata[2] = 32;
  first[0] = 7;
  second[0] = 7;
  if (MulticoreRuntime::HasOnlyGuestBandGradients(nullptr) ||
      !MulticoreRuntime::HasOnlyGuestBandGradients(metadata.data()) ||
      !MulticoreRuntime::HasOnlyGuestBandGradients(first)) {
    throw std::runtime_error("guest-only band metadata not recognized");
  }
  for (auto record : {first, second}) {
    record[11] = 1;
    if (MulticoreRuntime::HasOnlyGuestBandGradients(metadata.data())) {
      throw std::runtime_error("owner gradient worker reservation was removed");
    }
    record[11] = 0;
    record[0] = 1;
    if (MulticoreRuntime::HasOnlyGuestBandGradients(metadata.data()) ||
        MulticoreRuntime::HasOnlyGuestBandGradients(record)) {
      throw std::runtime_error("full-expert publication workers were removed");
    }
    record[0] = 7;
  }
  CheckLocalOutputProtocol(metadata.data());
}

void CheckFinalGradientLayout() {
  std::array<uint64_t, 64> final_record = {};
  auto final = final_record.data();
  final[0] = 13;
  final[11] = 3;
  final[18] = 4096;
  final[19] = 3;
  final[20] = 2;
  final[21] = 36;
  final[22] = 39;
  final[23] = 42;
  final[36] = 1;
  final[38] = 2;
  for (uint32_t expert = 0; expert < 4; ++expert) {
    const auto bytes = MulticoreRuntime::GradientOutputElementBytes(final, expert, 3, sizeof(float));
    if (bytes != (expert == 1 ? sizeof(uint16_t) : sizeof(float))) {
      throw std::runtime_error("compact scratch, direct home or guest stride is incorrect");
    }
  }
  final[38] = 3;
  bool rejected_scratch = false;
  try { MulticoreRuntime::GradientScratchIndex(final, 2); }
  catch (const std::runtime_error &) { rejected_scratch = true; }
  if (!rejected_scratch) throw std::runtime_error("out-of-bounds scratch index accepted");
}

void CheckBandPublication() {
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
}

void CheckBandPartition() {
  std::array<int, 9> visits = {};
  for (int worker = 0; worker < 20; ++worker) {
    auto part = MulticoreRuntime::PartitionGradientBands(worker, 20, 9);
    if (part.worker == 0) {
      for (int band = part.group; band < 9; band += part.groups) ++visits[band];
    }
  }
  for (int count : visits) if (count != 1) throw std::runtime_error("independent band groups overlap");
}

int main() {
  CheckGuestWorkerReservation();
  CheckFinalGradientLayout();
  CheckBandPublication();
  CheckBandPartition();
  std::cout << "Verified final-tile fence, empty bands, duplicate rejection and disjoint groups\n";
}
