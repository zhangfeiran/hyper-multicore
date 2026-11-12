/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "shmem.h"

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void one_sided_probe_kernel(
    GM_ADDR destination, GM_ADDR source, GM_ADDR control, uint32_t bytes, int32_t peer,
    int32_t generation, int32_t mode, uint64_t wait_cycles) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECOUT> buffer;
  pipe.InitBuffer(buffer, 128);
  auto local = buffer.Get<uint8_t>();
  auto scratch = reinterpret_cast<__ubuf__ uint8_t *>(local.GetPhyAddr());
  auto ready = reinterpret_cast<__gm__ int32_t *>(control);
  auto tag = ready + 32;
  auto result = reinterpret_cast<__gm__ uint64_t *>(ready + 128);
  uint64_t started = AscendC::GetSystemCycle();
  if (mode < 0) {
    while (AscendC::GetSystemCycle() - started < wait_cycles) {}
    result[0] = 0;
    result[1] = AscendC::GetSystemCycle() - started;
    dcci_cachelines(reinterpret_cast<__gm__ uint8_t *>(result), 32);
    return;
  }
  if (mode == 0) {
    if (bytes) {
      aclshmem_putmem(destination, source, bytes, peer);
      aclshmemx_mte_quiet();
    }
    aclshmemx_signal_op(ready, generation, ACLSHMEM_SIGNAL_SET, peer);
  } else {
    if (bytes) {
      aclshmemx_roce_put_nbi(reinterpret_cast<__gm__ uint8_t *>(destination),
                           reinterpret_cast<__gm__ uint8_t *>(source), scratch, bytes, peer, 0);
    }
    if (mode == 1) {
      aclshmemx_roce_quiet(peer, scratch, 0);
    }
    // The tag source is symmetric and immutable until the final quiet completes.
    // A single AIV owns QP0 for this peer; no remote atomic Add is required.
    aclshmemx_roce_put_nbi(reinterpret_cast<__gm__ uint8_t *>(ready),
                         reinterpret_cast<__gm__ uint8_t *>(tag), scratch, sizeof(int32_t), peer, 0);
    aclshmemx_roce_quiet(peer, scratch, 0);
  }
  uint64_t issued = AscendC::GetSystemCycle();
  uint64_t status = 0;
  while (true) {
    dcci_cachelines(reinterpret_cast<__gm__ uint8_t *>(ready), 32);
    if (*reinterpret_cast<__gm__ volatile int32_t *>(ready) >= generation) {
      break;
    }
    if (AscendC::GetSystemCycle() - issued > wait_cycles) {
      status = 1;
      break;
    }
  }
  result[0] = status;
  result[1] = issued - started;
  result[2] = AscendC::GetSystemCycle() - started;
  result[3] = static_cast<uint64_t>(*ready);
  auto words = reinterpret_cast<__gm__ uint32_t *>(destination);
  uint32_t count = bytes ? bytes / sizeof(uint32_t) : 8;
  for (uint32_t sample = 0; sample < 16; ++sample) {
    uint32_t index = static_cast<uint64_t>(count - 1) * sample / 15;
    dcci_cachelines(reinterpret_cast<__gm__ uint8_t *>(words + index), sizeof(uint32_t));
    result[4 + sample] = *reinterpret_cast<__gm__ volatile uint32_t *>(words + index);
  }
  dcci_cachelines(reinterpret_cast<__gm__ uint8_t *>(result), 160);
}

void launch_one_sided_probe(void *stream, void *destination, void *source, void *control, uint32_t bytes,
                            int32_t peer, int32_t generation, int32_t mode, uint64_t wait_cycles) {
  one_sided_probe_kernel<<<1, nullptr, stream>>>((GM_ADDR)destination, (GM_ADDR)source,
                                                (GM_ADDR)control, bytes, peer, generation,
                                                mode, wait_cycles);
}

// Keep both kernels in one translation unit: AscendC defines a per-unit tiling global.
namespace {
constexpr uint32_t kPeers = 8;
constexpr uint32_t kExperts = 8;
constexpr uint32_t kTokens = 2048;
constexpr uint32_t kRowBytes = 6144 * 2;
constexpr uint32_t kRowWords = kRowBytes / sizeof(uint32_t);

__aicore__ inline int32_t ReadFlag(__gm__ int32_t *flag) {
  dcci_cachelines(reinterpret_cast<__gm__ uint8_t *>(flag), 32);
  return *reinterpret_cast<__gm__ volatile int32_t *>(flag);
}
}  // namespace

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void pipeline_probe_kernel(
    GM_ADDR destination, GM_ADDR source, GM_ADDR control, uint32_t rows,
    int32_t rank, int32_t generation, uint64_t wait_cycles) {
  uint32_t worker = AscendC::GetBlockIdx();
  uint32_t local_rank = rank % kPeers;
  uint32_t opposite = rank < kPeers ? kPeers : 0;
  auto flags = reinterpret_cast<__gm__ int32_t *>(control);
  auto all_results = reinterpret_cast<__gm__ uint64_t *>(control);
  uint64_t started = AscendC::GetSystemCycle();
  if (worker < kPeers) {
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECOUT> buffer;
    pipe.InitBuffer(buffer, 128);
    auto local = buffer.Get<uint8_t>();
    auto scratch = reinterpret_cast<__ubuf__ uint8_t *>(local.GetPhyAddr());
    uint32_t peer = opposite + worker;
    auto remote_flag = flags + local_rank * 32;
    for (uint32_t expert = 0; expert < kExperts; ++expert) {
      uint64_t output_row = static_cast<uint64_t>(expert * kPeers + local_rank) * rows;
      for (uint32_t row = 0; row < rows; ++row) {
        uint32_t route = ((worker * kExperts + expert) * rows + row) * 17 + rank * 13;
        uint32_t source_row = route % kTokens;
        aclshmemx_roce_put_nbi(destination + (output_row + row) * kRowBytes,
                             source + static_cast<uint64_t>(source_row) * kRowBytes,
                             scratch, kRowBytes, peer, 0);
      }
      // Immutable prefix-mask records keep every outstanding WQE's source stable.
      // This AIV is the sole QP0 producer for peer, including data and credit writes.
      auto tag = reinterpret_cast<__gm__ uint8_t *>(flags + 256 + expert);
      aclshmemx_roce_put_nbi(reinterpret_cast<__gm__ uint8_t *>(remote_flag), tag,
                           scratch, sizeof(int32_t), peer, 0);
    }
    aclshmemx_roce_quiet(peer, scratch, 0);
    auto result = all_results + 160 + worker * 16;
    result[0] = started;
    result[1] = AscendC::GetSystemCycle();
    result[2] = kExperts * (rows + 1);
    result[3] = static_cast<uint64_t>(kExperts) * rows * kRowBytes;
    dcci_cachelines(reinterpret_cast<__gm__ uint8_t *>(result), 128);
    return;
  }

  auto samples = all_results + 336;
  auto snapshots = all_results + 480;
  uint64_t status = 0;
  all_results[321] = started;
  for (uint32_t expert = 0; expert < kExperts; ++expert) {
    while (true) {
      bool ready = true;
      for (uint32_t peer = 0; peer < kPeers; ++peer) {
        int32_t value = ReadFlag(flags + peer * 32);
        ready = ready && (static_cast<uint32_t>(value) >> 8) == static_cast<uint32_t>(generation) &&
                (value & (1 << expert)) != 0;
      }
      if (ready) break;
      if (AscendC::GetSystemCycle() - started > wait_cycles) {
        status = expert + 1;
        break;
      }
    }
    if (status) break;
    all_results[322 + expert] = AscendC::GetSystemCycle();
    for (uint32_t peer = 0; peer < kPeers; ++peer) {
      if (expert == 0) snapshots[peer] = ReadFlag(flags + peer * 32);
      auto data = reinterpret_cast<__gm__ uint32_t *>(destination) +
                  static_cast<uint64_t>(expert * kPeers + peer) * rows * kRowWords;
      uint32_t last = rows ? rows * kRowWords - 1 : 0;
      dcci_cachelines(reinterpret_cast<__gm__ uint8_t *>(data), 32);
      dcci_cachelines(reinterpret_cast<__gm__ uint8_t *>(data + last), 32);
      samples[(expert * kPeers + peer) * 2] = *reinterpret_cast<__gm__ volatile uint32_t *>(data);
      samples[(expert * kPeers + peer) * 2 + 1] =
          *reinterpret_cast<__gm__ volatile uint32_t *>(data + last);
    }
  }
  all_results[320] = status;
  dcci_cachelines(reinterpret_cast<__gm__ uint8_t *>(all_results + 320), 1536);
}

void launch_pipeline_probe(void *stream, uint8_t *destination, uint8_t *source, uint8_t *control,
                           uint32_t rows, int32_t rank, int32_t generation, uint64_t wait_cycles) {
  pipeline_probe_kernel<<<kPeers + 1, nullptr, stream>>>(destination, source, control, rows,
                                                       rank, generation, wait_cycles);
}
