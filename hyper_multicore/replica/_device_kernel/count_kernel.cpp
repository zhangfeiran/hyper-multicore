// Copyright 2026 Huawei Technologies Co., Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#include "acl/acl.h"
#include "kernel_operator.h"  // NOLINT(build/include_subdir): flat CANN SDK header.
#define HOT_COUNT_DEVICE __aicore__ inline
#define HOT_COUNT_LOCAL __ubuf__
#include "./count_algorithm.h"

namespace {
constexpr int kBlocks = 32;
constexpr int kMaximumExperts = 512;
constexpr int kMaximumTopK = 8;
constexpr int kTileRows = 128;
constexpr int kStrideAlignment = 64;
constexpr int kMaximumStride = 576;
}  // namespace

inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg{0};

__global__ __aicore__ void replica_count_partials(GM_ADDR input, GM_ADDR partials, int64_t rows, int top_k, int experts,
                                                  int stride) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> tile_buffer, count_buffer;
  pipe.InitBuffer(tile_buffer, kTileRows * kMaximumTopK * sizeof(int64_t));
  pipe.InitBuffer(count_buffer, kMaximumStride * sizeof(int64_t));
  auto tile = tile_buffer.Get<int64_t>();
  auto memory = reinterpret_cast<__ubuf__ int64_t *>(tile.GetPhyAddr());
  auto counts = reinterpret_cast<__ubuf__ int64_t *>(count_buffer.Get<int64_t>().GetPhyAddr());
  for (int expert = 0; expert <= experts; ++expert) counts[expert] = 0;
  const int64_t per_block = (rows + kBlocks - 1) / kBlocks;
  const int64_t first = AscendC::GetBlockIdx() * per_block;
  const int64_t last = rows < first + per_block ? rows : first + per_block;
  AscendC::GlobalTensor<int64_t> source, output;
  source.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(input));
  output.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(partials));
  for (int64_t offset = first; offset < last; offset += kTileRows) {
    const int length = last - offset < kTileRows ? static_cast<int>(last - offset) : kTileRows;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
    AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(length * top_k * sizeof(int64_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int64_t> padding{false, 0, 0, 0};
    AscendC::DataCopyPad(tile, source[offset * top_k], copy, padding);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    for (int row = 0; row < length; ++row) hot::CountRow(memory + row * top_k, counts, top_k, experts);
  }
  // Each producer owns cache-line-aligned partials; no scalar writeback shares a line with another core.
  const int base = AscendC::GetBlockIdx() * stride;
  for (int expert = 0; expert <= experts; ++expert) output.SetValue(base + expert, counts[expert]);
  AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
    output);
}

__global__ __aicore__ void replica_count_merge(GM_ADDR partials, GM_ADDR result, int experts, int stride) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
  pipe.InitBuffer(buffer, kBlocks * kMaximumStride * sizeof(int64_t));
  auto local = buffer.Get<int64_t>();
  AscendC::GlobalTensor<int64_t> source, output;
  source.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(partials));
  output.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(result));
  AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(kBlocks * stride * sizeof(int64_t)), 0, 0, 0};
  AscendC::DataCopyPadExtParams<int64_t> padding{false, 0, 0, 0};
  AscendC::DataCopyPad(local, source, copy, padding);
  AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  auto memory = reinterpret_cast<__ubuf__ int64_t *>(local.GetPhyAddr());
  for (int expert = 0; expert <= experts; ++expert) {
    int64_t value = 0;
    for (int block = 0; block < kBlocks; ++block) value += memory[block * stride + expert];
    output.SetValue(expert, expert == experts ? static_cast<int64_t>(value != 0) : value);
  }
  AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
    output);
}

extern "C" int launch_replica_counts(void *stream, void *input, void *partials, void *result, int64_t rows, int top_k,
                                     int experts, int stride) {
  if (rows < 0 || top_k < 1 || top_k > kMaximumTopK || experts < 1 || experts > kMaximumExperts ||
      stride != (experts + kStrideAlignment) / kStrideAlignment * kStrideAlignment) {
    return ACL_ERROR_INVALID_PARAM;
  }
  replica_count_partials<<<kBlocks, nullptr, stream>>>(static_cast<uint8_t *>(input), static_cast<uint8_t *>(partials),
                                                       rows, top_k, experts, stride);
  const int status = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
  if (status) return status;
  replica_count_merge<<<1, nullptr, stream>>>(static_cast<uint8_t *>(partials), static_cast<uint8_t *>(result), experts,
                                              stride);
  return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
}

extern "C" int replica_counts_abi_version() { return 2; }
