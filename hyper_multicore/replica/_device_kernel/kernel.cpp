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
#include "kernel_operator.h"  // NOLINT(build/include_subdir): flat CANN SDK header.
#include "acl/acl.h"
#define HOT_DEVICE __aicore__ inline
#define HOT_LOCAL __ubuf__
#define HOT_OUTPUT __gm__
#include "./algorithm.h"
using AscendC::CacheLine;
using AscendC::DataCacheCleanAndInvalid;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::DcciDst;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::SetFlag;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;
using AscendC::WaitFlag;
inline __gm__ struct OpSystemRunCfg g_opSystemRunCfg{0};
__global__ __aicore__ void replica_planner(GM_ADDR input, GM_ADDR output, GM_ADDR dispatch, int ranks, int experts,
                                           int budget, int64_t capacity, int64_t target, int64_t minimum,
                                           int64_t admission, int64_t saved_rows) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> scratch;
  pipe.InitBuffer(scratch, 184320);
  auto local = scratch.Get<int64_t>();
  GlobalTensor<int64_t> counts;
  counts.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(input));
  DataCopyExtParams copy{1, static_cast<uint32_t>(ranks * experts * sizeof(int64_t)), 0, 0, 0};
  DataCopyPadExtParams<int64_t> padding{false, 0, 0, 0};
  DataCopyPad(local, counts, copy, padding);
  SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
  auto memory = reinterpret_cast<__ubuf__ int64_t *>(local.GetPhyAddr());
  hot::Solver(memory, ranks, experts, budget)
    .Run(reinterpret_cast<__gm__ int64_t *>(output), reinterpret_cast<__gm__ int32_t *>(dispatch), capacity, target,
         minimum, admission, saved_rows);
  GlobalTensor<int64_t> control;
  control.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(output));
  DataCacheCleanAndInvalid<int64_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(control);
}
extern "C" int launch_planner(void *stream, void *counts, void *control, void *dispatch, int ranks, int experts,
                              int budget, int64_t capacity, int64_t target, int64_t minimum, int64_t admission,
                              int64_t saved_rows) {
  replica_planner<<<1, nullptr, stream>>>(static_cast<uint8_t *>(counts), static_cast<uint8_t *>(control),
                                          static_cast<uint8_t *>(dispatch), ranks, experts, budget, capacity, target,
                                          minimum, admission, saved_rows);
  return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
}

extern "C" int planner_abi_version() { return 4; }
