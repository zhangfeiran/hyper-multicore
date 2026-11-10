// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef HYPER_MULTICORE_BF16_GMM_SHARED_WORKER_H_
#define HYPER_MULTICORE_BF16_GMM_SHARED_WORKER_H_

#include "blaze_worker.h"

namespace MulticoreBf16 {
constexpr uint32_t SHARED_BLAZE_TAG = 0x53424631U;

template <class Task>
__aicore__ inline void ExecuteSharedBlazeTask(const Task &task, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling_address,
                                              uint32_t logical_cube, uint32_t cubes) {
  if ASCEND_IS_AIV {
    return;
  }
  AscendC::AscendCUtils::SetOverflow(1);
  GET_TILING_DATA_MEMBER(GMMTilingData, mmTilingData, tiling, tiling_address);
  const Problem problem{
    static_cast<uint64_t>(tiling.M), static_cast<uint64_t>(tiling.N), static_cast<uint64_t>(tiling.Ka), 0, 0, 0};
  if (task.inputs[0].transpose_flag == 1) {
    RunBlaze<asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>(problem, a, b, c, tiling, logical_cube, cubes);
  } else if (task.inputs[1].transpose_flag == 1) {
    RunBlaze<asc::te::nd_ext_layout_ptn, asc::te::dn_ext_layout_ptn>(problem, a, b, c, tiling, logical_cube, cubes);
  } else {
    RunBlaze<asc::te::nd_ext_layout_ptn, asc::te::nd_ext_layout_ptn>(problem, a, b, c, tiling, logical_cube, cubes);
  }
}
}  // namespace MulticoreBf16
#endif
