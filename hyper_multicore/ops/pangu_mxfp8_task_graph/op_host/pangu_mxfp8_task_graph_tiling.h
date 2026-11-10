// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef PANGU_MXFP8_TASK_GRAPH_TILING_H_
#define PANGU_MXFP8_TASK_GRAPH_TILING_H_
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(PanguMxfp8TaskGraphTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bufferCount);
TILING_DATA_FIELD_DEF(uint32_t, cubeCount);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PanguMxfp8TaskGraph, PanguMxfp8TaskGraphTilingData)
}  // namespace optiling
#endif
