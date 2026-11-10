// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef PANGU_MULTICORE_MX_TASK_FLAGS_H_
#define PANGU_MULTICORE_MX_TASK_FLAGS_H_

#include <stdint.h>

namespace MulticoreMx {
constexpr uint32_t MX_PROJECTION_READY_MASK = 0x0000FFFFU;
constexpr uint32_t MX_READY_DEPTH8 = 0x00010000U;
constexpr uint32_t MX_READY_DEPTH1 = 0x00020000U;
}  // namespace MulticoreMx
#endif
