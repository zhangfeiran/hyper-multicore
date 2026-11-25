// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef HYPER_MULTICORE_OPS_COMMON_SHMEM_TASK_OPTIONS_H_
#define HYPER_MULTICORE_OPS_COMMON_SHMEM_TASK_OPTIONS_H_
#include <cstdint>
#include "../runtime/runtime_config.hpp"

namespace MulticoreRuntime {
// Mirrors the option definitions in tasks/alltoall.py.
enum class ShmemTaskPartition : uint32_t {
  FIXED_ROWS = 0,
  RUNTIME_ROWS = 2,
};
enum class ShmemTransferEngine : uint32_t { MTE = 0, UDMA = 16 };
constexpr uint32_t SHMEM_UDMA_QP1 = 1U << 5;
inline __aicore__ ShmemTransferEngine GetShmemTransferEngine(const TaskDesc &task) {
  return static_cast<ShmemTransferEngine>(task.extra_value_1 & 16U);
}
inline __aicore__ uint32_t GetShmemUdmaQp(const TaskDesc &task) {
  return (task.extra_value_1 & SHMEM_UDMA_QP1) ? 1 : 0;
}
}  // namespace MulticoreRuntime
#endif  // HYPER_MULTICORE_OPS_COMMON_SHMEM_TASK_OPTIONS_H_
