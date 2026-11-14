/*
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include <iostream>
#include "hyper_multicore/ops/runtime/core_type.h"
#include "hyper_multicore/ops/runtime/normal_task_trace_layout.h"

using namespace MulticoreRuntime;

#ifdef EXPECT_CUBE_ROLE
#ifdef HMC_RUNTIME_CUBE
  static_assert(EXPECT_CUBE_ROLE == 1, "Vector compile selected a Cube trace slot");
#else
  static_assert(EXPECT_CUBE_ROLE == 0, "Cube compile selected a Vector trace slot");
#endif
#endif

int main() {
  constexpr uint64_t bytes = (24 + 48) * (64 + 64 * 1024);
  if (!NormalTaskTraceStorageValid(1024, 48, 24, bytes) ||
      NormalTaskTraceStorageValid(1024, 48, 24, bytes - 1) ||
      NormalTaskTraceStorageValid(0, 48, 24, bytes) ||
      NormalTaskTraceStorageValid(16385, 48, 24, bytes) ||
      NormalTaskTraceStorageValid(1, 0, 24, bytes) ||
      NormalTaskTraceStorageValid(1, 47, 24, bytes) ||
      NormalTaskTraceStorageValid(1, 48, 23, bytes) ||
      !NormalTaskTraceWorkerValid(23, 48, true) || NormalTaskTraceWorkerValid(24, 48, true) ||
      !NormalTaskTraceWorkerValid(47, 48, false) || NormalTaskTraceWorkerValid(48, 48, false) ||
      NormalTaskTraceWorkerValid(8, 16, true) || NormalTaskTraceWorkerValid(16, 16, false)) {
    return 1;
  }
  const NormalTaskTraceHeader header{NORMAL_TASK_TRACE_MAGIC, 10, 20, 1, 0, 1024, 2, 47, 1, {0, 0}};
  const NormalTaskTraceRecord record{11, 12, 13, 14, 123, 142, 47, 99, 100, 0, 576, 8};
  std::cout.write(reinterpret_cast<const char *>(&header), sizeof(header));
  std::cout.write(reinterpret_cast<const char *>(&record), sizeof(record));
  return 0;
}
