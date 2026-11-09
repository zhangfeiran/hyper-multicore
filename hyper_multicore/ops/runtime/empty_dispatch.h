/** Copyright 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
 */
#ifndef HYPER_MULTICORE_RUNTIME_EMPTY_DISPATCH_H_
#define HYPER_MULTICORE_RUNTIME_EMPTY_DISPATCH_H_

#include <cstdint>

#if defined(__CCE_AICORE__)
#define HP_EMPTY_DISPATCH_INLINE __aicore__ inline
#else
#define HP_EMPTY_DISPATCH_INLINE inline
#endif

namespace MulticoreRuntime {

constexpr uint32_t BATCH_EMPTY_DISPATCH = 0x40000000;

// Called only for empty tiles. Each group retains one credit per original tile;
// active tiles still signal individually after their payload is visible.
HP_EMPTY_DISPATCH_INLINE int32_t EmptyDispatchCredit(uint32_t marker, int32_t tile, int32_t tiles,
                                                    int32_t start, int32_t elements, int32_t tile_elements) {
  if (marker != BATCH_EMPTY_DISPATCH) return 1;
  return start - tile_elements < elements ? tiles - tile : 0;
}

}  // namespace MulticoreRuntime
#undef HP_EMPTY_DISPATCH_INLINE
#endif  // HYPER_MULTICORE_RUNTIME_EMPTY_DISPATCH_H_
