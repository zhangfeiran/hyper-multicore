// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef HYPER_MULTICORE_EMPTY_TAIL_CREDIT_H_
#define HYPER_MULTICORE_EMPTY_TAIL_CREDIT_H_

#include <cstdint>

#if defined(__CCE_AICORE__) || defined(__NPU_ARCH__)
#define MULTICORE_EMPTY_TAIL_INLINE __aicore__ inline
#else
#define MULTICORE_EMPTY_TAIL_INLINE inline
#endif

namespace MulticoreRuntime {

MULTICORE_EMPTY_TAIL_INLINE int32_t EmptyTailCompletionCredit(
    int64_t tile_index, int64_t tile_count, int64_t elements, int64_t tile_elements) {
  // The first empty tile contributes the whole tail. Active tiles still signal
  // only after their payload completes, so the original threshold stays valid.
  const int64_t first_empty = (elements + tile_elements - 1) / tile_elements;
  return tile_index == first_empty ? static_cast<int32_t>(tile_count - first_empty) : 0;
}

}  // namespace MulticoreRuntime

#undef MULTICORE_EMPTY_TAIL_INLINE
#endif
