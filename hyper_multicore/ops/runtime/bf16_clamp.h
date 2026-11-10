// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef HYPER_MULTICORE_BF16_CLAMP_H_
#define HYPER_MULTICORE_BF16_CLAMP_H_
#include <cstdint>
#if defined(__CCE_AICORE__) || defined(__NPU_ARCH__)
#define HYPER_CLAMP_INLINE __aicore__ inline
#else
#define HYPER_CLAMP_INLINE inline
#endif

namespace MulticoreRuntime {
HYPER_CLAMP_INLINE float Bf16ClampLimit(float value) {
  union { float value; uint32_t bits; } encoded{value};
  if (!(value > 0.0f) || (encoded.bits & 0x7f800000U) == 0x7f800000U) return value;
  const uint32_t rounded = (encoded.bits + 0x7fffU + ((encoded.bits >> 16) & 1U)) & 0xffff0000U;
  // Keep an enabled tiny limit enabled; zero denotes the legacy unclipped path.
  if (rounded == 0) return value;
  encoded.bits = rounded;
  return encoded.value;
}
}  // namespace MulticoreRuntime
#undef HYPER_CLAMP_INLINE
#endif
