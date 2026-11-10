// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef PANGU_MX_SOURCE_ROW_LAYOUT_H_
#define PANGU_MX_SOURCE_ROW_LAYOUT_H_

#include "mx_launch_descriptor.h"

namespace MulticoreMx {
constexpr uint32_t MX_SOURCE_ROW_VERSION = 14;
constexpr uint32_t MX_SOURCE_ROW_RESERVED = 13;
constexpr uint32_t MX_SOURCE_SCALE_POSITION = 27;
constexpr uint32_t MX_SOURCE_INDEX_POSITION = 28;
constexpr uint32_t MX_SOURCE_ROW_BUFFER_COUNT = 29;
constexpr uint32_t MX_SOURCE_ROW_READY_TASK = 166;

#ifdef __NPU_ARCH__
#define MX_SOURCE_INLINE __aicore__ inline
#define MX_SOURCE_GM __gm__
#else
#define MX_SOURCE_INLINE inline
#define MX_SOURCE_GM
#endif

MX_SOURCE_INLINE bool IsSourceRowProtocolValid(const MX_SOURCE_GM MxLaunchDescriptor *launch) {
  return launch->version == MX_SOURCE_ROW_VERSION && launch->reserved == MX_SOURCE_ROW_RESERVED &&
         launch->buffer_count == MX_SOURCE_ROW_BUFFER_COUNT && launch->hidden >= 1024 && launch->hidden <= 8192 &&
         launch->hidden % 1024 == 0 && launch->local_tokens > 0 && launch->local_tokens <= (1 << 24);
}

MX_SOURCE_INLINE bool AreSourceRowPlanesValid(const MX_SOURCE_GM MxLaunchDescriptor *launch) {
  const uint64_t row_bytes = launch->buffer_bytes[16];
  return row_bytes > 0 && row_bytes <= UINT64_MAX / 2 && row_bytes % launch->hidden == 0 &&
         launch->buffer_bytes[0] == row_bytes * 2 && launch->buffer_bytes[17] == row_bytes / 32 &&
         launch->buffer_bytes[18] == row_bytes;
}

MX_SOURCE_INLINE bool IsSourceRowLaunchValid(const MX_SOURCE_GM MxLaunchDescriptor *launch) {
  if (!IsSourceRowProtocolValid(launch)) return false;
  const uint64_t width = launch->hidden;
  const uint64_t tokens = launch->local_tokens;
  const uint64_t source_indices = launch->buffer_bytes[MX_SOURCE_INDEX_POSITION];
  if (source_indices == 0 || source_indices % (tokens * sizeof(int32_t))) return false;
  const uint64_t top_k = source_indices / (tokens * sizeof(int32_t));
  return top_k <= 16 && tokens * top_k <= (1 << 24) &&
         launch->buffer_bytes[2] == tokens * width &&
         launch->buffer_bytes[MX_SOURCE_SCALE_POSITION] == tokens * (width / 32) && AreSourceRowPlanesValid(launch);
}

MX_SOURCE_INLINE uint64_t SourceRowPhysicalBufferBytes(const MX_SOURCE_GM MxLaunchDescriptor *launch,
                                                       uint32_t position) {
  const bool source = launch->version == MX_SOURCE_ROW_VERSION && launch->reserved == MX_SOURCE_ROW_RESERVED;
  return source && position >= 16 && position <= 18 ? launch->buffer_bytes[0] : launch->buffer_bytes[position];
}

MX_SOURCE_INLINE bool AreSourceRowBaseAliasesValid(const MX_SOURCE_GM MxLaunchDescriptor *launch,
                                                   uint64_t base, uint64_t row, uint64_t scale, uint64_t column) {
  return base != 0 && launch->buffer_bytes[0] <= UINT64_MAX - base && row == base && scale == base && column == base;
}

}  // namespace MulticoreMx
#undef MX_SOURCE_INLINE
#undef MX_SOURCE_GM
#endif
