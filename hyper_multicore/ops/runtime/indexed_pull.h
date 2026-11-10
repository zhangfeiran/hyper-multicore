/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under CANN Open Software License Agreement Version 1.0.
 * See LICENSE in the root of the software repository for the full text.
 */
#ifndef HYPER_MULTICORE_INDEXED_PULL_H
#define HYPER_MULTICORE_INDEXED_PULL_H

#include "../bf16_gmm/tile_ready.h"

// Forward-only scalar route publication at port 16. Blaze F2 leaves this
// legacy workspace argument unused; all backward transport remains unchanged.
namespace MulticoreRuntime {
  template <typename T, bool Count = false>
  __aicore__ inline void PullOriginalRows(__gm__ T *source, __gm__ T *destination, __ubuf__ int32_t *metadata,
                                         uint64_t scratch, int64_t rows, int64_t hidden, uint32_t top_k,
                                         uint32_t source_tokens, uint64_t *statistics) {
    constexpr int64_t CHUNK = 3072;
    uint64_t value_bytes = CHUNK * sizeof(T);
    if (scratch + 2 * value_bytes > 192 * 1024) {
      AscendC::Trap();
    }
    __ubuf__ T *buffers[2] = {reinterpret_cast<__ubuf__ T *>(scratch),
                            reinterpret_cast<__ubuf__ T *>(scratch + value_bytes)};
    for (int64_t col = 0; col < hidden; col += CHUNK) {
      int64_t elements = hidden - col < CHUNK ? hidden - col : CHUNK;
      uint64_t bytes = elements * sizeof(T);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
      AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
      int64_t first = metadata[0] / top_k;
      if (first < 0 || first >= source_tokens) { AscendC::Trap(); }
      aclshmemi_copy_gm2ub(buffers[0], source + first * hidden + col, bytes);
      if constexpr (Count) { statistics[0] += bytes; }
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      for (int64_t row = 0; row < rows; ++row) {
        uint32_t parity = row & 1;
        AscendC::TEventID event = parity ? EVENT_ID1 : EVENT_ID0;
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        if (row + 1 < rows) {
          AscendC::TEventID next = parity ? EVENT_ID0 : EVENT_ID1;
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(next);
          int64_t token = metadata[(row + 1) * 4] / top_k;
          if (token < 0 || token >= source_tokens) { AscendC::Trap(); }
          aclshmemi_copy_gm2ub(buffers[1 - parity], source + token * hidden + col, bytes);
          if constexpr (Count) { statistics[0] += bytes; }
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(next);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);
        aclshmemi_copy_ub2gm(destination + row * hidden + col, buffers[parity], bytes);
        if constexpr (Count) { statistics[1] += bytes; }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
      }
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }
  }

  template <typename T, bool Count = false, bool Publish = false>
  __aicore__ inline void PullPeerRows(
      __gm__ T *source, __gm__ T *destination, __gm__ int32_t *remote_metadata,
      __ubuf__ int32_t *sorted_metadata, uint64_t scratch, int64_t rows, int64_t hidden,
      uint32_t top_k, uint32_t source_tokens, int64_t rank, int64_t source_pe, int64_t local_experts,
      __gm__ int64_t *target_offsets, uint64_t *statistics,
      __gm__ int32_t *event_counters, uint32_t event_base) {
    constexpr int64_t CHUNK = 3072;
    // Match the indexed bridge chunk while honoring the SDK 16 KiB UB lease.
    constexpr uint64_t TOKEN_METADATA_BYTES = 16 * 4 * sizeof(int32_t);
    auto *token_metadata = reinterpret_cast<__ubuf__ int32_t *>(scratch);
    constexpr uint64_t CREDIT_BYTES = Publish ? 64 * sizeof(int32_t) : 0;
    auto *credits = reinterpret_cast<__ubuf__ int32_t *>(scratch + TOKEN_METADATA_BYTES);
    auto *values = reinterpret_cast<__ubuf__ T *>(scratch + TOKEN_METADATA_BYTES + CREDIT_BYTES);
    if constexpr (Publish) {
      if (local_experts > 64) { AscendC::Trap(); }
      for (int64_t expert = 0; expert < local_experts; ++expert) { credits[expert] = 0; }
    }
    if (top_k == 0 || top_k > 16 || scratch + TOKEN_METADATA_BYTES + CREDIT_BYTES + CHUNK * sizeof(T) > 192 * 1024) {
      AscendC::Trap();
    }
    for (int64_t row = 0; row < rows; ++row) {
      if constexpr (Publish) {
        // Scalar owner flags arrive with the sorted metadata block. Nonowners
        // never fetch the token's K-record fanout metadata.
        if (sorted_metadata[row * 4 + 1] == 0) { continue; }
      }
      int64_t slot = sorted_metadata[row * 4];
      int64_t token = slot / top_k;
      if (slot < 0 || token >= source_tokens) { AscendC::Trap(); }
      aclshmemi_copy_gm2ub(token_metadata, remote_metadata + token * top_k * 4,
                         top_k * 4 * sizeof(int32_t));
      if constexpr (Count) { statistics[2] += top_k * 4 * sizeof(int32_t); }
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
      int64_t owner = -1;
      for (uint32_t selected = 0; selected < top_k; ++selected) {
        int64_t expert = token_metadata[selected * 4 + 2];
        if (expert / local_experts == rank) { owner = selected; break; }
      }
      if (owner < 0) { AscendC::Trap(); }
      // Exactly the first selected expert on this destination pulls the token.
      // The prepass kernel boundary makes every fanout row ready before GMM.
      if (slot % top_k != owner) { continue; }
      if constexpr (Count) { ++statistics[3]; }
      for (int64_t col = 0; col < hidden; col += CHUNK) {
        int64_t elements = hidden - col < CHUNK ? hidden - col : CHUNK;
        uint64_t bytes = elements * sizeof(T);
        aclshmemi_copy_gm2ub(values, source + token * hidden + col, bytes);
        if constexpr (Count) { statistics[0] += bytes; }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        for (uint32_t selected = 0; selected < top_k; ++selected) {
          int64_t expert = token_metadata[selected * 4 + 2];
          if (expert / local_experts != rank) { continue; }
          int64_t ordinal = token_metadata[selected * 4 + 3];
          if (ordinal < 0 || ordinal >= source_tokens) { AscendC::Trap(); }
          int64_t offset = target_offsets[source_pe * local_experts + expert % local_experts];
          aclshmemi_copy_ub2gm(destination + offset + ordinal * hidden + col, values, bytes);
          if constexpr (Count) { statistics[1] += bytes; }
        }
        // Every local destination has consumed this UB row before its reuse.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
      }
      if constexpr (Publish) {
        for (uint32_t selected = 0; selected < top_k; ++selected) {
          int64_t expert = token_metadata[selected * 4 + 2];
          if (expert / local_experts == rank) { ++credits[expert % local_experts]; }
        }
      }
    }
    if constexpr (Publish) {
      // One credit per complete original route row, after every H chunk and
      // destination store drains. No TPipe may overwrite live route metadata.
      PipeBarrier<PIPE_ALL>();
      for (int64_t expert = 0; expert < local_experts; ++expert) {
        int32_t delta = credits[expert];
        if (delta != 0) { (void)AscendC::AtomicAdd<int32_t>(event_counters + event_base + expert, delta); }
      }
      PipeBarrier<PIPE_ALL>();
    }
  }

  __aicore__ inline void StoreIngressStatistics(const TaskDesc &task, GM_ADDR metadata,
                                                int64_t tokens, uint64_t *statistics) {
    if (task.inputs[1].dim[2] != 1) { return; }
    GlobalTensor<uint64_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(
      metadata + tokens * task.inputs[1].dim[3] * 4 * sizeof(int32_t)) + task.task_index * 16, 16);
    uint32_t columns = task.task_type == TaskType::TASK_MOE_TOKEN_GET && task.extra_value_1 == 6 ? 6 : 4;
    for (uint32_t column = 0; column < columns; ++column) { output.SetValue(column, statistics[column]); }
    DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(output);
    PipeBarrier<PIPE_ALL>();
  }

  template <bool Count>
  __aicore__ inline void AddInputTileInterval(__gm__ int32_t *table, uint64_t stride, int64_t expert,
                                             int64_t first, int64_t rows, int64_t expert_rows, uint64_t *statistics) {
    if (first < 0 || rows <= 0 || first + rows > expert_rows) { AscendC::Trap(); }
    constexpr int64_t TILE_ROWS = MulticoreBf16::INPUT_TILE_ROWS;
    while (rows > 0) {
      uint64_t tile = first / TILE_ROWS;
      if (tile >= stride) { AscendC::Trap(); }
      int64_t delta = TILE_ROWS - first % TILE_ROWS;
      if (delta > rows) { delta = rows; }
      (void)AscendC::AtomicAdd<int32_t>(table + expert * stride + tile, static_cast<int32_t>(delta));
      if constexpr (Count) { ++statistics[4]; }
      first += delta;
      rows -= delta;
    }
  }

  template <bool Count>
  __aicore__ inline void PublishInputTiles(
      __gm__ int32_t *table, uint64_t stride, __gm__ int64_t *groups, __gm__ int64_t *target_offsets,
      __ubuf__ int32_t *metadata, __ubuf__ int32_t *counts, __ubuf__ int32_t *firsts,
      __ubuf__ int32_t *contiguous, int64_t rows, uint32_t top_k, int64_t hidden,
      int64_t source_pe, int64_t rank, int64_t local_experts, uint64_t *statistics) {
    for (int64_t expert = 0; expert < local_experts; ++expert) {
      if (counts[expert] == 0) { continue; }
      int64_t start = expert == 0 ? 0 : groups[expert - 1];
      int64_t expert_rows = groups[expert] - start;
      int64_t destination = target_offsets[source_pe * local_experts + expert] / hidden - start;
      if (contiguous[expert] != 0) {
        AddInputTileInterval<Count>(
            table, stride, expert, destination + firsts[expert], counts[expert], expert_rows, statistics);
        continue;
      }
      // A native scalar sort need not preserve original-slot order. The live
      // block supplies an exact fallback without inventing credit for a gap.
      if constexpr (Count) { ++statistics[5]; }
      for (int64_t row = 0; row < rows; ++row) {
        for (uint32_t selected = 0; selected < top_k; ++selected) {
          int64_t slot = (row * top_k + selected) * 2;
          if (metadata[slot] == rank * local_experts + expert) {
            AddInputTileInterval<Count>(
                table, stride, expert, destination + metadata[slot + 1], 1, expert_rows, statistics);
          }
        }
      }
    }
  }

  struct TokenPullGeometry {
    uint64_t metadata_bytes;
    int64_t chunk_elements;
  };

  template <typename T>
  __aicore__ inline TokenPullGeometry GetTokenPullGeometry(int64_t hidden) {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510
    constexpr int64_t FULL_ROW_ELEMENTS = 5120;
    // A smaller metadata window fits one complete row in the same 16 KiB
    // lease, reducing payload DMA calls without changing row publication.
    if (sizeof(T) == sizeof(uint16_t) && hidden == FULL_ROW_ELEMENTS) {
      return {4096, FULL_ROW_ELEMENTS};
    }
#else
    (void)hidden;
#endif
    return {8192, 3072};
  }

  template <typename T, bool Count = false, bool InputTiles = false>
  __aicore__ inline void PullTokenBlocks(
      const TaskDesc &task, __gm__ T *source, __gm__ T *destination, __gm__ int32_t *remote_metadata,
      __gm__ int64_t *target_offsets, __gm__ int32_t *event_counters, uint64_t scratch,
      int64_t token_start, int64_t token_rows, int64_t hidden, uint32_t top_k,
      int64_t rank, int64_t source_pe, int64_t local_experts, int64_t source_tokens, uint64_t *statistics,
      __gm__ int64_t *expert_groups = nullptr, __gm__ int32_t *input_tiles = nullptr, uint64_t tile_stride = 0) {
    const auto geometry = GetTokenPullGeometry<T>(hidden);
    const uint64_t META_BYTES = geometry.metadata_bytes;
    constexpr uint64_t CREDIT_BYTES = 64 * sizeof(int32_t);
    const int64_t CHUNK = geometry.chunk_elements;
    auto *metadata = reinterpret_cast<__ubuf__ int32_t *>(scratch);
    auto *credits = reinterpret_cast<__ubuf__ int32_t *>(scratch + META_BYTES);
    auto *firsts = credits + 64;
    auto *contiguous = firsts + 64;
    auto *values = reinterpret_cast<__ubuf__ T *>(scratch + META_BYTES + (InputTiles ? 3 : 1) * CREDIT_BYTES);
    for (int64_t expert = 0; expert < local_experts; ++expert) { credits[expert] = 0; }
    int64_t metadata_tokens = META_BYTES / (top_k * 2 * sizeof(int32_t));
    for (int64_t begin = 0; begin < token_rows; begin += metadata_tokens) {
      if constexpr (InputTiles) {
        for (int64_t expert = 0; expert < local_experts; ++expert) {
          credits[expert] = 0;
          firsts[expert] = 0;
          contiguous[expert] = 1;
        }
      }
      int64_t rows = token_rows - begin < metadata_tokens ? token_rows - begin : metadata_tokens;
      uint64_t metadata_bytes = rows * top_k * 2 * sizeof(int32_t);
      aclshmemi_copy_gm2ub(metadata, remote_metadata + (token_start + begin) * top_k * 2, metadata_bytes);
      if constexpr (Count) { statistics[2] += metadata_bytes; }
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
      for (int64_t row = 0; row < rows; ++row) {
        auto *route = metadata + row * top_k * 2;
        bool wanted = false;
        for (uint32_t selected = 0; selected < top_k; ++selected) {
          if (route[selected * 2] / local_experts == rank) { wanted = true; break; }
        }
        if (!wanted) { continue; }
        if constexpr (Count) { ++statistics[3]; }
        int64_t token = token_start + begin + row;
        for (int64_t col = 0; col < hidden; col += CHUNK) {
          int64_t elements = hidden - col < CHUNK ? hidden - col : CHUNK;
          uint64_t bytes = elements * sizeof(T);
          aclshmemi_copy_gm2ub(values, source + token * hidden + col, bytes);
          if constexpr (Count) { statistics[0] += bytes; }
          AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
          for (uint32_t selected = 0; selected < top_k; ++selected) {
            int64_t expert = route[selected * 2];
            if (expert / local_experts != rank) { continue; }
            int64_t ordinal = route[selected * 2 + 1];
            if (ordinal < 0 || ordinal >= source_tokens) { AscendC::Trap(); }
            int64_t offset = target_offsets[source_pe * local_experts + expert % local_experts];
            aclshmemi_copy_ub2gm(destination + offset + ordinal * hidden + col, values, bytes);
            if constexpr (Count) { statistics[1] += bytes; }
          }
          AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        for (uint32_t selected = 0; selected < top_k; ++selected) {
          int64_t expert = route[selected * 2];
          if (expert / local_experts == rank) {
            int64_t local = expert % local_experts;
            if constexpr (InputTiles) {
              int32_t ordinal = route[selected * 2 + 1];
              if (credits[local] == 0) {
                firsts[local] = ordinal;
              } else if (ordinal != firsts[local] + credits[local]) {
                contiguous[local] = 0;
              }
            }
            ++credits[local];
          }
        }
      }
      if constexpr (InputTiles) {
        PipeBarrier<PIPE_ALL>();
        PublishInputTiles<Count>(input_tiles, tile_stride, expert_groups, target_offsets, metadata, credits,
                                 firsts, contiguous, rows, top_k, hidden, source_pe, rank, local_experts, statistics);
        PipeBarrier<PIPE_ALL>();
      }
    }
    // Keep deltas live across all metadata blocks. Publish each complete row
    // once, after the entire original-token job has drained every fanout store.
    if constexpr (!InputTiles) {
      PipeBarrier<PIPE_ALL>();
      for (int64_t expert = 0; expert < local_experts; ++expert) {
        int32_t delta = credits[expert];
        if (delta != 0) {
          (void)AscendC::AtomicAdd<int32_t>(event_counters + task.trigger_event + expert, delta);
        }
      }
      PipeBarrier<PIPE_ALL>();
    }
  }

  template <typename T>
  __aicore__ inline void ExecuteTokenPull(const TaskDesc &task, GM_ADDR *input_list, uint32_t tiling_index) {
#ifdef __DAV_CUBE__
    AscendC::Trap();
#else
    namespace plane = hyper_multicore::shmem::data_plane;
    int64_t ep = getExtraValueFromTiling(input_list[tiling_index], 1);
    int64_t experts = getExtraValueFromTiling(input_list[tiling_index], 2);
    int64_t hidden = getExtraValueFromTiling(input_list[tiling_index], 3);
    int64_t tokens = getExtraValueFromTiling(input_list[tiling_index], 4);
    int64_t rank = getExtraValueFromTiling(input_list[tiling_index], 0) % ep;
    uint32_t top_k = task.inputs[1].dim[3];
    int64_t blocks = (tokens + task.task_split_value - 1) / task.task_split_value;
    int64_t source_pe = task.task_index / blocks;
    int64_t token_start = (task.task_index % blocks) * task.task_split_value;
    int64_t token_rows = tokens - token_start < task.task_split_value ? tokens - token_start : task.task_split_value;
    uint64_t scratch;
    uint64_t capacity;
    bool input_tiles = task.extra_value_1 == 6;
    uint64_t required_bytes = 8192 + (input_tiles ? 3 : 1) * 64 * sizeof(int32_t) + 3072 * sizeof(T);
    if (top_k == 0 || top_k > 16 || source_pe >= ep || experts / ep > 64 ||
        (task.extra_value_1 != 5 && !input_tiles) ||
        !hyper_multicore::shmem::cann::device::query_mte_resources(source_pe, scratch, capacity) ||
        capacity < required_bytes || scratch + required_bytes > 192 * 1024) {
      AscendC::Trap();
    }
    auto *source = reinterpret_cast<__gm__ T *>(plane::remote_ptr(input_list[2], source_pe));
    auto *metadata = reinterpret_cast<__gm__ int32_t *>(plane::remote_ptr(input_list[16], source_pe));
    if (source == nullptr || metadata == nullptr) { AscendC::Trap(); }
    auto *destination = reinterpret_cast<__gm__ T *>(input_list[0]);
    auto *target_offsets = reinterpret_cast<__gm__ int64_t *>(input_list[1]);
    auto *event_counters = reinterpret_cast<__gm__ int32_t *>(input_list[24]);
    uint64_t statistics[6] = {0, 0, 0, 0, 0, 0};
    if (input_tiles) {
      auto *table = reinterpret_cast<__gm__ int32_t *>(
          input_list[16] + MulticoreBf16::InputTileTableOffset(tokens, top_k, experts));
      auto *groups = reinterpret_cast<__gm__ int64_t *>(input_list[6]);
      uint64_t stride = MulticoreBf16::InputTileStride(tokens, top_k, ep);
      if (task.inputs[1].dim[2] == 1) {
        PullTokenBlocks<T, true, true>(task, source, destination, metadata, target_offsets, event_counters, scratch,
                                      token_start, token_rows, hidden, top_k, rank, source_pe, experts / ep,
                                      tokens, statistics, groups, table, stride);
      } else {
        PullTokenBlocks<T, false, true>(task, source, destination, metadata, target_offsets, event_counters, scratch,
                                       token_start, token_rows, hidden, top_k, rank, source_pe, experts / ep,
                                       tokens, statistics, groups, table, stride);
      }
    } else if (task.inputs[1].dim[2] == 1) {
      PullTokenBlocks<T, true>(task, source, destination, metadata, target_offsets, event_counters, scratch,
                              token_start, token_rows, hidden, top_k, rank, source_pe, experts / ep,
                              tokens, statistics);
    } else {
      PullTokenBlocks<T, false>(task, source, destination, metadata, target_offsets, event_counters, scratch,
                               token_start, token_rows, hidden, top_k, rank, source_pe, experts / ep,
                              tokens, statistics);
    }
    StoreIngressStatistics(task, input_list[16], tokens, statistics);
#endif
  }

  template <typename T>
  __aicore__ inline void ExecuteIndexedPull(const TaskDesc &task, GM_ADDR *input_list, uint32_t tiling_index) {
#ifdef __DAV_CUBE__
    AscendC::Trap();
#else
    if (task.extra_value_1 == 3) { return; }
    int64_t ep = getExtraValueFromTiling(input_list[tiling_index], 1);
    int64_t experts = getExtraValueFromTiling(input_list[tiling_index], 2);
    int64_t hidden = getExtraValueFromTiling(input_list[tiling_index], 3);
    int64_t tokens = getExtraValueFromTiling(input_list[tiling_index], 4);
    int64_t tiles = task.task_split_num / experts;
    int source_pe = task.task_index / ((experts / ep) * tiles);
    int64_t size = reinterpret_cast<__gm__ int32_t *>(input_list[task.inputs[3].input_position])[
      task.inputs[3].base_ptr_offset];
    int64_t start = (task.task_index % tiles) * static_cast<int64_t>(task.task_split_value) * hidden;
    uint64_t statistics[4] = {0, 0, 0, 0};
    if (start >= size) {
      StoreIngressStatistics(task, input_list[task.extra_value_0], tokens, statistics);
      return;
    }
    int64_t tile_elements = static_cast<int64_t>(task.task_split_value) * hidden;
    int64_t elements = size - start < tile_elements ? size - start : tile_elements;
    int64_t destination_offset = reinterpret_cast<__gm__ int64_t *>(input_list[task.inputs[0].input_position])[
      task.inputs[0].base_ptr_offset] + start;
    int64_t source_offset = reinterpret_cast<__gm__ int64_t *>(input_list[task.inputs[2].input_position])[
      task.inputs[2].base_ptr_offset] + start;
    namespace plane = hyper_multicore::shmem::data_plane;
    constexpr int64_t META_ROWS = 128;
    // Each task drains all transfers before another task reuses this worker's UB.
    uint64_t scratch_ub;
    uint64_t transport_bytes;
    if (!hyper_multicore::shmem::cann::device::query_mte_resources(
          source_pe, scratch_ub, transport_bytes) || transport_bytes < 128 * 4 * sizeof(int32_t) + 2 * 3072 * sizeof(T)) {
      AscendC::Trap();
    }
    auto *remote_data = reinterpret_cast<__gm__ T *>(plane::remote_ptr(
      input_list[task.inputs[1].input_position], source_pe));
    auto *remote_metadata = reinterpret_cast<__gm__ int32_t *>(plane::remote_ptr(
      input_list[task.extra_value_0], source_pe));
    if (remote_data == nullptr || remote_metadata == nullptr) {
      AscendC::Trap();
    }
    auto *destination = reinterpret_cast<__gm__ T *>(input_list[task.outputs[0].input_position]) + destination_offset;
    int64_t source_row = source_offset / hidden;
    int64_t total_rows = elements / hidden;
    for (int64_t begin = 0; begin < total_rows; begin += META_ROWS) {
      int64_t rows = total_rows - begin < META_ROWS ? total_rows - begin : META_ROWS;
      auto *metadata = reinterpret_cast<__ubuf__ int32_t *>(scratch_ub);
      aclshmemi_copy_gm2ub(metadata, remote_metadata + (source_row + begin) * 4,
                           rows * 4 * sizeof(int32_t));
      if (task.inputs[1].dim[2] == 1) { statistics[2] += rows * 4 * sizeof(int32_t); }
      // No TPipe is alive in the transport task. Own a fixed event rather
      // than querying FetchEventID through the previous task's pipe pointer.
      AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
      if (task.extra_value_1 == 2 || task.extra_value_1 == 4) {
        int64_t rank = getExtraValueFromTiling(input_list[tiling_index], 0) % ep;
        uint32_t current_expert = (task.task_index / tiles) % (experts / ep);
        uint32_t event_base = task.trigger_event - current_expert;
        auto *event_counters = reinterpret_cast<__gm__ int32_t *>(input_list[24]);
        if (task.extra_value_1 == 4) {
          if (task.inputs[1].dim[2] == 1) {
            PullPeerRows<T, true, true>(remote_data,
            reinterpret_cast<__gm__ T *>(input_list[task.outputs[0].input_position]), remote_metadata, metadata,
            scratch_ub + META_ROWS * 4 * sizeof(int32_t), rows, hidden, task.inputs[1].dim[3], tokens,
            rank, source_pe, experts / ep,
            reinterpret_cast<__gm__ int64_t *>(input_list[task.inputs[0].input_position]), statistics,
            event_counters, event_base);
          } else {
            PullPeerRows<T, false, true>(remote_data,
            reinterpret_cast<__gm__ T *>(input_list[task.outputs[0].input_position]), remote_metadata, metadata,
            scratch_ub + META_ROWS * 4 * sizeof(int32_t), rows, hidden, task.inputs[1].dim[3], tokens,
            rank, source_pe, experts / ep,
            reinterpret_cast<__gm__ int64_t *>(input_list[task.inputs[0].input_position]), statistics,
            event_counters, event_base);
          }
        } else {
          if (task.inputs[1].dim[2] == 1) {
            PullPeerRows<T, true, false>(remote_data,
            reinterpret_cast<__gm__ T *>(input_list[task.outputs[0].input_position]), remote_metadata, metadata,
            scratch_ub + META_ROWS * 4 * sizeof(int32_t), rows, hidden, task.inputs[1].dim[3], tokens,
            rank, source_pe, experts / ep,
            reinterpret_cast<__gm__ int64_t *>(input_list[task.inputs[0].input_position]), statistics,
            event_counters, event_base);
          } else {
            PullPeerRows<T, false, false>(remote_data,
            reinterpret_cast<__gm__ T *>(input_list[task.outputs[0].input_position]), remote_metadata, metadata,
            scratch_ub + META_ROWS * 4 * sizeof(int32_t), rows, hidden, task.inputs[1].dim[3], tokens,
            rank, source_pe, experts / ep,
            reinterpret_cast<__gm__ int64_t *>(input_list[task.inputs[0].input_position]), statistics,
            event_counters, event_base);
          }
        }
      } else {
        if (task.inputs[1].dim[2] == 1) {
          PullOriginalRows<T, true>(remote_data, destination + begin * hidden, metadata,
                           scratch_ub + META_ROWS * 4 * sizeof(int32_t), rows, hidden,
                           task.inputs[1].dim[3], tokens, statistics);
          statistics[3] += rows;
        } else {
          PullOriginalRows<T, false>(remote_data, destination + begin * hidden, metadata,
                           scratch_ub + META_ROWS * 4 * sizeof(int32_t), rows, hidden,
                           task.inputs[1].dim[3], tokens, statistics);
        }
      }
    }
    StoreIngressStatistics(task, input_list[task.extra_value_0], tokens, statistics);
#endif
  }

}  // namespace MulticoreRuntime
#endif
