/* Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * Licensed under the Apache License, Version 2.0. */
#ifndef HMC_MOE_TRANSPORT_TASK_H_
#define HMC_MOE_TRANSPORT_TASK_H_
#include "data_plane/rma.h"
#include "runtime/runtime_config.hpp"
#include "common/mega_moe/task_options.h"
#include "common/mega_moe/probability_workspace.h"
#include "common/mega_moe/a3_indexed_transfer.h"
#include "common/mega_moe/a3_indexed_push.h"
#include "common/mega_moe/a3_recompute_call.h"
#include "common/mega_moe/a3_recompute_gradient_tiles.h"

namespace MulticoreMoE {
namespace plane = hyper_multicore::shmem::data_plane;

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510 && !defined(__DAV_CUBE__)
template <typename T>
__aicore__ inline void PutIndexedA5(GM_ADDR target, GM_ADDR src, const __gm__ int32_t *indices,
    int64_t destination, int64_t first, int64_t rows, int64_t indexed_rows, int64_t hidden, int32_t peer) {
  if (hidden != 4096 && hidden != 5120) AscendC::Trap();
  hyper_multicore::PutIndexedRows<T>(target, destination, src, indices,
      first, rows, indexed_rows, hidden, peer, nullptr, false);
}
#endif

__aicore__ inline void TransferProbability(const MulticoreRuntime::TaskDesc &task, GM_ADDR *inputs,
                                          uint64_t first, int64_t count, int64_t source, int64_t destination,
                                          int64_t hidden, int32_t peer, plane::TransferEngine engine, bool pull) {
  using MulticoreRuntime::MegaMoeTransportPayload;
  using MulticoreRuntime::ProbabilityLayout;
  using MulticoreRuntime::ProbabilityPlane;
  const auto payload = MulticoreRuntime::GetMegaMoeTransportPayload(task);
  if (payload != MegaMoeTransportPayload::TOKENS) {
    if (first % hidden || count % hidden || source % hidden || destination % hidden) AscendC::Trap();
    const auto &aux = task.outputs[task.num_outputs - 1];
    const bool gradient = payload == MegaMoeTransportPayload::TOKENS_WITH_PROBABILITY_GRADIENTS;
    GM_ADDR buffer = inputs[aux.input_position];
    auto scalar_dst = ProbabilityPlane(buffer, aux.dim[1], gradient ? ProbabilityLayout::RETURNED_GRADIENT
                                                                  : ProbabilityLayout::RECEIVED) +
                      (destination + first) / hidden * sizeof(float);
    auto scalar_src = ProbabilityPlane(buffer, aux.dim[1], gradient ? ProbabilityLayout::RECEIVED_GRADIENT
                                                                  : ProbabilityLayout::SOURCE) +
                      (source + first) / hidden * sizeof(float);
    if (pull) plane::get_complete<float>(scalar_dst, scalar_src, count / hidden, peer, engine);
    else plane::put_complete<float>(scalar_dst, scalar_src, count / hidden, peer, engine);
  }
}

/** Translate expert routing descriptors; transport mechanisms remain in SHMEM. */
template <typename T>
__aicore__ inline void ExecuteTransfer(const MulticoreRuntime::TaskDesc &task, GM_ADDR *inputs,
                                      GM_ADDR metadata, GM_ADDR events, int64_t indexed_rows = 0,
                                      const __gm__ A3RecomputeCall *recompute = nullptr,
                                      bool dispatch_prepared = false) {
  using MulticoreRuntime::GetShmemTransferEngine;
  using MulticoreRuntime::TaskType;
  auto topology = reinterpret_cast<__gm__ int64_t *>(metadata);
  const int64_t ep = topology[1], experts = topology[2], hidden = topology[3];
  if (ep <= 0 || experts <= 0 || experts % ep || hidden <= 0 || task.task_split_num % experts) AscendC::Trap();
  const int64_t fragments = task.task_split_num / experts;
  if (fragments <= 0) AscendC::Trap();
  const int32_t peer = task.task_index / ((experts / ep) * fragments);
  if (peer >= ep) AscendC::Trap();
  const auto engine = static_cast<plane::TransferEngine>(static_cast<uint32_t>(GetShmemTransferEngine(task)) >> 4);
  const bool pull = task.task_type == TaskType::TASK_SHMEM_GET_MEM;
  auto event = reinterpret_cast<__gm__ int32_t *>(events) + task.trigger_event;
  if (dispatch_prepared && !pull && task.inputs[1].input_position == 2) {
    plane::signal(event, 1, plane::SignalOp::Add, peer);
    return;
  }
  auto sizes = reinterpret_cast<__gm__ int32_t *>(inputs[task.inputs[3].input_position]);
  const int64_t total = sizes[task.inputs[3].base_ptr_offset];
  if (total < 0) AscendC::Trap();
  const uint64_t extent = static_cast<uint64_t>(task.task_split_value) * hidden;
  if (extent == 0) AscendC::Trap();
  const uint64_t tile = task.task_index % fragments;
  const uint64_t valid_tiles = total / extent + (total % extent != 0);
  if (tile < valid_tiles) {
    const uint64_t first = tile * extent;
    const int64_t count = total - first < extent ? total - first : extent;
    auto destination_offsets = reinterpret_cast<__gm__ int64_t *>(inputs[task.inputs[0].input_position]);
    auto source_offsets = reinterpret_cast<__gm__ int64_t *>(inputs[task.inputs[2].input_position]);
    const int64_t destination = destination_offsets[task.inputs[0].base_ptr_offset];
    const int64_t source = source_offsets[task.inputs[2].base_ptr_offset];
    const uint64_t limit = ~uint64_t{0} / sizeof(T);
    if (first > limit || source < 0 || destination < 0 || static_cast<uint64_t>(source) > limit - first ||
        static_cast<uint64_t>(destination) > limit - first) AscendC::Trap();
    auto dst = inputs[task.outputs[0].input_position] + (destination + first) * sizeof(T);
    auto src = inputs[task.inputs[1].input_position] + (source + first) * sizeof(T);
    if (indexed_rows > 0 && task.inputs[1].input_position == 2 && !pull) {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510 && !defined(__DAV_CUBE__)
      if ((source + first) % hidden || count % hidden
          || (source + first + count) / hidden > indexed_rows * A5_INDEXED_TOP_K) AscendC::Trap();
      const auto indices = reinterpret_cast<const __gm__ int32_t *>(source_offsets + experts);
      const int64_t row = (source + first) / hidden;
      const int64_t rows = count / hidden;
      if (recompute != nullptr) {
        PutIndexedA5<T>(reinterpret_cast<GM_ADDR>(recompute->received_g),
            reinterpret_cast<GM_ADDR>(recompute->source_g), indices,
            destination + first, row, rows, indexed_rows, hidden, peer);
        constexpr int64_t packet = kProbabilityPacketElements * sizeof(float);
        plane::put_complete<float>(reinterpret_cast<GM_ADDR>(recompute->received_p) +
            (destination + first) / hidden * packet, reinterpret_cast<GM_ADDR>(recompute->source_p) + row * packet,
            rows * kProbabilityPacketElements, peer, engine);
      }
      PutIndexedA5<T>(inputs[task.outputs[0].input_position], inputs[2], indices,
          destination + first, row, rows, indexed_rows, hidden, peer);
#else
      AscendC::Trap();
#endif
    } else if (recompute != nullptr && task.inputs[1].input_position == 11 && !pull) {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510 && !defined(__DAV_CUBE__)
      if ((source + first) % hidden || (destination + first) % hidden || count % hidden) AscendC::Trap();
      const int64_t row = (source + first) / hidden;
      const int64_t rows = count / hidden;
      auto g = reinterpret_cast<GM_ADDR>(recompute->received_g);
      auto p = reinterpret_cast<GM_ADDR>(recompute->received_p);
      auto dp = reinterpret_cast<GM_ADDR>(recompute->local_dp);
      constexpr int64_t packet = kProbabilityPacketElements * sizeof(float);
      if (hidden != 4096 && hidden != 5120) AscendC::Trap();
      RecomputedProbabilityGradientRows(g, inputs[11], dp, hidden, row, row + rows, 1);
      ScaleRecomputedGradientRows(g, p, g, hidden, row, row + rows, 1, kProbabilityPacketElements);
      plane::put_complete<float>(reinterpret_cast<GM_ADDR>(recompute->returned_dp) +
          (destination + first) / hidden * packet, dp + row * packet,
          rows * kProbabilityPacketElements, peer, engine);
#else
      AscendC::Trap();
#endif
    } else if (pull) {
      plane::get_complete<T>(dst, src, count, peer, engine);
    } else {
      plane::put_complete<T>(dst, src, count, peer, engine, MulticoreRuntime::GetShmemUdmaQp(task));
    }
    // Finish every payload before the original data credit. The RMA provider
    // supplies complete semantics even when token and scalar engines differ.
    TransferProbability(task, inputs, first, count, source, destination, hidden, peer, engine, pull);
  }
  // Empty Push fragments still satisfy the original multi-source ADD contract.
  // Pull completion is published locally by the runtime after this function returns.
  if (!pull) plane::signal(event, 1, plane::SignalOp::Add, peer);
}
}  // namespace MulticoreMoE

#endif  // HMC_MOE_TRANSPORT_TASK_H_
