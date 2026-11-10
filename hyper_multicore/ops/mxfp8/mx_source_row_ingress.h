// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef PANGU_MX_SOURCE_ROW_INGRESS_H_
#define PANGU_MX_SOURCE_ROW_INGRESS_H_

#include "mx_source_row_layout.h"

namespace MulticoreMx {
#ifdef __NPU_ARCH__
namespace ingress_plane = hyper_multicore::shmem::data_plane;

struct SourceRowTransfer {
  int64_t peer, source_begin, target_begin, rows, top_k;
};

__aicore__ inline bool IsSourceRowTaskValid(const MulticoreRuntime::TaskDesc &task) {
  return task.num_inputs == 4 && task.num_outputs == 1 && task.inputs[0].input_position == 1 &&
         task.inputs[1].input_position == 2 && task.inputs[2].input_position == 3 &&
         task.inputs[3].input_position == 4 && task.outputs[0].input_position == 0;
}

__aicore__ inline bool AreSourceRowOffsetsValid(int64_t width, int64_t elements, int64_t target, int64_t source) {
  return elements >= 0 && elements % width == 0 && target >= 0 && target % width == 0 &&
         source >= 0 && source % width == 0;
}

__aicore__ inline SourceRowTransfer ReadSourceRowTransfer(const MulticoreRuntime::TaskDesc &task, GM_ADDR *inputs,
                                                          const __gm__ MxLaunchDescriptor *launch) {
  const int64_t width = launch->hidden;
  const int64_t top_k = launch->buffer_bytes[MX_SOURCE_INDEX_POSITION] / sizeof(int32_t) / launch->local_tokens;
  const int64_t local_experts = launch->experts / launch->ep;
  const int64_t fragments = task.task_split_num / launch->experts;
  if (fragments <= 0 || task.task_split_num % launch->experts || task.task_split_value <= 0) AscendC::Trap();
  const int64_t peer = launch->rank / launch->ep * launch->ep + task.task_index / (local_experts * fragments);
  const int64_t target_offset = *reinterpret_cast<__gm__ int64_t *>(
      inputs[1] + task.inputs[0].base_ptr_offset * task.inputs[0].data_type);
  const int64_t source_offset = *reinterpret_cast<__gm__ int64_t *>(
      inputs[3] + task.inputs[2].base_ptr_offset * task.inputs[2].data_type);
  const int64_t elements = *reinterpret_cast<__gm__ int32_t *>(
      inputs[4] + task.inputs[3].base_ptr_offset * task.inputs[3].data_type);
  if (!AreSourceRowOffsetsValid(width, elements, target_offset, source_offset)) AscendC::Trap();
  const int64_t first = (task.task_index % fragments) * task.task_split_value;
  const int64_t available = elements / width;
  if (first >= available) return {peer, 0, 0, 0, top_k};
  const int64_t rows = available - first < task.task_split_value ? available - first : task.task_split_value;
  return {peer, source_offset / width + first, target_offset / width + first, rows, top_k};
}

__aicore__ inline __gm__ uint8_t *MapSourceRowDestination(GM_ADDR *inputs, int64_t peer) {
  auto state = aclshmemi_get_state();
  if (peer != aclshmem_my_pe() && ((state->topo_list[peer] & ACLSHMEM_TRANSPORT_MTE) == 0 ||
      state->p2p_device_heap_base == nullptr ||
      reinterpret_cast<uintptr_t>(state->p2p_device_heap_base[peer]) == 0)) AscendC::Trap();
  auto destination = reinterpret_cast<__gm__ uint8_t *>(ingress_plane::remote_ptr(inputs[0], peer));
  if (destination == nullptr) AscendC::Trap();
  return destination;
}

__aicore__ inline void CopySourceRowBlocks(GM_ADDR *inputs, const __gm__ MxLaunchDescriptor *launch,
                                           const SourceRowTransfer &transfer, __gm__ uint8_t *destination) {
  const int64_t width = launch->hidden;
  const int64_t scale_width = width / 32;
  const int64_t source_slots = launch->local_tokens * transfer.top_k;
  auto state = aclshmemi_get_state();
  const uint64_t scratch = state->mte_config.aclshmem_ub;
  const uint64_t scratch_bytes = state->mte_config.ub_size;
  constexpr uint64_t metadata_bytes = 64;
  if (scratch_bytes < metadata_bytes + width + scale_width) AscendC::Trap();
  uint64_t batch_limit = (scratch_bytes - metadata_bytes) / (width + scale_width);
  if (batch_limit > metadata_bytes / sizeof(int32_t)) batch_limit = metadata_bytes / sizeof(int32_t);
  auto indices = reinterpret_cast<__ubuf__ int32_t *>(scratch);
  auto data = reinterpret_cast<__ubuf__ uint8_t *>(scratch + metadata_bytes);
  auto source_indices = reinterpret_cast<__gm__ int32_t *>(inputs[MX_SOURCE_INDEX_POSITION]);
  for (int64_t first_row = 0; first_row < transfer.rows;) {
    const int64_t remaining = transfer.rows - first_row;
    const int64_t batch = remaining < batch_limit ? remaining : batch_limit;
    auto scales = data + batch * width;
    aclshmemi_copy_gm2ub(indices, source_indices + transfer.source_begin + first_row, batch * sizeof(int32_t));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
    for (int64_t row = 0; row < batch; ++row) {
      const int64_t slot = indices[row];
      if (slot < 0 || slot >= source_slots) AscendC::Trap();
      const int64_t token = slot / transfer.top_k;
      aclshmemi_copy_gm2ub(data + row * width, inputs[2] + token * width, width);
      aclshmemi_copy_gm2ub(scales + row * scale_width,
                          inputs[MX_SOURCE_SCALE_POSITION] + token * scale_width, scale_width);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
    aclshmemi_copy_ub2gm(destination + (transfer.target_begin + first_row) * width, data, batch * width);
    aclshmemi_copy_ub2gm(destination + launch->buffer_bytes[16] + (transfer.target_begin + first_row) * scale_width,
                        scales, batch * scale_width);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    first_row += batch;
  }
}

__aicore__ inline void ExecuteSourceRowPut(const MulticoreRuntime::TaskDesc &task, GM_ADDR *inputs,
                                           const __gm__ MxLaunchDescriptor *launch) {
  if (!IsSourceRowLaunchValid(launch) || !IsSourceRowTaskValid(task)) AscendC::Trap();
  if (inputs[16] != inputs[0] || inputs[17] != inputs[0] + launch->buffer_bytes[16]) AscendC::Trap();
  const SourceRowTransfer transfer = ReadSourceRowTransfer(task, inputs, launch);
  if (transfer.peer < 0 || transfer.peer >= launch->ep) AscendC::Trap();
  const int64_t source_slots = launch->local_tokens * transfer.top_k;
  if (transfer.source_begin + transfer.rows > source_slots ||
      static_cast<uint64_t>(transfer.target_begin + transfer.rows) > launch->buffer_bytes[16] / launch->hidden)
    AscendC::Trap();
  if (transfer.rows != 0) {
    auto destination = MapSourceRowDestination(inputs, transfer.peer);
    CopySourceRowBlocks(inputs, launch, transfer, destination);
  }
  // The one original PUT credit covers both complete planes, including all scale stores.
  aclshmemx_mte_quiet();
  auto signal = reinterpret_cast<__gm__ int32_t *>(inputs[65]) + task.trigger_event;
  ingress_plane::signal(signal, 1, ingress_plane::SignalOp::Add, transfer.peer);
}
#endif
}  // namespace MulticoreMx
#endif
