// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#ifndef PANGU_MX_TASK_GRAPH_WORKER_H
#define PANGU_MX_TASK_GRAPH_WORKER_H
#include "kernel_operator.h"
#include "mxfp8/mx_launch_descriptor.h"
#include "mxfp8/mx_byte_tensor_list.h"
#include "mxfp8/mx_source_row_layout.h"
#include "runtime/worker_kernel.h"
#ifdef __DAV_CUBE__
#include "mxfp8/mx_gmm_worker.h"
#else
#include "mxfp8_quant/mx_quant_worker.h"
#define DTYPE_DISPATCH_TARGET bfloat16_t
#include "put_mem_signal/put_mem_signal_kernel.cpp"
#include "mxfp8/mx_source_row_ingress.h"
#endif

using namespace AscendC;
namespace MulticoreRuntime {
template <VectorWorkerPolicy Policy>
class MxTaskWorker : public KernelWorkerBase<MxTaskWorker<Policy>, Policy, RuntimeStorageBoundsPolicy::NATIVE_TILING> {
 public:
  static constexpr uint32_t EVENT_IDX = 65;
  static constexpr uint32_t DIAGNOSTIC_IDX = 66;
  static constexpr uint32_t TILING_IDX = 67;
  __aicore__ inline void ExecuteComputeKernel(TaskDesc task) {
    auto launch = reinterpret_cast<__gm__ MulticoreMx::MxLaunchDescriptor *>(this->input_list[TILING_IDX]);
    if (task.num_inputs > 4 || task.num_outputs > 4) Trap();
    for (uint32_t i = 0; i < task.num_inputs; ++i) {
      if (task.inputs[i].input_position >= launch->buffer_count) Trap();
    }
    for (uint32_t i = 0; i < task.num_outputs; ++i) {
      if (task.outputs[i].input_position >= launch->buffer_count) Trap();
    }
#ifdef __DAV_CUBE__
    switch (static_cast<uint32_t>(task.task_type)) {
      case 0:
      case 10:
        break;
      case 160:
        MulticoreMx::ExecuteMxGmmTask(task, this->input_list, launch,
                                     this->replica_matrix_bases_[task.outputs[0].input_position == 6 ? 3 : 2],
                                     this->home_experts_, this->replica_trace_, this->replica_trace_task_);
        break;
      default:
        Trap();
    }
#else
    ExecuteVectorTask(task);
#endif
  }

 private:
#ifndef __DAV_CUBE__
  __aicore__ inline void ExecuteVectorTask(TaskDesc task) {
    switch (static_cast<uint32_t>(task.task_type)) {
      case 0:
      case 10:
        break;
      case MulticoreMx::MX_SOURCE_ROW_READY_TASK:
        if (reinterpret_cast<__gm__ MulticoreMx::MxLaunchDescriptor *>(this->input_list[TILING_IDX])->version != 14)
          Trap();
        break;
#ifdef PANGU_MX_ROW_ONLY
      case 164:
      case 165:
#endif
      case 161:
      case 162:
      case 163:
      case 153:
        MulticoreMx::ExecuteMxQuantTask(task, this->input_list);
        break;
      case 105:
        ExecuteShmemPutMem(task);
        break;
      case 141:
        ExecuteShmemGetMem<bfloat16_t>(task);
        break;
      default:
        Trap();
    }
  }

  template <typename T>
  __aicore__ inline void ExecuteShmemGetMem(const TaskDesc &task) {
    GM_ADDR tiling = this->input_list[TILING_IDX];
    const int64_t ep = getExtraValueFromTiling(tiling, 1);
    const int64_t experts = getExtraValueFromTiling(tiling, 2);
    const int64_t hidden = getExtraValueFromTiling(tiling, 3);
    const int64_t tiles = task.task_split_num / experts;
    const int64_t source_pe = task.task_index / ((experts / ep) * tiles);
    const int64_t size = reinterpret_cast<__gm__ int32_t *>(
        this->input_list[task.inputs[3].input_position])[task.inputs[3].base_ptr_offset];
    const int64_t start = (task.task_index % tiles) * static_cast<int64_t>(task.task_split_value) * hidden;
    if (start >= size) return;
    const int64_t tile_elements = static_cast<int64_t>(task.task_split_value) * hidden;
    const int64_t elements = size - start < tile_elements ? size - start : tile_elements;
    const int64_t destination = reinterpret_cast<__gm__ int64_t *>(
        this->input_list[task.inputs[0].input_position])[task.inputs[0].base_ptr_offset];
    const int64_t source = reinterpret_cast<__gm__ int64_t *>(
        this->input_list[task.inputs[2].input_position])[task.inputs[2].base_ptr_offset];
    PullToLocal<T>(this->input_list[task.outputs[0].input_position] + (destination + start) * sizeof(T),
                   this->input_list[task.inputs[1].input_position] + (source + start) * sizeof(T),
                   elements, source_pe);
  }

  __aicore__ inline void ExecuteShmemPutMem(TaskDesc task_desc) {
    auto launch = reinterpret_cast<__gm__ MulticoreMx::MxLaunchDescriptor *>(this->input_list[TILING_IDX]);
    if (launch->version == 14 && task_desc.inputs[1].input_position == 2) {
      MulticoreMx::ExecuteSourceRowPut(task_desc, this->input_list, launch);
      return;
    }
    GM_ADDR signal = this->input_list[EVENT_IDX];
    GM_ADDR transport_probe = TransportProbeCounters();

    int64_t ep = getExtraValueFromTiling(this->input_list[TILING_IDX], 1);
    int64_t expert_num = getExtraValueFromTiling(this->input_list[TILING_IDX], 2);
    int64_t expert_num_single_rank = expert_num / ep;
    int64_t hidden_size = getExtraValueFromTiling(this->input_list[TILING_IDX], 3);

    int64_t rank_id = (*(__gm__ int64_t *)(this->input_list[TILING_IDX])) / ep * ep;
    int64_t single_expert_task_num = task_desc.task_split_num / expert_num;
    int64_t target_pe = rank_id + task_desc.task_index / (expert_num_single_rank * single_expert_task_num);

    int64_t target_offset_ = *((__gm__ int64_t *)(this->input_list[task_desc.inputs[0].input_position] +
                                                  task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type));
    int64_t src_offset_ = *((__gm__ int64_t *)(this->input_list[task_desc.inputs[2].input_position] +
                                               task_desc.inputs[2].base_ptr_offset * task_desc.inputs[2].data_type));
    int32_t size_ = *((__gm__ int32_t *)(this->input_list[task_desc.inputs[3].input_position] +
                                         task_desc.inputs[3].base_ptr_offset * task_desc.inputs[3].data_type));
    int32_t send_data_size_ = task_desc.task_split_value * hidden_size;

    int32_t value = task_desc.task_index % single_expert_task_num;
    int32_t start = value * task_desc.task_split_value * hidden_size;
    int32_t end = (value + 1) * task_desc.task_split_value * hidden_size;
    if (start >= size_) {
      aclshmemx_signal_op(reinterpret_cast<__gm__ int32_t *>(signal) + task_desc.trigger_event, 1, ACLSHMEM_SIGNAL_ADD,
                          target_pe);
      hyper_multicore::RecordTokenTransport(
        transport_probe, hyper_multicore::TokenTransportProbeSlot::EMPTY);
      return;
    } else {
      if (end > size_) {
        send_data_size_ = size_ - start;
      }
    }
    target_offset_ = target_offset_ + static_cast<int64_t>(start);
    src_offset_ = src_offset_ + static_cast<int64_t>(start);
    send_data_size_ = static_cast<int64_t>(send_data_size_);

    constexpr uint32_t kPeerQp0Marker = 0x4D585130U;
    if (task_desc.extra_value_0 == kPeerQp0Marker) {
      const bool peer_protocol = launch->version >= 15 && launch->version <= 18 &&
                                 launch->reserved + 1 == launch->version;
      if (!peer_protocol || this->replica_udma_legacy_qp_bank_) Trap();
      if (TryPeerQp0Put(task_desc, target_pe, target_offset_, src_offset_, send_data_size_, transport_probe)) return;
    }

    constexpr uint32_t kMixedCombineMarker = 0x4D584D00U;
    const uint32_t transport = static_cast<uint32_t>(task_desc.extra_value_0);
    const bool mixed_combine = (transport & 0xFFFFFFFCU) == kMixedCombineMarker;
    if (mixed_combine && ((transport & 3U) == 0 || task_desc.inputs[1].input_position == 2 ||
                         (AscendC::GetBlockIdx() & 1U) == 0)) Trap();
    // Keep quiet and credit publication per task while testing concurrent data engines.
    const bool prefer_mapped_mte = !mixed_combine || (AscendC::GetBlockIdx() / 2) % 4 < (transport & 3U);
    put_mem_signal_kernel(this->input_list[task_desc.outputs[0].input_position] +
                            task_desc.outputs[0].base_ptr_offset * task_desc.outputs[0].data_type,
                          target_offset_,
                          this->input_list[task_desc.inputs[1].input_position] +
                            task_desc.inputs[1].base_ptr_offset * task_desc.inputs[1].data_type,
                          src_offset_, send_data_size_, signal, static_cast<int64_t>(task_desc.trigger_event), 1,
                          nullptr, 1, target_pe, false, this->replica_udma_legacy_qp_bank_ ? 16 : 0,
                          prefer_mapped_mte, transport_probe, MteScratchBytes(transport));
  }

  __aicore__ inline bool TryPeerQp0Put(const TaskDesc &task, int32_t peer, int64_t destination_offset,
                                       int64_t source_offset, int64_t elements, GM_ADDR probe) {
    auto state = aclshmemi_get_state();
    if (peer == aclshmem_my_pe()) return false;
    const uint32_t ep = getExtraValueFromTiling(this->input_list[TILING_IDX], 1);
    const uint32_t rank = aclshmem_my_pe() % ep;
    const uint32_t remote = peer % ep;
    const uint32_t owner = 2 * (((remote + ep - rank) % ep) % 32) + 1;
    if (this->worker_id_ != owner) Trap();
    if (!(state->topo_list[peer] & ACLSHMEM_TRANSPORT_UDMA)) return false;
    const uint32_t qps = aclshmemi_udma_qp_info_fetch()->qp_num;
    if (qps != 1 && qps != 8 && qps != 16 && qps != 32) Trap();
    auto destination = reinterpret_cast<__gm__ uint8_t *>(this->input_list[task.outputs[0].input_position] +
        task.outputs[0].base_ptr_offset * task.outputs[0].data_type + destination_offset * sizeof(bfloat16_t));
    auto source = reinterpret_cast<__gm__ uint8_t *>(this->input_list[task.inputs[1].input_position] +
        task.inputs[1].base_ptr_offset * task.inputs[1].data_type + source_offset * sizeof(bfloat16_t));
    TPipe pipe;
    TBuf<TPosition::VECOUT> staging;
    pipe.InitBuffer(staging, ACLSHMEM_UDMA_MTE_STAGING_UB_SIZE);
    auto scratch = reinterpret_cast<__ubuf__ uint8_t *>(staging.Get<uint8_t>().GetPhyAddr());
    uint64_t bytes = static_cast<uint64_t>(elements) * sizeof(bfloat16_t);
    while (bytes) {
      const uint32_t count = bytes < (1ULL << 28) ? bytes : (1ULL << 28);
      aclshmemx_udma_qp_put_nbi<uint8_t>(destination, source, scratch, count, peer, 0, 0);
      aclshmemx_udma_qp_quiet(peer, 0);
      destination += count;
      source += count;
      bytes -= count;
    }
    hyper_multicore::shmem::data_plane::signal(
      reinterpret_cast<__gm__ int32_t *>(this->input_list[EVENT_IDX]) + task.trigger_event,
      1, hyper_multicore::shmem::data_plane::SignalOp::Add, peer);
    hyper_multicore::RecordTokenTransport(probe, hyper_multicore::TokenTransportProbeSlot::REMOTE_UDMA);
    return true;
  }

  __aicore__ static inline uint32_t MteScratchBytes(uint32_t transport) {
    if ((transport & 0xFFFFFFFCU) != 0x4D585500U) return 0;
    const uint32_t scale = transport & 3U;
    if ((scale != 1 && scale != 2) || (AscendC::GetBlockIdx() & 1U) == 0) AscendC::Trap();
    return 16384U << scale;
  }

  __aicore__ inline GM_ADDR TransportProbeCounters() {
    static_assert(CYCLE_TRACE_RESERVED_WORDS == 9, "MX transport probe requires the existing eight padding words");
    if (getDiagnosticMode(this->runtimeConfigPtr) != DiagnosticMode::PROFILE) return nullptr;
    const uint64_t aic_stride = sizeof(CycleTraceCoreHeader) +
                               uint64_t(getAicProfileRecordCapacity(this->runtimeConfigPtr)) * sizeof(CycleTraceRecord);
    const uint64_t aiv_stride = sizeof(CycleTraceCoreHeader) +
                               uint64_t(getAivProfileRecordCapacity(this->runtimeConfigPtr)) * sizeof(CycleTraceRecord);
    auto header = reinterpret_cast<__gm__ CycleTraceCoreHeader *>(
      this->input_list[DIAGNOSTIC_IDX] + getCubeWorkerCapacity(this->runtimeConfigPtr) * aic_stride +
      this->worker_id_ * aiv_stride);
    if (header->coreType != 2 || header->blockId != this->worker_id_) AscendC::Trap();
    // The existing decoder requires reserved[0] to remain zero; the following eight words are ignored padding.
    auto probe = header->reserved + 1;
    if (probe[0] != hyper_multicore::TOKEN_TRANSPORT_PROBE_MAGIC) {
      auto state = aclshmemi_get_state();
      const uint32_t world = static_cast<uint32_t>(aclshmem_n_pes());
      const uint32_t rank = static_cast<uint32_t>(aclshmem_my_pe());
      if (world == 0 || world > 0xFF || rank >= world) AscendC::Trap();
      bool all_remote_udma = false;
      probe[7] = TransportProbeTopology(world, rank, all_remote_udma);
      probe[5] = (uint32_t(all_remote_udma) << 31) | (world << 16) | rank;
      probe[6] = state->mte_config.ub_size;
      probe[0] = hyper_multicore::TOKEN_TRANSPORT_PROBE_MAGIC;
    }
    return reinterpret_cast<GM_ADDR>(probe);
  }

  __aicore__ static inline uint32_t TransportProbeTopology(uint32_t world, uint32_t rank, bool &all_remote_udma) {
    auto state = aclshmemi_get_state();
    uint32_t capabilities = 0, mappings = 0;
    all_remote_udma = state->topo_list != nullptr;
    for (uint32_t peer = 0; peer < world; ++peer) {
      if (peer != rank &&
          (state->topo_list == nullptr || (state->topo_list[peer] & ACLSHMEM_TRANSPORT_UDMA) == 0))
        all_remote_udma = false;
      if (peer < 16 && state->topo_list != nullptr && (state->topo_list[peer] & ACLSHMEM_TRANSPORT_MTE) != 0)
        capabilities |= uint32_t(1) << peer;
      if (peer < 16 && state->p2p_device_heap_base != nullptr && state->p2p_device_heap_base[peer] != nullptr)
        mappings |= uint32_t(1) << peer;
    }
    return (mappings << 16) | capabilities;
  }
#endif
};
}  // namespace MulticoreRuntime
#include "mx_role_worker.h"
#include "mx_planned_worker.h"
__aicore__ inline void LoadMxGraphInputs(GM_ADDR buffers, const __gm__ MulticoreMx::MxLaunchDescriptor *launch,
                                         GM_ADDR *inputs) {
  const bool source_row = launch->version == 14 && launch->reserved == 13;
  if (source_row && !MulticoreMx::IsSourceRowLaunchValid(launch)) Trap();
  auto list_words = reinterpret_cast<__gm__ uint64_t *>(buffers);
  const uint64_t pointer_offset = list_words[0];
  if (!MulticoreMx::IsMxByteListHeaderValid(pointer_offset, launch->buffer_count)) Trap();
  auto data_pointers = list_words + pointer_offset / sizeof(uint64_t);
  for (uint32_t i = 0; i < launch->buffer_count; ++i) {
    const uint64_t physical_bytes = MulticoreMx::SourceRowPhysicalBufferBytes(launch, i);
    if (!MulticoreMx::IsMxByteVectorShapeValid(list_words[1 + 2 * i], list_words[2 + 2 * i], physical_bytes))
      Trap();
    inputs[i] = reinterpret_cast<GM_ADDR>(data_pointers[i]);
  }
  if (source_row) {
    if (!MulticoreMx::AreSourceRowBaseAliasesValid(launch, reinterpret_cast<uint64_t>(inputs[0]),
          reinterpret_cast<uint64_t>(inputs[16]), reinterpret_cast<uint64_t>(inputs[17]),
          reinterpret_cast<uint64_t>(inputs[18]))) Trap();
    inputs[17] += launch->buffer_bytes[16];
  }
}

__aicore__ inline void ValidateMxWindowRuntime(GM_ADDR runtime, const __gm__ MulticoreMx::MxLaunchDescriptor *launch) {
#if __NPU_ARCH__ != 3510
  Trap();
#endif
  MulticoreRuntime::ReadyHandshakeMeta meta;
  MulticoreRuntime::getReadyHandshakeMeta(runtime, &meta);
  if (meta.completion_event != 0 || launch->cube_count != 32 || MulticoreRuntime::getNumWorkers(runtime) != 64 ||
      *reinterpret_cast<__gm__ uint32_t *>(runtime + offsetof(MulticoreRuntime::RuntimeHeader, protocol_version)) != 0 ||
      *reinterpret_cast<__gm__ int32_t *>(runtime + MulticoreRuntime::getTaskIndexNumOffset(runtime) +
                                          3 * sizeof(int32_t)) != -1)
    Trap();
}

__aicore__ inline bool HasMxPrefetch(const __gm__ MulticoreMx::MxLaunchDescriptor *launch) {
  return (launch->version == 12 && launch->reserved == 11) ||
         (launch->version == 13 && launch->reserved == 12) ||
         (launch->version == 17 && launch->reserved == 16) ||
         (launch->version == 18 && launch->reserved == 17);
}

__aicore__ inline bool HasMxGradientReturn(const __gm__ MulticoreMx::MxLaunchDescriptor *launch) {
  return (launch->version == 11 && launch->reserved == 10) ||
         (launch->version == 13 && launch->reserved == 12) ||
         (launch->version == 16 && launch->reserved == 15) ||
         (launch->version == 18 && launch->reserved == 17);
}

template <MulticoreRuntime::VectorWorkerPolicy Policy>
__aicore__ inline void RunMxTaskGraph(GM_ADDR buffers, GM_ADDR runtime, GM_ADDR events, GM_ADDR diagnostic,
                                      GM_ADDR launch_metadata) {
  auto launch = reinterpret_cast<__gm__ MulticoreMx::MxLaunchDescriptor *>(launch_metadata);
  static_assert(Policy == MulticoreRuntime::VectorWorkerPolicy::LEGACY_ODD,
                "The integrated MX payload supports legacy and window2 queues only");
  const bool prefetch = HasMxPrefetch(launch);
  const bool gradients = HasMxGradientReturn(launch);
  const bool source_row = launch->version == 14 && launch->reserved == 13;
  const bool peer_qp0 = launch->version >= 15 && launch->version <= 18 &&
                        launch->reserved + 1 == launch->version;
  const bool window = (launch->version == 6 && launch->reserved == 5) || gradients || prefetch || source_row || peer_qp0;
  const bool legacy = launch->version == 1 && launch->reserved == 0;
  if (launch->magic != MulticoreMx::MX_LAUNCH_MAGIC || (!window && !legacy) || launch->buffer_count == 0 ||
      launch->buffer_count > 64 || !MulticoreMx::IsMxLaunchTopologyValid(launch->ep, launch->experts))
    Trap();
  GM_ADDR inputs[68] = {};
  LoadMxGraphInputs(buffers, launch, inputs);
  inputs[64] = runtime;
  inputs[65] = events;
  inputs[66] = diagnostic;
  inputs[67] = launch_metadata;
  if (window) {
    ValidateMxWindowRuntime(runtime, launch);
    MulticoreRuntime::MxPlannedWorker worker;
    worker.Init(GetBlockIdx(), runtime, inputs);
    worker.ProcessPlanned(gradients);
    return;
  }
  MulticoreRuntime::MxTaskWorker<Policy> worker;
  worker.Init(GetBlockIdx(), runtime, inputs);
  worker.Process();
}
#endif
