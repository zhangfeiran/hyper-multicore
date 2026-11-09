/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file worker_kernel.cpp
 * @brief MoE-FFN backward — op-specific compute kernels + KernelWorker specialization.
 */

#include <cstddef>

#include "kernel_operator.h"
#include "swi_glu/swi_glu.cpp"
#include "swi_glu_grad/swi_glu_grad.cpp"
#include "grouped_matmul/grouped_matmul.cpp"
#include "put_mem_signal/put_mem_signal_kernel.cpp"
#include "runtime/empty_dispatch.h"
#include "runtime/worker_kernel.h"
#include "runtime/gradient_order_policy.h"

using namespace AscendC;

namespace MulticoreRuntime {

class KernelWorker : public KernelWorkerBase<KernelWorker> {
 public:
  // input_list layout for hyper_mega_moe_grad (backward):
  //   [30] = tiling params  [31] = all_event_counters  [32] = diagnostic_buffer
  //   [25] = gmm_workspace  [26] = swi_glu_grad_workspace
  static constexpr uint32_t TILING_IDX              = 30;
  static constexpr uint32_t EVENT_IDX               = 31;
  static constexpr uint32_t DIAGNOSTIC_IDX          = 32;
  static constexpr uint32_t WORKSPACE_IDX           = 25;
  static constexpr uint32_t SWIGLU_GRAD_WORKSPACE_IDX = 26;
  static constexpr uint32_t SWIGLU_DYNAMIC_FIELDS_OFFSET =
    static_cast<uint32_t>(offsetof(SwiGluTilingData, rowLen));
  static constexpr int64_t SWIGLU_DYNAMIC_FIELDS_BYTES =
    offsetof(SwiGluTilingData, baseRowLen) + sizeof(uint32_t) - SWIGLU_DYNAMIC_FIELDS_OFFSET;

  static constexpr uint32_t GMM_BASE_M_OFFSET = static_cast<uint32_t>(
    offsetof(GMMTilingData, gmmBaseParams) + offsetof(GMMBaseParams, m));
  static constexpr uint32_t GMM_MATMUL_M_OFFSET = static_cast<uint32_t>(
    offsetof(GMMTilingData, mmTilingData) + offsetof(TCubeTiling, M));
  static constexpr int64_t GMM_MATMUL_M_FIELDS_BYTES =
    offsetof(TCubeTiling, singleCoreM) + sizeof(int32_t) - offsetof(TCubeTiling, M);
  static constexpr uint32_t GMM_MATMUL_K_OFFSET = static_cast<uint32_t>(
    offsetof(GMMTilingData, mmTilingData) + offsetof(TCubeTiling, Ka));
  static constexpr int64_t GMM_MATMUL_K_FIELDS_BYTES =
    offsetof(TCubeTiling, singleCoreK) + sizeof(int32_t) - offsetof(TCubeTiling, Ka);

  __aicore__ inline void ResolveTaskDescriptor(TaskId &task_id, TaskDesc &task) {
#ifdef __DAV_C220_CUBE__
    if ((task.extra_value_0 & GRADIENT_ORDER_MAGIC_MASK) != GRADIENT_ORDER_MAGIC) return;
    if (task.extra_value_0 == GRADIENT_LANE_MARKER || task.extra_value_0 == GRADIENT_ACTIVE_LANE_MARKER) return;
    const uint32_t policy = (task.extra_value_0 >> 20) & 0xF;
    const uint32_t join_event = task.extra_value_0 & GRADIENT_ORDER_EVENT_MASK;
    const bool original_w1 = task.outputs[0].input_position == 18;
    const uint32_t experts = task.inputs[2].dim[0];
    ValidateGradientTask(task, experts, policy);
    PrepareGradientOrder(task, experts, policy);
    if (SkipGradientShape(policy, original_w1)) return;
    const uint32_t original_expert = task.task_index / this->core_num;
    const uint32_t lane = task.task_index % this->core_num;
    const uint32_t own_base = task_id - task.task_index;
    uint32_t selected = task_id;
    if (policy == 4) {
      if (!gradient_join_) return;
      const uint32_t gate_base = original_w1 ? task.tiling_data_offset : own_base;
      const uint32_t w1_base = original_w1 ? own_base : task.tiling_data_offset;
      const uint32_t ordinal = original_expert + (original_w1 ? experts : 0);
      selected = (ordinal % 2 ? w1_base : gate_base) + (ordinal / 2) * this->core_num + lane;
    } else {
      selected = own_base + gradient_order_[original_expert] * this->core_num + lane;
    }
    if (selected >= this->runtime_task_capacity) AscendC::Trap();
    getTaskDesc(this->runtimeConfigPtr, &task, selected);
    task_id = selected;
    ApplyGradientJoin(task, policy, join_event, original_expert);
#endif
  }

  __aicore__ inline void ExecuteComputeKernel(TaskDesc task_desc) {
    switch (task_desc.task_type) {
      case TaskType::TASK_BEGIN_TASK_GRAPH:
        break;
      case TaskType::TASK_MATMUL:
        ExecuteMatmul(task_desc);
        break;
      case TaskType::TASK_GROUPED_MATMUL:
        ExecuteGroupedMatmul(task_desc);
        break;
      case TaskType::TASK_SHMEM_PUT_MEM_SIGNAL:
        ExecuteShmemPutMem(task_desc);
        break;
      case TaskType::TASK_SHMEM_GET_MEM:
        ExecuteShmemGetMem<DTYPE_DISPATCH_TARGET>(task_desc);
        break;
      case TaskType::TASK_SWI_GLU_GRAD:
        ExecuteSwiGluGrad(task_desc);
        break;
      default:
        break;
    }
  }

 private:
  __aicore__ inline bool SkipGradientShape(uint32_t policy, bool original_w1) const {
    return policy == GRADIENT_SHAPE_POLICY && (!gradient_join_ || !original_w1);
  }

  __aicore__ inline void ApplyGradientJoin(TaskDesc &task, uint32_t policy, uint32_t join_event,
                                          uint32_t original_expert) {
#ifdef __DAV_C220_CUBE__
    if (policy == 3 || policy >= 6) {
      if (join_event >= this->runtime_event_capacity) AscendC::Trap();
      task.dependent_event = (policy == 3 || original_expert == 0) ? join_event : EVENT_INVALID_ID;
    }
#endif
  }

  bool gradient_order_ready_ = false;
  bool gradient_join_ = false;
  uint32_t active_w1_ = 0;
  uint32_t active_w2_ = 0;
  int64_t gradient_rows_[GRADIENT_ORDER_MAX_EXPERTS];
  uint32_t gradient_order_[GRADIENT_ORDER_MAX_EXPERTS];

  __aicore__ inline void ValidateGradientTask(const TaskDesc &task, uint32_t experts, uint32_t policy) {
    if (this->home_experts_ != 0 || task.task_type != TaskType::TASK_GROUPED_MATMUL ||
        experts == 0 || experts > GRADIENT_ORDER_MAX_EXPERTS || policy < 1 || policy == 5 ||
        policy > GRADIENT_SHAPE_POLICY) AscendC::Trap();
    const uint32_t position = task.outputs[0].input_position;
    if (position != 18 && !(position == 12 && policy == 4) &&
        !(position == 6 && policy == GRADIENT_SHAPE_POLICY)) AscendC::Trap();
  }

  __aicore__ inline uint32_t GradientLaneExpert(const TaskDesc &task, uint32_t expert, bool publishes_bands) {
    const uint32_t mode = task.extra_value_0 & (GRADIENT_ORDER_MAGIC_MASK | 0x00F00000U);
    if (mode != GRADIENT_LANE_MARKER && mode != GRADIENT_ACTIVE_LANE_MARKER &&
        mode != (GRADIENT_ORDER_MAGIC | (7U << 20)) &&
        !(mode == GRADIENT_SHAPE_MARKER && gradient_join_)) {
      return GRADIENT_LANE_DISABLED;
    }
    const uint32_t position = task.outputs[0].input_position;
    if (!getTransposeData(task.inputs[0].transpose_flag) ||
        (position != 6 && position != 18)) AscendC::Trap();
    // Called only for nonempty experts. All workers visit the same phase
    // sequence, so the first real GEMM keeps the original lane assignment.
    if (mode == GRADIENT_ACTIVE_LANE_MARKER) {
      // Keep the tile-to-producer mapping of the real communication bands.
      if (publishes_bands) return GRADIENT_LANE_DISABLED;
      const uint32_t ordinal = position == 6 ? active_w2_++ : active_w1_++;
      return ordinal == 0 ? GRADIENT_LANE_DISABLED : ordinal;
    }
    if (this->home_experts_ != 0) AscendC::Trap();
    return expert;
  }

  __aicore__ inline void PrepareGradientOrder(const TaskDesc &task, uint32_t experts, uint32_t policy) {
    if (gradient_order_ready_) return;
    GlobalTensor<int64_t> glist;
    glist.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(input_list[task.inputs[2].input_position]), experts);
    int64_t previous = 0;
    for (uint32_t i = 0; i < experts; ++i) {
      const int64_t current = glist.GetValue(i);
      if (current < previous) AscendC::Trap();
      gradient_rows_[i] = current - previous;
      previous = current;
    }
    gradient_join_ = BuildGradientExpertOrder(gradient_rows_, experts, policy, gradient_order_);
    gradient_order_ready_ = true;
  }

  __aicore__ inline float GetSwiGluClampLimit(const TaskDesc &task_desc) {
    union {
      uint32_t bits;
      float value;
    } encoded = {};
    encoded.bits = task_desc.extra_value_0;
    return encoded.value;
  }

  __aicore__ inline bool getTransposeData(uint32_t data) { return data == 1; }

  __aicore__ inline void ExecuteMatmul(const TaskDesc &task_desc) {}

  __aicore__ inline void cacheWriteThrough(__gm__ uint8_t *sourceAddr, int64_t length) {
    if (length <= 0) {
      return;
    }
    __gm__ uint8_t *start =
      (__gm__ uint8_t *)((int64_t)sourceAddr / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    __gm__ uint8_t *end =
      (__gm__ uint8_t *)(((int64_t)sourceAddr + length - 1) / AscendC::CACHE_LINE_SIZE * AscendC::CACHE_LINE_SIZE);
    AscendC::GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(start);
    for (uint32_t i = 0; i <= end - start; i += AscendC::CACHE_LINE_SIZE) {
      __asm__ __volatile__("");
      AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::SINGLE_CACHE_LINE,
                                        AscendC::DcciDst::CACHELINE_OUT>(global[i]);
      __asm__ __volatile__("");
    }
  }

  __aicore__ inline void ExecuteSwiGluGrad(const TaskDesc &task_desc) {
    if (task_desc.inputs[0].dynamic_shape == 1 || task_desc.outputs[0].dynamic_shape == 1) {
      DynamicData dynamic_data;
      getDynamicData(this->runtimeConfigPtr, &(dynamic_data));
      if (dynamic_data.dynamic_type == DynamicType::DYNAMIC_DSV3_MOE) {
        GM_ADDR grouped_list = input_list[dynamic_data.dynamic_input_position];
        GlobalTensor<int64_t> grouped_list_tensor;
        int64_t ep = getExtraValueFromTiling(input_list[TILING_IDX], 1);
        int64_t expert_num = getExtraValueFromTiling(input_list[TILING_IDX], 2);
        int64_t expert_num_single_rank = expert_num / ep;
        grouped_list_tensor.SetGlobalBuffer((__gm__ int64_t *)(grouped_list), expert_num_single_rank);

        uint32_t grouped_list_shape = dynamic_data.dynamic_group_size;
        uint32_t per_task_num = task_desc.task_split_num / grouped_list_shape;
        uint32_t data_index = task_desc.task_index / per_task_num;
        uint32_t task_index = task_desc.task_index % per_task_num;
        int64_t start = 0;
        int64_t end = grouped_list_tensor.GetValue(data_index);
        if (data_index != 0) {
          start = grouped_list_tensor.GetValue(data_index - 1);
        }
        int64_t current_seq_start = task_index * task_desc.task_split_value;
        int64_t current_seq_end = (task_index + 1) * task_desc.task_split_value;
        int64_t base_ptr_offset = start + current_seq_start;

        if (base_ptr_offset >= end) {
          return;
        }
        int64_t input_0_offset = task_desc.inputs[0].dynamic_shape == 1
                                   ? base_ptr_offset * task_desc.inputs[0].dim[1] * task_desc.inputs[0].data_type
                                   : task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type;
        int64_t input_1_offset = task_desc.inputs[1].dynamic_shape == 1
                                   ? base_ptr_offset * task_desc.inputs[1].dim[1] * task_desc.inputs[1].data_type
                                   : task_desc.inputs[1].base_ptr_offset * task_desc.inputs[1].data_type;
        int64_t output_0_offset = task_desc.outputs[0].dynamic_shape == 1
                                    ? base_ptr_offset * task_desc.outputs[0].dim[1] * task_desc.outputs[0].data_type
                                    : task_desc.outputs[0].base_ptr_offset * task_desc.outputs[0].data_type;
        if (start + current_seq_end <= end) {
          swi_glu_grad(input_list[task_desc.inputs[0].input_position] + input_0_offset,
                       input_list[task_desc.inputs[1].input_position] + input_1_offset,
                       input_list[task_desc.outputs[0].input_position] + output_0_offset,
                       input_list[SWIGLU_GRAD_WORKSPACE_IDX],
                       input_list[task_desc.tiling_data_position] + task_desc.tiling_data_offset,
                       GetSwiGluClampLimit(task_desc));
          return;
        } else {
          GM_ADDR tiling_data_addr = input_list[task_desc.tiling_data_position] + 80 * (AscendC::GetBlockIdx() + 1);
          __gm__ SwiGluTilingData *tilingdata_data = reinterpret_cast<__gm__ SwiGluTilingData *>(tiling_data_addr);
          tilingdata_data->rowLen = end - (start + current_seq_start);
          if (tilingdata_data->rowLen == 0) {
            return;
          }
          tilingdata_data->baseRowLen = tilingdata_data->rowLen < SWIGLU_DYNAMIC_TAIL_BASE_ROW_LIMIT
                                           ? tilingdata_data->rowLen : SWIGLU_DYNAMIC_TAIL_BASE_ROW_LIMIT;
          cacheWriteThrough(tiling_data_addr + SWIGLU_DYNAMIC_FIELDS_OFFSET, SWIGLU_DYNAMIC_FIELDS_BYTES);
          PipeBarrier<PIPE_ALL>();

          swi_glu_grad(input_list[task_desc.inputs[0].input_position] + input_0_offset,
                       input_list[task_desc.inputs[1].input_position] + input_1_offset,
                       input_list[task_desc.outputs[0].input_position] + output_0_offset,
                       input_list[SWIGLU_GRAD_WORKSPACE_IDX],
                       tiling_data_addr, GetSwiGluClampLimit(task_desc));
        }
      }
      return;
    }
  }

  __aicore__ inline void PrepareOwnerGroupedTiling(const TaskDesc &task_desc, GM_ADDR grouped_list_real,
      uint32_t expert_num_single_rank, uint32_t data_index, int64_t value, GM_ADDR tiling_data_addr) {
    GlobalTensor<int64_t> grouped_list_tensor_real;
    grouped_list_tensor_real.SetGlobalBuffer((__gm__ int64_t *)(grouped_list_real), expert_num_single_rank);

    for (uint32_t i = 0; i < expert_num_single_rank; i++) {
      if (i > data_index) {
        grouped_list_tensor_real.SetValue(i, value);
      } else if (i == data_index) {
        grouped_list_tensor_real.SetValue(i, value);
      } else {
        grouped_list_tensor_real.SetValue(i, 0);
      }
    }
    cacheWriteThrough(grouped_list_real, expert_num_single_rank * sizeof(int64_t));

    __gm__ GMMTilingData *tilingdata_data = reinterpret_cast<__gm__ GMMTilingData *>(tiling_data_addr);
    if (getTransposeData(task_desc.inputs[0].transpose_flag)) {
      tilingdata_data->mmTilingData.Ka = value;
      tilingdata_data->mmTilingData.Kb = value;
      tilingdata_data->mmTilingData.singleCoreK = value;
      cacheWriteThrough(tiling_data_addr + GMM_MATMUL_K_OFFSET, GMM_MATMUL_K_FIELDS_BYTES);
    } else {
      tilingdata_data->gmmBaseParams.m = value;
      tilingdata_data->mmTilingData.M = value;
      tilingdata_data->mmTilingData.singleCoreM = value;
      cacheWriteThrough(tiling_data_addr + GMM_BASE_M_OFFSET, sizeof(uint32_t));
      cacheWriteThrough(tiling_data_addr + GMM_MATMUL_M_OFFSET, GMM_MATMUL_M_FIELDS_BYTES);
    }
    PipeBarrier<PIPE_ALL>();
  }

  __aicore__ inline bool UsesOwnerEoiGradient(const TaskDesc &task) const {
    // EOI writes directly into parameter layout; preserve its validated group-list path.
    const uint32_t position = task.outputs[0].input_position;
    return this->home_experts_ == 0 &&
           ((position == 18 && task.inputs[0].input_position == 10) ||
            (position == 6 && task.inputs[0].input_position == 0));
  }

  __aicore__ inline void ResolveReplicaMatrices(const TaskDesc &task_desc, uint32_t data_index,
      GM_ADDR &weight_base, GM_ADDR &output_base, int64_t &output_0_offset) {
    if (this->home_experts_ > 0) {
      if (getTransposeData(task_desc.inputs[0].transpose_flag)) {
        const int64_t matrix_bytes = static_cast<int64_t>(task_desc.inputs[0].dim[1]) *
                                     task_desc.inputs[1].dim[1] * task_desc.outputs[0].data_type;
        const uint32_t position = task_desc.outputs[0].input_position;
        output_base = this->GetExpertMatrix(position, data_index, matrix_bytes, position == 18 ? 2 : 3);
        output_0_offset = 0;
      } else {
        const int64_t matrix_bytes = static_cast<int64_t>(task_desc.inputs[1].dim[1]) *
                                     task_desc.inputs[1].dim[2] * task_desc.inputs[1].data_type;
        const uint32_t position = task_desc.inputs[1].input_position;
        weight_base = this->GetExpertMatrix(position, data_index, matrix_bytes, position == 11 ? 0 : 1);
      }
    }
  }

  struct GroupedCall {
    GM_ADDR groups;
    int64_t rows;
    bool single_group;
    uint32_t expert;
    uint32_t lane;
  };

  __aicore__ inline GroupedCall PrepareGroupedCall(const TaskDesc &task, uint32_t experts,
      uint32_t expert, int64_t rows, GM_ADDR tiling, bool bands) {
    if (UsesOwnerEoiGradient(task)) {
      GM_ADDR groups = this->runtimeConfigPtr + this->grouped_matmul_group_list_offset_;
      PrepareOwnerGroupedTiling(task, groups, experts, expert, rows, tiling);
      return {groups, 0, false, UINT32_MAX, GRADIENT_LANE_DISABLED};
    }
    const bool physical = this->home_experts_ > 0;
    return {nullptr, rows, physical, physical ? 0 : expert, GradientLaneExpert(task, expert, bands)};
  }

  __aicore__ inline void ExecuteGroupedMatmul(const TaskDesc &task_desc) {
    if (task_desc.inputs[0].dynamic_shape == 1 || task_desc.inputs[1].dynamic_shape == 1) {
      DynamicData dynamic_data;
      getDynamicData(this->runtimeConfigPtr, &(dynamic_data));
      if (dynamic_data.dynamic_type == DynamicType::DYNAMIC_DSV3_MOE) {
        GM_ADDR grouped_list = input_list[task_desc.inputs[2].input_position] +
                               task_desc.inputs[2].base_ptr_offset * task_desc.inputs[2].data_type;
        GlobalTensor<int64_t> grouped_list_tensor;
        int64_t ep = getExtraValueFromTiling(input_list[TILING_IDX], 1);
        int64_t expert_num = getExtraValueFromTiling(input_list[TILING_IDX], 2);
        int64_t expert_num_single_rank = expert_num / ep;
        grouped_list_tensor.SetGlobalBuffer((__gm__ int64_t *)(grouped_list), expert_num_single_rank);

        uint32_t data_index = task_desc.task_index / this->core_num;
        int64_t value = grouped_list_tensor.GetValue(data_index);
        int64_t start = 0;
        if (data_index != 0) {
          value = value - grouped_list_tensor.GetValue(data_index - 1);
          start = grouped_list_tensor.GetValue(data_index - 1);
        }
        uint32_t band_output_rows = 0, band_rows = 0;
        int32_t band_epoch = 0;
        GM_ADDR band_flags = this->GetGradientBandProduction(
            task_desc.outputs[0].input_position, data_index, &band_output_rows, &band_rows, &band_epoch);
        if (value == 0) {
          if (band_flags != nullptr) {
            MulticoreRuntime::GradientBandProducer producer;
            producer.Set(band_flags, band_epoch, band_rows);
            producer.SetProfile(this->replica_trace_, task_desc.outputs[0].input_position == 18,
                                getExtraValueFromTiling(input_list[TILING_IDX], 0) % ep, data_index);
            producer.Begin(band_output_rows, this->worker_id_, this->core_num);
            producer.PublishEmpty();
          }
          return;
        }
        GM_ADDR tiling_data_addr = input_list[task_desc.tiling_data_position] + 2016 * AscendC::GetBlockIdx();
        const GroupedCall call = PrepareGroupedCall(task_desc, expert_num_single_rank, data_index, value,
                                                     tiling_data_addr, band_flags != nullptr);
        int64_t input_0_offset = task_desc.inputs[0].dynamic_shape == 1
                                   ? start * task_desc.inputs[0].dim[1] * task_desc.inputs[0].data_type
                                   : task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type;
        int64_t input_1_offset = task_desc.inputs[1].dynamic_shape == 1
                                   ? start * task_desc.inputs[1].dim[1] * task_desc.inputs[1].data_type
                                   : task_desc.inputs[1].base_ptr_offset * task_desc.inputs[1].data_type;
        int64_t output_0_offset = task_desc.outputs[0].dynamic_shape == 1
                                    ? start * task_desc.outputs[0].dim[1] * task_desc.outputs[0].data_type
                                    : task_desc.outputs[0].base_ptr_offset * task_desc.outputs[0].data_type;

        GM_ADDR weight_base = input_list[task_desc.inputs[1].input_position];
        GM_ADDR output_base = input_list[task_desc.outputs[0].input_position];
        ResolveReplicaMatrices(task_desc, data_index, weight_base, output_base, output_0_offset);
        grouped_matmul(input_list[task_desc.inputs[0].input_position] + input_0_offset,
                       weight_base + input_1_offset, nullptr, nullptr, nullptr,
                       nullptr, nullptr, call.groups, nullptr,
                       output_base + output_0_offset, input_list[WORKSPACE_IDX],
                       tiling_data_addr, getTransposeData(task_desc.inputs[0].transpose_flag),
                       getTransposeData(task_desc.inputs[1].transpose_flag),
                       task_desc.outputs[0].data_type == sizeof(float), band_flags, band_epoch, band_rows,
                       this->replica_trace_, task_desc.outputs[0].input_position == 18,
                       getExtraValueFromTiling(input_list[TILING_IDX], 0) % ep, data_index,
                       call.rows, call.single_group, call.expert, call.lane);
      }
      return;
    }
  }

  __aicore__ inline void ExecuteShmemPutMem(const TaskDesc &task_desc) {
    GM_ADDR signal = input_list[EVENT_IDX];

    int64_t ep = getExtraValueFromTiling(input_list[TILING_IDX], 1);
    int64_t expert_num = getExtraValueFromTiling(input_list[TILING_IDX], 2);
    int64_t expert_num_single_rank = expert_num / ep;
    int64_t hidden_size = getExtraValueFromTiling(input_list[TILING_IDX], 3);

    int64_t rank_id = (*(__gm__ int64_t *)(input_list[TILING_IDX])) / ep * ep;
    int64_t single_expert_task_num = task_desc.task_split_num / expert_num;
    int64_t target_pe = rank_id + task_desc.task_index / (expert_num_single_rank * single_expert_task_num);

    int64_t target_offset_ = *((__gm__ int64_t *)(input_list[task_desc.inputs[0].input_position] +
                                                  task_desc.inputs[0].base_ptr_offset * task_desc.inputs[0].data_type));
    int64_t src_offset_ = *((__gm__ int64_t *)(input_list[task_desc.inputs[2].input_position] +
                                               task_desc.inputs[2].base_ptr_offset * task_desc.inputs[2].data_type));
    int32_t size_ = *((__gm__ int32_t *)(input_list[task_desc.inputs[3].input_position] +
                                         task_desc.inputs[3].base_ptr_offset * task_desc.inputs[3].data_type));
    int32_t send_data_size_ = task_desc.task_split_value * hidden_size;

    int32_t value = task_desc.task_index % single_expert_task_num;
    int32_t start = value * task_desc.task_split_value * hidden_size;
    int32_t end = (value + 1) * task_desc.task_split_value * hidden_size;
    if (start >= size_) {
      const int32_t credit = EmptyDispatchCredit(task_desc.extra_value_0, value, single_expert_task_num,
                                                 start, size_, send_data_size_);
      if (credit != 0) {
        aclshmemx_signal_op(reinterpret_cast<__gm__ int32_t *>(signal) + task_desc.trigger_event, credit,
                           ACLSHMEM_SIGNAL_ADD, target_pe);
      }
      return;
    } else {
      if (end > size_) {
        send_data_size_ = size_ - start;
      }
    }
    target_offset_ = target_offset_ + static_cast<int64_t>(start);
    src_offset_ = src_offset_ + static_cast<int64_t>(start);
    send_data_size_ = static_cast<int64_t>(send_data_size_);

    put_mem_signal_kernel(input_list[task_desc.outputs[0].input_position] +
                            task_desc.outputs[0].base_ptr_offset * task_desc.outputs[0].data_type,
                          target_offset_,
                          input_list[task_desc.inputs[1].input_position] +
                            task_desc.inputs[1].base_ptr_offset * task_desc.inputs[1].data_type,
                          src_offset_, send_data_size_, signal, static_cast<int64_t>(task_desc.trigger_event), 1,
                          nullptr, 1, target_pe, false);
  }
};

}  // namespace MulticoreRuntime

extern "C" inline __aicore__ void worker_kernel(uint32_t worker_id, __gm__ uint8_t *runtimeConfigPtr,
                                                GM_ADDR *input_list) {
  MulticoreRuntime::KernelWorker worker;
  worker.Init(worker_id, runtimeConfigPtr, input_list);
  worker.Process();
}
