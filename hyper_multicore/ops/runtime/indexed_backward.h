/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under CANN Open Software License Agreement Version 1.0.
 * See LICENSE in the root of the software repository for the full text.
 */
#ifndef HYPER_MULTICORE_INDEXED_BACKWARD_H_
#define HYPER_MULTICORE_INDEXED_BACKWARD_H_

#include "adv_api/reduce/reduce.h"

namespace MulticoreRuntime::BackwardIndexed {

constexpr uint64_t HEADER_OFFSET = 512;
constexpr uint64_t RECORD_OFFSET = 640;
constexpr uint64_t UB_LIMIT = 192 * 1024;
constexpr int64_t RECORD_ROWS = 128;
constexpr int64_t BATCH_ROWS = 8;

template <typename T>
__aicore__ inline LocalTensor<T> LocalAt(uint64_t address, uint32_t elements) {
  LocalTensor<T> result;
  TBuffAddr buffer{};
  buffer.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
  buffer.bufferAddr = address;
#if ASCENDC_CPU_DEBUG
  buffer.dataLen = elements * sizeof(T);
  buffer.absAddr = reinterpret_cast<uint8_t *>(address);
#endif
  result.SetAddr(buffer);
  result.SetSize(elements);
  return result;
}

__aicore__ inline uint64_t ApplicationScratch(int peer, uint64_t &transport) {
  uint64_t lease;
  if (!hyper_multicore::shmem::cann::device::query_mte_resources(peer, transport, lease) ||
      lease < RECORD_ROWS * 4 * sizeof(int32_t)) {
    AscendC::Trap();
  }
  return (transport + lease + 31) / 32 * 32;
}

__aicore__ inline int64_t TokenIndex(__ubuf__ int32_t *records, int64_t row, int64_t top_k, int64_t tokens) {
  int64_t slot = records[row * 4];
  if (slot < 0 || slot >= tokens * top_k) { AscendC::Trap(); }
  return slot / top_k;
}

template <typename T>
__aicore__ inline void ReadBatch(__gm__ T *source, LocalTensor<T> batch, __ubuf__ int32_t *records,
                                int64_t rows, int64_t hidden, int64_t top_k, int64_t tokens) {
  int64_t first = TokenIndex(records, 0, top_k, tokens);
  int64_t stride = rows > 1 ? TokenIndex(records, 1, top_k, tokens) - first : 1;
  bool uniform = stride > 0 && static_cast<uint64_t>(stride - 1) * hidden * sizeof(T) <= UINT32_MAX;
  for (int64_t row = 1; row < rows; ++row) {
    if (TokenIndex(records, row, top_k, tokens) != first + row * stride) { uniform = false; }
  }
  if (uniform) {
    GlobalTensor<T> remote;
    remote.SetGlobalBuffer(source);
    DataCopyExtParams parameters{static_cast<uint16_t>(rows), static_cast<uint32_t>(hidden * sizeof(T)),
                                static_cast<uint32_t>((stride - 1) * hidden * sizeof(T)), 0, 0};
    DataCopyPadExtParams<T> padding{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(batch, remote[first * hidden], parameters, padding);
  } else {
    for (int64_t row = 0; row < rows; ++row) {
      int64_t token = TokenIndex(records, row, top_k, tokens);
      aclshmemi_copy_gm2ub(reinterpret_cast<__ubuf__ T *>(batch.GetPhyAddr()) + row * hidden,
                          source + token * hidden, hidden * sizeof(T));
    }
  }
}

template <typename T>
__aicore__ inline void ScaleBatch(LocalTensor<T> batch, LocalTensor<float> temporary,
                                 __ubuf__ int32_t *records, int64_t rows, int64_t hidden) {
  for (int64_t row = 0; row < rows; ++row) {
    union { int32_t bits; float value; } probability;
    probability.bits = records[row * 4 + 1];
    Cast(temporary, batch[row * hidden], RoundMode::CAST_NONE, hidden);
    PipeBarrier<PIPE_V>();
    Muls(temporary, temporary, probability.value, hidden);
    PipeBarrier<PIPE_V>();
    Cast(batch[row * hidden], temporary, RoundMode::CAST_RINT, hidden);
    PipeBarrier<PIPE_V>();
  }
}

template <typename T>
__aicore__ inline void PullRows(__gm__ T *source, __gm__ T *destination, __ubuf__ int32_t *records,
                               uint64_t scratch, int64_t rows, int64_t hidden, int64_t top_k, int64_t tokens) {
  uint64_t values = BATCH_ROWS * hidden * sizeof(T);
  if (scratch + values + hidden * sizeof(float) > UB_LIMIT) { AscendC::Trap(); }
  auto batch = LocalAt<T>(scratch, BATCH_ROWS * hidden);
  auto temporary = LocalAt<float>(scratch + values, hidden);
  for (int64_t begin = 0; begin < rows; begin += BATCH_ROWS) {
    int64_t count = rows - begin < BATCH_ROWS ? rows - begin : BATCH_ROWS;
    ReadBatch(source, batch, records + begin * 4, count, hidden, top_k, tokens);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    ScaleBatch(batch, temporary, records + begin * 4, count, hidden);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    aclshmemi_copy_ub2gm(destination + begin * hidden, reinterpret_cast<__ubuf__ T *>(scratch),
                        count * hidden * sizeof(T));
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
  }
}

template <typename T>
__aicore__ inline void ExecuteGet(const TaskDesc &task, GM_ADDR *inputs, uint32_t tiling_position) {
#ifndef __DAV_CUBE__
  int64_t ep = getExtraValueFromTiling(inputs[tiling_position], 1);
  int64_t experts = getExtraValueFromTiling(inputs[tiling_position], 2);
  int64_t hidden = getExtraValueFromTiling(inputs[tiling_position], 3);
  int64_t tokens = getExtraValueFromTiling(inputs[tiling_position], 4);
  int64_t tiles = task.task_split_num / experts;
  int peer = task.task_index / ((experts / ep) * tiles);
  int64_t size = reinterpret_cast<__gm__ int32_t *>(inputs[4])[task.inputs[3].base_ptr_offset];
  int64_t start = task.task_index % tiles * static_cast<int64_t>(task.task_split_value) * hidden;
  if (start >= size) { return; }
  int64_t elements = size - start;
  if (elements > task.task_split_value * hidden) { elements = task.task_split_value * hidden; }
  int64_t source_row = (reinterpret_cast<__gm__ int64_t *>(inputs[3])[task.inputs[2].base_ptr_offset] + start) / hidden;
  int64_t target = reinterpret_cast<__gm__ int64_t *>(inputs[1])[task.inputs[0].base_ptr_offset] + start;
  uint64_t transport;
  uint64_t scratch = ApplicationScratch(peer, transport);
  namespace plane = hyper_multicore::shmem::data_plane;
  auto *source = reinterpret_cast<__gm__ T *>(plane::remote_ptr(inputs[2], peer));
  auto *metadata_address = reinterpret_cast<__gm__ uint8_t *>(plane::remote_ptr(inputs[25], peer));
  if (source == nullptr || metadata_address == nullptr) { AscendC::Trap(); }
  auto *remote = reinterpret_cast<__gm__ int32_t *>(metadata_address + RECORD_OFFSET);
  auto *destination = reinterpret_cast<__gm__ T *>(inputs[0]) + target;
  int64_t rows = elements / hidden;
  for (int64_t begin = 0; begin < rows; begin += RECORD_ROWS) {
    int64_t count = rows - begin < RECORD_ROWS ? rows - begin : RECORD_ROWS;
    auto *records = reinterpret_cast<__ubuf__ int32_t *>(transport);
    aclshmemi_copy_gm2ub(records, remote + (source_row + begin) * 4, count * 4 * sizeof(int32_t));
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    PullRows(source, destination + begin * hidden, records, scratch, count, hidden, task.inputs[1].dim[3], tokens);
  }
#endif
}

template <typename T>
class ProbabilityGradient {
 public:
  __aicore__ inline ProbabilityGradient(GM_ADDR *inputs, const TaskDesc &task) : inputs_(inputs), task_(task) {}

  __aicore__ inline void Run() {
    auto *header = reinterpret_cast<__gm__ uint64_t *>(inputs_[25] + HEADER_OFFSET);
    if (header[0] == 0) { return; }
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(header[2]));
    expert_ = reinterpret_cast<__gm__ T *>(header[1]);
    hidden_ = task_.inputs[0].dim[1];
    top_k_ = task_.outputs[0].dim[1];
    int64_t tokens = task_.inputs[0].dim[0];
    int64_t first = task_.task_index * static_cast<int64_t>(task_.task_split_value);
    int64_t rows = tokens - first < task_.task_split_value ? tokens - first : task_.task_split_value;
    if (first >= tokens || rows <= 0) { return; }
    Init(rows);
    auto *remote = reinterpret_cast<__gm__ int32_t *>(inputs_[25] + RECORD_OFFSET);
    aclshmemi_copy_gm2ub(metadata_, remote + first * top_k_ * 4, rows * top_k_ * 4 * sizeof(int32_t));
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    for (int64_t row = 0; row < rows; ++row) { RunToken(first + row, row); }
    PipeBarrier<PIPE_ALL>();
    DataCopyExtParams parameters{1, static_cast<uint32_t>(rows * top_k_ * sizeof(float)), 0, 0, 0};
    DataCopyPad(output_[first * top_k_], result_, parameters);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    PipeBarrier<PIPE_ALL>();
  }

 private:
  __aicore__ inline void Init(int64_t rows) {
    uint64_t transport;
    uint64_t scratch = ApplicationScratch(0, transport);
    uint64_t value_bytes = hidden_ * sizeof(T);
    uint64_t float_bytes = hidden_ * sizeof(float);
    uint64_t metadata_bytes = rows * top_k_ * 4 * sizeof(int32_t);
    if (scratch + 2 * value_bytes + 3 * float_bytes + metadata_bytes + rows * top_k_ * sizeof(float) + 32 > UB_LIMIT) {
      AscendC::Trap();
    }
    value_ = LocalAt<T>(scratch, hidden_);
    dy_ = LocalAt<T>(scratch + value_bytes, hidden_);
    values_float_ = LocalAt<float>(scratch + 2 * value_bytes, hidden_);
    dy_float_ = LocalAt<float>(scratch + 2 * value_bytes + float_bytes, hidden_);
    reduce_work_ = LocalAt<float>(scratch + 2 * value_bytes + 2 * float_bytes, hidden_);
    uint64_t metadata_start = scratch + 2 * value_bytes + 3 * float_bytes;
    metadata_ = reinterpret_cast<__ubuf__ int32_t *>(metadata_start);
    result_ = LocalAt<float>(metadata_start + metadata_bytes, rows * top_k_);
    reduced_ = LocalAt<float>(metadata_start + metadata_bytes + rows * top_k_ * sizeof(float), 8);
  }

  __aicore__ inline void RunToken(int64_t token, int64_t local) {
    aclshmemi_copy_gm2ub(reinterpret_cast<__ubuf__ T *>(dy_.GetPhyAddr()),
                        reinterpret_cast<__gm__ T *>(inputs_[2]) + token * hidden_, hidden_ * sizeof(T));
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(dy_float_, dy_, RoundMode::CAST_NONE, hidden_);
    PipeBarrier<PIPE_V>();
    for (int64_t selected = 0; selected < top_k_; ++selected) {
      int64_t index = metadata_[(local * top_k_ + selected) * 4 + 2];
      if (index < 0 || index >= static_cast<int64_t>(task_.inputs[0].dim[0]) * top_k_) { AscendC::Trap(); }
      aclshmemi_copy_gm2ub(reinterpret_cast<__ubuf__ T *>(value_.GetPhyAddr()),
                          expert_ + index * hidden_, hidden_ * sizeof(T));
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      Cast(values_float_, value_, RoundMode::CAST_NONE, hidden_);
      PipeBarrier<PIPE_V>();
      Mul(values_float_, values_float_, dy_float_, hidden_);
      PipeBarrier<PIPE_V>();
      ReduceSum(reduced_, values_float_, reduce_work_, hidden_);
      SetFlag<HardEvent::V_S>(EVENT_ID0);
      WaitFlag<HardEvent::V_S>(EVENT_ID0);
      result_.SetValue(local * top_k_ + selected, reduced_.GetValue(0));
    }
  }

  GM_ADDR *inputs_;
  TaskDesc task_;
  int64_t hidden_ = 0;
  int64_t top_k_ = 0;
  __gm__ T *expert_ = nullptr;
  __ubuf__ int32_t *metadata_ = nullptr;
  GlobalTensor<float> output_;
  LocalTensor<T> value_, dy_;
  LocalTensor<float> values_float_, dy_float_, reduce_work_, result_, reduced_;
};

}  // namespace MulticoreRuntime::BackwardIndexed

#endif  // HYPER_MULTICORE_INDEXED_BACKWARD_H_
