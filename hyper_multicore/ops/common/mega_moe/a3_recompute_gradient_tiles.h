// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef HMC_A3_RECOMPUTE_GRADIENT_TILES_H_
#define HMC_A3_RECOMPUTE_GRADIENT_TILES_H_

#include "a3_indexed_push.h"
#include "kernel_operator.h"

namespace MulticoreMoE {
constexpr int32_t kRecomputeFastReductionWidth = 4096;
constexpr int32_t kProbabilityPacketElements = 8;

template <AscendC::HardEvent Event>
__aicore__ inline void RecomputeTileSync() {
  AscendC::SetFlag<Event>(EVENT_ID0);
  AscendC::WaitFlag<Event>(EVENT_ID0);
}

// Preserve the vendor unpermute-grad FP32 tree independently for each row.
__aicore__ inline void ReduceRecomputeProducts(const AscendC::LocalTensor<float> &products, uint32_t rows) {
  AscendC::BlockReduceSum(products, products, 64 * rows, 64, 1, 1, 8);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::BlockReduceSum(products, products, 8 * rows, 64, 1, 1, 8);
  AscendC::PipeBarrier<PIPE_V>();
  AscendC::BlockReduceSum(products, products, rows, 64, 1, 1, 8);
  AscendC::PipeBarrier<PIPE_V>();
}

// Use the same power-of-eight reduction and aligned tail folds as unpermute-grad.
__aicore__ inline void ReduceRecomputeRow(const AscendC::LocalTensor<float> &output,
                                         const AscendC::LocalTensor<float> &products, uint32_t hidden) {
  constexpr uint32_t vector_width = 64;
  constexpr uint32_t block_width = 8;
  const uint32_t width = hidden >= 4096 ? 4096 : (hidden >= 512 ? 512 : 64);
  while (hidden > width) {
    const uint32_t half = (((hidden + 1) / 2 + block_width - 1) / block_width) * block_width;
    const uint32_t split = half > width ? half : width;
    AscendC::Add(products, products, products[split], hidden - split);
    AscendC::PipeBarrier<PIPE_V>();
    hidden = split;
  }
  while (hidden >= vector_width) {
    AscendC::BlockReduceSum(products, products, hidden / vector_width, vector_width, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    hidden /= block_width;
  }
  AscendC::WholeReduceSum(output, products, hidden, 1, 1, 1, 8);
}

// Split the two consumers of G. Scaling can publish dV before V is available.
__aicore__ inline void ScaleRecomputedGradientRows(GM_ADDR raw_g, GM_ADDR raw_p, GM_ADDR scaled_g,
                                                   uint32_t hidden, int64_t first, int64_t end, int64_t stride,
                                                   int64_t probability_stride = 1) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> input_buffer, fp32_buffer;
  pipe.InitBuffer(input_buffer, hidden * sizeof(bfloat16_t));
  pipe.InitBuffer(fp32_buffer, hidden * sizeof(float));
  auto raw = input_buffer.Get<bfloat16_t>();
  auto values = fp32_buffer.Get<float>();
  AscendC::GlobalTensor<bfloat16_t> gm_inputs, gm_outputs;
  AscendC::GlobalTensor<float> gm_probs;
  gm_inputs.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(raw_g));
  gm_outputs.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(scaled_g));
  gm_probs.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(raw_p));
  for (int64_t row = first; row < end; row += stride) {
    const int64_t offset = row * hidden;
    const float p = gm_probs.GetValue(row * probability_stride);
    AscendC::DataCopy(raw, gm_inputs[offset], hidden);
    RecomputeTileSync<AscendC::HardEvent::MTE2_V>();
    AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE, hidden);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(values, values, p, hidden);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(raw, values, AscendC::RoundMode::CAST_RINT, hidden);
    RecomputeTileSync<AscendC::HardEvent::V_MTE3>();
    AscendC::DataCopy(gm_outputs[offset], raw, hidden);
    RecomputeTileSync<AscendC::HardEvent::MTE3_MTE2>();
  }
  RecomputeTileSync<AscendC::HardEvent::MTE3_S>();
  pipe.Destroy();
}

__aicore__ inline void RecomputedProbabilityGradientRows(GM_ADDR raw_g, GM_ADDR raw_v, GM_ADDR packets,
                                                        uint32_t hidden, int64_t first, int64_t end, int64_t stride) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> g_buffer, v_buffer, floats_buffer, products_buffer, packet_buffer;
  pipe.InitBuffer(g_buffer, hidden * sizeof(bfloat16_t));
  pipe.InitBuffer(v_buffer, hidden * sizeof(bfloat16_t));
  pipe.InitBuffer(floats_buffer, hidden * sizeof(float));
  pipe.InitBuffer(products_buffer, hidden * sizeof(float));
  pipe.InitBuffer(packet_buffer, kProbabilityPacketElements * sizeof(float));
  auto g = g_buffer.Get<bfloat16_t>();
  auto v = v_buffer.Get<bfloat16_t>();
  auto f = floats_buffer.Get<float>();
  auto products = products_buffer.Get<float>();
  auto packet = packet_buffer.Get<float>();
  AscendC::GlobalTensor<bfloat16_t> gm_g, gm_v;
  AscendC::GlobalTensor<float> gm_packets;
  gm_g.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(raw_g));
  gm_v.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(raw_v));
  gm_packets.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(packets));
  AscendC::Duplicate(packet, 0.0f, kProbabilityPacketElements);
  AscendC::PipeBarrier<PIPE_V>();
  for (int64_t row = first; row < end; row += stride) {
    const int64_t offset = row * hidden;
    AscendC::DataCopy(g, gm_g[offset], hidden);
    AscendC::DataCopy(v, gm_v[offset], hidden);
    RecomputeTileSync<AscendC::HardEvent::MTE2_V>();
    AscendC::Cast(f, g, AscendC::RoundMode::CAST_NONE, hidden);
    AscendC::Cast(products, v, AscendC::RoundMode::CAST_NONE, hidden);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(products, products, f, hidden);
    AscendC::PipeBarrier<PIPE_V>();
    if (hidden == kRecomputeFastReductionWidth) {
      ReduceRecomputeProducts(products, 1);
      AscendC::WholeReduceSum(packet, products, 8, 1, 1, 1, 8);
    } else {
      ReduceRecomputeRow(packet, products, hidden);
    }
    AscendC::Duplicate(f, 0.0f, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(packet, f, packet, 1);
    RecomputeTileSync<AscendC::HardEvent::V_MTE3>();
    AscendC::DataCopy(gm_packets[row * kProbabilityPacketElements], packet, kProbabilityPacketElements);
    RecomputeTileSync<AscendC::HardEvent::MTE3_V>();
    RecomputeTileSync<AscendC::HardEvent::MTE3_MTE2>();
  }
  RecomputeTileSync<AscendC::HardEvent::MTE3_S>();
  pipe.Destroy();
}

__aicore__ static __attribute__((noinline)) void RecomputedGradientRows(
    GM_ADDR raw_g, GM_ADDR raw_p, GM_ADDR raw_v, GM_ADDR packets, uint32_t hidden,
    int64_t first, int64_t rows, uint32_t batch_rows) {
  constexpr uint32_t bytes_per_element = 2 * sizeof(bfloat16_t) + 2 * sizeof(float);
  const uint32_t bytes_per_row = hidden * bytes_per_element + kProbabilityPacketElements * sizeof(float);
  if ((batch_rows != 1 && batch_rows != 3) || batch_rows * bytes_per_row > 192 * 1024) AscendC::Trap();
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> g_buffer, v_buffer, fp32_buffer, products_buffer, packet_buffer;
  const uint32_t elements = batch_rows * hidden;
  pipe.InitBuffer(g_buffer, elements * sizeof(bfloat16_t));
  pipe.InitBuffer(v_buffer, elements * sizeof(bfloat16_t));
  pipe.InitBuffer(fp32_buffer, elements * sizeof(float));
  pipe.InitBuffer(products_buffer, elements * sizeof(float));
  pipe.InitBuffer(packet_buffer, batch_rows * kProbabilityPacketElements * sizeof(float));
  const auto g = g_buffer.Get<bfloat16_t>();
  const auto v = v_buffer.Get<bfloat16_t>();
  const auto values = fp32_buffer.Get<float>();
  const auto products = products_buffer.Get<float>();
  const auto packet = packet_buffer.Get<float>();
  AscendC::GlobalTensor<bfloat16_t> gm_g, gm_v;
  AscendC::GlobalTensor<float> gm_probs, gm_packets;
  gm_g.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(raw_g));
  gm_v.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(raw_v));
  gm_probs.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(raw_p));
  gm_packets.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(packets));
  AscendC::Duplicate(packet, 0.0f, batch_rows * kProbabilityPacketElements);
  AscendC::PipeBarrier<PIPE_V>();

  for (int64_t row = first; row < first + rows; row += batch_rows) {
    const uint32_t remaining = first + rows - row;
    const uint32_t active = remaining < batch_rows ? remaining : batch_rows;
    const int64_t offset = row * hidden;
    const uint32_t count = active * hidden;
    AscendC::DataCopy(g, gm_g[offset], count);
    AscendC::DataCopy(v, gm_v[offset], count);
    RecomputeTileSync<AscendC::HardEvent::MTE2_V>();
    AscendC::Cast(values, g, AscendC::RoundMode::CAST_NONE, count);
    AscendC::Cast(products, v, AscendC::RoundMode::CAST_NONE, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(products, products, values, count);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < active; ++i) {
      const auto g_row = values[i * hidden];
      AscendC::Muls(g_row, g_row, gm_probs.GetValue((row + i) * kProbabilityPacketElements), hidden);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(g, values, AscendC::RoundMode::CAST_RINT, count);

    if (hidden == kRecomputeFastReductionWidth) {
      ReduceRecomputeProducts(products, active);
      for (uint32_t i = 0; i < active; ++i) {
        const uint32_t at = i * kProbabilityPacketElements;
        AscendC::WholeReduceSum(packet[at], products[at], kProbabilityPacketElements, 1, 1, 1, 8);
      }
    } else {
      for (uint32_t i = 0; i < active; ++i) {
        ReduceRecomputeRow(packet[i * kProbabilityPacketElements], products[i * hidden], hidden);
      }
    }
    AscendC::Duplicate(values, 0.0f, kProbabilityPacketElements);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < active; ++i) {
      const auto dp = packet[i * kProbabilityPacketElements];
      AscendC::Add(dp, values, dp, 1);
    }
    RecomputeTileSync<AscendC::HardEvent::V_MTE3>();
    AscendC::DataCopy(gm_packets[row * kProbabilityPacketElements], packet, active * kProbabilityPacketElements);
    // The raw G is dead on these rows after Mul, so B can read pG in the same plane.
    AscendC::DataCopy(gm_g[offset], g, count);
    RecomputeTileSync<AscendC::HardEvent::MTE3_V>();
    RecomputeTileSync<AscendC::HardEvent::MTE3_MTE2>();
  }
  RecomputeTileSync<AscendC::HardEvent::MTE3_S>();
  pipe.Destroy();
}

// Each physical row has exactly one R producer. Publish completed rows after
// the gradient routine has drained its MTE3 writes to the pG plane.
__aicore__ static __attribute__((noinline)) void PublishRecomputedGradientRows(uint64_t address, uint32_t rows) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECOUT> buffer;
  constexpr uint32_t words = 8;
  pipe.InitBuffer(buffer, words * sizeof(int32_t));
  auto values = buffer.Get<int32_t>();
  values.SetValue(0, rows);
  for (uint32_t i = 1; i < words; ++i) values.SetValue(i, 0);
  AscendC::GlobalTensor<int32_t> flag;
  flag.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(address), words);
  AscendC::SetAtomicAdd<int32_t>();
  RecomputeTileSync<AscendC::HardEvent::S_MTE3>();
  AscendC::DataCopy(flag, values, words);
  RecomputeTileSync<AscendC::HardEvent::MTE3_S>();
  AscendC::SetAtomicNone();
  pipe.Destroy();
}

}  // namespace MulticoreMoE
#endif  // HMC_A3_RECOMPUTE_GRADIENT_TILES_H_
