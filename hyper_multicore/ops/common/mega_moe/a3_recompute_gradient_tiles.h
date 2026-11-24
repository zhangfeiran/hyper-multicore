// Copyright 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
#ifndef HMC_A3_RECOMPUTE_GRADIENT_TILES_H_
#define HMC_A3_RECOMPUTE_GRADIENT_TILES_H_

#include "kernel_operator.h"

namespace MulticoreMoE {
constexpr int32_t kRecomputeHidden = 4096;
constexpr int32_t kProbabilityPacketElements = 8;

template <AscendC::HardEvent Event>
__aicore__ inline void RecomputeTileSync() {
  AscendC::SetFlag<Event>(EVENT_ID0);
  AscendC::WaitFlag<Event>(EVENT_ID0);
}

// Split the two consumers of G. Scaling can publish dV before V is available.
__aicore__ inline void ScaleRecomputedGradientRows(GM_ADDR raw_g, GM_ADDR raw_p, GM_ADDR scaled_g,
                                                   int64_t first, int64_t end, int64_t stride,
                                                   int64_t probability_stride = 1) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> input_buffer, fp32_buffer;
  pipe.InitBuffer(input_buffer, kRecomputeHidden * sizeof(bfloat16_t));
  pipe.InitBuffer(fp32_buffer, kRecomputeHidden * sizeof(float));
  auto raw = input_buffer.Get<bfloat16_t>();
  auto values = fp32_buffer.Get<float>();
  AscendC::GlobalTensor<bfloat16_t> gm_inputs, gm_outputs;
  AscendC::GlobalTensor<float> gm_probs;
  gm_inputs.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(raw_g));
  gm_outputs.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(scaled_g));
  gm_probs.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(raw_p));
  for (int64_t row = first; row < end; row += stride) {
    const int64_t offset = row * kRecomputeHidden;
    const float p = gm_probs.GetValue(row * probability_stride);
    AscendC::DataCopy(raw, gm_inputs[offset], kRecomputeHidden);
    RecomputeTileSync<AscendC::HardEvent::MTE2_V>();
    AscendC::Cast(values, raw, AscendC::RoundMode::CAST_NONE, kRecomputeHidden);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(values, values, p, kRecomputeHidden);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(raw, values, AscendC::RoundMode::CAST_RINT, kRecomputeHidden);
    RecomputeTileSync<AscendC::HardEvent::V_MTE3>();
    AscendC::DataCopy(gm_outputs[offset], raw, kRecomputeHidden);
    RecomputeTileSync<AscendC::HardEvent::MTE3_MTE2>();
  }
  RecomputeTileSync<AscendC::HardEvent::MTE3_S>();
  pipe.Destroy();
}

__aicore__ inline void RecomputedProbabilityGradientRows(GM_ADDR raw_g, GM_ADDR raw_v, GM_ADDR packets,
                                                        int64_t first, int64_t end, int64_t stride) {
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> g_buffer, v_buffer, floats_buffer, products_buffer, packet_buffer;
  pipe.InitBuffer(g_buffer, kRecomputeHidden * sizeof(bfloat16_t));
  pipe.InitBuffer(v_buffer, kRecomputeHidden * sizeof(bfloat16_t));
  pipe.InitBuffer(floats_buffer, kRecomputeHidden * sizeof(float));
  pipe.InitBuffer(products_buffer, kRecomputeHidden * sizeof(float));
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
    const int64_t offset = row * kRecomputeHidden;
    AscendC::DataCopy(g, gm_g[offset], kRecomputeHidden);
    AscendC::DataCopy(v, gm_v[offset], kRecomputeHidden);
    RecomputeTileSync<AscendC::HardEvent::MTE2_V>();
    AscendC::Cast(f, g, AscendC::RoundMode::CAST_NONE, kRecomputeHidden);
    AscendC::Cast(products, v, AscendC::RoundMode::CAST_NONE, kRecomputeHidden);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(products, products, f, kRecomputeHidden);
    AscendC::PipeBarrier<PIPE_V>();
    // The A3 vendor unpermute-grad FP32 reduction tree for a full H4096 row.
    AscendC::BlockReduceSum(products, products, 64, 64, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::BlockReduceSum(products, products, 8, 64, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::BlockReduceSum(products, products, 1, 64, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::WholeReduceSum(packet, products, 8, 1, 1, 1, 8);
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

}  // namespace MulticoreMoE
#endif  // HMC_A3_RECOMPUTE_GRADIENT_TILES_H_
