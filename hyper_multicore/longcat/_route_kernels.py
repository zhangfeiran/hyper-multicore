# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Optional Ascend Triton kernels preserving ordered BF16 route arithmetic."""

from functools import lru_cache
from typing import Any

# Native tensor APIs are the model adapter contract.
import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_round(value):
    bits = value.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    finite = (bits & 0x7f800000) != 0x7f800000
    return tl.where(finite, rounded.to(tl.float32, bitcast=True), value)


@triton.jit
def _identity_probability_product(gradient, hidden, output, row_start,
                                  H: tl.constexpr, K: tl.constexpr, B: tl.constexpr):  # pylint: disable=unused-argument
    token = tl.program_id(0) + row_start
    columns = tl.program_id(1) * B + tl.arange(0, B)
    dy = tl.load(gradient + token * H + columns, columns < H, 0).to(tl.float32)
    x = tl.load(hidden + token * H + columns, columns < H, 0).to(tl.float32)
    tl.store(output + token * H + columns, dy * x, columns < H)


@triton.jit
def _real_probability_product(gradient, experts, token_ids, output, row_start,
                              H: tl.constexpr, K: tl.constexpr, B: tl.constexpr):  # pylint: disable=unused-argument
    route = tl.program_id(0) + row_start
    columns = tl.program_id(1) * B + tl.arange(0, B)
    token = tl.load(token_ids + route)
    dy = tl.load(gradient + token * H + columns, columns < H, 0).to(tl.float32)
    expert = tl.load(experts + route * H + columns, columns < H, 0).to(tl.float32)
    tl.store(output + route * H + columns, dy * expert, columns < H)


@lru_cache(maxsize=None)
def _supports_probability_products(device: torch.device) -> bool:
    """Select the verified chip family for the original-width FP32 product kernels."""
    return str(torch.npu.get_device_name(device)).startswith("Ascend910_93")


@triton.jit
def _combine_kernel(hidden, experts, inverse, weights, output,
                   row_start, H: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0) + row_start
    columns = tl.program_id(1) * B + tl.arange(0, B)
    source = tl.load(hidden + token * H + columns, columns < H, 0).to(tl.float32)
    acc = tl.full((B,), 0, tl.float32)
    for slot in range(K):
        route = tl.load(inverse + token * K + slot)
        probability = tl.load(weights + token * K + slot)
        expert = tl.load(experts + tl.maximum(route, 0) * H + columns,
                         (columns < H) & (route >= 0), 0).to(tl.float32)
        value = tl.where(route >= 0, expert, source) * probability
        rounded = _bf16_round(value)
        acc = _bf16_round(acc + rounded)
    tl.store(output + token * H + columns, acc.to(tl.bfloat16, fp_downcast_rounding="rtne"), columns < H)


@triton.jit
def _identity_gradient_kernel(gradient, inverse, weights, output,
                             row_start, H: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0) + row_start
    columns = tl.program_id(1) * B + tl.arange(0, B)
    dy = tl.load(gradient + token * H + columns, columns < H, 0).to(tl.float32)
    acc = tl.full((B,), 0, tl.float32)
    for position in range(K):
        slot = K - 1 - position
        route = tl.load(inverse + token * K + slot)
        probability = tl.load(weights + token * K + slot)
        value = _bf16_round(tl.where(route < 0, dy * probability, 0))
        acc = _bf16_round(acc + value)
    tl.store(output + token * H + columns, acc.to(tl.bfloat16, fp_downcast_rounding="rtne"), columns < H)


@triton.jit
def _expert_gradient_kernel(gradient, token_ids, slot_ids, weights, output,
                           row_start, H: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    route = tl.program_id(0) + row_start
    columns = tl.program_id(1) * B + tl.arange(0, B)
    token = tl.load(token_ids + route)
    slot = tl.load(slot_ids + route)
    probability = tl.load(weights + token * K + slot)
    dy = tl.load(gradient + token * H + columns, columns < H, 0).to(tl.float32)
    contribution = _bf16_round(dy * probability).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(output + route * H + columns, contribution, columns < H)


@triton.jit
def _input_gradient_kernel(real_gradient, identity_gradient, inverse, output,
                          row_start, H: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0) + row_start
    columns = tl.program_id(1) * B + tl.arange(0, B)
    acc = tl.load(identity_gradient + token * H + columns, columns < H, 0).to(tl.float32)
    for position in range(K):
        slot = K - 1 - position
        route = tl.load(inverse + token * K + slot)
        value = tl.load(real_gradient + tl.maximum(route, 0) * H + columns,
                        (columns < H) & (route >= 0), 0).to(tl.float32)
        acc = _bf16_round(acc + value)
    tl.store(output + token * H + columns, acc.to(tl.bfloat16, fp_downcast_rounding="rtne"), columns < H)


@triton.jit
def _real_slot_count(inverse, token, K: tl.constexpr):
    slots = tl.arange(0, triton.next_power_of_2(K))
    routes = tl.load(inverse + token * K + slots, slots < K, -1)
    return tl.sum(((slots < K) & (routes >= 0)).to(tl.int32), 0)


@triton.jit
def _identity_prefix_gradient_kernel(gradient, inverse, weights, output,
                                     row_start, H: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0) + row_start
    columns = tl.program_id(1) * B + tl.arange(0, B)
    real = _real_slot_count(inverse, token, K)
    dy = tl.load(gradient + token * H + columns, columns < H, 0).to(tl.float32)
    acc = tl.full((B,), 0, tl.float32)
    for position in range(K - real):
        slot = K - 1 - position
        probability = tl.load(weights + token * K + slot)
        value = _bf16_round(dy * probability)
        acc = _bf16_round(acc + value)
    if real > 0:
        acc = tl.where(acc == 0, 0., acc)
    tl.store(output + token * H + columns, acc.to(tl.bfloat16, fp_downcast_rounding="rtne"), columns < H)


@triton.jit
def _input_prefix_gradient_kernel(real_gradient, identity_gradient, inverse, output,
                                  row_start, H: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0) + row_start
    columns = tl.program_id(1) * B + tl.arange(0, B)
    real = _real_slot_count(inverse, token, K)
    acc = tl.load(identity_gradient + token * H + columns, columns < H, 0).to(tl.float32)
    if real < K:
        acc = tl.where(acc == 0, 0., acc)
    for position in range(real):
        slot = real - 1 - position
        route = tl.load(inverse + token * K + slot)
        value = tl.load(real_gradient + route * H + columns, columns < H, 0).to(tl.float32)
        acc = _bf16_round(acc + value)
    tl.store(output + token * H + columns, acc.to(tl.bfloat16, fp_downcast_rounding="rtne"), columns < H)


@triton.jit
def _empty_input_gradient_kernel(identity_gradient, output,
                                 row_start, H: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0) + row_start
    columns = tl.program_id(1) * B + tl.arange(0, B)
    value = tl.load(identity_gradient + token * H + columns, columns < H, 0).to(tl.float32)
    if K > 0:
        value = tl.where(value == 0, 0., value)
    tl.store(output + token * H + columns, value.to(tl.bfloat16, fp_downcast_rounding="rtne"), columns < H)


def _launch(kernel: Any, rows: int, hidden_size: int, *args: torch.Tensor, top_k: int) -> None:
    """Launch only nonempty grids and disable multiply/add reassociation."""
    # A complete expert-gradient row reuses its scalar token and probability
    # loads. Bound the tile so the pointwise rounding fits the tested UB budget.
    full_row = (kernel is _expert_gradient_kernel or kernel is _identity_probability_product
                or kernel is _real_probability_product)
    block = triton.next_power_of_2(hidden_size) if full_row and hidden_size <= 8192 else 2048
    if kernel is _input_prefix_gradient_kernel:
        block = 4096
    columns = triton.cdiv(hidden_size, block)
    maximum_rows = max(1, 65535 // columns)
    for offset in range(0, rows, maximum_rows):
        count = min(maximum_rows, rows - offset)
        kernel[(count, columns)](*args, row_start=offset, H=hidden_size, K=top_k,
                                 B=block, enable_fp_fusion=False)


class OrderedGather(torch.autograd.Function):
    """Retain one identity alias and accumulate real dX after identity dX."""

    @staticmethod
    def forward(ctx: Any, hidden: torch.Tensor, token: torch.Tensor,
                inverse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather real assignments and retain one storage-sharing identity view.

        Args:
            ctx: Autograd context retaining source shape and route mapping.
            hidden: Contiguous BF16 source states.
            token: Source token index for each real assignment.
            inverse: Assignment index for each sorted route slot, or -1 for identity.
        """
        ctx.save_for_backward(inverse)
        ctx.shape = hidden.shape
        return hidden.index_select(0, token), hidden.view_as(hidden)

    @staticmethod
    def backward(ctx: Any, real_gradient: torch.Tensor, identity_gradient: torch.Tensor) -> tuple:
        """Accumulate descending identity slots before descending real slots.

        Args:
            ctx: Retained source shape and route mapping.
            real_gradient: Compact real-assignment input gradients.
            identity_gradient: Ordered sum of local identity gradients.
        """
        inverse, = ctx.saved_tensors
        output = torch.empty_like(identity_gradient)
        specialized = ctx.shape[1] == 6144 and inverse.shape[1] == 12
        if specialized and not real_gradient.shape[0]:
            _launch(_empty_input_gradient_kernel, ctx.shape[0], ctx.shape[1],
                    identity_gradient, output, top_k=inverse.shape[1])
            return output, None, None
        # The adapter sorts real experts before identity experts. Skip their
        # zero contributions while retaining the descending BF16 sum order.
        kernel = (_input_prefix_gradient_kernel
                  if specialized and real_gradient.shape[0] < ctx.shape[0] * inverse.shape[1]
                  else _input_gradient_kernel)
        _launch(kernel, ctx.shape[0], ctx.shape[1],
               real_gradient, identity_gradient, inverse, output, top_k=inverse.shape[1])
        return output, None, None


class OrderedCombine(torch.autograd.Function):
    """Fuse forward and ordered identity dX; retain native FP32 reductions for dP."""

    @staticmethod
    def forward(ctx: Any, hidden: torch.Tensor, expert_output: torch.Tensor, weights: torch.Tensor,
                inverse: torch.Tensor, token: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        """Add weighted contributions in ascending expert order with per-slot casts.

        Args:
            ctx: Autograd context retaining expert outputs and probabilities.
            hidden: Original token states used by identity experts.
            expert_output: Unweighted real-assignment expert outputs.
            weights: FP32 probabilities in sorted expert order.
            inverse: Assignment index per sorted slot, or -1 for identity.
            token: Source token index per real assignment.
            slot: Sorted slot index per real assignment.
        """
        ctx.save_for_backward(hidden, expert_output, weights, inverse, token, slot)
        output = torch.empty_like(hidden)
        _launch(_combine_kernel, hidden.shape[0], hidden.shape[1],
               hidden, expert_output, inverse, weights, output, top_k=weights.shape[1])
        return output

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple:
        """Preserve native FP32 probability reductions and ordered BF16 input sums.

        Args:
            ctx: Retained source, expert outputs and route tensors.
            gradient: Upstream token output gradient.
        """
        hidden, expert_output, weights, inverse, token, slot = ctx.saved_tensors
        gradient = gradient.contiguous()
        expert_gradient = torch.empty_like(expert_output)
        top_k = weights.shape[1]
        specialized = hidden.shape[1] == 6144 and top_k == 12
        if specialized and token.numel() == hidden.shape[0] * top_k:
            identity_gradient = torch.zeros_like(hidden)
        else:
            identity_gradient = torch.empty_like(hidden)
            kernel = (_identity_prefix_gradient_kernel
                      if specialized and token.numel() else _identity_gradient_kernel)
            _launch(kernel, hidden.shape[0], hidden.shape[1],
                    gradient, inverse, weights, identity_gradient, top_k=top_k)
        # NPU nonzero can return a strided column view. Both Triton kernels
        # index token IDs as contiguous storage, so share the packed copy.
        packed_token = token.contiguous()
        _launch(_expert_gradient_kernel, token.numel(), hidden.shape[1],
               gradient, packed_token, slot.contiguous(), weights, expert_gradient, top_k=top_k)
        product_fusion = specialized and _supports_probability_products(gradient.device)
        if product_fusion:
            identity_product = torch.empty(hidden.shape, device=hidden.device, dtype=torch.float32)
            _launch(_identity_probability_product, hidden.shape[0], hidden.shape[1],
                    gradient, hidden, identity_product, top_k=top_k)
            identity_probability = identity_product.sum(-1)
            del identity_product
        else:
            identity_probability = (gradient.float() * hidden.float()).sum(-1)
        probability_gradient = torch.where(inverse < 0, identity_probability[:, None], 0.)
        if token.numel():
            if product_fusion:
                real_product = torch.empty(expert_output.shape, device=hidden.device, dtype=torch.float32)
                _launch(_real_probability_product, token.numel(), hidden.shape[1],
                        gradient, expert_output, packed_token, real_product, top_k=top_k)
                real_probability = real_product.sum(-1)
                del real_product
            else:
                real_probability = (expert_output.float() * gradient.index_select(0, token).float()).sum(-1)
            probability_gradient.index_put_((token, slot), real_probability)
        return identity_gradient, expert_gradient, probability_gradient, None, None, None
