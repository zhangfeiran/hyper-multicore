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
"""Independent native NPU EP expert reference for LongCat validation."""

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu

from hyper_multicore.replica.communication import differentiable_all_to_all_single


class _GroupedMatmul(torch.autograd.Function):
    """Compute native GMM forward and both gradients in expert-major order."""

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, weights: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        """Run the NPU grouped matrix multiplication."""
        ctx.save_for_backward(inputs, weights, groups)
        return torch_npu.npu_grouped_matmul(x=[inputs], weight=[weights], bias=[], group_list=groups,
                                            split_item=3, group_type=0, group_list_type=0)[0]

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple:
        """Use native grouped GEMMs for input and expert-weight gradients."""
        inputs, weights, groups = ctx.saved_tensors
        gradient = gradient.contiguous()
        dx = torch_npu.npu_grouped_matmul(x=[gradient], weight=[weights.transpose(-1, -2)], bias=[],
                                          group_list=groups, split_item=3, group_type=0, group_list_type=0)[0]
        dw = torch.zeros_like(weights) if inputs.numel() == 0 else torch_npu.npu_grouped_matmul(
            x=[inputs.transpose(-1, -2)], weight=[gradient], bias=[], group_list=groups,
            split_item=2, group_type=2, group_list_type=0)[0]
        return dx, dw, None


def _permute(hidden: torch.Tensor, ids: torch.Tensor) -> tuple:
    """Keep an autograd-connected empty input when a source has no rows."""
    if not hidden.shape[0]:
        return hidden, ids.new_empty((0,), dtype=torch.int32)
    return torch_npu.npu_moe_token_permute(hidden, ids.to(torch.int32).contiguous())


def _unpermute(hidden: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    """Restore one row per compact K=1 assignment."""
    return hidden[:0] if not mapping.numel() else torch_npu.npu_moe_token_unpermute(hidden, mapping)


def _exchange_counts(counts: torch.Tensor, group: Any, ep_size: int) -> tuple:
    """Exchange per-expert route counts and compute all-to-all split sizes."""
    received = torch.empty_like(counts)
    if ep_size > 1:
        dist.all_to_all_single(received, counts.contiguous(), group=group)
    else:
        received.copy_(counts)
    local = counts.numel() // ep_size
    segments = torch.stack((counts.view(ep_size, local).sum(1), received.view(ep_size, local).sum(1))).tolist()
    return received, segments[0], segments[1]


def _expert_gemms(hidden: torch.Tensor, weights: tuple, counts: torch.Tensor, precision: str) -> torch.Tensor:
    """Apply GMM, SwiGLU and GMM using exactly the requested rounding mode."""
    gate, down = weights
    if not hidden.shape[0]:
        return hidden + (gate.reshape(-1)[:1].sum() + down.reshape(-1)[:1].sum()) * 0
    groups = counts.to(torch.int64).cumsum(0)
    projected = _GroupedMatmul.apply(hidden, gate.transpose(1, 2), groups)
    if precision == "fused":
        intermediate = torch_npu.npu_swiglu(projected)
    elif precision == "hf":
        first, second = projected.chunk(2, -1)
        intermediate = F.silu(first) * second
    else:
        raise ValueError("precision must be fused or hf")
    return _GroupedMatmul.apply(intermediate, down.transpose(1, 2), groups)


class NativeExperts:
    """Expose a K=1 executor using native permute, all-to-all, GMM and SwiGLU."""

    def __init__(self, num_experts: int, ep_size: int, group: Any, precision: str = "fused") -> None:
        """Bind global expert count and explicit expert process group."""
        self.num_experts, self.ep_size, self.group, self.precision = num_experts, ep_size, group, precision

    def __call__(self, hidden: torch.Tensor, ids: torch.Tensor, _weights: torch.Tensor,
                 *, expert_weights: tuple) -> torch.Tensor:
        """Return unweighted expert outputs in compact assignment order."""
        gate, down = expert_weights
        local = self.num_experts // self.ep_size
        gate = gate[:local]
        counts = torch.bincount(ids.reshape(-1).long(), minlength=self.num_experts)
        received_counts, send_sizes, receive_sizes = _exchange_counts(counts, self.group, self.ep_size)
        source, source_mapping = _permute(hidden, ids)
        received = (differentiable_all_to_all_single(source, send_sizes, receive_sizes, self.group)
                    if self.ep_size > 1 else source)
        local_ids = torch.arange(local, device=hidden.device).repeat(self.ep_size)
        local_ids = local_ids.repeat_interleave(received_counts).reshape(-1, 1)
        expert_input, owner_mapping = _permute(received, local_ids)
        local_counts = received_counts.view(self.ep_size, local).sum(0)
        expert_output = _expert_gemms(expert_input, (gate, down), local_counts, self.precision)
        owner_output = _unpermute(expert_output, owner_mapping)
        returned = (differentiable_all_to_all_single(owner_output, receive_sizes, send_sizes, self.group)
                    if self.ep_size > 1 else owner_output)
        return _unpermute(returned, source_mapping)
