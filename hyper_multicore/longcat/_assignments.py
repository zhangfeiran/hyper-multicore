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
"""Execute a variable number of expert assignments through a K=1 executor."""

from __future__ import annotations

from typing import Any, Callable

# LongCat assignment compaction uses the project's native Torch backend.
import torch
import torch.distributed as dist


def _validate_inputs(hidden_states: torch.Tensor, expert_ids: torch.Tensor,
                     routing_weights: torch.Tensor, active: torch.Tensor) -> None:
    """Validate local route metadata before constructing assignment tensors."""
    if hidden_states.ndim != 2 or expert_ids.ndim != 2:
        raise ValueError("masked experts require token and route matrices")
    if (active.dtype != torch.bool or active.shape != expert_ids.shape
            or routing_weights.shape != expert_ids.shape or expert_ids.shape[0] != hidden_states.shape[0]):
        raise ValueError("active routes and weights must match [tokens, top_k]")
    if expert_ids.dtype not in (torch.int32, torch.int64) or not routing_weights.is_floating_point():
        raise TypeError("expert IDs must be integer and routing weights floating point")
    if any(value.device != hidden_states.device for value in (expert_ids, routing_weights, active)):
        raise ValueError("masked expert inputs must share one device")


def masked_expert_forward(
    executor: Any,
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    active: torch.Tensor,
    *,
    expert_weights: tuple[torch.Tensor, torch.Tensor],
    ep_group: Any = None,
    route_error: torch.Tensor | None = None,
    input_gather: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compact active slots, execute real experts, and restore original tokens.

    Args:
        executor: Callable K=1 expert executor exposing num_experts.
        hidden_states: Original [T, H] token matrix.
        expert_ids: [T, K] global expert IDs; inactive IDs are ignored.
        routing_weights: Differentiable [T, K] weights without renormalization.
        active: Boolean [T, K] mask selecting real expert assignments.
        expert_weights: Current local unsharded Gate/Up and Down weights.
        ep_group: Explicit expert group; None selects standalone execution.
        route_error: Optional caller-detected route error, agreed across the EP group.
        input_gather: Optional source gather with a custom backward; called even for an empty route.

    Returns:
        Weighted real-expert contribution in the original token order.

    Note:
        An indexed gather aliases the original activation and passes explicit row metadata.
        Ordinary gather paths materialize [R, H] scratch for R active slots.
    """
    _validate_inputs(hidden_states, expert_ids, routing_weights, active)
    token, slot = active.nonzero(as_tuple=True)
    ids = expert_ids[token, slot].reshape(-1, 1).contiguous()
    invalid = ((ids < 0) | (ids >= executor.num_experts)).any().to(torch.int64)
    if route_error is not None:
        invalid = torch.maximum(invalid, route_error.to(dtype=torch.int64, device=invalid.device))
    agreement = torch.stack((invalid.new_tensor(ids.shape[0]), invalid))
    if ep_group is not None:
        dist.all_reduce(agreement, group=ep_group)
    total, invalid_count = agreement.tolist()
    if invalid_count:
        raise ValueError("an active route contains an invalid real expert ID")
    states = input_gather(hidden_states, token, slot) if input_gather is not None else None
    if total == 0:
        # Preserve unused expert parameters as grad=None, including AdamW decay.
        return hidden_states * 0 + routing_weights.sum().to(hidden_states.dtype) * 0
    if states is None:
        states = hidden_states.index_select(0, token)
    options = ({"_indexed_source": (hidden_states, token)}
               if getattr(input_gather, "indexed", False) else {})
    unweighted = executor(
        states, ids, routing_weights.new_ones((ids.shape[0], 1)), expert_weights=expert_weights, **options,
    )
    weighted = (unweighted.float() * routing_weights[token, slot, None]).to(hidden_states.dtype)
    # The empty dependency keeps remote expert backward reachable on empty sources.
    output = hidden_states.new_zeros(hidden_states.shape) + unweighted[:0].sum().to(hidden_states.dtype)
    # One assignment per token in each slot keeps the caller's accumulation order.
    for position in range(expert_ids.shape[1]):
        selected = slot == position
        slot_tokens = token[selected]
        if slot_tokens.numel():
            # Some accelerator index_copy implementations leave empty-index output unwritten.
            contribution = torch.zeros_like(hidden_states).index_copy(0, slot_tokens, weighted[selected])
            output = output + contribution
    return output
