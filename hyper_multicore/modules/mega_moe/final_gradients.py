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
"""Compact FP32 owner partials and explicit last-peer final output ownership."""

from dataclasses import dataclass
from typing import Any

import torch

from hyper_multicore._loader import NativeComponentUnavailableError
from hyper_multicore.torch import ops


@dataclass(frozen=True)
class OwnerGradientLayout:
    """Use zero for direct home outputs and one-based indices for FP32 scratch."""

    home_to_scratch: tuple[int, ...]
    last_edges: tuple[int, ...]
    owners: tuple[int, ...]


def resolve_final_gradient_bf16(requested: bool | None) -> bool:
    """Negotiate final-store support once per plan, keeping explicit requests fail-closed.

    Args:
        requested: None selects a supported payload, False disables final stores,
            and True requires the exact final-store protocol.

    Returns:
        Whether the actual payload supports the qualified final-store protocol.
    """
    if requested is not None and not isinstance(requested, bool):
        raise ValueError("Final gradient mode must be a boolean or None for automatic selection")
    if requested is False:
        return False
    ops._load_native()  # pylint: disable=protected-access
    version = getattr(torch.ops.hyper_multicore, "mega_moe_final_gradient_version", None)
    supported = version is not None and version() == 1
    if requested is True and not supported:
        raise NativeComponentUnavailableError("Final BF16 gradients require a matching rebuilt multicore payload")
    return supported


def require_final_gradient_native() -> None:
    """Require the final-store ABI independently of the older no-owned BF16 ABI."""
    resolve_final_gradient_bf16(True)


def owner_gradient_layout(route: Any) -> OwnerGradientLayout:
    """Find the last edge of each home expert in the unchanged stable peer order."""
    home = route.plan.config.home_experts
    edges = sorted((edge for edge in route.plan.transfers if edge.owner_rank == route.rank),
                   key=lambda edge: edge.target_rank)
    owners = tuple(sorted({edge.owner_slot for edge in edges}))
    if any(owner < 0 or owner >= home for owner in owners):
        raise ValueError("Owned gradient expert is outside the home matrix")
    indices = {owner: index + 1 for index, owner in enumerate(owners)}
    last = {edge.owner_slot: index for index, edge in enumerate(edges)}
    return OwnerGradientLayout(tuple(indices.get(expert, 0) for expert in range(home)),
                               tuple(int(last[edge.owner_slot] == index) for index, edge in enumerate(edges)), owners)


def allocate_owner_scratch(route: Any, gradient: torch.Tensor, layout: OwnerGradientLayout) -> torch.Tensor:
    """Zero cold local partials while retaining only actually owned FP32 matrices."""
    if (gradient.dtype != torch.bfloat16 or gradient.ndim != 3
            or gradient.shape[0] != len(layout.home_to_scratch)):
        raise ValueError("Final BF16 outputs require complete home expert matrices")
    scratch = gradient.new_empty((len(layout.owners), *gradient.shape[1:]), dtype=torch.float32)
    counts = route.plan.destination_counts[route.rank]
    for index, owner in enumerate(layout.owners):
        if not counts[owner]:
            scratch[index].zero_()
    return scratch
