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
"""Validated shape, topology, and hardware specification for MegaMoe."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from hyper_multicore.replica.capacity import ExpertReplicaConfig

def _route_metadata_rows(routed_slots: int, num_experts: int, ep_size: int, mode: str) -> int:
    """Keep peer-visible records, diagnostic lines and local tile credits disjoint."""
    rows = routed_slots + 8 * num_experts * ((routed_slots + 127) // 128)
    if mode == "peer_tile":
        counters = ((ep_size * routed_slots + 127) // 128) * (num_experts // ep_size)
        rows += (counters + 3) // 4
    return rows


_COMMUNICATION_SPLIT = 128
_DEFAULT_CAPACITY_FACTOR = 1.25
_A5_SHARED_FORWARD_MIN_ROWS = 512


def _supports_indexed_backward_pair(forward: str, dispatch: str, replica_slots: int = 0) -> bool:
    """Match original-row sources while keeping peer replica pairings disabled."""
    return dispatch == "pull" and (
        forward == "indexed" or (forward in ("peer_token", "peer_tile") and replica_slots == 0)
    )


@dataclass(frozen=True)
class MegaMoeSpec:
    """Shape, capacity, topology, and hardware values bound on first use."""

    local_num_tokens: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    initial_capacity_factor: float | None
    receive_capacity: int
    ep_size: int
    ep_group: Any | None
    rank_id: int
    num_cube_cores: int
    dispatch_mode: str = "push"
    capacity_growth_factor: float | None = _DEFAULT_CAPACITY_FACTOR
    dispatch_split: int = _COMMUNICATION_SPLIT
    combine_split: int = _COMMUNICATION_SPLIT
    swiglu_split: int = _COMMUNICATION_SPLIT
    swiglu_limit: float | None = None
    max_local_num_tokens: int | None = None
    schedule_tokens: int | None = None
    forward_ingress_mode: str = "legacy"
    backward_ingress_mode: str = "legacy"
    shared_intermediate_size: int = 0
    shared_placement: str = "idle"
    forward_shared_expert: bool = True
    backward_shared_expert: bool = False
    replica_slots_per_rank: int = 0
    logical_num_experts: int = 0
    replica_transport: str = "shmem_signal_kernel_gradient_bands"
    expert_weight_layout: str = "EIO"
    auto_tokens: bool = False
    compute_stages: int = 3
    mx_vector_policy: str = "legacy_odd"
    mx_swiglu_row_only: bool = False
    mx_ready_depth: int = 0
    mx_poll_policy: str = "0"
    mx_schedule_dir: str = ""
    mx_gradient_udma: bool = False

    @property
    def dynamic_tokens(self) -> bool:
        """Report variable-length execution with automatic or bounded storage."""
        return self.auto_tokens or self.max_local_num_tokens is not None

    @property
    def plan_tokens(self) -> int:
        """Return schedule coverage independently of reserved source storage."""
        return self.local_num_tokens if self.schedule_tokens is None else self.schedule_tokens

    @property
    def local_experts(self) -> int:
        """Return execution slots on this rank, including enabled guest slots."""
        return self.num_experts // self.ep_size

    @property
    def maximum_receive_capacity(self) -> int:
        """Return the bound for dynamic growth without requiring preallocation."""
        if not self.replica_slots_per_rank:
            return _align_capacity(self.ep_size * self.routed_slots)
        upper = ExpertReplicaConfig(self.logical_num_experts, self.ep_size, 0).maximum_receive_rows(
            self.local_num_tokens, self.top_k)
        return upper if self.dispatch_mode == 'pull' else max(upper, self.receive_capacity)

    @property
    def routed_slots(self) -> int:
        """Return routed rows produced by local Top-K expansion."""
        return self.local_num_tokens * self.top_k


def _align_capacity(capacity: int) -> int:
    """Align receive capacity to the baseline compute tile, independent of communication tiles."""
    return (
        (capacity + _COMMUNICATION_SPLIT - 1)
        // _COMMUNICATION_SPLIT
        * _COMMUNICATION_SPLIT
    )


def _resolve_receive_capacity(
    initial_capacity_factor: float | None,
    routed_slots: int,
    ep_size: int,
) -> int:
    """Resolve initial receive rows without exceeding the lossless route bound."""
    factor = 1.0 if initial_capacity_factor is None else initial_capacity_factor
    requested = math.ceil(min(factor, ep_size) * routed_slots)
    return _align_capacity(requested)


def _grow_receive_capacity(spec: MegaMoeSpec, current: int, required: int) -> tuple[int, int]:
    """Return geometric and minimum capacities without exceeding the lossless bound."""
    upper = spec.maximum_receive_capacity
    if required > upper:
        raise ValueError("MegaMoe route exceeds the lossless token bound")
    requested = max(required, math.ceil(min(upper, current * spec.capacity_growth_factor)))
    return min(upper, _align_capacity(requested)), _align_capacity(required)


def initial_receive_capacity(specification: Mapping[str, Any]) -> int:
    """Resolve the receive reservation independently of the physical guest slot cap.

    Args:
        specification: Validated unbound executor configuration.
    """
    capacity = _resolve_receive_capacity(
        specification["initial_capacity_factor"],
        specification["local_num_tokens"] * specification["top_k"], specification["ep_size"],
    )
    return capacity


def _resolve_capacity_factors(
    dispatch_mode: str,
    initial_capacity_factor: float | None,
    capacity_growth_factor: float | None,
) -> tuple[float | None, float | None]:
    """Resolve push defaults and reject capacity knobs for pull before allocation."""
    if dispatch_mode not in ("push", "pull"):
        raise ValueError("dispatch_mode must be push or pull")
    if dispatch_mode == "pull":
        if initial_capacity_factor is not None or capacity_growth_factor is not None:
            raise ValueError("initial_capacity_factor and capacity_growth_factor are only supported for push")
        return None, None
    factors = {
        "initial_capacity_factor": (
            _DEFAULT_CAPACITY_FACTOR if initial_capacity_factor is None else initial_capacity_factor),
        "capacity_growth_factor": (
            _DEFAULT_CAPACITY_FACTOR if capacity_growth_factor is None else capacity_growth_factor),
    }
    for name, value in factors.items():
        valid_type = isinstance(value, (int, float)) and not isinstance(value, bool)
        try:
            valid = valid_type and math.isfinite(value) and value >= 1.0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"{name} must be a finite number at least 1.0, got {value!r}")
    return float(factors["initial_capacity_factor"]), float(factors["capacity_growth_factor"])


def bind_mega_moe_spec(
    specification: Mapping[str, Any],
    tensor: torch.Tensor,
) -> MegaMoeSpec:
    """Bind hardware and process-group values to the first NPU input.

    Args:
        specification: Constructor-validated static resource specification.
        tensor: First input tensor used to resolve device-specific limits.

    Returns:
        Bound specification cached for subsequent forwards.

    Raises:
        ValueError: If the device or EP process group does not match the specification.
        TypeError: If the activation dtype is unsupported.
        RuntimeError: If the device does not report a valid Cube-core count.
    """
    device = torch.device(tensor.device)
    if device.type != "npu":
        raise ValueError(f"MegaMoeExperts requires an NPU input, got {device}.")
    if tensor.dtype != torch.bfloat16:
        raise TypeError(
            f"MegaMoeExperts supports BF16 activations, got {tensor.dtype}."
        )

    ep_group = specification["ep_group"]
    ep_size = specification["ep_size"]
    if dist.is_initialized():
        if ep_group is None and dist.get_world_size() > 1:
            raise ValueError("Pass ep_group explicitly for distributed MegaMoeExperts")
        group_size = dist.get_world_size(ep_group)
        rank_id = dist.get_rank(ep_group)
    else:
        if ep_group is not None:
            raise RuntimeError(
                "ep_group cannot be used before torch.distributed initialization."
            )
        group_size = 1
        rank_id = 0
    if group_size != ep_size:
        raise ValueError(
            f"ep_group world size ({group_size}) does not match ep_size ({ep_size})."
        )

    device_index = (
        device.index if device.index is not None else torch.npu.current_device()
    )
    device_limits = torch.npu.get_device_limit(device_index)
    num_cube_cores = device_limits.get("cube_core_num")
    if not isinstance(num_cube_cores, int) or num_cube_cores <= 0:
        raise RuntimeError(
            "torch.npu.get_device_limit() returned an invalid cube_core_num "
            f"for NPU {device_index}: {num_cube_cores!r}."
        )

    local_num_tokens = specification["local_num_tokens"]
    top_k = specification["top_k"]
    forward_ingress_mode = specification.get("forward_ingress_mode", "legacy")
    if forward_ingress_mode != "legacy" and num_cube_cores != 32:
        raise ValueError("Indexed forward ingress requires the Ascend 950 32-Cube BF16 kernel.")
    backward_ingress_mode = specification.get("backward_ingress_mode", "legacy")
    if backward_ingress_mode != "legacy" and num_cube_cores != 32:
        raise ValueError("Indexed backward ingress requires the Ascend 950 32-Cube BF16 kernel.")
    initial_capacity_factor = specification["initial_capacity_factor"]
    swiglu_limit = specification["swiglu_limit"]
    shared_intermediate_size = specification.get("shared_intermediate_size", 0)
    forward_shared_expert = specification.get("forward_shared_expert", True)
    if (not specification.get("auto_tokens", False) and num_cube_cores == 32
            and shared_intermediate_size and local_num_tokens < _A5_SHARED_FORWARD_MIN_ROWS):
        # Smaller integrated shared forwards have not met the strict gradient
        # agreement gate. The native path retains that contract for these rows.
        forward_shared_expert = False
    receive_capacity = initial_receive_capacity(specification)
    return MegaMoeSpec(
        local_num_tokens=local_num_tokens,
        hidden_size=specification["hidden_size"],
        intermediate_size=specification["intermediate_size"],
        num_experts=specification["num_experts"],
        logical_num_experts=specification.get("logical_num_experts", specification["num_experts"]),
        replica_slots_per_rank=specification.get("replica_slots_per_rank", 0),
        compute_stages=specification.get("compute_stages", 3),
        mx_vector_policy=specification.get("mx_vector_policy", "legacy_odd"),
        mx_swiglu_row_only=specification.get("mx_swiglu_row_only", False),
        mx_ready_depth=specification.get("mx_ready_depth", 0),
        mx_poll_policy=specification.get("mx_poll_policy", "0"),
        mx_gradient_udma=specification.get("mx_gradient_udma", False),
        replica_transport=specification.get("replica_transport", "shmem_signal_kernel_gradient_bands"),
        top_k=top_k,
        initial_capacity_factor=initial_capacity_factor,
        receive_capacity=receive_capacity,
        ep_size=ep_size,
        ep_group=ep_group,
        rank_id=rank_id,
        num_cube_cores=num_cube_cores,
        swiglu_limit=swiglu_limit,
        dispatch_mode=specification.get("dispatch_mode", "push"),
        expert_weight_layout=specification.get("expert_weight_layout", "EIO"),
        capacity_growth_factor=specification.get("capacity_growth_factor", _DEFAULT_CAPACITY_FACTOR),
        dispatch_split=specification.get("dispatch_split", _COMMUNICATION_SPLIT),
        combine_split=specification.get("combine_split", _COMMUNICATION_SPLIT),
        max_local_num_tokens=specification.get("max_local_num_tokens"),
        auto_tokens=specification.get("auto_tokens", False),
        forward_ingress_mode=forward_ingress_mode,
        backward_ingress_mode=backward_ingress_mode,
        shared_intermediate_size=shared_intermediate_size,
        shared_placement=specification.get("shared_placement", "idle"),
        forward_shared_expert=forward_shared_expert,
        backward_shared_expert=specification.get("backward_shared_expert", False),
    )
