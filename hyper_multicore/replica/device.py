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
"""Device-resident bounded placement and lossless source quota assignment."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import torch

from .capacity import ExpertReplicaConfig, _integer
from .admission import reduction_basis_points
from .device_kernel import launch_device_planner
from .plan import ExpertReplicaTransfer


@dataclass(frozen=True)
class ReplicaPlanSummary:
    """Host-only transfer and allocation controls; source quotas stay on device."""

    config: ExpertReplicaConfig
    slot_to_logical: tuple[tuple[int, ...], ...]
    destination_counts: tuple[tuple[int, ...], ...]
    rank_splits: tuple[tuple[int, ...], ...]

    @cached_property
    def physical_to_logical(self) -> tuple[int, ...]:
        """Return fixed physical ownership, with -1 for unused guest slots."""
        return tuple(expert for row in self.slot_to_logical for expert in row)

    @cached_property
    def destination_loads(self) -> tuple[int, ...]:
        """Provide the dynamic capacity decision without reading device quotas."""
        return tuple(sum(row) for row in self.destination_counts)

    @cached_property
    def transfers(self) -> tuple[ExpertReplicaTransfer, ...]:
        """Provide the globally identical ordinary P2P/SHMEM transfer order."""
        home = self.config.home_experts
        return tuple(ExpertReplicaTransfer(expert, expert // home, expert % home, rank, slot)
                     for rank, row in enumerate(self.slot_to_logical)
                     for slot, expert in enumerate(row) if slot >= home and expert >= 0)


@dataclass(frozen=True)
class DeviceExpertExecutionPlan:
    """Keep per-source routing data on device and expose one compact control copy."""

    config: ExpertReplicaConfig
    slot_to_logical: torch.Tensor
    destination_counts: torch.Tensor
    dispatch_counts: torch.Tensor
    control: torch.Tensor
    source_order: torch.Tensor
    source_counts: torch.Tensor

    def host_summary(self) -> ReplicaPlanSummary:
        """Read placement, destination boundaries and rank splits once, never quotas."""
        values = self.control.cpu().tolist()
        if values[0]:
            raise ValueError("Device replica plan has invalid counts, conservation or capacity")
        ranks, width = self.config.ep_size, self.config.slots_per_rank
        size = ranks * width
        slots = tuple(tuple(values[1 + rank * width:1 + (rank + 1) * width]) for rank in range(ranks))
        counts = tuple(tuple(values[1 + size + rank * width:1 + size + (rank + 1) * width])
                       for rank in range(ranks))
        splits = tuple(tuple(values[1 + 2 * size + rank * ranks:1 + 2 * size + (rank + 1) * ranks])
                       for rank in range(ranks))
        return ReplicaPlanSummary(self.config, slots, counts, splits)

    def source_runs(self, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return device physical slots in stable logical occurrence order."""
        return self.source_order, self.source_counts[rank]


@torch.no_grad()
def build_device_expert_replica_plan(counts_by_source: torch.Tensor, config: ExpertReplicaConfig,
                                     *, capacity_limit: int | None = None, target_load: int | None = None,
                                     minimum_replica_rows: int = 0,
                                     minimum_load_reduction: float = 0.0,
                                     minimum_saved_rows: int = 0) -> DeviceExpertExecutionPlan:
    """Execute the shared integer policy on NPU without reading source counts.

    The fused solver matches CPU constructive placement, target-load preference,
    greedy improvement, capacity-safe small-copy consolidation and rebalancing.
    Each invocation owns its outputs, including when backward is deferred.
    Offline calibrated cost refinement is available only with the CPU planner.

    Args:
        counts_by_source: Exact source-rank histograms on the NPU.
        config: Logical topology and replica slot budget.
        capacity_limit: Hard maximum receive rows, or inferred capacity when None.
        target_load: Soft desired receive rows.
        minimum_replica_rows: Soft minimum rows served by each guest copy.
        minimum_load_reduction: Required fractional peak reduction for optional copies.
        minimum_saved_rows: Required absolute peak reduction for optional copies.

    Returns:
        Invocation-owned plan with exact per-source dispatch quotas.
    """
    for name, value in (("capacity_limit", capacity_limit), ("target_load", target_load),
                        ("minimum_replica_rows", minimum_replica_rows), ("minimum_saved_rows", minimum_saved_rows)):
        if value is not None:
            _integer(value, name, 0)
            if value > torch.iinfo(torch.int64).max:
                raise ValueError(f"{name} exceeds the device integer range")
    if counts_by_source.shape != (config.ep_size, config.num_experts):
        raise ValueError("Device source counts do not match the replica topology")
    if counts_by_source.dtype not in (torch.int32, torch.int64):
        raise ValueError("Device source counts require int32 or int64")
    control, dispatch = launch_device_planner(counts_by_source, config, capacity_limit, target_load,
                                               minimum_replica_rows,
                                               reduction_basis_points(minimum_load_reduction), minimum_saved_rows)
    size = config.physical_experts
    shape = (config.ep_size, config.slots_per_rank)
    summary_size = 1 + 2 * size + config.ep_size * config.ep_size
    return DeviceExpertExecutionPlan(config, control[1:1 + size].view(shape),
                                     control[1 + size:1 + 2 * size].view(shape), dispatch, control[:summary_size],
                                     control[summary_size:summary_size + size],
                                     control[summary_size + size:].view(config.ep_size, size))


def validate_planner_backend(backend: str, minimum_rows: int, cost_model: object) -> None:
    """Reject unsupported host-only cost refinement before any communication.

    Args:
        backend: CPU or device planner selection.
        minimum_rows: Soft minimum rows served by a guest copy.
        cost_model: Optional host calibration, unsupported by device planning.
    """
    if backend not in ("cpu", "device"):
        raise ValueError("replica_planner must be cpu or device")
    _integer(minimum_rows, "replica_min_rows", 0)
    if backend == "device" and cost_model is not None:
        raise ValueError("Device planning does not support a host cost model")
