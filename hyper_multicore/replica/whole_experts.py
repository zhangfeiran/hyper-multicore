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
"""Project admitted guest slots onto indivisible logical-expert execution."""

from __future__ import annotations

from .admission import reduction_basis_points
from .capacity import _integer
from .plan import ExpertExecutionPlan
from ._selection import _optional_peak_is_useful


def _owner_loads(plan: ExpertExecutionPlan) -> tuple[list[int], list[int]]:
    """Reduce the immutable source counts into original expert and owner loads."""
    totals = [sum(row[expert] for row in plan.logical_counts) for expert in range(plan.config.num_experts)]
    home = plan.config.home_experts
    loads = [sum(totals[rank * home:(rank + 1) * home]) for rank in range(plan.config.ep_size)]
    return totals, loads


def _destination_key(loads: list[int], rows: int, target: int, owner: int) -> tuple[int, int, bool, int]:
    """Prefer lower peaks and residual imbalance, retaining the owner on exact ties."""
    proposed = [value + (rows if rank == target else 0) for rank, value in enumerate(loads)]
    return max(proposed), sum(value * value for value in proposed), target != owner, target


def _whole_targets(plan: ExpertExecutionPlan, totals: list[int], owner_loads: list[int]) -> dict[int, tuple[int, int]]:
    """Select at most one of the existing guest locations for each logical expert."""
    choices = {}
    for transfer in plan.transfers:
        choices.setdefault(transfer.logical_expert, []).append((transfer.target_rank, transfer.target_slot))
    loads = owner_loads.copy()
    targets = {}
    for expert in sorted(choices, key=lambda value: (-totals[value], value)):
        rows = totals[expert]
        if not rows:
            continue
        owner, slot = divmod(expert, plan.config.home_experts)
        loads[owner] -= rows
        selected = min([(owner, slot), *choices[expert]],
                       key=lambda location: _destination_key(loads, rows, location[0], owner))
        loads[selected[0]] += rows
        if selected[0] != owner:
            targets[expert] = selected
    return targets


def _materialize_whole(plan: ExpertExecutionPlan, targets: dict[int, tuple[int, int]]) -> ExpertExecutionPlan:
    """Retain every source's complete expert segment without splitting its reduction."""
    config = plan.config
    slots = [list(range(rank * config.home_experts, (rank + 1) * config.home_experts))
             + [-1] * config.replica_slots_per_rank for rank in range(config.ep_size)]
    for expert, (rank, slot) in targets.items():
        slots[rank][slot] = expert
    dispatch = []
    for source in plan.logical_counts:
        row = [0] * config.physical_experts
        for expert, count in enumerate(source):
            target, slot = targets.get(expert, divmod(expert, config.home_experts))
            row[target * config.slots_per_rank + slot] = count
        dispatch.append(tuple(row))
    return ExpertExecutionPlan(config, plan.logical_counts, tuple(map(tuple, slots)), tuple(dispatch))


def build_whole_expert_plan(plan: ExpertExecutionPlan, *, capacity_limit: int | None = None,
                            minimum_load_reduction: float = 0.0,
                            minimum_saved_rows: int = 0) -> ExpertExecutionPlan:
    """Keep logical-expert reductions whole within an admitted physical guest budget.

    Args:
        plan: Valid fragment plan supplying candidate guest locations and source counts.
        capacity_limit: Hard receive-row bound already agreed by the runtime.
        minimum_load_reduction: Required fractional peak reduction for optional copies.
        minimum_saved_rows: Required absolute peak reduction for optional copies.

    Returns:
        A complete-expert placement, or owner-only placement when its gain is insufficient.

    Raises:
        ValueError: No whole-expert placement or owner fallback fits the hard capacity.
    """
    _integer(minimum_saved_rows, "minimum_saved_rows", 0)
    if capacity_limit is not None:
        _integer(capacity_limit, "capacity_limit", 0)
    admission = reduction_basis_points(minimum_load_reduction)
    totals, loads = _owner_loads(plan)
    limit = max(loads) if capacity_limit is None else capacity_limit
    candidate = _materialize_whole(plan, _whole_targets(plan, totals, loads))
    peak = max(candidate.destination_loads)
    useful = _optional_peak_is_useful(peak, loads, limit,
                                     admission, minimum_saved_rows)
    if peak <= limit and useful:
        return candidate
    if max(loads) > limit:
        raise ValueError("Whole-expert replicas cannot satisfy the receive capacity; token splitting is required")
    return _materialize_whole(plan, {})
