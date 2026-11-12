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
"""Keep MXFP8 expert reductions complete while using bounded guest slots."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

from .admission import reduction_basis_points
from .capacity import ExpertReplicaConfig, _integer
from .plan import ExpertExecutionPlan
from .planner import _counts_matrix, _home_layout, _materialize as _materialize_home
from .topology import _validate_rank_nodes


def build_whole_expert_plan(counts: Sequence[Sequence[int]], config: ExpertReplicaConfig, *, capacity_limit: int,
                            minimum_load_reduction: float, minimum_saved_rows: int = 0,
                            minimum_replica_rows: int = 0,
                            rank_nodes: tuple[int, ...] | None = None) -> ExpertExecutionPlan:
    """Seed complete reductions directly after the unchanged compact admission gate.

    Args:
        counts: Exact CPU logical-expert histograms from the existing count gather.
        config: Validated logical topology and physical guest budget.
        capacity_limit: Common receive bound applied to the final complete plan.
        minimum_load_reduction: Existing optional migration threshold.
        minimum_saved_rows: Existing minimum absolute peak reduction.
        minimum_replica_rows: Existing minimum optional guest size.
        rank_nodes: Immutable node identities in expert-parallel rank order.

    Returns:
        The same complete plan as converting an intermediate split-K placement.
    """
    matrix = _counts_matrix(counts)
    if len(matrix) != config.ep_size or len(matrix[0]) != config.num_experts:
        raise ValueError("Exact counts must match the complete expert topology")
    _, _, home = _home_layout(matrix, config)
    # Necessary migrations can make an over-capacity home layout feasible.
    seed_limit = capacity_limit if not config.replica_slots_per_rank else None
    seed = _materialize_home(matrix, home, config, seed_limit)
    return place_whole_experts(
        seed, capacity_limit=capacity_limit, minimum_load_reduction=minimum_load_reduction,
        minimum_saved_rows=minimum_saved_rows, minimum_replica_rows=minimum_replica_rows, rank_nodes=rank_nodes,
    )


def _materialize(plan: ExpertExecutionPlan, targets: list[int]) -> ExpertExecutionPlan:
    """Retain home parameter identity and one complete physical destination per expert."""
    config = plan.config
    home, width = config.home_experts, config.slots_per_rank
    slots, destinations = [], {}
    for rank in range(config.ep_size):
        guests = [expert for expert, target in enumerate(targets) if target == rank and expert // home != rank]
        row = tuple(range(rank * home, (rank + 1) * home)) + tuple(guests)
        slots.append(row + (-1,) * (width - len(row)))
        for slot, expert in enumerate(slots[-1]):
            if expert >= 0 and targets[expert] == rank:
                destinations[expert] = rank * width + slot
    dispatch = []
    for source in plan.logical_counts:
        row = [0] * config.physical_experts
        for expert, count in enumerate(source):
            row[destinations[expert]] = count
        dispatch.append(tuple(row))
    result = ExpertExecutionPlan(config, plan.logical_counts, tuple(slots), tuple(dispatch))
    result.validate()
    return result


@lru_cache(maxsize=64, typed=True)
def place_whole_experts(plan: ExpertExecutionPlan, *, capacity_limit: int,
                        minimum_load_reduction: float, minimum_saved_rows: int = 0,
                        minimum_replica_rows: int = 0,
                        rank_nodes: tuple[int, ...] | None = None) -> ExpertExecutionPlan:
    """Place complete expert reductions with deterministic minimax moves and pruning.

    Splitting the K reduction can change quantization groups and FP32 gradient
    products. A complete expert retains its original source order on one home
    or guest destination, using the existing weight and gradient protocols.

    Args:
        plan: Validated immutable CPU plan from the shared replica planner.
        capacity_limit: Common receive bound already owned by the invocation.
        minimum_load_reduction: Required peak reduction for optional migration.
        minimum_saved_rows: Minimum absolute receive reduction for optional migration.
        minimum_replica_rows: Minimum useful optional guest size.
        rank_nodes: Immutable node labels in EP-rank order, when available.

    Returns:
        Immutable complete-K plan preserving logical counts and home parameters.

    Raises:
        ValueError: Complete experts cannot satisfy the common receive capacity.
    """
    plan.validate()
    _integer(capacity_limit, "Complete expert capacity", 0)
    _integer(minimum_saved_rows, "Complete expert minimum saved rows", 0)
    _integer(minimum_replica_rows, "Complete expert minimum replica rows", 0)
    reduction_basis_points(minimum_load_reduction)
    config = plan.config
    nodes = _validate_rank_nodes(rank_nodes or (), config.ep_size) or (0,) * config.ep_size
    if not config.replica_slots_per_rank:
        return plan
    owners = [expert // config.home_experts for expert in range(config.num_experts)]
    weights = [sum(row[expert] for row in plan.logical_counts) for expert in range(config.num_experts)]
    targets = owners.copy()
    loads = [sum(weights[start:start + config.home_experts])
             for start in range(0, config.num_experts, config.home_experts)]
    baseline_peak = max(loads)
    lightest_home = min(loads)
    capacity_required = baseline_peak > capacity_limit
    guests = [0] * config.ep_size
    cross, guest_total = 0, 0

    def select_move() -> tuple[tuple[int, int, int, int], tuple[int, int, int] | None]:
        """Choose the first strict improvement using the existing deterministic score."""
        square = sum(load * load for load in loads)
        current = (max(loads), square, cross, guest_total)
        leaders = sorted(range(config.ep_size), key=loads.__getitem__, reverse=True)[:3]
        outside = tuple(tuple(next((loads[r] for r in leaders if r not in (source, target)), 0)
                              for target in range(config.ep_size)) for source in range(config.ep_size))
        best = current
        selected = None
        for expert, weight in enumerate(weights):
            if not weight:
                continue
            source, owner = targets[expert], owners[expert]
            for target in range(config.ep_size):
                if target == source or (target != owner and (guests[target] >= config.replica_slots_per_rank
                        or (weight < minimum_replica_rows and not capacity_required))):
                    continue
                remaining, arriving = loads[source] - weight, loads[target] + weight
                peak = max(remaining, arriving, outside[source][target])
                candidate_square = square - loads[source] ** 2 - loads[target] ** 2 + remaining ** 2 + arriving ** 2
                candidate_cross = cross - int(nodes[source] != nodes[owner]) + int(nodes[target] != nodes[owner])
                candidate_guests = guest_total - int(source != owner) + int(target != owner)
                score = peak, candidate_square, candidate_cross, candidate_guests
                if score < best:
                    best, selected = score, (expert, source, target)
        return best, selected

    def prune_guests() -> None:
        """Prune toward a whole-expert target instead of the admission ceiling."""
        # Allow one largest expert beside the lightest original home workload.
        # This avoids moving cold matrices merely to free a few rows for a guest.
        target = max(max(loads), max(weights) + lightest_home)
        ceiling = min(capacity_limit, target)
        if not capacity_required:
            ceiling = min(ceiling, baseline_peak - 1, baseline_peak - minimum_saved_rows,
                          int(baseline_peak * (1 - minimum_load_reduction)))
        for _ in range(config.num_experts):
            changed = False
            order = sorted(range(config.num_experts),
                           key=lambda expert: (-int(nodes[targets[expert]] != nodes[owners[expert]]),
                                               weights[expert], expert))
            for expert in order:
                source, owner, weight = targets[expert], owners[expert], weights[expert]
                if source != owner and loads[owner] + weight <= ceiling:
                    loads[source] -= weight
                    loads[owner] += weight
                    targets[expert] = owner
                    changed = True
            if not changed:
                break

    for _ in range(config.num_experts + config.physical_experts):
        best, selected = select_move()
        if selected is None:
            break
        expert, source, target = selected
        weight, owner = weights[expert], owners[expert]
        loads[source] -= weight
        loads[target] += weight
        guests[source] -= int(source != owner)
        guests[target] += int(target != owner)
        targets[expert] = target
        cross, guest_total = best[2:]
    prune_guests()
    peak = max(loads)

    def admitted() -> bool:
        """Keep necessary capacity moves independent of optional benefit thresholds."""
        return (peak <= capacity_limit if capacity_required else
                peak < baseline_peak and baseline_peak - peak >= minimum_saved_rows
                and peak <= baseline_peak * (1 - minimum_load_reduction))

    if not admitted():
        targets = owners.copy()
        peak = baseline_peak
    if peak > capacity_limit:
        raise ValueError("A complete expert cannot satisfy the common receive capacity without splitting K")
    result = _materialize(plan, targets)
    if max(result.destination_loads) > baseline_peak or max(result.replica_slots_used) > config.replica_slots_per_rank:
        raise RuntimeError("Complete expert placement increased the home peak or exceeded its slot budget")
    return result
