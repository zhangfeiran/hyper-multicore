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
"""Logical routing to stable physical expert slots."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property, lru_cache
import math
import struct
from typing import Any

import torch
import torch.distributed as dist

from hyper_multicore.replica.recompute import get_recompute_state

from .admission import reduction_basis_points
from ._metadata import _async_device_bytes, _recompute_metadata, _ReplicaCountHeaderCache
from .capacity import ExpertReplicaConfig, _integer, align_capacity
from .cost import ExpertReplicaCostModel
from .count_kernel import try_count_payload
from .host import _column_totals
from .device import DeviceExpertExecutionPlan, ReplicaPlanSummary, build_device_expert_replica_plan
from .device import validate_planner_backend
from .plan import ExpertExecutionPlan, ExpertReplicaTransfer
from .planner import (
    _build_expert_replica_plan as build_expert_replica_plan,
    _optional_peak_is_useful,
    _UncalibratedCandidate,
    _admitted_uncalibrated_copies,
    _complete_guest_search_enabled,
    _projected_rank_nodes,
)
from .topology import _validate_rank_nodes
from .whole import build_whole_expert_plan


@dataclass(frozen=True)
class _HomeOnlyPlan:
    """Keep an ordinary-route decision without materializing virtual CPU quotas.

    The count table is invocation-owned. Diagnostic callers may explicitly read
    logical_counts, but normal execution consumes only the loads and empty edges.
    """

    config: ExpertReplicaConfig
    counts: torch.Tensor
    destination_loads: tuple[int, ...]

    @property
    def transfers(self) -> tuple[ExpertReplicaTransfer, ...]:
        """An ordinary invocation has no weight or gradient transfer edges."""
        return ()

    @cached_property
    def logical_counts(self) -> tuple[tuple[int, ...], ...]:
        """Read exact source counts only when explicitly requested by diagnostics."""
        return tuple(tuple(row) for row in self.counts.cpu().tolist())

    @cached_property
    def destination_counts(self) -> tuple[tuple[int, ...], ...]:
        """Provide exact diagnostic slot totals without affecting the hot path."""
        totals = tuple(map(sum, zip(*self.logical_counts)))
        home, padding = self.config.home_experts, (0,) * self.config.replica_slots_per_rank
        return tuple(totals[start:start + home] + padding for start in range(0, self.config.num_experts, home))


@dataclass(frozen=True)
class _GatheredReplicaCounts:
    """Reuse the native route's validated count agreement without another collective."""

    counts: torch.Tensor
    source_summary: list[list[int]]
    host_counts: list[list[int]] | None = None


def _host_column_totals(counts: list[list[int]]) -> list[int]:
    """Use the existing host agreement without freezing or reading device counts again."""
    totals = _column_totals(counts)
    return list(map(sum, zip(*counts))) if totals is None else totals


@dataclass(frozen=True)
class ReplicaRoute:
    """Invocation-owned logical plan and device execution metadata.

    The optional dynamic summary stores invalid/token/autograd headers, logical
    count sum/min/max, and destination load for every EP rank on the host.
    """

    plan: ExpertExecutionPlan | ReplicaPlanSummary | _HomeOnlyPlan
    physical_ids: torch.Tensor
    counts_by_source: torch.Tensor
    rank: int
    group: object
    transport: object | None = None
    device_plan: DeviceExpertExecutionPlan | None = None
    home_only: bool = False
    dynamic_summary: tuple[tuple[int, ...], ...] | None = None
    plan_reused: bool = False
    plan_bypassed: bool = False
    _host_counts: tuple[tuple[int, ...], ...] | None = None


@dataclass(frozen=True)
class _RejectedReplicaQuotas:
    """Retain source-independent rejection evidence without holding device tensors."""

    config: ExpertReplicaConfig
    totals: tuple[int, ...]
    loads: tuple[int, ...]
    target: int
    admission: int
    saved_rows: int
    minimum_rows: int

    def matches(self, config: ExpertReplicaConfig, totals: list[int], loads: tuple[int, ...],
                target: int, admission: int, saved_rows: int, minimum_rows: int) -> bool:
        """Require the same quota inputs and benefit policy after fresh count agreement."""
        return (self.config == config and self.totals == tuple(totals) and self.loads == loads
                and (self.target, self.admission, self.saved_rows, self.minimum_rows)
                == (target, admission, saved_rows, minimum_rows))


class _RejectedQuotaMemo:
    """Keep one exact CPU-only rejection after fresh count and capacity agreement."""

    def __init__(self) -> None:
        """Start without reusable rejection evidence or retained device storage."""
        self.signature: tuple[Any, ...] | None = None
        self.hits = 0

    def contains(self, signature: tuple[Any, ...]) -> bool:
        """Reuse only the complete pure-search inputs, never invocation metadata.

        Args:
            signature: Exact topology, expert totals, loads, capacity and benefit-policy inputs.
        """
        if self.signature != signature:
            return False
        self.hits += 1
        return True


@lru_cache(maxsize=64, typed=True)
def _cached_complete_admission(signature: tuple[Any, ...]) -> tuple[tuple[tuple[int, int], ...], ...] | None:
    """Retain only immutable CPU quotas after an exact fresh-count admission signature."""
    decision, limit, resident, complete_search, _, projected_nodes = signature
    copies = _admitted_uncalibrated_copies(
        list(decision.totals), list(decision.loads), decision.config, limit, resident, decision.target,
        decision.minimum_rows, decision.admission, decision.saved_rows,
        allow_complete_search=complete_search, projected_rank_nodes=projected_nodes)
    return None if copies is None else tuple(tuple(row.items()) for row in copies)


def _complete_admission_copies(signature: tuple[Any, ...]) -> list[dict[int, int]] | None:
    """Keep mutable quota dictionaries invocation-owned instead of sharing cached planner state."""
    _integer(signature[0].minimum_rows, 'minimum_replica_rows', 0)
    copies = _cached_complete_admission(signature)
    return None if copies is None else [dict(row) for row in copies]


class _RecomputedReplicaPlans:
    """Retain immutable CPU plans and home-decision evidence for one invocation."""

    def __init__(self) -> None:
        """Initialize ordered call records without retaining any device tensors."""
        self.entries: list[tuple[Any, ExpertExecutionPlan | None]] = []
        self.cursor = 0

    def clear(self) -> None:
        """Release all plans after complete or interrupted recomputation."""
        self.entries.clear()
        self.cursor = 0

    def lookup(self, signature: Any, counts: tuple[tuple[int, ...], ...]) -> ExpertExecutionPlan | None:
        """Advance in invocation order and reject changed counts or static policy."""
        if self.cursor >= len(self.entries):
            return None
        expected, plan = self.entries[self.cursor]
        self.cursor += 1
        return plan if plan is not None and signature == expected and counts == plan.logical_counts else None


def _record_home_decision(decision: _RejectedReplicaQuotas | None = None) -> None:
    """Preserve replay ordering even when a call never enters the CPU planner."""
    state = get_recompute_state()
    if state is None:
        return
    cache = state.get_resource(_RecomputedReplicaPlans, _RecomputedReplicaPlans)
    if state.is_recomputing:
        if cache.cursor < len(cache.entries):
            cache.cursor += 1
    else:
        cache.entries.append((decision, None))


def _validate_complete_expert_policy(selected: bool, backend: str) -> None:
    """Reject incomplete backend support before any collective or device operation."""
    if not isinstance(selected, bool) or (selected and backend != "cpu"):
        raise ValueError("Complete expert selection requires a boolean flag and the CPU planner")


def _build_route_plan(counts: Any, config: ExpertReplicaConfig, complete_experts: bool,
                      options: dict[str, Any]) -> ExpertExecutionPlan:
    """Select complete MX reductions before checkpoint caching and metadata upload."""
    if not complete_experts:
        return build_expert_replica_plan(counts, config.replica_slots_per_rank, **options)
    if options.get('cost_model') is not None:
        raise ValueError("Complete MXFP8 experts require costs calibrated for complete reductions")
    _validate_complete_options(options, config)
    return build_whole_expert_plan(
        counts, config, capacity_limit=options.get('capacity_limit'),
        minimum_load_reduction=options.get('minimum_load_reduction', 0.0),
        minimum_saved_rows=options.get('minimum_saved_rows', 0),
        minimum_replica_rows=options.get('minimum_replica_rows', 0),
        rank_nodes=options.get('_rank_nodes') or None,
        transfer_row_credit=options.get('_replica_transfer_row_credit', 0),
    )


def _validate_complete_options(options: dict[str, Any], config: ExpertReplicaConfig) -> None:
    """Retain shared boundary checks while skipping unused split-K computation."""
    allowed = {'target_load', 'minimum_replica_rows', 'capacity_limit', 'cost_model', 'minimum_load_reduction',
               'minimum_saved_rows', 'receive_capacity', 'adaptive', '_rank_nodes',
               '_replica_transfer_row_credit', '_uncalibrated_candidate'}
    unknown = options.keys() - allowed
    if unknown:
        raise TypeError("Unknown complete expert options: " + ', '.join(sorted(unknown)))
    _integer(options.get('_replica_transfer_row_credit', 0), "Replica transfer row credit", 0)
    for name in ('target_load', 'receive_capacity'):
        value = options.get(name)
        if value is not None:
            _integer(value, name, 0)
    adaptive = options.get('adaptive', False)
    if not isinstance(adaptive, bool) or (adaptive and not config.replica_slots_per_rank):
        raise ValueError("Adaptive replicas require a boolean flag and a positive slot cap")
    candidate = options.get('_uncalibrated_candidate')
    if candidate is not None and not isinstance(candidate, _UncalibratedCandidate):
        raise ValueError("Complete expert admission requires a validated compact candidate")


def _plan_replica_route(counts: Any, config: ExpertReplicaConfig, group: object, rank: int,
                        *, _complete_experts: bool = False, **options: Any) -> tuple[ExpertExecutionPlan, bool]:
    """Replay the original decision after the usual count exchange validates current inputs."""
    state = get_recompute_state()
    if state is None:
        return _build_route_plan(counts, config, _complete_experts, options), False
    frozen = tuple(tuple(row) for row in counts)
    # Heap allowance and gradient-dependent credits can change between phases.
    # Replaying the original decision preserves the checkpoint's physical graph.
    policy = tuple((key, type(value), value) for key, value in sorted(options.items())
                   if key not in ('receive_capacity', 'capacity_limit', '_replica_transfer_row_credit',
                                  '_uncalibrated_candidate'))
    signature = (config, id(group), rank, policy, _complete_experts)
    cache = state.get_resource(_RecomputedReplicaPlans, _RecomputedReplicaPlans)
    if state.is_recomputing:
        plan = cache.lookup(signature, frozen)
        limit = options.get('capacity_limit')
        if plan is not None and (limit is None or max(plan.destination_loads) <= limit):
            return plan, True
    plan = _build_route_plan(frozen, config, _complete_experts, options)
    if not state.is_recomputing:
        cache.entries.append((signature, plan))
    return plan, False


def stable_expert_order(ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Sort small expert keys on device while keeping permutation indices integer.

    FP32 represents every expert ID exactly below 2**24. Token indices are never
    converted to floating point, so large token batches keep exact inverses.

    Args:
        ids: Logical expert IDs to order stably.
        num_experts: Logical expert count determining the exact sorting key dtype.

    Returns:
        Integer permutation preserving the order of equal expert IDs.
    """
    keys = ids.float() if num_experts <= 2**24 else ids
    return torch.argsort(keys, stable=True)


def _upload_route_metadata(plan: ExpertExecutionPlan, rank: int,
                           device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Upload one owned buffer with aligned int64 runs and int32 dispatch counts."""
    cache, recomputing = _recompute_metadata(plan)
    if cache is not None and recomputing:
        cached = cache.lookup(plan, rank, device, "dispatch")
        if cached is not None:
            return cached
    runs = plan.source_runs_for(rank) if plan.transfers else ()
    slots = tuple(slot for slot, _ in runs)
    lengths = tuple(count for _, count in runs)
    data = struct.pack(f"<{2 * len(runs)}q", *slots, *lengths) + plan._dispatch_counts_bytes
    # The pinned allocator retains the source through its copy event. Device
    # views own the result through remapping and every downstream consumer.
    packed = _async_device_bytes(data, device)
    run_bytes = 16 * len(runs)
    indices = packed[:run_bytes].view(torch.int64).reshape(2, len(runs))
    dispatch = packed[run_bytes:].view(torch.int32).reshape(plan.config.ep_size, plan.config.physical_experts)
    result = indices[0], indices[1], dispatch
    if cache is not None and not recomputing:
        cache.remember(plan, rank, device, "dispatch", result)
    return result


def _home_route_counts(plan: ExpertExecutionPlan, rank: int, logical_counts: torch.Tensor) -> torch.Tensor:
    """Retain owned compact counts without creating guest dispatch metadata."""
    cache, recomputing = _recompute_metadata(plan)
    if cache is not None and recomputing:
        cached = cache.lookup(plan, rank, logical_counts.device, "home")
        if cached is not None:
            return cached[0]
    counts = logical_counts.to(torch.int32).contiguous()
    if cache is not None and not recomputing:
        # A no-op conversion may still alias a producer's reusable count buffer.
        if counts is logical_counts:
            counts = counts.clone()
        cache.remember(plan, rank, logical_counts.device, "home", (counts,))
    return counts


def _remap_replica_ids(ids: torch.Tensor, plan: ExpertExecutionPlan,
                       slots: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Keep TopK order, bypassing quota expansion when all experts stay home."""
    if not plan.transfers:
        home = plan.config.home_experts
        return ids + torch.div(ids, home, rounding_mode="floor") * plan.config.replica_slots_per_rank
    destinations = torch.repeat_interleave(slots, lengths, output_size=ids.numel())
    order = stable_expert_order(ids.flatten(), plan.config.num_experts)
    remapped = torch.empty_like(ids.flatten())
    remapped.scatter_(0, order, destinations)
    return remapped.reshape_as(ids)


def _gather_replica_counts(payload: torch.Tensor, size: int, group: object) -> torch.Tensor:
    """Gather rank-major counts into one invocation-owned contiguous buffer."""
    gathered = torch.empty((size, payload.numel()), dtype=payload.dtype, device=payload.device)
    if size > 1:
        work = dist.all_gather_into_tensor(gathered.flatten(), payload, group=group, async_op=True)
        work.wait()
    else:
        gathered[0].copy_(payload)
    return gathered


def _replica_count_payload(ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Keep route validation and logical counts in one collective payload."""
    fused = try_count_payload(ids, num_experts)
    if fused is not None:
        return fused
    return _torch_replica_count_payload(ids, num_experts)


def _torch_replica_count_payload(ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Validate and count exactly when the optional native counter is unavailable."""
    ordered = (ids.float() if num_experts <= 2**24 else ids).sort(dim=1).values
    invalid = ((ids < 0) | (ids >= num_experts)).any()
    invalid = invalid | (ordered[:, 1:] == ordered[:, :-1]).any()
    # The clamped IDs have a known range. A fixed integer histogram avoids
    # bincount's device min/max scalar reads while retaining exact counts.
    clamped = ids.clamp(0, num_experts - 1).flatten()
    counts = torch.zeros(num_experts, dtype=torch.int64, device=ids.device)
    counts.scatter_add_(0, clamped, torch.ones_like(clamped))
    return torch.cat((counts, invalid.reshape(1).to(counts.dtype)))


def _automatic_capacity_options(
    maximum_tokens: int, top_k: int, target_load: int | None, receive_capacity: int | None,
    factor: float | None, budget: Callable[[int], int] | None = None,
) -> tuple[int | None, int | None]:
    """Keep the preferred reservation separate from the allocator's hard receive limit."""
    reservation = (receive_capacity if factor is None else
                   align_capacity(math.ceil(maximum_tokens * top_k * factor)))
    admission = reservation
    if budget is not None:
        admission = budget(maximum_tokens)
        _integer(admission, 'Push receive capacity limit', 0)
    target = reservation if factor is not None and target_load is None else target_load
    return target, admission


def _compact_home_cost_dominates(
    totals: list[int], rows: list[list[int]], config: ExpertReplicaConfig, model: ExpertReplicaCostModel,
) -> bool:
    """Apply the existing adaptive planner's optimistic bound to compact statistics."""
    if model.critical_path is not None:
        return False
    measured = min(model.forward_ms[-1][0], model.backward_ms[-1][0])
    if max(totals) > measured:
        return False
    if model.effective_group_ms:
        home = config.home_experts
        tasks = max(sum(count > 0 for count in totals[start:start + home])
                    for start in range(0, config.num_experts, home))
        source_floor = model.effective_source_row_floor * max(row[4] for row in rows)
        received = max(max(row[7] for row in rows), source_floor)
        tokens = max((row[4] + row[7] - 2 * row[8]) / 2 for row in rows)
        fixed, row_ms, _, _, token_ms, active_ms = model.effective_group_ms
        owner = fixed * tasks + row_ms * received + token_ms * tokens
        optimistic = row_ms * max(sum(row[7] for row in rows) / config.ep_size, source_floor) + active_ms
        return owner <= optimistic + max(model.minimum_gain_ms, model.minimum_replica_gain_ms or 0)
    durations: dict[int, tuple[float, float]] = {}
    costs = []
    home = config.home_experts
    for rank, row in enumerate(rows):
        forward, backward = 0.0, 0.0
        for count in totals[rank * home:(rank + 1) * home]:
            if count not in durations:
                durations[count] = model.compute_ms(count), model.compute_ms(count, backward=True)
            forward += durations[count][0]
            backward += durations[count][1]
        # The diagonal counts are exactly the owner's locally originated rows.
        source_rows, received_rows, local_rows = row[4], row[7], row[8]
        token_phase = (source_rows + received_rows - 2 * local_rows) * model.token_ms_per_row / 2
        costs.append((forward + token_phase, backward + token_phase))
    owner = (model.forward_repeats * (max(value[0] for value in costs) + model.planner_overhead_ms)
             + max(value[1] for value in costs))
    average = sum(row[7] for row in rows) / config.ep_size
    rates = [min(duration / count for count, duration in table if count)
             for table in (model.forward_ms, model.backward_ms)]
    optimistic = (model.forward_repeats * (rates[0] * average + model.planner_overhead_ms
                                         + model.replica_overhead_ms) + rates[1] * average)
    return owner <= optimistic + max(model.minimum_gain_ms, model.minimum_replica_gain_ms or 0)


def _try_compact_home_route(
    matrix: torch.Tensor | None, topk_ids: torch.Tensor, config: ExpertReplicaConfig, group: object, rank: int,
    *, target_load: int | None, receive_capacity: int | None, factor: float | None,
    minimum_load_reduction: float, minimum_saved_rows: int,
    cost_model: ExpertReplicaCostModel | None, rank_nodes: tuple[int, ...],
    minimum_replica_rows: int = 0,
    capacity_budget: Callable[[int], int] | None = None,
    gathered_counts: _GatheredReplicaCounts | None = None,
    rejected_quota_memo: _RejectedQuotaMemo | None = None,
    complete_experts: bool = False,
) -> ReplicaRoute | _UncalibratedCandidate | None:
    """Read bounded statistics before deciding whether a full CPU table is needed."""
    # A larger allowance must not replace the original checkpoint graph with
    # a newly affordable home-only shortcut during recomputation.
    state = get_recompute_state()
    cache = None
    if state is not None and state.is_recomputing:
        cache = state.get_resource(_RecomputedReplicaPlans, _RecomputedReplicaPlans)
        if cache.cursor < len(cache.entries) and cache.entries[cache.cursor][1] is not None:
            return None
    _integer(minimum_saved_rows, "minimum_saved_rows", 0)
    admission = reduction_basis_points(minimum_load_reduction)
    if target_load is not None:
        _integer(target_load, "target_load", 0)
    nodes = _validate_rank_nodes(rank_nodes, config.ep_size)
    if cost_model is not None:
        if not isinstance(cost_model, ExpertReplicaCostModel):
            raise ValueError("cost_model must be an ExpertReplicaCostModel")
        if cost_model.ep_size != config.ep_size:
            raise ValueError("Replica cost model EP size does not match the plan")
        if nodes and cost_model._rank_nodes and nodes != cost_model._rank_nodes:
            raise ValueError("Replica topology differs from the cost model topology")
    if gathered_counts is None:
        logical = matrix[:, :config.num_experts]
        grouped = logical.reshape(config.ep_size, config.ep_size, config.home_experts).sum(2)
        loads = grouped.sum(0)
        # Wire fields: ID error, size error, tokens, autograd mask, sum, min, max,
        # owner receive rows and the owner's locally originated rows.
        summary = torch.cat((matrix[:, config.num_experts:], logical.sum(1, keepdim=True),
                             logical.amin(1, keepdim=True), logical.amax(1, keepdim=True),
                             loads[:, None], grouped.diagonal()[:, None]), dim=1)
        payload = summary.flatten() if cost_model is None else torch.cat((summary.flatten(), logical.sum(0)))
        compact = payload.cpu().tolist()
        width = summary.shape[1]
        rows = [compact[start:start + width] for start in range(0, config.ep_size * width, width)]
        totals = compact[config.ep_size * width:]
    else:
        logical = gathered_counts.counts
        if cost_model is None:
            rows = [[0] + list(row) for row in gathered_counts.source_summary]
            totals = []
        elif gathered_counts.host_counts is None:
            grouped = logical.reshape(config.ep_size, config.ep_size, config.home_experts).sum(2, dtype=torch.int64)
            compact = torch.cat((grouped.diagonal(), logical.sum(0, dtype=torch.int64))).cpu().tolist()
            diagonal, totals = compact[:config.ep_size], compact[config.ep_size:]
        else:
            host = gathered_counts.host_counts
            diagonal = [sum(row[owner * config.home_experts:(owner + 1) * config.home_experts])
                        for owner, row in enumerate(host)]
            totals = _host_column_totals(host)
        if cost_model is not None:
            # Native agreement already validated the source controls and owner loads.
            rows = [[0] + list(row) + [local] for row, local in zip(gathered_counts.source_summary, diagonal)]
    if any(row[0] for row in rows):
        raise ValueError("hot replication requires in-range, distinct expert IDs per token")
    maximum_tokens = max(row[2] for row in rows)
    top_k = topk_ids.shape[1]
    target_load, receive_capacity = _automatic_capacity_options(
        maximum_tokens, top_k, target_load, receive_capacity, factor, capacity_budget)
    upper = config.maximum_receive_rows(max(1, maximum_tokens), top_k, alignment=1)
    owner_upper = ExpertReplicaConfig(config.num_experts, config.ep_size, 0).maximum_receive_rows(
        max(1, maximum_tokens), top_k, alignment=1)
    upper = owner_upper if receive_capacity is None else min(owner_upper, max(upper, receive_capacity))
    if capacity_budget is not None:
        upper = min(upper, receive_capacity)
    destination_loads = tuple(row[7] for row in rows)
    peak = max(destination_loads)
    resident = min(upper, receive_capacity) if receive_capacity is not None else upper
    if peak > upper:
        return None
    decision = None
    balanced = min(destination_loads) == peak
    if not balanced:
        if peak > resident:
            return None
        if cost_model is None:
            average = (sum(destination_loads) + config.ep_size - 1) // config.ep_size
            if _optional_peak_is_useful(average, destination_loads, resident, admission, minimum_saved_rows):
                if gathered_counts is not None and gathered_counts.host_counts is not None:
                    totals = _host_column_totals(gathered_counts.host_counts)
                else:
                    totals = logical.sum(0, dtype=torch.int64).cpu().tolist()
                target = max(average, target_load or 0)
                rejection = _RejectedReplicaQuotas(config, tuple(totals), destination_loads,
                                                   target, admission, minimum_saved_rows, minimum_replica_rows)
                complete_search = _complete_guest_search_enabled()
                projected_nodes = _projected_rank_nodes(nodes)
                signature = (rejection, upper, resident, complete_search, nodes, projected_nodes)
                previous = None
                if cache is not None and cache.cursor < len(cache.entries):
                    previous = cache.entries[cache.cursor][0]
                # Allocator limits above the owner peak cannot change the
                # quota search or fragment pruning, whose bound is owner peak.
                replay_rejected = isinstance(previous, _RejectedReplicaQuotas) and previous.matches(
                    config, totals, destination_loads, target, admission, minimum_saved_rows, minimum_replica_rows)
                memo_rejected = rejected_quota_memo is not None and rejected_quota_memo.contains(signature)
                if not (replay_rejected or memo_rejected):
                    candidate = (_complete_admission_copies(signature) if complete_experts else
                                 _admitted_uncalibrated_copies(
                                     totals, list(destination_loads), config, upper, resident, target,
                                     minimum_replica_rows, admission, minimum_saved_rows,
                                     allow_complete_search=complete_search, projected_rank_nodes=projected_nodes))
                    if candidate is not None and any(expert // config.home_experts != rank and count
                                                     for rank, row in enumerate(candidate)
                                                     for expert, count in row.items()):
                        if rejected_quota_memo is not None:
                            rejected_quota_memo.signature = None
                        return _UncalibratedCandidate(config, tuple(totals), candidate, upper, resident,
                                                      target, minimum_replica_rows, admission, minimum_saved_rows)
                if state is None or not state.is_recomputing:
                    decision = rejection
                    if rejected_quota_memo is not None:
                        rejected_quota_memo.signature = signature
        elif not _compact_home_cost_dominates(totals, rows, config, cost_model):
            return None
    counts = logical.to(torch.int32).contiguous()
    plan = _HomeOnlyPlan(config, counts, destination_loads)
    _record_home_decision(decision)
    dynamic_summary = tuple(tuple(row[1:8]) for row in rows)
    return ReplicaRoute(plan, topk_ids, counts, rank, group, home_only=True,
                        dynamic_summary=dynamic_summary, plan_bypassed=True)


def prepare_replica_route(
    topk_ids: torch.Tensor, config: ExpertReplicaConfig, group: object = None,
    *, target_load: int | None = None, minimum_replica_rows: int = 0,
    planner_backend: str = "cpu",
    compact_home: bool = False,
    minimum_load_reduction: float = 0.0,
    minimum_saved_rows: int = 0,
    dynamic_header: tuple[int, int, int] | None = None,
    max_local_num_tokens: int | None = None,
    receive_capacity: int | None = None,
    _auto_capacity_factor: float | None = None,
    _capacity_budget: Callable[[int], int] | None = None,
    _rank_nodes: tuple[int, ...] = (),
    _cost_model: ExpertReplicaCostModel | None = None,
    _trusted_counts: torch.Tensor | None = None,
    _gathered_counts: _GatheredReplicaCounts | None = None,
    _replica_transfer_row_credit: int = 0,
    _rejected_quota_memo: _RejectedQuotaMemo | None = None,
    _count_header_cache: _ReplicaCountHeaderCache | None = None,
    _complete_experts: bool = False,
) -> ReplicaRoute:
    """Gather logical counts once, plan replicas, and preserve every TopK slot.

    Route validation is collective: invalid IDs and repeated experts are reported
    by all ranks before any sparse weight transfers. All ranks must supply the
    same Top-K width and configuration; source token counts may differ.

    Args:
        topk_ids: Integer expert IDs, with distinct experts in each token's row.
        config: Logical expert topology and per-rank replica budget.
        group: EP process group; None selects the default group.
        target_load: Optional preferred destination load, capped by the receive bound.
        minimum_replica_rows: Minimum useful row count for a guest copy.
        planner_backend: CPU or device placement implementation.
        compact_home: Keep logical IDs and counts when the global plan has no guests.
            The caller must execute that invocation with a home-only kernel topology.
        minimum_load_reduction: Required receive-peak reduction for optional copies.
        minimum_saved_rows: Required absolute receive-peak reduction for optional copies.
        dynamic_header: Optional invalid/token-count/autograd header fused into the count gather.
        max_local_num_tokens: Optional source-row reservation, validated again by the dynamic route summary.
        receive_capacity: Globally agreed resident receive rows for capacity-aware CPU cost refinement.
        _auto_capacity_factor: Internal Push reservation factor applied to the gathered actual source size.
        _capacity_budget: Internal allocator query shared by Push with and without replicas.
        _trusted_counts: Internal exact Router histogram, using the executor's trusted-count contract.
        _gathered_counts: Internal validated native count agreement for CPU planning.
        _replica_transfer_row_credit: Internal conservative activation-row credit per deleted matrix edge.
        _count_header_cache: Internal immutable control uploads owned by MX execution resources.
        _complete_experts: Internal MXFP8 training policy preserving complete expert reductions.

    Returns:
        Invocation-owned placement, physical IDs and dispatch counts.
    """
    validate_planner_backend(planner_backend, minimum_replica_rows, _cost_model)
    _validate_complete_expert_policy(_complete_experts, planner_backend)
    if receive_capacity is not None:
        _integer(receive_capacity, "receive_capacity", 0)
    if _capacity_budget is not None and not callable(_capacity_budget):
        raise ValueError('Replica capacity budget must be callable')
    if max_local_num_tokens is not None:
        _integer(max_local_num_tokens, "max_local_num_tokens", 1)
    if _auto_capacity_factor is not None and (
            isinstance(_auto_capacity_factor, bool) or not isinstance(_auto_capacity_factor, (int, float))
            or not math.isfinite(_auto_capacity_factor) or _auto_capacity_factor < 1):
        raise ValueError("Automatic replica capacity factor must be finite and at least 1")
    if topk_ids.ndim != 2 or topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must be a two-dimensional integer tensor")
    rank = dist.get_rank(group) if dist.is_initialized() else 0
    _integer(_replica_transfer_row_credit, "Replica transfer row credit", 0)
    size = dist.get_world_size(group) if dist.is_initialized() else 1
    if size != config.ep_size:
        raise ValueError("replica configuration must match the EP group size")
    tokens, top_k = topk_ids.shape
    config.maximum_receive_rows(max(1, tokens), top_k, alignment=1)
    ids = topk_ids
    if _gathered_counts is not None:
        if planner_backend != "cpu" or _gathered_counts.counts.shape != (size, config.num_experts):
            raise ValueError("Gathered replica counts require a matching CPU planner topology")
    elif _trusted_counts is None:
        payload = _replica_count_payload(ids.to(torch.int64), config.num_experts)
        if dynamic_header is not None:
            payload = torch.cat((payload, payload.new_tensor(dynamic_header)))
    else:
        if (_trusted_counts.shape != (config.num_experts,) or _trusted_counts.device != ids.device
                or _trusted_counts.dtype not in (torch.int32, torch.int64)):
            raise ValueError("Trusted replica counts must be an integer expert histogram on the route device")
        header = (0,) + (() if dynamic_header is None else dynamic_header)
        counts = _trusted_counts.to(torch.int64)
        upload = _async_device_bytes if _count_header_cache is None else _count_header_cache.upload
        descriptor = upload(struct.pack(f"<{len(header)}q", *header), ids.device).view(torch.int64)
        payload = torch.cat((counts, descriptor))
    if _gathered_counts is None:
        matrix = _gather_replica_counts(payload, size, group)
        logical_counts = matrix[:, :config.num_experts]
    else:
        logical_counts = _gathered_counts.counts
    if planner_backend == "device":
        source_summary = None
        if dynamic_header is not None:
            summary = torch.cat((matrix[:, config.num_experts + 1:], logical_counts.sum(1, keepdim=True),
                                 logical_counts.amin(1, keepdim=True), logical_counts.amax(1, keepdim=True)), dim=1)
            source_summary = summary.cpu().tolist()
        maximum_tokens = (max(row[1] for row in source_summary) if source_summary is not None
                          else int(logical_counts.sum(dim=1).max()) // top_k)
        target_load, receive_capacity = _automatic_capacity_options(
            maximum_tokens, top_k, target_load, receive_capacity, _auto_capacity_factor, _capacity_budget)
        upper = config.maximum_receive_rows(max(1, maximum_tokens), top_k, alignment=1)
        if _capacity_budget is not None:
            upper = min(upper, receive_capacity)
        device_plan = build_device_expert_replica_plan(
            logical_counts, config, capacity_limit=upper,
            target_load=min(upper, target_load) if target_load is not None else None,
            minimum_replica_rows=minimum_replica_rows, minimum_load_reduction=minimum_load_reduction,
            minimum_saved_rows=minimum_saved_rows)
        device_plan.control[:1].bitwise_or_(matrix[:, config.num_experts].any().to(torch.int64))
        plan = device_plan.host_summary()
        dynamic_summary = (None if source_summary is None else
                           tuple(tuple(row) + (load,) for row, load in zip(source_summary, plan.destination_loads)))
        if compact_home and not plan.transfers:
            return ReplicaRoute(plan, topk_ids, logical_counts.to(torch.int32).contiguous(),
                                rank, group, device_plan=device_plan, home_only=True, dynamic_summary=dynamic_summary)
        slots, lengths = device_plan.source_runs(rank)
        remapped = _remap_replica_ids(ids.to(torch.int64), plan, slots, lengths)
        return ReplicaRoute(plan, remapped.to(topk_ids.dtype), device_plan.dispatch_counts.to(torch.int32),
                            rank, group, device_plan=device_plan, dynamic_summary=dynamic_summary)
    prepared_candidate = None
    if (compact_home and dynamic_header is not None and config.replica_slots_per_rank
            and (_gathered_counts is None or _cost_model is not None
                 or minimum_load_reduction or minimum_saved_rows)):
        ordinary = _try_compact_home_route(
            matrix if _gathered_counts is None else None, topk_ids, config, group, rank,
            target_load=target_load, receive_capacity=receive_capacity,
            factor=_auto_capacity_factor, minimum_load_reduction=minimum_load_reduction,
            minimum_saved_rows=minimum_saved_rows,
            minimum_replica_rows=minimum_replica_rows,
            cost_model=_cost_model, rank_nodes=_rank_nodes, capacity_budget=_capacity_budget,
            gathered_counts=_gathered_counts, rejected_quota_memo=_rejected_quota_memo,
            complete_experts=_complete_experts)
        if isinstance(ordinary, ReplicaRoute):
            return ordinary
        prepared_candidate = ordinary
    # The deterministic host planner shares the route-count synchronization with
    # split sizing; no per-expert device-to-host reads occur below.
    if _gathered_counts is None:
        host = matrix.cpu().tolist()
        if any(row[config.num_experts] for row in host):
            raise ValueError("hot replication requires in-range, distinct expert IDs per token")
        logical_host = [row[:config.num_experts] for row in host]
    else:
        logical_host = (_gathered_counts.host_counts if _gathered_counts.host_counts is not None
                        else logical_counts.cpu().tolist())
    # Every planner must use the same bound when source ranks have unequal or zero rows.
    maximum_tokens = max(sum(row) for row in logical_host) // top_k
    target_load, receive_capacity = _automatic_capacity_options(
        maximum_tokens, top_k, target_load, receive_capacity, _auto_capacity_factor, _capacity_budget)
    upper = config.maximum_receive_rows(max(1, maximum_tokens), top_k, alignment=1)
    if config.replica_slots_per_rank:
        owner_upper = ExpertReplicaConfig(config.num_experts, config.ep_size, 0).maximum_receive_rows(
            max(1, maximum_tokens), top_k, alignment=1)
        upper = (owner_upper if receive_capacity is None else
                 min(owner_upper, max(upper, receive_capacity)))
    if _capacity_budget is not None:
        upper = min(upper, receive_capacity)
    plan, reused = _plan_replica_route(
        logical_host, config, group, rank,
        _complete_experts=_complete_experts,
        target_load=min(upper, target_load) if target_load is not None else None,
        minimum_replica_rows=minimum_replica_rows, capacity_limit=upper, cost_model=_cost_model,
        minimum_load_reduction=minimum_load_reduction,
        minimum_saved_rows=minimum_saved_rows,
        receive_capacity=receive_capacity, adaptive=bool(config.replica_slots_per_rank),
        _rank_nodes=_rank_nodes,
        _replica_transfer_row_credit=_replica_transfer_row_credit,
        _uncalibrated_candidate=prepared_candidate,
    )
    if max(plan.destination_loads) > upper:
        raise RuntimeError("replica planner exceeded the theoretical receive bound")
    # Distinct logical Top-K counts bound every physical split, including home-only plans.
    dynamic_summary = None
    if dynamic_header is not None:
        if _gathered_counts is None:
            dynamic_summary = tuple(
                tuple(row[config.num_experts + 1:]) + (sum(counts), min(counts), max(counts), load)
                for row, counts, load in zip(host, logical_host, plan.destination_loads))
        else:
            # Native agreement already validated these source controls and
            # statistics. Only destination loads change with replica placement.
            dynamic_summary = tuple(tuple(row[:6]) + (load,) for row, load in
                                    zip(_gathered_counts.source_summary, plan.destination_loads))
    if compact_home and not plan.transfers:
        return ReplicaRoute(plan, topk_ids, _home_route_counts(plan, rank, logical_counts),
                            rank, group, home_only=True, dynamic_summary=dynamic_summary, plan_reused=reused,
                            _host_counts=tuple(map(tuple, logical_host)))
    slots, lengths, device_counts = _upload_route_metadata(plan, rank, ids.device)
    remapped = _remap_replica_ids(ids.to(torch.int64), plan, slots, lengths)
    return ReplicaRoute(plan, remapped.to(topk_ids.dtype), device_counts, rank, group,
                        dynamic_summary=dynamic_summary, plan_reused=reused, _host_counts=plan.dispatch_counts)
