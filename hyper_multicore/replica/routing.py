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
"""Invocation-owned, lossless CPU replica routing and exact checkpoint replay."""

from __future__ import annotations

import math
import struct
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache
import os
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ._metadata import _async_device_bytes, _recompute_metadata
from .admission import reduction_basis_points
from .capacity import ExpertReplicaConfig, _integer, align_capacity
from .plan import ExpertExecutionPlan, ExpertReplicaTransfer
from .planner import build_expert_replica_plan
from ._selection import _UncalibratedCandidate, _admitted_uncalibrated_copies, _optional_peak_is_useful
from .host import _column_totals
from .count_kernel import try_count_payload
from .topology import _validate_rank_nodes
from .recompute import get_recompute_state

_HostReplicaCounts = list[list[int]] | np.ndarray
_SEVERE_OWNER_SKEW = 3


@lru_cache(maxsize=1)
def _hot_replay_readback_enabled() -> bool:
    """Select the exact checkpoint readback experiment independently of upload reuse."""
    value = os.environ.get("HYPER_MULTICORE_MOE_HOT_REPLAY_READBACK", "0")
    if value not in ("0", "1"):
        raise ValueError("HYPER_MULTICORE_MOE_HOT_REPLAY_READBACK must be 0 or 1")
    return value == "1"


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
        return tuple(totals[start : start + home] + padding for start in range(0, self.config.num_experts, home))


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

    def matches(
        self,
        config: ExpertReplicaConfig,
        totals: list[int],
        loads: tuple[int, ...],
        target: int,
        admission: int,
        saved_rows: int,
        minimum_rows: int,
    ) -> bool:
        """Require the same quota inputs and benefit policy after fresh count agreement."""
        return (
            self.config == config
            and self.totals == tuple(totals)
            and self.loads == loads
            and (self.target, self.admission, self.saved_rows, self.minimum_rows)
            == (target, admission, saved_rows, minimum_rows)
        )


def _read_gathered_host_counts(counts: torch.Tensor) -> _HostReplicaCounts:
    """Keep the original INT32 readback unboxed when a checkpoint can replay it."""
    host = counts.cpu()
    if counts.dtype == torch.int32 and _hot_replay_readback_enabled() and get_recompute_state() is not None:
        return host.numpy()
    return host.tolist()


def _host_column_totals(counts: _HostReplicaCounts) -> list[int]:
    """Use the existing host agreement without freezing or reading device counts again."""
    if isinstance(counts, np.ndarray):
        return counts.sum(axis=0, dtype=np.int64).tolist()
    totals = _column_totals(counts)
    return list(map(sum, zip(*counts))) if totals is None else totals


@dataclass(frozen=True)
class _GatheredReplicaCounts:
    """Borrow the executor's already validated count exchange."""

    counts: torch.Tensor
    source_summary: list[list[int]]
    host_counts: _HostReplicaCounts | None = None


@dataclass(frozen=True)
class ReplicaRoute:
    """Retain one immutable execution plan and current token remapping."""

    plan: ExpertExecutionPlan | _HomeOnlyPlan
    physical_ids: torch.Tensor
    counts_by_source: torch.Tensor
    rank: int
    group: object
    transport: object | None = None
    home_only: bool = False
    dynamic_summary: tuple[tuple[int, ...], ...] | None = None
    plan_reused: bool = False
    plan_bypassed: bool = False
    _host_counts: tuple[tuple[int, ...], ...] | None = None


class _RecomputedReplicaPlans:
    """Retain immutable CPU plans and home-decision evidence for one invocation."""

    def __init__(self) -> None:
        """Initialize ordered call records without retaining any device tensors."""
        self.entries: list[tuple[Any, ExpertExecutionPlan | None]] = []
        self.binary_counts: dict[int, bytes] = {}
        self.cursor = 0

    def clear(self) -> None:
        """Release all plans after complete or interrupted recomputation."""
        self.entries.clear()
        self.binary_counts.clear()
        self.cursor = 0

    def lookup(self, signature: Any, counts: tuple[tuple[int, ...], ...] | np.ndarray) -> ExpertExecutionPlan | None:
        """Advance in invocation order and reject changed counts or static policy."""
        if self.cursor >= len(self.entries):
            return None
        expected, plan = self.entries[self.cursor]
        self.cursor += 1
        if plan is None or signature != expected:
            return None
        if isinstance(counts, np.ndarray):
            saved = self.binary_counts.get(id(plan))
            if saved is not None and counts.dtype == np.int32:
                if counts.shape != (plan.config.ep_size, plan.config.num_experts):
                    return None
                return plan if counts.tobytes() == saved else None
            counts = tuple(map(tuple, counts.tolist()))
        return plan if counts == plan.logical_counts else None


def _has_home_recompute_hint() -> bool:
    """Hint ordinary-route overlap without trusting or consuming fresh counts."""
    state = get_recompute_state()
    if state is None or not state.is_recomputing:
        return False
    cache = state.get_resource(_RecomputedReplicaPlans, _RecomputedReplicaPlans)
    return cache.cursor < len(cache.entries) and cache.entries[cache.cursor][1] is None


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


def _plan_signature(config, group, rank, options):
    """Exclude only invocation-local capacity and admitted-quota observations."""
    policy = tuple(
        (
            (key, type(value), value)
            for key, value in sorted(options.items())
            if key not in ("receive_capacity", "capacity_limit", "transfer_row_credit", "_uncalibrated_candidate")
        )
    )
    signature = (config, id(group), rank, policy)
    return signature


def _plan_replica_route(
    counts: Any, config: ExpertReplicaConfig, group: object, rank: int, **options: Any
) -> tuple[ExpertExecutionPlan, bool]:
    """Replay the original decision after the usual count exchange validates current inputs."""
    state = get_recompute_state()
    binary = isinstance(counts, np.ndarray)
    if state is None:
        if binary:
            counts = counts.tolist()
        return (build_expert_replica_plan(counts, config.replica_slots_per_rank, **options), False)
    logical = counts if binary else tuple((tuple(row) for row in counts))
    signature = _plan_signature(config, group, rank, options)
    cache = state.get_resource(_RecomputedReplicaPlans, _RecomputedReplicaPlans)
    if state.is_recomputing:
        plan = cache.lookup(signature, logical)
        limit = options.get("capacity_limit")
        if plan is not None and (limit is None or max(plan.destination_loads) <= limit):
            return (plan, True)
    if binary:
        logical = tuple(map(tuple, counts.tolist()))
    plan = build_expert_replica_plan(logical, config.replica_slots_per_rank, **options)
    if not state.is_recomputing:
        cache.entries.append((signature, plan))
        if binary and counts.dtype == np.int32 and plan.transfers:
            cache.binary_counts[id(plan)] = counts.tobytes()
    return (plan, False)


def stable_expert_order(ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Return stable integer permutation indices using exact expert sorting keys.

    Args:
        ids: Integer expert identifiers.
        num_experts: Upper bound determining whether FP32 keys are exact.
    """
    return torch.argsort(ids.float() if num_experts <= 2**24 else ids, stable=True)


def _upload_route_metadata(
    plan: ExpertExecutionPlan, rank: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        if counts is logical_counts:
            counts = counts.clone()
        cache.remember(plan, rank, logical_counts.device, "home", (counts,))
    return counts


def _remap_replica_ids(
    ids: torch.Tensor, plan: ExpertExecutionPlan, slots: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Consume current stable occurrences without caching token permutations."""
    if not plan.transfers:
        return (
            ids + torch.div(ids, plan.config.home_experts, rounding_mode="floor") * plan.config.replica_slots_per_rank
        )
    destinations = torch.repeat_interleave(slots, lengths, output_size=ids.numel())
    order = stable_expert_order(ids.flatten(), plan.config.num_experts)
    remapped = torch.empty_like(ids.flatten())
    remapped.scatter_(0, order, destinations)
    return remapped.reshape_as(ids)


def _gather_replica_counts(payload: torch.Tensor, size: int, group: object) -> torch.Tensor:
    """Own the gathered count matrix and wait before any control reads."""
    gathered = torch.empty((size, payload.numel()), dtype=payload.dtype, device=payload.device)
    if size > 1:
        dist.all_gather_into_tensor(gathered.flatten(), payload, group=group, async_op=True).wait()
    else:
        gathered[0].copy_(payload)
    return gathered


def _replica_count_payload(ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Keep route validation and logical counts in one collective payload."""
    fused = try_count_payload(ids, num_experts)
    if fused is not None:
        return fused
    return _torch_replica_count_payload(ids, num_experts)


def _automatic_capacity_options(
    maximum_tokens: int,
    top_k: int,
    target_load: int | None,
    receive_capacity: int | None,
    factor: float | None,
    budget: Callable[[int], int] | None = None,
) -> tuple[int | None, int | None]:
    """Keep the soft reservation separate from the actual allocator allowance."""
    reservation = receive_capacity if factor is None else align_capacity(math.ceil(maximum_tokens * top_k * factor))
    allowance = reservation if budget is None else budget(maximum_tokens)
    if allowance is not None:
        _integer(allowance, "Push receive capacity limit", 0)
    return (reservation if factor is not None and target_load is None else target_load), allowance


def _validate_route_options(
    ids: torch.Tensor,
    config: ExpertReplicaConfig,
    group: object,
    minimum_rows: int,
    reduction: float,
    factor: float | None,
) -> tuple[int, int]:
    """Validate metadata before collective routing without scalar device reads."""
    if ids.ndim != 2 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must be a two-dimensional integer tensor")
    _integer(minimum_rows, "minimum_replica_rows", 0)
    reduction_basis_points(reduction)
    if factor is not None and (
        isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor < 1
    ):
        raise ValueError("Automatic replica capacity factor must be finite and at least 1")
    rank = dist.get_rank(group) if dist.is_initialized() else 0
    size = dist.get_world_size(group) if dist.is_initialized() else 1
    if size != config.ep_size:
        raise ValueError("replica configuration must match the EP group size")
    config.maximum_receive_rows(max(1, ids.shape[0]), ids.shape[1], alignment=1)
    return rank, size


def _route_counts(
    ids: torch.Tensor,
    config: ExpertReplicaConfig,
    group: object,
    header: tuple[int, int, int] | None,
    trusted: torch.Tensor | None,
    gathered: _GatheredReplicaCounts | None,
    matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list, list]:
    """Reuse a validated agreement, or gather IDs/counts and controls once."""
    if gathered is not None:
        if gathered.counts.shape != (config.ep_size, config.num_experts):
            raise ValueError("Gathered replica counts require a matching planner topology")
        host = gathered.host_counts if gathered.host_counts is not None else _read_gathered_host_counts(gathered.counts)
        return gathered.counts, host, [list(row[:6]) for row in gathered.source_summary]
    if matrix is None:
        payload = _source_count_payload(ids, config, trusted, header)
        matrix = _gather_replica_counts(payload, config.ep_size, group)
    host = matrix.cpu().tolist()
    if any(row[config.num_experts] for row in host):
        raise ValueError("hot replication requires in-range, distinct expert IDs per token")
    counts = [row[: config.num_experts] for row in host]
    controls = (
        []
        if header is None
        else [row[config.num_experts + 1 :] + [sum(count), min(count), max(count)] for row, count in zip(host, counts)]
    )
    return matrix[:, : config.num_experts], counts, controls


def _source_count_payload(ids, config, trusted, header):
    """Keep trusted-count validation identical before compact and full agreement."""
    if trusted is None:
        payload = _replica_count_payload(ids.to(torch.int64), config.num_experts)
    else:
        if trusted.shape != (config.num_experts,) or trusted.device != ids.device:
            raise ValueError("Trusted counts must match the logical experts and route device")
        if trusted.dtype not in (torch.int32, torch.int64):
            raise ValueError("Trusted replica counts must be an integer histogram")
        payload = torch.cat((trusted.to(torch.int64), trusted.new_zeros(1, dtype=torch.int64)))
    return payload if header is None else torch.cat((payload, payload.new_tensor(header)))


def _validate_source_controls(controls: list, top_k: int) -> None:
    """Reject inconsistent group participation before any sparse weight transfer."""
    if not controls:
        return
    if any(row[0] or row[1] < 0 for row in controls):
        raise RuntimeError("Replica route exceeds native size limits")
    if any(row[2] != controls[0][2] for row in controls):
        raise RuntimeError("Replica autograd participation differs across EP ranks")
    if any(row[3] != row[1] * top_k or row[4] < 0 or row[5] > row[1] for row in controls):
        raise ValueError("Replica counts must describe the actual distinct Top-K routes")


def _prepare_compact_candidate(
    topk_ids,
    config,
    group,
    rank,
    trusted,
    gathered,
    *,
    dynamic_header,
    target_load,
    receive_capacity,
    factor,
    minimum_load_reduction,
    minimum_saved_rows,
    rank_nodes,
    minimum_replica_rows,
    capacity_budget,
):
    """Gather at most once and retain admitted quotas for full source materialization."""
    if config.replica_slots_per_rank >= 2:
        gathered = _prepare_severe_skew_counts(gathered, config, minimum_load_reduction, minimum_saved_rows)
    matrix = None
    if gathered is None:
        payload = _source_count_payload(topk_ids, config, trusted, dynamic_header)
        matrix = _gather_replica_counts(payload, config.ep_size, group)
    ordinary = _try_compact_home_route(
        matrix,
        topk_ids,
        config,
        group,
        rank,
        gathered_counts=gathered,
        target_load=target_load,
        receive_capacity=receive_capacity,
        factor=factor,
        minimum_load_reduction=minimum_load_reduction,
        minimum_saved_rows=minimum_saved_rows,
        rank_nodes=rank_nodes,
        minimum_replica_rows=minimum_replica_rows,
        capacity_budget=capacity_budget,
    )
    return matrix, ordinary, gathered


def _prepare_severe_skew_counts(gathered, config, minimum_load_reduction, minimum_saved_rows):
    """Read a severe multi-guest table once for both admission and materialization."""
    if gathered is None or gathered.host_counts is not None or config.replica_slots_per_rank < 2:
        return gathered
    if not (minimum_load_reduction or minimum_saved_rows):
        return gathered
    if gathered.counts.shape != (config.ep_size, config.num_experts):
        raise ValueError("Gathered replica counts require a matching planner topology")
    loads = tuple(row[6] for row in gathered.source_summary)
    total = sum(loads)
    if not total or max(loads) * config.ep_size < _SEVERE_OWNER_SKEW * total:
        return gathered
    average = (total + config.ep_size - 1) // config.ep_size
    if not _optional_peak_is_useful(
        average, loads, max(loads), reduction_basis_points(minimum_load_reduction), minimum_saved_rows
    ):
        return gathered
    return replace(gathered, host_counts=_read_gathered_host_counts(gathered.counts))


def prepare_replica_route(
    topk_ids: torch.Tensor,
    config: ExpertReplicaConfig,
    group: object = None,
    *,
    target_load: int | None = None,
    minimum_replica_rows: int = 0,
    compact_home: bool = False,
    minimum_load_reduction: float = 0.0,
    minimum_saved_rows: int = 0,
    dynamic_header: tuple[int, int, int] | None = None,
    receive_capacity: int | None = None,
    _auto_capacity_factor: float | None = None,
    _capacity_budget: Callable[[int], int] | None = None,
    _rank_nodes: tuple[int, ...] = (),
    _trusted_counts: torch.Tensor | None = None,
    _gathered_counts: _GatheredReplicaCounts | None = None,
    _replica_transfer_row_credit: int = 0,
) -> ReplicaRoute:
    """Gather exact counts, solve once and remap this invocation's current TopK IDs.

    Args:
        topk_ids: Current distinct logical expert IDs for every token.
        config: Logical ownership and fixed guest budget.
        group: Explicit EP process group, or WORLD.
        target_load: Soft desired receive rows.
        minimum_replica_rows: Soft minimum rows per copied matrix.
        compact_home: Select ordinary logical IDs if the plan has no transfers.
        minimum_load_reduction: Required peak reduction for optional copies.
        minimum_saved_rows: Required absolute peak reduction.
        dynamic_header: Invocation size/token/autograd controls fused with counts.
        receive_capacity: Agreed receive reservation.
        _auto_capacity_factor: Executor-owned reservation factor.
        _capacity_budget: Executor-owned hard allowance callback.
        _rank_nodes: Actual immutable node labels.
        _trusted_counts: Router-owned exact histogram.
        _gathered_counts: Already validated native count agreement.
        _replica_transfer_row_credit: Runtime-derived mandatory-gradient row credit.

    Returns:
        Owned plan, current mapping and integer execution metadata.
    """
    rank, _ = _validate_route_options(
        topk_ids, config, group, minimum_replica_rows, minimum_load_reduction, _auto_capacity_factor
    )
    _integer(_replica_transfer_row_credit, "Replica transfer row credit", 0)
    matrix, candidate = (None, None)
    if compact_home and dynamic_header is not None and config.replica_slots_per_rank:
        matrix, ordinary, _gathered_counts = _prepare_compact_candidate(
            topk_ids,
            config,
            group,
            rank,
            _trusted_counts,
            _gathered_counts,
            target_load=target_load,
            receive_capacity=receive_capacity,
            factor=_auto_capacity_factor,
            minimum_load_reduction=minimum_load_reduction,
            minimum_saved_rows=minimum_saved_rows,
            minimum_replica_rows=minimum_replica_rows,
            rank_nodes=_rank_nodes,
            capacity_budget=_capacity_budget,
            dynamic_header=dynamic_header,
        )
        if isinstance(ordinary, ReplicaRoute):
            return ordinary
        candidate = ordinary
    logical, host, controls = _route_counts(
        topk_ids, config, group, dynamic_header, _trusted_counts, _gathered_counts, matrix
    )
    _validate_source_controls(controls, topk_ids.shape[1])
    maximum = max(row[1] for row in controls) if controls else max(map(sum, host)) // topk_ids.shape[1]
    target, allowance = _automatic_capacity_options(
        maximum, topk_ids.shape[1], target_load, receive_capacity, _auto_capacity_factor, _capacity_budget
    )
    upper = config.maximum_receive_rows(max(1, maximum), topk_ids.shape[1], alignment=1)
    owner = ExpertReplicaConfig(config.num_experts, config.ep_size, 0).maximum_receive_rows(
        max(1, maximum), topk_ids.shape[1], alignment=1
    )
    upper = owner if allowance is None else min(owner, max(upper, allowance))
    if _capacity_budget is not None:
        upper = min(upper, allowance)
    plan, reused = _plan_replica_route(
        host,
        config,
        group,
        rank,
        target_load=min(upper, target) if target is not None else None,
        minimum_replica_rows=minimum_replica_rows,
        capacity_limit=upper,
        receive_capacity=allowance,
        minimum_load_reduction=minimum_load_reduction,
        minimum_saved_rows=minimum_saved_rows,
        rank_nodes=_rank_nodes,
        transfer_row_credit=_replica_transfer_row_credit,
        _uncalibrated_candidate=candidate,
    )
    summary = tuple((tuple(row) + (load,) for row, load in zip(controls, plan.destination_loads))) if controls else None
    if compact_home and (not plan.transfers):
        return ReplicaRoute(
            plan,
            topk_ids,
            _home_route_counts(plan, rank, logical),
            rank,
            group,
            home_only=True,
            dynamic_summary=summary,
            plan_reused=reused,
            _host_counts=plan.logical_counts,
        )
    slots, lengths, counts = _upload_route_metadata(plan, rank, topk_ids.device)
    remapped = _remap_replica_ids(topk_ids.to(torch.int64), plan, slots, lengths)
    return ReplicaRoute(
        plan,
        remapped.to(topk_ids.dtype),
        counts,
        rank,
        group,
        dynamic_summary=summary,
        plan_reused=reused,
        _host_counts=plan.dispatch_counts,
    )


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


def _compact_source_summary(matrix, config, gathered):
    """Read bounded validation and owner statistics before freezing a full table."""
    if gathered is not None:
        if gathered.counts.shape != (config.ep_size, config.num_experts):
            raise ValueError("Gathered replica counts require a matching planner topology")
        return gathered.counts, [[0] + list(row) for row in gathered.source_summary]
    logical = matrix[:, : config.num_experts]
    grouped = logical.reshape(config.ep_size, config.ep_size, config.home_experts).sum(2)
    summary = torch.cat(
        (
            matrix[:, config.num_experts :],
            logical.sum(1, keepdim=True),
            logical.amin(1, keepdim=True),
            logical.amax(1, keepdim=True),
            grouped.sum(0)[:, None],
            grouped.diagonal()[:, None],
        ),
        dim=1,
    )
    return logical, summary.cpu().tolist()


def _compact_capacity(config, rows, top_k, target, receive, factor, budget):
    """Use the same theoretical and allocator bounds as full source planning."""
    maximum = max(row[2] for row in rows)
    target, receive = _automatic_capacity_options(maximum, top_k, target, receive, factor, budget)
    upper = config.maximum_receive_rows(max(1, maximum), top_k, alignment=1)
    owner = ExpertReplicaConfig(config.num_experts, config.ep_size, 0).maximum_receive_rows(
        max(1, maximum), top_k, alignment=1
    )
    upper = owner if receive is None else min(owner, max(upper, receive))
    if budget is not None:
        upper = min(upper, receive)
    resident = min(upper, receive) if receive is not None else upper
    return target, upper, resident


def _compact_candidate(
    logical, gathered, config, loads, target, upper, resident, minimum_rows, admission, saved_rows, previous
):
    """Solve source-independent quotas once, retaining rejected replay evidence."""
    totals = (
        _host_column_totals(gathered.host_counts)
        if gathered is not None and gathered.host_counts is not None
        else logical.sum(0, dtype=torch.int64).cpu().tolist()
    )
    if isinstance(previous, _RejectedReplicaQuotas) and previous.matches(
        config, totals, loads, target, admission, saved_rows, minimum_rows
    ):
        return None, totals
    copies = _admitted_uncalibrated_copies(
        totals, list(loads), config, upper, resident, target, minimum_rows, admission, saved_rows
    )
    active = copies is not None and any(
        expert // config.home_experts != rank and count
        for rank, row in enumerate(copies)
        for expert, count in row.items()
    )
    candidate = (
        _UncalibratedCandidate(
            config, tuple(totals), copies, upper, resident, target, minimum_rows, admission, saved_rows
        )
        if active
        else None
    )
    return candidate, totals


def _compact_replay_entry(state):
    """Preserve forward graph selection even if a recompute has a larger allowance."""
    if state is None or not state.is_recomputing:
        return False, None
    cache = state.get_resource(_RecomputedReplicaPlans, _RecomputedReplicaPlans)
    entry = cache.entries[cache.cursor] if cache.cursor < len(cache.entries) else (None, None)
    return entry[1] is not None, entry[0]


def _try_compact_home_route(
    matrix: torch.Tensor | None,
    topk_ids: torch.Tensor,
    config: ExpertReplicaConfig,
    group: object,
    rank: int,
    *,
    target_load: int | None,
    receive_capacity: int | None,
    factor: float | None,
    minimum_load_reduction: float,
    minimum_saved_rows: int,
    rank_nodes: tuple[int, ...],
    minimum_replica_rows: int = 0,
    capacity_budget: Callable[[int], int] | None = None,
    gathered_counts: _GatheredReplicaCounts | None = None,
) -> ReplicaRoute | _UncalibratedCandidate | None:
    """Keep rejected no-table routes compact and pass admitted quotas to materialization."""
    state = get_recompute_state()
    physical_forward, previous = _compact_replay_entry(state)
    if physical_forward:
        return None
    _integer(minimum_saved_rows, "minimum_saved_rows", 0)
    admission = reduction_basis_points(minimum_load_reduction)
    if target_load is not None:
        _integer(target_load, "target_load", 0)
    _validate_rank_nodes(rank_nodes, config.ep_size)
    logical, rows = _compact_source_summary(matrix, config, gathered_counts)
    if any(row[0] for row in rows):
        raise ValueError("hot replication requires in-range, distinct expert IDs per token")
    _validate_source_controls([row[1:7] for row in rows], topk_ids.shape[1])
    target, upper, resident = _compact_capacity(
        config, rows, topk_ids.shape[1], target_load, receive_capacity, factor, capacity_budget
    )
    loads = tuple(row[7] for row in rows)
    peak = max(loads)
    if not _compact_home_fits(loads, upper, resident):
        return None
    decision = None
    average = (sum(loads) + config.ep_size - 1) // config.ep_size
    if min(loads) != peak and _optional_peak_is_useful(average, loads, resident, admission, minimum_saved_rows):
        target = max(average, target or 0)
        candidate, totals = _compact_candidate(
            logical,
            gathered_counts,
            config,
            loads,
            target,
            upper,
            resident,
            minimum_replica_rows,
            admission,
            minimum_saved_rows,
            previous,
        )
        if candidate is not None:
            return candidate
        if state is not None and not state.is_recomputing:
            decision = _RejectedReplicaQuotas(
                config, tuple(totals), loads, target, admission, minimum_saved_rows, minimum_replica_rows
            )
    counts = logical.to(torch.int32).contiguous()
    _record_home_decision(decision)
    return ReplicaRoute(
        _HomeOnlyPlan(config, counts, loads),
        topk_ids,
        counts,
        rank,
        group,
        home_only=True,
        dynamic_summary=tuple(tuple(row[1:8]) for row in rows),
        plan_bypassed=True,
    )


def _compact_home_fits(loads, upper, resident):
    """Retain hard receive limits and mandatory-capacity replicas before bypass."""
    peak = max(loads)
    return peak <= upper and (min(loads) == peak or peak <= resident)
