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
"""Measured execution costs for bounded, deterministic replica quota refinement."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field
from functools import cached_property

from .capacity import ExpertReplicaConfig, _integer
from .critical_path import DeviceCriticalPathCalibration
from .plan import ExpertExecutionPlan
from .topology import _validate_rank_nodes
from .host import _ordered_compute, _rank_compute


def _nonnegative(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")


def _default_schedule(backend: str, transport: str) -> str:
    """Identify the original calibrated queue contract for one backend."""
    if backend == 'native':
        return 'native_grouped_v1'
    return {'shmem_signal_kernel_gradient_bands': 'fixed_queue_gradient_bands_v1',
            'shmem_signal_kernel_gradient': 'fixed_queue_w13_first_v1'}.get(transport, 'fixed_queue_v1')


def _effective_group_features(copies: list[dict[int, int]], counts: tuple[tuple[int, ...], ...],
                              home: int, nodes: tuple[int, ...],
                              source_totals: list[int] | None = None,
                              source_row_floor: float = 0., token_work: bool = True) -> tuple[float, ...]:
    """Summarize group component maxima without treating synchronized rank waits as local work."""
    if len(nodes) != len(copies):
        raise ValueError('Effective group scoring requires the measured EP topology')
    if source_totals is None:
        source_totals = [sum(row) for row in counts]
    local, remote = [0] * len(copies), [0] * len(copies)
    for target, experts in enumerate(copies):
        for expert, rows in experts.items():
            owner = expert // home
            if rows and owner != target:
                edges = local if nodes[owner] == nodes[target] else remote
                edges[owner] += 1
                edges[target] += 1
    tasks, received, tokens = [], [], []
    for rank, experts in enumerate(copies):
        tasks.append(sum(rows > 0 for rows in experts.values()))
        received.append(sum(experts.values()))
        if token_work:
            hits = sum(min(counts[rank][expert], rows) for expert, rows in experts.items())
            tokens.append((source_totals[rank] + received[-1] - 2 * hits) / 2)
    received_max = max(max(received), source_row_floor * max(source_totals))
    return max(tasks), received_max, max(local), max(remote), max(tokens, default=0.), int(any(local) or any(remote))


@dataclass(frozen=True)
class ExpertReplicaCostModel:
    """Offline-calibrated millisecond costs for one shape and execution backend.

    Args:
        calibration_version: Explicit schema version; version 2 uses exposed communication
            and separate forward/backward rank maxima. Old calibrations must be regenerated.
        backend: Native HCCL execution or MegaMoe push/pull.
        hidden_size: Calibrated hidden dimension.
        intermediate_size: Calibrated SwiGLU intermediate dimension.
        ep_size: Calibrated expert-parallel group size.
        transport: Calibrated weight/gradient transport mode; native requires p2p.
        forward_ms: Increasing (rows, milliseconds) knots for one expert forward.
        backward_ms: Increasing knots for one expert backward with FP32 partials.
        weight_ms: Full weight-copy cost per replica, charged in each direction.
        gradient_ms: Exposed gradient-return cost per replica after local compute.
        token_ms_per_row: Dispatch/combine cost per remote send or receive row.
        schedule: Validated schedule identity, inferred from backend/transport if omitted.
        gradient_dtype: Weight partial precision; only float32 calibrations are accepted.
        forward_repeats: Number of forward executions, including checkpoint replay.
        replica_overhead_ms: Exposed replica-only scheduling budget per forward.
            Excludes host planning work already performed for every candidate.
        planner_overhead_ms: Common host budget per forward, including owner-only
            candidates. This invariant budget must not bias replica admission.
        quota_search_budget_ms: Measured host budget for attempting finer quotas.
        minimum_gain_ms: Minimum predicted worst-rank gain, including measurement
            uncertainty and the budget for extra host decision work.
        backward_weight_ms: Measured backward weight-copy cost per replica. None
            retains the forward copy cost; zero represents verified resident reuse.
        minimum_replica_gain_ms: Required gain over a feasible owner-only plan.
            None preserves legacy refinement without a separate admission margin.
        intra_node_weight_ms: Optional measured forward-copy cost between ranks on one node.
        inter_node_weight_ms: Optional measured forward-copy cost between different nodes.
        intra_node_gradient_ms: Optional exposed gradient-return cost on one node.
        inter_node_gradient_ms: Optional exposed gradient-return cost across nodes.
        intra_node_backward_weight_ms: Optional nonresident backward-copy cost on one node.
        inter_node_backward_weight_ms: Optional nonresident backward-copy cost across nodes.
        weight_fanout_ms: Optional (copies, local_ms, remote_ms) knots for one expert's
            outgoing forward copies. Different experts and incoming copies are charged
            separately. Mixed links use a conservative weighted combination of endpoints.
        _rank_nodes: Internal immutable EP-rank node labels, bound during runtime setup.
        critical_path: Optional exact-signature device dependency calibration. It
            replaces residual-based scoring with physical services and resource queues.

    Tables start at (0, 0), use nondecreasing times and cover the intended expert
    row counts. Out-of-range plans retain the original policy. Every EP rank
    must use identical calibration data. This is an approximate cost model,
    not a prediction guarantee or runtime calibration/weight cache.
    Missing link classes retain uniform costs. Legacy scoring uses exposed
    gradient residuals; critical-path scoring requires separate physical services.
    """

    calibration_version: int
    backend: str
    hidden_size: int
    intermediate_size: int
    ep_size: int
    transport: str
    forward_ms: tuple[tuple[int, float], ...]
    backward_ms: tuple[tuple[int, float], ...]
    weight_ms: float
    gradient_ms: float
    token_ms_per_row: float = 0.0
    minimum_gain_ms: float = 0.0
    schedule: str | None = None
    gradient_dtype: str = "float32"
    forward_repeats: int = 1
    replica_overhead_ms: float = 0.0
    quota_search_budget_ms: float = 0.0
    planner_overhead_ms: float = 0.0
    backward_weight_ms: float | None = None
    minimum_replica_gain_ms: float | None = None
    intra_node_weight_ms: float | None = None
    inter_node_weight_ms: float | None = None
    intra_node_gradient_ms: float | None = None
    inter_node_gradient_ms: float | None = None
    intra_node_backward_weight_ms: float | None = None
    inter_node_backward_weight_ms: float | None = None
    weight_fanout_ms: tuple[tuple[int, float, float], ...] = ()
    _rank_nodes: tuple[int, ...] = field(default=(), repr=False)
    critical_path: DeviceCriticalPathCalibration | None = None
    effective_group_ms: tuple[float, ...] = ()
    effective_source_row_floor: float = 0.

    def __post_init__(self) -> None:
        """Freeze measured inputs and reject inconsistent calibration contracts."""
        coefficients = tuple(self.effective_group_ms)
        if coefficients:
            if len(coefficients) != 6 or self.critical_path is not None:
                raise ValueError('Effective group calibration requires six coefficients and no physical template')
            for value in coefficients:
                _nonnegative(value, 'Effective group milliseconds')
        object.__setattr__(self, 'effective_group_ms', coefficients)
        _nonnegative(self.effective_source_row_floor, 'Effective source-row floor')
        if self.effective_source_row_floor > 1 or self.effective_source_row_floor and not coefficients:
            raise ValueError('Source-row floor requires group calibration and a value between zero and one')
        if isinstance(self.critical_path, dict):
            object.__setattr__(self, 'critical_path', DeviceCriticalPathCalibration(**self.critical_path))
        if self.critical_path is not None:
            if not isinstance(self.critical_path, DeviceCriticalPathCalibration):
                raise ValueError('critical_path must contain a matching device calibration')
            if len(self.critical_path.communication_domains) != self.ep_size:
                raise ValueError('Critical-path communication domains must match EP size')
            if self.backend not in ('push', 'pull') or self.transport != 'shmem_signal_kernel_gradient_bands':
                raise ValueError('Critical-path template requires Push/Pull SHMEM gradient bands')
        if isinstance(self.calibration_version, bool) or self.calibration_version != 2:
            raise ValueError("Regenerate replica calibration with calibration_version=2")
        expected_schedule = _default_schedule(self.backend, self.transport)
        schedules = (expected_schedule,)
        if expected_schedule == 'fixed_queue_gradient_bands_v1':
            schedules += ('fixed_queue_unmapped_guest_suffix_v1', 'fixed_queue_global_guest_communication_suffix_v1')
        if self.schedule is not None and self.schedule not in schedules:
            raise ValueError("Replica calibration schedule does not match the execution")
        if self.schedule in ('fixed_queue_unmapped_guest_suffix_v1',
                             'fixed_queue_global_guest_communication_suffix_v1') and self.critical_path is not None:
            raise ValueError('Critical-path template does not support the unmapped guest schedule')
        object.__setattr__(self, "schedule", self.schedule or expected_schedule)
        if self.gradient_dtype != "float32":
            raise ValueError("Replica cost calibration requires float32 weight partials")
        if self.backend not in ("native", "push", "pull") or not isinstance(self.transport, str) or not self.transport:
            raise ValueError("Cost model requires native, push or pull and a transport mode")
        if self.transport not in ("p2p", "shmem_signal_kernel_gradient", "shmem_signal_kernel_gradient_bands"):
            raise ValueError("Cost model requires p2p or a supported SHMEM kernel-gradient transport")
        if self.backend == "native" and self.transport != "p2p":
            raise ValueError("Native cost models require HCCL p2p")
        for name in ("hidden_size", "intermediate_size", "ep_size"):
            _integer(getattr(self, name), name)
        _integer(self.forward_repeats, "forward_repeats")
        for name in ("forward_ms", "backward_ms"):
            table = tuple(tuple(point) for point in getattr(self, name))
            if len(table) < 2 or any(len(point) != 2 for point in table) or table[0] != (0, 0):
                raise ValueError(f"{name} must start at (0, 0) and contain at least two knots")
            previous_rows, previous_ms = -1, 0.0
            for rows, duration in table:
                _integer(rows, "calibration rows", 0)
                _nonnegative(duration, "calibration milliseconds")
                if rows <= previous_rows or duration < previous_ms:
                    raise ValueError("Calibration rows must increase and times must not decrease")
                previous_rows, previous_ms = rows, duration
            object.__setattr__(self, name, table)
        for name in ("weight_ms", "gradient_ms", "token_ms_per_row", "minimum_gain_ms", "replica_overhead_ms",
                     "quota_search_budget_ms", "planner_overhead_ms"):
            _nonnegative(getattr(self, name), name)
        for name in ("backward_weight_ms", "minimum_replica_gain_ms", "intra_node_weight_ms", "inter_node_weight_ms",
                     "intra_node_gradient_ms", "inter_node_gradient_ms", "intra_node_backward_weight_ms",
                     "inter_node_backward_weight_ms"):
            if getattr(self, name) is not None:
                _nonnegative(getattr(self, name), name)
        fanout = tuple(tuple(point) for point in self.weight_fanout_ms)
        if fanout:
            if len(fanout) < 2 or fanout[0] != (0, 0, 0) or any(len(point) != 3 for point in fanout):
                raise ValueError('weight_fanout_ms must start at (0, 0, 0) and contain increasing copy counts')
            previous = (-1, 0.0, 0.0)
            for point in fanout:
                _integer(point[0], 'fanout copies', 0)
                for duration in point[1:]:
                    _nonnegative(duration, 'fanout milliseconds')
                if point[0] <= previous[0] or any(value < old for value, old in zip(point[1:], previous[1:])):
                    raise ValueError('Fanout copy counts must increase and durations must not decrease')
                previous = point
        object.__setattr__(self, 'weight_fanout_ms', fanout)
        object.__setattr__(self, '_rank_nodes', _validate_rank_nodes(self._rank_nodes, self.ep_size))

    def weight_copy_ms(self, copies: int, local_copies: int) -> float:
        """Charge one expert's outgoing forward weights without extrapolating fanout knots."""
        _integer(copies, 'outgoing copies', 0)
        _integer(local_copies, 'local outgoing copies', 0)
        if local_copies > copies:
            raise ValueError('Local outgoing copies cannot exceed all outgoing copies')
        if not copies:
            return 0.0
        local, remote = (row[0] for row in self._link_costs[:2])
        if not self.weight_fanout_ms or not self._rank_nodes or copies > self.weight_fanout_ms[-1][0]:
            return local_copies * local + (copies - local_copies) * remote
        index = bisect_left(self.fanout_knots, copies)
        left, right = self.weight_fanout_ms[index - 1], self.weight_fanout_ms[index]
        fraction = (copies - left[0]) / (right[0] - left[0])
        local_ms, remote_ms = (start + (end - start) * fraction for start, end in zip(left[1:], right[1:]))
        return (local_copies * local_ms + (copies - local_copies) * remote_ms) / copies

    @cached_property
    def fanout_knots(self) -> tuple[int, ...]:
        """Cache copy-count interpolation keys for invocation-local trials."""
        return tuple(point[0] for point in self.weight_fanout_ms)

    def transfer_ms(self, owner: int, target: int) -> tuple[float, float, float]:
        """Return forward-copy, backward-copy and exposed gradient costs for one edge.

        Missing node-specific measurements retain the original uniform coefficients.
        Verified resident backward reuse always suppresses the backward weight copy.
        """
        _integer(owner, 'owner rank', 0)
        _integer(target, 'target rank', 0)
        if owner >= self.ep_size or target >= self.ep_size:
            raise ValueError('Replica transfer ranks must belong to the EP group')
        if not self._rank_nodes:
            return self._link_costs[2]
        return self._link_costs[int(self._rank_nodes[owner] != self._rank_nodes[target])]

    @cached_property
    def _link_costs(self) -> tuple[tuple[float, float, float], ...]:
        """Resolve optional measurements once while preserving verified backward reuse."""
        backward = self.weight_ms if self.backward_weight_ms is None else self.backward_weight_ms
        base = self.weight_ms, backward, self.gradient_ms
        result = []
        for prefix in ('intra_node', 'inter_node'):
            values = tuple(getattr(self, f'{prefix}_{name}_ms') for name in ('weight', 'backward_weight', 'gradient'))
            weight, backward_copy, gradient = (fallback if value is None else value
                                               for value, fallback in zip(values, base))
            if self.backward_weight_ms is None and values[1] is None:
                backward_copy = weight
            result.append((weight, 0.0 if backward == 0 else backward_copy, gradient))
        return tuple(result) + (base,)

    def validate_execution(self, backend: str, hidden: int, intermediate: int, ep_size: int, transport: str) -> None:
        """Reject accidental reuse on another shape, backend or transport."""
        if (backend, hidden, intermediate, ep_size, transport) != (
                self.backend, self.hidden_size, self.intermediate_size, self.ep_size, self.transport):
            raise ValueError("Replica cost calibration does not match the execution shape/backend/transport")

    @cached_property
    def row_knots(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return interpolation keys without rebuilding them for each candidate."""
        return tuple(point[0] for point in self.forward_ms), tuple(point[0] for point in self.backward_ms)

    def compute_ms(self, rows: int, *, backward: bool = False) -> float:
        """Interpolate within the measured range; never extrapolate silently."""
        _integer(rows, "expert rows", 0)
        table = self.backward_ms if backward else self.forward_ms
        index = bisect_left(self.row_knots[int(backward)], rows)
        if index == len(table):
            raise ValueError("Expert rows exceed the calibrated cost range")
        if table[index][0] == rows:
            return table[index][1]
        left_rows, left_ms = table[index - 1]
        right_rows, right_ms = table[index]
        return left_ms + (right_ms - left_ms) * (rows - left_rows) / (right_rows - left_rows)

    def _phase_inputs(self, experts: dict[int, int], home: int, rank: int,
                      backward: bool, owner: tuple | None) -> tuple | None:
        """Prepare ordered inputs only for the concrete model's interpolation contract."""
        if type(self) is not ExpertReplicaCostModel:
            return None
        compute = self.backward_ms if backward else self.forward_ms
        return _ordered_compute(experts, home, rank, compute, compute if owner is None else owner, owner is None)

    def estimate_ms(self, plan: ExpertExecutionPlan) -> float:
        """Score separate phase bottlenecks, excluding invariant optimizer work.

        This conservative heuristic is not a prediction of an implicit global
        barrier. It does not assume that the same rank is slowest in F and B,
        or credit overlap without a measured schedule-prefix window.
        """
        copies = [{expert: rows for expert, rows in zip(slots, counts) if expert >= 0}
                  for slots, counts in zip(plan.slot_to_logical, plan.destination_counts)]
        return _QuotaCostState(self, copies, plan.logical_counts, plan.config).score()


class _QuotaCostState:
    """Invocation-local scores; a quota trial changes only its owner and target."""

    def __init__(self, model: ExpertReplicaCostModel, copies: list[dict[int, int]],
                 counts: tuple[tuple[int, ...], ...], config: ExpertReplicaConfig,
                 durations: dict[int, tuple[float, float]] | None = None) -> None:
        """Initialize scores and invariants for one quota-refinement invocation.

        Args:
            model: Matching offline execution-cost calibration.
            copies: Working quotas mutated by the refinement caller.
            counts: Immutable per-source logical-expert counts.
            config: Expert topology and guest budget.
            durations: Optional shared interpolation cache for this invocation.
        """
        self.model = model
        self.copies = copies
        self.counts = counts
        self.home = config.home_experts
        self.config = config
        if model.critical_path is not None:
            return
        self.backward_weight_ms = model.weight_ms if model.backward_weight_ms is None else model.backward_weight_ms
        self.link_costs = model._link_costs[:2] if model._rank_nodes and len(set(model._link_costs)) > 1 else None
        self.durations = {} if durations is None else durations
        self.weight_durations = {}
        self.transfers = [0] * config.ep_size
        self.local_transfers = [0] * config.ep_size
        self.outgoing = ([{} for _ in range(config.ep_size)]
                         if model.weight_fanout_ms and model._rank_nodes else None)
        for target, experts in enumerate(copies):
            for expert, rows in experts.items():
                owner = expert // self.home
                if rows and owner != target:
                    self.transfers[target] += 1
                    self.transfers[owner] += 1
                    if model._rank_nodes and model._rank_nodes[owner] == model._rank_nodes[target]:
                        self.local_transfers[target] += 1
                        self.local_transfers[owner] += 1
                    if self.outgoing is not None:
                        total, local = self.outgoing[owner].get(expert, (0, 0))
                        same_node = int(model._rank_nodes[owner] == model._rank_nodes[target])
                        self.outgoing[owner][expert] = total + 1, local + same_node
        self.costs = [self._rank_cost(rank, self.transfers[rank]) for rank in range(config.ep_size)]
        self.transfer_edges = sum(self.transfers) // 2

    @cached_property
    def source_totals(self) -> list[int]:
        """Sum immutable source rows only when token costs or refinement need them."""
        return [sum(row) for row in self.counts]

    def score(self, removed: int = 0) -> float:
        """Charge replay and replica-only scheduling work without assuming overlap."""
        if self.model.effective_group_ms:
            if removed:
                raise ValueError('Effective group scoring requires final explicit quotas')
            features = _effective_group_features(self.copies, self.counts, self.home, self.model._rank_nodes,
                                                 self.source_totals, self.model.effective_source_row_floor,
                                                 token_work=bool(self.model.effective_group_ms[4]))
            return sum(value * coefficient for value, coefficient in zip(features, self.model.effective_group_ms))
        if self.model.critical_path is not None:
            if removed:
                raise ValueError('Critical-path scoring requires final explicit quotas')
            return self.model.critical_path.estimate(self.model, self.copies, self.counts, self.config).total_ms
        host = (self.model.planner_overhead_ms
                + (self.model.replica_overhead_ms if self.transfer_edges > removed else 0.0))
        return (self.model.forward_repeats * (max(value[0] for value in self.costs) + host)
                + max(value[1] for value in self.costs))

    def _rank_cost(self, rank: int, transfers: int, local_transfers: int | None = None,
                   outgoing_removed: tuple[int, int] | None = None) -> tuple[float, float]:
        """Keep the original home/guest accumulation order, without float deltas."""
        experts = self.copies[rank]
        durations = self.durations
        token_rate = self.model.token_ms_per_row
        forward, backward = [0.0, 0.0], [0.0, 0.0]
        local = 0
        # Subclasses may override interpolation; only the concrete model has
        # the native primitive's exact compute_ms contract.
        native = (_rank_compute(experts, self.counts[rank] if token_rate else None, self.home, rank,
                                durations, self.model.forward_ms, self.model.backward_ms)
                  if type(self.model) is ExpertReplicaCostModel and not self.model.effective_group_ms else None)
        if native is None:
            for expert, rows in experts.items():
                guest = int(expert // self.home != rank)
                duration = durations.get(rows)
                if duration is None:
                    duration = self.model.compute_ms(rows), self.model.compute_ms(rows, backward=True)
                    durations[rows] = duration
                forward[guest] += duration[0]
                backward[guest] += duration[1]
                if token_rate:
                    local += min(self.counts[rank][expert], rows)
        else:
            forward, backward, local = [native[0]], [native[1]], native[2]
        # Without measured link concurrency or queue-prefix windows, neither
        # full duplex nor whole-home compute can hide these transfers safely.
        ready = transfers * self.model.weight_ms
        backward_copy = transfers * self.backward_weight_ms
        gradients = transfers * self.model.gradient_ms
        if transfers and self.link_costs is not None:
            local_count = self.local_transfers[rank] if local_transfers is None else local_transfers
            remote_count = transfers - local_count
            ready, backward_copy, gradients = (transfers * local if local == remote else
                                               local_count * local + remote_count * remote
                                               for local, remote in zip(*self.link_costs))
        if transfers and self.outgoing is not None:
            ready = self._forward_weights(rank, transfers, local_transfers, outgoing_removed)
        token_phase = 0.0
        if token_rate:
            remote_rows = self.source_totals[rank] + sum(experts.values()) - 2 * local
            token_phase = remote_rows * token_rate / 2
        return (sum(forward) + ready + token_phase,
                sum(backward) + backward_copy + gradients + token_phase)

    def _forward_weights(self, rank: int, transfers: int, local_transfers: int | None,
                         removed: tuple[int, int] | None) -> float:
        """Amortize only the same expert's sends; retain linear incoming and backward costs."""
        outgoing, local_outgoing, duration = 0, 0, 0.0
        for expert, (copies, local) in self.outgoing[rank].items():
            if removed is not None and removed[0] == expert:
                copies, local = copies - 1, local - removed[1]
            key = copies, local
            if key not in self.weight_durations:
                self.weight_durations[key] = self.model.weight_copy_ms(copies, local)
            duration += self.weight_durations[key]
            outgoing += copies
            local_outgoing += local
        local_total = self.local_transfers[rank] if local_transfers is None else local_transfers
        incoming, local_incoming = transfers - outgoing, local_total - local_outgoing
        local_ms, remote_ms = (row[0] for row in self.model._link_costs[:2])
        return duration + local_incoming * local_ms + (incoming - local_incoming) * remote_ms

    def trial_score(self, owner: int, target: int, expert: int, unchanged: tuple[float, float]) -> float:
        """Refresh the two changed ranks against the accepted edge counts.

        Args:
            owner: Original expert owner whose home quota was changed.
            target: Guest rank whose quota was changed on an accepted active edge.
            expert: Logical expert shared by these ranks.
            unchanged: Phase maxima of ranks unaffected by this edge.

        Returns:
            Separate forward/backward rank maxima for the trial quotas.
        """
        if self.model.effective_group_ms:
            return self.score()
        removed = int(not self.copies[target][expert])
        local_removed = int(bool(removed and self.model._rank_nodes
                                 and self.model._rank_nodes[owner] == self.model._rank_nodes[target]))
        for rank in (owner, target):
            self.costs[rank] = self._rank_cost(rank, self.transfers[rank] - removed,
                                               self.local_transfers[rank] - local_removed,
                                               (expert, local_removed) if removed and rank == owner else None)
        host = (self.model.planner_overhead_ms
                + (self.model.replica_overhead_ms if self.transfer_edges > removed else 0.0))
        forward = max(unchanged[0], self.costs[owner][0], self.costs[target][0])
        backward = max(unchanged[1], self.costs[owner][1], self.costs[target][1])
        return self.model.forward_repeats * (forward + host) + backward

    def accept(self, owner: int, target: int, expert: int) -> None:
        """Restore scores for the chosen quota before considering another edge.

        Args:
            owner: Original expert owner with its chosen home quota restored.
            target: Guest rank with its chosen quota restored, including a zero quota.
            expert: Logical expert whose edge may now be removed.
        """
        removed = int(not self.copies[target][expert])
        local_removed = int(bool(removed and self.model._rank_nodes
                                 and self.model._rank_nodes[owner] == self.model._rank_nodes[target]))
        self.transfer_edges -= removed
        if removed and self.outgoing is not None:
            total, local = self.outgoing[owner][expert]
            self.outgoing[owner][expert] = total - 1, local - local_removed
        for rank in (owner, target):
            self.transfers[rank] -= removed
            self.local_transfers[rank] -= local_removed
            self.costs[rank] = self._rank_cost(rank, self.transfers[rank])


def _refine_edge(target: int, expert: int, limit: int, knots: list[int], costs: _QuotaCostState) -> bool:
    """Compare the original bounded candidates for one existing guest edge."""
    copies = costs.copies
    current = copies[target].get(expert, 0)
    if not current:
        return False
    owner = expert // costs.home
    total = current + copies[owner][expert]
    loads = [sum(row.values()) for row in copies]
    lower = max(0, total + loads[owner] - copies[owner][expert] - limit)
    upper = min(total, limit - loads[target] + current)
    quotas = {current, lower, upper, total // 2}
    quotas.update(value for point in knots for value in (point, total - point))
    measured = min(costs.model.forward_ms[-1][0], costs.model.backward_ms[-1][0])
    best, best_cost = current, costs.score()
    unchanged = tuple(max((value[phase] for rank, value in enumerate(costs.costs)
                           if rank not in (owner, target)), default=0.0) for phase in (0, 1))
    for quota in sorted(value for value in quotas if lower <= value <= upper and value != current
                        and max(value, total - value) <= measured):
        copies[target][expert], copies[owner][expert] = quota, total - quota
        cost = costs.trial_score(owner, target, expert, unchanged)
        if cost + costs.model.minimum_gain_ms < best_cost:
            best, best_cost = quota, cost
    copies[target][expert], copies[owner][expert] = best, total - best
    costs.accept(owner, target, expert)
    if not best:
        del copies[target][expert]
    return best != current


def _prune_small_edges(costs: _QuotaCostState, limit: int, minimum_rows: int, measured: int) -> None:
    """Prefer fewer transfers on score plateaus without increasing the receive peak."""
    if not minimum_rows:
        return
    copies = costs.copies
    loads = [sum(row.values()) for row in copies]
    ceiling = min(limit, max(loads))
    candidates = sorted((rows, target, expert) for target, row in enumerate(copies)
                        for expert, rows in row.items() if expert // costs.home != target and 0 < rows < minimum_rows)
    for rows, target, expert in candidates:
        owner = expert // costs.home
        owner_rows = copies[owner][expert]
        if loads[owner] + rows > ceiling or owner_rows + rows > measured:
            continue
        previous = costs.score()
        unchanged = tuple(max((value[phase] for rank, value in enumerate(costs.costs)
                               if rank not in (owner, target)), default=0.0) for phase in (0, 1))
        copies[target][expert], copies[owner][expert] = 0, owner_rows + rows
        if costs.trial_score(owner, target, expert, unchanged) <= previous:
            loads[owner] += rows
            loads[target] -= rows
            costs.accept(owner, target, expert)
            del copies[target][expert]
        else:
            copies[target][expert], copies[owner][expert] = rows, owner_rows
            costs.accept(owner, target, expert)


def refine_replica_quotas(copies: list[dict[int, int]], counts: tuple[tuple[int, ...], ...],
                          config: ExpertReplicaConfig, limit: int, model: ExpertReplicaCostModel,
                          *, minimum_replica_rows: int = 0,
                          _durations: dict[int, tuple[float, float]] | None = None) -> None:
    """Prune cheap small copies, then accept strict worst-rank gains from quota search.

    Args:
        copies: Mutable per-rank logical-expert row quotas, updated in place.
        counts: Exact per-source logical-expert counts.
        config: Expert topology and per-rank guest budget.
        limit: Maximum permissible destination row count.
        model: Matching offline execution-cost calibration.
        minimum_replica_rows: Try cheaper deletions below this row count before
            applying the expensive quota-search budget. Zero preserves the original policy.
        _durations: Optional interpolation cache owned by the current planner invocation.
    """
    _integer(minimum_replica_rows, "minimum_replica_rows", 0)
    if model.ep_size != config.ep_size:
        raise ValueError("Replica cost model EP size does not match the plan")
    measured = min(model.forward_ms[-1][0], model.backward_ms[-1][0])
    if max(rows for row in copies for rows in row.values()) > measured:
        return
    home = config.home_experts
    edges = sorted((target, expert) for target, row in enumerate(copies)
                   for expert, rows in row.items() if expert // home != target and rows)
    if not edges:
        return
    knots = sorted(set(model.row_knots[0] + model.row_knots[1]))
    costs = _QuotaCostState(model, copies, counts, config, _durations)
    totals = list(map(sum, zip(*counts)))
    home_copies = [{expert: totals[expert] for expert in range(rank * home, (rank + 1) * home)}
                   for rank in range(config.ep_size)]
    home_cost = None
    if max(totals) <= measured and max(sum(row.values()) for row in home_copies) <= limit:
        home_cost = _QuotaCostState(model, home_copies, counts, config, costs.durations).score()
        if model.minimum_replica_gain_ms is None and home_cost + model.minimum_gain_ms < costs.score():
            copies[:] = home_copies
            return
    # Small-copy deletion has a bounded linear candidate pass and accepts
    # equal scores with fewer transfers. It does not require expensive quota search.
    _prune_small_edges(costs, limit, minimum_replica_rows, measured)
    edges = [(target, expert) for target, expert in edges if copies[target].get(expert, 0)]
    if not edges:
        return
    forward_rate = min(duration / rows for rows, duration in model.forward_ms if rows)
    backward_rate = min(duration / rows for rows, duration in model.backward_ms if rows)
    average_rows = sum(costs.source_totals) / config.ep_size
    moving = {expert for _, expert in edges}
    fixed = [max(sum(costs.durations[rows][phase] for expert, rows in row.items() if expert not in moving)
                 for row in copies) for phase in (0, 1)]
    # Owner-only execution was compared above. Every other profitable supported
    # candidate retains a guest and cannot move experts outside these edges.
    optimistic = (model.forward_repeats * (max(forward_rate * average_rows, fixed[0])
                                         + model.replica_overhead_ms + model.planner_overhead_ms)
                  + max(backward_rate * average_rows, fixed[1]))
    # Two deterministic sweeps bound Python work; candidate count depends on
    # calibration knots and existing edges, not the number of routed tokens.
    if costs.score() - optimistic > model.minimum_gain_ms + model.quota_search_budget_ms:
        for _ in range(2):
            changed = False
            for target, expert in edges:
                changed |= _refine_edge(target, expert, limit, knots, costs)
            if not changed:
                break
    if (home_cost is not None and model.minimum_replica_gain_ms is not None
            and costs.score() + model.minimum_replica_gain_ms >= home_cost):
        copies[:] = home_copies
