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
"""Calibrated device dependency and resource model for expert replica decisions."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from typing import Any, TYPE_CHECKING

from .capacity import ExpertReplicaConfig

if TYPE_CHECKING:
    from .cost import ExpertReplicaCostModel


def execution_fingerprint(metadata: dict[str, Any]) -> str:
    """Identify an execution contract without including installation prefixes."""
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _duration(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a finite nonnegative duration')


def _curve(points: Any, name: str) -> tuple[tuple[int, float], ...]:
    result = tuple(tuple(point) for point in points)
    if len(result) < 2 or result[0] != (0, 0):
        raise ValueError(f'{name} must start at (0, 0) and contain two or more knots')
    previous = (-1, 0.0)
    for point in result:
        if len(point) != 2:
            raise ValueError(f'{name} requires (rows, milliseconds) knots')
        rows, milliseconds = point
        _duration(milliseconds, name)
        if type(rows) is not int or rows <= previous[0] or milliseconds < previous[1]:
            raise ValueError(f'{name} requires increasing rows and nondecreasing times')
        previous = point
    return result


def _interpolate(points: tuple[tuple[int, float], ...], rows: int) -> float:
    index = bisect_left(points, (rows, -math.inf))
    if index == len(points):
        raise ValueError('Owner rows exceed the critical-path calibration range')
    if points[index][0] == rows:
        return points[index][1]
    left, right = points[index - 1], points[index]
    return left[1] + (right[1] - left[1]) * (rows - left[0]) / (right[0] - left[0])


@dataclass(frozen=True)
class CriticalPathEstimate:
    """Expose phase makespans and the owner update tail separately from host costs."""

    forward_ms: float
    recompute_ms: float
    backward_ms: float
    host_ms: float
    owner_ready_ms: tuple[float, ...]
    tasks: tuple[tuple[str, float, float, tuple[str, ...]], ...] = ()

    @property
    def total_ms(self) -> float:
        """Return one complete F/R/B prediction, excluding invariant optimizer work."""
        return self.forward_ms + self.recompute_ms + self.backward_ms + self.host_ms


class _Timeline:
    """List-schedule immutable jobs on exclusive calibrated resource domains."""

    def __init__(self, record: bool) -> None:
        self.available: dict[str, float] = {}
        self.tasks: list[tuple[str, float, float, tuple[str, ...]]] = []
        self.record = record

    def add(self, name: str, duration: float, ready: float, resources: tuple[str, ...]) -> float:
        """Honor both producer dependencies and resource completion before starting."""
        start = max(ready, max((self.available.get(key, 0.0) for key in resources), default=0.0))
        finish = start + duration
        for key in resources:
            self.available[key] = finish
        if self.record:
            self.tasks.append((name, start, finish, resources))
        return finish


@dataclass(frozen=True)
class DeviceCriticalPathCalibration:
    """Measured execution-specific data for the full-expert ready/ACK model.

    The first template uses a complete expert as its scheduling unit. Gradient
    release defaults to the full producer fence. Earlier release and overlap
    require explicit measured data; this is an advisory model, not a bound.
    Physical gradient service times must not be copied from exposed residuals.
    Communication domains identify measured endpoint bottlenecks on each node.
    """

    execution_signature: str
    home_forward_ms: tuple[tuple[int, float], ...]
    home_backward_ms: tuple[tuple[int, float], ...]
    intra_node_gradient_ms: float
    inter_node_gradient_ms: float
    communication_domains: tuple[int, ...]
    schema_version: int = 1
    template: str = 'replica_ready_ack_v1'
    overlap_verified: bool = False
    gradient_bands: tuple[tuple[int, float, float, float], ...] = ()
    gradient_workers: int = 1
    credit_ms: float = 0.0
    weight_ack_ms: float = 0.0
    gradient_ack_ms: float = 0.0
    planner_host_ms: float = 0.0
    rejection_host_ms: float = 0.0
    uncertainty_ms: float = 0.0

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('Unsupported critical-path calibration schema')
        if self.template != 'replica_ready_ack_v1':
            raise ValueError('Unsupported critical-path execution template')
        if (not isinstance(self.execution_signature, str) or len(self.execution_signature) != 64
                or any(char not in '0123456789abcdef' for char in self.execution_signature)):
            raise ValueError('Critical-path calibration needs an execution SHA256 signature')
        for name in ('home_forward_ms', 'home_backward_ms'):
            object.__setattr__(self, name, _curve(getattr(self, name), name))
        for name in ('intra_node_gradient_ms', 'inter_node_gradient_ms', 'credit_ms', 'weight_ack_ms',
                     'gradient_ack_ms', 'planner_host_ms', 'rejection_host_ms', 'uncertainty_ms'):
            _duration(getattr(self, name), name)
        domains = tuple(self.communication_domains)
        if not domains or any(type(value) is not int or value < 0 for value in domains):
            raise ValueError('Critical-path communication domains must be nonnegative integer labels')
        object.__setattr__(self, 'communication_domains', domains)
        if type(self.overlap_verified) is not bool:
            raise ValueError('overlap_verified must be a boolean')
        if type(self.gradient_workers) is not int or not 1 <= self.gradient_workers <= 4:
            raise ValueError('gradient_workers must be between one and four')
        bands = tuple(tuple(band) for band in self.gradient_bands)
        previous_projection, previous = -1, 0.0
        for band in bands:
            if len(band) != 4 or type(band[0]) is not int or band[0] not in (0, 1):
                raise ValueError('Gradient bands require (projection, ready fraction, local ms, remote ms)')
            for value in band[1:]:
                _duration(value, 'gradient band')
            if band[0] < previous_projection:
                raise ValueError('Gradient projections must follow W2 then W13')
            if band[0] != previous_projection:
                previous = 0.0
            if not previous <= band[1] <= 1:
                raise ValueError('Gradient band release fractions must be ordered within [0, 1]')
            previous_projection, previous = band[0], band[1]
        if bands and {band[0] for band in bands} != {0, 1}:
            raise ValueError('Gradient band calibration must cover both weight projections')
        if bands and not self.overlap_verified:
            raise ValueError('Early gradient release requires measured overlap')
        object.__setattr__(self, 'gradient_bands', bands)

    @property
    def owner_coverage(self) -> int:
        """Return the calibrated owner-only row domain without extrapolation."""
        return min(self.home_forward_ms[-1][0], self.home_backward_ms[-1][0])

    def _links(self, owner: int, target: int, nodes: tuple[int, ...]) -> tuple[str, ...]:
        domains = self.communication_domains
        # Input/output domains are distinct only when the calibration explicitly
        # certified overlap; otherwise an endpoint is conservatively half-duplex.
        source = f'link:{nodes[owner]}:{domains[owner]}'
        destination = f'link:{nodes[target]}:{domains[target]}'
        suffixes = (':out', ':in') if self.overlap_verified else ('', '')
        return tuple(dict.fromkeys((source + suffixes[0], destination + suffixes[1])))

    def estimate(self, model: ExpertReplicaCostModel, copies: list[dict[int, int]],
                 counts: tuple[tuple[int, ...], ...], config: ExpertReplicaConfig,
                 *, trace: bool = False, baseline: bool = False) -> CriticalPathEstimate:
        """Predict final owner readiness using copies, compute queues and peer order.

        Args:
            model: Matching expert compute and weight-transfer calibration.
            copies: Exact per-destination integer expert quotas.
            counts: Current logical source histograms.
            config: Physical expert-slot budget.
            trace: Retain scheduled jobs for an offline explanation.
            baseline: Exclude the separately measured rejection-path overhead.
        """
        if len(self.communication_domains) != config.ep_size:
            raise ValueError('Critical-path communication domains must match EP size')
        nodes = model._rank_nodes
        if len(nodes) != config.ep_size:
            raise ValueError('Critical-path scoring requires an explicit EP node mapping')
        home = config.home_experts
        edges = sorted((expert // home, target, expert, rows) for target, row in enumerate(copies)
                       for expert, rows in row.items() if rows and expert // home != target)
        if baseline and edges:
            raise ValueError('A true baseline cannot contain expert replicas')
        forward = self._phase(model, copies, counts, config, nodes, edges, backward=False, trace=trace)
        backward = self._phase(model, copies, counts, config, nodes, edges, backward=True, trace=trace)
        host = self.planner_host_ms if edges else (0.0 if baseline else self.rejection_host_ms)
        return CriticalPathEstimate(forward[0], (model.forward_repeats - 1) * forward[0], backward[0], host,
                                    tuple(backward[1]), tuple(forward[2] + backward[2]))

    def _phase(self, model: ExpertReplicaCostModel, copies: list[dict[int, int]],
               counts: tuple[tuple[int, ...], ...], config: ExpertReplicaConfig, nodes: tuple[int, ...],
               edges: list[tuple[int, int, int, int]], *, backward: bool, trace: bool,
               ) -> tuple[float, list[float], list[tuple[str, float, float, tuple[str, ...]]]]:
        timeline, home = _Timeline(trace), config.home_experts
        prefix = 'backward' if backward else 'forward'
        ready = {}
        batched = set()
        for owner, target, expert, _ in edges:
            copy_ms = model.transfer_ms(owner, target)[int(backward)]
            if backward and model.backward_weight_ms == 0:
                ready[target, expert] = 0.0
                continue
            if not backward and model.weight_fanout_ms:
                if (owner, expert) in batched:
                    continue
                batched.add((owner, expert))
                targets = [peer for source, peer, logical, _ in edges if (source, logical) == (owner, expert)]
                local = sum(nodes[owner] == nodes[peer] for peer in targets)
                copy_ms = model.weight_copy_ms(len(targets), local)
                resources = tuple(dict.fromkeys(key for peer in targets for key in self._links(owner, peer, nodes)))
                finish = timeline.add(f'{prefix}:fanout:{owner}:{expert}', copy_ms, self.credit_ms, resources)
                for peer in targets:
                    ready[peer, expert] = finish
                continue
            credit = timeline.add(f'{prefix}:credit:{owner}:{target}:{expert}', self.credit_ms, 0.0, ())
            ready[target, expert] = timeline.add(f'{prefix}:weight:{owner}:{target}:{expert}', copy_ms, credit,
                                                self._links(owner, target, nodes))
        barrier = 0.0 if self.overlap_verified else max(ready.values(), default=0.0)
        enabled = bool(edges)
        home_ends, compute_ends, guest_ready = [], [], {}
        for rank, row in enumerate(copies):
            owner_curve = self.home_backward_ms if backward else self.home_forward_ms
            native = (model._phase_inputs(row, home, rank, backward, None if enabled else owner_curve)
                      if type(self) is DeviceCriticalPathCalibration else None)
            home_duration = 0.0
            if native is None:
                for expert, rows in row.items():
                    if expert // home == rank:
                        home_duration += (model.compute_ms(rows, backward=backward) if enabled else
                                          _interpolate(owner_curve, rows))
            else:
                home_duration = native[0]
            compute = (f'cube:{rank}',)
            home_end = timeline.add(f'{prefix}:home:{rank}', home_duration, barrier, compute)
            home_ends.append(home_end)
            finish = home_end
            guests = (((expert, model.compute_ms(rows, backward=backward)) for expert, rows in row.items()
                       if rows and expert // home != rank) if native is None else native[1])
            for expert, duration in guests:
                finish = timeline.add(f'{prefix}:guest:{rank}:{expert}', duration,
                                      max(home_end, ready[rank, expert]), compute)
                guest_ready[rank, expert] = finish - duration, finish
            # The old per-row token coefficient is retained only when measured;
            # the complete graph's dispatch/combine residual must not be added twice.
            remote = sum(counts[rank]) + sum(row.values()) - 2 * sum(
                min(counts[rank][expert], rows) for expert, rows in row.items())
            finish = timeline.add(f'{prefix}:token:{rank}', remote * model.token_ms_per_row, finish,
                                  (f'token:{rank}',))
            compute_ends.append(finish)
        owner_ends = list(compute_ends)
        if backward:
            reduction_ends = list(home_ends)
            queues: dict[tuple[int, int], list[tuple[int, int]]] = {}
            for owner, target, expert, _ in edges:
                queues.setdefault((owner, expert), []).append((target, expert))
            all_compute = max(compute_ends, default=0.0)
            bands = self.gradient_bands or ((0, 1.0, self.intra_node_gradient_ms, self.inter_node_gradient_ms),)

            def release_time(target: int, expert: int, band: int) -> float:
                start, finish = guest_ready[target, expert]
                return start + (finish - start) * bands[band][1]

            for projection in sorted({band[0] for band in bands}):
                pending = []
                projection_ready = list(reduction_ends)
                for (owner, expert), peers in queues.items():
                    target, _ = peers[0]
                    for band, values in enumerate(bands):
                        if values[0] != projection:
                            continue
                        release = max(projection_ready[owner], release_time(target, expert, band),
                                      0.0 if self.overlap_verified else all_compute)
                        heapq.heappush(pending, (release, owner, expert, band, 0))
                while pending:
                    release, owner, expert, band, index = heapq.heappop(pending)
                    target, _ = queues[owner, expert][index]
                    same_node = nodes[owner] == nodes[target]
                    duration = bands[band][2 if same_node else 3]
                    resources = self._links(target, owner, nodes) + (
                        f'gradient:{owner}:{band % self.gradient_workers}',)
                    name = f'backward:gradient:{owner}:{target}:{expert}'
                    if self.gradient_bands:
                        name += f':projection{projection}:band{band}'
                    finish = timeline.add(name, duration, release, resources)
                    owner_ends[owner] = max(owner_ends[owner], finish)
                    reduction_ends[owner] = max(reduction_ends[owner], finish)
                    if index + 1 < len(queues[owner, expert]):
                        peer, _ = queues[owner, expert][index + 1]
                        release = max(finish, projection_ready[owner], release_time(peer, expert, band),
                                      0.0 if self.overlap_verified else all_compute)
                        heapq.heappush(pending, (release, owner, expert, band, index + 1))
            # Native workers join all owned projections/bands before publishing
            # the final ACK sweep. An early peer cannot release its pool alone.
            for owner, target, expert, _ in edges:
                ack = timeline.add(f'backward:ack:{owner}:{target}:{expert}', self.gradient_ack_ms,
                                   reduction_ends[owner], ())
                owner_ends[target] = max(owner_ends[target], ack)
        else:
            for owner, target, expert, _ in edges:
                ack = timeline.add(f'forward:ack:{owner}:{target}:{expert}', self.weight_ack_ms,
                                   max(ready[target, expert], compute_ends[target]), ())
                owner_ends[owner] = max(owner_ends[owner], ack)
        return max(owner_ends, default=0.0), owner_ends, timeline.tasks
