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
"""Offline v2 dependency reference for static native gradient-band traversal.

Every physical worker keeps its actual band/owned-edge queue. This reference
checks causal ordering, not an unmeasured network arbitration policy. Weight
SDMA and the owner's AIV MTE2/Vector/MTE3 return engine remain separate.
Missing producer/stripe measurements return uncovered instead of invented times.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from .plan import ExpertExecutionPlan


TEMPLATE = 'static_band_physical_workers_v2'
ObservationKey = tuple[int, int, int, int, int]


@dataclass(frozen=True)
class GradientBand:
    """Keep the physical projection-local ordinal and ragged element count."""

    ordinal: int
    row_offset: int
    rows: int
    elements: int


def gradient_band_layout(rows: int, columns: int, target_bytes: int = 4 * 1024 * 1024) -> tuple[GradientBand, ...]:
    """Match complete-row FP32 bands and the native maximum of 64 bands."""
    if any(type(value) is not int or value <= 0 for value in (rows, columns, target_bytes)):
        raise ValueError('Gradient matrix dimensions and band bytes must be positive integers')
    band_rows = max(1, min(rows, target_bytes // (columns * 4)))
    count = (rows + band_rows - 1) // band_rows
    if count > 64:
        raise ValueError('Gradient band count exceeds the native maximum of 64')
    return tuple(GradientBand(index, index * band_rows, min(band_rows, rows - index * band_rows),
                              min(band_rows, rows - index * band_rows) * columns) for index in range(count))


@dataclass(frozen=True)
class BandObservation:
    """Use logical elapsed durations; raw unsynchronized die counters are invalid.

    Arrays contain one value per physical producer/return worker, including
    no-tile producer contributions. Missing observations remain None. The
    observer must translate into one declared logical elapsed domain without
    subtracting timestamps from unrelated rank clocks.
    """

    owner_ready_ms: tuple[float | None, ...]
    guest_ready_ms: tuple[float | None, ...]
    stripe_service_ms: tuple[float | None, ...]
    owner_epoch: int
    guest_epoch: int
    clock_domain: str = 'logical_elapsed_ms'

    def __post_init__(self) -> None:
        for name in ('owner_ready_ms', 'guest_ready_ms', 'stripe_service_ms'):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class ReturnTask:
    """Immutable queue job; producer readiness and predecessor completion are distinct."""

    key: str
    rank: int
    worker: int
    projection: int
    ready_ms: float
    duration_ms: float
    dependencies: tuple[str, ...]
    engine: str


@dataclass(frozen=True)
class ReturnOperation:
    """One physical worker's immutable projection/band/owned-edge queue item."""

    projection: int
    band: int
    owner_rank: int
    logical_expert: int
    owner_slot: int
    target_rank: int
    target_slot: int


def static_band_queue(plan: ExpertExecutionPlan, rank: int, worker: int, projection: int,
                      bands: int, physical_workers: int) -> tuple[ReturnOperation, ...]:
    """Return queue order without requiring or fabricating producer timing data."""
    if (any(type(value) is not int for value in (rank, worker, projection, bands, physical_workers))
            or not 0 <= rank < plan.config.ep_size or physical_workers <= 0
            or not 0 <= worker < physical_workers or projection not in (0, 1) or not 1 <= bands <= 64):
        raise ValueError('Invalid rank, physical worker, projection or native band count')
    groups = min(4, bands, physical_workers)
    owned = tuple(edge for edge in plan.transfers if edge.owner_rank == rank)
    return tuple(ReturnOperation(projection, band, rank, edge.logical_expert, edge.owner_slot,
                                 edge.target_rank, edge.target_slot)
                 for band in range(worker % groups, bands, groups) for edge in owned)


@dataclass(frozen=True)
class StaticBandEstimate:
    """Keep uncovered inputs and gradient readiness separate from kernel completion."""

    covered: bool
    missing: tuple[str, ...]
    owner_gradient_ready_ms: tuple[float, ...] | None = None
    rank_completion_ms: tuple[float, ...] | None = None
    tasks: tuple[ReturnTask, ...] = ()
    trace: tuple[tuple[str, float, float], ...] = ()
    template: str = TEMPLATE
    production_calibrated: bool = False


def _duration(value: float | None) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


def static_band_reference(plan: ExpertExecutionPlan, band_counts: tuple[int, int], physical_workers: int,
                          observations: Mapping[ObservationKey, BandObservation],
                          local_dw_end_ms: tuple[float | None, ...], *, epochs: tuple[int, int] = (1, 1),
                          ack_service_ms: float) -> StaticBandEstimate:
    """Evaluate the deployed static queue, per-worker projections and ACK quorums.

    Args:
        plan: Exact immutable dispatch plan, preserving physical guest-slot order.
        band_counts: Separate W2/W13 band counts; ordinals restart at each projection.
        physical_workers: Reserved return/producer workers, not the up-to-four band groups.
        observations: (projection, owner, target, target_slot, band) records.
        local_dw_end_ms: All local AIC dW completion per rank, in the declared logical domain.
        epochs: Expected W2/W13 epochs for this invocation.
        ack_service_ms: Explicit per-edge ACK service; never inferred from matrix bandwidth.

    Returns:
        Uncovered with reasons when observations are missing. Otherwise a dependency
        timeline with no readiness reordering inside a worker's fixed queue. It does
        not install a runtime model or claim fitted production service durations.
    """
    plan.validate()
    if type(physical_workers) is not int or physical_workers <= 0:
        raise ValueError('physical_workers must be a positive integer')
    if len(band_counts) != 2 or any(type(value) is not int or not 1 <= value <= 64 for value in band_counts):
        raise ValueError('Two projection-local native band counts in [1, 64] are required')
    if len(epochs) != 2 or any(type(value) is not int or value <= 0 for value in epochs):
        raise ValueError('Two positive projection epochs are required')
    if not _duration(ack_service_ms):
        raise ValueError('ACK service must be a finite nonnegative duration')
    size = plan.config.ep_size
    if len(local_dw_end_ms) != size:
        raise ValueError('local_dw_end_ms must cover every rank')
    missing = [f'local_dw:{rank}' for rank, value in enumerate(local_dw_end_ms) if not _duration(value)]
    owned = {rank: tuple(edge for edge in plan.transfers if edge.owner_rank == rank) for rank in range(size)}
    # Transfers already follow target-rank/slot order. Preserve that stable order
    # across all experts owned by the same rank, not just within one expert.
    incoming = {rank: tuple(edge for edge in plan.transfers if edge.target_rank == rank) for rank in range(size)}
    for projection, bands in enumerate(band_counts):
        for edge in plan.transfers:
            for band in range(bands):
                key = projection, edge.owner_rank, edge.target_rank, edge.target_slot, band
                observation = observations.get(key)
                if observation is None:
                    missing.append(f'observation:{key}')
                    continue
                if observation.clock_domain != 'logical_elapsed_ms':
                    missing.append(f'unaligned_clock:{key}')
                if (type(observation.owner_epoch) is not int or type(observation.guest_epoch) is not int
                        or (observation.owner_epoch, observation.guest_epoch) !=
                        (epochs[projection], epochs[projection])):
                    missing.append(f'producer_epoch:{key}')
                for name in ('owner_ready_ms', 'guest_ready_ms'):
                    values = getattr(observation, name)
                    if len(values) != physical_workers or any(not _duration(value) for value in values):
                        missing.append(f'{name}:{key}')
                groups = min(4, bands, physical_workers)
                services = observation.stripe_service_ms
                if len(services) != physical_workers or any(
                        not _duration(services[worker]) for worker in range(physical_workers)
                        if worker % groups == band % groups):
                    missing.append(f'stripe_service_ms:{key}')
    if missing:
        return StaticBandEstimate(False, tuple(missing))

    tasks: list[ReturnTask] = []
    worker_tails: dict[tuple[int, int, int], str] = {}
    ack_tails: dict[tuple[int, int], str] = {}
    ack_keys: dict[tuple[int, int, int, int], str] = {}
    gradient_tails: dict[tuple[int, int], tuple[str, ...]] = {}
    for projection, bands in enumerate(band_counts):
        for rank in range(size):
            for worker in range(physical_workers):
                previous = None
                if projection:
                    previous = (ack_tails[0, rank] if worker == 0 else worker_tails[0, rank, worker])
                for operation in static_band_queue(plan, rank, worker, projection, bands, physical_workers):
                    band, target, slot = operation.band, operation.target_rank, operation.target_slot
                    observation = observations[projection, rank, target, slot, band]
                    key = f'return:{projection}:{rank}:{worker}:{band}:{target}:{slot}'
                    tasks.append(ReturnTask(key, rank, worker, projection,
                        max(max(observation.owner_ready_ms), max(observation.guest_ready_ms)),
                        observation.stripe_service_ms[worker], (previous,) if previous is not None else (),
                        'aiv_mte2_vector_mte3'))
                    previous = key
                key = f'worker_done:{projection}:{rank}:{worker}'
                tasks.append(ReturnTask(key, rank, worker, projection, 0., 0.,
                                        (previous,) if previous is not None else (), 'local_control'))
                worker_tails[projection, rank, worker] = key
            quorum = tuple(worker_tails[projection, rank, worker] for worker in range(physical_workers))
            gradient_tails[projection, rank] = quorum
            key = f'quorum:{projection}:{rank}'
            tasks.append(ReturnTask(key, rank, 0, projection, 0., 0., quorum, 'local_control'))
            previous = key
            for edge in owned[rank]:
                key = f'ack:{projection}:{rank}:{edge.target_rank}:{edge.target_slot}'
                tasks.append(ReturnTask(key, rank, 0, projection, 0., ack_service_ms, (previous,), 'ack_signal'))
                ack_keys[projection, rank, edge.target_rank, edge.target_slot] = key
                previous = key
            ack_tails[projection, rank] = previous

    completion_keys = []
    for rank in range(size):
        previous = ack_tails[1, rank]
        for projection in range(2):
            for edge in incoming[rank]:
                key = f'incoming_ack:{projection}:{rank}:{edge.owner_rank}:{edge.target_slot}'
                dependency = ack_keys[projection, edge.owner_rank, rank, edge.target_slot]
                tasks.append(ReturnTask(key, rank, 0, projection, 0., 0., (previous, dependency), 'local_control'))
                previous = key
        key = f'kernel_complete:{rank}'
        tasks.append(ReturnTask(key, rank, -1, 1, local_dw_end_ms[rank], 0.,
            tuple(worker_tails[1, rank, worker] for worker in range(physical_workers)) + (previous,), 'local_control'))
        completion_keys.append(key)

    finishes: dict[str, float] = {}
    trace = []
    for task in tasks:
        start = max(task.ready_ms, max((finishes[key] for key in task.dependencies), default=0.))
        finish = start + task.duration_ms
        finishes[task.key] = finish
        trace.append((task.key, start, finish))
    owner_ready = tuple(max(local_dw_end_ms[rank], max(finishes[key] for key in gradient_tails[1, rank]))
                        for rank in range(size))
    return StaticBandEstimate(True, (), owner_ready, tuple(finishes[key] for key in completion_keys),
                              tuple(tasks), tuple(trace))
