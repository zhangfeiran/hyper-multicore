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
"""Projection-ready SDMA weights and fused FP32 replica-gradient return."""

from __future__ import annotations

import math
import struct
import threading
from contextlib import contextmanager
from typing import Any, Iterator

import torch

from ._metadata import _async_device_bytes
from .capacity import _integer
from .pool import ReplicaPool, ReplicaPrefetch
from .residency import _ReplicaWeightCache
from .routing import ReplicaRoute
from .transport import _gradient_accumulators

KERNEL_GRADIENT_TRANSPORT = "shmem_signal_kernel_gradient"
GRADIENT_BAND_TRANSPORT = "shmem_signal_kernel_gradient_bands"
KERNEL_GRADIENT_TRANSPORT_MODES = (KERNEL_GRADIENT_TRANSPORT, GRADIENT_BAND_TRANSPORT)
_SIGNAL_BYTES = 64
_SIGNAL_CHANNELS = 5
_MAX_EPOCH = 2**31 - 1
_GRADIENT_BAND_BYTES = 4 * 1024 * 1024
_MAX_GRADIENT_BANDS = 64
_MAX_GRADIENT_BAND_CORES = 32


def _aligned(size: int) -> int:
    return (size + _SIGNAL_BYTES - 1) // _SIGNAL_BYTES * _SIGNAL_BYTES


def gradient_band_layout(shape: tuple[int, ...]) -> tuple[int, int] | None:
    """Derive bounded, whole-row publication bands from an FP32 matrix shape.

    Args:
        shape: Expert matrix dimensions, excluding the expert slot dimension.

    Returns:
        Rows per band and band count, or None for an unsupported shape.
    """
    if len(shape) != 2 or any((not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape)):
        return None
    rows, columns = shape
    band_rows = max(1, min(rows, _GRADIENT_BAND_BYTES // (columns * 4)))
    bands = (rows + band_rows - 1) // band_rows
    return (band_rows, bands) if bands <= _MAX_GRADIENT_BANDS else None


def signal_storage_bytes(
    shapes: tuple[tuple[int, ...], ...],
    slots: int,
    ep_size: int,
    element_size: int,
    *,
    gradient_bands: bool = False,
    storage_slots: int | None = None,
) -> int:
    """Size symmetric B slots, FP32 gradients and per-projection ready words."""
    _integer(slots, "slots")
    _integer(ep_size, "ep_size")
    _integer(element_size, "element_size")
    if element_size not in (2, 4):
        raise ValueError("Signal transport requires two- or four-byte weights")
    if not shapes or any((not shape for shape in shapes)):
        raise ValueError("Signal transport requires nonempty expert matrix shapes")
    for shape in shapes:
        for dim in shape:
            _integer(dim, "expert matrix dimension")
    storage_slots = slots if storage_slots is None else _integer(storage_slots, "storage_slots")
    if storage_slots > slots:
        raise ValueError("Replica matrix storage cannot exceed the signal slot count")
    matrices = sum(
        (_aligned(storage_slots * math.prod(shape) * size) for size in (element_size, 4) for shape in shapes)
    )
    layouts = [gradient_band_layout(shape) for shape in shapes] if gradient_bands else []
    band_bytes = sum(
        (slots * layout[1] * _MAX_GRADIENT_BAND_CORES * _SIGNAL_BYTES for layout in layouts if layout is not None)
    )
    return matrices + (_SIGNAL_CHANNELS * ep_size + len(shapes)) * slots * _SIGNAL_BYTES + band_bytes


def _transfer_edges(route: ReplicaRoute) -> tuple[list, list]:
    """Keep owner sends and guest receives in the immutable plan order."""
    incoming = [item for item in route.plan.transfers if item.target_rank == route.rank]
    outgoing = [item for item in route.plan.transfers if item.owner_rank == route.rank]
    return incoming, outgoing


class SignalReplicaTransport:
    """Lease symmetric slots with credit, projection readiness and final ACKs.

    SDMA weight copies run on one stream per peer. Fused backward workers return
    FP32 gradients; schedules without complete projection fences use parallel
    SDMA internally. All ranks submit the same invocation and epoch sequence.
    """

    def __init__(
        self,
        runtime: Any,
        storage: torch.Tensor,
        slots: int,
        ep_size: int,
        *,
        gradient_bands: bool = False,
        storage_slots: int | None = None,
    ) -> None:
        """Bind externally owned, aligned symmetric uint8 storage.

        Args:
            runtime: Stream-ordered SDMA put/get, signal/wait and host-barrier provider.
            storage: Symmetric storage retained through all remote consumers.
            slots: Guest expert budget per rank.
            ep_size: Number of ranks in the runtime's peer namespace.
            gradient_bands: Enable Cube-produced row bands; include their bytes in storage sizing.
            storage_slots: Resident matrix count, bounded by the unchanged signal stride.
        """
        if (
            storage.dtype != torch.uint8
            or storage.ndim != 1
            or (not storage.is_contiguous())
            or storage.data_ptr() % _SIGNAL_BYTES
        ):
            raise ValueError("Signal replica storage must be a cache-line-aligned contiguous uint8 vector")
        _integer(slots, "slots")
        _integer(ep_size, "ep_size")
        self.storage_slots = slots if storage_slots is None else _integer(storage_slots, "storage_slots")
        if self.storage_slots > slots:
            raise ValueError("Replica matrix storage cannot exceed the signal slot count")
        self.runtime, self.storage, self.slots, self.ep_size = (runtime, storage, slots, ep_size)
        self.pool = self.signals = self.projection_signals = None
        self.gradient_bands = gradient_bands
        self._gradient_band_storage = {}
        self._gradient_band_tables = {}
        self.epoch = 0
        self._layout = None
        self._lock = threading.Lock()
        self._streams, self._gradient_streams, self._gradient_scratch = ({}, {}, {})
        self._resident_generation = self._resident_ready = None
        self._weight_cache = _ReplicaWeightCache()
        self._last_weight_copy_counts = (0, 0)
        self.last_kernel_gradient_matrices = ()

    @property
    def kernel_gradients(self) -> bool:
        """Allow fused projection publication, accumulation and acknowledgement."""
        return True

    def kernel_gradient_band_signals(self, matrix: int, cores: int) -> tuple[torch.Tensor, int, int] | None:
        """Borrow a compact per-Cube band table from the existing symmetric inbox.

        Args:
            matrix: Projection index in the bound weight/gradient tuple.
            cores: Active Cube producer count, identical on every EP rank.

        Returns:
            Table, rows per band and band count, or None to retain full-expert return.

        Raises:
            ValueError: The provider has no active backward lease or the indices are invalid.
        """
        if not self.gradient_bands or not self._lock.locked() or self.pool is None:
            raise ValueError("Gradient bands require an active backward lease")
        _integer(matrix, "matrix", 0)
        _integer(cores, "cores")
        if matrix >= len(self._layout):
            raise ValueError("Gradient band matrix index is outside the bound layout")
        if cores > _MAX_GRADIENT_BAND_CORES or matrix not in self._gradient_band_storage:
            return None
        band_rows, bands = gradient_band_layout(self._layout[matrix][0])
        key = (matrix, cores)
        if key not in self._gradient_band_tables:
            length = self.slots * bands * cores * _SIGNAL_BYTES
            self._gradient_band_tables[key] = (
                self._gradient_band_storage[matrix][:length]
                .view(torch.int32)
                .reshape(self.slots, bands, cores, _SIGNAL_BYTES // 4)
            )
        return (self._gradient_band_tables[key], band_rows, bands)

    @property
    def gradient_scratch_bytes(self) -> int:
        """Return the bounded local FP32 scratch used by the internal fallback."""
        return sum((tensor.numel() * tensor.element_size() for tensor in self._gradient_scratch.values()))

    def kernel_gradient_signals(self) -> tuple[int, int, int]:
        """Reserve one projection epoch inside an active backward lease."""
        if not self._lock.locked() or self.pool is None or self.pool.gradients is None:
            raise ValueError("Kernel gradients require an active projection SDMA lease")
        return (self._advance(), self.signals[3].data_ptr(), self.signals[4].data_ptr())

    def _bind(self, weights: tuple[torch.Tensor, ...]) -> None:
        if not weights:
            raise ValueError("Signal transport requires nonempty expert weights")
        layout = tuple(((tuple(weight.shape[1:]), weight.dtype, weight.device) for weight in weights))
        if self.pool is not None:
            if layout != self._layout:
                raise ValueError("Signal transport matrix layout cannot change without reinitialization")
            return
        if any((weight.device != self.storage.device or weight.dtype != weights[0].dtype for weight in weights)):
            raise ValueError("Signal weights must share the symmetric storage device and one dtype")
        shapes = tuple((tuple(weight.shape[1:]) for weight in weights))
        required = signal_storage_bytes(
            shapes,
            self.slots,
            self.ep_size,
            weights[0].element_size(),
            gradient_bands=self.gradient_bands,
            storage_slots=self.storage_slots,
        )
        if required > self.storage.numel():
            raise ValueError("Symmetric signal replica storage is too small")
        views, offset = ([], 0)
        for dtype, size in ((weights[0].dtype, weights[0].element_size()), (torch.float32, 4)):
            for shape in shapes:
                length = self.storage_slots * math.prod(shape) * size
                views.append(self.storage[offset : offset + length].view(dtype).reshape(self.storage_slots, *shape))
                offset += _aligned(length)
        signal_end = offset + _SIGNAL_CHANNELS * self.ep_size * self.slots * _SIGNAL_BYTES
        self.signals = (
            self.storage[offset:signal_end]
            .view(torch.int32)
            .reshape(_SIGNAL_CHANNELS, self.ep_size, self.slots, _SIGNAL_BYTES // 4)
        )
        ready_end = signal_end + len(weights) * self.slots * _SIGNAL_BYTES
        self.projection_signals = (
            self.storage[signal_end:ready_end].view(torch.int32).reshape(len(weights), self.slots, _SIGNAL_BYTES // 4)
        )
        self.signals.zero_()
        self.projection_signals.zero_()
        signal_end = ready_end
        if self.gradient_bands:
            for matrix, shape in enumerate(shapes):
                band_layout = gradient_band_layout(shape)
                if band_layout is None:
                    continue
                length = self.slots * band_layout[1] * _MAX_GRADIENT_BAND_CORES * _SIGNAL_BYTES
                self._gradient_band_storage[matrix] = self.storage[signal_end : signal_end + length]
                self._gradient_band_storage[matrix].zero_()
                signal_end += length
        self.runtime.host_barrier()
        self.pool = ReplicaPool(tuple(views[: len(weights)]), self.storage_slots, borrow=True)
        self.pool.gradients = tuple(views[len(weights) :])
        self._layout = layout

    def _advance(self, *, reserve: int = 0) -> int:
        if self.epoch >= _MAX_EPOCH - reserve:
            self.runtime.host_barrier()
            self.signals.zero_()
            self.projection_signals.zero_()
            for table in self._gradient_band_storage.values():
                table.zero_()
            self.runtime.host_barrier()
            self.epoch = 0
            self._resident_generation = self._resident_ready = None
            self._weight_cache.clear()
        self.epoch += 1
        return self.epoch

    def _word(self, channel: int, peer: int, slot: int) -> torch.Tensor:
        return self.signals[channel, peer, slot, :1]

    def _publish(self, channel: int, peer_index: int, slot: int, target: int, epoch: int) -> None:
        self.runtime.signal(self._word(channel, peer_index, slot), epoch, target)

    def _wait(self, channel: int, peer: int, slot: int, epoch: int) -> None:
        self.runtime.wait_signal(self._word(channel, peer, slot), epoch, comparison="ge")

    def _validate_route_storage(self, route: ReplicaRoute) -> None:
        """Reject a route outside the provider's topology or allocated matrix prefix."""
        if route.plan.config.replica_slots_per_rank != self.slots or route.plan.config.ep_size != self.ep_size:
            raise ValueError("Signal transport topology does not match the replica route")
        if self.storage_slots < self.slots and any(
            (
                not route.plan.config.home_experts
                <= transfer.target_slot
                < route.plan.config.home_experts + self.storage_slots
                for transfer in route.plan.transfers
            )
        ):
            raise ValueError("The replica route addresses an unallocated matrix slot")

    def _saved_generation_is_resident(self, backward: bool, overlap: bool, generation: object) -> bool:
        """Accept only this backward's saved weights before the reserved epoch limit."""
        return (
            backward
            and overlap
            and generation is not None
            and generation is self._resident_generation
            and self._resident_ready is not None
            and self.epoch < _MAX_EPOCH - 2
        )

    @contextmanager
    def lease(
        self,
        weights: tuple[torch.Tensor, ...],
        route: ReplicaRoute,
        *,
        backward: bool = False,
        overlap: bool = False,
        reuse_generation: object | None = None,
        _overwrite_gradients: bool = False,
    ) -> Iterator[ReplicaPool | ReplicaPrefetch]:
        """Keep invocation epochs fresh while owners reuse unchanged resident payloads."""
        if reuse_generation is not None and (not backward):
            raise ValueError("Replica weights may be reused only by their saved backward context")
        self._validate_route_storage(route)
        home = route.plan.config.home_experts
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent signal replica leases are unsupported")
        try:
            self._last_weight_copy_counts = (0, 0)
            self._bind(weights)
            active_slots = (
                max(
                    (item.target_slot - home + 1 for item in route.plan.transfers if item.target_rank == route.rank),
                    default=0,
                )
                if backward
                else None
            )
            with self.pool.lease(
                backward=backward, active_slots=active_slots, _overwrite_gradients=_overwrite_gradients
            ):
                if self._saved_generation_is_resident(backward, overlap, reuse_generation):

                    yield ReplicaPrefetch(
                        self.pool,
                        self.pool.wait_weights,
                        projection_ready=self._resident_ready,
                        weight_generation=self._resident_generation,
                    )
                else:
                    self._resident_generation = self._resident_ready = None
                    with self._overlapped_prefetch(weights, route, backward=backward) as prefetched:
                        if overlap:
                            self._resident_generation = object()
                            self._resident_ready = prefetched.projection_ready
                            prefetched.weight_generation = self._resident_generation
                            yield prefetched
                        else:
                            prefetched.wait_weights()
                            yield self.pool
        except BaseException:
            self._weight_cache.clear()
            self._resident_generation = self._resident_ready = None
            raise
        finally:
            self._lock.release()

    def _copy_projections(
        self,
        outgoing: list,
        copies: Any,
        sources: dict,
        value: torch.Tensor,
        order: tuple[int, ...],
        home: int,
        backend: Any,
        streams: dict,
    ) -> None:
        """Publish a fresh projection epoch after each copied or resident payload."""
        for item in outgoing:
            stream = streams[item.target_rank]
            with backend.stream(stream):
                slot = item.target_slot - home
                value.record_stream(stream)
                for index in order:
                    if (item.target_rank, item.target_slot) not in copies.reused:
                        source = sources[item.owner_slot][index]
                        source.record_stream(stream)
                        self.runtime.put(self.pool.weights[index][slot], source, item.target_rank, use_sdma=True)
                    self.runtime.put(self.projection_signals[index, slot, :1], value, item.target_rank, use_sdma=True)

    @contextmanager
    def _overlapped_prefetch(
        self, weights: tuple[torch.Tensor, ...], route: ReplicaRoute, *, backward: bool = False
    ) -> Iterator[ReplicaPrefetch]:
        epoch = self._advance(reserve=2 if backward else 0)
        home = route.plan.config.home_experts
        incoming, outgoing = _transfer_edges(route)
        copies = self._weight_cache.prepare(weights, route)
        self._last_weight_copy_counts = (len(outgoing) - len(copies.reused), len(copies.reused))
        value = _async_device_bytes(struct.pack("<i", epoch), weights[0].device).view(torch.int32)
        for item in incoming:
            self._publish(0, route.rank, item.target_slot - home, item.owner_rank, epoch)
        for item in outgoing:
            self._wait(0, item.target_rank, item.target_slot - home, epoch)
        sources = {
            slot: tuple((weight[slot].contiguous() for weight in weights)) for slot in copies.required_owner_slots
        }

        def _wait(matrix_index):
            for item in incoming:
                self.runtime.wait_signal(
                    self.projection_signals[matrix_index, item.target_slot - home, :1], epoch, comparison="ge"
                )

        order = tuple(range(len(weights)))
        if backward:
            order = order[::-1]
        peers = sorted({item.target_rank for item in outgoing})
        with self._copy_streams(peers, self._streams, weights[0].device) as (backend, streams):
            self._copy_projections(outgoing, copies, sources, value, order, home, backend, streams)
            ready = (tuple((signal.data_ptr() for signal in self.projection_signals)), epoch)
            prefetched = ReplicaPrefetch(self.pool, _wait, projection_ready=ready)
            try:
                yield prefetched
            finally:
                prefetched.wait_weights()
                for item in incoming:
                    self._publish(2, route.rank, item.target_slot - home, item.owner_rank, epoch)
                for item in outgoing:
                    self._wait(2, item.target_rank, item.target_slot - home, epoch)
        self._weight_cache.commit(copies)

    @contextmanager
    def _copy_streams(
        self, keys: list[int], cache: dict[int, Any], device: torch.device
    ) -> Iterator[tuple[Any, dict[int, Any]]]:
        backend = getattr(torch, device.type)
        caller = backend.current_stream(device)
        streams = {}
        try:
            if keys:
                ready = backend.Event()
                ready.record(caller)
                for key in keys:
                    if key not in cache:
                        cache[key] = backend.Stream(device=device)
                    stream = cache[key]
                    stream.wait_event(ready)
                    streams[key] = stream
            yield (backend, streams)
        finally:
            for stream in streams.values():
                done = backend.Event()
                done.record(stream)
                caller.wait_event(done)

    def return_gradients(
        self, gradients: tuple[torch.Tensor, ...], guests: tuple[torch.Tensor, ...], route: ReplicaRoute
    ) -> tuple[torch.Tensor, ...]:
        """Preserve borrowed home gradients during the internal SDMA fallback."""
        return self._return_gradients(_gradient_accumulators(gradients, consume=False), guests, route)

    def return_gradients_owned(
        self, gradients: tuple[torch.Tensor, ...], guests: tuple[torch.Tensor, ...], route: ReplicaRoute
    ) -> tuple[torch.Tensor, ...]:
        """Consume exclusive FP32 home buffers during the internal fallback."""
        return self._return_gradients(_gradient_accumulators(gradients, consume=True), guests, route)

    def _return_gradients(
        self, result: tuple[torch.Tensor, ...], guests: tuple[torch.Tensor, ...], route: ReplicaRoute
    ) -> tuple[torch.Tensor, ...]:
        epoch = self._advance()
        home = route.plan.config.home_experts
        incoming = [item for item in route.plan.transfers if item.target_rank == route.rank]
        owned = sorted(
            (item for item in route.plan.transfers if item.owner_rank == route.rank), key=lambda item: item.target_rank
        )
        for item in incoming:
            self._publish(3, route.rank, item.target_slot - home, item.owner_rank, epoch)
        if owned:
            with self._copy_streams(list(range(len(result))), self._gradient_streams, result[0].device) as (
                backend,
                streams,
            ):
                for index, (output, guest) in enumerate(zip(result, guests)):
                    with backend.stream(streams[index]):
                        output.record_stream(streams[index])
                        if index not in self._gradient_scratch:
                            self._gradient_scratch[index] = torch.empty_like(output[0])
                        scratch = self._gradient_scratch[index]
                        for item in owned:
                            slot = item.target_slot - home
                            self._wait(3, item.target_rank, slot, epoch)
                            self.runtime.get(scratch, guest[slot], item.target_rank, use_sdma=True)
                            output[item.owner_slot].add_(scratch)
            for item in owned:
                self._publish(4, route.rank, item.target_slot - home, item.target_rank, epoch)
        for item in incoming:
            self._wait(4, item.owner_rank, item.target_slot - home, epoch)
        return result
