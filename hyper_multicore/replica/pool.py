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
"""Process-group shared guest storage with stream-ordered exclusive leases."""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import contextmanager
from typing import Iterator
from weakref import WeakKeyDictionary

import torch
import torch.distributed as dist

from .capacity import _integer


class ReplicaPool:
    """Reuse B guest slots across layers without retaining parameter snapshots."""

    def __init__(self, weights: tuple[torch.Tensor, ...], slots: int, *, borrow: bool = False) -> None:
        """Allocate guest matrices or borrow externally owned execution views."""
        self.weights = weights if borrow else tuple(weight.new_empty((slots, *weight.shape[1:])) for weight in weights)
        self._borrowed = borrow
        self.gradients = None
        self._lock = threading.Lock()
        self._event = None
        self.projection_ready = None

    def wait_weights(self, matrix_index: int | None = None) -> None:
        """Keep the eager pool compatible with deferred guest consumers."""
        if matrix_index is not None and not 0 <= matrix_index < len(self.weights):
            raise ValueError("Replica matrix index is out of range")

    def _prepare_gradients(self, active_slots: int | None, overwrite: bool) -> None:
        """Allocate FP32 partials and clear only storage that the consumer may read."""
        if self.gradients is None:
            self.gradients = tuple(torch.empty_like(weight, dtype=torch.float32) for weight in self.weights)
        if active_slots != 0 and not overwrite:
            for gradient in self.gradients:
                target = gradient if active_slots is None else gradient[:active_slots]
                target.zero_()

    def _record_consumer(self, backend: object, stream: object) -> None:
        """Protect reusable allocations until the current lease's final consumer."""
        if stream is None:
            return
        self._event = backend.Event()
        self._event.record(stream)
        if not self._borrowed:
            for tensor in self.weights + (self.gradients or ()):
                tensor.record_stream(stream)

    @contextmanager
    def lease(
        self, *, backward: bool = False, active_slots: int | None = None, _overwrite_gradients: bool = False
    ) -> Iterator[ReplicaPool]:
        """Order consumers and clear active guests unless complete overwrites are proven."""
        if active_slots is not None:
            _integer(active_slots, "Active guest slots", 0)
            if active_slots > self.weights[0].shape[0]:
                raise ValueError("Active guest slots exceed the allocated pool")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent expert replica pool leases are unsupported")
        backend, stream = None, None
        try:
            device = self.weights[0].device
            backend = None if device.type == "cpu" else getattr(torch, device.type)
            stream = None if backend is None else backend.current_stream(device)
            if self._event is not None:
                stream.wait_event(self._event)
            if backward:
                self._prepare_gradients(active_slots, _overwrite_gradients)
            yield self
        finally:
            try:
                self._record_consumer(backend, stream)
            finally:
                self._lock.release()


class ReplicaPrefetch:
    """Invocation-owned view that separates local work from guest readiness."""

    def __init__(
        self,
        pool: ReplicaPool,
        wait: Callable[..., None],
        *,
        projection_ready: tuple[tuple[int, ...], int] | None = None,
        weight_generation: object | None = None,
    ) -> None:
        """Borrow leased tensors and expose device ready base/epoch to fused consumers."""
        self.weights = pool.weights
        self.gradients = pool.gradients
        self.projection_ready = projection_ready
        self.weight_generation = weight_generation
        self._wait = wait
        self._waited = set()

    def wait_weights(self, matrix_index: int | None = None) -> None:
        """Order guest reads of one matrix, or every matrix before releasing the slot."""
        if matrix_index is not None and not 0 <= matrix_index < len(self.weights):
            raise ValueError("Replica matrix index is out of range")
        indices = range(len(self.weights)) if matrix_index is None else (matrix_index,)
        for index in indices:
            if index not in self._waited:
                self._wait(index)
                self._waited.add(index)


_POOLS = WeakKeyDictionary()
_REGISTRY_LOCK = threading.Lock()


def replica_pool(weights: tuple[torch.Tensor, ...], slots: int, group: object) -> ReplicaPool:
    """Get one pool per group, device, dtype and expert matrix layout.

    The registry holds no process-group references in its values, so destroying
    the group also releases its pools. Pools never retain owner parameters.
    """
    group = dist.group.WORLD if group is None else group
    if group is None:
        raise ValueError("Replica pools require an initialized process group")
    key = (slots, tuple((weight.device, weight.dtype, tuple(weight.shape[1:])) for weight in weights))
    with _REGISTRY_LOCK:
        pools = _POOLS.setdefault(group, {})
        if key not in pools:
            pools[key] = ReplicaPool(weights, slots)
        return pools[key]
