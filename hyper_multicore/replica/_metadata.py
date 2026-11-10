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
"""Owned asynchronous uploads for small replica protocol descriptors."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import os
from threading import RLock
from typing import Any

import torch

from hyper_multicore.replica.recompute import get_recompute_state

from .plan import ExpertExecutionPlan

_MAX_COUNT_HEADERS = 8


class _ReplicaCountHeaderCache:
    """Keep exact control bytes immutable within one executor resource lifecycle."""

    def __init__(self) -> None:
        """Retain only bounded header tensors and their asynchronous upload events."""
        self.entries: OrderedDict[tuple[torch.device, bytes], tuple[torch.Tensor, Any | None]] = OrderedDict()
        self._lock = RLock()

    def clear(self) -> None:
        """Release owned references while consumer streams retain outstanding reads."""
        with self._lock:
            self.entries.clear()

    def upload(self, data: bytes, device: torch.device) -> torch.Tensor:
        """Reuse identical bytes after ordering the current stream behind their upload.

        Args:
            data: Immutable serialized control fields for the current invocation.
            device: Concrete route device retaining the header buffer.

        Returns:
            Owned read-only bytes ready for consumption on the current stream.
        """
        with self._lock:
            device = torch.device("cpu") if device.type == "cpu" else device
            key = device, data
            stream = None if device.type == "cpu" else getattr(torch, device.type).current_stream(device)
            entry = self.entries.get(key)
            if entry is not None:
                value, ready = entry
                if ready is not None:
                    stream.wait_event(ready)
                    value.record_stream(stream)
                self.entries.move_to_end(key)
                return value
            value = _async_device_bytes(data, device)
            ready = None
            if stream is not None:
                ready = getattr(torch, device.type).Event()
                ready.record(stream)
            self.entries[key] = value, ready
            while len(self.entries) > _MAX_COUNT_HEADERS:
                self.entries.popitem(last=False)
            return value


@lru_cache(maxsize=1)
def _metadata_reuse_enabled() -> bool:
    """Keep the unvalidated device experiment behind one internal switch."""
    return os.environ.get("HYPER_PARALLEL_REPLICA_METADATA_REUSE", "0") == "1"


@dataclass(frozen=True)
class _UploadedRouteMetadata:
    """Retain only an immutable CPU plan, integer descriptors and upload readiness."""

    plan: ExpertExecutionPlan
    values: tuple[torch.Tensor, ...]
    ready: Any | None


class _RecomputedRouteMetadata:
    """Reuse descriptors only after the existing exact CPU plan replay succeeds."""

    def __init__(self) -> None:
        """Initialize an invocation-local registry without retaining activations."""
        self.entries: dict[tuple[int, int, torch.device, str], _UploadedRouteMetadata] = {}

    def clear(self) -> None:
        """Release cache references while saved graphs and stream events retain their uses."""
        self.entries.clear()

    @staticmethod
    def _key(plan: ExpertExecutionPlan, rank: int, device: torch.device, kind: str) -> tuple:
        """Normalize CPU aliases while retaining accelerator device indices."""
        device = torch.device("cpu") if device.type == "cpu" else device
        return id(plan), rank, device, kind

    def lookup(self, plan: ExpertExecutionPlan, rank: int, device: torch.device,
               kind: str) -> tuple[torch.Tensor, ...] | None:
        """Order the replay stream after upload and retain every asynchronous read."""
        entry = self.entries.get(self._key(plan, rank, device, kind))
        if entry is None or entry.plan is not plan:
            return None
        if device.type != "cpu":
            stream = getattr(torch, device.type).current_stream(device)
            stream.wait_event(entry.ready)
            for value in entry.values:
                value.record_stream(stream)
        return entry.values

    def remember(self, plan: ExpertExecutionPlan, rank: int, device: torch.device,
                 kind: str, values: tuple[torch.Tensor, ...]) -> None:
        """Publish exclusively owned integer metadata after its current-stream upload."""
        if any(value.requires_grad or value.grad_fn is not None or value.dtype not in (torch.int32, torch.int64)
               for value in values):
            raise ValueError("Replica metadata reuse requires graph-free integer descriptors")
        ready = None
        if device.type != "cpu":
            backend = getattr(torch, device.type)
            ready = backend.Event()
            ready.record(backend.current_stream(device))
        self.entries[self._key(plan, rank, device, kind)] = _UploadedRouteMetadata(plan, values, ready)


def _recompute_metadata(plan: ExpertExecutionPlan) -> tuple[_RecomputedRouteMetadata | None, bool]:
    """Find a cache only for enabled positive-B checkpoint invocations."""
    if not plan.config.replica_slots_per_rank or not _metadata_reuse_enabled():
        return None, False
    state = get_recompute_state()
    if state is None:
        return None, False
    return state.get_resource(_RecomputedRouteMetadata, _RecomputedRouteMetadata), state.is_recomputing


def _owned_host_bytes(data: bytes, device: torch.device) -> torch.Tensor:
    """Keep asynchronous sources alive through the pinned allocator's copy event."""
    source = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    if device.type != "cpu":
        source = source.pin_memory(device=device.type)
    return source


def _async_device_bytes(data: bytes, device: torch.device) -> torch.Tensor:
    """Upload owned pinned bytes without draining earlier work on the device stream."""
    return _owned_host_bytes(data, device).to(device, non_blocking=True)


def _copy_device_bytes(data: bytes, destination: torch.Tensor) -> None:
    """Update a leased device suffix directly without allocating a device intermediate."""
    destination.copy_(_owned_host_bytes(data, destination.device), non_blocking=True)
