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
"""Workspace-owned exact plans and immutable integer route descriptors."""

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, NamedTuple

import torch

from .recompute import get_recompute_state


class _CountSnapshot(NamedTuple):
    """Bind validated controls to one immutable full-wire count snapshot."""

    counts: torch.Tensor
    raw: tuple
    summary: tuple | None
    rows: tuple | None
    defer_receive_limit: bool


def _freeze_controls(value: Any) -> tuple:
    """Preserve scalar types and ordered quota seeds in a private planner key."""
    if isinstance(value, (tuple, list)):
        return type(value), tuple(_freeze_controls(item) for item in value)
    if isinstance(value, dict):
        return dict, tuple((key, _freeze_controls(item)) for key, item in value.items())
    if is_dataclass(value):
        return type(value), tuple((field.name, _freeze_controls(getattr(value, field.name))) for field in fields(value))
    return type(value), value


def _integer_tensors(value: Any) -> tuple[torch.Tensor, ...]:
    """Find retained descriptor views without retaining activations or parameters."""
    tensors, pending = [], [value]
    while pending:
        item = pending.pop()
        if isinstance(item, torch.Tensor):
            if item.requires_grad or item.grad_fn is not None or item.dtype not in (torch.int32, torch.int64):
                raise ValueError("Replica descriptor reuse requires graph-free integer tensors")
            tensors.append(item)
        elif isinstance(item, (tuple, list)):
            pending.extend(item)
        elif is_dataclass(item):
            pending.extend(getattr(item, field.name) for field in fields(item))
    return tuple(tensors)


@dataclass
class _RouteData:
    """An immutable descriptor and the completion event of its producing stream."""

    key: Any
    value: Any
    tensors: tuple[torch.Tensor, ...]
    stream: Any = None
    ready: Any = None

    @classmethod
    def capture(cls, key: Any, value: Any) -> "_RouteData":
        """Record upload completion on the stream producing all descriptor views."""
        result = cls(key, value, _integer_tensors(value))
        if result.tensors and result.tensors[0].device.type != "cpu":
            device = result.tensors[0].device
            backend = getattr(torch, device.type)
            result.stream = backend.current_stream(device)
            result.ready = backend.Event()
            result.ready.record(result.stream)
        return result

    def consume(self) -> Any:
        """Order cross-stream readers and retain each allocation through their use."""
        if self.ready is not None:
            device = self.tensors[0].device
            current = getattr(torch, device.type).current_stream(device)
            if current != self.stream:
                current.wait_event(self.ready)
                for tensor in self.tensors:
                    tensor.record_stream(current)
        return self.value


class _ReplicaRouteCache:
    """Keep one full CPU plan and one entry for each bounded descriptor kind."""

    def __init__(self) -> None:
        """Initialize empty, bounded plan and descriptor ownership."""
        self.plan_key = self.plan = None
        self.entries: dict[str, _RouteData] = {}
        self.count_table: _RouteData | None = None
        self.hits: dict[str, int] = {}
        self.misses: dict[str, int] = {}

    def clear(self) -> None:
        """Release cache ownership; saved graphs and stream records retain live uses."""
        self.plan_key = self.plan = None
        self.entries.clear()
        self.count_table = None
        self.hits.clear()
        self.misses.clear()

    @staticmethod
    def make_plan_key(counts: Any, config: Any, group: Any, rank: int, options: dict) -> tuple:
        """Compare all freshly validated source counts and every planning control."""
        return id(group), rank, config, tuple(map(tuple, counts)), _freeze_controls(options)

    def lookup_plan(self, key: tuple) -> Any:
        """Reuse only an exact decision; checkpoint replay keeps its existing registry."""
        if get_recompute_state() is None and self.plan_key == key:
            self.hits["plan"] = self.hits.get("plan", 0) + 1
            return self.plan
        return None

    def remember_plan(self, key: tuple, plan: Any) -> None:
        """Bound plan retention and discard descriptors after a planning change."""
        self.plan_key, self.plan = key, plan
        self.entries.clear()
        self.misses["plan"] = self.misses.get("plan", 0) + 1

    def get_data(self, kind: str, key: Any, factory: Callable[[], Any]) -> Any:
        """Reuse immutable descriptors only when an exact CPU key is available."""
        if key is None or get_recompute_state() is not None:
            return factory()
        entry = self.entries.get(kind)
        if entry is not None and entry.key == key:
            self.hits[kind] = self.hits.get(kind, 0) + 1
            return entry.consume()
        value = factory()
        self.entries[kind] = _RouteData.capture(key, value)
        self.misses[kind] = self.misses.get(kind, 0) + 1
        return value

    def previous_counts(self, key: Any) -> Any:
        """Borrow immutable counts only for the same device and routing topology."""
        if self.count_table is not None and self.count_table.key == key:
            return self.count_table.consume()
        return None

    def remember_counts(
        self, key: Any, counts: torch.Tensor, raw: list[list[int]], *, summary: tuple | None = None,
        rows: list[list[int]] | None = None, defer_receive_limit: bool = False,
    ) -> None:
        """Retain one validated device table and a separately owned CPU snapshot."""
        value = _CountSnapshot(
            counts, tuple(tuple(row) for row in raw), summary,
            None if rows is None else tuple(tuple(row) for row in rows), defer_receive_limit,
        )
        self.count_table = _RouteData.capture(key, value)
        self.misses["counts"] = self.misses.get("counts", 0) + 1
