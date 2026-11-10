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
"""Bounded Host submission facts, kept separate from uncalibrated Device clocks."""

from __future__ import annotations

import os
from threading import Lock, local
from time import perf_counter_ns
from typing import Any, Callable

from hyper_multicore.replica.recompute import is_recomputing


class ReplicaHostCapture:
    """Retain primitive metadata without extending Tensor or routing graph lifetimes."""

    def __init__(self, enabled: Callable[[], bool], step: Callable[[], int], limit: int = 16384) -> None:
        """Bind one exclusive profiler and a fixed Host record budget."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("Host replica record limit must be a nonnegative integer")
        self._enabled = enabled
        self._step = step
        self._limit = limit
        self._lock = Lock()
        self._thread = local()
        self._next_lease = 0
        self._events: list[dict[str, Any]] = []
        self._dropped = 0

    def is_enabled(self) -> bool:
        """Avoid building Host metadata outside the active capture window."""
        return self._enabled()

    @property
    def current_lease(self) -> int | None:
        """Identify the lease enclosing this thread's kernel submission."""
        return getattr(self._thread, "lease", None)

    def begin_lease(self, route: Any, backward: bool, provider_id: int = 0) -> tuple[int | None, int] | None:
        """Snapshot CPU transfer metadata; never read lazy histogram properties."""
        if not self._enabled():
            return None
        with self._lock:
            identity = self._next_lease
            self._next_lease += 1
        previous = self.current_lease
        self._thread.lease = identity
        transfers = [{key: int(getattr(edge, key)) for key in
                      ("logical_expert", "owner_rank", "owner_slot", "target_rank", "target_slot")}
                     for edge in route.plan.transfers]
        phase = "backward" if backward else "recompute" if is_recomputing() else "forward"
        self.event("LeaseBegin", phase=phase, provider_id=provider_id, ep_rank=route.rank,
                   ep_size=route.plan.config.ep_size,
                   home_experts=route.plan.config.home_experts, transfers=transfers)
        return previous, identity

    def end_lease(self, token: tuple[int | None, int] | None) -> None:
        """Release thread context even when the lease exits exceptionally."""
        if token is None:
            return
        try:
            self.event("LeaseEnd")
        finally:
            self._thread.lease = token[0]

    def begin(self, kind: str, **metadata: Any) -> dict[str, Any] | None:
        """Start a Host API interval, which is not a Device transfer duration."""
        if not self._enabled():
            return None
        return {"kind": kind, "lease_id": self.current_lease, "step": self._step(),
                "start_ns": perf_counter_ns(), "metadata": metadata}

    def end(self, record: dict[str, Any] | None) -> None:
        """Complete at most a fixed number of Host records."""
        if record is None:
            return
        record["end_ns"] = perf_counter_ns()
        with self._lock:
            if len(self._events) >= self._limit:
                self._dropped += 1
            else:
                self._events.append(record)

    def event(self, kind: str, **metadata: Any) -> None:
        """Record a bounded point observation."""
        self.end(self.begin(kind, **metadata))

    def take(self) -> dict[str, Any]:
        """Return one window without implying Host/Device or inter-host clock alignment."""
        with self._lock:
            result = {"schema_version": 1, "clock_domain": f"host_monotonic:pid{os.getpid()}",
                      "events": self._events, "dropped_events": self._dropped,
                      "scope": "Host API submission intervals, not SDMA completion or Device time"}
            self._events = []
            self._dropped = 0
        return result
