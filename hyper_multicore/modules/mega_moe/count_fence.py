# Copyright 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Readiness contract for Push forward after the expert count exchange."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .workspace import MegaMoeWorkspace

COUNT_FENCE_BIT = 1 << 10


@dataclass(frozen=True)
class CountFence:
    """Track the exact SHMEM buffers cleared before the EP count exchange."""

    events: torch.Tensor
    receive: torch.Tensor
    combine: torch.Tensor
    lease_generation: int
    stream: Any

    @classmethod
    def before_gather(cls, workspace: MegaMoeWorkspace, stream: Any, *, launch_stream: Any = None):
        """Prepare counters when already bound and order them after the last lease."""
        if (not workspace.forward_ready_initialized or workspace.forward_event_counters is None
                or workspace.expert_buffer is None or workspace.routed_buffer is None or workspace.in_use):
            return None
        workspace.wait_for_reuse(stream=stream)
        events = workspace.prepare_event_counters(forward=True)
        return cls(events, workspace.expert_buffer, workspace.routed_buffer, workspace.lease_generation,
                   stream if launch_stream is None else launch_stream)

    def matches(self, workspace: MegaMoeWorkspace) -> bool:
        """A capacity restart replaces these symmetric allocations on every EP rank."""
        return (workspace.forward_event_counters is self.events and workspace.expert_buffer is self.receive
                and workspace.routed_buffer is self.combine and workspace.forward_ready_initialized)

    def claim_events(self, workspace: MegaMoeWorkspace, counter_bytes: int | None) -> torch.Tensor:
        """Reuse the collective's counters for the immediately following lease."""
        if (not self.matches(workspace) or workspace.lease_generation != self.lease_generation + 1
                or workspace.lease_stream != self.stream):
            raise RuntimeError("MegaMoE workspace lease changed after route agreement")
        offset = 0 if counter_bytes is None else workspace.event_counter_bytes - counter_bytes
        if not 0 <= offset < workspace.event_counter_bytes:
            raise ValueError("MegaMoE event counters are outside the reserved workspace")
        return self.events[offset:] if offset else self.events
