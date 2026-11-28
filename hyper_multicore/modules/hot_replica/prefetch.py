# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage one HotReplica invocation's count agreement outside the expert workspace."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from ..mega_moe.route import (
    _INVALID_COUNTS_MASK, _can_fuse_count, _dynamic_route_payload, _permute_and_count,
    _route_input_mask, _start_dynamic_gather, _validate_topk_inputs,
)
from .bounded_placement import decode_placement_wire


@dataclass
class PendingWire:
    """Keep the communication storage and private pinned readback alive until consumption."""

    payload: torch.Tensor
    gathered: torch.Tensor
    work: Any
    host: torch.Tensor
    ready: Any
    permuted: tuple[torch.Tensor, torch.Tensor] | None = None

    def snapshot(self, spec, workspace, stream):
        """Wait only for this readback and publish descriptors for the consumer stream."""
        self.ready.synchronize()
        self.payload.record_stream(stream)
        self.gathered.record_stream(stream)
        if self.permuted is not None:
            for tensor in self.permuted:
                tensor.record_stream(stream)
        wire = self.host.numpy()
        previous = workspace.placement_wire
        if previous is not None and previous.stream == stream and np.array_equal(previous.wire, wire):
            return previous
        # The snapshot owns this allocation via NumPy, including across another branch's readback.
        return decode_placement_wire(wire, spec, workspace, stream)


def _control_stream(resources, device):
    """Order the shared count stream after this invocation's producer."""
    spec = resources.spec
    placement = resources.placement
    if (placement is None or not placement.uses_placement_wire() or not spec.auto_tokens
            or spec.ep_size * spec.num_experts > 16384):
        raise ValueError("Scheduled routing requires bounded HotReplica placement with automatic capacity")
    producer = torch.npu.current_stream(device)
    control = placement.route_control_stream
    if control is None:
        control = placement.route_control_stream = torch.npu.Stream(device=device)
    control.wait_stream(producer)
    return control


def _gather_to_host(payload, spec, control):
    gathered, work = _start_dynamic_gather(payload, spec)
    if work is not None:
        work.wait()  # Orders D2H after HCCL on the control stream.
    host = torch.empty(gathered.shape, dtype=gathered.dtype, device="cpu", pin_memory=True)
    host.copy_(gathered, non_blocking=True)
    ready = torch.npu.Event()
    ready.record(control)
    return PendingWire(payload, gathered, work, host, ready)


def start_replay_vote(record, hidden, ids, probs, counts, resources, mask):
    """Stage the existing vote and preserve the checkpoint mode's replay contract."""
    from .checkpoint import local_replay_vote, resolve_replay_vote

    flag = local_replay_vote(record, hidden, ids, probs, counts, resources.spec, mask)
    control = _control_stream(resources, hidden.device)
    with torch.npu.stream(control):
        flag.record_stream(control)
        pending = _gather_to_host(flag, resources.spec, control)

    def finish():
        pending.ready.synchronize()
        return resolve_replay_vote(record, pending.host.flatten().tolist())

    return finish


def start_wire(hidden, ids, probs, counts, resources, mask):
    """Queue the histogram, EP gather and pinned D2H behind the router's producer."""
    spec = resources.spec
    control = _control_stream(resources, hidden.device)
    with torch.npu.stream(control), torch.inference_mode(False), torch.no_grad():
        _validate_topk_inputs(hidden, ids, probs)
        flat = ids.reshape(-1)
        mask = _route_input_mask(counts, spec, flat.device, mask)
        ids.record_stream(control)
        if counts is not None and not mask & _INVALID_COUNTS_MASK:
            counts.record_stream(control)
        within_limit = hidden.shape[0] * max(spec.hidden_size, spec.top_k) <= 2**31 - 1
        permuted = None
        if within_limit and _can_fuse_count(hidden, spec, counts):
            hidden.record_stream(control)
            permuted, counts = _permute_and_count(hidden, ids, spec)
        _, payload, _ = _dynamic_route_payload(hidden, flat, spec, counts, mask)
        pending = _gather_to_host(payload, spec, control)
        pending.permuted = permuted
        return pending
