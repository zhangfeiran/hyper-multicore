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
"""Exact MX projection P2P on a private stream with SDMA-published readiness."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import struct
from typing import Any

import torch
import torch.distributed as dist

from hyper_multicore.replica._metadata import _async_device_bytes
from hyper_multicore.replica.pool import replica_pool
from hyper_multicore.replica.transport import _global_peer_rank

from .mxfp8_weights import MXFP8PhysicalWeight


def _current_axes(weights: tuple[MXFP8PhysicalWeight, ...], backward: bool) -> tuple[torch.Tensor, ...]:
    """Read only the current phase's exact data and scale bytes."""
    fields = ("row_data", "row_scale") if backward else ("col_data", "col_scale")
    originals = tuple(getattr(weight, field) for weight in weights for field in fields)
    if any(tensor is None for tensor in originals):
        raise ValueError("MX projection prefetch requires every current data/scale axis")
    return originals


def _bind_ready_storage(provider: Any, weights: tuple[torch.Tensor, ...]) -> None:
    """Bind only shape aliases; the existing bounded BF16 weight views remain unused."""
    aliases = tuple(provider.storage[:provider.slots * weight[0].numel() * 2].view(torch.bfloat16).reshape(
        provider.slots, *weight.shape[1:]) for weight in weights)
    provider._bind(aliases)


def _projection_operations(aliases: tuple[torch.Tensor, ...], pool: Any, route: Any, matrix: int) -> list[Any]:
    """Keep all ranks in one projection/data-scale/transfer P2P order."""
    operations = []
    home = route.plan.config.home_experts
    for transfer in route.plan.transfers:
        for index in (2 * matrix, 2 * matrix + 1):
            if route.rank == transfer.owner_rank:
                peer = _global_peer_rank(route.group, transfer.target_rank)
                operations.append(dist.P2POp(dist.isend, aliases[index][transfer.owner_slot], peer, route.group))
            elif route.rank == transfer.target_rank:
                peer = _global_peer_rank(route.group, transfer.owner_rank)
                operations.append(dist.P2POp(dist.irecv, pool.weights[index][transfer.target_slot - home],
                                             peer, route.group))
    return operations


class MXProjectionPrefetch:
    """Own one reusable copy stream without retaining FSDP weight generations."""

    def __init__(self) -> None:
        """Defer device stream creation until the first active guest invocation."""
        self.stream = None

    @contextmanager
    def lease(self, weights: tuple[MXFP8PhysicalWeight, ...], route: Any,
              provider: Any, *, backward: bool) -> Iterator[tuple[tuple[torch.Tensor, ...], torch.Tensor]]:
        """Publish each received projection through a local SDMA copy, then join all consumers.

        Args:
            weights: Current FSDP physical axes; no snapshots are cached across phases.
            route: Exact immutable sparse transfers for this invocation.
            provider: Serial workspace-owned symmetric signal and FP32 inbox.
            backward: Select fresh row axes and second-projection-first copy order.
        """
        originals = _current_axes(weights, backward)
        aliases = tuple(tensor.view(torch.uint8).reshape(tensor.shape[0], -1) for tensor in originals)
        _bind_ready_storage(provider, originals[::2])
        epoch = provider._advance(reserve=3)
        ready = provider.projection_signals
        content = struct.pack("<5Q", epoch, ready[0].data_ptr(), ready[1].data_ptr(),
                              route.plan.config.home_experts, route.plan.config.replica_slots_per_rank)
        information = _async_device_bytes(content, originals[0].device)
        stamp = _async_device_bytes(struct.pack("<i", epoch), originals[0].device).view(torch.int32)
        backend = torch.npu
        caller = backend.current_stream(originals[0].device)
        if self.stream is None:
            self.stream = backend.Stream(device=originals[0].device)
        pool = replica_pool(aliases, route.plan.config.replica_slots_per_rank, route.group)
        with pool.lease():
            # Zero-row guest tasks return before reading weights. Current receives
            # overwrite every byte of active slots before their readiness is published.
            active = {item.target_slot - route.plan.config.home_experts for item in route.plan.transfers
                      if item.target_rank == route.rank}
            start = backend.Event()
            start.record(caller)
            self.stream.wait_event(start)
            # Request handles outlive submission and the native consumer lease.
            work = []
            try:
                with backend.stream(self.stream):
                    for matrix in ((1, 0) if backward else (0, 1)):
                        operations = _projection_operations(aliases, pool, route, matrix)
                        if operations:
                            requests = dist.batch_isend_irecv(operations)
                            work.extend(requests)
                            for request in requests:
                                request.wait()
                        for slot in sorted(active):
                            provider.runtime.put(ready[matrix, slot, :1], stamp, route.rank, use_sdma=True)
                    for tensor in (*aliases, *pool.weights, stamp):
                        tensor.record_stream(self.stream)
                guests = tuple(guest.view(tensor.dtype).reshape(
                    route.plan.config.replica_slots_per_rank, *tensor.shape[1:])
                    for tensor, guest in zip(originals, pool.weights))
                yield guests, information
            finally:
                done = backend.Event()
                done.record(self.stream)
                caller.wait_event(done)
