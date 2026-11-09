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
"""Invocation metadata for ordered projection returns inside fused backward."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

import torch

from hyper_multicore.replica._metadata import _async_device_bytes
from hyper_multicore.replica.routing import ReplicaRoute

from .plan import MegaMoePlan


@dataclass(frozen=True)
class KernelGradientReturn:
    """Retain ordinary metadata and completion words through kernel submission."""

    metadata: torch.Tensor
    completion: torch.Tensor
    matrices: tuple[int, ...] = (1,)
    resources: tuple[torch.Tensor, ...] = ()


@dataclass(frozen=True)
class _GradientBandReturn:
    """Separate ordinary owner flags from borrowed symmetric guest flags."""

    owner: torch.Tensor
    guest: torch.Tensor
    rows: int


def _validate_gradient_pair(gradient: torch.Tensor, guest: torch.Tensor | None) -> None:
    """Reject borrowed, cast-requiring or incompatible gradient storage."""
    if gradient.dtype != torch.float32 or gradient.requires_grad or (not gradient.is_contiguous()):
        raise ValueError("Kernel gradient return requires detached contiguous FP32 owner gradients")
    if (
        guest is None
        or guest.dtype != torch.float32
        or guest.requires_grad
        or (not guest.is_contiguous())
        or (guest.device != gradient.device)
        or (guest.shape[1:] != gradient.shape[1:])
    ):
        raise ValueError("Kernel gradient return requires matching detached contiguous FP32 guest gradients")


def _matrix_record(
    route: ReplicaRoute,
    provider: Any,
    events: tuple[int, ...],
    gradient: torch.Tensor,
    guest: torch.Tensor,
    completion: torch.Tensor,
    bands: _GradientBandReturn | None = None,
) -> list[int]:
    """Encode one epoch with complete local producers before ordered peer reads."""
    home = route.plan.config.home_experts
    incoming = [item for item in route.plan.transfers if item.target_rank == route.rank]
    owned = sorted(
        (item for item in route.plan.transfers if item.owner_rank == route.rank), key=lambda item: item.target_rank
    )
    epoch, ready, ack = provider.kernel_gradient_signals()
    values = [
        1,
        route.rank,
        route.plan.config.replica_slots_per_rank,
        epoch,
        gradient[0].numel(),
        gradient.data_ptr(),
        guest.data_ptr(),
        ready,
        ack,
        completion.data_ptr(),
        len(incoming),
        len(owned),
    ]
    if bands is not None:
        values[0] = 7
        values.extend(
            (
                bands.owner.data_ptr(),
                bands.guest.data_ptr(),
                bands.rows,
                gradient.shape[-2],
                gradient.shape[-1],
                bands.owner.shape[1],
            )
        )
    for item in incoming:
        values.extend((item.owner_rank, item.target_slot - home, events[item.target_slot]))
    for item in owned:
        values.extend((item.target_rank, item.target_slot - home, item.owner_slot, events[item.owner_slot]))
    return values


def _prepare_gradient_bands(
    plan: MegaMoePlan, provider: Any, pairs: list[tuple]
) -> tuple[_GradientBandReturn | None, ...]:
    """Select the band protocol before advancing either shared ready/ACK epoch."""
    fallback = (None,) * len(pairs)
    if not getattr(provider, "gradient_bands", False) or len(pairs) != 2:
        return fallback
    if any((owner.ndim != 3 for _, owner, _ in pairs)):
        return fallback
    cores = plan.spec.num_cube_cores
    tables = [provider.kernel_gradient_band_signals(matrix, cores) for matrix in (1, 0)]
    if any((table is None for table in tables)):
        return fallback
    result = []
    for (_, owner, _), (guest_flags, band_rows, bands) in zip(pairs, tables):
        flags = torch.zeros((owner.shape[0], bands, cores, 16), dtype=torch.int32, device=owner.device)
        result.append(_GradientBandReturn(flags, guest_flags, band_rows))
    return tuple(result)


def prepare_kernel_gradient_return(
    plan: MegaMoePlan,
    route: ReplicaRoute | None,
    provider: Any,
    gradient: torch.Tensor,
    guest: torch.Tensor | None,
    *,
    gradient_w13: torch.Tensor | None = None,
    guest_w13: torch.Tensor | None = None,
) -> KernelGradientReturn | None:
    """Encode proven projection fences and retain FP32 peer order and ACK leases."""
    if (
        route is None
        or not route.plan.transfers
        or (not plan.replica_w2_events)
        or (not getattr(provider, "kernel_gradients", False))
    ):
        return None
    pairs = [(plan.replica_w2_events, gradient, guest)]
    w13_events = getattr(plan, "replica_w13_events", ())
    if w13_events and gradient_w13 is not None:
        pairs.append((w13_events, gradient_w13, guest_w13))
    for _, owner, replica in pairs:
        _validate_gradient_pair(owner, replica)
    bands = _prepare_gradient_bands(plan, provider, pairs)
    completion = torch.zeros((len(pairs), plan.spec.num_cube_cores, 16), dtype=torch.int32, device=gradient.device)
    records = [
        _matrix_record(route, provider, events, owner, replica, completion[index], bands[index])
        for index, (events, owner, replica) in enumerate(pairs)
    ]
    values = records[0] if len(records) == 1 else [2, 3, 3 + len(records[0])] + records[0] + records[1]
    data = struct.pack("<" + "Q" * len(values), *values)
    metadata = _async_device_bytes(data, gradient.device)
    if len(records) == 1:
        completion = completion[0]
    stream = torch.npu.current_stream(gradient.device)
    resources = tuple((band.owner for band in bands if band is not None))
    for tensor in (metadata, completion, *resources):
        tensor.record_stream(stream)
    return KernelGradientReturn(metadata, completion, (1,) if len(records) == 1 else (1, 0), resources)
