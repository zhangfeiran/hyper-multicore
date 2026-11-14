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
from .local_gradients import use_local_gradient_bf16
from .final_gradients import OwnerGradientLayout, allocate_owner_scratch, owner_gradient_layout


@dataclass(frozen=True)
class KernelGradientReturn:
    """Retain ordinary metadata and completion words through kernel submission."""

    metadata: torch.Tensor
    completion: torch.Tensor
    matrices: tuple[int, ...] = (1,)
    resources: tuple[torch.Tensor, ...] = ()
    scratch_resources: tuple[torch.Tensor, ...] = ()


@dataclass(frozen=True)
class _GradientBandReturn:
    """Separate ordinary owner flags from borrowed symmetric guest flags."""

    owner: torch.Tensor
    guest: torch.Tensor
    rows: int


def _validate_gradient_pair(gradient: torch.Tensor, guest: torch.Tensor | None, *, local_bf16: bool = False) -> None:
    """Reject borrowed, cast-requiring or incompatible gradient storage."""
    expected = torch.bfloat16 if local_bf16 else torch.float32
    if gradient.dtype != expected or gradient.requires_grad or (not gradient.is_contiguous()):
        name = "BF16" if local_bf16 else "FP32"
        raise ValueError(f"Kernel gradient return requires detached contiguous {name} owner gradients")
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
    scratch: torch.Tensor | None = None,
    layout: OwnerGradientLayout | None = None,
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
        values[0] = 11 if gradient.dtype == torch.bfloat16 else 7
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
    if layout is not None:
        values[0] = 13
        mapping_offset = 24 + len(incoming) * 3 + len(owned) * 4
        values.extend((scratch.data_ptr(), home, len(layout.owners), mapping_offset,
                       mapping_offset + home, mapping_offset + home + len(owned)))
    for item in incoming:
        values.extend((item.owner_rank, item.target_slot - home, events[item.target_slot]))
    for item in owned:
        values.extend((item.target_rank, item.target_slot - home, item.owner_slot, events[item.owner_slot]))
    if layout is not None:
        values.extend(layout.home_to_scratch)
        values.extend(layout.last_edges)
    return values


def _prepare_gradient_storage(
    plan: MegaMoePlan, provider: Any, pairs: list[tuple]
) -> tuple[tuple[_GradientBandReturn | None, ...], torch.Tensor]:
    """Own disjoint flags in one zeroed allocation before reserving any epochs."""
    cores = plan.spec.num_cube_cores
    completion_shape = len(pairs), cores, 16
    device = pairs[0][1].device
    fallback = (None,) * len(pairs)
    tables = []
    if (getattr(provider, "gradient_bands", False) and len(pairs) == 2
            and all(owner.ndim == 3 for _, owner, _ in pairs)):
        tables = [provider.kernel_gradient_band_signals(matrix, cores) for matrix in (1, 0)]
    if not tables or any(table is None for table in tables):
        return fallback, torch.zeros(completion_shape, dtype=torch.int32, device=device)
    completion_elements = len(pairs) * cores * 16
    lengths = [owner.shape[0] * table[2] * cores * 16 for (_, owner, _), table in zip(pairs, tables)]
    storage = torch.zeros(completion_elements + sum(lengths), dtype=torch.int32, device=device)
    completion = storage[:completion_elements].view(completion_shape)
    offset = completion_elements
    result = []
    for (_, owner, _), (guest_flags, band_rows, bands), length in zip(pairs, tables, lengths):
        flags = storage[offset:offset + length].view(owner.shape[0], bands, cores, 16)
        result.append(_GradientBandReturn(flags, guest_flags, band_rows))
        offset += length
    return tuple(result), completion


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
    pairs = _gradient_projection_pairs(plan, route, provider, gradient, guest, gradient_w13, guest_w13)
    if pairs is None:
        return None
    local_bf16 = gradient.dtype == torch.bfloat16
    _validate_return_projections(plan, route, provider, pairs, local_bf16)
    bands, completion = _prepare_gradient_storage(plan, provider, pairs)
    if local_bf16 and any(band is None for band in bands):
        raise ValueError("Local BF16 return cannot fall back to full-expert gradient publication")
    layout, scratch_resources = _prepare_owner_scratch(plan, route, pairs, local_bf16)
    records = [
        _matrix_record(route, provider, events, owner, replica, completion[index], bands[index],
                       scratch_resources[index] if layout else None, layout)
        for index, (events, owner, replica) in enumerate(pairs)
    ]
    values = records[0] if len(records) == 1 else [2, 3, 3 + len(records[0])] + records[0] + records[1]
    data = struct.pack("<" + "Q" * len(values), *values)
    metadata = _async_device_bytes(data, gradient.device)
    if len(records) == 1:
        completion = completion[0]
    stream = torch.npu.current_stream(gradient.device)
    resources = tuple((band.owner for band in bands if band is not None))
    for tensor in (metadata, completion, *resources, *scratch_resources):
        tensor.record_stream(stream)
    return KernelGradientReturn(metadata, completion, (1,) if len(records) == 1 else (1, 0), resources,
                                scratch_resources)


def _gradient_projection_pairs(plan, route, provider, gradient, guest, gradient_w13, guest_w13):
    """Select only producer-backed projections before allocating return storage."""
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
    return pairs


def _validate_return_projections(plan, route, provider, pairs, local_bf16):
    """Validate every output and guest pair under one shared two-projection choice."""
    if local_bf16 and (
        len(pairs) != 2 or not use_local_gradient_bf16(plan, route, provider, (pairs[1][1], pairs[0][1]))
    ):
        raise ValueError("Kernel gradient return requires detached contiguous FP32 owner gradients; "
                         "local BF16 requires both band projections and no owned remote gradient edges")
    for _, owner, replica in pairs:
        _validate_gradient_pair(owner, replica, local_bf16=local_bf16)


def _prepare_owner_scratch(plan, route, pairs, local_bf16):
    """Allocate only the current route's actually owned FP32 matrices."""
    final_bf16 = local_bf16 and getattr(plan.spec, "_final_gradient_bf16", False)
    layout = owner_gradient_layout(route) if final_bf16 else None
    scratch_resources = tuple(allocate_owner_scratch(route, owner, layout) for _, owner, _ in pairs) if layout else ()
    return layout, scratch_resources
