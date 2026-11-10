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
"""Exact byte transport for MX weight axes and FP32 owner-gradient reduction."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Iterator

import torch

from hyper_multicore.replica.routing import ReplicaRoute
from hyper_multicore.replica.transport import prefetch_weights, return_gradients

from .mxfp8_weights import MXFP8PhysicalWeight


@contextmanager
def lease_replica_weights(
    weights: tuple[MXFP8PhysicalWeight, ...], route: ReplicaRoute | None, *, backward: bool
) -> Iterator[tuple[torch.Tensor, ...]]:
    """Borrow phase-current guest buffers until the caller submits every consumer.

    Home tasks read the original FSDP storage directly. Guest tasks use separate
    buffers in weight/data-scale order, without copying home matrices. The pool
    records its consumer-stream event only after the native launch is submitted.

    Args:
        weights: Current physical owner weight axes, never cached across FSDP phases.
        route: Immutable replica plan and exact physical dispatch metadata.
    """
    with ExitStack() as leases:
        guests = ()
        if route is not None and not route.home_only:
            fields = ("row_data", "row_scale") if backward else ("col_data", "col_scale")
            originals = []
            for weight in weights:
                for field in fields:
                    tensor = getattr(weight, field)
                    if tensor is None:
                        raise ValueError(f"MXFP8 replica phase is missing {field}.")
                    originals.append(tensor)
            aliases = tuple(tensor.view(torch.uint8).reshape(tensor.shape[0], -1) for tensor in originals)
            config = route.plan.config
            pool = leases.enter_context(prefetch_weights(aliases, route))
            # Empty guest tasks never read weights; prefetch fully overwrites active slots.
            guests = tuple(
                guest.view(tensor.dtype).reshape((config.replica_slots_per_rank, *tensor.shape[1:]))
                for tensor, guest in zip(originals, pool.weights)
            )
        yield guests


def reduce_replica_gradients(
    gradients: tuple[torch.Tensor, ...], route: ReplicaRoute | None
) -> tuple[torch.Tensor, ...]:
    """Merge guest partials in FP32 and return only the logical owner matrices.

    Args:
        gradients: Fresh FP32 physical partials, including active guests.
        route: Immutable replica plan and exact physical dispatch metadata.
    """
    if route is None or route.home_only:
        return gradients
    if any(gradient.dtype != torch.float32 for gradient in gradients):
        raise TypeError("MXFP8 replica owner reduction requires FP32 gradient partials.")
    return return_gradients(gradients, route, consume=True)
