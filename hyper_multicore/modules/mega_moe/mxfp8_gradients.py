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
"""Projection-complete FP32 returns with bounded helpers in the MX window queue."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import replace
import struct
from typing import Any

import torch

from hyper_multicore.replica.plan import ExpertExecutionPlan
from hyper_multicore.scheduler.mxfp8 import MX_GMM_TASK_TYPE

from .function import _runtime_image
from .kernel_gradients import prepare_kernel_gradient_return
from .mxfp8_gradient_udma import mx_udma_gradient_scratch

MX_GRADIENT_HELPERS = 4


def configure_mx_gradient_helpers(config: Any) -> None:
    """Reserve four even AIV workers without dropping or reordering communication.

    Mathematical tasks are movable because each descriptor retains its logical
    quantization worker. Merge the original queues by window position before
    redistributing; appending displaced producers could create a wait cycle.

    Args:
        config: A completed window2 CSR runtime with 32 Cube and 64 AIV workers.
    """
    if config.task_index_num[3] != -1:
        raise ValueError("MX gradient helpers require the window2 CSR queue")
    queues = [list(config.vector_task_indices[config.mix_task_indices[i]:config.mix_task_indices[i + 1]])
              for i in range(64)]
    math = queues[::2]
    if len({len(queue) for queue in math}) != 1:
        raise ValueError("MX gradient helpers require equal logical mathematics queues")
    work = [queue[index] for index in range(len(math[0])) for queue in math]
    consumers = 32 - MX_GRADIENT_HELPERS
    for lane in range(32):
        queues[2 * lane] = [] if lane < MX_GRADIENT_HELPERS else work[lane - MX_GRADIENT_HELPERS::consumers]
    original = list(config.vector_task_indices[:config.task_index_num[1]])
    flattened = [task for queue in queues for task in queue]
    if sorted(flattened) != sorted(original):
        raise ValueError("MX gradient helpers changed task coverage")
    offset = 0
    for worker, queue in enumerate(queues):
        config.mix_task_indices[worker] = offset
        for task in queue:
            config.vector_task_indices[offset] = task
            offset += 1
    config.mix_task_indices[64] = offset


def mx_projection_events(config: Any, experts: int, outputs: tuple[int, ...]) -> tuple[int, ...]:
    """Extract only complete 32-Cube TN fences from the actual launch graph.

    Args:
        config: Completed backward runtime containing home and guest TN tasks.
        experts: Number of local physical expert slots.
        outputs: Home and guest output positions for one projection.
    """
    events: dict[int, list[int]] = {}
    for task in config.all_tasks[:config.task_num]:
        if task.task_type == MX_GMM_TASK_TYPE and task.outputs[0].input_position in outputs:
            expert = task.task_index // 32
            events.setdefault(expert, []).append(task.trigger_event)
    result = []
    for expert in range(experts):
        producers = events.get(expert, [])
        if len(producers) != 32 or len(set(producers)) != 1 or config.all_event_num_triggers[producers[0]] != 32:
            raise ValueError("MX projection return requires all 32 Cube producers per expert")
        result.append(producers[0])
    return tuple(result)


@contextmanager
def lease_mx_gradient_inbox(provider: Any, weights: tuple[torch.Tensor, ...], route: Any) -> Iterator[Any]:
    """Reuse the existing bounded signal layout without submitting weight transport.

    The provider's BF16 weight views remain unused. Shape-only aliases into its
    own storage bind the existing FP32 suffix without retaining owner parameters.
    Direct access is confined here because the provider's public lease also
    transfers BF16 weights, whereas MX keeps exact FP8 and scale P2P transport.

    Args:
        provider: Workspace-owned signal transport with its reserved symmetric inbox.
        weights: Phase-current MX axes supplying logical matrix dimensions.
        route: Immutable invocation topology shared by the expert group.
    """
    if (route.plan.config.replica_slots_per_rank != provider.slots
            or route.plan.config.ep_size != provider.ep_size):
        raise ValueError("MX gradient inbox topology does not match the replica route")
    if not provider._lock.acquire(blocking=False):
        raise RuntimeError("Concurrent MX gradient inbox leases are unsupported")
    try:
        aliases = tuple(provider.storage[:provider.slots * weight[0].numel() * 2].view(torch.bfloat16).reshape(
            provider.slots, *weight.shape[1:]) for weight in weights)
        provider._bind(aliases)
        provider._advance(reserve=2)
        empty = None
        if isinstance(route.plan, ExpertExecutionPlan):
            home = route.plan.config.home_experts
            counts = route.plan.destination_counts[route.rank]
            guests = route.plan.slot_to_logical[route.rank][home:]
            empty = tuple(slot for slot, expert in enumerate(guests) if expert >= 0 and counts[home + slot] == 0)
        with provider.pool.lease(backward=True, _overwrite_gradients=empty is not None):
            # Complete-K MX TN producers overwrite nonempty matrices. Only zero-row
            # guests return gradients without a producer; inactive slots have no readers.
            for slot in empty or ():
                for gradient in provider.pool.gradients:
                    gradient[slot].zero_()
            yield provider.pool
    finally:
        provider._lock.release()


@contextmanager
def lease_mx_gradient_return(resources: Any, plan: Any, route: Any,
                             gradients: tuple[torch.Tensor, torch.Tensor],
                             weights: tuple[Any, Any]) -> Iterator[Any]:
    """Retain symmetric FP32 destinations and final ACKs through kernel submission.

    Args:
        resources: Serial workspace and its bounded symmetric inbox provider.
        plan: Current physical backward task graph.
        route: Immutable sparse replica transfers for this autograd invocation.
        gradients: Fresh FP32 owner matrices in first/second projection order.
        weights: Phase-current physical weight axes providing logical matrix shapes.
    """
    if not plan.backward.host.kernel_gradients:
        yield plan.backward
        return
    provider, workspace = resources.workspace.replica_provider, resources.workspace
    if route is None or route.home_only or provider is None:
        raise ValueError("MX kernel gradient return requires an active sparse replica route")
    with lease_mx_gradient_inbox(provider, tuple(weight.row_data for weight in weights), route) as pool:
        first, second = pool.gradients
        udma = (replace(mx_udma_gradient_scratch(provider), peer_qps=plan.backward.gradient_udma_qps)
                if plan.spec.mx_gradient_udma else None)
        metadata = prepare_kernel_gradient_return(plan, route, provider, gradients[1], second,
                                                 gradient_w13=gradients[0], guest_w13=first, udma=udma)
        if metadata is None or metadata.matrices != (1, 0):
            raise RuntimeError("MX kernel gradient return requires both complete projection fences")
        home = route.plan.config.home_experts
        # MX weights are already ordered by P2P. The existing envelope's weight
        # readiness fields are unused by MX tasks; only gradient metadata is read.
        ready = tuple(tensor.data_ptr() for tensor in provider.projection_signals)
        content = struct.pack("<II9Q", 0x53505754, 5, home, 0, 0, first.data_ptr(), second.data_ptr(),
                              *ready, 1, metadata.metadata.data_ptr())
        runtime = _runtime_image(plan.backward.runtime, content, workspace.replica_runtime_images, static=False)
        provider.last_kernel_gradient_matrices = metadata.matrices
        yield replace(plan.backward, runtime=runtime)
