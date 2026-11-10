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
"""Sparse eager weight prefetch and bounded FP32 gradient return."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.distributed as dist

from .pool import ReplicaPool, ReplicaPrefetch, replica_pool
from .routing import ReplicaRoute


def _exchange(operations: list) -> None:
    """Order every tensor consumer after the asynchronous P2P operations."""
    if operations:
        for work in dist.batch_isend_irecv(operations):
            work.wait()


def _global_peer_rank(group: object, peer: int) -> int:
    """Default-group ranks already are global; subgroup ranks need translation."""
    return peer if group is None else dist.get_global_rank(group, peer)


@contextmanager
def prefetch_weights(weights: tuple[torch.Tensor, ...], route: ReplicaRoute,
                     *, backward: bool = False, provider: object = None,
                     overlap: bool = False,
                     reuse_generation: object | None = None,
                     _overwrite_gradients: bool = False) -> Iterator[ReplicaPool | ReplicaPrefetch]:
    """Borrow home weights and fill only the shared B guest slots.

    Ordinary pool allocations remain valid across push SHMEM heap growth.
    The lease spans all consumers, including gradient return in backward.
    With overlap=True, an enabled provider may defer guest readiness. Call
    wait_weights(matrix_index) before each guest projection read, or honor the
    projection_ready metadata in a fused kernel.

    A backward may pass the opaque weight_generation saved from its own forward
    view, after autograd checks the saved parameter versions. Every rank must use
    the same invocation sequence. A provider accepts reuse only while that exact
    generation remains resident; receiver-local cache decisions are unsafe.
    Signal owners may skip unchanged payload puts using global slot-write history,
    while preserving every current credit, readiness publication and acknowledgement.
    The internal overwrite flag requires complete-K stores to every active guest
    gradient before publication. Other consumers retain zero initialization.
    """
    if reuse_generation is not None and not backward:
        raise ValueError("Replica weights may be reused only by their saved backward context")
    provider = route.transport if provider is None else provider
    if provider is not None and callable(getattr(provider, "lease", None)):
        options = {"overlap": True} if overlap else {}
        if reuse_generation is not None:
            options["reuse_generation"] = reuse_generation
        if _overwrite_gradients:
            options["_overwrite_gradients"] = True
        with provider.lease(weights, route, backward=backward, **options) as pool:
            yield pool
        return
    config = route.plan.config
    if not dist.is_initialized() and config.ep_size == 1 and not route.plan.transfers:
        pool = ReplicaPool(weights, config.replica_slots_per_rank)
    else:
        pool = replica_pool(weights, config.replica_slots_per_rank, route.group)
    with pool.lease(backward=backward):
        if provider is not None:
            provider.prefetch(weights, pool.weights, route)
            yield pool
            return
        operations = []
        for transfer in route.plan.transfers:
            for weight, output in zip(weights, pool.weights):
                if route.rank == transfer.owner_rank:
                    peer = _global_peer_rank(route.group, transfer.target_rank)
                    operations.append(dist.P2POp(
                        dist.isend, weight[transfer.owner_slot].contiguous(), peer, route.group))
                elif route.rank == transfer.target_rank:
                    peer = _global_peer_rank(route.group, transfer.owner_rank)
                    guest = output[transfer.target_slot - config.home_experts]
                    operations.append(dist.P2POp(dist.irecv, guest, peer, route.group))
        _exchange(operations)
        yield pool


def _gradient_accumulators(gradients: tuple[torch.Tensor, ...], *, consume: bool) -> tuple[torch.Tensor, ...]:
    """Reuse explicitly owned FP32 buffers or preserve borrowed inputs by copying."""
    if not consume:
        return tuple(gradient.float().clone() for gradient in gradients)
    if any(gradient.dtype != torch.float32 or gradient.requires_grad for gradient in gradients):
        raise ValueError("Consumed owner gradients must be detached FP32 tensors")
    return gradients


def return_gradients(gradients: tuple[torch.Tensor, ...], route: ReplicaRoute,
                     guests: tuple[torch.Tensor, ...] | None = None,
                     provider: object = None, *, consume: bool = False) -> tuple[torch.Tensor, ...]:
    """Accumulate guest partials in FP32 before returning owner gradients.

    One target rank returns at a time. Each owner needs at most B incoming
    gradient slots, independent of the number of EP ranks or remote replicas.
    All ranks execute rounds in identical order, including ranks with no work.

    With consume=True, the caller transfers exclusive ownership of detached
    FP32 home gradients. These must be fresh invocation buffers, disjoint from
    parameters, saved tensors, guest slots, and other live gradient buffers.
    The caller must use the returned tensors and stop using the original values.
    Providers opt in through return_gradients_owned; legacy providers retain
    their original three-argument, non-consuming call.
    """
    provider = route.transport if provider is None else provider
    if provider is not None:
        if consume and callable(getattr(provider, "return_gradients_owned", None)):
            return provider.return_gradients_owned(gradients, guests, route)
        return provider.return_gradients(gradients, guests, route)
    home = route.plan.config.home_experts
    result = _gradient_accumulators(tuple(gradient[:home] for gradient in gradients), consume=consume)
    for target in range(route.plan.config.ep_size):
        operations = []
        incoming = []
        outgoing = []
        for transfer in route.plan.transfers:
            if transfer.target_rank != target:
                continue
            for index, gradient in enumerate(gradients):
                if route.rank == target:
                    value = (gradient[transfer.target_slot] if guests is None else
                             guests[index][transfer.target_slot - home]).float().contiguous()
                    outgoing.append(value)
                    peer = _global_peer_rank(route.group, transfer.owner_rank)
                    operations.append(dist.P2POp(dist.isend, value, peer, route.group))
                elif route.rank == transfer.owner_rank:
                    value = torch.empty_like(result[index][transfer.owner_slot])
                    incoming.append((index, transfer.owner_slot, value))
                    peer = _global_peer_rank(route.group, target)
                    operations.append(dist.P2POp(dist.irecv, value, peer, route.group))
        _exchange(operations)
        for index, slot, value in incoming:
            result[index][slot].add_(value)
    return result
