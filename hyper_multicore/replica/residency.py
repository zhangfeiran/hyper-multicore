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
"""Owner-local evidence for unchanged payloads in shared remote guest slots."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any
from weakref import ReferenceType, ref

import torch

from .routing import ReplicaRoute


@dataclass(frozen=True, eq=False)
class _WeightSnapshot:
    """Weak identities and version metadata without retaining parameter graphs."""

    tensors: tuple[ReferenceType, ...]
    storages: tuple[ReferenceType, ...]
    metadata: tuple[tuple[Any, ...], ...]

    @classmethod
    def capture(cls, weights: tuple[torch.Tensor, ...]) -> _WeightSnapshot | None:
        """Decline values without version counters; rebinding must also miss."""
        if any(torch.is_inference(weight) for weight in weights):
            return None
        storages = tuple(weight.untyped_storage() for weight in weights)
        metadata = tuple((weight._version, weight.data_ptr(), weight.storage_offset(),
                          tuple(weight.shape), weight.stride(), weight.dtype, weight.device,
                          getattr(weight, "_hp_materialization_token", None))
                         for weight in weights)
        return cls(tuple(ref(weight) for weight in weights), tuple(ref(storage) for storage in storages), metadata)

    def matches(self, other: _WeightSnapshot) -> bool:
        """Require live object identities, layouts and unchanged value versions."""
        if self.metadata != other.metadata:
            return False
        return all(left() is not None and left() is right()
                   for left, right in zip(self.tensors + self.storages, other.tensors + other.storages))


@dataclass(frozen=True)
class _WeightCopies:
    """Stage slot updates until the existing copy/readiness protocol completes."""

    reused: frozenset[tuple[int, int]]
    updates: tuple[tuple[tuple[int, int], int, _WeightSnapshot], ...]
    required_owner_slots: frozenset[int]


class _ReplicaWeightCache:
    """Remember only this owner's last writes, bounded by destination slots.

    Every rank observes the same global transfer sequence. Writes by another
    owner invalidate our evidence, while inactive slots retain their payload.
    Only the owner skips a matrix put; receivers still await fresh readiness.
    """

    def __init__(self) -> None:
        """Start with no evidence for the provider's symmetric heap generation."""
        self._entries: dict[tuple[int, int], tuple[int, _WeightSnapshot]] = {}

    def clear(self) -> None:
        """Invalidate all evidence after rollover, failed execution or heap replacement."""
        self._entries.clear()

    def prepare(self, weights: tuple[torch.Tensor, ...], route: ReplicaRoute) -> _WeightCopies:
        """Check current owner values and invalidate every slot being reassigned."""
        enabled = os.environ.get('HYPER_PARALLEL_REPLICA_WEIGHT_RESIDENCY', '1')
        if enabled not in ('0', '1'):
            raise ValueError('HYPER_PARALLEL_REPLICA_WEIGHT_RESIDENCY must be 0 or 1')
        if enabled == '0':
            self.clear()
        snapshot = _WeightSnapshot.capture(weights)
        reused, updates, required = set(), [], set()
        home = route.plan.config.home_experts
        for transfer in route.plan.transfers:
            key = (transfer.target_rank, transfer.target_slot - home)
            previous = self._entries.pop(key, None)
            if transfer.owner_rank != route.rank:
                continue
            if (snapshot is not None and previous is not None
                    and previous[0] == transfer.owner_slot and previous[1].matches(snapshot)):
                reused.add((transfer.target_rank, transfer.target_slot))
            else:
                required.add(transfer.owner_slot)
            if snapshot is not None:
                updates.append((key, transfer.owner_slot, snapshot))
        return _WeightCopies(frozenset(reused), tuple(updates), frozenset(required))

    def commit(self, copies: _WeightCopies) -> None:
        """Publish host evidence only after the unchanged transfer protocol succeeds."""
        self._entries.update((key, (slot, snapshot)) for key, slot, snapshot in copies.updates)
