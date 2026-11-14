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
"""Owner-authorized reuse of unchanged remote guest payloads."""

import gc
import random
import unittest
import weakref
from types import SimpleNamespace

import torch

from hyper_multicore.replica.residency import _ReplicaWeightCache, mark_replica_weights_materialized


def _materialized(weights):
    """Provide explicit materialization evidence for these test-owned values."""
    for value in weights:
        if isinstance(value, torch.Tensor):
            mark_replica_weights_materialized(value)
        else:
            _materialized(value)
    return weights


def _route(*assignments: tuple[int, int, int, int], rank: int = 0) -> SimpleNamespace:
    """Build immutable transfer fields without distributed initialization."""
    transfers = tuple(
        (
            SimpleNamespace(owner_rank=owner, owner_slot=expert, target_rank=target, target_slot=slot)
            for owner, expert, target, slot in assignments
        )
    )
    return SimpleNamespace(rank=rank, plan=SimpleNamespace(transfers=transfers, config=SimpleNamespace(home_experts=2)))


class TestReplicaWeightCache(unittest.TestCase):
    """Track value identity independently of microbatch routes and gradient epochs."""

    def test_external_weights_without_authority_always_copy(self) -> None:
        """Pointer and Torch version alone cannot prove an external materialization."""
        weights = (torch.ones(2, 3, 4), torch.ones(2, 4, 2))
        cache, route = _ReplicaWeightCache(), _route((0, 1, 2, 3))
        first = cache.prepare(weights, route)
        cache.commit(first)
        self.assertFalse(cache.prepare(weights, route).reused)

    def test_materialization_marker_invalidates_preserved_torch_version(self) -> None:
        """A framework must refresh evidence after data replacement or an all-gather."""
        weights = _materialized((torch.ones(2, 3, 4), torch.ones(2, 4, 2)))
        cache, route = _ReplicaWeightCache(), _route((0, 1, 2, 3))
        cache.commit(cache.prepare(weights, route))
        hit = cache.prepare(weights, route)
        self.assertTrue(hit.reused)
        cache.commit(hit)
        previous = weights[0]._version
        weights[0].data.zero_()
        self.assertEqual(weights[0]._version, previous)
        mark_replica_weights_materialized(*weights)
        self.assertFalse(cache.prepare(weights, route).reused)

    def test_microbatch_and_row_changes_preserve_weight_hits(self) -> None:
        """Token work and invocation metadata do not change an unchanged expert matrix."""
        weights = _materialized((torch.ones(2, 3, 4), torch.ones(2, 4, 2)))
        cache = _ReplicaWeightCache()
        route = _route((0, 1, 2, 3))
        initial = cache.prepare(weights, route)
        self.assertEqual(initial.reused, frozenset())
        cache.commit(initial)
        route.microbatch = 3
        route.plan.logical_counts = ((17, 2),)
        following = cache.prepare(weights, route)
        self.assertEqual(following.reused, frozenset({(2, 3)}))
        cache.commit(following)

    def test_any_parameter_update_invalidates_the_combined_payload(self) -> None:
        """An optimizer-like update to either projection requires both fresh matrices."""
        for changed in range(2):
            with self.subTest(changed=changed):
                weights = _materialized((torch.ones(2, 3, 4), torch.ones(2, 4, 2)))
                cache = _ReplicaWeightCache()
                route = _route((0, 0, 1, 2))
                cache.commit(cache.prepare(weights, route))
                weights[changed].add_(1)
                self.assertEqual(cache.prepare(weights, route).reused, frozenset())

    def test_remote_owner_overwrite_invalidates_only_the_written_slot(self) -> None:
        """All ranks observe the global writes even when they are not the new owner."""
        weights = _materialized((torch.ones(2, 3, 4),))
        cache = _ReplicaWeightCache()
        original = _route((0, 0, 1, 2), (0, 1, 1, 3))
        cache.commit(cache.prepare(weights, original))
        overwrite = _route((2, 0, 1, 2))
        cache.commit(cache.prepare(weights, overwrite))
        restored = cache.prepare(weights, original)
        self.assertEqual(restored.reused, frozenset({(1, 3)}))

    def test_owner_expert_and_destination_are_part_of_identity(self) -> None:
        """A shared parameter tensor does not make its different expert slices equivalent."""
        weights = _materialized((torch.ones(2, 3, 4),))
        for changed in (_route((0, 1, 1, 2)), _route((0, 0, 2, 2)), _route((0, 0, 1, 3))):
            with self.subTest(transfers=changed.plan.transfers):
                cache = _ReplicaWeightCache()
                cache.commit(cache.prepare(weights, _route((0, 0, 1, 2))))
                self.assertEqual(cache.prepare(weights, changed).reused, frozenset())

    def test_different_home_width_invalidates_the_same_physical_guest_slot(self) -> None:
        """Physical slot identity must survive changes to the logical expert prefix."""
        weights = _materialized((torch.ones(2, 3, 4),))
        cache = _ReplicaWeightCache()
        original = _route((0, 0, 1, 2))
        cache.commit(cache.prepare(weights, original))
        overwrite = _route((2, 0, 1, 4))
        overwrite.plan.config.home_experts = 4
        cache.commit(cache.prepare((torch.ones(4, 3, 4),), overwrite))
        self.assertEqual(cache.prepare(weights, original).reused, frozenset())

    def test_storage_replacement_invalidates_without_a_version_increment(self) -> None:
        """FSDP-style rebinding cannot reuse a payload based only on its numeric version."""
        weights = _materialized((torch.ones(2, 3, 4),))
        route = _route((0, 0, 1, 2))
        cache = _ReplicaWeightCache()
        cache.commit(cache.prepare(weights, route))
        version = weights[0]._version
        weights[0].data = weights[0].clone()
        self.assertEqual(weights[0]._version, version)
        self.assertEqual(cache.prepare(weights, route).reused, frozenset())

    def test_cache_does_not_keep_parameter_graph_or_storage_alive(self) -> None:
        """Residency metadata must not pin layer weights or their autograd graphs."""
        weights = _materialized((torch.ones(2, 3, 4, requires_grad=True) * 2,))
        tensor_ref = weakref.ref(weights[0])
        storage_ref = weakref.ref(weights[0].untyped_storage())
        cache = _ReplicaWeightCache()
        cache.commit(cache.prepare(weights, _route((0, 0, 1, 2))))
        del weights
        gc.collect()
        self.assertIsNone(tensor_ref())
        self.assertIsNone(storage_ref())
        self.assertEqual(cache.prepare((torch.ones(2, 3, 4),), _route((0, 0, 1, 2))).reused, frozenset())

    def test_uncommitted_or_cleared_writes_cannot_hit(self) -> None:
        """Failed copies and a new protocol generation must take the normal copy path."""
        weights = _materialized((torch.ones(2, 3, 4),))
        route = _route((0, 0, 1, 2))
        cache = _ReplicaWeightCache()
        cache.prepare(weights, route)
        self.assertEqual(cache.prepare(weights, route).reused, frozenset())
        cache.commit(cache.prepare(weights, route))
        cache.clear()
        self.assertEqual(cache.prepare(weights, route).reused, frozenset())

    def test_inference_tensors_without_versions_always_copy(self) -> None:
        """Unversioned values cannot provide a safe parameter-value identity."""
        with torch.inference_mode():
            weights = _materialized((torch.ones(2, 3, 4),))
        cache = _ReplicaWeightCache()
        route = _route((0, 0, 1, 2))
        cache.commit(cache.prepare(weights, route))
        pending = cache.prepare(weights, route)
        self.assertEqual(pending.reused, frozenset())
        self.assertEqual(pending.required_owner_slots, frozenset({0}))

    def test_history_is_bounded_by_live_destination_slots(self) -> None:
        """Different layers replace metadata instead of growing a per-layer matrix cache."""
        cache = _ReplicaWeightCache()
        for _ in range(100):
            weights = _materialized((torch.ones(2, 3, 4),))
            cache.commit(cache.prepare(weights, _route((0, 0, 1, 2), (0, 1, 2, 2))))
        self.assertEqual(len(cache._entries), 2)

    def test_global_slot_history_matches_remote_values_under_owner_changes(self) -> None:
        """One owner's update or another owner's overwrite cannot cause a stale hit."""
        rng = random.Random(1731)
        weights = _materialized([[(torch.randn(2, 3, 4),) for _ in range(2)] for _ in range(3)])
        caches = [_ReplicaWeightCache() for _ in range(3)]
        remote = {}
        for step in range(80):
            layer = rng.randrange(2)
            if step % 7 == 0:
                weights[rng.randrange(3)][layer][0].add_(1)
            assignments = []
            for target in range(3):
                for slot in range(2, 2 + rng.randrange(3)):
                    owner = rng.choice([rank for rank in range(3) if rank != target])
                    assignments.append((owner, rng.randrange(2), target, slot))
            pending = [
                cache.prepare(weights[rank][layer], _route(*assignments, rank=rank))
                for rank, cache in enumerate(caches)
            ]
            for owner, expert, target, slot in assignments:
                expected = weights[owner][layer][0][expert]
                if (target, slot) in pending[owner].reused:
                    torch.testing.assert_close(remote[target, slot], expected, rtol=0, atol=0)
                else:
                    remote[target, slot] = expected.clone()
            for cache, copies in zip(caches, pending):
                cache.commit(copies)
                self.assertLessEqual(len(cache._entries), 6)
