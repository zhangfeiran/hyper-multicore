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
"""Guest storage ownership, reuse and stream ordering tests."""

import gc
import unittest
import weakref
from unittest.mock import MagicMock, patch

import torch

from hyper_multicore.replica.pool import ReplicaPool, ReplicaPrefetch, replica_pool
from tests.common.mark_utils import arg_mark


class _Group:
    """Weak-referenceable process-group stand-in."""


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
class TestReplicaPool(unittest.TestCase):
    """Exercise the pool independently of accelerator and collective setup."""

    def test_complete_overwrite_borrows_stale_gradients_without_clearing(self) -> None:
        """Only an explicit complete-store caller may consume uncleared guest storage."""
        pool = ReplicaPool((torch.ones(6, 2, 3),), 6)
        pool.gradients = (torch.full((6, 2, 3), float("nan")),)
        original = pool.gradients[0]
        with pool.lease(backward=True, active_slots=3, _overwrite_gradients=True):
            self.assertIs(original, pool.gradients[0])
            self.assertTrue(torch.isnan(original).all())
            original[:3].fill_(9)
        with pool.lease(backward=True, active_slots=3):
            torch.testing.assert_close(original[:3], torch.zeros(3, 2, 3))
            self.assertTrue(torch.isnan(original[3:]).all())

    def test_backward_clears_only_the_active_guest_prefix(self) -> None:
        """Keep inactive storage untouched and clear newly activated slots on later leases."""
        pool = ReplicaPool((torch.ones(6, 2, 3),), 6)
        pool.gradients = (torch.full((6, 2, 3), 17.0),)
        with pool.lease(backward=True, active_slots=2):
            torch.testing.assert_close(pool.gradients[0][:2], torch.zeros(2, 2, 3))
            torch.testing.assert_close(pool.gradients[0][2:], torch.full((4, 2, 3), 17.0))
            pool.gradients[0][:2].fill_(9)
        with pool.lease(backward=True, active_slots=0):
            self.assertEqual(pool.gradients[0][0, 0, 0], 9)
        with pool.lease(backward=True, active_slots=4):
            torch.testing.assert_close(pool.gradients[0][:4], torch.zeros(4, 2, 3))
            torch.testing.assert_close(pool.gradients[0][4:], torch.full((2, 2, 3), 17.0))
        with pool.lease(backward=True):
            torch.testing.assert_close(pool.gradients[0], torch.zeros(6, 2, 3))

    def test_invalid_active_prefix_releases_the_lease(self) -> None:
        """Reject malformed active widths without permanently taking the pool lock."""
        pool = ReplicaPool((torch.ones(1, 2, 2),), 2)
        for count in (-1, 3, True, 1.5):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "Active guest slots"):
                with pool.lease(backward=True, active_slots=count):
                    self.fail("Invalid active width was accepted")
        with pool.lease():
            self.assertIsNone(pool.gradients)

    def test_projection_prefetch_waits_once_per_matrix(self):
        """Join each projection once, including a final whole-slot wait."""
        pool = ReplicaPool((torch.ones(1, 2, 2), torch.ones(1, 2, 1)), 1)
        waited = []
        prefetched = ReplicaPrefetch(pool, waited.append, projection_ready=((4096, 8192), 17))
        prefetched.wait_weights(0)
        prefetched.wait_weights(0)
        prefetched.wait_weights()
        self.assertEqual(waited, [0, 1])
        with self.assertRaisesRegex(ValueError, "out of range"):
            prefetched.wait_weights(2)

    def test_layer_reuse_and_group_lifetime(self):
        """Different layers reuse B storage without pinning parameters or groups."""
        group = _Group()
        owner = torch.ones(6, 8, 16)
        owner_ref = weakref.ref(owner)
        pool = replica_pool((owner,), 1, group)
        self.assertEqual(pool.weights[0].shape, (1, 8, 16))
        self.assertIs(pool, replica_pool((owner.clone(),), 1, group))
        self.assertIsNot(pool, replica_pool((owner,), 2, group))
        del owner
        self.assertIsNone(owner_ref())
        pool_ref = weakref.ref(pool)
        del pool, group
        gc.collect()
        self.assertIsNone(pool_ref())

    def test_lease_reuse_and_exception_cleanup(self):
        """Nested claims fail; failures release ownership; backward zeros guests."""
        pool = ReplicaPool((torch.ones(6, 8, 16),), 2)
        with self.assertRaisesRegex(ValueError, "consumer"):
            with pool.lease(backward=True):
                pool.gradients[0].fill_(12)
                with self.assertRaisesRegex(RuntimeError, "Concurrent"):
                    with pool.lease():
                        self.fail("nested claim unexpectedly succeeded")
                raise ValueError("consumer failed")
        previous = pool.gradients[0]
        with pool.lease(backward=True):
            self.assertIs(pool.gradients[0], previous)
            torch.testing.assert_close(previous, torch.zeros_like(previous))
            self.assertEqual(previous.dtype, torch.float32)

    def test_stream_waits_before_reuse(self):
        """A later stream must wait for the previous lease's final consumer."""
        device = torch.device("cuda", 0)
        tensor = MagicMock(device=device)
        backend = MagicMock()
        streams = [MagicMock(), MagicMock()]
        backend.current_stream.side_effect = streams
        pool = ReplicaPool((torch.ones(1, 1, 1),), 1)
        pool.weights = (tensor,)
        with patch.object(torch, "cuda", backend):
            with pool.lease():
                pass
            event = backend.Event.return_value
            with pool.lease():
                streams[1].wait_event.assert_called_once_with(event)
        self.assertEqual(tensor.record_stream.call_count, 2)
        self.assertEqual(event.record.call_count, 2)

    def test_stream_lookup_failure_releases_host_lock(self):
        """An unavailable device stream cannot permanently strand the pool lock."""
        pool = ReplicaPool((torch.ones(1, 1, 1),), 1)
        original = pool.weights
        pool.weights = (MagicMock(device=torch.device("cuda", 0)),)
        with patch.object(torch.cuda, "current_stream", side_effect=RuntimeError("device unavailable")):
            with self.assertRaisesRegex(RuntimeError, "device unavailable"):
                with pool.lease():
                    self.fail("stream lookup unexpectedly succeeded")
        pool.weights = original
        with pool.lease():
            self.assertIs(pool.weights, original)
