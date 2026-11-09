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
"""Signal storage, epoch rollover and lease boundary contracts."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from hyper_multicore.replica import build_expert_replica_plan
from hyper_multicore.replica.residency import mark_replica_weights_materialized
from hyper_multicore.replica.signal_transport import SignalReplicaTransport, gradient_band_layout, signal_storage_bytes
from tests.common.mark_utils import arg_mark


def _materialized(weights):
    """Provide explicit materialization evidence for these test-owned values."""
    mark_replica_weights_materialized(*weights)
    return weights


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
class TestSignalReplicaTransport(unittest.TestCase):
    """Keep host boundary checks alongside real-stream NPU protocol acceptance."""

    def test_resident_payload_hit_still_publishes_fresh_projection_ready(self) -> None:
        """Owners skip matrices across microbatches without reusing a readiness epoch."""
        weights = _materialized((torch.ones(1, 2, 2, dtype=torch.bfloat16), torch.ones(1, 2, 1, dtype=torch.bfloat16)))
        runtime = MagicMock()
        storage = torch.empty(signal_storage_bytes(((2, 2), (2, 1)), 1, 2, 2), dtype=torch.uint8)
        provider = SignalReplicaTransport(runtime, storage, 1, 2)
        transfer = SimpleNamespace(owner_rank=0, owner_slot=0, target_rank=1, target_slot=1)
        route = SimpleNamespace(
            rank=0,
            plan=SimpleNamespace(
                config=SimpleNamespace(replica_slots_per_rank=1, ep_size=2, home_experts=1), transfers=(transfer,)
            ),
        )
        with patch.object(torch, "cpu", MagicMock()), patch.object(torch.Tensor, "record_stream"):
            with provider.lease(weights, route, overlap=True) as first:
                initial_epoch = first.projection_ready[1]
            self.assertEqual(provider._last_weight_copy_counts, (1, 0))
            runtime.put.reset_mock()
            with provider.lease(weights, route, overlap=True) as second:
                self.assertGreater(second.projection_ready[1], initial_epoch)
                self.assertIsNot(second.weight_generation, first.weight_generation)
            self.assertEqual(provider._last_weight_copy_counts, (0, 1))
            self.assertEqual(runtime.put.call_count, len(weights))
            self.assertTrue(all((call.args[1].dtype == torch.int32 for call in runtime.put.call_args_list)))
            weights[0].add_(1)
            runtime.put.reset_mock()
            with provider.lease(weights, route, overlap=True):
                pass
            self.assertEqual(provider._last_weight_copy_counts, (1, 0))
            self.assertEqual(runtime.put.call_count, 2 * len(weights))

    def test_failed_consumer_invalidates_payload_evidence(self) -> None:
        """A caller error cannot leave a reusable generation after protocol cleanup."""
        weights = _materialized((torch.ones(1, 2, 2, dtype=torch.bfloat16),))
        runtime = MagicMock()
        storage = torch.empty(signal_storage_bytes(((2, 2),), 1, 2, 2), dtype=torch.uint8)
        provider = SignalReplicaTransport(runtime, storage, 1, 2)
        transfer = SimpleNamespace(owner_rank=0, owner_slot=0, target_rank=1, target_slot=1)
        route = SimpleNamespace(
            rank=0,
            plan=SimpleNamespace(
                config=SimpleNamespace(replica_slots_per_rank=1, ep_size=2, home_experts=1), transfers=(transfer,)
            ),
        )
        with patch.object(torch, "cpu", MagicMock()), patch.object(torch.Tensor, "record_stream"):
            with self.assertRaisesRegex(RuntimeError, "consumer failed"):
                with provider.lease(weights, route, overlap=True):
                    raise RuntimeError("consumer failed")
            self.assertIsNone(provider._resident_generation)
            with provider.lease(weights, route, overlap=True):
                pass
            self.assertEqual(provider._last_weight_copy_counts, (1, 0))

    def test_complete_store_lease_preserves_gradient_storage_until_kernel_overwrite(self) -> None:
        """The fused caller opts out of clearing without changing the saved weight generation."""
        weights = _materialized((torch.ones(1, 2, 2, dtype=torch.bfloat16),))
        storage = torch.empty(signal_storage_bytes(((2, 2),), 1, 2, 2), dtype=torch.uint8)
        provider = SignalReplicaTransport(MagicMock(), storage, 1, 2)
        provider._bind(weights)
        provider.pool.gradients[0].fill_(float("nan"))
        generation = object()
        provider._resident_generation = generation
        provider._resident_ready = ((4096,), 1)
        route = SimpleNamespace(
            rank=0,
            plan=SimpleNamespace(
                config=SimpleNamespace(replica_slots_per_rank=1, ep_size=2, home_experts=1),
                transfers=(SimpleNamespace(target_rank=0, target_slot=1),),
            ),
        )
        with provider.lease(
            weights, route, backward=True, overlap=True, reuse_generation=generation, _overwrite_gradients=True
        ) as view:
            self.assertTrue(torch.isnan(view.gradients[0]).all())
            self.assertIs(view.weight_generation, generation)
        with provider.lease(weights, route, backward=True, overlap=True, reuse_generation=generation) as view:
            torch.testing.assert_close(view.gradients[0], torch.zeros(1, 2, 2))

    def test_reused_generation_clears_sparse_guest_prefix_only(self) -> None:
        """Keep inactive gradient storage untouched even when the weight exchange is skipped."""
        weights = _materialized((torch.ones(1, 2, 2, dtype=torch.bfloat16),))
        size = signal_storage_bytes(((2, 2),), 6, 2, 2)
        provider = SignalReplicaTransport(MagicMock(), torch.empty(size, dtype=torch.uint8), 6, 2)
        provider._bind(weights)
        provider.pool.gradients[0].fill_(17)
        generation = object()
        provider._resident_generation = generation
        provider._resident_ready = ((4096,), 1)
        route = SimpleNamespace(
            rank=0,
            plan=SimpleNamespace(
                config=SimpleNamespace(replica_slots_per_rank=6, ep_size=2, home_experts=1),
                transfers=(
                    SimpleNamespace(target_rank=0, target_slot=1),
                    SimpleNamespace(target_rank=0, target_slot=3),
                ),
            ),
        )
        with provider.lease(weights, route, backward=True, overlap=True, reuse_generation=generation) as view:
            self.assertIs(view.weight_generation, generation)
            torch.testing.assert_close(view.gradients[0][:3], torch.zeros(3, 2, 2))
            torch.testing.assert_close(view.gradients[0][3:], torch.full((3, 2, 2), 17.0))
        route.plan.transfers = ()
        provider.pool.gradients[0].fill_(11)
        with provider.lease(weights, route, backward=True, overlap=True, reuse_generation=generation) as view:
            torch.testing.assert_close(view.gradients[0], torch.full((6, 2, 2), 11.0))

    def test_compact_matrix_storage_keeps_the_signal_protocol_stride(self) -> None:
        """Lease only resident matrices while retaining the full native signal namespace."""
        shapes = ((512, 512), (256, 512))
        weights = _materialized(tuple((torch.ones((1, *shape), dtype=torch.bfloat16) for shape in shapes)))
        runtime = MagicMock()
        full = signal_storage_bytes(shapes, 6, 2, 2, gradient_bands=True)
        size = signal_storage_bytes(shapes, 6, 2, 2, gradient_bands=True, storage_slots=2)
        provider = SignalReplicaTransport(
            runtime, torch.empty(size, dtype=torch.uint8), 6, 2, gradient_bands=True, storage_slots=2
        )
        config = SimpleNamespace(replica_slots_per_rank=6, ep_size=2, home_experts=1)
        missing = SimpleNamespace(
            rank=0, plan=SimpleNamespace(config=config, transfers=(SimpleNamespace(target_rank=1, target_slot=3),))
        )
        with self.assertRaisesRegex(ValueError, "unallocated matrix"):
            with provider.lease(weights, missing):
                self.fail("A route may not read a slot outside the current allocation")
        runtime.host_barrier.assert_not_called()
        provider._bind(weights)
        self.assertEqual(full - size, 4 * (512 * 512 + 256 * 512) * 6)
        self.assertEqual(tuple(provider.signals.shape), (5, 2, 6, 16))
        self.assertEqual(tuple(provider.projection_signals.shape), (2, 6, 16))
        self.assertEqual([tuple(w.shape) for w in provider.pool.weights], [(2, *shape) for shape in shapes])
        self.assertEqual([tuple(g.shape) for g in provider.pool.gradients], [(2, *shape) for shape in shapes])
        for matrix, shape in enumerate(shapes):
            band_bytes = provider._gradient_band_storage[matrix].numel()
            self.assertEqual(band_bytes, 6 * gradient_band_layout(shape)[1] * 32 * 64)

    def test_band_storage_is_bounded_and_preserves_existing_transport_options(self):
        """Reserve all guest signal tables before execution without changing legacy modes."""
        shapes = ((5120, 3584), (1792, 5120))
        self.assertEqual(gradient_band_layout(shapes[0]), (292, 18))
        self.assertEqual(gradient_band_layout(shapes[1]), (204, 9))
        original = signal_storage_bytes(shapes, 2, 16, 2)
        extended = signal_storage_bytes(shapes, 2, 16, 2, gradient_bands=True)
        self.assertEqual(extended - original, 2 * (18 + 9) * 32 * 64)
        for shape in ((0, 3), (1, True), (2, 3, 4), (65, 1048576)):
            self.assertIsNone(gradient_band_layout(shape))

    def test_band_tables_are_inbox_views_and_reset_at_reserved_rollover(self):
        """Compact views survive calls and reset with all protocol words before epoch reuse."""
        shapes = ((16, 32), (32, 16))
        size = signal_storage_bytes(shapes, 2, 4, 2, gradient_bands=True)
        storage = torch.empty(size, dtype=torch.uint8)
        runtime = MagicMock()
        provider = SignalReplicaTransport(runtime, storage, 2, 4, gradient_bands=True)
        provider._bind(tuple((torch.ones(2, *shape, dtype=torch.bfloat16) for shape in shapes)))
        with provider._lock:
            pointers = []
            for matrix in (0, 1):
                flags, rows, bands = provider.kernel_gradient_band_signals(matrix, 24)
                self.assertEqual((rows, bands), (shapes[matrix][0], 1))
                self.assertEqual(tuple(flags.shape), (2, 1, 24, 16))
                self.assertEqual(flags.untyped_storage().data_ptr(), storage.data_ptr())
                self.assertIs(provider.kernel_gradient_band_signals(matrix, 24)[0], flags)
                flags.fill_(17)
                pointers.append(flags.data_ptr())
            self.assertNotEqual(pointers[0], pointers[1])
            self.assertIsNone(provider.kernel_gradient_band_signals(0, 33))
            provider.epoch = 2**31 - 2
            self.assertEqual(provider._advance(reserve=2), 1)
            self.assertEqual(runtime.host_barrier.call_count, 3)
            for matrix in (0, 1):
                self.assertEqual(int(provider.kernel_gradient_band_signals(matrix, 24)[0].count_nonzero()), 0)
            self.assertEqual(provider.kernel_gradient_signals()[0], 2)
            self.assertEqual(provider.kernel_gradient_signals()[0], 3)
        with self.assertRaisesRegex(ValueError, "active backward lease"):
            provider.kernel_gradient_band_signals(0, 24)

    def test_saved_generation_reuse_and_backward_epoch_reservation(self):
        """Reuse a live forward generation and roll over before publishing backward weights."""
        weights = _materialized((torch.ones(1, 2, 2, dtype=torch.bfloat16), torch.ones(1, 2, 1, dtype=torch.bfloat16)))
        runtime = MagicMock()
        storage = torch.empty(signal_storage_bytes(((2, 2), (2, 1)), 1, 2, 2), dtype=torch.uint8)
        provider = SignalReplicaTransport(runtime, storage, 1, 2)
        route = SimpleNamespace(plan=build_expert_replica_plan([[1, 0], [0, 1]], 1), rank=0)
        with patch.object(torch, "cpu", MagicMock()):
            with provider.lease(weights, route, overlap=True) as view:
                generation = view.weight_generation
            with provider.lease(weights, route, backward=True, overlap=True, reuse_generation=generation) as view:
                self.assertIs(view.weight_generation, generation)
            provider.epoch = 2**31 - 3
            with provider.lease(weights, route, backward=True, overlap=True, reuse_generation=generation) as view:
                self.assertIsNot(view.weight_generation, generation)
                self.assertEqual(view.projection_ready[1], 1)
                barriers = runtime.host_barrier.call_count
                self.assertEqual(provider.kernel_gradient_signals()[0], 2)
                self.assertEqual(provider.kernel_gradient_signals()[0], 3)
                self.assertEqual(runtime.host_barrier.call_count, barriers)
            with self.assertRaisesRegex(ValueError, "saved backward context"):
                with provider.lease(weights, route, reuse_generation=generation):
                    self.fail("forward accepted a backward-only reuse token")

    def test_projection_storage_and_rollover_keep_signals_disjoint(self):
        """Zero both publications on rollover without aliasing credit/ACK cache lines."""
        shapes = ((2, 2), (2, 1))
        size = signal_storage_bytes(shapes, 1, 2, 2)
        runtime = MagicMock()
        provider = SignalReplicaTransport(runtime, torch.empty(size, dtype=torch.uint8), 1, 2)
        weights = _materialized((torch.ones(1, 2, 2, dtype=torch.bfloat16), torch.ones(1, 2, 1, dtype=torch.bfloat16)))
        provider._bind(weights)
        self.assertGreaterEqual(
            provider.projection_signals.data_ptr(), provider.signals.data_ptr() + provider.signals.numel() * 4
        )
        provider.signals.fill_(17)
        provider.projection_signals.fill_(17)
        provider.epoch = 2**31 - 1
        self.assertEqual(provider._advance(), 1)
        self.assertEqual(int(provider.signals.count_nonzero()), 0)
        self.assertEqual(int(provider.projection_signals.count_nonzero()), 0)
        self.assertEqual(runtime.host_barrier.call_count, 3)

    def test_invalid_layout_fails_before_collectives(self):
        """Reject malformed public sizing arguments and insufficient storage early."""
        for shapes, slots, size, element in (
            (((),), 1, 2, 2),
            (((2, 2),), True, 2, 2),
            (((2, 2),), 1, 1.5, 2),
            (((2, 2),), 1, 2, 1),
            (((2, 0),), 1, 2, 2),
        ):
            with self.assertRaises(ValueError):
                signal_storage_bytes(shapes, slots, size, element)
        runtime = MagicMock()
        provider = SignalReplicaTransport(runtime, torch.empty(64, dtype=torch.uint8), 1, 2)
        route = SimpleNamespace(plan=build_expert_replica_plan([[1, 0], [0, 1]], 1), rank=0)
        with self.assertRaisesRegex(ValueError, "too small"):
            with provider.lease((torch.ones(1, 2, 2),), route):
                self.fail("undersized storage unexpectedly accepted")
        self.assertFalse(runtime.host_barrier.called)

    def test_rollover_resets_only_after_quiescence(self):
        """A bounded integer epoch resets collectively instead of wrapping stale signals."""
        runtime = MagicMock()
        size = signal_storage_bytes(((2, 2),), 1, 2, 2)
        provider = SignalReplicaTransport(runtime, torch.empty(size, dtype=torch.uint8), 1, 2)
        provider.epoch = 2**31 - 1
        route = SimpleNamespace(plan=build_expert_replica_plan([[1, 0], [0, 1]], 1), rank=0)
        with patch.object(torch, "cpu", MagicMock()), provider.lease(
            (torch.ones(1, 2, 2, dtype=torch.bfloat16),), route
        ):
            self.assertEqual(provider.epoch, 1)
            self.assertEqual(runtime.host_barrier.call_count, 3)
            self.assertEqual(int(provider.signals.count_nonzero()), 0)
            with self.assertRaisesRegex(RuntimeError, "Concurrent"):
                with provider.lease((torch.ones(1, 2, 2, dtype=torch.bfloat16),), route):
                    self.fail("nested lease unexpectedly succeeded")
