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
"""CPU checks for immutable metadata reuse after the original exact plan guard."""

from __future__ import annotations

from types import SimpleNamespace
import os
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_multicore.replica.recompute import create_recompute_contexts, get_recompute_state
from hyper_multicore.replica import _metadata, routing
from hyper_multicore.replica.capacity import ExpertReplicaConfig
from hyper_multicore.replica.planner import build_expert_replica_plan


class TestReplicaMetadataSetting(unittest.TestCase):
    """Reject malformed switches instead of silently measuring the disabled path."""

    def tearDown(self) -> None:
        """Restore the cached setting after every isolated check."""
        _metadata._metadata_reuse_enabled.cache_clear()

    def test_default_is_disabled(self) -> None:
        """An absent setting preserves the guarded default."""
        with patch.dict(os.environ):
            os.environ.pop("HYPER_MULTICORE_REPLICA_METADATA_REUSE", None)
            _metadata._metadata_reuse_enabled.cache_clear()
            self.assertFalse(_metadata._metadata_reuse_enabled())

    def test_accepts_binary_switch(self) -> None:
        """Only the two explicit binary settings select a mode."""
        for value, expected in (("0", False), ("1", True)):
            with self.subTest(value=value), patch.dict(os.environ, {"HYPER_MULTICORE_REPLICA_METADATA_REUSE": value}):
                _metadata._metadata_reuse_enabled.cache_clear()
                self.assertIs(_metadata._metadata_reuse_enabled(), expected)

    def test_rejects_malformed_switch(self) -> None:
        """Invalid benchmark controls must fail before a false comparison."""
        for value in ("1-state", "", "true", "2"):
            with self.subTest(value=value), patch.dict(os.environ, {"HYPER_MULTICORE_REPLICA_METADATA_REUSE": value}):
                _metadata._metadata_reuse_enabled.cache_clear()
                with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                    _metadata._metadata_reuse_enabled()


class TestReplicaMetadataReplay(unittest.TestCase):
    """Retain metadata without weakening count/policy checks or caching token positions."""

    def setUp(self) -> None:
        """Enable only the experimental CPU path for each isolated case."""
        setting = patch.object(_metadata, "_metadata_reuse_enabled", return_value=True)
        setting.start()
        self.addCleanup(setting.stop)
        self.config = ExpertReplicaConfig(4, 2, 1)
        self.group = object()
        self.counts = ((2, 1, 1, 0), (2, 1, 0, 1))
        self.options = {"target_load": 4, "capacity_limit": 8}

    def _plan(self, counts: tuple | None = None, **changes: object) -> tuple:
        """Use the real invocation-local CPU guard before touching metadata."""
        return routing._plan_replica_route(
            self.counts if counts is None else counts, self.config, self.group, 0, **(self.options | changes)
        )

    def test_home_hint_is_replay_local_and_does_not_consume_records(self) -> None:
        """A home hint predicts local work without replacing the fresh plan guard."""
        self.assertFalse(routing._has_home_recompute_hint())
        forward, replay = create_recompute_contexts()
        with forward:
            routing._record_home_decision()
            self.assertFalse(routing._has_home_recompute_hint())
        with replay:
            self.assertTrue(routing._has_home_recompute_hint())
            self.assertTrue(routing._has_home_recompute_hint())
            routing._record_home_decision()
            self.assertFalse(routing._has_home_recompute_hint())
        self.assertFalse(routing._has_home_recompute_hint())

    def test_hot_plan_is_not_a_home_hint(self) -> None:
        """An existing expert plan cannot trigger speculative home permutation."""
        forward, replay = create_recompute_contexts()
        with forward:
            self._plan()
        with replay:
            self.assertFalse(routing._has_home_recompute_hint())

    def test_replay_reuses_storage_and_remaps_current_ids(self) -> None:
        """Feature: Exact metadata hit.
        Description: Replay the same counts with a different per-token ordering.
        Expectation: Upload occurs once while remap uses the current logical IDs.
        """
        forward, replay = create_recompute_contexts()
        with patch.object(routing, "_async_device_bytes", wraps=routing._async_device_bytes) as upload:
            with forward:
                plan, reused = self._plan()
                self.assertFalse(reused)
                original = routing._upload_route_metadata(plan, 0, torch.device("cpu"))
                cache = get_recompute_state().get_resource(_metadata._RecomputedRouteMetadata, lambda: None)
            with replay:
                same, reused = self._plan()
                self.assertTrue(reused)
                self.assertIs(same, plan)
                current = routing._upload_route_metadata(same, 0, torch.device("cpu"))
                self.assertTrue(all(a is b for a, b in zip(original, current)))
                before = routing._remap_replica_ids(torch.tensor([[0, 1], [0, 2]]), plan, *current[:2])
                after = routing._remap_replica_ids(torch.tensor([[2, 0], [1, 0]]), plan, *current[:2])
                self.assertFalse(torch.equal(before, after))
                self.assertEqual(upload.call_count, 1)
            self.assertFalse(cache.entries)
        self.assertTrue(all(not value.requires_grad and value.grad_fn is None for value in original))
        self.assertEqual(tuple(map(tuple, original[2].tolist())), plan.dispatch_counts)

    def test_changed_counts_and_policy_repack_after_cpu_miss(self) -> None:
        """Feature: Exact CPU guard preservation.
        Description: Change the histogram or planning policy during replay.
        Expectation: A new plan cannot use the original metadata entry.
        """
        for change in ("counts", "policy"):
            with self.subTest(change=change):
                forward, replay = create_recompute_contexts()
                with patch.object(routing, "_async_device_bytes", wraps=routing._async_device_bytes) as upload:
                    with forward:
                        original, _ = self._plan()
                        first = routing._upload_route_metadata(original, 0, torch.device("cpu"))
                    with replay:
                        if change == "counts":
                            plan, hit = self._plan(((1, 2, 1, 0), (2, 1, 0, 1)))
                        else:
                            plan, hit = self._plan(target_load=3)
                        self.assertFalse(hit)
                        result = routing._upload_route_metadata(plan, 0, torch.device("cpu"))
                        self.assertIsNot(result[2], first[2])
                    self.assertEqual(upload.call_count, 2)

    def test_equal_value_new_plan_identity_still_misses(self) -> None:
        """Feature: Producer identity.
        Description: Create an equal-valued plan without an original CPU replay hit.
        Expectation: Metadata is rebuilt rather than selected by a weak value hash.
        """
        forward, replay = create_recompute_contexts()
        with patch.object(routing, "_async_device_bytes", wraps=routing._async_device_bytes) as upload:
            with forward:
                original, _ = self._plan()
                routing._upload_route_metadata(original, 0, torch.device("cpu"))
            with replay:
                other = build_expert_replica_plan(self.counts, 1, **self.options)
                self.assertEqual(other, original)
                routing._upload_route_metadata(other, 0, torch.device("cpu"))
            self.assertEqual(upload.call_count, 2)

    def test_compact_home_owns_counts_and_never_uploads_guest_dispatch(self) -> None:
        """Feature: Home-only metadata.
        Description: Retain compact counts across producer buffer mutation and replay.
        Expectation: Owned counts are reused without dense guest metadata upload.
        """
        counts = ((1, 1, 0, 0), (0, 0, 1, 1))
        producer = torch.tensor(counts, dtype=torch.int32)
        forward, replay = create_recompute_contexts()
        with patch.object(routing, "_async_device_bytes", side_effect=AssertionError("Home has no dense upload")):
            with forward:
                plan, _ = self._plan(counts)
                self.assertFalse(plan.transfers)
                first = routing._home_route_counts(plan, 0, producer)
                self.assertNotEqual(first.data_ptr(), producer.data_ptr())
            producer.fill_(99)
            with replay:
                plan, hit = self._plan(counts)
                self.assertTrue(hit)
                result = routing._home_route_counts(plan, 0, torch.tensor(counts, dtype=torch.int32))
                self.assertIs(result, first)
                torch.testing.assert_close(result, torch.tensor(counts, dtype=torch.int32))

    def test_b0_disabled_and_noncheckpoint_do_not_create_entries(self) -> None:
        """Feature: Baseline preservation.
        Description: Inspect B0, disabled and non-checkpoint cache eligibility.
        Expectation: None creates an invocation metadata registry.
        """
        plan = build_expert_replica_plan(self.counts, 0)
        with patch.object(_metadata, "get_recompute_state", side_effect=AssertionError("B0 must bypass")):
            self.assertEqual(_metadata._recompute_metadata(plan), (None, False))
        positive = build_expert_replica_plan(self.counts, 1)
        self.assertEqual(_metadata._recompute_metadata(positive), (None, False))
        with patch.object(_metadata, "_metadata_reuse_enabled", return_value=False), patch.object(
            _metadata, "get_recompute_state", side_effect=AssertionError("Disabled must bypass")
        ):
            self.assertEqual(_metadata._recompute_metadata(positive), (None, False))

    def test_saved_tensors_outlive_early_stop_and_later_replay_rebuilds(self) -> None:
        """Feature: Invocation cleanup.
        Description: Clear an interrupted replay and then replay the retained graph again.
        Expectation: Saved metadata remains readable while stale cache entries are gone.
        """
        forward, replay = create_recompute_contexts()
        with forward:
            plan, _ = self._plan()
            values = routing._upload_route_metadata(plan, 0, torch.device("cpu"))
            cache = get_recompute_state().get_resource(_metadata._RecomputedRouteMetadata, lambda: None)
        with self.assertRaisesRegex(RuntimeError, "early stop"):
            with replay:
                same, hit = self._plan()
                self.assertTrue(hit)
                self.assertIs(routing._upload_route_metadata(same, 0, torch.device("cpu"))[2], values[2])
                raise RuntimeError("Synthetic early stop")
        self.assertFalse(cache.entries)
        self.assertEqual(tuple(map(tuple, values[2].tolist())), plan.dispatch_counts)
        with replay:
            rebuilt, hit = self._plan()
            self.assertFalse(hit)
            self.assertIsNot(routing._upload_route_metadata(rebuilt, 0, torch.device("cpu"))[2], values[2])

    def test_accelerator_stream_wait_and_allocator_lifetime_are_explicit(self) -> None:
        """Feature: Cross-stream ownership.
        Description: Mock the accelerator stream/event API without starting a device.
        Expectation: Replay waits on upload readiness and records each consumer stream.
        """
        plan = build_expert_replica_plan(self.counts, 1)
        cache = _metadata._RecomputedRouteMetadata()
        stream, ready = Mock(), Mock()
        backend = SimpleNamespace(current_stream=Mock(return_value=stream), Event=Mock(return_value=ready))
        values = tuple(
            SimpleNamespace(dtype=torch.int32, requires_grad=False, grad_fn=None, record_stream=Mock())
            for _ in range(3)
        )
        with patch.object(_metadata.torch, "cuda", backend):
            cache.remember(plan, 0, torch.device("cuda:0"), "dispatch", values)
            self.assertIs(cache.lookup(plan, 0, torch.device("cuda:0"), "dispatch"), values)
        ready.record.assert_called_once_with(stream)
        stream.wait_event.assert_called_once_with(ready)
        for value in values:
            value.record_stream.assert_called_once_with(stream)
        self.assertIsNone(cache.lookup(plan, 1, torch.device("cpu"), "dispatch"))
        cache.clear()
        self.assertFalse(cache.entries)

    def test_cache_rejects_activation_or_noninteger_payloads(self) -> None:
        """Feature: Graph-free metadata.
        Description: Try to store an activation instead of integer route descriptors.
        Expectation: The cache rejects the payload before retaining its graph.
        """
        plan = build_expert_replica_plan(self.counts, 1)
        cache = _metadata._RecomputedRouteMetadata()
        with self.assertRaisesRegex(ValueError, "graph-free integer"):
            cache.remember(plan, 0, torch.device("cpu"), "dispatch", (torch.ones(2, requires_grad=True),))
        self.assertFalse(cache.entries)
