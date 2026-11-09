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
"""Invocation ownership and exact route replay with actual Torch checkpoints."""

import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.checkpoint import DefaultDeviceType

from hyper_multicore.replica import ExpertReplicaConfig, routing
from hyper_multicore.replica.recompute import create_recompute_contexts, get_recompute_state, replica_checkpoint
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
class TestReplicaRecompute(unittest.TestCase):
    """Keep plans bound to checkpoint calls, including balanced bypass ordinals."""

    def setUp(self) -> None:
        """Keep CPU checkpoints independent of an installed accelerator default."""
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")
        self.config = ExpertReplicaConfig(4, 2, 1)
        self.counts = ((20, 10, 0, 0), (20, 10, 0, 0))

    def test_both_checkpoint_modes_reuse_exact_plan_after_home_ordinal(self) -> None:
        """Actual backward replays the original invocation and preserves autograd values."""
        for reentrant in (False, True):
            with self.subTest(reentrant=reentrant):
                observations = []

                def operation(value: torch.Tensor) -> torch.Tensor:
                    """Use the same home/hot call order during F and R."""
                    state = get_recompute_state()
                    routing._record_home_decision()
                    plan, reused = routing._plan_replica_route(self.counts, self.config, None, 0)
                    observations.append((state.invocation_id, state.is_recomputing, plan, reused))
                    return value.sin() * value

                value = torch.arange(1.0, 5.0, requires_grad=True)
                expected = value.detach().clone().requires_grad_()
                (expected.sin() * expected).sum().backward()
                output = replica_checkpoint(operation, value, use_reentrant=reentrant)
                output.sum().backward()
                torch.testing.assert_close(value.grad, expected.grad)
                self.assertEqual(len(observations), 2)
                first, second = observations[0], observations[1]
                self.assertIs(first[0], second[0])
                self.assertEqual((first[1], second[1], first[3], second[3]), (False, True, False, True))
                self.assertIs(first[2], second[2])
                self.assertIsNone(get_recompute_state())

    def test_changed_source_counts_and_smaller_capacity_cannot_replay(self) -> None:
        """An unchanged column total is insufficient, and the current hard bound wins."""
        forward, recompute = create_recompute_contexts()
        with forward:
            original, _ = routing._plan_replica_route(self.counts, self.config, None, 0, capacity_limit=60)
        changed = ((21, 9, 0, 0), (19, 11, 0, 0))
        with recompute:
            actual, reused = routing._plan_replica_route(changed, self.config, None, 0, capacity_limit=60)
        self.assertFalse(reused)
        self.assertEqual(actual.logical_counts, changed)
        self.assertIsNot(actual, original)
        forward, recompute = create_recompute_contexts()
        with forward:
            routing._plan_replica_route(self.counts, self.config, None, 0, capacity_limit=60)
        with recompute, self.assertRaises(ValueError):
            routing._plan_replica_route(self.counts, self.config, None, 0, capacity_limit=29)

    def test_binary_counts_replay_rejects_changed_source_rows(self) -> None:
        """Equal expert totals do not permit a replay when source assignments change."""
        original = np.asarray(self.counts, dtype=np.int32)
        changed = np.asarray(((21, 9, 0, 0), (19, 11, 0, 0)), dtype=np.int32)
        variants = ((original, original.copy(), True), (original, changed, False),
                    (original.tolist(), original.copy(), True), (original, original.tolist(), True))
        for first_counts, current, expected_reuse in variants:
            with self.subTest(
                expected_reuse=expected_reuse, forward_type=type(first_counts), replay_type=type(current)
            ):
                forward, recompute = create_recompute_contexts()
                with forward:
                    first, _ = routing._plan_replica_route(first_counts, self.config, None, 0, capacity_limit=60)
                with recompute:
                    second, reused = routing._plan_replica_route(current, self.config, None, 0, capacity_limit=60)
                self.assertEqual(reused, expected_reuse)
                expected = current.tolist() if isinstance(current, np.ndarray) else current
                self.assertEqual(second.logical_counts, tuple(map(tuple, expected)))
                self.assertEqual(second is first, expected_reuse)

    def test_optional_checkpoint_readback_reaches_gathered_counts(self) -> None:
        """Ordinary calls keep lists; enabled checkpoint calls retain exact integer bytes."""
        counts = torch.tensor(self.counts, dtype=torch.int32)
        gathered = routing._GatheredReplicaCounts(counts, [[0, 30, 15, 30, 0, 20, 60]] * 2)
        ids = torch.empty((30, 1), dtype=torch.int64)
        with patch.object(routing, "_hot_replay_readback_enabled", return_value=True):
            _, outside, _ = routing._route_counts(ids, self.config, None, None, None, gathered)
            forward, _ = create_recompute_contexts()
            with forward:
                _, inside, _ = routing._route_counts(ids, self.config, None, None, None, gathered)
        self.assertIsInstance(outside, list)
        self.assertIsInstance(inside, np.ndarray)
        np.testing.assert_array_equal(inside, counts.numpy())

    def test_nested_exception_restores_outer_scope_and_clears_partial_entries(self) -> None:
        """A failed inner invocation cannot poison the next consumer's ordinal."""
        outer, _ = create_recompute_contexts()
        inner, _ = create_recompute_contexts()
        with outer:
            state = get_recompute_state()
            with self.assertRaisesRegex(RuntimeError, "consumer"):
                with inner:
                    routing._plan_replica_route(self.counts, self.config, None, 0)
                    raise RuntimeError("consumer failed")
            self.assertIs(get_recompute_state(), state)
        self.assertIsNone(get_recompute_state())
        with inner:
            cache = get_recompute_state().get_resource(routing._RecomputedReplicaPlans, routing._RecomputedReplicaPlans)
            self.assertEqual(cache.entries, [])

    def test_deferred_reverse_backward_keeps_invocations_separate(self) -> None:
        """Reverse order backward cannot consume another layer's retained plan."""
        observed = {}
        inputs, outputs = [], []
        for index in range(2):
            counts = tuple(tuple(value + index for value in row) for row in self.counts)

            def operation(value: torch.Tensor, logical: tuple = counts, layer: int = index) -> torch.Tensor:
                """Capture the layer's immutable logical counts in the closure."""
                plan, reused = routing._plan_replica_route(logical, self.config, None, 0)
                observed.setdefault(layer, []).append((plan, reused))
                return value.sin() * value

            value = torch.full((3,), 1.0 + index, requires_grad=True)
            inputs.append(value)
            outputs.append(replica_checkpoint(operation, value))
        for output in reversed(outputs):
            output.sum().backward()
        for layer, value in enumerate(inputs):
            expected = value.detach().sin() + value.detach() * value.detach().cos()
            torch.testing.assert_close(value.grad, expected)
            first, replay = observed[layer]
            self.assertIs(first[0], replay[0])
            self.assertEqual((first[1], replay[1]), (False, True))

    def test_retain_graph_replay_remains_correct_after_cache_release(self) -> None:
        """A second backward recomputes safely after the first replay clears its cache."""
        value = torch.ones(3, requires_grad=True)

        def operation(argument: torch.Tensor) -> torch.Tensor:
            """Exercise real saved-tensor hooks after a planner resource."""
            routing._plan_replica_route(self.counts, self.config, None, 0)
            return argument.sin() * argument

        result = replica_checkpoint(operation, value).sum()
        result.backward(retain_graph=True)
        first = value.grad.clone()
        value.grad = None
        result.backward()
        torch.testing.assert_close(value.grad, first)
        self.assertIsNone(get_recompute_state())

    def test_external_context_override_fails_before_execution(self) -> None:
        """Do not silently replace the invocation contract with a foreign context."""
        with patch("hyper_multicore.replica.recompute.checkpoint") as execute:
            with self.assertRaisesRegex(ValueError, "owns context_fn"):
                replica_checkpoint(lambda x: x, torch.ones(1), context_fn=lambda: ())
            execute.assert_not_called()
