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
"""Bounded planner conservation and logical-route contract tests."""

from dataclasses import replace
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_multicore.replica import (
    ExpertReplicaConfig,
    build_expert_replica_plan,
    planner,
    transport,
)
from hyper_multicore.replica.routing import (
    ReplicaRoute,
)
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestHotReplica(unittest.TestCase):
    """Check capacity against independent distinct-TopK route generation."""

    def test_fast_integer_validation_keeps_rejecting_zero_booleans(self) -> None:
        """Zero-valued malformed counts must not bypass validation or conservation checks."""
        plan = build_expert_replica_plan([[8, 0], [0, 0]], 1)
        for value in (False, True, 0.0, -1, '0', None):
            with self.subTest(value=value):
                counts = [[8, value], [0, 0]]
                with self.assertRaises(ValueError):
                    build_expert_replica_plan(counts, 1)
                dispatch = [list(row) for row in plan.dispatch_counts]
                dispatch[0][-1] = value
                with self.assertRaises(ValueError):
                    replace(plan, dispatch_counts=dispatch).validate()

    def test_transposed_totals_preserve_integer_subclasses_and_dimensions(self) -> None:
        """Keep accepted integer counts and reject rectangular truncation explicitly."""
        class Count(int):
            """Exercise the existing integer subclass contract."""

        plan = build_expert_replica_plan([[Count(8), Count(0)], [Count(0), Count(0)]], 1)
        plan.validate()
        expected = tuple(tuple(sum(row[rank * 2 + slot] for row in plan.dispatch_counts)
                               for slot in range(2)) for rank in range(2))
        self.assertEqual(plan.destination_counts, expected)
        with self.assertRaisesRegex(ValueError, 'dispatch row'):
            replace(plan, dispatch_counts=[row[:-1] for row in plan.dispatch_counts]).destination_counts

    def test_owned_p2p_return_preserves_guest_storage_and_fp32_order(self):
        """Consume only home rows while preserving guest slices and cancellation order."""
        plan = build_expert_replica_plan([[100, 0, 0, 0]] * 4, 1)
        route = ReplicaRoute(plan, torch.empty(0), torch.empty(0), 0, None)
        gradient = torch.full((2, 2, 2), 5.0)
        gradient[0].fill_(0.5)
        expected = gradient[:1].clone()
        values = (0.0, float(2**24), 1.0, -float(2**24))
        for peer in range(1, 4):
            expected.add_(values[peer])

        def _operation(op, tensor, peer, group):
            return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)

        def _exchange(operations):
            for pending in operations:
                pending.tensor.fill_(values[pending.peer])

        with patch.object(transport.dist, "P2POp", side_effect=_operation), \
                patch.object(transport.dist, "get_global_rank", side_effect=lambda _group, peer: peer), \
                patch.object(transport, "_exchange", side_effect=_exchange):
            result, = transport.return_gradients((gradient,), route, consume=True)
        self.assertEqual(result.data_ptr(), gradient.data_ptr())
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        torch.testing.assert_close(gradient[1], torch.full((2, 2), 5.0))

    def test_owned_return_rejects_non_fp32_or_autograd_owned_inputs(self):
        """An ownership opt-in must not silently convert or mutate saved differentiable inputs."""
        plan = build_expert_replica_plan([[1]], 0)
        route = ReplicaRoute(plan, torch.empty(0), torch.empty(0), 0, None)
        for value in (torch.ones(1, 2, 2, dtype=torch.bfloat16), torch.ones(1, 2, 2, requires_grad=True)):
            with self.assertRaisesRegex(ValueError, "detached FP32"), patch.object(transport, "_exchange") as exchange:
                transport.return_gradients((value,), route, consume=True)
            exchange.assert_not_called()

    def test_copy_threshold_keeps_mandatory_work_and_consolidates_fragments(self):
        """One existing copy absorbs necessary work before another small copy is removed."""
        counts = [[512] * 8 + [0] * 16] * 4
        config = ExpertReplicaConfig(24, 4, 1)
        upper = config.maximum_receive_rows(512, 8, alignment=1)
        for target in (None, 5120, 7168, 11008):
            with self.subTest(target=target):
                plan = build_expert_replica_plan(counts, 1, target_load=min(target, upper) if target else None,
                                                minimum_replica_rows=4096, capacity_limit=upper)
                plan.validate()
                self.assertEqual(len(plan.transfers), 1)
                self.assertLessEqual(max(plan.destination_loads), upper)
                self.assertGreater(max(plan.destination_loads), 8192)

    def test_copy_consolidation_uses_updated_guest_rows_after_removals(self) -> None:
        """Preserve capacity and ordered choices when earlier removals grow another guest."""
        counts = ((9, 7, 4, 4, 3, 4, 2, 4),) + ((0,) * 8,) * 3
        copies = [{0: 6, 1: 4}, {2: 4, 3: 4, 0: 2}, {4: 3, 5: 4, 0: 1, 1: 2}, {6: 2, 7: 4, 1: 1}]
        expected = [{0: 9, 1: 1}, {2: 4, 3: 4}, {4: 3, 5: 4, 1: 3}, {6: 2, 7: 4, 1: 3}]

        planner._prune_small_copies(copies, counts, [16, 8, 7, 6], ExpertReplicaConfig(8, 4, 2), 3, 10)

        self.assertEqual([list(row.items()) for row in copies], [list(row.items()) for row in expected])
        self.assertEqual([sum(row.get(expert, 0) for row in copies) for expert in range(8)], list(counts[0]))
        self.assertLessEqual(max(sum(row.values()) for row in copies), 10)

    def test_direct_copy_removal_precedes_indexed_consolidation(self) -> None:
        """Build replacement evidence after an earlier direct removal changed the quotas."""
        counts = ((8, 7, 4, 4, 3, 4, 2, 4),) + ((0,) * 8,) * 3
        copies = [{0: 5, 1: 4}, {2: 4, 3: 4, 0: 2}, {4: 3, 5: 4, 0: 1, 1: 2}, {6: 2, 7: 4, 1: 1}]
        expected = [{0: 6, 1: 4}, {2: 4, 3: 4, 0: 2}, {4: 3, 5: 4, 1: 3}, {6: 2, 7: 4}]

        planner._prune_small_copies(copies, counts, [15, 8, 7, 6], ExpertReplicaConfig(8, 4, 2), 3, 10)

        self.assertEqual([list(row.items()) for row in copies], [list(row.items()) for row in expected])

    def test_copy_threshold_rejects_invalid_policy_or_infeasible_capacity(self):
        """Invalid thresholds and impossible receive limits fail before execution."""
        for threshold in (-1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "minimum_replica_rows"):
                build_expert_replica_plan([[2, 0], [2, 0]], 1, minimum_replica_rows=threshold)
        with self.assertRaisesRegex(ValueError, "capacity_limit"):
            build_expert_replica_plan([[2, 0], [2, 0]], 1, capacity_limit=1)

    def test_random_routes(self):
        """Exercise different EP degrees, home counts, skew and spare slots."""
        rng = random.Random(718)
        for _ in range(240):
            ranks, home, tokens = rng.randint(2, 6), rng.randint(1, 6), rng.randint(1, 32)
            experts = ranks * home
            top_k = rng.randint(1, experts)
            counts = [[0] * experts for _ in range(ranks)]
            for row in counts:
                for _ in range(tokens):
                    for expert in rng.sample(range(experts), top_k):
                        row[expert] += 1
            for budget in (0, 1, home, home + 1):
                plan = build_expert_replica_plan(counts, budget)
                self.assertEqual(plan, build_expert_replica_plan(counts, budget))
                plan.validate()
                self.assertLessEqual(max(plan.destination_loads),
                                     plan.config.maximum_receive_rows(tokens, top_k, alignment=1))

    def test_single_hot_expert_balances_with_one_slot(self):
        """One hot expert uses all ranks without changing its parameter owner."""
        plan = build_expert_replica_plan([[100, 0, 0, 0]] * 4, 1)
        self.assertEqual(plan.destination_loads, (100, 100, 100, 100))
        self.assertTrue(all(transfer.owner_rank == 0 for transfer in plan.transfers))

    def test_invalid_config_and_counts(self):
        """Reject impossible topology and non-integer route counts."""
        for args in ((4, 3, 1), (4, 2, -1), (4, 2, True)):
            with self.assertRaises(ValueError):
                ExpertReplicaConfig(*args)
        for counts in (([1, -1],), ([True, 1],), ([1, 2], [1])):
            with self.assertRaises(ValueError):
                build_expert_replica_plan(counts, 1)
