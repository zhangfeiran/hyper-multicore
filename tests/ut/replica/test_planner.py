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
"""Validate public heuristic semantics independently of an accelerator runtime."""

import dataclasses
import inspect
import random
import unittest

from hyper_multicore.replica import ExpertExecutionPlan, ExpertReplicaConfig, build_expert_replica_plan


class TestReplicaPlanner(unittest.TestCase):
    """Cover ownership, capacity-required copies and production heuristic regressions."""

    def test_capacity_bounds(self) -> None:
        """Keep distinct-TopK worst-case bounds independent from actual allocation."""
        expected = ((0, 1, 98304), (1, 1, 87385), (1, 128, 87424), (6, 1, 32768))
        for budget, alignment, rows in expected:
            with self.subTest(budget=budget, alignment=alignment):
                self.assertEqual(ExpertReplicaConfig(24, 4, budget).maximum_receive_rows(
                    4096, 8, alignment=alignment), rows)

    def test_zero_budget_keeps_owner_dispatch(self) -> None:
        """B0 conserves original ownership even under an extreme hotspot."""
        counts = ((4096, 0, 0, 0),) * 2
        plan = build_expert_replica_plan(counts, 0)
        plan.validate()
        self.assertEqual(plan.dispatch_counts, counts)
        self.assertEqual(plan.destination_loads, (8192, 0))
        self.assertFalse(plan.transfers)

    def test_required_copies_override_optional_threshold(self) -> None:
        """A strict benefit threshold cannot reject copies needed for hard capacity."""
        counts = ((16, 0, 0, 0, 0, 0, 0, 0),) * 4
        plan = build_expert_replica_plan(counts, 1, capacity_limit=16,
                                          minimum_replica_rows=1024, minimum_load_reduction=1.0)
        plan.validate()
        self.assertEqual(plan.destination_loads, (16, 16, 16, 16))
        self.assertEqual(len(plan.transfers), 3)
        with self.assertRaises(ValueError):
            build_expert_replica_plan(counts, 1, capacity_limit=15)

    def test_impossible_optional_gain_keeps_home(self) -> None:
        """Reject optional work when even ideal balance cannot meet the gain floor."""
        counts = ((3, 2, 2, 1), (2, 3, 1, 2))
        plan = build_expert_replica_plan(counts, 2, minimum_load_reduction=0.4)
        plan.validate()
        self.assertFalse(plan.transfers)

    def test_retained_guest_refill_keeps_matrix_budget(self) -> None:
        """Preserve the production critical-expert refill after locality and consolidation."""
        source = (512, 0, 4096, 0, 0, 0, 0, 0, 512, 0, 1024, 0, 2048, 1024, 1024, 4096)
        plan = build_expert_replica_plan(
            (source,) * 4, 1, target_load=5 * sum(source) // 4,
            capacity_limit=4 * sum(source), receive_capacity=4 * sum(source),
            minimum_replica_rows=1024, minimum_load_reduction=0.4,
            rank_nodes=(0, 0, 1, 1), transfer_row_credit=1152)
        plan.validate()
        self.assertEqual(len(plan.transfers), 3)
        self.assertLessEqual(max(plan.destination_loads), 17920)
        self.assertLess(max(map(max, plan.destination_counts)), 14336)
        self.assertEqual(max(plan.replica_slots_used), 1)

    def test_sparse_completed_seed_rescues_hotspot(self) -> None:
        """Keep the default completed single-guest rescue without experimental searches."""
        hot = (397, 396, 396, 396, 396, 396)
        quotient, remainder = divmod(4096 * 6 - sum(hot), 192 - len(hot))
        source = hot + tuple(quotient + (index < remainder) for index in range(192 - len(hot)))
        plan = build_expert_replica_plan(
            (source,) * 32, 2, target_load=30720, capacity_limit=786432,
            receive_capacity=786432, minimum_replica_rows=1024, minimum_load_reduction=0.4,
            rank_nodes=(0,) * 16 + (1,) * 16)
        plan.validate()
        self.assertTrue(plan.transfers)
        self.assertLessEqual(len(plan.transfers), 8)
        self.assertLessEqual(max(plan.destination_loads), 40000)

    def test_empty_source_can_receive_guest_work(self) -> None:
        """An empty source still participates as an execution destination."""
        counts = ((64, 0, 0, 0), (0, 0, 0, 0))
        plan = build_expert_replica_plan(counts, 1, capacity_limit=32)
        plan.validate()
        self.assertEqual(sum(plan.dispatch_counts[1]), 0)
        self.assertEqual(plan.destination_loads, (32, 32))

    def test_all_empty_and_large_integer_counts(self) -> None:
        """Retain empty-route behavior and arbitrary-precision Python accounting."""
        for counts in (((0, 0, 0, 0),) * 2, ((2**65, 0, 0, 0),) * 2):
            with self.subTest(counts=counts):
                plan = build_expert_replica_plan(counts, 1)
                plan.validate()
                self.assertEqual(sum(plan.destination_loads), sum(map(sum, counts)))

    def test_invalid_inputs_and_required_controls(self) -> None:
        """Reject malformed histograms and booleans without allocating or communicating."""
        for counts in ((), ((),), ((1, 0), (1,)), ((-1, 0),), ((True, 0),), ((1.0, 0),)):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                build_expert_replica_plan(counts, 1)
        for option in ("minimum_replica_rows", "minimum_saved_rows", "transfer_row_credit"):
            with self.subTest(option=option), self.assertRaises(ValueError):
                build_expert_replica_plan(((1,),), 1, **{option: None})
        for nodes in ((True, False), (-1, 0), (0,)):
            with self.subTest(nodes=nodes), self.assertRaises(ValueError):
                build_expert_replica_plan(((4, 0),) * 2, 1, rank_nodes=nodes)

    def test_immutable_plan_and_diagnostic_validation(self) -> None:
        """Input mutations and cached summaries cannot change a saved execution plan."""
        counts = [[8, 0], [8, 0]]
        plan = build_expert_replica_plan(counts, 1)
        counts[0][0] = 0
        self.assertEqual(plan.logical_counts[0][0], 8)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            plan.config = ExpertReplicaConfig(2, 2, 0)
        object.__setattr__(plan, "destination_counts", ((0, 0), (0, 0)))
        with self.assertRaisesRegex(ValueError, "cached destination"):
            plan.validate()

    def test_random_source_conservation_and_determinism(self) -> None:
        """Exercise ragged sources, changing budgets and interleaved node membership."""
        rng = random.Random(920104)
        for iteration in range(80):
            counts = tuple(tuple(rng.randrange(40) for _ in range(12)) for _ in range(4))
            budget = rng.choice((0, 1, 2, 6))
            options = {"rank_nodes": (7, 2, 7, 2), "minimum_replica_rows": 4,
                       "minimum_load_reduction": 0.4}
            with self.subTest(iteration=iteration):
                plan = build_expert_replica_plan(counts, budget, **options)
                plan.validate()
                self.assertEqual(plan, build_expert_replica_plan(counts, budget, **options))
                self.assertEqual(tuple(map(sum, counts)), tuple(map(sum, plan.dispatch_counts)))
                self.assertLessEqual(max(plan.replica_slots_used), budget)

    def test_no_cost_model_compatibility_argument(self) -> None:
        """The standalone heuristic API has no dormant calibration branch."""
        parameters = inspect.signature(build_expert_replica_plan).parameters
        self.assertNotIn("cost_model", parameters)
        self.assertNotIn("adaptive", parameters)


class TestPlanValidation(unittest.TestCase):
    """Reject malformed saved plans before any runtime adapter consumes them."""

    def test_owner_replacement_and_unassigned_work_are_rejected(self) -> None:
        """Home ownership, empty slots and source conservation are independent invariants."""
        config = ExpertReplicaConfig(2, 2, 1)
        logical = ((1, 0), (0, 1))
        variants = (
            (((1, -1), (1, -1)), ((1, 0, 0, 0), (0, 0, 1, 0))),
            (((0, -1), (1, -1)), ((0, 1, 0, 0), (0, 0, 1, 0))),
            (((0, -1), (1, -1)), ((0, 0, 0, 0), (0, 0, 1, 0))),
        )
        for slots, dispatch in variants:
            with self.subTest(slots=slots, dispatch=dispatch), self.assertRaises(ValueError):
                ExpertExecutionPlan(config, logical, slots, dispatch).validate()
