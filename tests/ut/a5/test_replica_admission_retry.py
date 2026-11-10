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
"""Retry otherwise rejected preference targets without changing accepted placements."""

import os
import unittest
from dataclasses import replace
from unittest.mock import patch

from hyper_multicore.replica import planner
from hyper_multicore.replica.cost import ExpertReplicaCostModel


class TestReplicaAdmissionRetry(unittest.TestCase):
    """Keep allocation bounds independent of the optional load-reduction target."""

    @staticmethod
    def diffuse_counts() -> list[list[int]]:
        """Represent 4096 distinct TopK6 routes per source with a diffuse hot owner."""
        owner = 5529
        remaining = 4096 * 6 - owner
        totals = [owner] + [remaining // 7 + int(rank < remaining % 7) for rank in range(7)]
        counts = [rows // 6 + int(expert < rows % 6) for rows in totals for expert in range(6)]
        return [list(counts) for _ in range(8)]

    def test_uncalibrated_preference_keeps_owner_without_retry(self) -> None:
        """A smaller receive peak alone does not justify the optional retry."""
        counts = self.diffuse_counts()
        for budget in (2, 6):
            with self.subTest(budget=budget):
                options = dict(adaptive=True, minimum_replica_rows=1024, minimum_load_reduction=.4,
                               capacity_limit=196608, receive_capacity=196608)
                expected = planner._build_expert_replica_plan(counts, budget, target_load=24576, **options)
                with patch.object(planner, '_uncalibrated_copies', wraps=planner._uncalibrated_copies) as select:
                    actual = planner._build_expert_replica_plan(counts, budget, target_load=30720, **options)
                actual.validate()
                self.assertTrue(expected.transfers)
                self.assertFalse(actual.transfers)
                self.assertEqual([call.args[5] for call in select.call_args_list], [30720])

    def test_calibrated_retry_must_cover_search_and_uncertainty(self) -> None:
        """A paired group estimate must cover both fixed replica and extra search costs."""
        counts = self.diffuse_counts()
        model = ExpertReplicaCostModel(
            calibration_version=2, backend='push', hidden_size=128, intermediate_size=256,
            ep_size=8, transport='p2p', forward_ms=((0, 0.), (50000, .05)),
            backward_ms=((0, 0.), (50000, .05)), weight_ms=0., gradient_ms=0.,
            effective_group_ms=(0., .001, 0., 0., 0., 16.), _rank_nodes=(0, 0, 0, 0, 1, 1, 1, 1),
            minimum_gain_ms=1., quota_search_budget_ms=2.,
        )
        options = dict(adaptive=True, target_load=30720, minimum_replica_rows=1024, minimum_load_reduction=.4,
                       capacity_limit=196608, receive_capacity=196608)

        admitted = planner._build_expert_replica_plan(counts, 2, cost_model=model, **options)
        admitted.validate()
        self.assertTrue(admitted.transfers)
        self.assertEqual(max(admitted.destination_loads), 24576)
        for budget in (0., 3.):
            with self.subTest(search_budget_ms=budget):
                rejected = planner._build_expert_replica_plan(
                    counts, 2, cost_model=replace(model, quota_search_budget_ms=budget), **options)
                rejected.validate()
                self.assertFalse(rejected.transfers)

    def test_profitable_preference_keeps_first_candidate(self) -> None:
        """Existing concentrated-hot choices must not pay for a second search."""
        counts = [[4096] * 6 + [0] * 42 for _ in range(8)]
        for budget in (2, 6):
            with self.subTest(budget=budget):
                with patch.object(planner, '_uncalibrated_copies', wraps=planner._uncalibrated_copies) as select:
                    actual = planner._build_expert_replica_plan(
                        counts, budget, adaptive=True, target_load=30720, minimum_replica_rows=1024,
                        minimum_load_reduction=.4, capacity_limit=196608, receive_capacity=196608)
                actual.validate()
                self.assertTrue(actual.transfers)
                self.assertEqual(select.call_count, 1)

    def test_impossible_fraction_keeps_owner_without_retry(self) -> None:
        """An impossible admission bound must remain a cheap owner-only decision."""
        with patch.object(planner, '_uncalibrated_copies', wraps=planner._uncalibrated_copies) as select:
            actual = planner._build_expert_replica_plan(
                self.diffuse_counts(), 6, adaptive=True, target_load=30720,
                minimum_replica_rows=1024, minimum_load_reduction=.8,
                capacity_limit=196608, receive_capacity=196608)
        self.assertFalse(actual.transfers)
        self.assertEqual(select.call_count, 0)

    def test_required_capacity_does_not_use_optional_retry(self) -> None:
        """Mandatory replicas remain valid even if optional gain is impossible."""
        counts = [[4096] * 6 + [0] * 42 for _ in range(8)]
        with patch.object(planner, '_uncalibrated_copies', wraps=planner._uncalibrated_copies) as select:
            actual = planner._build_expert_replica_plan(
                counts, 2, adaptive=True, target_load=30720, minimum_replica_rows=1000000,
                minimum_load_reduction=.99, capacity_limit=49152, receive_capacity=49152)
        actual.validate()
        self.assertTrue(actual.transfers)
        self.assertLessEqual(max(actual.destination_loads), 49152)
        self.assertEqual(select.call_count, 1)

    def test_complete_candidate_fits_two_physical_guests_and_keeps_all_routes(self) -> None:
        """A full redistribution that fits every receiver should not be quota-thinned."""
        source = [4096, 0, 9, 97, 0, 0, 4096, 0, 4096, 4096, 0, 0,
                  0, 0, 1200, 0, 0, 0, 0, 3354, 0, 11, 0, 82,
                  0, 0, 1, 0, 0, 0, 224, 0, 7, 2993, 0, 0,
                  0, 0, 0, 0, 0, 208, 0, 0, 1, 5, 0, 0]
        counts = [source] * 8
        options = dict(adaptive=True, target_load=30720, minimum_replica_rows=1024,
                       minimum_load_reduction=.4, capacity_limit=196608, receive_capacity=196608)
        with patch.dict(os.environ, {"HYPER_PARALLEL_REPLICA_COMPLETE_GUEST_SEARCH": "0"}):
            baseline = planner._build_expert_replica_plan(counts, 2, **options)
        with patch.dict(os.environ, {"HYPER_PARALLEL_REPLICA_COMPLETE_GUEST_SEARCH": "1"}):
            candidate = planner._build_expert_replica_plan(counts, 2, **options)
        candidate.validate()
        self.assertEqual(candidate.config.replica_slots_per_rank, 2)
        self.assertLessEqual(max(candidate.replica_slots_used), 2)
        self.assertEqual(max(candidate.destination_loads), 24576)
        self.assertGreater(max(baseline.destination_loads), 30720)

    def test_projected_quota_keeps_two_slots_and_exact_dispatch(self) -> None:
        """An affordable wider quota can reproduce the B6 plan with two slots."""
        source = [123, 332, 981, 924, 1371, 1361, 1320, 243, 777, 911, 116, 568,
                  262, 829, 148, 223, 267, 22, 678, 28, 1312, 240, 386, 8,
                  666, 35, 18, 694, 33, 582, 1982, 3907, 735, 1240, 2737, 678,
                  1067, 186, 225, 1432, 2, 78, 1588, 547, 106, 58, 391, 351]
        baseline, projected, wider = self._projected_plans(source)
        projected.validate()
        self.assertEqual(projected.config.replica_slots_per_rank, 2)
        self.assertEqual(max(projected.destination_loads), 33240)
        self.assertLess(max(projected.destination_loads), max(baseline.destination_loads))
        self.assertEqual(projected.destination_loads, wider.destination_loads)
        self.assertEqual(projected.transfers, wider.transfers)
        for left, right in zip(projected.dispatch_counts, wider.dispatch_counts, strict=True):
            self.assertEqual(left, tuple(row for index, row in enumerate(right)
                                         if index % wider.config.slots_per_rank < projected.config.slots_per_rank))

    def test_projection_rejects_unpaid_or_infeasible_wider_quotas(self) -> None:
        """Retain the first B2 route if wider fanout is unpriced or exceeds two slots."""
        pressure = [423, 31, 1181, 56, 311, 1067, 665, 110, 3694, 265, 558, 3556,
                    205, 1409, 188, 205, 66, 191, 1145, 336, 45, 170, 78, 964,
                    1978, 163, 830, 127, 178, 675, 763, 473, 62, 903, 1534, 288,
                    808, 903, 900, 768, 146, 289, 881, 1311, 209, 3, 1384, 273]
        occupancy = [15, 214, 86, 130, 4096, 456, 1, 101, 367, 30, 55, 116,
                     136, 1996, 1201, 93, 904, 63, 1926, 389, 1446, 264, 1836, 3548,
                     411, 0, 105, 82, 32, 857, 0, 812, 491, 556, 104, 1254,
                     2883, 4, 165, 765, 634, 772, 1195, 2, 1433, 233, 125, 384]
        for source, fits in ((pressure, True), (occupancy, False)):
            with self.subTest(wider_fits=fits):
                baseline, projected, wider = self._projected_plans(source)
                self.assertEqual(max(wider.replica_slots_used) <= 2, fits)
                self.assertLess(max(wider.destination_loads), max(projected.destination_loads))
                self.assertEqual(projected, baseline)

    @staticmethod
    def _projected_plans(source: list[int]) -> tuple[planner.ExpertExecutionPlan, ...]:
        """Use the same two-node histogram for old B2, projected B2, and B6."""
        options = dict(adaptive=True, target_load=40960, minimum_replica_rows=1024,
                       minimum_load_reduction=.4, capacity_limit=196608, receive_capacity=196608,
                       _rank_nodes=(0,) * 4 + (1,) * 4)
        plans = []
        for budget, enabled in ((2, '0'), (2, '1'), (6, '0')):
            with patch.dict(os.environ, {"HYPER_PARALLEL_REPLICA_COMPLETE_GUEST_SEARCH": "1",
                                         "HYPER_PARALLEL_REPLICA_PROJECTED_GUEST_SEARCH": enabled}):
                plans.append(planner._build_expert_replica_plan([source] * 8, budget, **options))
        return tuple(plans)
