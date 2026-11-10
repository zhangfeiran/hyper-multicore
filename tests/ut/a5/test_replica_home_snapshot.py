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
"""Reuse invocation-local integer totals without sharing mutable baseline quotas."""

from copy import deepcopy
import random
import unittest
from unittest.mock import patch

from hyper_multicore.replica import planner, plan as plan_module
from hyper_multicore.replica.capacity import ExpertReplicaConfig
from hyper_multicore.replica.cost import ExpertReplicaCostModel
from tests.common.mark_utils import arg_mark


class TestReplicaHomeSnapshot(unittest.TestCase):
    """Preserve quotas, float scores and mutable ownership while removing a second matrix pass."""

    def test_final_destination_quotas_match_independent_dispatch_reduction(self) -> None:
        """Reuse final quotas while retaining exact integer destination metadata."""
        generator = random.Random(20261105)
        for budget in (0, 1, 2, 6):
            for iteration in range(12):
                counts = tuple(tuple(generator.randrange(50) + (300 if expert == 0 else 0)
                                     for expert in range(24)) for _ in range(4))
                result = planner._build_expert_replica_plan(
                    counts, budget, adaptive=bool(budget), _rank_nodes=(0, 1, 0, 1))
                width = result.config.slots_per_rank
                expected = tuple(tuple(sum(source[rank * width + slot] for source in result.dispatch_counts)
                                       for slot in range(width)) for rank in range(4))
                with self.subTest(budget=budget, iteration=iteration), patch.object(
                        plan_module, "_column_totals", side_effect=AssertionError("Unexpected second reduction")):
                    self.assertEqual(result.destination_counts, expected)
                    self.assertEqual(result.destination_loads, tuple(map(sum, expected)))
                result.validate()

    def test_validation_rejects_destination_cache_drift(self) -> None:
        """Diagnostic validation must independently check any prefilled quota cache."""
        counts = ((10, 1, 0, 0, 3, 2, 0, 0),) * 4
        result = planner._build_expert_replica_plan(counts, 1, adaptive=True)
        original = result.destination_counts
        changed = (tuple(value + 1 for value in original[0]), *original[1:])
        object.__setattr__(result, "destination_counts", changed)
        with self.assertRaisesRegex(ValueError, "cached destination quotas"):
            result.validate()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_snapshot_builds_independent_owner_quotas_without_reading_counts_again(self) -> None:
        """Feature: Call-local owner metadata reuse.
        Description: Rebuild an owner layout from exact immutable totals while denying a second matrix read.
        Expectation: All values match, and mutating returned containers changes neither input nor baseline.
        """
        config = ExpertReplicaConfig(8, 4, 2)
        counts = ((3, 1, 0, 8, 0, 0, 4, 0),) * 4
        original = planner._home_layout(counts, config)
        snapshot = tuple(tuple(values) for values in original[:2])

        class UnreadCounts:
            """Reject accidental iteration of already summarized counts."""

            def __iter__(self) -> object:
                """Reject a redundant full count read."""
                raise AssertionError("The snapshot path read the full count matrix again")

        actual = planner._home_layout(UnreadCounts(), config, _home_counts=snapshot)
        self.assertEqual(actual, original)
        for new, old in zip(actual, original):
            self.assertIsNot(new, old)
        saved = deepcopy(original)
        actual[0][0] = 99
        actual[1][0] = 99
        actual[2][0].clear()
        self.assertEqual(original, saved)
        self.assertEqual(snapshot, tuple(tuple(values) for values in saved[:2]))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_adaptive_snapshot_preserves_quota_order_and_calibrated_scores(self) -> None:
        """Feature: Snapshot compatibility across planner policies.
        Description: Compare fresh and reused summaries on seeded uncalibrated and calibrated searches.
        Expectation: Every quota, insertion order and score is identical and the snapshot stays immutable.
        """
        generator = random.Random(20261105)
        config = ExpertReplicaConfig(12, 4, 2)
        model = ExpertReplicaCostModel(
            calibration_version=2, backend="push", hidden_size=16, intermediate_size=8, ep_size=4,
            transport="p2p", forward_ms=((0, 0), (4096, 4)), backward_ms=((0, 0), (4096, 4)),
            weight_ms=.01, gradient_ms=.01, forward_repeats=2, backward_weight_ms=0)
        for iteration in range(24):
            counts = tuple(tuple(generator.randrange(30) + (200 if rank == 0 and expert == 0 else 0)
                                 for expert in range(12)) for rank in range(4))
            totals, loads, baseline = planner._home_layout(counts, config)
            saved = deepcopy(baseline)
            snapshot = tuple(totals), tuple(loads)
            target = (sum(loads) + 3) // 4
            for costs in (None, model):
                options = {"minimum_replica_rows": 16, "admission": 2000, "rank_nodes": (0, 1, 1, 0)}
                expected = planner._adaptive_copies(counts, config, max(loads), max(loads), costs, target, **options)
                actual = planner._adaptive_copies(counts, config, max(loads), max(loads), costs, target,
                                                  _home_counts=snapshot, **options)
                with self.subTest(iteration=iteration, calibrated=costs is not None):
                    self.assertEqual(actual, expected)
                    self.assertEqual([tuple(row.items()) for row in actual[0]],
                                     [tuple(row.items()) for row in expected[0]])
                    self.assertEqual(snapshot, (tuple(totals), tuple(loads)))
                    for row in actual[0]:
                        row.clear()
                    self.assertEqual(baseline, saved)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_real_planner_summarizes_the_count_matrix_once(self) -> None:
        """Feature: Reached summary reuse.
        Description: Run a real hot placement through the public planning boundary.
        Expectation: Its adaptive search receives the same immutable summary and returns a valid replica plan.
        """
        counts = ((600, 300, 50, 50, 20, 10, 0, 0),) * 4
        original = planner._home_layout
        observations = []

        def observe(*arguments: object, **options: object) -> tuple:
            """Retain actual full-summary and reuse decisions without replacing their arithmetic."""
            observations.append(options.get("_home_counts"))
            return original(*arguments, **options)

        with patch.object(planner, "_home_layout", side_effect=observe):
            result = planner._build_expert_replica_plan(counts, 2, adaptive=True)
        result.validate()
        self.assertTrue(result.transfers)
        self.assertEqual(sum(item is None for item in observations), 1)
        self.assertEqual(sum(item is not None for item in observations), 1)
