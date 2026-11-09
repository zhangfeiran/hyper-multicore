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
"""Prove compact rejection and accepted-quota reuse without distributed setup."""

import unittest
from unittest.mock import patch

import torch

from hyper_multicore.replica import ExpertReplicaConfig, planner, routing


class TestCompactAdmission(unittest.TestCase):
    """Keep rejected routes bounded and preserve complete source quotas on acceptance."""

    def setUp(self) -> None:
        """Construct a feasible distinct-TopK route whose optional replica is rejected."""
        self.config = ExpertReplicaConfig(8, 4, 1)
        self.counts = ((600, 360, 200, 200, 200, 200, 200, 88),) * 4
        self.ids = torch.repeat_interleave(torch.arange(8), torch.tensor(self.counts[0])).reshape(2, 1024).T
        totals = list(map(sum, zip(*self.counts)))
        self.loads = [sum(totals[start : start + 2]) for start in range(0, 8, 2)]
        self.summary = [[0, 1024, 15, sum(row), min(row), max(row), load] for row, load in zip(self.counts, self.loads)]
        self.gathered = routing._GatheredReplicaCounts(torch.tensor(self.counts, dtype=torch.int32), self.summary)

    def _prepare(self, **options):
        with patch.object(routing, "dist") as collective:
            collective.is_initialized.return_value = True
            collective.get_rank.return_value = 0
            collective.get_world_size.return_value = 4
            return routing.prepare_replica_route(
                self.ids,
                self.config,
                compact_home=True,
                dynamic_header=(0, 1024, 15),
                _rank_nodes=(0, 0, 1, 1),
                _gathered_counts=self.gathered,
                **options,
            )

    def test_rejected_quota_avoids_full_table_and_materialization(self) -> None:
        """An admission failure cannot hide full Python count preparation behind home-only output."""
        with patch.object(routing, "_route_counts", side_effect=AssertionError("full table reached")):
            route = self._prepare(minimum_replica_rows=1024, minimum_load_reduction=0.4)
        self.assertTrue(route.home_only)
        self.assertTrue(route.plan_bypassed)
        self.assertIs(route.physical_ids, self.ids)
        self.assertEqual(route.plan.destination_loads, tuple(self.loads))
        self.assertEqual(tuple(route.counts_by_source.shape), (4, 8))

    def test_admitted_quota_is_not_searched_twice(self) -> None:
        """The full planner consumes exactly the quotas selected from compact statistics."""
        with (
            patch.object(
                routing, "_admitted_uncalibrated_copies", wraps=routing._admitted_uncalibrated_copies
            ) as admit,
            patch.object(planner, "_admitted_uncalibrated_copies", side_effect=AssertionError("duplicate search")),
        ):
            route = self._prepare(minimum_replica_rows=0, minimum_load_reduction=0.0)
        admit.assert_called_once()
        self.assertFalse(route.home_only)
        route.plan.validate()
        self.assertEqual(route.plan.logical_counts, self.counts)
        counts = torch.bincount(route.physical_ids.flatten(), minlength=self.config.physical_experts)
        torch.testing.assert_close(counts, route.counts_by_source[0].long())

    def test_invalid_source_controls_precede_compact_admission(self) -> None:
        """An early home bypass must still reject false totals and autograd disagreement."""
        for column, value in ((0, 1), (2, 0), (3, 2047), (5, 1025)):
            controls = [list(row) for row in self.summary]
            controls[1][column] = value
            self.gathered = routing._GatheredReplicaCounts(torch.tensor(self.counts), controls)
            with self.subTest(column=column), patch.object(routing, "_compact_candidate") as admit:
                with self.assertRaises((RuntimeError, ValueError)):
                    self._prepare(minimum_replica_rows=1024, minimum_load_reduction=0.4)
                admit.assert_not_called()
