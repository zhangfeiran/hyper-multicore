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
"""CPU contracts for planner count formats and optional execution dependencies."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from hyper_multicore.experimental import HotReplicaPlanner, PlacementRequest
from hyper_multicore.modules.hot_replica.bounded_placement import render_geometry
from hyper_multicore.modules.hot_replica._planner.aligned import materialize_aligned_moves
from hyper_multicore.modules.hot_replica._planner.capacity import ExpertReplicaConfig
from hyper_multicore.modules.hot_replica._planner.receive import ReplicaCost, SourceQuota, receive_limit, select_replica_plan


class TestHotReplicaPlacement(unittest.TestCase):
    """Reject invalid metadata before any optional device kernel is called."""

    def test_receive_limit_is_exact_and_auto_has_no_limit(self) -> None:
        self.assertIsNone(receive_limit("auto", 525, 4))
        self.assertEqual(receive_limit(102, 525, 4), 134)
        self.assertEqual(receive_limit(100, 525, 4), 132)

    def test_capacity_placement_preserves_source_intervals(self) -> None:
        """Split the same logical expert in source-rank order for both wire dtypes."""
        for dtype in (torch.int32, torch.int64):
            with self.subTest(dtype=dtype):
                counts = torch.tensor([[4, 0], [2, 0]], dtype=dtype)
                plan = select_replica_plan(counts.numpy(), ExpertReplicaConfig(2, 2, 1), soft_target=3,
                                           source_quota=SourceQuota.RANK_ORDER)
                ids = torch.zeros(4, 1, dtype=torch.int32)
                geometry = render_geometry(plan, SimpleNamespace(rank_id=0), ids.device, 4, prepare_metadata=False)
                result = geometry.lower(torch.empty(4, 4), ids, torch.arange(4, dtype=torch.int32))
                torch.testing.assert_close(result.virtual_ids, torch.tensor([[0], [0], [0], [3]], dtype=torch.int32))
                torch.testing.assert_close(result.counts_by_source,
                                           torch.tensor([[3, 0, 0, 1], [0, 0, 0, 2]], dtype=torch.int32))
                torch.testing.assert_close(result.copies, torch.tensor([[-1], [0]], dtype=torch.int32))
                self.assertEqual((result.receive_capacity, result.published_slots), (3, 0))

    def test_capacity_placement_honors_an_explicit_guest_budget(self) -> None:
        counts = ((4,) * 6 + (0,) * 6,) * 2
        ordinary = select_replica_plan(counts, ExpertReplicaConfig(12, 2, 6), soft_target=24)
        capped = select_replica_plan(counts, ExpertReplicaConfig(12, 2, 2), soft_target=24)
        self.assertEqual((ordinary.replica_slots_used[1], max(ordinary.destination_loads)), (3, 24))
        self.assertEqual((capped.compact_borrowed_width, capped.replica_slots_used[1],
                          max(capped.destination_loads)), (2, 2, 32))
        self.assertEqual(sum(capped.destination_loads), 48)

    def test_wide_counts_reject_overflow_before_allocator(self) -> None:
        """A large INT64 wire count must never wrap to a negative planner value."""
        request = PlacementRequest(((2**31, 0), (0, 0)), 2**31, 2**31, 0, 4, 4, 24, False)
        with self.assertRaisesRegex(ValueError, "INT32 planner limit"):
            HotReplicaPlanner().plan(request)

    def test_invalid_topologies_fail_before_allocator(self) -> None:
        """An invalid rank or expert extent cannot launch an asymmetric plan."""
        request = PlacementRequest(((0, 0), (0, 0)), 0, 0, 0, 4, 4, 24, False)
        for changes in ({"counts_by_source": ((0, 0, 0),) * 2}, {"rank": 2}, {"counts_by_source": ()}):
            with self.subTest(changes=changes):
                with (
                    patch("hyper_multicore.modules.hot_replica.planner.select_replica_plan") as allocate,
                    self.assertRaises(ValueError),
                ):
                    HotReplicaPlanner().plan(replace(request, **changes))
                allocate.assert_not_called()


def _capacity_plan(matrix, config, *, quantum, limit, row_credit=0):
    """Use K64 interval materialization for the MXFP8 precision."""
    return select_replica_plan(
        matrix, config, quantum=quantum, limit=limit, prices=ReplicaCost(phases=((row_credit, row_credit),)),
        materialize=materialize_aligned_moves if quantum == 64 else None,
    )


def test_explicit_limit_balances_bf16_and_k64():
    # One exact pair per hot owner; the two precisions differ only at lowering.
    counts = ((512, 0, 512, 0, 0, 0, 0, 0),) * 4
    config = ExpertReplicaConfig(8, 4, 2)
    for quantum in (1, 64):
        plan = _capacity_plan(counts, config, quantum=quantum, limit=1024)
        plan.validate()
        assert plan.destination_loads == (1024,) * 4
        assert plan.maximum_transfer_degree == 1


def test_shared_search_uses_only_reachable_peer_for_hard_limit():
    config = ExpertReplicaConfig(3, 3, 1)
    counts = ((64, 0, 0),) * 3
    edges = ((True, False, True), (False, True, True), (True, True, True))
    plan = select_replica_plan(counts, config, limit=96, nodes=(0, 0, 1), peer_edges=edges)
    plan.validate()
    assert plan.destination_loads == (96, 0, 96)
    edge, = plan.transfers
    assert (edge.owner_rank, edge.target_rank) == (0, 2)


def test_k64_capacity_allows_multiple_guests_and_a_short_tail():
    config = ExpertReplicaConfig(4, 4, 1)
    matrix = ((129, 0, 0, 0), (128, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
    plan = _capacity_plan(matrix, config, quantum=64, limit=65)
    plan.validate()
    assert sorted(plan.destination_loads) == [64, 64, 64, 65]
    assert len(plan.transfers) == 3


def test_mx_uses_its_minimum_move_only_for_optional_replication():
    config = ExpertReplicaConfig(2, 2, 1)
    counts = ((1024, 0),) * 2
    prices = ReplicaCost(minimum_replica_rows=8192)
    options = dict(quantum=64, prices=prices, materialize=materialize_aligned_moves)
    assert select_replica_plan(counts, config, **options) is None
    plan = select_replica_plan(counts, config, limit=1024, **options)
    assert plan.destination_loads == (1024, 1024)


@pytest.mark.parametrize("quantum,limit", [(1, 5), (64, 226)])
def test_small_hard_cap_can_repair_an_unsuccessful_candidate(quantum, limit):
    rng = np.random.default_rng(3)
    wire = rng.integers(0, 3, (3, 6))
    if quantum == 64:
        wire = wire * 40 + rng.integers(0, 8, wire.shape)
    plan = _capacity_plan(tuple(map(tuple, wire.tolist())), ExpertReplicaConfig(6, 3, 1),
                               quantum=quantum, limit=limit)
    plan.validate()
    assert max(plan.destination_loads) <= limit


def test_capacity_repair_budget_applies_to_a_four_rank_layout():
    wire = ((1, 2, 3, 3, 0, 0, 3, 3),) + ((0,) * 8,) * 3
    plan = _capacity_plan(wire, ExpertReplicaConfig(8, 4, 1), quantum=1, limit=4)
    plan.validate()
    assert max(plan.destination_loads) == 4
    assert max(plan.replica_slots_used) == 1


def test_k64_rebalances_full_blocks_without_moving_short_tails():
    totals = (148, 304, 336, 101, 40, 211, 233, 272, 225, 250, 320, 320, 324, 301, 252, 321)
    counts = (totals,) + ((0,) * 16,) * 7
    plan = _capacity_plan(counts, ExpertReplicaConfig(16, 8, 2), quantum=64, limit=555)
    plan.validate()
    assert max(plan.destination_loads) <= 555
    for edge in plan.transfers:
        start, end = plan.intervals[edge.target_slot + edge.target_rank * plan.config.slots_per_rank]
        assert start % 64 == end % 64 == 0


@pytest.mark.parametrize("quantum", [1, 64])
def test_hard_limit_allows_a_cheaper_plan_below_the_first_feasible_peak(quantum):
    matrix = ((256, 0, 0, 0),) + ((0,) * 4,) * 3
    config = ExpertReplicaConfig(4, 4, 1)
    for credit, expected in [(0, (64, 3)), (80, (128, 1))]:
        plan = _capacity_plan(matrix, config, quantum=quantum, limit=128, row_credit=credit)
        plan.validate()
        assert (max(plan.destination_loads), len(plan.transfers)) == expected
