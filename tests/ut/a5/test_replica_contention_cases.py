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
"""Validate contention topology, conserved quotas and untrimmed role statistics."""

from dataclasses import replace
import unittest

from tests.ut.a5.replica_contention_cases import case_plan, make_case, summarize_ranks


class TestReplicaContentionCases(unittest.TestCase):
    """CPU checks need no process group or device runtime."""

    def test_complete_ep64_matrix_and_noncontiguous_nodes(self) -> None:
        """All 22 meaningful cases conserve routes and respect the B1 guest cap."""
        for nodes in (tuple(rank // 16 for rank in range(64)), tuple(rank % 4 for rank in range(64))):
            total = 0
            for owners in (1, 4):
                for fanout in (1, 2, 4, 8):
                    for placement in ('intra', 'inter', 'mixed'):
                        if placement == 'mixed' and fanout == 1:
                            continue
                        with self.subTest(owners=owners, fanout=fanout, placement=placement, nodes=nodes[:8]):
                            case = make_case(nodes, fanout, owners, placement)
                            plan = case_plan(case, 64, 6)
                            self.assertEqual(sum(map(sum, plan.logical_counts)), owners * fanout)
                            self.assertEqual(sum(plan.replica_slots_used), owners * fanout)
                            self.assertLessEqual(max(plan.replica_slots_used), 1)
                            self.assertEqual({(edge.owner_rank, edge.target_rank) for edge in plan.transfers},
                                             set(case.edges))
                            self.assertEqual([case.role(rank) for rank in case.owners], ['owner'] * owners)
                            for owner in case.owners:
                                local = sum(nodes[source] == nodes[target] for source, target in case.edges
                                            if source == owner)
                                expected = {'intra': fanout, 'inter': 0, 'mixed': (fanout + 1) // 2}[placement]
                                self.assertEqual(local, expected)
                            total += 1
            self.assertEqual(total, 22)

    def test_local_smoke_shapes(self) -> None:
        """The local experiment exercises fanout and concurrent owners separately."""
        for size, owners, fanout in ((2, 1, 1), (4, 1, 2), (4, 2, 1)):
            case = make_case((0,) * size, fanout, owners, 'intra')
            self.assertEqual(len(case_plan(case, size, 6).transfers), owners * fanout)

    def test_reject_unavailable_or_ambiguous_placement(self) -> None:
        """Unsupported combinations fail explicitly instead of silently changing topology."""
        for fanout, owners, placement in ((1, 1, 'mixed'), (1, 1, 'inter'), (2, 2, 'intra'),
                                           (0, 1, 'intra'), (1, 1, 'unknown')):
            with self.subTest(fanout=fanout, owners=owners, placement=placement):
                with self.assertRaises(ValueError):
                    make_case((0, 0, 0, 0), fanout, owners, placement)
        case = make_case((0, 0, 0, 0), 2, 1, 'intra')
        with self.assertRaises(ValueError):
            case_plan(replace(case, edges=((0, 1), (0, 1))), 4, 6)

    @staticmethod
    def records() -> list[dict]:
        """Include an idle critical rank and maxima that move between iterations."""
        values = ((1, 10, 2), (5, 1, 3), (7, 2, 4))
        return [dict(ep_rank=rank, global_rank=rank + 64, role=role, checks_passed=True,
                     samples=[dict(weight_ms=value, physical_gradient_ms=value,
                                   weight_wall_ms=value, gradient_wall_ms=value) for value in values[rank]])
                for rank, role in enumerate(('owner', 'guest', 'idle'))]

    def test_rank_max_before_median_and_idle_attribution(self) -> None:
        """An idle rank is retained and no slow sample is removed."""
        result = summarize_ranks(self.records())
        medians = result['median_of_iteration_rank_max_ms']
        self.assertEqual(medians['all']['weight_ms'], 7)
        self.assertEqual(medians['active']['weight_ms'], 5)
        self.assertEqual(medians['owner']['weight_ms'], 2)
        self.assertEqual(result['samples'][0]['critical_global_ranks']['weight_ms'], [66])
        self.assertFalse(result['trimming'])

    def test_reject_missing_rank_sample_and_nonfinite_time(self) -> None:
        """Incomplete or invalid evidence cannot produce a successful summary."""
        records = self.records()
        with self.assertRaises(ValueError):
            summarize_ranks(records[1:])
        records[1]['samples'].pop()
        with self.assertRaises(ValueError):
            summarize_ranks(records)
        records = self.records()
        records[0]['samples'][0]['weight_ms'] = float('nan')
        with self.assertRaises(ValueError):
            summarize_ranks(records)
