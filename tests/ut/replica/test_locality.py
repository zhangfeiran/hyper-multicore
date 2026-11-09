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
"""Validate source/node locality independently from planning objectives."""

import unittest

from hyper_multicore.replica import ExpertExecutionPlan, ExpertReplicaConfig
from hyper_multicore.replica._dispatch import _materialize
from hyper_multicore.replica._retained import _accept_retained_proposal, _rebalance_retained_copies


def _traffic(plan: ExpertExecutionPlan, nodes: tuple[int, ...]) -> tuple[int, int]:
    """Count actual off-rank and off-node routed rows from physical dispatch."""
    width = plan.config.slots_per_rank
    remote = sum(rows for source, row in enumerate(plan.dispatch_counts)
                 for physical, rows in enumerate(row) if source != physical // width)
    cross = sum(rows for source, row in enumerate(plan.dispatch_counts)
                for physical, rows in enumerate(row) if nodes[source] != nodes[physical // width])
    return remote, cross


class TestReplicaLocality(unittest.TestCase):
    """Preserve exact traffic and critical-fragment guards from the source implementation."""

    def test_interleaved_nodes_preserve_local_source_intersections(self) -> None:
        """Actual membership, rather than contiguous rank assumptions, removes node traffic."""
        config = ExpertReplicaConfig(4, 4, 1)
        counts = ((1, 0, 0, 0), (1, 0, 0, 0), (3, 0, 0, 0), (3, 0, 0, 0))
        copies = [{0: 4}, {1: 0, 0: 4}, {2: 0}, {3: 0}]
        nodes = (0, 1, 1, 0)
        ordinary = _materialize(counts, copies, config)
        local = _materialize(counts, copies, config, rank_nodes=nodes)
        local.validate()
        self.assertEqual(_traffic(ordinary, nodes), (6, 6))
        self.assertEqual(_traffic(local, nodes), (6, 0))
        self.assertEqual(local.destination_counts, ordinary.destination_counts)
        self.assertEqual(local.transfers, ordinary.transfers)

    def test_node_ingress_guard_rejects_rank_safe_refill(self) -> None:
        """A compute improvement cannot spend uncredited node ingress."""
        copies = [{0: 8}, {1: 2, 0: 14}, {2: 5, 0: 8}, {3: 5, 0: 7}]
        proposal = [{0: 9}, {1: 2, 0: 10}, {2: 5, 0: 8}, {3: 5, 0: 10}]
        counts = ((8, 0, 1, 1), (13, 0, 1, 3), (4, 1, 1, 1), (12, 1, 2, 0))
        self.assertIs(_accept_retained_proposal(copies, proposal, counts, ()), proposal)
        self.assertIs(_accept_retained_proposal(copies, proposal, counts, (0, 1, 0, 2)), copies)

    def test_critical_guest_refill_never_adds_a_matrix(self) -> None:
        """An already hot guest may return work to the owner's existing matrices."""
        copies = [{0: 1}, {1: 0, 0: 11}, {2: 0, 0: 2}]
        counts = ((4, 0, 0), (4, 0, 0), (6, 0, 0))
        changed = _rebalance_retained_copies(copies, counts, 1, (0, 0, 1))
        self.assertEqual([row[0] for row in changed], [5, 5, 4])
        self.assertEqual([tuple(row) for row in changed], [tuple(row) for row in copies])
        self.assertEqual([row[0] for row in copies], [1, 11, 2])

    def test_refill_keeps_source_local_rows(self) -> None:
        """Keep the old plan if all hot rows are already local to their execution ranks."""
        copies = [{0: 1}, {1: 0, 0: 11}, {2: 0, 0: 2}]
        counts = ((1, 0, 0), (11, 0, 0), (2, 0, 0))
        self.assertIs(_rebalance_retained_copies(copies, counts, 1, (0, 0, 1)), copies)
