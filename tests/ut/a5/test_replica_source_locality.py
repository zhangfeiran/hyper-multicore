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
"""Quota-preserving token locality, independent of expert placement costs."""

import random
import unittest
from unittest.mock import patch

from hyper_multicore.replica.capacity import ExpertReplicaConfig
from hyper_multicore.replica import planner


def traffic(plan: object, nodes: tuple[int, ...]) -> tuple[int, int]:
    """Return rows crossing ranks and nodes for one dispatch direction."""
    width = plan.config.slots_per_rank
    remote = sum(count for source, row in enumerate(plan.dispatch_counts)
                 for physical, count in enumerate(row) if source != physical // width)
    cross_node = sum(count for source, row in enumerate(plan.dispatch_counts)
                     for physical, count in enumerate(row) if nodes[source] != nodes[physical // width])
    return remote, cross_node


class TestReplicaSourceLocality(unittest.TestCase):
    """Minimize fabric traffic without changing quotas, slots or receive peaks."""

    def test_interleaved_sources_avoid_cross_node_exchange(self) -> None:
        """Rank-order filling can exchange two entire local source segments."""
        config = ExpertReplicaConfig(4, 4, 1)
        counts = ((1, 0, 0, 0), (1, 0, 0, 0), (3, 0, 0, 0), (3, 0, 0, 0))
        copies = [{0: 4}, {1: 0, 0: 4}, {2: 0}, {3: 0}]
        original = planner._materialize(counts, copies, config)
        local = planner._materialize(counts, copies, config, rank_nodes=(0, 1, 1, 0))
        local.validate()
        self.assertEqual(traffic(original, (0, 1, 1, 0)), (6, 6))
        self.assertEqual(traffic(local, (0, 1, 1, 0)), (6, 0))
        self.assertEqual(local.destination_counts, original.destination_counts)
        self.assertEqual(local.transfers, original.transfers)

    def test_random_quotas_reach_rank_and_node_lower_bounds(self) -> None:
        """Saturating nested locality classes attains both conservation bounds."""
        generator = random.Random(29104)
        for iteration in range(100):
            size, experts = 6, 12
            config = ExpertReplicaConfig(experts, size, 2)
            nodes = (7, 2, 7, 9, 2, 9)
            counts = tuple(tuple(generator.randrange(5) for _ in range(experts)) for _ in range(size))
            copies = [{expert: sum(row[expert] for row in counts)
                       for expert in range(rank * 2, rank * 2 + 2)} for rank in range(size)]
            for rank in range(size):
                for expert in generator.sample([e for e in range(experts) if e // 2 != rank], 2):
                    owner = expert // 2
                    moved = generator.randrange(copies[owner][expert] + 1)
                    copies[owner][expert] -= moved
                    copies[rank][expert] = moved
            original = planner._materialize(counts, copies, config)
            plan = planner._materialize(counts, copies, config, rank_nodes=nodes)
            rank_bound = sum(max(counts[r][e] - copies[r].get(e, 0), 0)
                             for r in range(size) for e in range(experts))
            node_bound = sum(max(sum(counts[r][e] - copies[r].get(e, 0)
                                     for r in range(size) if nodes[r] == node), 0)
                             for node in set(nodes) for e in range(experts))
            with self.subTest(iteration=iteration):
                plan.validate()
                self.assertEqual(traffic(plan, nodes), (rank_bound, node_bound))
                self.assertLessEqual(traffic(plan, nodes)[1], traffic(original, nodes)[1])
                self.assertEqual(plan.destination_counts, original.destination_counts)
                self.assertEqual(plan.slot_to_logical, original.slot_to_logical)
                self.assertEqual(plan, planner._materialize(counts, copies, config, rank_nodes=nodes))

    def test_public_planner_threads_topology_into_final_materialization(self) -> None:
        """The runtime entry point must use locality after selecting the layout."""
        counts = [[100, 100, 0, 0]] * 4
        nodes = (0, 1, 1, 0)
        with patch.object(planner, '_materialize', wraps=planner._materialize) as materialize:
            plan = planner._build_expert_replica_plan(counts, 1, adaptive=True, _rank_nodes=nodes)
        plan.validate()
        self.assertTrue(plan.transfers)
        self.assertEqual(materialize.call_args.kwargs.get('rank_nodes'), nodes)

    def test_degenerate_topology_keeps_dispatch_identical(self) -> None:
        """Unknown, one-node and one-rank-per-node topologies need no extra pass."""
        config = ExpertReplicaConfig(4, 4, 1)
        counts = ((1, 0, 0, 0), (1, 0, 0, 0), (3, 0, 0, 0), (3, 0, 0, 0))
        copies = [{0: 4}, {1: 0, 0: 4}, {2: 0}, {3: 0}]
        original = planner._materialize(counts, copies, config)
        for nodes in ((), (0, 0, 0, 0), (0, 1, 2, 3)):
            with self.subTest(nodes=nodes):
                self.assertEqual(original, planner._materialize(counts, copies, config, rank_nodes=nodes))

    def test_exhaustive_small_transport_including_empty_sources(self) -> None:
        """Check every supply/quota composition up to five rows on three ranks."""
        config = ExpertReplicaConfig(3, 3, 1)
        nodes = (4, 9, 4)
        for total in range(6):
            compositions = [(first, second, total - first - second)
                            for first in range(total + 1) for second in range(total - first + 1)]
            for supply in compositions:
                counts = tuple((value, 0, 0) for value in supply)
                for demand in compositions:
                    copies = [{0: demand[0]}, {1: 0, 0: demand[1]}, {2: 0, 0: demand[2]}]
                    plan = planner._materialize(counts, copies, config, rank_nodes=nodes)
                    remote = sum(max(supply[r] - demand[r], 0) for r in range(3))
                    cross = abs(supply[1] - demand[1])
                    with self.subTest(supply=supply, demand=demand):
                        plan.validate()
                        self.assertEqual(traffic(plan, nodes), (remote, cross))

    def test_home_only_skips_node_matching(self) -> None:
        """Topology metadata must not add matching work to the owner-only path."""
        counts = ((1, 0, 0, 0),) * 4
        config = ExpertReplicaConfig(4, 4, 1)
        copies = [{0: 4}, {1: 0}, {2: 0}, {3: 0}]
        with patch.object(planner, '_fill_node_quotas', side_effect=AssertionError('Unexpected node matching')):
            plan = planner._materialize(counts, copies, config, rank_nodes=(0, 1, 1, 0))
        self.assertFalse(plan.transfers)


if __name__ == '__main__':
    unittest.main()
