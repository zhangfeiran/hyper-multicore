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
"""Prove the integer invariants used to accept uncalibrated locality proposals."""

from collections import Counter
from copy import deepcopy
import random
import unittest

from hyper_multicore.replica.capacity import ExpertReplicaConfig
from hyper_multicore.replica.planner import _locality_copies
from tests.common.mark_utils import arg_mark


def _objective(copies: list[dict[int, int]], config: ExpertReplicaConfig,
                nodes: tuple[int, ...]) -> tuple[int, int]:
    """Return the original uncalibrated receive-peak and cross-node-edge key."""
    remote = sum(bool(rows) and expert // config.home_experts != rank
                 and nodes[expert // config.home_experts] != nodes[rank]
                 for rank, row in enumerate(copies) for expert, rows in row.items())
    return max(sum(row.values()) for row in copies), remote


def _totals(copies: list[dict[int, int]]) -> Counter:
    """Count every logical expert independently of insertion order and physical owner."""
    result = Counter()
    for row in copies:
        result.update(row)
    return result


class TestReplicaLocalityInvariants(unittest.TestCase):
    """Keep the monotonicity proof executable before removing redundant objective scans."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_move_merge_swap_and_noop_preserve_quotas_and_objective(self) -> None:
        """Feature: Integer locality admission.
        Description: Exercise new local guests, guest merges, two-edge swaps and a saturated no-op.
        Expectation: Changed proposals strictly improve the original key without mutating their input.
        """
        config, nodes = ExpertReplicaConfig(4, 4, 1), (0, 0, 1, 1)
        cases = (
            ([{0: 14}, {1: 0}, {2: 10, 0: 4}, {3: 0}], [{0: 14}, {1: 0, 0: 4}, {2: 10}, {3: 0}]),
            ([{0: 14}, {1: 0, 0: 3}, {2: 10, 0: 4}, {3: 0}],
             [{0: 14}, {1: 0, 0: 7}, {2: 10}, {3: 0}]),
            ([{0: 10}, {1: 6, 3: 4}, {2: 6, 0: 4}, {3: 10}],
             [{0: 10}, {1: 6, 0: 4}, {2: 6, 3: 4}, {3: 10}]),
            ([{0: 10}, {1: 10}, {2: 6, 0: 4}, {3: 10}],
             [{0: 10}, {1: 10}, {2: 6, 0: 4}, {3: 10}]),
        )
        for selected, expected in cases:
            original = deepcopy(selected)
            with self.subTest(selected=selected):
                proposal = _locality_copies(selected, config, nodes)
                self.assertEqual(proposal, expected)
                self.assertEqual(selected, original)
                self.assertEqual(_totals(proposal), _totals(original))
                if proposal != original:
                    self.assertLess(_objective(proposal, config, nodes), _objective(original, config, nodes))
                else:
                    self.assertIs(proposal, selected)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_seeded_placements_preserve_strict_order_and_slot_cap(self) -> None:
        """Feature: Locality proof across topology and guest budgets.
        Description: Explore sparse owner quotas and interleaved physical nodes with fixed random inputs.
        Expectation: Every changed proposal conserves quotas, fits B and strictly wins the unchanged integer key.
        """
        generator = random.Random(20261105)
        changed = 0
        for nodes in ((0, 0, 1, 1), (101, 108, 101, 108)):
            for budget in (1, 2, 3):
                config = ExpertReplicaConfig(12, 4, budget)
                for iteration in range(80):
                    selected = [{expert: generator.randrange(20) for expert in range(rank * 3, (rank + 1) * 3)}
                                for rank in range(4)]
                    for rank, row in enumerate(selected):
                        foreign = [expert for expert in range(12) if expert // 3 != rank]
                        for expert in generator.sample(foreign, generator.randrange(budget + 1)):
                            row[expert] = generator.randrange(1, 20)
                    original = deepcopy(selected)
                    proposal = _locality_copies(selected, config, nodes)
                    with self.subTest(nodes=nodes, budget=budget, iteration=iteration):
                        self.assertEqual(selected, original)
                        self.assertEqual(_totals(proposal), _totals(original))
                        self.assertLessEqual(_objective(proposal, config, nodes)[0],
                                             _objective(original, config, nodes)[0])
                        self.assertTrue(all(sum(expert // 3 != rank and bool(rows)
                                                for expert, rows in row.items()) <= budget
                                            for rank, row in enumerate(proposal)))
                        if proposal != original:
                            changed += 1
                            self.assertLess(_objective(proposal, config, nodes), _objective(original, config, nodes))
        self.assertGreater(changed, 100)
