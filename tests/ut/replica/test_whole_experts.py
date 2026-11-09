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
"""CPU invariants for complete-expert execution within bounded replica slots."""

import random
import unittest

from hyper_multicore.replica.recompute import create_recompute_contexts
from hyper_multicore.replica import routing
from hyper_multicore.replica.planner import build_expert_replica_plan
from hyper_multicore.replica.whole_experts import build_whole_expert_plan
from tests.common.mark_utils import arg_mark


class TestWholeExpertReplicas(unittest.TestCase):
    """Keep every logical expert's source rows on one execution rank."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_invalid_capacity_and_admission_are_rejected(self) -> None:
        """Feature: Whole-expert policy validation.

        Description: Supply malformed receive capacities and admission thresholds.
        Expectation: Reject each policy before materializing a placement.
        """
        proposal = build_expert_replica_plan(((4, 0), (4, 0)), 1)
        for options in ({"capacity_limit": -1}, {"capacity_limit": True}, {"minimum_saved_rows": -1},
                        {"minimum_load_reduction": -0.1}, {"minimum_load_reduction": 1.1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                build_whole_expert_plan(proposal, **options)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_checkpoint_replays_whole_plans_and_separates_fragment_policies(self) -> None:
        """Feature: Whole-expert checkpoint replay.

        Description: Replay matching whole policies and change granularity in a second invocation.
        Expectation: Reuse the exact whole plan only for the same policy and logical counts.
        """
        counts = ((20, 20, 0, 0),) * 2
        config, group = routing.ExpertReplicaConfig(4, 2, 1), object()
        for replay_whole in (True, False):
            with self.subTest(replay_whole=replay_whole):
                forward, replay = create_recompute_contexts()
                with forward:
                    first, hit = routing._plan_replica_route(counts, config, group, 0, _whole_experts=True)
                    self.assertFalse(hit)
                with replay:
                    second, hit = routing._plan_replica_route(counts, config, group, 0,
                                                            _whole_experts=replay_whole)
                self.assertEqual(hit, replay_whole)
                if replay_whole:
                    self.assertIs(first, second)
                else:
                    self.assertIsNot(first, second)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_eight_hot_experts_spread_without_splitting_reductions(self) -> None:
        """Feature: Whole-expert replica placement.

        Description: Balance eight hot owner experts over six bounded guest slots.
        Expectation: Every expert remains whole, every source row survives, and each rank gets two hot experts.
        """
        counts = tuple((2048,) * 8 + (0,) * 24 for _ in range(4))
        proposal = build_expert_replica_plan(counts, 2, target_load=30720, receive_capacity=30720,
                                            minimum_replica_rows=1024, capacity_limit=98304,
                                            minimum_load_reduction=0.4)
        whole = build_whole_expert_plan(proposal, capacity_limit=98304, minimum_load_reduction=0.4)
        self.assertEqual(whole.destination_loads, (16384,) * 4)
        self.assertEqual(len(whole.transfers), 6)
        self.assertEqual(whole.logical_counts, counts)
        self.assertEqual(tuple(map(sum, whole.dispatch_counts)), tuple(map(sum, counts)))
        for expert in range(8):
            positive = [(rank, slot) for rank, row in enumerate(whole.slot_to_logical)
                        for slot, logical in enumerate(row)
                        if logical == expert and whole.destination_counts[rank][slot] > 0]
            self.assertEqual(len(positive), 1)
            rank, slot = positive[0]
            self.assertEqual(tuple(row[rank * whole.config.slots_per_rank + slot]
                                   for row in whole.dispatch_counts), (2048,) * 4)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_single_hot_expert_falls_back_without_an_illusory_peak_gain(self) -> None:
        """Feature: Whole-expert admission.

        Description: Moving an indivisible single hot expert cannot reduce the maximum load.
        Expectation: The owner remains selected when the declared minimum improvement is unavailable.
        """
        counts = ((4096,) + (0,) * 7,) * 2
        proposal = build_expert_replica_plan(counts, 1, capacity_limit=8192, minimum_load_reduction=0.4)
        whole = build_whole_expert_plan(proposal, capacity_limit=8192, minimum_load_reduction=0.4)
        self.assertEqual(whole.transfers, ())
        self.assertEqual(whole.destination_loads, (8192, 0))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_indivisible_capacity_failure_is_explicit(self) -> None:
        """Feature: Whole-expert hard capacity.

        Description: An expert larger than the receive bound cannot fit as one reduction.
        Expectation: The planner fails instead of silently splitting or discarding assignments.
        """
        counts = ((4096,) + (0,) * 7,) * 2
        proposal = build_expert_replica_plan(counts, 1, capacity_limit=4096)
        with self.assertRaisesRegex(ValueError, "token splitting is required"):
            build_whole_expert_plan(proposal, capacity_limit=4096)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_random_placements_preserve_counts_and_guest_budgets(self) -> None:
        """Feature: Whole-expert plan invariants.

        Description: Exercise uneven, zero, and multi-owner source distributions.
        Expectation: Peaks never grow, guest budgets hold, and every active expert has one destination.
        """
        generator = random.Random(931)
        for ep in (2, 4, 8):
            for case in range(16):
                with self.subTest(ep=ep, case=case):
                    counts = tuple(tuple(generator.randrange(24) for _ in range(ep * 8)) for _ in range(ep))
                    proposal = build_expert_replica_plan(counts, 2)
                    whole = build_whole_expert_plan(proposal)
                    self.assertEqual(tuple(map(sum, whole.dispatch_counts)), tuple(map(sum, counts)))
                    owner = [sum(sum(row[r * 8:(r + 1) * 8]) for row in counts) for r in range(ep)]
                    self.assertLessEqual(max(whole.destination_loads), max(owner))
                    for rank, row in enumerate(whole.slot_to_logical):
                        self.assertLessEqual(sum(logical >= 0 for logical in row[8:]), 2)
                    for expert in range(ep * 8):
                        self.assertEqual(sum(whole.destination_counts[rank][slot] > 0
                                             for rank, row in enumerate(whole.slot_to_logical)
                                             for slot, logical in enumerate(row) if logical == expert),
                                         int(sum(row[expert] for row in counts) > 0))
