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
"""Node-aware candidate costs without changing bounded replica business APIs."""

from dataclasses import replace
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hyper_multicore.replica.cost import ExpertReplicaCostModel, _QuotaCostState
from hyper_multicore.replica.capacity import ExpertReplicaConfig
from hyper_multicore.replica.planner import _build_expert_replica_plan
from hyper_multicore.replica import planner, topology
from hyper_multicore.modules.mega_moe import module as module_api


class TestReplicaTopology(unittest.TestCase):
    """Prefer cheaper links while preserving capacity and checkpoint ownership."""

    @staticmethod
    def model(**options: object) -> ExpertReplicaCostModel:
        """Use exact linear compute and independent node-link coefficients."""
        values = dict(calibration_version=2, backend='push', hidden_size=16, intermediate_size=8,
                      ep_size=4, transport='p2p', forward_ms=((0, 0), (4096, 4096)),
                      backward_ms=((0, 0), (4096, 4096)), weight_ms=10, gradient_ms=3,
                      forward_repeats=2, backward_weight_ms=0)
        return ExpertReplicaCostModel(**(values | options))

    def test_noncontiguous_rank_mapping_prefers_same_node_on_equal_load(self) -> None:
        """EP rank order need not match machine order or the global rank stride."""
        counts = [[300, 0, 0, 300, 0, 0] + [0] * 6] + [[0] * 12] * 3
        plan = _build_expert_replica_plan(counts, 1, adaptive=True, capacity_limit=600,
                                         cost_model=self.model(), _rank_nodes=(0, 1, 1, 0))
        plan.validate()
        self.assertTrue(plan.transfers)
        self.assertTrue(all((0, 1, 1, 0)[edge.owner_rank] == (0, 1, 1, 0)[edge.target_rank]
                            for edge in plan.transfers))

    def test_calibrated_slower_local_links_do_not_force_local_placement(self) -> None:
        """Measured total cost wins over a fixed same-machine preference."""
        counts = [[300, 0, 0, 300, 0, 0] + [0] * 6] + [[0] * 12] * 3
        model = self.model(intra_node_weight_ms=100, inter_node_weight_ms=1,
                           intra_node_gradient_ms=30, inter_node_gradient_ms=1)
        nodes = (0, 1, 1, 0)
        plan = _build_expert_replica_plan(counts, 1, adaptive=True, capacity_limit=600,
                                         cost_model=model, _rank_nodes=nodes)
        self.assertTrue(plan.transfers)
        self.assertTrue(all(nodes[edge.owner_rank] != nodes[edge.target_rank] for edge in plan.transfers))
        plan.validate()

    def test_hot_node_can_spill_to_other_nodes_to_satisfy_capacity(self) -> None:
        """Two overloaded local ranks cannot reserve their copies only on that node."""
        counts = [[300, 0, 0, 300, 0, 0] + [0] * 6] + [[0] * 12] * 3
        nodes = (0, 0, 1, 1)
        plan = _build_expert_replica_plan(counts, 3, adaptive=True, capacity_limit=150,
                                         cost_model=self.model(), _rank_nodes=nodes)
        plan.validate()
        self.assertLessEqual(max(plan.destination_loads), 150)
        self.assertTrue(any(nodes[edge.owner_rank] != nodes[edge.target_rank] for edge in plan.transfers))

    def test_missing_costs_keep_peak_first_then_reduce_cross_node_edges(self) -> None:
        """Uncalibrated locality may not increase the chosen receive peak."""
        counts = [[300, 0, 0, 300, 0, 0] + [0] * 6] + [[0] * 12] * 3
        original = _build_expert_replica_plan(counts, 1, adaptive=True, capacity_limit=600)
        local = _build_expert_replica_plan(counts, 1, adaptive=True, capacity_limit=600,
                                          _rank_nodes=(0, 1, 1, 0))
        self.assertEqual(max(local.destination_loads), max(original.destination_loads))
        self.assertTrue(all((0, 1, 1, 0)[edge.owner_rank] == (0, 1, 1, 0)[edge.target_rank]
                            for edge in local.transfers))
        local.validate()

    def test_uniform_or_unknown_topology_preserves_existing_plan(self) -> None:
        """The original selector remains exact when no same-node alternative exists."""
        counts = [[300, 0, 0, 300, 0, 0] + [0] * 6] + [[0] * 12] * 3
        original = _build_expert_replica_plan(counts, 1, adaptive=True, cost_model=self.model())
        for nodes in ((), (0, 0, 0, 0), (0, 1, 2, 3)):
            with self.subTest(nodes=nodes):
                self.assertEqual(original, _build_expert_replica_plan(counts, 1, adaptive=True,
                    cost_model=self.model(), _rank_nodes=nodes))

    def test_link_cost_validation_and_backward_residency(self) -> None:
        """A zero backward copy remains zero with differentiated forward link costs."""
        model = self.model(intra_node_weight_ms=2, inter_node_weight_ms=20,
                           intra_node_gradient_ms=1, inter_node_gradient_ms=7, _rank_nodes=(0, 1, 1, 0))
        self.assertEqual(model.transfer_ms(0, 3), (2, 0, 1))
        self.assertEqual(model.transfer_ms(0, 2), (20, 0, 7))
        for value in (-1, True, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.model(intra_node_weight_ms=value)
        for nodes in ((0, 1), (0, 1, False, 0), (0, -1, 0, 1)):
            with self.subTest(nodes=nodes), self.assertRaises(ValueError):
                self.model(_rank_nodes=nodes)

    def test_trial_cost_removes_the_correct_link_class(self) -> None:
        """Dropping a local edge must retain the owner's remaining inter-node charge."""
        model = self.model(intra_node_weight_ms=2, inter_node_weight_ms=20,
                           intra_node_gradient_ms=1, inter_node_gradient_ms=7, _rank_nodes=(0, 1, 1, 0))
        config = ExpertReplicaConfig(8, 4, 1)
        counts = ((12, 0, 0, 0, 0, 0, 0, 0),) + ((0,) * 8,) * 3
        copies = [{0: 6, 1: 0}, {2: 0, 3: 0, 0: 3}, {4: 0, 5: 0}, {6: 0, 7: 0, 0: 3}]
        state = _QuotaCostState(model, copies, counts, config)
        copies[0][0] += 3
        copies[3][0] = 0
        unchanged = tuple(max(state.costs[rank][phase] for rank in (1, 2)) for phase in (0, 1))
        trial = state.trial_score(0, 3, 0, unchanged)
        rebuilt = _QuotaCostState(model, copies, counts, config).score()
        self.assertEqual(trial, rebuilt)
        state.accept(0, 3, 0)
        self.assertEqual(state.score(), rebuilt)

    def test_startup_manifest_collects_node_identity_without_new_hot_collective(self) -> None:
        """Append identity to the existing resource consensus and preserve its configuration check."""
        resource = module_api._MegaMoeExecutionResources.__new__(module_api._MegaMoeExecutionResources)
        resource.spec = SimpleNamespace(ep_size=4, ep_group=object())

        def gather(output: list, value: tuple, **_kwargs: object) -> None:
            """Supply interleaved machine identities in EP rank order."""
            output[:] = [value[:-1] + (node,) for node in ('host-a', 'host-b', 'host-b', 'host-a')]

        with patch.object(module_api.dist, 'all_gather_object', side_effect=gather) as collective:
            resource._validate_distributed_configuration(({},))
        self.assertEqual(resource.rank_nodes, (0, 1, 1, 0))
        self.assertEqual(collective.call_count, 1)

    def test_final_refinement_cannot_regress_global_candidate_cost_or_capacity(self) -> None:
        """Compare final quotas, since a lower initial score need not survive pruning."""
        generator = random.Random(9713)
        nodes = (0, 1, 1, 0)
        model = self.model(intra_node_weight_ms=1, inter_node_weight_ms=30,
                           intra_node_gradient_ms=1, inter_node_gradient_ms=10)
        bound_model = replace(model, _rank_nodes=nodes)
        for budget in (1, 2, 3):
            for iteration in range(8):
                counts = [[generator.randrange(60) for _ in range(12)] for _ in range(4)]
                counts[0][iteration % 6] += 300
                options = dict(adaptive=True, cost_model=model, _rank_nodes=nodes, capacity_limit=4000,
                               minimum_replica_rows=16)
                with patch.object(planner, '_locality_copies', side_effect=lambda *_args: _args[0]):
                    control = _build_expert_replica_plan(counts, budget, **options)
                candidate = _build_expert_replica_plan(counts, budget, **options)
                with self.subTest(budget=budget, iteration=iteration):
                    candidate.validate()
                    self.assertLessEqual(bound_model.estimate_ms(candidate), bound_model.estimate_ms(control) + 1e-9)
                    self.assertLessEqual(max(candidate.destination_loads), max(control.destination_loads))

    def test_owner_only_plans_skip_extra_topology_search(self) -> None:
        """Node coefficients must not add CPU searches when the main policy has no replicas."""
        counts = [[10] * 12] * 4
        model = self.model(intra_node_weight_ms=1, inter_node_weight_ms=100)
        with patch.object(planner, '_locality_copies', side_effect=AssertionError('Unexpected local search')):
            plan = _build_expert_replica_plan(counts, 3, adaptive=True, cost_model=model, _rank_nodes=(0, 1, 1, 0))
        self.assertFalse(plan.transfers)

    def test_kernel_identity_and_hostname_fallback_do_not_assume_sixteen_ranks(self) -> None:
        """Containers sharing a kernel can have distinct hostnames without splitting the node."""
        with patch.object(topology.Path, 'read_text', return_value='boot-a\n'):
            self.assertEqual(topology._local_node_identity(), 'boot-a')
        with (patch.object(topology.Path, 'read_text', side_effect=OSError('Unavailable')),
              patch.object(topology.socket, 'gethostname', return_value='host-a')):
            self.assertEqual(topology._local_node_identity(), 'host-a')
        self.assertEqual(topology._node_ids(('a', 'b', 'a', 'b', 'c')), (0, 1, 0, 1, 2))



if __name__ == '__main__':
    unittest.main()
