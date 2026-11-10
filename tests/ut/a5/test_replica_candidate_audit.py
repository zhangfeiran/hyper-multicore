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
"""Check read-only layout promotion and candidate tracing invariants."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from hyper_multicore.replica.capacity import ExpertReplicaConfig
from hyper_multicore.replica.cost import ExpertReplicaCostModel, _QuotaCostState
from hyper_multicore.replica.planner import _build_expert_replica_plan, _materialize
from scripts.audit_replica_candidates import load_fixture, plan_digest, promoted_plan, traced_replay


class TestReplicaCandidateAudit(unittest.TestCase):
    """The audit must not redistribute source quotas or persist trace hooks."""

    def test_promotion_preserves_every_source_column_and_does_not_mutate_input(self) -> None:
        """Only inactive physical slots may be added when lifting a real layout."""
        config = ExpertReplicaConfig(6, 3, 1)
        counts = ((3, 0, 0, 0, 0, 0),) + ((0,) * 6,) * 2
        plan = _materialize(counts, [{0: 1, 1: 0}, {2: 0, 3: 0, 0: 1}, {4: 0, 5: 0, 0: 1}], config)
        before = asdict(plan)
        target = promoted_plan(plan, ExpertReplicaConfig(6, 3, 3))
        target.validate()
        self.assertEqual(asdict(plan), before)
        self.assertEqual(target.logical_counts, plan.logical_counts)
        self.assertEqual(target.destination_loads, plan.destination_loads)
        for source, row in enumerate(plan.dispatch_counts):
            for rank in range(3):
                self.assertEqual(target.dispatch_counts[source][rank * 5:rank * 5 + 3], row[rank * 3:rank * 3 + 3])
                self.assertEqual(target.dispatch_counts[source][rank * 5 + 3:rank * 5 + 5], (0, 0))
        with self.assertRaises(ValueError):
            promoted_plan(plan, ExpertReplicaConfig(6, 2, 3))

    def test_trace_keeps_scores_plans_and_global_methods_unchanged(self) -> None:
        """Trace interception returns the original plan and restores methods after use."""
        model = ExpertReplicaCostModel(calibration_version=2, backend='push', hidden_size=16,
            intermediate_size=8, ep_size=2, transport='shmem_signal_kernel_gradient_bands',
            forward_ms=((0, 0.), (32, 32.)), backward_ms=((0, 0.), (32, 32.)),
            weight_ms=1., gradient_ms=1., _rank_nodes=(0, 1))
        counts, config = ((8, 0), (0, 0)), ExpertReplicaConfig(2, 2, 1)
        options = dict(target_load=4, receive_capacity=8, capacity_limit=8, adaptive=True, _rank_nodes=(0, 1))
        expected = _build_expert_replica_plan(counts, 1, cost_model=model, **options)
        method = _QuotaCostState.score
        plan, trace = traced_replay(counts, config, model, options)
        self.assertEqual(plan, expected)
        self.assertEqual(trace['selected_score_ms'], model.estimate_ms(plan))
        self.assertIs(_QuotaCostState.score, method)
        self.assertTrue(trace['thin_candidates'])
        self.assertTrue(trace['scored_candidates'])

    def test_loader_rejects_a_corrupted_plan_before_replaying_it(self) -> None:
        """A changed physical row cannot be accepted with the saved plan checksum."""
        config = ExpertReplicaConfig(2, 2, 0)
        plan = _build_expert_replica_plan(((2, 0), (0, 2)), 0)
        payload = dict(schema_version=1, plan=asdict(plan), plan_sha256=plan_digest(plan), original_decision={})
        payload['plan']['config'] = asdict(config)
        payload['plan_sha256'] = '0' * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'fixture.json'
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_fixture(path)

    def test_loader_labels_load_only_as_a_different_policy(self) -> None:
        """A missing effective model must not be blamed on cost-model scoring."""
        plan = _build_expert_replica_plan(((2, 0), (0, 2)), 0)
        payload = dict(schema_version=1, plan=asdict(plan), plan_sha256=plan_digest(plan),
                       original_decision={'model': None})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'fixture.json'
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'load-only'):
                load_fixture(path)
