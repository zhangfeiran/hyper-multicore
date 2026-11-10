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
"""Check device critical paths against explicit dependency and resource examples."""

from dataclasses import asdict, replace
import json
import unittest
from unittest.mock import patch

from hyper_multicore.replica import policy
from hyper_multicore.replica.capacity import ExpertReplicaConfig
from hyper_multicore.replica.cost import ExpertReplicaCostModel
from hyper_multicore.replica.critical_path import (
    DeviceCriticalPathCalibration, execution_fingerprint,
)
from hyper_multicore.replica.planner import _build_expert_replica_plan, _materialize


class TestReplicaCriticalPath(unittest.TestCase):
    """Use hand-computable workloads rather than fitting expected predictions."""

    @staticmethod
    def calibration(size: int = 2, **changes: object) -> DeviceCriticalPathCalibration:
        """Create explicitly synthetic physical-service data for CPU contracts."""
        values = dict(execution_signature='a' * 64, home_forward_ms=((0, 0), (128, 128)),
                      home_backward_ms=((0, 0), (128, 128)), intra_node_gradient_ms=3.,
                      inter_node_gradient_ms=3., communication_domains=tuple(range(size)),
                      planner_host_ms=5., rejection_host_ms=.5, gradient_ack_ms=1.)
        return DeviceCriticalPathCalibration(**(values | changes))

    @classmethod
    def model(cls, size: int = 2, **changes: object) -> ExpertReplicaCostModel:
        """Deliberately distinguish physical gradient service from legacy residuals."""
        values = dict(calibration_version=2, backend='push', hidden_size=16, intermediate_size=8,
                      ep_size=size, transport='shmem_signal_kernel_gradient_bands',
                      forward_ms=((0, 0), (128, 128)), backward_ms=((0, 0), (128, 128)),
                      weight_ms=2., gradient_ms=999., backward_weight_ms=0., forward_repeats=2,
                      _rank_nodes=tuple(range(size)), critical_path=cls.calibration(size))
        return ExpertReplicaCostModel(**(values | changes))

    @staticmethod
    def example() -> tuple:
        """Split eight exact logical rows evenly between owner and guest."""
        counts = ((8, 0, 0, 0), (0, 0, 0, 0))
        config = ExpertReplicaConfig(4, 2, 1)
        copies = [{0: 4, 1: 0}, {2: 0, 3: 0, 0: 4}]
        return counts, config, copies

    def test_full_fence_forward_recompute_and_owner_ack_tail(self) -> None:
        """Two forward copies, one physical gradient and one ACK have explicit paths."""
        counts, config, copies = self.example()
        model = self.model()
        estimate = model.critical_path.estimate(model, copies, counts, config, trace=True)
        self.assertEqual((estimate.forward_ms, estimate.recompute_ms, estimate.backward_ms), (6., 6., 8.))
        self.assertEqual(estimate.owner_ready_ms, (7., 8.))
        self.assertEqual(estimate.total_ms, 25.)
        plan = _materialize(counts, copies, config)
        self.assertEqual(model.estimate_ms(plan), estimate.total_ms)
        self.assertFalse(any(task[0].startswith('backward:weight:') for task in estimate.tasks))

    def test_measured_overlap_changes_dependencies_and_not_matrix_bytes(self) -> None:
        """Home work can hide copies only in a profile certifying that overlap."""
        counts, config, _ = self.example()
        copies = [{0: 6, 1: 0}, {2: 0, 3: 0, 0: 2}]
        model = self.model()
        serial = model.critical_path.estimate(model, copies, counts, config, trace=True)
        overlap = replace(model.critical_path, overlap_verified=True)
        concurrent = overlap.estimate(model, copies, counts, config, trace=True)
        self.assertEqual(serial.forward_ms, 8.)
        self.assertEqual(concurrent.forward_ms, 6.)
        home = next(t for t in serial.tasks if t[0]=='forward:home:0')
        self.assertEqual(home[1], 2.)

    def test_gradient_returns_keep_same_expert_peer_order(self) -> None:
        """Four workers cannot reorder two contributions to the same owner element."""
        counts = ((8, 0, 0), (0, 0, 0), (0, 0, 0))
        copies = [{0: 2}, {1: 0, 0: 3}, {2: 0, 0: 3}]
        model = self.model(3, weight_ms=0., critical_path=self.calibration(
            3, gradient_workers=4, gradient_ack_ms=0., planner_host_ms=0.))
        estimate = model.critical_path.estimate(model, copies, counts, ExpertReplicaConfig(3, 3, 1), trace=True)
        jobs = [task for task in estimate.tasks if task[0].startswith('backward:gradient:')]
        self.assertEqual([task[0] for task in jobs], ['backward:gradient:0:1:0', 'backward:gradient:0:2:0'])
        self.assertEqual([(task[1], task[2]) for task in jobs], [(3., 6.), (6., 9.)])
        self.assertEqual(estimate.backward_ms, 9.)

    def test_shared_domain_serializes_otherwise_independent_owners(self) -> None:
        """Shared endpoint contention cannot be priced as independent rank maxima."""
        counts = ((4, 4, 4, 4),) + ((0, 0, 0, 0),) * 3
        copies = [{0: 0, 1: 4}, {1: 0, 0: 4}, {2: 0, 3: 4}, {3: 0, 2: 4}]
        model = self.model(4, weight_ms=0., _rank_nodes=(0,) * 4,
                           critical_path=self.calibration(4, gradient_workers=4, gradient_ack_ms=0.))
        separate = model.critical_path.estimate(model, copies, counts, ExpertReplicaConfig(4, 4, 1))
        shared = replace(model.critical_path, communication_domains=(0,) * 4)
        contended = shared.estimate(model, copies, counts, ExpertReplicaConfig(4, 4, 1))
        self.assertEqual(separate.backward_ms, 10.)
        self.assertEqual(contended.backward_ms, 16.)

    def test_weight_fanout_uses_measured_batch_service(self) -> None:
        """A measured two-peer copy completes both guests after the shared batch."""
        counts = ((8, 0, 0), (0, 0, 0), (0, 0, 0))
        copies = [{0: 2}, {1: 0, 0: 3}, {2: 0, 0: 3}]
        model = self.model(3, weight_fanout_ms=((0, 0., 0.), (1, 2., 2.), (2, 3., 3.)))
        estimate = model.critical_path.estimate(model, copies, counts, ExpertReplicaConfig(3, 3, 1), trace=True)
        fanout = [task for task in estimate.tasks if task[0].startswith('forward:fanout:')]
        self.assertEqual(len(fanout), 1)
        self.assertEqual(fanout[0][1:3], (0., 3.))
        self.assertEqual(estimate.forward_ms, 6.)

    def test_final_ack_waits_for_every_owned_peer(self) -> None:
        """The first completed peer cannot release its guest pool before the owner join."""
        counts = ((8, 0, 0), (0, 0, 0), (0, 0, 0))
        copies = [{0: 2}, {1: 0, 0: 3}, {2: 0, 0: 3}]
        model = self.model(3, weight_ms=0.)
        estimate = model.critical_path.estimate(model, copies, counts, ExpertReplicaConfig(3, 3, 1), trace=True)
        ack = [task for task in estimate.tasks if task[0].startswith('backward:ack:')]
        self.assertEqual([(task[1],task[2]) for task in ack], [(9.,10.),(9.,10.)])
        self.assertEqual(estimate.owner_ready_ms, (9.,10.,10.))

    def test_true_owner_curve_and_rejection_cost_are_separate(self) -> None:
        """Rejection overhead is visible and cannot be counted as replica benefit."""
        counts, config, _ = self.example()
        copies = [{0: 8, 1: 0}, {2: 0, 3: 0}]
        model = self.model()
        true = model.critical_path.estimate(model, copies, counts, config, baseline=True)
        rejected = model.critical_path.estimate(model, copies, counts, config)
        self.assertEqual(true.total_ms, 24.)
        self.assertEqual(rejected.total_ms, 24.5)
        model = replace(model, critical_path=replace(model.critical_path, rejection_host_ms=5.))
        plan = _build_expert_replica_plan(counts, 1, target_load=4, capacity_limit=8, receive_capacity=8,
                                         adaptive=True, cost_model=model, _rank_nodes=(0, 1))
        self.assertFalse(plan.transfers)

    def test_new_scoring_reaches_real_planner_and_changes_admission(self) -> None:
        """A physical-service DAG can accept a plan rejected by the residual heuristic."""
        counts, _, _ = self.example()
        calibration = self.calibration(home_forward_ms=((0, 0), (128, 256)),
            home_backward_ms=((0, 0), (128, 256)), intra_node_gradient_ms=.01,
            inter_node_gradient_ms=.01, planner_host_ms=0., gradient_ack_ms=0.)
        model = self.model(weight_ms=0., critical_path=calibration)
        options = dict(target_load=4, capacity_limit=8, receive_capacity=8, adaptive=True, _rank_nodes=(0, 1))
        legacy = _build_expert_replica_plan(counts, 1, cost_model=replace(model, critical_path=None), **options)
        with patch('hyper_multicore.replica.planner.refine_replica_quotas',
                   side_effect=AssertionError('Legacy two-rank score is invalid for resource queues')):
            selected = _build_expert_replica_plan(counts, 1, cost_model=model, **options)
        self.assertFalse(legacy.transfers)
        self.assertTrue(selected.transfers)
        selected.validate()
        self.assertLess(model.estimate_ms(selected), model.estimate_ms(legacy))

    def test_capacity_required_copies_do_not_need_a_feasible_baseline(self) -> None:
        """An expensive replica is retained when the explicit hard capacity requires it."""
        counts, _, _ = self.example()
        plan = _build_expert_replica_plan(counts, 1, target_load=4, capacity_limit=4, receive_capacity=4,
                                         adaptive=True, cost_model=self.model(), _rank_nodes=(0, 1))
        plan.validate()
        self.assertTrue(plan.transfers)
        self.assertLessEqual(max(plan.destination_loads), 4)

    def test_early_ready_requires_explicit_verified_overlap(self) -> None:
        """A band release hint cannot silently introduce unmeasured overlap credit."""
        with self.assertRaisesRegex(ValueError, 'Early gradient release'):
            self.calibration(gradient_bands=((0, .5, 1., 1.), (1, 1., 1., 1.)))
        self.assertEqual(self.calibration(overlap_verified=True,
                         gradient_bands=((0, .5, 1., 1.), (1, 1., 1., 1.))).gradient_bands,
                         ((0, .5, 1., 1.), (1, 1., 1., 1.)))

    def test_each_gradient_band_waits_its_producer_and_ack_waits_all_bands(self) -> None:
        """One ready band never permits reading the unproduced remainder of a matrix."""
        counts, config, _ = self.example()
        copies = [{0: 2, 1: 0}, {2: 0, 3: 0, 0: 6}]
        calibration = self.calibration(overlap_verified=True, gradient_workers=2,
                                       gradient_bands=((0, .5, 1., 1.), (1, 1., 1., 1.)), gradient_ack_ms=1.)
        model = self.model(weight_ms=0., critical_path=calibration)
        estimate = calibration.estimate(model, copies, counts, config, trace=True)
        jobs = {task[0]:task for task in estimate.tasks}
        self.assertEqual(jobs['backward:gradient:0:1:0:projection0:band0'][1:3], (3., 4.))
        self.assertEqual(jobs['backward:gradient:0:1:0:projection1:band1'][1:3], (6., 7.))
        self.assertEqual(jobs['backward:ack:0:1:0'][1:3], (7., 8.))
        self.assertEqual(estimate.backward_ms, 8.)

    def test_w13_return_waits_the_owner_w2_projection_join(self) -> None:
        """A ready W13 band cannot bypass the deployed W2-first return sequence."""
        counts, config, _ = self.example()
        copies = [{0:2,1:0}, {2:0,3:0,0:6}]
        calibration = self.calibration(overlap_verified=True,
            gradient_bands=((0,1.,2.,2.),(1,.5,1.,1.)), gradient_ack_ms=0.)
        model = self.model(weight_ms=0., critical_path=calibration)
        estimate = calibration.estimate(model, copies, counts, config, trace=True)
        jobs = {task[0]:task for task in estimate.tasks}
        self.assertEqual(jobs['backward:gradient:0:1:0:projection0:band0'][1:3], (6.,8.))
        self.assertEqual(jobs['backward:gradient:0:1:0:projection1:band1'][1:3], (8.,9.))

    def test_json_round_trip_freezes_all_calibrated_state(self) -> None:
        """JSON calibration does not leave mutable tables in the recompute signature."""
        model = self.model()
        restored = ExpertReplicaCostModel(**json.loads(json.dumps(asdict(model))))
        self.assertEqual(restored, model)
        self.assertEqual(hash(restored), hash(model))

    def test_invalid_services_and_signatures_fail_closed(self) -> None:
        """Reject invalid profile data before any distributed operation."""
        for changes in ({'inter_node_gradient_ms':-1.}, {'execution_signature':'unknown'},
                        {'communication_domains':(True, 0)}, {'gradient_workers':5},
                        {'home_forward_ms':((0, 0), (128, float('nan')))}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.calibration(**changes)

    def test_execution_template_rejects_other_transport_protocols(self) -> None:
        """A P2P or native execution cannot reuse the kernel-gradient dependency graph."""
        for changes in ({'transport': 'p2p'}, {'backend': 'native', 'transport': 'p2p'}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'Critical-path template'):
                self.model(**changes)

    def test_original_template_rejects_the_unmapped_guest_schedule(self) -> None:
        """A new fingerprint cannot rename the old dependency estimator as a new graph."""
        with self.assertRaisesRegex(ValueError, 'Critical-path template'):
            self.model(schedule='fixed_queue_unmapped_guest_suffix_v1')

    def test_profile_selection_requires_the_exact_execution_fingerprint(self) -> None:
        """A changed kernel, dtype or topology cannot consume a device profile."""
        key = ('push', 16, 8, 2, 'shmem_signal_kernel_gradient_bands', 4, 2, 'test_soc')
        model = self.model()
        legacy = replace(model, critical_path=None)
        policy._resolve_cost_model.cache_clear()
        self.addCleanup(policy._resolve_cost_model.cache_clear)
        with patch.object(policy, '_calibrations', return_value={key:legacy, key+('a'*64,):model}):
            matched = policy._resolve_cost_model(*key, True, 'a'*64)
            stale = policy._resolve_cost_model(*key, True, 'b'*64)
            self.assertEqual(matched.critical_path, model.critical_path)
            self.assertIsNone(stale.critical_path)
        self.assertEqual(execution_fingerprint({'kernel':'a','dtype':'bf16'}),
                         execution_fingerprint({'dtype':'bf16','kernel':'a'}))
        self.assertNotEqual(execution_fingerprint({'kernel':'a'}), execution_fingerprint({'kernel':'b'}))
