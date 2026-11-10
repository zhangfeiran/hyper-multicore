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
"""Deterministic cost-guided quota selection with independent capacity checks."""

import random
import unittest
from dataclasses import replace
from typing import Tuple, Union
from unittest.mock import PropertyMock, patch

from hyper_multicore.replica import (
    ExpertReplicaConfig,
    ExpertReplicaCostModel,
    cost,
)
from hyper_multicore.replica.planner import (
    _build_expert_replica_plan as build_expert_replica_plan, _materialize,
)


class TestReplicaCost(unittest.TestCase):
    """Keep measured costs advisory and preserve every routing invariant."""

    def test_zero_token_coefficient_skips_unused_source_expert_reads(self) -> None:
        """A zero token term retains the score without walking per-expert source hits."""
        class NoHits(tuple):
            """Allow aggregate iteration while rejecting unused expert-index reads."""

            def __getitem__(self, _index: Union[int, slice]) -> Union[int, Tuple[int, ...]]:
                raise AssertionError('Source expert hit was read')

        counts = ((10, 4, 0, 0), (6, 0, 0, 0))
        guarded = tuple(NoHits(row) for row in counts)
        copies = [{0: 10, 1: 4}, {2: 0, 3: 0, 0: 6}]
        config = ExpertReplicaConfig(4, 2, 1)
        model = self._model(effective_group_ms=(.5, .1, .2, .3, 0., .4), _rank_nodes=(0, 1))
        expected = cost._QuotaCostState(model, copies, counts, config).score()
        actual = cost._QuotaCostState(model, copies, guarded, config).score()
        self.assertEqual(actual, expected)
        positive = replace(model, effective_group_ms=(.5, .1, .2, .3, .25, .4))
        with self.assertRaisesRegex(AssertionError, 'Source expert hit'):
            cost._QuotaCostState(positive, copies, guarded, config).score()

    def test_resident_backward_weights_keep_forward_copy_cost(self) -> None:
        """Residency removes one backward copy without discounting forward or replay."""
        plan = build_expert_replica_plan([[8, 0, 0, 0]] * 2, 1, capacity_limit=16)
        model = self._model(weight_ms=3, gradient_ms=2, forward_repeats=2)
        resident = replace(model, backward_weight_ms=0)
        self.assertAlmostEqual(model.estimate_ms(plan) - resident.estimate_ms(plan), 3)
        repeated = replace(model, forward_repeats=3)
        self.assertAlmostEqual(repeated.estimate_ms(plan) -
                               replace(repeated, backward_weight_ms=0).estimate_ms(plan), 3)
        home = build_expert_replica_plan([[8, 0, 0, 0]] * 2, 0, capacity_limit=16)
        self.assertEqual(model.estimate_ms(home), resident.estimate_ms(home))

    def test_optional_replica_must_clear_explicit_owner_gain_margin(self) -> None:
        """Prefer the feasible owner when replica savings do not clear calibration uncertainty."""
        counts = [[10, 4, 0, 0]] * 2
        table = ((0, 0), (128, 128))
        original = self._model(forward_ms=table, backward_ms=table, forward_repeats=2,
                               quota_search_budget_ms=1000)
        replica = build_expert_replica_plan(counts, 1, capacity_limit=28, cost_model=original)
        self.assertTrue(replica.transfers)
        for margin in (41, 42, 43):
            with self.subTest(margin=margin):
                model = replace(original, minimum_replica_gain_ms=margin)
                loose = build_expert_replica_plan(counts, 1, capacity_limit=28, cost_model=model)
                self.assertEqual(bool(loose.transfers), margin < 42)
                tight = build_expert_replica_plan(counts, 1, capacity_limit=14, cost_model=model)
                tight.validate()
                self.assertTrue(tight.transfers)
                self.assertEqual(max(tight.destination_loads), 14)

    def test_profit_admission_does_not_extrapolate_uncovered_owner(self) -> None:
        """A mandatory covered split cannot compare against an unmeasured full expert."""
        model = self._model(forward_ms=((0, 0), (16, 1)), backward_ms=((0, 0), (16, 2)),
                            minimum_replica_gain_ms=1000)
        plan = build_expert_replica_plan([[10, 4, 0, 0]] * 2, 1, capacity_limit=14, cost_model=model)
        self.assertTrue(plan.transfers)
        plan.validate()

    def test_residency_and_admission_costs_validate_optional_values(self) -> None:
        """Missing fields preserve the legacy schema; explicit costs must be nonnegative."""
        for name in ('backward_weight_ms', 'minimum_replica_gain_ms'):
            for value in (-1, True, float('nan'), float('inf')):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    self._model(**{name: value})
            self.assertEqual(getattr(self._model(**{name: 0}), name), 0)

    @staticmethod
    def _model(**options):
        values = {"calibration_version": 2, "backend": "native", "hidden_size": 8,
                  "intermediate_size": 4, "ep_size": 2, "transport": "p2p",
                  "forward_ms": ((0, 0), (8, 1), (16, 1.1), (128, 3)),
                  "backward_ms": ((0, 0), (8, 1), (16, 1.1), (128, 3)), "weight_ms": 0.0, "gradient_ms": 0.0}
        return ExpertReplicaCostModel(**(values | options))

    def test_zero_token_score_does_not_aggregate_source_rows(self) -> None:
        """An unused token coefficient must not force a source-by-expert sum."""
        counts, config = ((10, 0, 0, 0), (6, 0, 0, 0)), ExpertReplicaConfig(4, 2, 1)
        copies = [{0: 8, 1: 0}, {2: 0, 3: 0, 0: 8}]
        with patch.object(cost._QuotaCostState, 'source_totals', new_callable=PropertyMock,
                          side_effect=AssertionError('Unexpected source aggregation')):
            score = cost._QuotaCostState(self._model(), copies, counts, config).score()
        self.assertGreater(score, 0)

    def test_nonzero_token_score_reuses_source_totals(self) -> None:
        """Token-sensitive scoring must retain exact invocation-local source totals."""
        counts, config = ((10, 0, 0, 0), (6, 0, 0, 0)), ExpertReplicaConfig(4, 2, 1)
        copies = [{0: 8, 1: 0}, {2: 0, 3: 0, 0: 8}]
        state = cost._QuotaCostState(self._model(token_ms_per_row=.01), copies, counts, config)
        self.assertEqual(state.source_totals, [10, 6])
        self.assertIs(state.source_totals, state.source_totals)
        self.assertGreater(state.score(), cost._QuotaCostState(self._model(), copies, counts, config).score())

    def test_group_quota_trials_sum_each_immutable_source_once(self) -> None:
        """Quota changes must refresh their score without rescanning immutable source rows."""
        counts, config = ((10, 0, 0, 0), (6, 0, 0, 0)), ExpertReplicaConfig(4, 2, 1)
        copies = [{0: 8, 1: 0}, {2: 0, 3: 0, 0: 8}]
        model = self._model(effective_group_ms=(.5, 1., 2., 3., .25, 4.), _rank_nodes=(0, 1))
        state = cost._QuotaCostState(model, copies, counts, config)
        trials = (8, 3, 0, 7, 8)
        expected, actual = [], []
        with patch.object(cost, 'sum', wraps=sum, create=True) as aggregate:
            for quota in trials:
                copies[0][0], copies[1][0] = 16 - quota, quota
                actual.append(state.trial_score(0, 1, 0, (0., 0.)))
                actual.append(state.score())
            source_sums = [sum(call.args[0] is row for call in aggregate.call_args_list) for row in counts]
        for quota in trials:
            copies[0][0], copies[1][0] = 16 - quota, quota
            features = cost._effective_group_features(copies, counts, config.home_experts, model._rank_nodes)
            expected.extend([sum(value * coefficient for value, coefficient
                                 in zip(features, model.effective_group_ms))] * 2)
        self.assertEqual(actual, expected)
        self.assertEqual(source_sums, [1, 1])

    def test_small_copy_is_removed_on_score_plateau_before_search_guard(self) -> None:
        """A cheap deletion must not require a strict gain or expensive quota search."""
        counts = ((10, 0, 0, 0, 20, 0, 0, 0),) + ((0,) * 8,) * 3
        copies = [{0: 9, 1: 0}, {2: 0, 3: 0, 0: 1}, {4: 20, 5: 0}, {6: 0, 7: 0}]
        config = ExpertReplicaConfig(8, 4, 1)
        model = self._model(ep_size=4, forward_ms=((0, 0), (128, 128)),
                            backward_ms=((0, 0), (128, 128)), weight_ms=1,
                            minimum_gain_ms=1000, quota_search_budget_ms=1000)
        before = _materialize(counts, copies, config, 20)
        with patch.object(cost, "_refine_edge", side_effect=AssertionError("Expensive quota search")):
            cost.refine_replica_quotas(copies, counts, config, 20, model, minimum_replica_rows=2)
        after = _materialize(counts, copies, config, 20)
        after.validate()
        self.assertTrue(before.transfers)
        self.assertFalse(after.transfers)
        self.assertEqual(max(after.destination_loads), max(before.destination_loads))
        self.assertEqual(model.estimate_ms(after), model.estimate_ms(before))

    def test_small_copy_required_by_current_peak_is_retained(self) -> None:
        """Deleting a small fragment must not increase the existing receive reservation."""
        counts = ((10, 0, 0, 0, 9, 0, 0, 0),) + ((0,) * 8,) * 3
        copies = [{0: 9, 1: 0}, {2: 0, 3: 0, 0: 1}, {4: 9, 5: 0}, {6: 0, 7: 0}]
        config = ExpertReplicaConfig(8, 4, 1)
        before = _materialize(counts, copies, config, 9)
        model = self._model(ep_size=4, minimum_gain_ms=1000, quota_search_budget_ms=1000)
        cost.refine_replica_quotas(copies, counts, config, 9, model, minimum_replica_rows=2)
        after = _materialize(counts, copies, config, 9)
        after.validate()
        self.assertEqual(after, before)

    def test_small_copy_with_worse_compute_score_is_retained(self) -> None:
        """A safe row peak alone does not justify moving into a slower compute bucket."""
        counts = ((3, 1, 0, 0, 2, 2, 0, 0),) + ((0,) * 8,) * 3
        copies = [{0: 2, 1: 1}, {2: 0, 3: 0, 0: 1}, {4: 2, 5: 2}, {6: 0, 7: 0}]
        config = ExpertReplicaConfig(8, 4, 1)
        before = _materialize(counts, copies, config, 4)
        table = ((0, 0), (1, 1), (2, 10), (3, 100), (128, 200))
        model = self._model(ep_size=4, forward_ms=table, backward_ms=table,
                            minimum_gain_ms=1000, quota_search_budget_ms=1000)
        cost.refine_replica_quotas(copies, counts, config, 4, model, minimum_replica_rows=2)
        after = _materialize(counts, copies, config, 4)
        after.validate()
        self.assertEqual(after, before)

    def test_small_copy_pruning_does_not_extrapolate_owner_rows(self) -> None:
        """Covered fragments cannot be merged into an unmeasured expert row count."""
        counts = ((16, 0, 0, 0, 10, 10, 0, 0),) + ((0,) * 8,) * 3
        copies = [{0: 12, 1: 0}, {2: 0, 3: 0, 0: 4}, {4: 10, 5: 10}, {6: 0, 7: 0}]
        config = ExpertReplicaConfig(8, 4, 1)
        before = _materialize(counts, copies, config, 20)
        model = self._model(ep_size=4, forward_ms=((0, 0), (12, 1)), backward_ms=((0, 0), (12, 1)),
                            minimum_gain_ms=1000, quota_search_budget_ms=1000)
        cost.refine_replica_quotas(copies, counts, config, 20, model, minimum_replica_rows=8)
        after = _materialize(counts, copies, config, 20)
        after.validate()
        self.assertEqual(after, before)

    def test_compute_cost_can_prefer_more_rows_on_one_rank(self):
        """Fewer expert fragments may finish sooner despite a larger receive peak."""
        counts = [[10, 4, 0, 0]] * 2
        model = self._model()
        baseline = build_expert_replica_plan(counts, 1, capacity_limit=28)
        candidate = build_expert_replica_plan(counts, 1, capacity_limit=28, cost_model=model)
        self.assertLess(model.estimate_ms(candidate), model.estimate_ms(baseline))
        self.assertGreater(max(candidate.destination_loads), max(baseline.destination_loads))
        self.assertLessEqual(max(candidate.destination_loads), 28)
        self.assertEqual(candidate.transfers, baseline.transfers)
        candidate.validate()

    def test_expensive_copy_is_removed_only_when_capacity_allows(self):
        """Count full-copy costs separately from changing the quota of a retained copy."""
        counts = [[10, 4, 0, 0]] * 2
        model = self._model(weight_ms=100, gradient_ms=100)
        loose = build_expert_replica_plan(counts, 1, capacity_limit=28, cost_model=model)
        self.assertFalse(loose.transfers)
        tight = build_expert_replica_plan(counts, 1, capacity_limit=14, cost_model=model)
        self.assertTrue(tight.transfers)
        self.assertEqual(max(tight.destination_loads), 14)

    def test_push_cost_cannot_grow_the_receive_buffer_for_optional_savings(self) -> None:
        """Keep replicas when removing them would exceed current or unavoidable receive rows."""
        counts = [[10, 4, 0, 0]] * 2
        model = self._model(weight_ms=100, gradient_ms=100)
        baseline = build_expert_replica_plan(counts, 1, capacity_limit=28)
        peak = max(baseline.destination_loads)
        for capacity in (0, peak):
            with self.subTest(receive_capacity=capacity):
                plan = build_expert_replica_plan(counts, 1, capacity_limit=28,
                                                receive_capacity=capacity, cost_model=model)
                plan.validate()
                self.assertTrue(plan.transfers)
                self.assertLessEqual(max(plan.destination_loads), peak)
        resident = build_expert_replica_plan(counts, 1, capacity_limit=28,
                                            receive_capacity=28, cost_model=model)
        self.assertFalse(resident.transfers)
        self.assertEqual(max(resident.destination_loads), 28)
        self.assertEqual(build_expert_replica_plan(counts, 1, capacity_limit=28, receive_capacity=1), baseline)
        for invalid in (-1, True, 1.5):
            with self.subTest(capacity=invalid), self.assertRaisesRegex(ValueError, "receive_capacity"):
                build_expert_replica_plan(counts, 1, receive_capacity=invalid)

    def test_capacity_guard_survives_final_optional_copy_admission(self) -> None:
        """A minimum-gain policy cannot undo capacity-required guests after quota refinement."""
        generator = random.Random(1718)
        model = self._model(weight_ms=.1, gradient_ms=.2)
        for threshold in (.0, .2, .4, .6):
            for _ in range(24):
                counts = [[generator.randrange(1, 12) for _ in range(4)] for _ in range(2)]
                limit = sum(map(sum, counts))
                baseline = build_expert_replica_plan(counts, 1, capacity_limit=limit,
                                                    minimum_load_reduction=threshold)
                peak = max(baseline.destination_loads)
                plan = build_expert_replica_plan(counts, 1, capacity_limit=limit, receive_capacity=max(0, peak-1),
                                                minimum_load_reduction=threshold, cost_model=model)
                plan.validate()
                self.assertLessEqual(max(plan.destination_loads), peak)

    def test_uncertainty_and_unmeasured_ranges_keep_original_plan(self):
        """Do not spend routing changes on tiny predictions or unsupported row counts."""
        counts = [[10, 4, 0, 0]] * 2
        baseline = build_expert_replica_plan(counts, 1, capacity_limit=28)
        for model in (self._model(minimum_gain_ms=1000),
                      self._model(forward_ms=((0, 0), (8, 1)), backward_ms=((0, 0), (8, 1)))):
            self.assertEqual(build_expert_replica_plan(counts, 1, capacity_limit=28, cost_model=model), baseline)

    def test_cost_refinement_preserves_source_conservation_and_budget(self):
        """Random legal plans may only improve prediction within the same hard bound."""
        rng = random.Random(914)
        model = self._model(weight_ms=0.1, gradient_ms=0.2, token_ms_per_row=0.01)
        for _ in range(40):
            counts = [[rng.randrange(10) for _ in range(4)] for _ in range(2)]
            baseline = build_expert_replica_plan(counts, 1)
            limit = max(baseline.destination_loads)
            candidate = build_expert_replica_plan(counts, 1, capacity_limit=limit, cost_model=model)
            candidate.validate()
            self.assertLessEqual(max(candidate.destination_loads), limit)
            self.assertLessEqual(model.estimate_ms(candidate), model.estimate_ms(baseline))
            self.assertTrue(set(candidate.transfers).issubset(set(baseline.transfers)))
            self.assertEqual(candidate, build_expert_replica_plan(counts, 1, capacity_limit=limit, cost_model=model))

    def test_cost_table_validation_and_execution_identity(self):
        """Reject malformed times, unsupported native transports and stale shape calibration."""
        for options in ({"weight_ms": float("nan")}, {"gradient_ms": -1}, {"minimum_gain_ms": True},
                        {"forward_ms": ((0, 0), (8, 2), (16, 1))}, {"backward_ms": ((1, 0), (8, 1))},
                        {"forward_ms": ((0, 0), (8, 1), (8, 2))}, {"transport": "shmem_signal"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self._model(**options)
        model = self._model()
        self.assertAlmostEqual(model.compute_ms(12), 1.05)
        with self.assertRaisesRegex(ValueError, "calibrated"):
            model.compute_ms(129)
        with self.assertRaisesRegex(ValueError, "does not match"):
            model.validate_execution("native", 16, 4, 2, "p2p")
        with self.assertRaisesRegex(ValueError, "EP size"):
            build_expert_replica_plan([[1, 1]], 1, cost_model=model)
        with self.assertRaises(ValueError):
            build_expert_replica_plan([[1, 1]], 1, cost_model=object())
        overlapping = replace(model, backend="push", transport="p2p",
                              schedule="fixed_queue_v1", weight_ms=10)
        serial = replace(overlapping, backend="native", transport="p2p", schedule="native_grouped_v1")
        plan = build_expert_replica_plan([[10, 4, 0, 0]] * 2, 1)
        self.assertEqual(overlapping.estimate_ms(plan), serial.estimate_ms(plan))

    def test_old_or_incompatible_calibration_is_rejected(self):
        """Version, schedule and dW dtype are part of the calibration contract."""
        for options in ({"calibration_version": 1}, {"calibration_version": True},
                        {"schedule": "whole_home_overlap"}, {"gradient_dtype": "bfloat16"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self._model(**options)

    def test_checkpoint_replay_charges_forward_and_host_decisions(self) -> None:
        """Replay repeats forward costs while weight-gradient work runs once."""
        plan = build_expert_replica_plan([[8, 0, 0, 0]] * 2, 1, capacity_limit=16)
        model = self._model(forward_ms=((0, 0), (128, 128)),
                            backward_ms=((0, 0), (128, 256)), replica_overhead_ms=5)
        peak = max(plan.destination_loads)
        self.assertTrue(plan.transfers)
        self.assertEqual(model.estimate_ms(plan), peak * 3 + 5)
        self.assertEqual(replace(model, forward_repeats=2).estimate_ms(plan), peak * 4 + 10)
        home = build_expert_replica_plan([[8, 0, 0, 0]] * 2, 0, capacity_limit=16)
        self.assertEqual(replace(model, forward_repeats=2).estimate_ms(home), 64)

    def test_host_budget_can_reject_all_optional_copies(self) -> None:
        """Compare the complete owner placement without violating a required receive bound."""
        counts = [[10, 4, 0, 0]] * 2
        model = self._model(replica_overhead_ms=100, forward_repeats=2)
        loose = build_expert_replica_plan(counts, 1, capacity_limit=28, cost_model=model)
        tight = build_expert_replica_plan(counts, 1, capacity_limit=14, cost_model=model)
        self.assertFalse(loose.transfers)
        self.assertTrue(tight.transfers)
        self.assertEqual(max(tight.destination_loads), 14)
        loose.validate()
        tight.validate()

    def test_common_planner_budget_does_not_bias_replica_admission(self) -> None:
        """Admission cannot save the host work already performed to construct either candidate."""
        counts = [[10, 4, 0, 0]] * 2
        original = self._model(forward_repeats=2)
        common = replace(original, planner_overhead_ms=100)
        for limit in (14, 28):
            with self.subTest(limit=limit):
                before = build_expert_replica_plan(counts, 1, capacity_limit=limit, cost_model=original)
                after = build_expert_replica_plan(counts, 1, capacity_limit=limit, cost_model=common)
                self.assertEqual(before, after)
                self.assertAlmostEqual(common.estimate_ms(after) - original.estimate_ms(after), 200)
        home = build_expert_replica_plan(counts, 0, capacity_limit=28)
        self.assertAlmostEqual(common.estimate_ms(home) - original.estimate_ms(home), 200)
        with self.assertRaises(ValueError):
            replace(original, planner_overhead_ms=-1)

    def test_replica_only_budget_is_distinct_from_common_planner_work(self) -> None:
        """Only exposed replica scheduling work disappears with the last guest edge."""
        counts = [[8, 0, 0, 0]] * 2
        replica = build_expert_replica_plan(counts, 1, capacity_limit=16)
        home = build_expert_replica_plan(counts, 0, capacity_limit=16)
        original = self._model(forward_repeats=2)
        measured = replace(original, planner_overhead_ms=50, replica_overhead_ms=5)
        self.assertAlmostEqual(measured.estimate_ms(replica) - original.estimate_ms(replica), 110)
        self.assertAlmostEqual(measured.estimate_ms(home) - original.estimate_ms(home), 100)

    def test_partial_coverage_does_not_extrapolate_to_unsplit_experts(self) -> None:
        """A covered split remains legal even when its original logical total is uncovered."""
        counts = [[10, 4, 0, 0]] * 2
        model = self._model(forward_ms=((0, 0), (16, 1)), backward_ms=((0, 0), (16, 2)),
                            weight_ms=100)
        plan = build_expert_replica_plan(counts, 1, capacity_limit=14, cost_model=model)
        self.assertTrue(plan.transfers)
        self.assertLessEqual(max(max(row) for row in plan.destination_counts), 16)
        self.assertLessEqual(max(plan.destination_loads), 14)
        plan.validate()

    def test_bands_identity_and_replay_validation(self) -> None:
        """Calibrated bands require their own schedule and a positive integral replay count."""
        model = self._model(backend="push", transport="shmem_signal_kernel_gradient_bands")
        self.assertEqual(model.schedule, "fixed_queue_gradient_bands_v1")
        for options in ({"forward_repeats": 0}, {"forward_repeats": True},
                        {"replica_overhead_ms": float("inf")}, {"replica_overhead_ms": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self._model(**options)

    def test_unreachable_gain_skips_quota_search_without_changing_capacity(self) -> None:
        """Avoid paying host search cost when even an optimistic compute bound cannot cover it."""
        counts = [[10, 4, 0, 0]] * 2
        baseline = build_expert_replica_plan(counts, 1, capacity_limit=14)
        model = self._model(quota_search_budget_ms=1000)
        for common in (0, 1000000):
            with self.subTest(planner_overhead_ms=common), patch.object(
                    cost, "_refine_edge", side_effect=AssertionError("Unprofitable quota search")):
                candidate = build_expert_replica_plan(
                    counts, 1, capacity_limit=14, cost_model=replace(model, planner_overhead_ms=common))
            self.assertEqual(candidate, baseline)
            candidate.validate()
        with self.assertRaisesRegex(ValueError, "quota_search_budget_ms"):
            self._model(quota_search_budget_ms=-1)
