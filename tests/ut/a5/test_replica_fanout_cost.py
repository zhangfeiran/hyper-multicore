# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Fanout-aware forward weights, exact incremental scores and explicit selector evidence."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from hyper_multicore.replica.capacity import ExpertReplicaConfig
from hyper_multicore.replica.cost import ExpertReplicaCostModel, _QuotaCostState
from hyper_multicore.replica import policy
def histogram_ids(counts: list[int], tokens: int, top_k: int) -> torch.Tensor:
    """Materialize exact counts as distinct per-token expert assignments."""
    if len(counts) < top_k or sum(counts) != tokens * top_k or any(type(n) is not int or not 0 <= n <= tokens
                                                                   for n in counts):
        raise ValueError('Routing histogram must conserve N*K valid entries')
    return torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts)).reshape(top_k, tokens).T.contiguous()


class TestReplicaFanoutCost(unittest.TestCase):
    """Use CPU-only cost states with independent expected protocol costs."""

    @staticmethod
    def model(**options: object) -> ExpertReplicaCostModel:
        """Use exact linear compute and independently measured forward-copy knots."""
        defaults = dict(calibration_version=2, backend='push', hidden_size=16, intermediate_size=8,
                        ep_size=8, transport='p2p', forward_ms=((0, 0), (1000, 10)),
                        backward_ms=((0, 0), (1000, 20)), weight_ms=3, gradient_ms=.5,
                        backward_weight_ms=0, intra_node_weight_ms=2, inter_node_weight_ms=3,
                        weight_fanout_ms=((0, 0, 0), (1, 2, 3), (2, 3, 4), (4, 5, 7)),
                        _rank_nodes=(0, 0, 1, 1, 2, 2, 3, 3))
        return ExpertReplicaCostModel(**(defaults | options))

    @staticmethod
    def state(model: ExpertReplicaCostModel, targets: tuple[int, ...]) -> _QuotaCostState:
        """Route ten rows to each guest from one logical expert."""
        counts = ((100,) + (0,) * 15,) + ((0,) * 16,) * 7
        copies = [{rank * 2: 0, rank * 2 + 1: 0} for rank in range(8)]
        copies[0][0] = 100 - len(targets) * 10
        for rank in targets:
            copies[rank][0] = 10
        return _QuotaCostState(model, copies, counts, ExpertReplicaConfig(16, 8, 1))

    def test_outgoing_curve_keeps_incoming_and_backward_charges_separate(self) -> None:
        """Two mixed-link sends cost 3.5, while guests retain independent single-copy costs."""
        state = self.state(self.model(), (1, 2))
        self.assertAlmostEqual(state.costs[0][0], .8 + 3.5)
        self.assertAlmostEqual(state.costs[1][0], .1 + 2)
        self.assertAlmostEqual(state.costs[2][0], .1 + 3)
        self.assertAlmostEqual(state.costs[0][1], 1.6 + 1)

    def test_interpolate_and_fallback_without_silent_extrapolation(self) -> None:
        """Unmeasured high fanout and unknown topology retain linear costs."""
        model = self.model()
        self.assertEqual(model.weight_copy_ms(0, 0), 0)
        self.assertEqual(model.weight_copy_ms(3, 3), 4)
        self.assertEqual(model.weight_copy_ms(3, 0), 5.5)
        self.assertEqual(model.weight_copy_ms(3, 1), 5)
        self.assertEqual(model.weight_copy_ms(5, 5), 10)
        unknown = self.state(replace(model, _rank_nodes=()), (1, 2))
        self.assertAlmostEqual(unknown.costs[0][0], .8 + 6)

    def test_legacy_empty_table_preserves_scores(self) -> None:
        """Omitting fanout calibration preserves exact existing arithmetic."""
        state = self.state(self.model(weight_fanout_ms=()), (1, 2))
        self.assertEqual(state.costs[0], (.8 + 5, 1.6 + 1))

    def test_trial_removal_and_accept_equal_independent_rebuild(self) -> None:
        """Removing a final guest updates owner fanout and target incoming counts together."""
        state = self.state(self.model(), (1, 2, 3))
        for target in (1, 2, 3):
            unchanged = tuple(max(value[phase] for rank, value in enumerate(state.costs)
                                  if rank not in (0, target)) for phase in (0, 1))
            state.copies[0][0] += state.copies[target][0]
            state.copies[target][0] = 0
            trial = state.trial_score(0, target, 0, unchanged)
            rebuilt = _QuotaCostState(state.model, state.copies, state.counts, ExpertReplicaConfig(16, 8, 1))
            self.assertEqual(trial, rebuilt.score())
            state.accept(0, target, 0)
            self.assertEqual(state.score(), rebuilt.score())
            self.assertEqual(state.costs, rebuilt.costs)

    def test_distinct_experts_do_not_share_fixed_cost(self) -> None:
        """An owner sending two experts twice pays two fanout-two curves."""
        counts = ((100, 100) + (0,) * 14,) + ((0,) * 16,) * 7
        copies = [{rank * 2: 0, rank * 2 + 1: 0} for rank in range(8)]
        copies[0] = {0: 80, 1: 80}
        for target, expert in ((1, 0), (2, 0), (3, 1), (4, 1)):
            copies[target][expert] = 10
        model = self.model(_rank_nodes=(0,) * 8)
        state = _QuotaCostState(model, copies, counts, ExpertReplicaConfig(16, 8, 1))
        self.assertAlmostEqual(state.costs[0][0], 1.6 + 2 * 3)

    def test_rank_can_own_one_expert_and_receive_another(self) -> None:
        """Incoming removal must preserve the target's unrelated outgoing fanout curve."""
        counts = ((100, 0, 100) + (0,) * 13,) + ((0,) * 16,) * 7
        copies = [{rank * 2: 0, rank * 2 + 1: 0} for rank in range(8)]
        copies[0][0], copies[1][2] = 70, 80
        for owner, targets in ((0, (1, 2, 3)), (1, (4, 5))):
            for target in targets:
                copies[target][owner * 2] = 10
        config = ExpertReplicaConfig(16, 8, 1)
        state = _QuotaCostState(self.model(), copies, counts, config)
        self.assertAlmostEqual(state.costs[1][0], .9 + 4 + 2)
        unchanged = tuple(max(value[phase] for rank, value in enumerate(state.costs) if rank not in (0, 1))
                          for phase in (0, 1))
        copies[0][0], copies[1][0] = 80, 0
        trial = state.trial_score(0, 1, 0, unchanged)
        rebuilt = _QuotaCostState(state.model, copies, counts, config)
        self.assertEqual(trial, rebuilt.score())
        state.accept(0, 1, 0)
        self.assertEqual(state.costs, rebuilt.costs)
        self.assertAlmostEqual(state.costs[1][0], .8 + 4)

    def test_invalid_tables_fail_before_planning(self) -> None:
        """Reject negative, unordered or incomplete calibration rows."""
        for table in (((1, 2, 3),), ((0, 0, 0), (1, -1, 3)), ((0, 0, 0), (2, 2, 3), (1, 3, 4)),
                      ((0, 0, 0), (1, 2, 3), (2, 1, 4)), ((0, 0, 0), (1, 2))):
            with self.subTest(table=table), self.assertRaises(ValueError):
                self.model(weight_fanout_ms=table)
        for total, local in ((1, 2), (-1, 0), (True, 0)):
            with self.subTest(total=total, local=local), self.assertRaises(ValueError):
                self.model().weight_copy_ms(total, local)


class TestSharedSocCalibration(unittest.TestCase):
    """The explicitly supported pair shares coefficients; other devices still fall back."""

    def tearDown(self) -> None:
        policy._resolve_cost_model.cache_clear()

    def test_supported_pair_shares_compute_and_records_actual_soc(self) -> None:
        """Cache misses report both actual and table identity on rank zero."""
        arguments = ('push', 5120, 2304, 64, 'shmem_signal_kernel_gradient_bands', 384, 6)
        policy._resolve_cost_model.cache_clear()
        with patch.object(policy.dist, 'is_initialized', return_value=False):
            with self.assertLogs(policy.__name__, level='INFO') as captured:
                alias = policy._resolve_cost_model(*arguments, 'Ascend910_9382', True)
                native = policy._resolve_cost_model(*arguments, 'Ascend910_9392', True)
                self.assertIsNone(policy._resolve_cost_model(*arguments, 'Ascend910_9381', True))
        self.assertEqual(alias, native)
        self.assertIsNotNone(alias)
        self.assertIn('actual_soc=Ascend910_9382', '\n'.join(captured.output))
        self.assertIn('table_soc=Ascend910_9392', '\n'.join(captured.output))
        self.assertIn('status=fallback', '\n'.join(captured.output))

    def test_exact_table_wins_over_alias_and_nonzero_ranks_do_not_log(self) -> None:
        """An explicit device calibration always takes precedence over a shared fallback."""
        arguments = ('push', 5120, 2304, 64, 'shmem_signal_kernel_gradient_bands', 384, 6)
        original = policy._resolve_cost_model(*arguments, 'Ascend910_9392', True)
        exact = replace(original, weight_ms=123)
        policy._resolve_cost_model.cache_clear()
        with patch.object(policy, '_calibrations', return_value={arguments + ('Ascend910_9382',): exact}), \
                patch.object(policy.dist, 'is_initialized', return_value=True), \
                patch.object(policy.dist, 'get_rank', return_value=7), \
                patch.object(policy._LOGGER, 'info') as log:
            selected = policy._resolve_cost_model(*arguments, 'Ascend910_9382', True)
        self.assertEqual(selected.weight_ms, 123)
        log.assert_not_called()


class TestHistoricalHistogramRealization(unittest.TestCase):
    """Exact per-source histograms need no original token IDs to reproduce communication load."""

    def test_counts_and_distinct_experts_per_token(self) -> None:
        """Column-major placement preserves quotas and prevents duplicate experts per row."""
        counts = [8, 7, 6, 5, 4, 2]
        ids = histogram_ids(counts, 8, 4)
        self.assertEqual(tuple(ids.shape), (8, 4))
        self.assertEqual(torch.bincount(ids.flatten(), minlength=len(counts)).tolist(), counts)
        self.assertTrue(all(len(set(row)) == 4 for row in ids.tolist()))

    def test_invalid_histograms_fail_before_device_setup(self) -> None:
        """Too many repeats or mismatched total quotas cannot masquerade as legal TopK routing."""
        for counts in ([9, 7], [4, 4], [-1, 9, 8], [True, 7, 8]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                histogram_ids(counts, 8, 2)
