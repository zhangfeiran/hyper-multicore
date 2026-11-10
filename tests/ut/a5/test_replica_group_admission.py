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
"""Keep compact admission consistent with effective group-component scoring."""

import unittest

from hyper_multicore.replica import routing
from hyper_multicore.replica.cost import ExpertReplicaCostModel


class TestReplicaGroupAdmission(unittest.TestCase):
    """A safe lower bound must use the active predictor's aggregation."""

    def test_compact_admission_uses_group_features_instead_of_legacy_row_curves(self) -> None:
        """Unused compatibility curves cannot reverse owner-only admission."""
        totals = [256, 176, 80, 0]
        rows = [[0, 0, 128, 31, 256, 0, 128, 432, 216],
                [0, 0, 128, 31, 256, 0, 128, 80, 40]]
        for rate, active, expected in ((100., 500., True), (0., 0., False)):
            model = ExpertReplicaCostModel(calibration_version=2, backend='push', hidden_size=16,
                intermediate_size=8, ep_size=2, transport='shmem_signal_kernel_gradient_bands',
                forward_ms=((0, 0.), (1000, 1000 * rate)), backward_ms=((0, 0.), (1000, 1000 * rate)),
                weight_ms=0., gradient_ms=0., effective_group_ms=(0., 1., 0., 0., 0., active))
            with self.subTest(legacy_rate=rate, active_cost=active):
                actual = routing._compact_home_cost_dominates(
                    totals, rows, routing.ExpertReplicaConfig(4, 2, 1), model)
                self.assertEqual(actual, expected)
