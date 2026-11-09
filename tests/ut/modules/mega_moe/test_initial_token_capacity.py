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
"""Initial automatic source reservation and symmetric resource ownership."""

import unittest

import torch

from hyper_multicore.modules.mega_moe.heap_manager import _workspace_layout
from hyper_multicore.modules.mega_moe.module import MegaMoeExperts
from tests.common.mark_utils import arg_mark


class TestInitialTokenCapacity(unittest.TestCase):
    """Exercise the optional source reservation before device binding."""

    # pylint: disable=protected-access

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_initial_source_reservation_and_resource_compatibility(self) -> None:
        """Feature: Initial automatic source reservation.

        Description: Reserve compact-row capacity before binding and reject incompatible sharing.
        Expectation: Physical source storage covers the floor; defaults and validation remain explicit.
        """
        default = MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=8, top_k=1)
        self.addCleanup(default.close)
        self.assertEqual(default.initial_token_capacity, 128)
        for requested, aligned in ((1, 128), (257, 384), (24576, 24576)):
            with self.subTest(requested=requested):
                module = MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=8, top_k=1,
                                        initial_token_capacity=requested)
                self.addCleanup(module.close)
                layout = _workspace_layout(module._resource_group.specification, torch.bfloat16)
                self.assertEqual(layout.token_capacity, aligned)
                self.assertGreaterEqual(layout.receive_capacity, aligned)
                if aligned != 128:
                    with self.assertRaisesRegex(ValueError, "identical configuration"):
                        MegaMoeExperts.share_execution_resources((default, module))
        for invalid in (0, -1, True, False, 1.5, None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "initial_token_capacity"):
                MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=8, top_k=1,
                               initial_token_capacity=invalid)
