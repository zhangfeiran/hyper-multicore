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
"""Preserve default accelerators and require evidence before skipping owner initialization."""

import os
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from hyper_multicore.modules.mega_moe.configuration import execution_options, qualify_topology_schedule
from hyper_multicore.modules.mega_moe.module import MegaMoeExperts
from hyper_multicore.modules.mega_moe.route import _owner_empty_experts, _read_replica_dynamic_route
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec


class TestExecutionConfiguration(unittest.TestCase):
    """Separate no-table execution controls from any calibration dependency."""

    @staticmethod
    def _spec():
        """Describe CPU-test topology without binding a runtime or native adapter."""
        return MegaMoeSpec(
            local_num_tokens=2,
            hidden_size=16,
            intermediate_size=8,
            num_experts=4,
            top_k=2,
            initial_capacity_factor=1.25,
            receive_capacity=128,
            ep_size=1,
            ep_group=None,
            rank_id=0,
            num_cube_cores=24,
            auto_tokens=True,
            _selective_owner_gradient_init=True,
        )

    def test_defaults_and_research_controls_enter_resource_identity(self) -> None:
        """Native auto scheduling is a default while explicit unsupported values fail closed."""
        with patch.dict(os.environ, {}, clear=True):
            module = MegaMoeExperts(
                hidden_size=16, intermediate_size=8, num_experts=4, top_k=2, replica_slots_per_rank=1
            )
            values = module._resource_group.specification
            self.assertEqual(values["_gradient_queue_order"], "auto")
            self.assertTrue(values["_selective_owner_gradient_init"])
            self.assertFalse(values["_batch_empty_combine"])
        with patch.dict(os.environ, {"HYPER_MULTICORE_MOE_GRADIENT_QUEUE_ORDER": "unsupported"}, clear=True):
            with self.assertRaisesRegex(ValueError, "GRADIENT_QUEUE_ORDER"):
                execution_options("push", 1)

    def test_owner_skip_requires_positive_k_and_no_guests(self) -> None:
        """Only a positive source covering every expert proves complete owner overwrites."""
        spec = self._spec()
        positive = [[0, 2, 15, 4, 1, 1, 4]]
        self.assertEqual(_owner_empty_experts(spec, positive, None), ())
        self.assertIsNone(_owner_empty_experts(spec, [[0, 2, 15, 4, 0, 2, 4]], None))
        self.assertIsNone(_owner_empty_experts(replace(spec, replica_slots_per_rank=1), positive, None))
        self.assertIsNone(_owner_empty_experts(replace(spec, _selective_owner_gradient_init=False), positive, None))

    def test_multi_node_auto_preserves_w2_communication_order(self) -> None:
        """Qualify owner and guest plans from physical EP topology before native selection."""
        nodes = (0,) * 16 + (1,) * 16
        for budget in (0, 1):
            spec = replace(self._spec(), replica_slots_per_rank=budget, _gradient_queue_order="auto")
            selected = qualify_topology_schedule(spec, nodes)
            self.assertEqual(selected._gradient_queue_order, "auto_w1")
            self.assertEqual(spec._gradient_queue_order, "auto")
            self.assertEqual(selected.replica_slots_per_rank, budget)
            self.assertIs(qualify_topology_schedule(spec, (0,) * 32), spec)
            explicit = replace(spec, _gradient_queue_order="device_active_lanes")
            self.assertIs(qualify_topology_schedule(explicit, nodes), explicit)

    def test_dynamic_summary_retains_owner_overwrite_evidence(self) -> None:
        """The normal wide-table summary can prove owner initialization without another readback."""
        spec = self._spec()
        payload = torch.tensor([[0, 2, 15, 1, 1, 1, 1]], dtype=torch.int32)
        _, _, controls = _read_replica_dynamic_route(payload, spec)
        self.assertEqual(_owner_empty_experts(spec, controls, None), ())
