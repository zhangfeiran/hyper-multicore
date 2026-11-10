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
"""Unit tests for the model-facing MegaMoe module."""

"""Keep small A5 shared gradients on the native forward path."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_multicore.modules.mega_moe.module import MegaMoeExperts
from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts, _resolve_a5_execution_spec
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec, bind_mega_moe_spec
from hyper_multicore.scheduler.config import TaskType


class TestSmallSharedForward(unittest.TestCase):
    """Verify resolved execution graphs at the hardware and row-count boundaries."""

    def bound(self, rows: int, cores: int) -> MegaMoeSpec:
        """Resolve the real module specification without allocating an NPU tensor."""
        module = MegaMoeExperts(hidden_size=32, intermediate_size=32,
                               num_experts=4, top_k=2, ep_size=1, create_parameters=False,
                               dispatch_mode='pull', forward_ingress_mode='indexed' if cores == 32 else 'legacy',
                               shared_intermediate_size=32, forward_shared_expert=True, backward_shared_expert=True)
        self.addCleanup(module.close)
        tensor = SimpleNamespace(device=torch.device('npu', 0), dtype=torch.bfloat16)
        with (patch('torch.distributed.is_initialized', return_value=False),
              patch.object(torch.npu, 'get_device_limit',
                           return_value={'cube_core_num': cores, 'vector_core_num': 2 * cores}, create=True)):
            return _resolve_a5_execution_spec(replace(
                bind_mega_moe_spec(module._resource_group.specification, tensor), local_num_tokens=rows))

    def test_small_a5_graph_omits_shared_forward_but_keeps_shared_backward(self) -> None:
        """The fallback must remove integrated launches and preserve the backward branch."""
        for rows in (128, 256, 384):
            with self.subTest(rows=rows):
                spec = self.bound(rows, 32)
                self.assertFalse(spec.forward_shared_expert)
                self.assertTrue(spec.backward_shared_expert)
                _, forward, _, backward = _build_runtime_artifacts(spec)
                fwd = [task for task in forward.all_tasks[:forward.task_num] if task.inputs[0].tensor_type == 2]
                bwd = [task for task in backward.all_tasks[:backward.task_num]
                       if task.inputs[0].tensor_type == 2 and task.task_type == TaskType.TASK_GROUPED_MATMUL]
                self.assertFalse(fwd)
                self.assertTrue(bwd)

    def test_validated_a5_target_retains_integrated_shared_forward(self) -> None:
        """Validated 512/4096-token graphs continue to include the requested shared work."""
        for rows in (512, 4096):
            with self.subTest(rows=rows):
                spec = self.bound(rows, 32)
                self.assertTrue(spec.forward_shared_expert)
                _, forward, _, _ = _build_runtime_artifacts(spec)
                self.assertTrue(any(task.inputs[0].tensor_type == 2
                                    for task in forward.all_tasks[:forward.task_num]))

    def test_a3_small_shape_keeps_its_existing_shared_graph(self) -> None:
        """The A5 precision policy must not disable the A3 implementation."""
        spec = self.bound(128, 24)
        self.assertTrue(spec.forward_shared_expert)
        _, forward, _, _ = _build_runtime_artifacts(spec)
        self.assertTrue(any(task.inputs[0].tensor_type == 2 for task in forward.all_tasks[:forward.task_num]))
