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
"""Verify compact backward publications and unchanged routed synchronization."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_multicore.modules.mega_moe import function as moe_function
from hyper_multicore.modules.mega_moe.module import MegaMoeExperts
from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.modules.mega_moe.workspace import _workspace_allocation_specs
from hyper_multicore.modules.mega_moe.indexed_backward import publish_backward_route
from hyper_multicore.modules.mega_moe.backward.storage import can_reuse_backward_dispatch
from hyper_multicore.torch import ops as native_ops
from hyper_multicore.scheduler.config import EVENT_INVALID_ID, TaskType


class TestIndexedBackward(unittest.TestCase):
    """Preserve backward mathematics while eliminating the sender's expanded dY."""

    def test_compact_source_reserves_only_original_rows(self) -> None:
        """Account for source storage separately from return and metadata buffers."""
        options = dict(dispatch_mode="pull", routed_slots=32768, receive_capacity=32768,
                       hidden_size=5120, ep_size=4, num_experts=32, dtype=torch.bfloat16,
                       top_k=8, forward_ingress_mode="indexed")
        legacy = dict(_workspace_allocation_specs(**options))
        compact = dict(_workspace_allocation_specs(**options, backward_ingress_mode="indexed"))
        self.assertEqual(legacy['source_buffer'].shape, (32768, 5120))
        self.assertEqual(compact['source_buffer'].shape, (4096, 5120))
        self.assertEqual(compact['routed_buffer'], legacy['routed_buffer'])
        self.assertEqual(compact['route_metadata'], legacy['route_metadata'])
        self.assertIn('backward_route_metadata', compact)

    def test_backward_tasks_keep_every_routed_event_and_cube_coordinate(self) -> None:
        """Require the new feature to change ingress selection without changing GMM joins."""
        for experts, ep in ((32, 4), (384, 64)):
            with self.subTest(ep=ep):
                spec = MegaMoeSpec(512, 5120, 1792, experts, 8, None, 4096, ep, None, 0, 32,
                                   dispatch_mode="pull", forward_ingress_mode="indexed")
                old = _build_runtime_artifacts(spec)
                new = _build_runtime_artifacts(replace(spec, backward_ingress_mode="indexed"))
                self.assertEqual(bytes(old[1]), bytes(new[1]))
                old_tasks = [task for task in old[3].all_tasks[:old[3].task_num]
                             if task.task_type != TaskType.TASK_TERMINATE]
                routed = [task for task in new[3].all_tasks[:new[3].task_num]
                          if task.task_type not in (TaskType.TASK_TERMINATE, TaskType.TASK_MOE_ROUTE_GRAD)]
                self.assertEqual(len(routed), len(old_tasks))
                for before, after in zip(old_tasks, routed):
                    self.assertEqual((before.dependent_event, before.trigger_event, before.task_index),
                                     (after.dependent_event, after.trigger_event, after.task_index))
                    if after.task_type == TaskType.TASK_SHMEM_GET_MEM:
                        self.assertEqual((after.extra_value_0, after.extra_value_1), (25, 8))
                extra = [task for task in new[3].all_tasks[:new[3].task_num]
                         if task.task_type == TaskType.TASK_MOE_ROUTE_GRAD]
                self.assertEqual(len(extra), 32)
                self.assertTrue(all(task.trigger_event == 0 for task in extra))
                self.assertTrue(all(task.dependent_event == EVENT_INVALID_ID for task in extra))
                self.assertEqual(list(old[3].all_event_num_triggers), list(new[3].all_event_num_triggers))

    def test_publication_keeps_weight_rounding_and_inverse_mapping(self) -> None:
        """Publish scalar weights and original dY without invoking expanded unpermutation."""
        source = torch.empty(3, 32)
        metadata = torch.zeros(46, 4, dtype=torch.int32)
        gradient = torch.randn(3, 32)
        expert_output = torch.randn(6, 32)
        probabilities = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
        order = torch.tensor([1, 2, 3, 5, 0, 4], dtype=torch.int32)
        inverse = torch.tensor([4, 0, 1, 2, 5, 3], dtype=torch.int32)
        ctx = SimpleNamespace(
            plan=SimpleNamespace(spec=SimpleNamespace(dispatch_mode='pull', backward_ingress_mode='indexed')),
            workspace=SimpleNamespace(source_buffer=source, backward_route_metadata=metadata),
            has_unpermute=True, token_rows=3, source_rows=6, topk_dtype=torch.bfloat16,
            needs_input_grad=(True,) * 11,
        )
        with patch.object(moe_function.multicore_ops, 'mega_moe_unpermute_grad_out') as expanded:
            published, grad_probabilities = moe_function._stage_backward_source(
                ctx, gradient, (inverse, expert_output, probabilities, order),
            )
        expanded.assert_not_called()
        self.assertEqual(published.data_ptr(), source.data_ptr())
        torch.testing.assert_close(published, gradient)
        records = metadata[40:]
        torch.testing.assert_close(records[:, 0], order)
        torch.testing.assert_close(records[:, 1], probabilities.flatten()[order.long()].view(torch.int32))
        torch.testing.assert_close(records[:, 2], inverse)
        self.assertEqual(grad_probabilities.dtype, torch.float32)
        self.assertEqual(tuple(grad_probabilities.shape), (3, 2))

    def test_unsupported_backward_modes_fail_before_parameter_allocation(self) -> None:
        """Reject incomplete ingress pairs and unsupported mode strings before device work."""
        for forward, backward in (('legacy', 'indexed'), ('indexed', 'unknown'), ('peer_dedup', 'indexed')):
            with self.subTest(forward=forward, backward=backward), self.assertRaises(ValueError):
                MegaMoeExperts(hidden_size=5120, intermediate_size=1792,
                               num_experts=32, top_k=8, dispatch_mode='pull',
                               forward_ingress_mode=forward, backward_ingress_mode=backward)

    def test_shared_directions_preserve_receive_reuse_and_route_gradient_coverage(self) -> None:
        """Keep the hybrid shared gradient graph independent of the new local dot tasks."""
        for shared_forward in (False, True):
            for shared_backward in (False, True):
                with self.subTest(forward=shared_forward, backward=shared_backward):
                    spec = MegaMoeSpec(512, 5120, 1792, 32, 8, None, 4096, 4, None, 0, 32,
                                       dispatch_mode='pull', forward_ingress_mode='indexed',
                                       backward_ingress_mode='indexed', shared_intermediate_size=2048,
                                       forward_shared_expert=shared_forward, backward_shared_expert=shared_backward)
                    config = _build_runtime_artifacts(spec)[3]
                    self.assertTrue(can_reuse_backward_dispatch(config, 8, 32))
                    route_tasks = [task for task in config.all_tasks[:config.task_num]
                                   if task.task_type == TaskType.TASK_MOE_ROUTE_GRAD]
                    self.assertEqual(len(route_tasks), 32)
                    self.assertEqual({task.task_index for task in route_tasks}, set(range(32)))

    def test_publication_owns_routes_and_can_omit_probability_gradient_storage(self) -> None:
        """Metadata reuse must not mutate another invocation's saved scalar route."""
        workspace = SimpleNamespace(source_buffer=torch.empty(2, 32),
                                    backward_route_metadata=torch.zeros(44, 4, dtype=torch.int32))
        order = torch.tensor([2, 0, 3, 1], dtype=torch.int32)
        inverse = torch.tensor([1, 3, 0, 2], dtype=torch.int32)
        original = order.clone()
        _, probabilities = publish_backward_route(
            workspace, torch.randn(2, 32), inverse, torch.randn(4, 32),
            torch.full((2, 2), 0.5), order, needs_probability_grad=False,
        )
        self.assertIsNone(probabilities)
        header = workspace.backward_route_metadata.view(torch.uint8).reshape(-1)[512:536].view(torch.int64)
        self.assertEqual((int(header[0]), int(header[2])), (0, 0))
        workspace.backward_route_metadata[40:, 0].zero_()
        torch.testing.assert_close(order, original)

    def test_indexed_backward_rejects_an_older_native_capability(self) -> None:
        """Do not allow a legacy device task switch to silently omit the new dot jobs."""
        native_ops.require_indexed_backward_support.cache_clear()
        try:
            with patch.object(native_ops, '_load_native'), patch.object(
                torch.ops.hyper_multicore, 'mega_moe_indexed_backward_version', return_value=0, create=True,
            ), self.assertRaisesRegex(RuntimeError, 'native version must be 1'):
                native_ops.require_indexed_backward_support()
        finally:
            native_ops.require_indexed_backward_support.cache_clear()
