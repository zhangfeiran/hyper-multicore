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
"""CPU contracts for shared tasks alongside the current Ascend950 routed plan."""

from dataclasses import replace
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_multicore.modules.mega_moe.plan import build_mega_moe_plan
from hyper_multicore.modules.mega_moe import function as moe_function
from hyper_multicore.modules.mega_moe import shared as moe_shared
from hyper_multicore.diagnostics import api as diagnostic_api
from hyper_multicore.diagnostics.common.call import DiagnosticCall
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.modules.mega_moe.backward.storage import can_reuse_backward_dispatch
from hyper_multicore.modules.mega_moe.shared_schedule import bind_shared_diagnostic_graph
from hyper_multicore.diagnostics.common.graph import validate_mega_kernel_diagnostic_contract
from hyper_multicore.scheduler.config import EVENT_INVALID_ID, TaskType


class TestA5SharedFusion(unittest.TestCase):
    """Keep routed scheduling and local shared ownership independently verifiable."""

    def test_a5_shared_backward_consumes_native_activation_gradients(self) -> None:
        """Keep A5 activation rounding outside the three fused shared gradient GEMMs."""
        for cores in (24, 32):
            with self.subTest(cores=cores):
                spec = MegaMoeSpec(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                                   num_experts=32, top_k=8, initial_capacity_factor=None,
                                   receive_capacity=32768, ep_size=4, ep_group=None, rank_id=0,
                                   num_cube_cores=cores, dispatch_mode="pull", shared_intermediate_size=2048,
                                   backward_shared_expert=True)
                graph, config = _build_runtime_artifacts(spec)[2:]
                names = {op.name for op in graph.topological_sort() if op.name.startswith("shared_")}
                expected = {"shared_w2_grad", "shared_gate_grad", "shared_w1_grad"}
                if cores == 24:
                    expected |= {"shared_act_grad", "shared_swiglu_grad"}
                self.assertEqual(names, expected)
                if cores == 32:
                    shared = [task for task in config.all_tasks[:config.task_num]
                              if task.inputs[0].tensor_type == 2]
                    self.assertEqual(len(shared), 3 * cores)
                    self.assertTrue(all(task.dependent_event == EVENT_INVALID_ID for task in shared))

    def test_native_shared_activation_buffers_keep_owned_results(self) -> None:
        """Pass native producer storage directly to the unchanged indirect pointer ABI."""
        dy = torch.randn(8, 4)
        source, first, second = torch.randn(8, 4), torch.randn(6, 4), torch.randn(4, 3)
        up, activation = torch.randn(8, 6), torch.randn(8, 3)
        derivative = torch.randn_like(up)
        with (
            patch.object(moe_shared, "pointer_table", side_effect=lambda tensors, **_: tensors),
            patch.object(torch.ops.npu, "npu_swiglu_backward", return_value=derivative) as native,
        ):
            tensors, pointers = moe_shared.shared_backward_buffers(
                dy, (source, first, second, up, activation), native_activation_grad=True,
            )
        native.assert_called_once()
        torch.testing.assert_close(tensors[4], dy @ second)
        self.assertIs(tensors[6], derivative)
        self.assertIs(pointers, tensors)
        self.assertEqual(tuple(tensors[10].shape), (4, 6))

    def test_independent_shared_switches_keep_abi_and_routed_storage(self) -> None:
        """Feature: Independent shared kernel directions.
        Description: Build EP4 and EP64 graphs with both shared directions and placements.
        Expectation: Preserve routed descriptors, native parameter counts and receive-buffer reuse.
        """
        for experts, ep in ((32, 4), (384, 64)):
            spec = MegaMoeSpec(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                               num_experts=experts, top_k=8, initial_capacity_factor=None,
                               receive_capacity=32768, ep_size=ep, ep_group=None, rank_id=0,
                               num_cube_cores=32, dispatch_mode="pull", forward_ingress_mode="indexed")
            with patch.dict(os.environ, {"HYPER_PARALLEL_SHMEM_DATA_ENGINE": "mte_udma",
                                         "HYPER_PARALLEL_SHMEM_UDMA_QP_NUM": "32"}):
                baseline = _build_runtime_artifacts(spec)
                for forward in (False, True):
                    for backward in (False, True):
                        for placement in ("idle", "front"):
                            with self.subTest(ep=ep, forward=forward, backward=backward, placement=placement):
                                current = replace(spec, shared_intermediate_size=2048, shared_placement=placement,
                                                  forward_shared_expert=forward, backward_shared_expert=backward)
                                graphs = _build_runtime_artifacts(current)
                                for graph, config, enabled, ports, original in (
                                    (graphs[0], graphs[1], forward, 28 if forward else 17, baseline[1]),
                                    (graphs[2], graphs[3], backward, 35 if backward else 20, baseline[3]),
                                ):
                                    shared = [task for task in config.all_tasks[:config.task_num]
                                              if task.inputs[0].tensor_type == 2]
                                    self.assertEqual(bool(shared), enabled)
                                    diagnostic = bind_shared_diagnostic_graph(graph, config)
                                    validate_mega_kernel_diagnostic_contract(diagnostic, config, ports)
                                    if not enabled:
                                        self.assertEqual(bytes(config), bytes(original))
                                self.assertTrue(can_reuse_backward_dispatch(graphs[3], spec.local_experts, 32))

    def test_shared_backward_keeps_unknown_reader_and_event_rejections(self) -> None:
        """Feature: Routed receive-buffer lifetime proof.
        Description: Add an unknown reader or an inconsistent event to a shared backward graph.
        Expectation: Disable buffer reuse for either invalid schedule.
        """
        spec = MegaMoeSpec(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                           num_experts=32, top_k=8, initial_capacity_factor=None,
                           receive_capacity=32768, ep_size=4, ep_group=None, rank_id=0,
                           num_cube_cores=32, dispatch_mode="pull", forward_ingress_mode="indexed",
                           shared_intermediate_size=2048, backward_shared_expert=True)
        config = _build_runtime_artifacts(spec)[3]
        self.assertTrue(can_reuse_backward_dispatch(config, 8, 32))
        shared = next(task for task in config.all_tasks[:config.task_num] if task.inputs[0].tensor_type == 2)
        old_port = shared.inputs[0].input_position
        shared.inputs[0].input_position = 0
        self.assertFalse(can_reuse_backward_dispatch(config, 8, 32))
        shared.inputs[0].input_position = old_port
        activation = next(task for task in config.all_tasks[:config.task_num]
                          if task.outputs[0].input_position == 8)
        config.all_event_num_triggers[activation.trigger_event] += 1
        self.assertFalse(can_reuse_backward_dispatch(config, 8, 32))

    def test_native_shared_backward_uses_the_routed_profile_binding(self) -> None:
        """Run the real backward launch boundary with an active CPU-only diagnostic session."""
        spec = MegaMoeSpec(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                           num_experts=32, top_k=8, initial_capacity_factor=None,
                           receive_capacity=32768, ep_size=4, ep_group=None, rank_id=0,
                           num_cube_cores=32, dispatch_mode="pull", forward_ingress_mode="indexed",
                           shared_intermediate_size=2048)
        plan = build_mega_moe_plan(spec, "cpu")
        tensor = torch.zeros(1)
        saved = SimpleNamespace(**{key: tensor for key in (
            "dispatch_target_off", "dispatch_src_off", "dispatch_size", "activation", "weight2", "up_proj",
            "weight1", "combine_target_off", "combine_src_off", "combine_size", "dispatch", "group_list",
        )})
        execution = moe_function._BackwardExecution(
            dispatch=tensor, grad_x=tensor, event_counters=tensor,
            gmm_workspace=tensor, swiglu_workspace=tensor,
            intermediates=moe_function._BackwardIntermediates(*(tensor for _ in range(5))),
        )
        session = Mock()
        session.is_capture_enabled.return_value = True
        session.prepare_call.side_effect = lambda context: DiagnosticCall(context.ordinary_arguments())
        with (
            patch.object(diagnostic_api, "_get_active_session", return_value=session),
            patch.object(moe_function.multicore_ops, "mega_moe_grad_with_diagnostic_buffer") as native,
        ):
            moe_function._launch_backward_kernel(plan, saved, tensor, execution)
        session.prepare_call.assert_called_once()
        native.assert_called_once()

    def test_shared_width_2048_keeps_current_routed_task_contracts(self) -> None:
        """Strip local shared tasks and recover every original routed descriptor's semantics."""
        spec = MegaMoeSpec(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                           num_experts=32, top_k=8, initial_capacity_factor=None,
                           receive_capacity=32768, ep_size=4, ep_group=None, rank_id=0,
                           num_cube_cores=32, dispatch_mode="pull", forward_ingress_mode="indexed")
        with patch.dict(os.environ, {"HYPER_PARALLEL_SHMEM_DATA_ENGINE": "mte_udma",
                                     "HYPER_PARALLEL_SHMEM_UDMA_QP_NUM": "32"}):
            baseline = _build_runtime_artifacts(spec)
            for placement in ("idle", "front"):
                with self.subTest(placement=placement):
                    fused = _build_runtime_artifacts(replace(spec, shared_intermediate_size=2048,
                                                           shared_placement=placement))
                    for graph, config, ports in ((fused[0], fused[1], 28), (fused[2], fused[3], 35)):
                        diagnostic = bind_shared_diagnostic_graph(graph, config)
                        validate_mega_kernel_diagnostic_contract(diagnostic, config, ports)
                    for graph, config, original in ((fused[0], fused[1], baseline[1]),
                                                     (fused[2], fused[3], baseline[3])):
                        shared = [task for task in config.all_tasks[:config.task_num]
                                  if task.inputs[0].tensor_type == 2]
                        if graph is fused[2]:
                            self.assertFalse(shared)
                            self.assertEqual(bytes(config), bytes(original))
                        else:
                            self.assertTrue(shared)
                            self.assertTrue(any(op.name.startswith("shared_") for op in graph.topological_sort()))
                        for queue_name, index in (("cube_task_indices", 0), ("vector_task_indices", 1)):
                            actual = [config.all_tasks[i] for i in getattr(config, queue_name)[:config.task_index_num[index]]
                                      if config.all_tasks[i].inputs[0].tensor_type != 2]
                            expected = [original.all_tasks[i] for i in getattr(original, queue_name)[:original.task_index_num[index]]]
                            signature = lambda task: (task.task_type, task.task_index, task.task_split_num,
                                                      task.task_split_value, task.dependent_event, task.trigger_event,
                                                      task.extra_value_0, task.extra_value_1, task.extra_value_2)
                            self.assertEqual([signature(task) for task in actual], [signature(task) for task in expected])
                        routed_gmm = [task for task in config.all_tasks[:config.task_num]
                                      if task.task_type == TaskType.TASK_GROUPED_MATMUL and task.inputs[0].tensor_type != 2]
                        self.assertTrue(all(task.extra_value_2 == 0x41350002 for task in routed_gmm)
                                        if graph is fused[0] else bool(routed_gmm))


if __name__ == "__main__":
    unittest.main()
