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
"""Validate W1 task ordering without changing communication or early GEMMs."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from hyper_multicore.modules.mega_moe.backward.graph import build_backward_graph
from hyper_multicore.modules.mega_moe.backward.gen_runtime_data import build_config_for_rank
from hyper_multicore.modules.mega_moe.backward.storage import (
    can_reuse_backward_dispatch,
    join_w1_phase_once,
    prioritize_activation_gradient,
)
from hyper_multicore.modules.mega_moe.gradient_queue import (
    ADAPTIVE_GRADIENT_QUEUE,
    mark_active_gradient_lanes,
    mark_device_gradient_queue,
    require_device_gradient_native,
    reverse_w1_gradient_queue,
)
from hyper_multicore.modules.mega_moe.module import MegaMoeExperts, _MegaMoeExecutionResources
from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts, _resolve_gradient_queue
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.scheduler.config import (
    EVENT_INVALID_ID,
    TaskSplitValue,
    TaskType,
    validate_runtime_config,
)


class TestGradientQueue(unittest.TestCase):
    """Preserve descriptor identity, dependencies and per-core phase boundaries."""

    def test_act_first_joins_w2_before_overwriting_dispatch(self) -> None:
        """Earlier ActGrad releases SwiGLU while Gate still joins both receive readers."""
        for mode in ("push", "pull"):
            with self.subTest(mode=mode):
                topology = TaskSplitValue(tp=1, ep=4, seq_size=512, all_expert_num=12, top_k=2, dispatch_mode=mode)
                graph = build_backward_graph(topology, hidden_size=1536, intermediate_size=768)
                graph.propagate_splits(topology)
                config = build_config_for_rank(graph, topology, rank_id=1, num_cube_cores=24)
                queue = config.cube_task_indices
                counts = sum(config.all_event_num_triggers)
                vectors = bytes(config.vector_task_indices)
                self.assertTrue(prioritize_activation_gradient(config, 3, 24))
                validate_runtime_config(config, topology, 24)
                self.assertEqual(vectors, bytes(config.vector_task_indices))
                self.assertEqual(counts, sum(config.all_event_num_triggers))
                for expert in range(3):
                    act, weight = (config.all_tasks[queue[expert * 48 + offset]] for offset in (0, 24))
                    self.assertEqual(act.outputs[0].input_position, 8)
                    self.assertEqual(weight.outputs[0].input_position, 6)
                    gate = config.all_tasks[queue[144 + expert * 24]]
                    self.assertEqual(weight.trigger_event, gate.dependent_event)
                self.assertTrue(can_reuse_backward_dispatch(config, 3, 24, act_first=True))
                self.assertFalse(can_reuse_backward_dispatch(config, 3, 24))
                config.all_tasks[queue[24]].trigger_event = EVENT_INVALID_ID
                self.assertFalse(can_reuse_backward_dispatch(config, 3, 24, act_first=True))

    def test_act_first_leaves_an_unknown_completion_graph_unchanged(self) -> None:
        """The original global completion must contain precisely the counted producers."""
        topology = TaskSplitValue(tp=1, ep=4, seq_size=512, all_expert_num=12, top_k=2, dispatch_mode="push")
        graph = build_backward_graph(topology, hidden_size=1536, intermediate_size=768)
        graph.propagate_splits(topology)
        config = build_config_for_rank(graph, topology, rank_id=1, num_cube_cores=24)
        term = next(task for task in config.all_tasks[: config.task_num] if task.task_type == TaskType.TASK_TERMINATE)
        config.all_event_num_triggers[term.dependent_event] -= 1
        damaged = bytes(config)
        self.assertFalse(prioritize_activation_gradient(config, 3, 24))
        self.assertEqual(damaged, bytes(config))

    def test_active_lane_marks_reject_incomplete_waves(self) -> None:
        """A divergent Cube wave cannot share the same active-expert ordinal."""
        topology = TaskSplitValue(tp=1, ep=4, seq_size=384, all_expert_num=12, top_k=2, dispatch_mode="push")
        graph = build_backward_graph(topology, hidden_size=1536, intermediate_size=768)
        graph.propagate_splits(topology)
        config = build_config_for_rank(graph, topology, rank_id=0, num_cube_cores=24)
        queue = config.cube_task_indices
        index = next(
            i for i in range(config.task_index_num[0]) if config.all_tasks[queue[i]].outputs[0].input_position == 6
        )
        queue[index], queue[index + 1] = queue[index + 1], queue[index]
        invalid = bytes(config)
        self.assertFalse(mark_active_gradient_lanes(config, 3, 24))
        self.assertEqual(bytes(config), invalid)
        queue[index], queue[index + 1] = queue[index + 1], queue[index]
        original = bytes(config)
        self.assertTrue(mark_active_gradient_lanes(config, 3, 24))
        marked = [task for task in config.all_tasks if task.extra_value_0 == 0x5A900000]
        self.assertEqual(len(marked), 2 * 3 * 24)
        for task in marked:
            task.extra_value_0 = 0
        self.assertEqual(bytes(config), original)

    def test_join_once_device_modes_keep_only_the_first_wait(self) -> None:
        """Both native candidates require a complete owner-only Gate phase proof."""
        spec = MegaMoeSpec(
            local_num_tokens=512,
            hidden_size=5120,
            intermediate_size=1792,
            num_experts=24,
            top_k=8,
            initial_capacity_factor=1.5,
            receive_capacity=6144,
            ep_size=4,
            ep_group=None,
            rank_id=1,
            num_cube_cores=24,
        )
        _, forward, _, reference = _build_runtime_artifacts(spec)
        modes = {"device_w1_asc_join_once": 6, "device_w1_asc_join_rotate": 7}
        for name, marker in modes.items():
            with self.subTest(mode=name):
                _, new_forward, _, actual = _build_runtime_artifacts(replace(spec, _gradient_queue_order=name))
                cube = [actual.all_tasks[index] for index in actual.cube_task_indices[: actual.task_index_num[0]]]
                weights = [task for task in cube if task.outputs[0].input_position == 18]
                self.assertEqual(len(weights), 6 * 24)
                self.assertTrue(all((task.extra_value_0 >> 20) & 0xF == marker for task in weights))
                waits = [task for task in weights if task.dependent_event != EVENT_INVALID_ID]
                self.assertEqual(len(waits), 24)
                self.assertEqual(len({task.dependent_event for task in waits}), 1)
                w2_marked = [task for task in cube if task.extra_value_0 == 0x5A500000]
                self.assertEqual(len(w2_marked), 144 if marker == 7 else 0)
                self.assertTrue(all(task.outputs[0].input_position == 6 for task in w2_marked))
                self.assertEqual(bytes(new_forward), bytes(forward))
                self.assertEqual(bytes(actual.cube_task_indices), bytes(reference.cube_task_indices))
                self.assertEqual(bytes(actual.vector_task_indices), bytes(reference.vector_task_indices))

    def test_lane_markers_keep_queues_events_and_activation_tasks(self) -> None:
        """Virtual Cube lanes must not reorder experts or change event ownership."""
        topology = TaskSplitValue(tp=1, ep=4, seq_size=512, all_expert_num=24, top_k=2, dispatch_mode="push")
        graph = build_backward_graph(topology, hidden_size=5120, intermediate_size=1792)
        graph.propagate_splits(topology)
        config = build_config_for_rank(graph, topology, rank_id=1, num_cube_cores=24)
        before = bytes(config)

        mark_device_gradient_queue(config, graph, "device_lane_rotate")

        marked = [task for task in config.all_tasks if task.extra_value_0 & 0xFF000000 == 0x5A000000]
        self.assertEqual(len(marked), 288)
        self.assertEqual({task.outputs[0].input_position for task in marked}, {6, 18})
        for task in marked:
            self.assertEqual((task.extra_value_0 >> 20) & 0xF, 5)
            self.assertEqual(task.tiling_data_offset, 0)
            task.extra_value_0 = 0
        self.assertEqual(bytes(config), before)
        self.assertTrue(can_reuse_backward_dispatch(config, 6, 24))

    def test_adaptive_markers_keep_original_fallback_descriptors(self) -> None:
        """An unselected rank must keep the original dependencies in both modes."""
        spec = MegaMoeSpec(
            local_num_tokens=512,
            hidden_size=5120,
            intermediate_size=1792,
            num_experts=24,
            top_k=8,
            initial_capacity_factor=1.5,
            receive_capacity=6144,
            ep_size=4,
            ep_group=None,
            rank_id=1,
            num_cube_cores=24,
        )
        for mode in ("push", "pull"):
            with self.subTest(mode=mode):
                base = replace(spec, dispatch_mode=mode)
                _, forward, _, reference = _build_runtime_artifacts(base)
                _, marked_forward, _, actual = _build_runtime_artifacts(
                    replace(base, _gradient_queue_order=ADAPTIVE_GRADIENT_QUEUE)
                )
                marked = [task for task in actual.all_tasks if task.extra_value_0 >> 20 == 0x5A8]
                self.assertEqual(len(marked), 288)
                self.assertEqual({task.outputs[0].input_position for task in marked}, {6, 18})
                self.assertEqual({task.inputs[2].input_position for task in marked}, {19})
                for task in marked:
                    task.extra_value_0 = task.tiling_data_offset = 0
                self.assertEqual(bytes(actual), bytes(reference))
                self.assertEqual(bytes(marked_forward), bytes(forward))

    def test_auto_plan_cache_reuses_a_bucket_without_a_token_whitelist(self) -> None:
        """A warmed bucket can select the same native policy for another length."""
        resources = _MegaMoeExecutionResources.__new__(_MegaMoeExecutionResources)
        resources._token_capacity = 4096
        resources._plans = {}
        resources.spec = MegaMoeSpec(
            local_num_tokens=128,
            hidden_size=4096,
            intermediate_size=2304,
            num_experts=32,
            top_k=4,
            initial_capacity_factor=1.25,
            receive_capacity=40960,
            ep_size=8,
            ep_group=None,
            rank_id=0,
            num_cube_cores=20,
            auto_tokens=True,
            _gradient_queue_order="auto",
            expert_weight_layout="EOI",
            dispatch_split=128,
            combine_split=256,
        )
        module = "hyper_multicore.modules.mega_moe"
        with (
            patch(module + ".plan.torch.device", return_value=SimpleNamespace(type="npu")),
            patch(module + ".plan.has_adaptive_gradient_native") as abi,
            patch(
                module + ".module.build_mega_moe_plan",
                side_effect=lambda spec, device: SimpleNamespace(spec=_resolve_gradient_queue(spec, device)),
            ),
        ):
            abi.return_value = False
            self.assertEqual(
                _resolve_gradient_queue(replace(resources.spec, schedule_tokens=896), "npu")._gradient_queue_order,
                "original",
            )
            abi.return_value = True
            large = resources.plan_for_tokens(896, "npu")
            self.assertEqual(large.spec._gradient_queue_order, "device_active_lanes_act_first")
            self.assertIs(resources.plan_for_tokens(768, "npu"), large)
            self.assertEqual(abi.call_count, 2)

    def test_auto_requires_physical_marker_support_before_changing_replica_lanes(self) -> None:
        """Old adapters may rotate ordinary lanes but cannot interpret physical markers."""
        spec = MegaMoeSpec(
            local_num_tokens=128,
            hidden_size=1536,
            intermediate_size=768,
            num_experts=16,
            top_k=2,
            initial_capacity_factor=None,
            receive_capacity=2048,
            ep_size=4,
            ep_group=None,
            rank_id=0,
            num_cube_cores=24,
            _gradient_queue_order="auto",
        )
        module = "hyper_multicore.modules.mega_moe"
        with (
            patch(module + ".plan.torch.device", return_value=SimpleNamespace(type="npu")),
            patch(module + ".plan.has_adaptive_gradient_native", side_effect=lambda version: version <= 3) as abi,
        ):
            self.assertEqual(
                _resolve_gradient_queue(spec, "npu")._gradient_queue_order, "device_active_lanes_act_first"
            )
            physical = replace(spec, replica_slots_per_rank=2, num_experts=24)
            self.assertEqual(_resolve_gradient_queue(physical, "npu")._gradient_queue_order, "original")
            self.assertEqual([call.args[0] for call in abi.call_args_list], [3, 4])
            cross_node = replace(spec, _gradient_queue_order="auto_w1")
            self.assertEqual(_resolve_gradient_queue(cross_node, "npu")._gradient_queue_order, "device_active_w1_lanes")

    def test_w1_join_once_keeps_queue_and_uses_full_gate_event(self) -> None:
        """Every Cube joins before its first W1; later W1 tasks need no extra wait."""
        topology = TaskSplitValue(tp=1, ep=4, seq_size=512, all_expert_num=24, top_k=2, dispatch_mode="push")
        graph = build_backward_graph(topology, hidden_size=5120, intermediate_size=1792)
        graph.propagate_splits(topology)
        config = build_config_for_rank(graph, topology, rank_id=1, num_cube_cores=24)
        tasks = [config.all_tasks[index] for index in config.cube_task_indices[: config.task_index_num[0]]]
        weights = [task for task in tasks if task.outputs[0].input_position == 18]
        original_deps = [task.dependent_event for task in weights]
        full_gate = original_deps[-1]
        config.all_event_num_triggers[full_gate] -= 1
        before = bytes(config)

        self.assertFalse(join_w1_phase_once(config, 6, 24))
        self.assertEqual(bytes(config), before)

        config.all_event_num_triggers[full_gate] += 1
        before = bytes(config)
        self.assertTrue(join_w1_phase_once(config, 6, 24))
        for index, task in enumerate(weights):
            self.assertEqual(task.dependent_event, full_gate if index < 24 else EVENT_INVALID_ID)
            task.dependent_event = original_deps[index]
        self.assertEqual(bytes(config), before)
        self.assertTrue(join_w1_phase_once(config, 6, 24))
        self.assertTrue(can_reuse_backward_dispatch(config, 6, 24))
        validate_runtime_config(config, topology, 24)

    def test_device_markers_preserve_host_queues_and_storage_reuse(self) -> None:
        """The device policy changes only GMM marker and opposite-phase metadata fields."""
        topology = TaskSplitValue(tp=1, ep=4, seq_size=512, all_expert_num=24, top_k=2, dispatch_mode="push")
        for mode in ("device_w1_asc", "device_w1_desc", "device_w1_asc_join", "device_late_auto"):
            with self.subTest(mode=mode):
                graph = build_backward_graph(topology, hidden_size=5120, intermediate_size=1792)
                graph.propagate_splits(topology)
                config = build_config_for_rank(graph, topology, rank_id=1, num_cube_cores=24)
                cube, vector = bytes(config.cube_task_indices), bytes(config.vector_task_indices)
                before = bytes(config)
                mark_device_gradient_queue(config, graph, mode)
                marked = [t for t in config.all_tasks if t.extra_value_0 & 0xFF000000 == 0x5A000000]
                self.assertEqual(len(marked), 288 if mode == "device_late_auto" else 144)
                self.assertTrue(all(t.outputs[0].input_position in (12, 18) for t in marked))
                self.assertEqual(bytes(config.cube_task_indices), cube)
                self.assertEqual(bytes(config.vector_task_indices), vector)
                self.assertTrue(can_reuse_backward_dispatch(config, 6, 24))
                for task in marked:
                    task.extra_value_0 = task.tiling_data_offset = 0
                self.assertEqual(bytes(config), before)

    def test_device_policy_requires_native_capability(self) -> None:
        """Old adapters must fail instead of silently treating marked tasks as ordinary."""
        with patch("torch.ops.hyper_multicore.mega_moe_schedule_version", create=True) as version:
            with patch("hyper_multicore.modules.mega_moe.gradient_queue.importlib.import_module"):
                version.return_value = 0
                with self.assertRaisesRegex(RuntimeError, "matching rebuilt"):
                    require_device_gradient_native()
                version.return_value = 1
                require_device_gradient_native()
                with self.assertRaisesRegex(RuntimeError, "matching rebuilt"):
                    require_device_gradient_native("device_lane_rotate")
                version.return_value = 2
                require_device_gradient_native("device_lane_rotate")
                with self.assertRaisesRegex(RuntimeError, "matching rebuilt"):
                    require_device_gradient_native("device_w1_asc_join_once")
                version.return_value = 3
                require_device_gradient_native("device_w1_asc_join_once")
                require_device_gradient_native("device_w1_asc_join_rotate")

    def test_w1_reversal_keeps_other_tasks_and_storage_reuse(self) -> None:
        """Reversing W1 alone must preserve communication, readers and core lanes."""
        for mode in ("push", "pull"):
            for local_experts in (1, 2, 6):
                with self.subTest(mode=mode, local_experts=local_experts):
                    topology = TaskSplitValue(
                        tp=1, ep=4, seq_size=512, all_expert_num=4 * local_experts, top_k=2, dispatch_mode=mode
                    )
                    graph = build_backward_graph(topology, hidden_size=5120, intermediate_size=1792)
                    graph.propagate_splits(topology)
                    config = build_config_for_rank(graph, topology, rank_id=1, num_cube_cores=24)
                    cube = list(config.cube_task_indices)
                    vector = bytes(config.vector_task_indices)
                    descriptors = bytes(config.all_tasks)
                    events = bytes(config.all_event_num_triggers)
                    original_reuse = can_reuse_backward_dispatch(config, local_experts, 24)
                    task_to_stage = {}
                    start = 0
                    for operation in graph.topological_sort():
                        task_to_stage.update({i: operation.name for i in range(start, start + operation.task_num)})
                        start += operation.task_num
                    reverse_w1_gradient_queue(config, graph, topology)
                    for index in range(config.task_index_num[0]):
                        before, after = cube[index], config.cube_task_indices[index]
                        stage = task_to_stage[before]
                        self.assertEqual(task_to_stage[after], stage)
                        if stage == "w1_grad":
                            old_task, new_task = config.all_tasks[before], config.all_tasks[after]
                            self.assertEqual(new_task.task_index // 24, local_experts - 1 - old_task.task_index // 24)
                            self.assertEqual(new_task.task_index % 24, old_task.task_index % 24)
                        else:
                            self.assertEqual(after, before)
                    self.assertEqual(sorted(cube), sorted(config.cube_task_indices))
                    self.assertEqual(bytes(config.vector_task_indices), vector)
                    self.assertEqual(bytes(config.all_tasks), descriptors)
                    self.assertEqual(bytes(config.all_event_num_triggers), events)
                    self.assertTrue(original_reuse)
                    self.assertEqual(can_reuse_backward_dispatch(config, local_experts, 24), original_reuse)
                    validate_runtime_config(config, topology, 24)
                    reverse_w1_gradient_queue(config, graph, topology)
                    self.assertEqual(list(config.cube_task_indices), cube)

    def test_default_queue_does_not_filter_ep_or_replica_budget_at_construction(self) -> None:
        """The automatic policy is the public default for arbitrary geometries."""
        cases = [(1, 0), (8, 0), (32, 0), (64, 0), (16, 2)]
        for ep_size, budget in cases:
            with self.subTest(ep_size=ep_size, budget=budget), patch.dict("os.environ", {}, clear=True):
                module = MegaMoeExperts(
                    hidden_size=1536,
                    intermediate_size=768,
                    top_k=2,
                    num_experts=4 * ep_size,
                    ep_size=ep_size,
                    replica_slots_per_rank=budget,
                    create_parameters=False,
                    expert_weight_layout="EIO" if budget else "EOI",
                    dispatch_split=256,
                    combine_split=128,
                )
                self.addCleanup(module.close)
                self.assertEqual(module._resource_group.specification["_gradient_queue_order"], "auto")

    def test_order_is_part_of_resource_identity(self) -> None:
        """Differently ordered layers must not share an execution resource group."""
        options = {"hidden_size": 16, "intermediate_size": 8, "num_experts": 4, "top_k": 2, "ep_size": 2}
        with patch.dict("os.environ", {"HYPER_MULTICORE_MOE_GRADIENT_QUEUE_ORDER": "original"}):
            original = MegaMoeExperts(**options)
        with patch.dict("os.environ", {"HYPER_MULTICORE_MOE_GRADIENT_QUEUE_ORDER": "w1_reverse"}):
            candidate = MegaMoeExperts(**options)
        self.addCleanup(original.close)
        self.addCleanup(candidate.close)
        self.assertNotEqual(original._resource_group.compatibility_key, candidate._resource_group.compatibility_key)

    def test_plan_changes_only_ordinary_backward(self) -> None:
        """Forward and actual guest schedules must keep their original images."""
        for slots in (0, 2):
            with self.subTest(slots=slots):
                spec = MegaMoeSpec(
                    local_num_tokens=512,
                    hidden_size=5120,
                    intermediate_size=1792,
                    num_experts=24 + 4 * slots,
                    top_k=8,
                    initial_capacity_factor=1.5,
                    receive_capacity=6144,
                    ep_size=4,
                    ep_group=None,
                    rank_id=1,
                    num_cube_cores=24,
                    replica_slots_per_rank=slots,
                    logical_num_experts=24,
                )
                _, forward, _, backward = _build_runtime_artifacts(spec)
                _, new_forward, _, new_backward = _build_runtime_artifacts(
                    replace(spec, _gradient_queue_order="w1_reverse")
                )
                self.assertEqual(bytes(forward), bytes(new_forward))
                if slots:
                    self.assertEqual(bytes(backward), bytes(new_backward))
                else:
                    self.assertNotEqual(bytes(backward), bytes(new_backward))
                    self.assertEqual(bytes(backward.vector_task_indices), bytes(new_backward.vector_task_indices))
                    self.assertEqual(bytes(backward.all_tasks), bytes(new_backward.all_tasks))
