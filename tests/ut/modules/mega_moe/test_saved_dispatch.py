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
"""Prove Combine copy coverage and preserve all scheduler completion contracts."""

import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from hyper_multicore.modules.mega_moe import saved_dispatch, plan as plan_module
from hyper_multicore.modules.mega_moe.configuration import execution_options
from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_multicore.modules.mega_moe.route import _numpy_route_metadata
from hyper_multicore.modules.mega_moe.saved_dispatch import build_saved_dispatch, resolve_saved_dispatch
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.scheduler.config import TaskType
from hyper_multicore.scheduler.runtime import serialize_runtime_config
from tests.ut.modules.mega_moe.saved_dispatch_simulator import simulate


def _spec(budget=0, compact=False):
    return MegaMoeSpec(
        4096, 128, 128, 4 * (2 + budget), 2, 1.25, 16384, 4, None, 0, 4,
        _compact_schedule=compact, _compact_communication=compact,
    )


def _combines(config):
    return [task_id for task_id in config.vector_task_indices[:config.task_index_num[1]]
            if config.all_tasks[task_id].task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
            and config.all_tasks[task_id].inputs[3].input_position == 15]


class TestSavedDispatch(unittest.TestCase):
    """Protect saved-X ownership independently of floating-point expert numerics."""

    def test_option_and_native_capability_fall_back(self):
        """Old payloads, CPU and unproven research modes retain the separate copy."""
        key = "HYPER_MULTICORE_MOE_FUSED_DISPATCH_SAVE"
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(execution_options("push", 6)["_fuse_saved_dispatch"])
        with patch.dict(os.environ, {key: "1"}, clear=True):
            self.assertTrue(execution_options("push", 6)["_fuse_saved_dispatch"])
            self.assertFalse(execution_options("pull", 6)["_fuse_saved_dispatch"])
        with patch.dict(os.environ, {key: "invalid"}, clear=True):
            with self.assertRaisesRegex(ValueError, "FUSED_DISPATCH_SAVE"):
                execution_options("push", 6)
        spec = replace(_spec(), _fuse_saved_dispatch=True)
        self.assertFalse(resolve_saved_dispatch(spec, "cpu")._fuse_saved_dispatch)
        for version, enabled in ((None, False), (0, False), (1, False), (2, False), (3, True), (4, False)):
            namespace = SimpleNamespace()
            if version is not None:
                namespace.mega_moe_saved_dispatch_version = lambda value=version: value
            with patch.object(saved_dispatch.torch, "device", return_value=SimpleNamespace(type="npu")), \
                 patch.object(saved_dispatch.torch.ops, "hyper_multicore", namespace), \
                 patch.object(saved_dispatch.multicore_ops, "_load_native"):
                self.assertEqual(resolve_saved_dispatch(spec, "cpu")._fuse_saved_dispatch, enabled)
                for changes in (
                    {"dispatch_mode": "pull"}, {"batch_experts": True}, {"hidden_size": 127},
                    {"apply_probs_before_fc2": True}, {"swiglu_limit": 7.0}, {"clamp_after_activation": True},
                ):
                    self.assertFalse(resolve_saved_dispatch(replace(spec, **changes), "cpu")._fuse_saved_dispatch)

    def test_copy_coverage_and_read_before_write_fences(self):
        """Owned X partitions cover all rows and GMM2 follows every X reader/save."""
        rng = np.random.default_rng(20261114)
        for budget in (0, 1, 6):
            for compact in (False, True):
                spec = _spec(budget, compact)
                original = _build_runtime_artifacts(spec)[1]
                before = serialize_runtime_config(original)
                changed = build_saved_dispatch(original, spec)
                self.assertEqual(serialize_runtime_config(original), before)
                counts = rng.integers(0, 258, size=(4, spec.num_experts), dtype=np.int32)
                counts[:, 1::3] = 0
                route = _numpy_route_metadata(counts, spec, 16384, "cpu")
                covered = np.zeros(int(route.group_list[-1]), dtype=np.uint8)
                saves = [changed.all_tasks[index] for index in changed.vector_task_indices[:changed.task_index_num[1]]
                         if changed.all_tasks[index].task_type == TaskType.TASK_SAVE_DISPATCH]
                self.assertEqual(len(saves), spec.local_experts * 8)
                for task in saves:
                    expert, lane = divmod(task.task_index, 8)
                    begin = 0 if expert == 0 else route.group_list[expert - 1].item()
                    rows = route.group_list[expert].item() - begin
                    split = (rows + 7) // 8
                    covered[begin + min(lane * split, rows):begin + min((lane + 1) * split, rows)] += 1
                    self.assertEqual(changed.all_event_num_triggers[task.trigger_event], 12)
                np.testing.assert_array_equal(covered, np.ones_like(covered))
                for task_id in _combines(changed):
                    self.assertEqual(changed.all_tasks[task_id].inputs[1].input_position, 0)
                for task_id in changed.cube_task_indices[:changed.task_index_num[0]]:
                    task = changed.all_tasks[task_id]
                    if task.inputs[0].input_position == 8:
                        self.assertEqual(task.outputs[0].input_position, 0)
                old_positions = {task: i for i, task in
                                 enumerate(original.vector_task_indices[:original.task_index_num[1]])}
                for i, task in enumerate(changed.vector_task_indices[:changed.task_index_num[1]]):
                    if task in old_positions and task != original.task_num:
                        self.assertEqual(i % 8, old_positions[task] % 8)

    def test_malformed_graph_is_rejected_without_mutation(self):
        """Reject missing producers, a bypassed join and an unexpected Y reader."""
        spec = _spec()
        for mutation in ("threshold", "reader", "missing"):
            config = _build_runtime_artifacts(spec)[1]
            task = config.all_tasks[_combines(config)[0]]
            if mutation == "reader":
                task.inputs[1].input_position = 7
            else:
                up = next(config.all_tasks[index] for index in config.cube_task_indices[:config.task_index_num[0]]
                          if config.all_tasks[index].outputs[0].input_position == 7)
                if mutation == "threshold":
                    config.all_event_num_triggers[up.trigger_event] -= 1
                else:
                    up.task_index += 1
            before = serialize_runtime_config(config)
            with self.assertRaisesRegex(ValueError, "saved-dispatch"):
                build_saved_dispatch(config, spec)
            self.assertEqual(serialize_runtime_config(config), before)

    def test_distributed_progress_and_guest_credit_pruning(self):
        """Both prefix strategies retain all X readers before overwrite on every rank."""
        for ep, budget, compact, nominal in (
            (4, 0, False, False), (4, 1, False, False), (4, 6, False, False),
            (4, 0, True, False), (4, 1, True, False), (4, 6, True, False),
            (4, 6, True, True), (32, 0, True, False), (32, 1, True, True), (32, 6, True, False),
        ):
            with self.subTest(ep=ep, budget=budget, compact=compact, nominal=nominal):
                spec = replace(_spec(budget, compact), local_num_tokens=512, ep_size=ep,
                               num_experts=ep * (2 + budget), nominal_schedule=nominal)
                configs = [build_saved_dispatch(_build_runtime_artifacts(replace(spec, rank_id=rank))[1], spec)
                           for rank in range(ep)]
                simulate(configs, ep, spec.local_experts, 4)

    def test_missing_join_credits_are_detected_by_delayed_simulator(self):
        """A false dependency proof must fail even when all task IDs remain present."""
        spec = replace(_spec(1, True), nominal_schedule=True)
        configs = [build_saved_dispatch(_build_runtime_artifacts(replace(spec, rank_id=rank))[1], spec)
                   for rank in range(4)]
        for config in configs:
            events = {config.all_tasks[index].trigger_event for index in
                      config.vector_task_indices[:config.task_index_num[1]]
                      if config.all_tasks[index].task_type == TaskType.TASK_SAVE_DISPATCH}
            for event in events:
                config.all_event_num_triggers[event] -= 8
        with self.assertRaisesRegex(AssertionError, "X readers/save still active"):
            simulate(configs, 4, spec.local_experts, 4)

    def test_plan_keeps_original_diagnostic_image(self):
        """Normal fused and original diagnostic images use independent physical layouts."""
        spec = replace(_spec(1, True), _fuse_saved_dispatch=True)
        with patch.object(plan_module, "resolve_saved_dispatch", side_effect=lambda value, device: value):
            plan = plan_module.build_mega_moe_plan(spec, "cpu")
        original_bytes = bytes(plan.fwd_runtime_config.tolist())
        fused_bytes = bytes(plan.fwd_saved_dispatch_config.tolist())
        self.assertNotEqual(original_bytes, fused_bytes)
        self.assertGreater(len(fused_bytes), len(original_bytes))
        self.assertIs(plan.fwd_diagnostics.runtime_config, plan.fwd_runtime_config)

    def test_full_shape_ep32_schedules(self):
        """The complete performance shape progresses with 24 Cube and 48 AIV lanes."""
        for compact in (False, True):
            spec = replace(_spec(6, compact), ep_size=32, num_experts=384,
                           hidden_size=5120, intermediate_size=1792, top_k=8, num_cube_cores=24,
                           receive_capacity=40960)
            configs = [build_saved_dispatch(_build_runtime_artifacts(replace(spec, rank_id=rank))[1], spec)
                       for rank in range(32)]
            simulate(configs, 32, 12, 24)

    def test_bounded_copy_lanes_preserve_exact_rows_and_all_join_credits(self):
        """Active copies are spread across workers while inactive lanes keep credits."""
        for requested in (0, 1, 2, 3, 4, 8, 24, 128):
            spec = replace(_spec(6, True), _saved_dispatch_lanes=requested)
            original = _build_runtime_artifacts(spec)[1]
            changed = build_saved_dispatch(original, spec)
            active = min(requested or 8, 8)
            selected = {index * 8 // active: index for index in range(active)}
            saves = [changed.all_tasks[index] for index in changed.vector_task_indices[:changed.task_index_num[1]]
                     if changed.all_tasks[index].task_type == TaskType.TASK_SAVE_DISPATCH]
            for rows in (0, 1, 7, 8, 9, 127, 128, 129, 32768):
                covered = np.zeros(rows, dtype=np.uint8)
                for task in saves[:8]:
                    self.assertEqual(task.task_split_value, active)
                    self.assertEqual(changed.all_event_num_triggers[task.trigger_event], 12)
                    lane = task.task_index % 8
                    if lane in selected:
                        split = (rows + active - 1) // active
                        index = selected[lane]
                        covered[min(index * split, rows):min((index + 1) * split, rows)] += 1
                np.testing.assert_array_equal(covered, np.ones_like(covered))
        for raw in ("bad", "-1", "129", "1.5"):
            with patch.dict(os.environ, {"HYPER_MULTICORE_MOE_SAVE_DISPATCH_LANES": raw}):
                with self.assertRaisesRegex(ValueError, "SAVE_DISPATCH_LANES"):
                    execution_options("push", 6)
