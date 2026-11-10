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
"""Validate descriptor scope and receiver-side readiness of empty Push credits."""

from dataclasses import replace
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from hyper_multicore.modules.mega_moe import plan as plan_module
from hyper_multicore.modules.mega_moe.empty_dispatch import EMPTY_DISPATCH_MARKER, mark_empty_combine
from hyper_multicore.modules.mega_moe.inactive_tasks import build_guest_task_prefixes
from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.scheduler.config import TaskType


class TestEmptyDispatch(unittest.TestCase):
    """Only owner-only Push dispatch participates in the experiment."""

    def test_compact_communication_preserves_edges_and_event_credits(self) -> None:
        """Check Dispatch and Combine tile coverage in both forward and backward graphs."""
        for slots in (0, 6):
            with self.subTest(slots=slots):
                spec = MegaMoeSpec(
                    local_num_tokens=1024, hidden_size=5120, intermediate_size=2304,
                    num_experts=24 + 4 * slots, top_k=6, initial_capacity_factor=1.25,
                    receive_capacity=12288, ep_size=4, ep_group=None, rank_id=3,
                    num_cube_cores=24, logical_num_experts=24, replica_slots_per_rank=slots,
                    dispatch_split=128, combine_split=256, _batch_empty_combine=True,
                )
                _, full_fwd, _, full_bwd = _build_runtime_artifacts(spec)
                _, compact_fwd, _, compact_bwd = _build_runtime_artifacts(replace(spec, _communication_rows=257))
                for full, compact, combine_position in ((full_fwd, compact_fwd, 15), (full_bwd, compact_bwd, 16)):
                    self.assertEqual(compact.event_capacity, full.event_capacity)
                    self.assertEqual(compact.task_index_num[0], full.task_index_num[0])
                    self.assertLess(compact.task_index_num[1], full.task_index_num[1])
                    self._assert_compact_edges(compact, spec, combine_position)

    def _assert_compact_edges(self, config, spec: MegaMoeSpec, combine_position: int) -> None:
        """Decode native descriptors to check whole tile coverage for every peer and slot."""
        groups = {}
        for index in config.vector_task_indices[: config.task_index_num[1]]:
            task = config.all_tasks[index]
            if task.task_type != TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
                continue
            position = int(task.inputs[3].input_position)
            self.assertIn(position, (4, combine_position))
            tiles = 3 if position == 4 else 2
            group, tile = divmod(task.task_index, tiles)
            self.assertEqual(task.inputs[3].base_ptr_offset, group)
            self.assertEqual(task.task_split_num, spec.num_experts * tiles)
            self.assertGreaterEqual(tiles * task.task_split_value, 257)
            if position == 4:
                self.assertEqual(config.all_event_num_triggers[task.trigger_event], tiles * spec.ep_size)
            groups.setdefault((position, group), []).append(tile)
        for position, tiles in ((4, 3), (combine_position, 2)):
            for group in range(spec.num_experts):
                self.assertEqual(sorted(groups[position, group]), list(range(tiles)))

    def test_only_dispatch_tasks_change_with_rotated_peer_queues(self) -> None:
        """RATR must retain every source-to-destination tile, dependency and trigger threshold."""
        for mode, slots in (("push", 0), ("push", 2), ("pull", 0)):
            for rank in (0, 3):
                with self.subTest(mode=mode, slots=slots, rank=rank):
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
                        rank_id=rank,
                        num_cube_cores=24,
                        logical_num_experts=24,
                        replica_slots_per_rank=slots,
                        dispatch_mode=mode,
                        dispatch_split=128,
                        combine_split=512,
                    )
                    _, fwd, _, bwd = _build_runtime_artifacts(spec)
                    _, new_fwd, _, new_bwd = _build_runtime_artifacts(replace(spec, _batch_empty_dispatch=True))
                    for original, candidate in ((fwd, new_fwd), (bwd, new_bwd)):
                        grouped: dict[int, list[int]] = {}
                        for task_id in candidate.vector_task_indices[: candidate.task_index_num[1]]:
                            task = candidate.all_tasks[task_id]
                            if (
                                mode == "push"
                                and not slots
                                and task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
                                and task.inputs[3].input_position == 4
                            ):
                                self.assertEqual(task.extra_value_0, EMPTY_DISPATCH_MARKER)
                                self.assertEqual(task.inputs[3].base_ptr_offset, task.task_index // 4)
                                grouped.setdefault(int(task.trigger_event), []).append(int(task.task_index % 4))
                                task.extra_value_0 = original.all_tasks[task_id].extra_value_0
                        if mode == "push" and not slots:
                            self.assertEqual(len(grouped), spec.num_experts)
                            for event, tiles in grouped.items():
                                self.assertEqual(sorted(tiles), [0, 1, 2, 3])
                                self.assertEqual(candidate.all_event_num_triggers[event], 4 * spec.ep_size)
                        self.assertEqual(bytes(original), bytes(candidate))

    def test_enabled_combine_checks_soc_before_native_abi(self) -> None:
        """The opt-in path reaches native validation only for a supported SoC."""
        spec = MegaMoeSpec(512, 5120, 1792, 24, 8, 1.25, 6144, 4, None, 0, 24, _batch_empty_combine=True)
        with (
            patch.object(plan_module, "_soc_name", return_value="Ascend910_9392") as soc,
            patch.object(
                plan_module, "require_empty_dispatch_native", side_effect=RuntimeError("Native ABI gate")
            ) as abi,
        ):
            with self.assertRaisesRegex(RuntimeError, "Native ABI gate"):
                plan_module.build_mega_moe_plan(spec, "cpu")
            abi.assert_called_once_with()
            abi.reset_mock()
            soc.return_value = "Ascend950"
            with self.assertRaisesRegex(ValueError, "requires Ascend910C"):
                plan_module.build_mega_moe_plan(spec, "cpu")
            abi.assert_not_called()

    def test_combine_home_groups_survive_guest_prefixes(self) -> None:
        """Both suffix policies keep all descriptors belonging to a marked home group."""
        for mode, budget in (("push", 0), ("push", 1), ("push", 6), ("pull", 1)):
            for rank in (0, 3):
                with self.subTest(mode=mode, budget=budget, rank=rank):
                    spec = MegaMoeSpec(
                        local_num_tokens=512,
                        hidden_size=5120,
                        intermediate_size=1792,
                        num_experts=24 + 4 * budget,
                        top_k=8,
                        initial_capacity_factor=1.5,
                        receive_capacity=6144,
                        ep_size=4,
                        ep_group=None,
                        rank_id=rank,
                        num_cube_cores=24,
                        logical_num_experts=24,
                        replica_slots_per_rank=budget,
                        dispatch_mode=mode,
                        dispatch_split=128,
                        combine_split=128,
                    )
                    _, old_fwd, _, old_bwd = _build_runtime_artifacts(spec)
                    enabled = replace(spec, _batch_empty_combine=True)
                    _, new_fwd, _, new_bwd = _build_runtime_artifacts(enabled)
                    for position, before, after in ((15, old_fwd, new_fwd), (16, old_bwd, new_bwd)):
                        marked = set()
                        count = after.task_index_num[1]
                        for index in after.vector_task_indices[:count]:
                            task = after.all_tasks[index]
                            is_combine = (
                                task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
                                and task.inputs[3].input_position == position
                            )
                            if is_combine and mode == "push" and task.inputs[3].base_ptr_offset % (6 + budget) < 6:
                                self.assertEqual(task.extra_value_0, EMPTY_DISPATCH_MARKER)
                                marked.add(index)
                        self.assertEqual(len(marked), 96 if mode == "push" else 0)
                        if mode == "push" and budget:
                            self._assert_prefix_keeps_marked(after, marked)
                        for index in marked:
                            after.all_tasks[index].extra_value_0 = 0
                        self.assertEqual(bytes(before), bytes(after))

    def _assert_prefix_keeps_marked(self, after, marked) -> None:
        """Independently decode every serialized prefix and retain marked credits."""
        for local in (False, True):
            prefixes = build_guest_task_prefixes(
                after,
                home_experts=6,
                ep_size=4,
                rank_local_communication=local,
            )
            self.assertTrue(prefixes)
            for prefix in prefixes:
                for variant in (prefix, *prefix.communication_variants):
                    patches = dict(variant.patches)
                    config_type = type(after)
                    (vector_count,) = struct.unpack_from(
                        "<i",
                        patches[config_type.task_index_num.offset],
                        4,
                    )
                    queue = patches[config_type.vector_task_indices.offset]
                    retained = {row[0] for row in struct.iter_unpack("<i", queue[: vector_count * 4])}
                    self.assertTrue(marked.issubset(retained))

    def test_combine_rejects_partial_dependency_groups_without_marking(self) -> None:
        """No incomplete peer-expert group can synthesize completion credits."""
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
            rank_id=0,
            num_cube_cores=24,
            logical_num_experts=24,
            dispatch_split=128,
            combine_split=128,
        )
        _, config, _, _ = _build_runtime_artifacts(spec)
        for index in config.vector_task_indices[: config.task_index_num[1]]:
            task = config.all_tasks[index]
            if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL and task.inputs[3].input_position == 15:
                task.dependent_event += 1
                break
        original = bytes(config)
        with self.assertRaisesRegex(ValueError, "producer or completion"):
            mark_empty_combine(config, ep_size=4, local_experts=6, home_experts=6, size_position=15)
        self.assertEqual(bytes(config), original)

    def test_device_credit_does_not_release_early(self) -> None:
        """Interleave sender tiles independently and require exact release in both modes."""
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("The empty dispatch test requires g++")
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder) / "check"
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-O2",
                    "-I",
                    str(root),
                    str(Path(__file__).with_name("empty_dispatch.cpp")),
                    "-o",
                    str(binary),
                ],
                check=True,
            )
            subprocess.run([str(binary)], check=True)
