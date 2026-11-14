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
"""Validate persistent schedules against distributed event progress and exact byte coverage."""
from copy import deepcopy
from dataclasses import replace
import math
import unittest

from hyper_multicore.modules.mega_moe.backward.storage import (
    can_reuse_backward_dispatch,
    prepare_w13_overlap_schedule,
    replica_w2_ready_events,
    replica_w13_ready_events,
)
from hyper_multicore.modules.mega_moe.gradient_queue import mark_active_gradient_lanes
from hyper_multicore.modules.mega_moe.diagnostics import _mega_moe_owner_options
from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts, _build_task_values
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.tasks.gmm import GroupedMatmulTaskMode
from hyper_multicore.scheduler.config import EVENT_INVALID_ID, TaskType
from tests.common.mark_utils import arg_mark


def _spec(ep, slots=0, rank=0):
    return MegaMoeSpec(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                       num_experts=ep * (6 + slots), top_k=8, initial_capacity_factor=1.25,
                       receive_capacity=40960, ep_size=ep, ep_group=None, rank_id=rank,
                       num_cube_cores=4, replica_slots_per_rank=slots, _compact_schedule=True)


def _execute(task, rank, counters, ep, experts):
    target = rank
    if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
        target = task.task_index // ((task.task_split_num // experts) * (experts // ep))
    counters[target][task.trigger_event] += 1
    if task.task_type == TaskType.TASK_GROUPED_MATMUL and task.extra_value_1 > 3:
        counters[rank][task.extra_value_1] += 1


def _simulate(configs, ep, experts, cores, *, ready_queue=True):
    counters = [[0] * config.event_capacity for config in configs]
    queues = []
    for rank, config in enumerate(configs):
        cube = list(config.cube_task_indices[:config.task_index_num[0]])
        vector = list(config.vector_task_indices[:config.task_index_num[1]])
        queues.extend((rank, False, cube[lane::cores]) for lane in range(cores))
        vector_workers = 2 * cores if ready_queue else cores
        queues.extend((rank, ready_queue, vector[lane::vector_workers]) for lane in range(vector_workers))
    remaining = sum(len(queue) for _, _, queue in queues)
    while remaining:
        progress = 0
        for rank, ready, queue in queues:
            config = configs[rank]
            choices = enumerate(queue[:256] if ready else queue[:1])
            for index, task_id in choices:
                task = config.all_tasks[task_id]
                dependency = task.dependent_event
                needed = config.all_event_num_triggers[dependency] if dependency != EVENT_INVALID_ID else 0
                if dependency != EVENT_INVALID_ID and counters[rank][dependency] < needed:
                    continue
                _execute(task, rank, counters, ep, experts)
                queue.pop(index)
                progress += 1
                break
        if not progress:
            raise AssertionError(f"Distributed compact schedule stalled with {remaining} tasks")
        remaining -= progress


class TestCompactSchedule(unittest.TestCase):
    """Check default owner and physical schedules without executing floating-point kernels."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_interface_batching_retains_all_expert_completion(self) -> None:
        """Keep the interface branch's complete grouped schedule on the B0 path."""
        spec = _spec(4)
        _, forward, _, backward = _build_runtime_artifacts(replace(spec, batch_experts=True))
        self.assertEqual(forward.task_index_num[0], 2 * spec.num_cube_cores)
        self.assertEqual(backward.task_index_num[0], 4 * spec.num_cube_cores)
        for config in (forward, backward):
            for task_id in config.cube_task_indices[:config.task_index_num[0]]:
                task = config.all_tasks[task_id]
                self.assertEqual(task.task_type, TaskType.TASK_GROUPED_MATMUL)
                self.assertIn(task.extra_value_1, (GroupedMatmulTaskMode.BATCHED, GroupedMatmulTaskMode.BATCHED_READY))
        with self.assertRaisesRegex(ValueError, "owner-only"):
            _build_runtime_artifacts(replace(_spec(4, 1), batch_experts=True))

    def test_diagnostic_queues_progress_without_ready_bypass(self) -> None:
        """Profile and tensor-dump queues must finish with their static worker policy."""
        for slots in (0, 1, 2):
            with self.subTest(slots=slots):
                specs = [_spec(4, slots, rank) for rank in range(4)]
                artifacts = [_build_runtime_artifacts(spec) for spec in specs]
                for phase in (1, 3):
                    diagnostic_configs = []
                    for spec, parts in zip(specs, artifacts):
                        graph, original = parts[phase - 1], parts[phase]
                        before = bytes(original)
                        config = deepcopy(original)
                        adapter = _mega_moe_owner_options(
                            _build_task_values(spec), num_cube_cores=4, compact_queues=True
                        )["graph_adapter"]
                        self.assertIs(adapter(graph, config), graph)
                        self.assertEqual(bytes(original), before)
                        diagnostic_configs.append(config)
                    _simulate(diagnostic_configs, 4, 4 * (6 + slots), 4, ready_queue=False)

    def test_replica_compaction_retains_both_gradient_returns(self) -> None:
        """Compact Vector queues must retain active W1 lanes and both projection fences."""
        for ep, slots in ((2, 1), (8, 2), (32, 1), (32, 8)):
            with self.subTest(ep=ep, slots=slots):
                spec = replace(_spec(ep, slots), num_cube_cores=24)
                _, _, _, config = _build_runtime_artifacts(spec, overlap_w13=False)
                experts = spec.local_experts
                self.assertTrue(can_reuse_backward_dispatch(config, experts, 24))
                self.assertTrue(mark_active_gradient_lanes(config, experts, 24, projections=(18,)))
                w2_events = replica_w2_ready_events(config, experts, 24)
                self.assertEqual(len(w2_events), experts)
                self.assertTrue(prepare_w13_overlap_schedule(config, experts, 24))
                self.assertEqual(replica_w2_ready_events(config, experts, 24), w2_events)
                self.assertEqual(len(replica_w13_ready_events(config, experts, 24)), experts)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_distributed_event_progress(self) -> None:
        """Feature: Persistent expert schedules.

        Description: Execute serialized event dependencies across independent rank queues.
        Expectation: Every rank finishes forward and backward, including physical producer events.
        """
        for ep in (2, 4, 8):
            for slots in (0, 1, 2):
                with self.subTest(ep=ep, slots=slots):
                    artifacts = [_build_runtime_artifacts(_spec(ep, slots, rank)) for rank in range(ep)]
                    for phase in (1, 3):
                        configs = [item[phase] for item in artifacts]
                        self.assertTrue(all(config.vector_ready_queue for config in configs))
                        _simulate(configs, ep, ep * (6 + slots), 4)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_edge_bounds_and_actual_rows_are_lossless(self) -> None:
        """Feature: Compact communication partitions.

        Description: Exercise exact coverage at empty, tile, lane and capacity boundaries.
        Expectation: Partitions cover each actual row exactly once and never exceed its range.
        """
        for bound in (128, 256, 4096):
            spec = replace(_spec(32), _communication_rows=bound, num_cube_cores=24)
            graph, config, _, _ = _build_runtime_artifacts(spec)
            lanes = graph.get_op("dispatch").task_num // spec.num_experts
            self.assertEqual(lanes, min(2, math.ceil(bound / 128)))
            self.assertEqual(config.cube_single_group, 1)
            for rows in (0, 1, 127, 128, 129, 4095, 4096):
                width = math.ceil(rows / lanes)
                intervals = [(min(lane * width, rows), min((lane + 1) * width, rows))
                             for lane in range(lanes)]
                self.assertEqual(sum(end - start for start, end in intervals), rows)
                self.assertEqual([end for _, end in intervals[:-1]], [start for start, _ in intervals[1:]])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_small_graph_retains_dense_schedule(self) -> None:
        """Feature: Conservative compact admission.

        Description: Use a graph whose descriptor reduction does not repay persistent partitioning.
        Expectation: Dense task geometry and the legacy Vector policy remain active.
        """
        spec = replace(_spec(2), local_num_tokens=128)
        _, forward, _, backward = _build_runtime_artifacts(spec)
        self.assertEqual((forward.vector_ready_queue, backward.vector_ready_queue), (0, 0))


if __name__ == "__main__":
    unittest.main()
