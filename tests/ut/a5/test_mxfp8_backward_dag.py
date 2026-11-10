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
"""Protect exact operands and event ownership in the opt-in B0 backward dependency graph."""

from collections import Counter
from dataclasses import replace
import os
import unittest
from unittest.mock import patch

from tests.common.mark_utils import arg_mark

from hyper_multicore.modules.mega_moe.mxfp8_graph import build_mxfp8_runtime, with_source_row_ingress
from hyper_multicore.modules.mega_moe.mxfp8_ingress import MX_SOURCE_ROW_READY_TASK
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.scheduler.config import TaskDescC, TaskType
from hyper_multicore.scheduler.mxfp8 import MX_GMM_TASK_TYPE
from hyper_multicore.scheduler.mxfp8_quant import MX_QUANT_ROW_TASK
from hyper_multicore.scheduler.runtime import serialize_runtime_config
from hyper_multicore.scheduler.vector_roles import configure_vector_roles
from hyper_multicore.scheduler.window_queue import configure_window_queue


OPTION = "HYPER_PARALLEL_MX_B0_BACKWARD_DAG"


class TestMXFP8BackwardDAG(unittest.TestCase):
    """Exercise actual graph generation without a native payload or process group."""

    def setUp(self) -> None:
        """Keep all graph options explicit and restore the caller's environment after each test."""
        environment = patch.dict(os.environ, {
            OPTION: "0", "PANGU_MX_GMM_TILE": "t6", "PANGU_MX_READY_DEPTH": "0",
            "PANGU_MX_SWIGLU_ROW_ONLY": "0", "HYPER_PARALLEL_SHMEM_DATA_ENGINE": "mte_udma",
            "HYPER_PARALLEL_SHMEM_UDMA_QP_NUM": "8", "PANGU_MX_REUSE_DISPATCH_WORKERS": "3",
        })
        environment.start()
        self.addCleanup(environment.stop)

    @staticmethod
    def _shape(ep: int) -> MegaMoeSpec:
        """Retain six physical home experts and production MX matrix dimensions."""
        return MegaMoeSpec(4096, 5120, 1792, ep * 6, 8, 1.25, 40960, ep, None, ep - 1, 32,
                           mx_vector_policy="window2", dispatch_split=4096, combine_split=4096, compute_stages=6)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_independent_gradients_allow_dx_only_combine(self) -> None:
        """Feature: Independent backward gradient branches.
        Description: Inspect EP4/16/64 predecessor relationships in actual runtime graphs.
        Expectation: Gradient branches share read-only inputs and combine waits only for completed DX writes.
        """
        expected = {
            "dw2": {"quant_dy"}, "da": {"quant_dy"}, "swiglu_grad_quant": {"da"},
            "dw1": {"swiglu_grad_quant"}, "dx": {"swiglu_grad_quant"},
            "combine": {"dx"},
        }
        os.environ[OPTION] = "1"
        for ep in (4, 16, 64):
            with self.subTest(ep=ep):
                runtime = build_mxfp8_runtime(self._shape(ep), forward=False, ready_depth=8)
                for name, predecessors in expected.items():
                    actual = {node.name for node in runtime.graph.get_op(name).predecessors}
                    self.assertEqual(actual, predecessors, f"stage={name}, actual={actual}, expected={predecessors}")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_device_events_wait_for_dx_and_retain_all_weight_work(self) -> None:
        """Feature: Actual Device gradient event dependencies.
        Description: Inspect serialized events and every task for EP4/16/64, beyond graph predecessor metadata.
        Expectation: Only DX credits combine; every independent weight task retains its completion event and queue.
        """
        for ep in (4, 16, 64):
            with self.subTest(ep=ep):
                os.environ[OPTION] = "0"
                spec = self._shape(ep)
                cubes, vectors = spec.num_cube_cores, 2 * spec.num_cube_cores
                reference = build_mxfp8_runtime(spec, forward=False, ready_depth=8)
                os.environ[OPTION] = "1"
                runtime = build_mxfp8_runtime(spec, forward=False, ready_depth=8)
                self.assertNotEqual(serialize_runtime_config(runtime.config),
                                    serialize_runtime_config(reference.config))
                stages, offset = {}, 0
                for operator in runtime.graph.topological_sort():
                    stages[operator.name] = (offset, operator.task_num)
                    offset += operator.task_num
                config = runtime.config
                quant_start, _ = stages["quant_dy"]
                swiglu_start, _ = stages["swiglu_grad_quant"]
                dw2_start, count = stages["dw2"]
                da_start, _ = stages["da"]
                dw1_start, _ = stages["dw1"]
                dx_start, _ = stages["dx"]
                for index in range(count):
                    expert = index // cubes
                    self.assertEqual(config.all_tasks[da_start + index].dependent_event,
                                     config.all_tasks[quant_start + expert * vectors].trigger_event)
                    self.assertEqual(config.all_tasks[dx_start + index].dependent_event,
                                     config.all_tasks[swiglu_start + expert * vectors].trigger_event)
                    dx_event = config.all_tasks[dx_start + index].trigger_event
                    self.assertEqual(config.all_event_num_triggers[dx_event], cubes)
                    for start in (dw1_start, dw2_start):
                        weight_event = config.all_tasks[start + index].trigger_event
                        original = reference.config.all_tasks[start + index].trigger_event
                        self.assertEqual(weight_event, original,
                                         f"actual_event={weight_event}, expected_event={original}")
                        self.assertNotEqual(weight_event, dx_event)
                        self.assertEqual(config.all_event_num_triggers[weight_event], cubes)
                triggers = Counter(task.trigger_event for task in config.all_tasks
                                   if task.task_type == MX_GMM_TASK_TYPE)
                for index in range(count):
                    event = config.all_tasks[dx_start + index].trigger_event
                    self.assertEqual(triggers[event], config.all_event_num_triggers[event])
                combine_start, combine_count = stages["combine"]
                dependencies = {task.dependent_event for task in
                                config.all_tasks[combine_start:combine_start + combine_count]}
                dx_events = {task.trigger_event for task in config.all_tasks[dx_start:dx_start + count]}
                self.assertEqual(dependencies, dx_events,
                                 f"actual_combine_dependencies={dependencies}, expected_DX_events={dx_events}")
                queued = list(config.cube_task_indices[:config.task_index_num[0]])
                gradients = {start + index for start in (da_start, dw1_start, dw2_start, dx_start)
                             for index in range(count)}
                self.assertEqual(Counter(queued), Counter(gradients),
                                 f"actual_Cube_tasks={len(queued)}, expected_gradient_tasks={len(gradients)}")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_actual_cube_queues_prioritize_activation_gradients(self) -> None:
        """Feature: Activation-gradient priority within the existing expert window.
        Description: Inspect every physical Cube queue at EP4/16/64 instead of graph edge order.
        Expectation: DA precedes DW2 and DX precedes DW1 without changing lane owners or vector queues.
        """
        for ep in (4, 16, 64):
            with self.subTest(ep=ep):
                spec = self._shape(ep)
                os.environ[OPTION] = "0"
                reference = build_mxfp8_runtime(spec, forward=False, ready_depth=8)
                os.environ[OPTION] = "1"
                current = build_mxfp8_runtime(spec, forward=False, ready_depth=8)
                cfg = current.config
                indices = list(cfg.cube_task_indices[:cfg.task_index_num[0]])
                original = list(reference.config.cube_task_indices[:reference.config.task_index_num[0]])
                self.assertCountEqual(indices, original, f"actual={indices}, expected_membership={original}")
                self.assertEqual(bytes(cfg.vector_task_indices), bytes(reference.config.vector_task_indices))
                self.assertEqual(bytes(cfg.mix_task_indices), bytes(reference.config.mix_task_indices))
                stages, offset = {}, 0
                for operator in current.graph.topological_sort():
                    stages[operator.name] = offset
                    offset += operator.task_num
                for lane in range(spec.num_cube_cores):
                    queue = indices[lane::spec.num_cube_cores]
                    positions = {task: index for index, task in enumerate(queue)}
                    for expert in range(spec.local_experts):
                        point = expert * spec.num_cube_cores + lane
                        for activation, weight in (("da", "dw2"), ("dx", "dw1")):
                            first, second = positions[stages[activation] + point], positions[stages[weight] + point]
                            self.assertLess(first, second, f"lane={lane}, expert={expert}, actual={first}, "
                                            f"expected_before={second}, activation={activation}, weight={weight}")
                        if expert:
                            ahead = positions[stages["da"] + point]
                            prior = positions[stages["dx"] + point - spec.num_cube_cores]
                            self.assertLess(ahead, prior, f"lane={lane}, actual_next_DA={ahead}, "
                                            f"expected_before_previous_DX={prior}")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_activation_priority_rejects_serial_or_forward_dependencies(self) -> None:
        """Feature: Activation-priority dependency validation.
        Description: Request priority before the backward fork is wired or with a forward runtime.
        Expectation: Reject both cases before changing any native queue or event byte.
        """
        for forward in (False, True):
            with self.subTest(forward=forward):
                spec = self._shape(16)
                runtime = build_mxfp8_runtime(spec, forward=forward, ready_depth=8)
                before = serialize_runtime_config(runtime.config)
                with self.assertRaisesRegex(ValueError, "independent gradient dependencies|backward"):
                    configure_window_queue(runtime.config, runtime.graph, spec,
                                           forward=forward, activation_first=True)
                actual = serialize_runtime_config(runtime.config)
                self.assertEqual(actual, before, f"actual_bytes={len(actual)}, expected_bytes={len(before)}")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_payloads_full_work_and_transport_ownership_preserved(self) -> None:
        """Feature: Backward task and operand preservation.
        Description: Compare complete GMM/quant records, task counts and communication descriptors.
        Expectation: Only dependencies may change; physical work, capacity and remote credits stay exact.
        """
        spec = self._shape(16)
        reference = build_mxfp8_runtime(spec, forward=False, ready_depth=8)
        os.environ[OPTION] = "1"
        actual = build_mxfp8_runtime(spec, forward=False, ready_depth=8)
        self.assertEqual(actual.gmm_descriptors, reference.gmm_descriptors)
        self.assertEqual(actual.quant_descriptors, reference.quant_descriptors)
        self.assertEqual(actual.buffer_count, reference.buffer_count)
        self.assertEqual(actual.config.task_capacity, reference.config.task_capacity)
        self.assertEqual(actual.config.event_capacity, reference.config.event_capacity)
        self.assertEqual(Counter(task.task_type for task in actual.config.all_tasks),
                         Counter(task.task_type for task in reference.config.all_tasks))
        left = [task for task in reference.config.all_tasks if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL]
        right = [task for task in actual.config.all_tasks if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL]
        self.assertEqual(len(left), len(right))
        for before, after in zip(left, right):
            first, second = TaskDescC.from_buffer_copy(bytes(before)), TaskDescC.from_buffer_copy(bytes(after))
            first.dependent_event = second.dependent_event = 0
            self.assertEqual(bytes(first), bytes(second))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_disabled_and_forward_runtime_bytes_preserved(self) -> None:
        """Feature: Opt-in B0 backward compatibility.
        Description: Compare default/zero backward and disabled/enabled forward runtime images.
        Expectation: Disabled and default bytes match; forward runtime, GMM and quant records remain exact.
        """
        spec = self._shape(16)
        normal_forward = build_mxfp8_runtime(spec, forward=True, ready_depth=8, swiglu_row_only=True)
        disabled = build_mxfp8_runtime(spec, forward=False, ready_depth=8)
        del os.environ[OPTION]
        default = build_mxfp8_runtime(spec, forward=False, ready_depth=8)
        self.assertEqual(serialize_runtime_config(default.config), serialize_runtime_config(disabled.config))
        os.environ[OPTION] = "1"
        actual_forward = build_mxfp8_runtime(spec, forward=True, ready_depth=8, swiglu_row_only=True)
        self.assertEqual(serialize_runtime_config(actual_forward.config),
                         serialize_runtime_config(normal_forward.config))
        self.assertEqual(actual_forward.gmm_descriptors, normal_forward.gmm_descriptors)
        self.assertEqual(actual_forward.quant_descriptors, normal_forward.quant_descriptors)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_source_row_forward_and_backward_switches_remain_independent(self) -> None:
        """Feature: Combined original-forward and backward optimization isolation.
        Description: Toggle both features at EP4/16/64 and compare complete phase-specific runtime images.
        Expectation: Backward priority preserves source-row forward bytes, and the source flag preserves backward work.
        """
        for ep in (4, 16, 64):
            with self.subTest(ep=ep):
                spec = self._shape(ep)
                os.environ[OPTION] = "0"
                original = build_mxfp8_runtime(spec, forward=True, swiglu_row_only=True, ready_depth=8)
                original_bytes = serialize_runtime_config(original.config)
                row_reference = with_source_row_ingress(spec, original)
                os.environ[OPTION] = "1"
                combined = with_source_row_ingress(spec, build_mxfp8_runtime(
                    spec, forward=True, swiglu_row_only=True, ready_depth=8))
                self.assertEqual(serialize_runtime_config(combined.config),
                                 serialize_runtime_config(row_reference.config))
                self.assertEqual(combined.gmm_descriptors, row_reference.gmm_descriptors)
                self.assertEqual(combined.quant_descriptors, row_reference.quant_descriptors)
                self.assertEqual(combined.buffer_count, 29)
                self.assertEqual(serialize_runtime_config(original.config), original_bytes)
                kinds = Counter(task.task_type for task in combined.config.all_tasks)
                self.assertEqual(kinds[MX_SOURCE_ROW_READY_TASK], spec.local_experts * 2 * spec.num_cube_cores)
                self.assertEqual(kinds[MX_QUANT_ROW_TASK], 0)
                backwards = []
                for source_flag in ("0", "1"):
                    with patch.dict(os.environ, {"PANGU_MX_SOURCE_ROW_FP8": source_flag}):
                        runtime = build_mxfp8_runtime(spec, forward=False, ready_depth=8)
                        backwards.append((serialize_runtime_config(runtime.config),
                                          runtime.gmm_descriptors, runtime.quant_descriptors))
                self.assertEqual(backwards[0], backwards[1])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_active_replicas_keep_the_original_graph_and_legacy_queue_is_rejected(self) -> None:
        """Feature: B0 backward scope validation.
        Description: Request the home-only option with active guests or a legacy queue.
        Expectation: Active guests retain their complete original graph; the legacy queue fails before construction.
        """
        os.environ[OPTION] = "1"
        spec = self._shape(16)
        guests = replace(spec, num_experts=128, logical_num_experts=96, replica_slots_per_rank=2,
                         replica_transport="shmem_signal_kernel_gradient")
        actual = build_mxfp8_runtime(guests, forward=False)
        with patch.dict(os.environ, {OPTION: "0"}):
            reference = build_mxfp8_runtime(guests, forward=False)
        self.assertEqual(serialize_runtime_config(actual.config), serialize_runtime_config(reference.config))
        self.assertEqual(actual.gmm_descriptors, reference.gmm_descriptors)
        self.assertEqual(actual.quant_descriptors, reference.quant_descriptors)
        with self.assertRaisesRegex(ValueError, "B0 backward DAG requires window2"):
            build_mxfp8_runtime(replace(spec, mx_vector_policy="legacy_odd"), forward=False)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_gradient_join_rejects_nonready_or_forward_without_mutation(self) -> None:
        """Feature: Gradient event join scope.
        Description: Request a join with nonready roles or a forward graph.
        Expectation: Both combinations fail before changing any runtime image byte.
        """
        runtime = build_mxfp8_runtime(self._shape(16), forward=False)
        before = serialize_runtime_config(runtime.config)
        for ready, forward in ((False, False), (True, True)):
            with self.subTest(ready=ready, forward=forward):
                with self.assertRaisesRegex(ValueError, "Gradient joins require ready backward roles"):
                    configure_vector_roles(runtime.config, runtime.graph, ready=ready,
                                           forward=forward, join_gradients=True)
                self.assertEqual(serialize_runtime_config(runtime.config), before)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_invalid_option_rejected(self) -> None:
        """Feature: B0 backward option validation.
        Description: Pass nonbinary, textual and empty option values.
        Expectation: Each invalid value raises a descriptive error before graph construction.
        """
        for value in ("true", "2", ""):
            with self.subTest(value=value):
                os.environ[OPTION] = value
                with self.assertRaisesRegex(ValueError, "must be exactly 0 or 1"):
                    build_mxfp8_runtime(self._shape(16), forward=False)
