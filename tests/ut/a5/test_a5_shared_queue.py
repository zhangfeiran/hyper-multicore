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
"""Cold scheduling gates for complete shared forward Cube rounds."""
from copy import deepcopy
from dataclasses import replace
import unittest

from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_multicore.modules.mega_moe.shared_schedule import _prioritize_shared_forward_rounds
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.scheduler.config import TaskType
from tests.common.mark_utils import arg_mark


class TestA5SharedQueue(unittest.TestCase):
    """Preserve events, descriptors, vector work and round ownership under cold reordering."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_forward_rounds_move_without_changing_backward_or_task_contracts(self) -> None:
        """Feature: Independent shared forward placement.
        Description: Reorder real shared/indexed plans for multiple token counts and ranks.
        Expectation: Only the forward Cube queue changes, with every task and terminal preserved.
        """
        base = MegaMoeSpec(512, 5120, 1792, 32, 8, None, 4096, 4, None, 0, 32,
                           dispatch_mode="pull", forward_ingress_mode="indexed", backward_ingress_mode="indexed",
                           shared_intermediate_size=2048, forward_shared_expert=True, backward_shared_expert=True)
        for tokens, rank, down in ((512, 0, False), (2048, 1, True), (4096, 3, False), (4096, 0, True)):
            with self.subTest(tokens=tokens, rank=rank, include_down=down):
                graph, runtime, _, backward = _build_runtime_artifacts(replace(base, local_num_tokens=tokens,
                                                                             rank_id=rank))
                original = deepcopy(runtime)
                backward_bytes = bytes(backward)
                queue = list(runtime.cube_task_indices[:runtime.task_index_num[0]])
                descriptors = [bytes(task) for task in runtime.all_tasks[:runtime.task_num]]

                _prioritize_shared_forward_rounds(runtime, graph, num_cube_cores=32, include_down=down)

                actual = list(runtime.cube_task_indices[:runtime.task_index_num[0]])
                self.assertCountEqual(actual, queue)
                self.assertEqual([bytes(task) for task in runtime.all_tasks[:runtime.task_num]], descriptors)
                self.assertEqual(list(runtime.all_event_num_triggers), list(original.all_event_num_triggers))
                self.assertEqual(list(runtime.vector_task_indices), list(original.vector_task_indices))
                self.assertEqual(bytes(backward), backward_bytes)
                self.assertEqual(
                    {index: position % 32 for position, index in enumerate(actual)},
                    {index: position % 32 for position, index in enumerate(queue)},
                )
                terminal = int(runtime.vector_task_indices[runtime.task_index_num[1] - 1])
                self.assertEqual(runtime.all_tasks[terminal].task_type, TaskType.TASK_TERMINATE)
                self.assertEqual(bytes(runtime.all_tasks[terminal]), bytes(original.all_tasks[terminal]))
                stage_ids = {}
                offset = 0
                for operator in graph.topological_sort():
                    stage_ids[operator.name] = list(range(offset, offset + operator.task_num))
                    offset += operator.task_num
                expected = stage_ids["shared_up_proj"]
                self.assertEqual(actual[:32], expected)
                _prioritize_shared_forward_rounds(runtime, graph, num_cube_cores=32, include_down=down)
                self.assertEqual(list(runtime.cube_task_indices[:runtime.task_index_num[0]]), actual)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_incompatible_graph_is_rejected_without_partial_mutation(self) -> None:
        """Feature: Fail-closed cold shared scheduling.
        Description: Apply the pass to a plain routed graph and an incomplete shared round.
        Expectation: Both invalid inputs fail before changing the RuntimeConfig bytes.
        """
        base = MegaMoeSpec(512, 5120, 1792, 32, 8, None, 4096, 4, None, 0, 32,
                           dispatch_mode="pull", forward_ingress_mode="indexed")
        for shared, cores in ((False, 32), (True, 31)):
            with self.subTest(shared=shared, cores=cores):
                spec = replace(base, shared_intermediate_size=2048) if shared else base
                graph, runtime, _, _ = _build_runtime_artifacts(spec)
                before = bytes(runtime)
                with self.assertRaisesRegex(ValueError, "complete Cube round"):
                    _prioritize_shared_forward_rounds(runtime, graph, num_cube_cores=cores)
                self.assertEqual(bytes(runtime), before)
