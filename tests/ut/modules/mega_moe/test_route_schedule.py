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
"""Exact route task bounds, source-capacity invariance and immutable cached plans."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from tests.common.mark_utils import arg_mark

from hyper_multicore.modules.mega_moe import module as module_impl
from hyper_multicore.modules.mega_moe import function as function_impl
from hyper_multicore.modules.mega_moe.module import MegaMoeExperts
from hyper_multicore.modules.mega_moe.forward.graph import build_forward_graph
from hyper_multicore.modules.mega_moe.route import RouteSchedule, _route_schedule
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.scheduler.config import TaskSplitValue


def make_spec(ep: int = 4) -> MegaMoeSpec:
    """Construct an automatic, non-replica CPU-only execution specification."""
    return MegaMoeSpec(local_num_tokens=128, hidden_size=6144, intermediate_size=2048,
                       num_experts=512, top_k=1, initial_capacity_factor=1.25,
                       receive_capacity=128, ep_size=ep, ep_group=None, rank_id=0,
                       num_cube_cores=24, auto_tokens=True, route_task_bounds=True)


class TestRouteSchedule(unittest.TestCase):
    """Route bounds cover every owner without changing source or workspace shape."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_global_bounds_and_local_empty_matrices(self) -> None:
        """Feature: Exact route task bounds.

        Description: Compare host/device counts across EP1, EP4 and EP16 owners.
        Expectation: Global maxima agree and empty matrix IDs match each owner's range.
        """
        for ep in (1, 4, 16):
            generator = torch.Generator().manual_seed(513 + ep)
            counts = torch.randint(0, 65, (ep, 512), generator=generator, dtype=torch.int32)
            counts[:, ::7] = 0
            original = counts.clone()
            host = counts.tolist()
            for rank in range(ep):
                spec = replace(make_spec(ep), rank_id=rank)
                schedule = _route_schedule(counts, spec)
                self.assertEqual(schedule, _route_schedule(None, spec, host))
                self.assertEqual(schedule.expert_rows, max(sum(row[e] for row in host) for e in range(512)))
                self.assertEqual(schedule.source_expert_rows, int(counts.max()))
                start = rank * spec.local_experts
                expected = tuple(e for e in range(spec.local_experts)
                                 if not any(row[start + e] for row in host))
                self.assertEqual(schedule.empty_experts, expected)
            torch.testing.assert_close(counts, original, rtol=0, atol=0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_empty_and_skewed_routes_retain_exact_coverage(self) -> None:
        """Feature: Empty and skewed route scheduling.

        Description: Move all traffic into one expert after an empty route.
        Expectation: Every active row is covered and inactive owners retain zero-safe initialization.
        """
        counts = torch.zeros((4, 512), dtype=torch.int32)
        empty = _route_schedule(counts, make_spec())
        self.assertEqual(empty, RouteSchedule(0, 0, tuple(range(128))))
        counts[:, 9] = torch.tensor([32, 128, 8192, 4096])
        schedule = _route_schedule(counts, make_spec())
        self.assertEqual(schedule.expert_rows, 12448)
        self.assertEqual(schedule.source_expert_rows, 8192)
        self.assertNotIn(9, schedule.empty_experts)
        self.assertEqual(len(schedule.empty_experts), 127)
        other = _route_schedule(counts, replace(make_spec(), rank_id=2))
        self.assertEqual(other.empty_experts, tuple(range(128)))
        self.assertEqual(other.expert_rows, schedule.expert_rows)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_static_and_replica_paths_keep_legacy_schedules(self) -> None:
        """Feature: Optional task bounds.

        Description: Disable bounds or select static and replica execution.
        Expectation: Each path retains its existing schedule policy.
        """
        self.assertIsNone(_route_schedule(None, replace(make_spec(), route_task_bounds=False)))
        self.assertIsNone(_route_schedule(None, replace(make_spec(), auto_tokens=False)))
        self.assertIsNone(_route_schedule(None, replace(make_spec(), replica_slots_per_rank=1)))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_task_extents_do_not_resize_native_tensor_operands(self) -> None:
        """Feature: Lossless schedule compaction.

        Description: Compare graph operands before and after bounding task extents.
        Expectation: Task counts decrease while native tensor shapes and argument positions remain identical.
        """
        baseline = TaskSplitValue(tp=1, ep=4, seq_size=16384, all_expert_num=512, top_k=1)
        bounded = replace(baseline, expert_rows_bound=256, source_expert_rows_bound=128)
        self.assertEqual(bounded.per_rank_seq, baseline.per_rank_seq)
        self.assertEqual(bounded.seq_all, baseline.seq_all)
        graphs = [build_forward_graph(values, hidden_size=6144, intermediate_size=2048,
                                      expert_weight_layout="EOI") for values in (baseline, bounded)]
        operands = [{tensor.name: (tensor.shape, tensor.param_position)
                     for node in graph.topological_sort() for tensor in (*node.inputs, *node.outputs)}
                    for graph in graphs]
        self.assertEqual(operands[0], operands[1])
        for name in ("dispatch", "swiglu", "combine"):
            nodes = [next(node for node in graph.topological_sort() if node.name == name) for graph in graphs]
            counts = [node.split_spec.task_num_fn(values) for node, values in zip(nodes, (baseline, bounded))]
            self.assertGreater(counts[0], counts[1])
            self.assertGreater(counts[1], 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_plan_cache_reuses_bounds_and_retains_old_backward_plan(self) -> None:
        """Feature: Immutable bounded task plans.

        Description: Reuse equal bounds and generate varying skew until cache eviction.
        Expectation: Bounds select covering plans and retained backward plans remain unchanged.
        """
        resources = object.__new__(module_impl._MegaMoeExecutionResources)
        resources.spec, resources.home_spec = make_spec(), None
        resources._token_capacity, resources._plans = 8192, {}
        with patch.object(module_impl, "build_mega_moe_plan",
                          side_effect=lambda spec, _device: SimpleNamespace(spec=spec)):
            first = resources.plan_for_tokens(4096, "cpu", schedule=RouteSchedule(200, 64, ()))
            same = resources.plan_for_tokens(4096, "cpu", schedule=RouteSchedule(255, 127, (2,)))
            self.assertIs(first, same)
            self.assertEqual(first.spec.schedule_expert_rows, 256)
            self.assertEqual(first.spec.schedule_source_expert_rows, 128)
            skewed = resources.plan_for_tokens(4096, "cpu", schedule=RouteSchedule(7000, 4096, ()))
            self.assertIsNot(first, skewed)
            self.assertEqual(skewed.spec.schedule_expert_rows, 8192)
            self.assertEqual(first.spec.schedule_expert_rows, 256)
            for tokens in (4096, 8192):
                for expert_bits in range(7, (tokens * 4).bit_length()):
                    for source_bits in range(7, min(tokens, 1 << expert_bits).bit_length()):
                        resources.plan_for_tokens(tokens, "cpu",
                                                  schedule=RouteSchedule(1 << expert_bits, 1 << source_bits, ()))
            self.assertLessEqual(len(resources._plans), 64)
            self.assertEqual(first.spec.schedule_expert_rows, 256)
            for invalid in (RouteSchedule(16385, 1, ()), RouteSchedule(1, 4097, ())):
                with self.subTest(invalid=invalid), self.assertRaisesRegex(RuntimeError, "per-expert task bounds"):
                    resources.plan_for_tokens(4096, "cpu", schedule=invalid)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_route_and_communication_bounds_are_consistent(self) -> None:
        """Both bounds cover the same route and reject a contradictory communication maximum."""
        resources = object.__new__(module_impl._MegaMoeExecutionResources)
        resources.spec = replace(make_spec(), _compact_communication=True)
        resources.home_spec, resources._token_capacity, resources._plans = None, 8192, {}
        with patch.object(module_impl, "build_mega_moe_plan",
                          side_effect=lambda spec, _device: SimpleNamespace(spec=spec)):
            route = RouteSchedule(expert_rows=500, source_expert_rows=200, empty_experts=())
            plan = resources.plan_for_tokens(4096, "cpu", schedule=route, maximum_edge_rows=200)
            self.assertEqual(plan.spec.schedule_source_expert_rows, 256)
            self.assertIsNone(plan.spec._communication_rows)
            self.assertIs(resources.plan_for_tokens(4096, "cpu", schedule=route, maximum_edge_rows=200), plan)
            with self.assertRaisesRegex(RuntimeError, "communication bound exceeds"):
                resources.plan_for_tokens(4096, "cpu", schedule=route, maximum_edge_rows=257)
            self.assertEqual(len(resources._plans), 1)
            resources._token_capacity, resources._plans = 384, {}
            partial = resources.plan_for_tokens(384, "cpu", maximum_edge_rows=300)
            self.assertEqual(partial.spec.plan_tokens, 384)
            self.assertIsNone(partial.spec._communication_rows)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_invalid_task_bounds_fail_before_serialization(self) -> None:
        """Feature: Task bound validation.

        Description: Supply noninteger, zero and out-of-capacity extents.
        Expectation: Invalid bounds fail before runtime serialization.
        """
        for name, maximum in (("expert_rows_bound", 512), ("source_expert_rows_bound", 128)):
            for value in (True, 0, -1, 1.5, maximum + 1):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    TaskSplitValue(tp=1, ep=4, seq_size=128, all_expert_num=512, top_k=1, **{name: value})

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_single_expert_tasks_require_boolean_and_distinct_resources(self) -> None:
        """Feature: Optional one-expert GMM addressing.

        Description: Validate option types and attempt resource sharing across different GMM modes.
        Expectation: Invalid types and incompatible mutable tiling images cannot enter shared execution.
        """
        arguments = {"hidden_size": 16, "intermediate_size": 8, "num_experts": 4, "top_k": 2}
        for invalid in (0, 1, None, "true"):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "single_expert_tasks"):
                MegaMoeExperts(**arguments, single_expert_tasks=invalid)
        ordinary = MegaMoeExperts(**arguments)
        single = MegaMoeExperts(**arguments, single_expert_tasks=True)
        self.addCleanup(ordinary.close)
        self.addCleanup(single.close)
        self.assertFalse(ordinary.single_expert_tasks)
        self.assertTrue(single.single_expert_tasks)
        with self.assertRaises(ValueError):
            MegaMoeExperts.share_execution_resources([ordinary, single])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_padded_gate_storage_requires_external_nonreplica_weights(self) -> None:
        """Feature: Inactive Gate/Up storage rows.

        Description: Validate configuration and independent Gate/Up versus Down storage shapes.
        Expectation: Only external, EP-divisible non-replica Gate/Up padding is accepted.
        """
        common = {"hidden_size": 4, "intermediate_size": 3, "num_experts": 4, "top_k": 1,
                  "ep_size": 2, "create_parameters": False, "expert_weight_layout": "EOI"}
        for invalid in (True, -1, 1.5, 1):
            with self.subTest(padding=invalid), self.assertRaises(ValueError):
                MegaMoeExperts(**common, gate_up_expert_padding=invalid)
        for extra in ({"create_parameters": True}, {"replica_slots_per_rank": 1},
                      {"expert_weight_layout": "EIO"}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                MegaMoeExperts(**(common | extra), gate_up_expert_padding=2)
        experts = MegaMoeExperts(**common, gate_up_expert_padding=2)
        self.addCleanup(experts.close)
        hidden = SimpleNamespace(dtype=torch.bfloat16, is_npu=True, device=torch.device("cpu"))
        gate, down = torch.empty(3, 6, 4, dtype=torch.bfloat16), torch.empty(2, 4, 3, dtype=torch.bfloat16)
        experts._validate_tensors(hidden, (gate, down))
        with self.assertRaisesRegex(ValueError, "gate_up_weight"):
            experts._validate_tensors(hidden, (gate[:2], down))
        with self.assertRaisesRegex(ValueError, "down_weight"):
            experts._validate_tensors(hidden, (gate, torch.empty(3, 4, 3, dtype=torch.bfloat16)))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
    def test_padded_gradient_initialization_clears_empty_and_unused_rows(self) -> None:
        """Feature: Direct full-shape expert gradients.

        Description: Poison allocations and vary the real expert emptiness set.
        Expectation: Empty real matrices and unused Gate/Up rows are zero; active matrices await complete stores.
        """
        spec = SimpleNamespace(replica_slots_per_rank=0, intermediate_size=3, hidden_size=4)
        gate, down = torch.ones(3, 6, 4), torch.ones(2, 4, 3)
        for empty in (None, (), (1,), (0, 1)):
            with self.subTest(empty=empty):
                with patch.object(function_impl.torch, "empty_like",
                                  side_effect=lambda value, **kwargs: torch.full_like(value, float("nan"), **kwargs)):
                    result = function_impl._allocate_backward_intermediates(
                        spec, 4, torch.ones(4, 4), gate, down, empty_experts=empty)
                torch.testing.assert_close(result.grad_weight1[2], torch.zeros_like(gate[2]))
                for expert in range(2):
                    for value in (result.grad_weight1, result.grad_weight2):
                        if empty is None or expert in empty:
                            torch.testing.assert_close(value[expert], torch.zeros_like(value[expert]))
                        else:
                            self.assertTrue(torch.isnan(value[expert]).all())
