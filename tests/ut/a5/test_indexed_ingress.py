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
"""Check indexed ingress row identity, backward isolation, and allocation ownership."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_multicore.modules.mega_moe import route as route_module
from hyper_multicore.modules.mega_moe.module import MegaMoeExperts
from hyper_multicore.modules.mega_moe.plan import (
    _build_ingress_config, _build_runtime_artifacts, _ingress_graph, _row_credit_element_offset,
)
from hyper_multicore.modules.mega_moe.function import (
    _patch_forward_row_thresholds, _reset_forward_tile_credits,
)
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec, _route_metadata_rows
from hyper_multicore.modules.mega_moe.workspace import _workspace_allocation_specs
from hyper_multicore.scheduler.config import TaskType, validate_runtime_config
from hyper_multicore.modules.mega_moe.plan import _build_task_values
from hyper_multicore.diagnostics import validate_mega_kernel_diagnostic_contract


class TestIndexedIngress(unittest.TestCase):
    """Keep the prototype confined to forward transport and scalar route state."""

    def test_route_uses_scalar_sort_and_preserves_owned_mapping(self) -> None:
        """Skew and empty experts retain exact sorted slots without any token permutation."""
        ids = torch.tensor([[3, 0], [0, 2], [3, 2]], dtype=torch.int32)
        hidden = torch.randn(3, 16)
        counts = torch.tensor([2, 0, 2, 2], dtype=torch.int32)
        spec = MegaMoeSpec(3, 16, 8, 4, 2, None, 128, 1, None, 0, 32,
                           dispatch_mode="pull", forward_ingress_mode="indexed")
        workspace = SimpleNamespace(in_use=True, source_buffer=torch.empty(6, 16))
        order = torch.tensor([1, 2, 3, 5, 0, 4], dtype=torch.int32)
        inverse = torch.tensor([4, 0, 1, 2, 5, 3], dtype=torch.int32)

        def scalar_permute(values: torch.Tensor, indices: torch.Tensor) -> tuple:
            """Emulate only the scalar-sort contract and reject expanded payloads."""
            self.assertEqual(tuple(values.shape), (6, 1), f"Expanded payload shape: {values.shape}")
            self.assertEqual(indices.flatten().tolist(), ids.flatten().tolist())
            return values[order.long()], inverse

        with (patch.object(route_module, "load_torch_npu", return_value=SimpleNamespace(npu_moe_token_permute=scalar_permute)),
              patch.object(route_module, "_permute_topk_input_out") as expanded):
            route = route_module.prepare_topk_route(hidden, ids, torch.ones(3, 2), spec, counts, workspace)
        expanded.assert_not_called()
        self.assertIsNone(route.routed_tokens, "Indexed ingress must not materialize token vectors")
        self.assertIs(route.tokens_per_expert, counts, "Final router counts should remain reusable")
        torch.testing.assert_close(route.permute_src_index, order)
        torch.testing.assert_close(route.unpermute_mapping, inverse)
        self.assertEqual(route.metadata.dispatch_size.tolist(), [2, 0, 2, 2])

    def test_forward_patch_preserves_backward_bytes_and_ready_dependencies(self) -> None:
        """The backward schedule and every forward event dependency remain unchanged."""
        legacy = MegaMoeSpec(512, 5120, 1792, 32, 8, None, 4096, 4, None, 0, 32, dispatch_mode="pull")
        old = _build_runtime_artifacts(legacy)
        new = _build_runtime_artifacts(replace(legacy, forward_ingress_mode="indexed"))
        self.assertEqual(bytes(old[3]), bytes(new[3]), "Indexed mode changed backward task bytes")
        changed_gets = 0
        for before, after in zip(old[1].all_tasks[:old[1].task_num], new[1].all_tasks[:new[1].task_num]):
            self.assertEqual((before.dependent_event, before.trigger_event, before.extra_value_2),
                             (after.dependent_event, after.trigger_event, after.extra_value_2))
            if after.task_type == TaskType.TASK_SHMEM_GET_MEM:
                changed_gets += 1
                self.assertEqual((after.extra_value_0, after.extra_value_1, after.inputs[1].dim[3]), (16, 1, 8))
            elif after.task_type == TaskType.TASK_GROUPED_MATMUL:
                self.assertEqual(after.extra_value_2, 0x41350002, "Port 16 requires tagged Blaze F2")
        self.assertGreater(changed_gets, 0, "Indexed ingress did not reach any transport task")

    def test_only_indexed_mode_reserves_symmetric_metadata(self) -> None:
        """Retain legacy heap sizing and give indexed state a tracked restart allocation."""
        kwargs = dict(dispatch_mode="pull", routed_slots=4096, receive_capacity=4096,
                      hidden_size=5120, ep_size=4, num_experts=32, dtype=torch.bfloat16)
        old = _workspace_allocation_specs(**kwargs)
        new = _workspace_allocation_specs(**kwargs, forward_ingress_mode="indexed")
        self.assertEqual(new[:-1], old, "Legacy symmetric allocation order changed")
        self.assertEqual((new[-1][0], new[-1][1].shape, new[-1][1].dtype),
                         ("route_metadata", (12288, 4), torch.int32))

    def test_unsupported_modes_fail_before_parameter_allocation(self) -> None:
        """Reject push, variable tokens, and inexact scalar indices without NPU work."""
        with self.assertRaisesRegex(ValueError, "pull"):
            MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=4, top_k=2,
                           forward_ingress_mode="indexed", dispatch_mode="push")
        with self.assertRaisesRegex(ValueError, "top_k"):
            MegaMoeExperts(hidden_size=16, intermediate_size=8, num_experts=32, top_k=17,
                           forward_ingress_mode="peer_token", dispatch_mode="pull")

    def test_peer_prepass_queues_only_independent_reads_and_keeps_outer_fences(self) -> None:
        """No per-expert dependency or unvisited terminate can block an owner fanout."""
        spec = MegaMoeSpec(512, 5120, 1792, 32, 8, None, 4096, 4, None, 0, 32,
                           dispatch_mode="pull", forward_ingress_mode="peer_dedup")
        artifacts = _build_runtime_artifacts(spec)
        full = artifacts[1]
        prepass = _build_ingress_config(full)
        indexed = _build_runtime_artifacts(replace(spec, forward_ingress_mode="indexed"))
        self.assertEqual(bytes(artifacts[3]), bytes(indexed[3]), "Peer dedup changed backward tasks")
        self.assertEqual(prepass.task_index_num[0], 0, "Prepass must not queue GMM")
        queued = [int(prepass.vector_task_indices[i]) for i in range(prepass.task_index_num[1])]
        expected = [i for i, task in enumerate(full.all_tasks[:full.task_num])
                    if task.task_type == TaskType.TASK_SHMEM_GET_MEM]
        self.assertEqual(set(queued), set(expected), "Prepass must queue every dispatch exactly once")
        self.assertEqual(len(queued), len(set(queued)), "Duplicate dispatch task")
        for index in queued:
            task = prepass.all_tasks[index]
            self.assertEqual((task.dependent_event, task.trigger_event, task.extra_value_1), (0xFFFFFFFF, 0, 2))
            self.assertEqual(full.all_tasks[index].extra_value_1, 3, "Main GET should be a no-op")
        self.assertEqual((prepass.ready_event, prepass.completion_event, prepass.protocol_version),
                         (full.ready_event, full.completion_event, full.protocol_version))
        self.assertGreater(prepass.ready_event, 8, "Dummy credit must not overlap source readiness")
        self.assertNotIn(full.task_num, queued, "Prepass must not wait on the old Combine terminate")

    def test_peer_ordinals_reconstruct_exact_expert_rows_with_one_owner(self) -> None:
        """Exercise token groups selecting one and two destination ranks and an empty expert."""
        ids = torch.tensor([[3, 0, 2, 1], [6, 7, 0, 4], [2, 0, 7, 1]], dtype=torch.int32)
        slots = torch.argsort(ids.flatten(), stable=True).int()
        counts = torch.bincount(ids.flatten().long(), minlength=8).int()
        ordinals = route_module._peer_row_ordinals(ids, slots, counts)
        values = torch.arange(3 * 16).reshape(3, 16)
        received = [torch.empty((int(count), 16), dtype=values.dtype) for count in counts]
        writes = [torch.zeros(int(count), dtype=torch.int32) for count in counts]
        reads = []
        for slot in slots.tolist():
            token, selected = divmod(slot, 4)
            peer = int(ids[token, selected]) // 4
            selected_slots = [j for j in range(4) if int(ids[token, j]) // 4 == peer]
            if selected != selected_slots[0]:
                continue
            reads.append((token, peer))
            for j in selected_slots:
                expert = int(ids[token, j])
                row = int(ordinals[token * 4 + j])
                received[expert][row].copy_(values[token])
                writes[expert][row] += 1
        expected_reads = {(token, int(expert) // 4) for token in range(3) for expert in ids[token]}
        self.assertEqual(set(reads), expected_reads)
        self.assertEqual(len(reads), len(expected_reads), "Token-peer was fetched more than once")
        for expert in range(8):
            expected_slots = [slot for slot in slots.tolist() if int(ids.flatten()[slot]) == expert]
            torch.testing.assert_close(received[expert], values[torch.tensor(expected_slots).long() // 4])
            torch.testing.assert_close(writes[expert], torch.ones_like(writes[expert]))

    def test_owner_prefix_keeps_all_queues_and_backward_but_changes_credit_unit(self) -> None:
        """Every worker finishes all its reads before any compute wait; no stage is dropped."""
        for rank in (0, 3):
            spec = MegaMoeSpec(512, 5120, 1792, 32, 8, None, 4096, 4, None, rank, 32,
                               dispatch_mode="pull", forward_ingress_mode="peer_overlap")
            current = _build_runtime_artifacts(spec)
            reference = _build_runtime_artifacts(replace(spec, forward_ingress_mode="indexed"))
            self.assertEqual(bytes(current[3]), bytes(reference[3]), "Overlap changed legacy backward")
            config = current[1]
            old = reference[1]
            queued = [int(config.vector_task_indices[i]) for i in range(config.task_index_num[1])]
            old_queued = [int(old.vector_task_indices[i]) for i in range(old.task_index_num[1])]
            self.assertEqual(set(queued), set(old_queued), "A compute/Combine/terminate task was lost")
            owners = [i for i in queued if config.all_tasks[i].task_type == TaskType.TASK_MOE_OWNER_GET]
            self.assertEqual(queued[:len(owners)], owners, "An owner was placed after a compute wait")
            for i in owners:
                task = config.all_tasks[i]
                self.assertEqual((task.dependent_event, task.extra_value_1), (0xFFFFFFFF, 4))
            self.assertEqual(_row_credit_element_offset(config, spec),
                             type(config).all_event_num_triggers.offset // 4 + rank * 8 + 1)

    def test_selected_runtime_threshold_patch_is_owned_and_exact_including_zero(self) -> None:
        """Ordinary and independent profiler images receive the same row-based thresholds."""
        counts = torch.tensor([0, 128, 3, 4096, 1, 0, 24, 50], dtype=torch.int32)
        plan = SimpleNamespace(row_credit_element_offset=32, spec=SimpleNamespace(local_experts=8))
        metadata = SimpleNamespace(expert_row_counts=counts)
        template = torch.zeros(256, dtype=torch.uint8)
        ordinary = template.clone()
        profiler = template.clone()
        for image in (ordinary, profiler):
            _patch_forward_row_thresholds(plan, metadata, image)
            torch.testing.assert_close(image.view(torch.int32)[32:40], counts)
            self.assertEqual(int(image.view(torch.int32)[:32].sum()), 0, "Runtime header was overwritten")
            self.assertEqual(int(image.view(torch.int32)[40:].sum()), 0, "Adjacent event thresholds changed")
        self.assertEqual(int(template.sum()), 0, "Patching selected images changed the immutable template")

    def test_owner_flags_skip_only_later_slots_on_the_same_peer(self) -> None:
        """Top-K order and repeated peer ownership are independent of expert sort order."""
        ids = torch.tensor([[3, 0, 2, 1], [6, 7, 0, 4], [2, 0, 7, 1]], dtype=torch.int32)
        slots = torch.argsort(ids.flatten(), stable=True).int()
        flags = route_module._peer_owner_flags(ids, slots, 4)
        expected = torch.tensor([1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.int32)
        torch.testing.assert_close(flags, expected[slots.long()])

    def test_token_jobs_compact_descriptors_preserve_math_and_cover_all_graph_tasks(self) -> None:
        """Include N128 where dividing the token-job count by E would produce zero."""
        for n in (128, 2048, 8192):
            for rank in range(4):
                spec = MegaMoeSpec(n, 5120, 1792, 32, 8, None, n * 8, 4, None, rank, 32,
                                   dispatch_mode="pull", forward_ingress_mode="peer_token")
                graph, current, _, backward = _build_runtime_artifacts(spec)
                _, reference, _, old_backward = _build_runtime_artifacts(replace(spec, forward_ingress_mode="indexed"))
                self.assertEqual(bytes(backward), bytes(old_backward), "Token jobs changed backward bytes")
                jobs = 4 * n // 128
                old_jobs = 32 * n // 128
                shift = old_jobs - jobs
                self.assertEqual(graph.get_op("dispatch").task_num, jobs)
                self.assertEqual(current.task_num, reference.task_num - shift)
                self.assertEqual(list(current.vector_task_indices[:jobs]), list(range(jobs)))
                for i in range(jobs):
                    task = current.all_tasks[i]
                    self.assertEqual((task.task_type, task.task_index, task.trigger_event, task.extra_value_1),
                                     (TaskType.TASK_MOE_TOKEN_GET, i, rank * 8 + 1, 5))
                for i in range(old_jobs, reference.task_num + 1):
                    self.assertEqual(bytes(current.all_tasks[i - shift]), bytes(reference.all_tasks[i]),
                                     f"Math/Combine/terminal descriptor changed at {i}")
                validate_runtime_config(current, _build_task_values(spec), spec.num_cube_cores)
                validate_mega_kernel_diagnostic_contract(graph, current, parameter_count=17)
                with self.assertRaises(ValueError):
                    validate_mega_kernel_diagnostic_contract(graph, current, parameter_count=16)
                self.assertEqual(graph.get_op("route_metadata").task_num, 0)
                self.assertEqual(graph.get_op("route_metadata").param_positions, [16])
                self.assertEqual(graph.get_op("dispatch").inputs[1].shape, [n, 5120])

    def test_prepass_diagnostic_graph_queues_every_declared_task(self) -> None:
        """The prepass cannot advertise unscheduled GMM/SwiGLU/Combine stages to tools."""
        for n in (128, 8192):
            spec = MegaMoeSpec(n, 5120, 1792, 32, 8, None, n * 8, 4, None, 0, 32,
                               dispatch_mode="pull", forward_ingress_mode="peer_dedup")
            graph, config, _, _ = _build_runtime_artifacts(spec)
            ingress = _ingress_graph(graph)
            prepass = _build_ingress_config(config)
            self.assertEqual(prepass.task_num, graph.get_op("dispatch").task_num)
            self.assertEqual(len(ingress.topological_sort()), 2)
            validate_mega_kernel_diagnostic_contract(ingress, prepass, parameter_count=17)

    def test_tile_ingress_preserves_arithmetic_and_queues_except_input_readiness(self) -> None:
        """Only the marked Up input dependency may change within the canonical math plan."""
        for n in (128, 2048, 8192):
            for rank in range(4):
                with self.subTest(tokens=n, rank=rank):
                    spec = MegaMoeSpec(n, 5120, 1792, 32, 8, None, n * 8, 4, None, rank, 32,
                                       dispatch_mode="pull", forward_ingress_mode="peer_tile")
                    graph, current, _, backward = _build_runtime_artifacts(spec)
                    _, old, _, old_backward = _build_runtime_artifacts(
                        replace(spec, forward_ingress_mode="peer_token"))
                    self.assertEqual(bytes(backward), bytes(old_backward), "Tile readiness changed backward")
                    self.assertEqual(current.task_num, old.task_num, "Tile mode changed the arithmetic task count")
                    validate_mega_kernel_diagnostic_contract(graph, current, parameter_count=17)
                    self.assertEqual(list(current.task_index_num), list(old.task_index_num), "Queues changed")
                    self.assertEqual(list(current.cube_task_indices), list(old.cube_task_indices), "Cube order changed")
                    self.assertEqual(list(current.vector_task_indices), list(old.vector_task_indices),
                                     "Vector producer order changed")
                    for index in range(current.task_num + 1):
                        actual, expected = current.all_tasks[index], old.all_tasks[index]
                        if actual.task_type == TaskType.TASK_MOE_TOKEN_GET:
                            self.assertEqual(actual.extra_value_1, 6, "Token producer did not select tile credits")
                            actual.extra_value_1 = 5
                        elif (actual.task_type == TaskType.TASK_GROUPED_MATMUL
                              and actual.outputs[0].input_position == 7):
                            self.assertEqual((actual.dependent_event, actual.extra_value_0, actual.extra_value_2),
                                             (0xFFFFFFFF, spec.top_k, 0x4D54494C), "Up uses whole-expert readiness")
                            actual.dependent_event = expected.dependent_event
                            actual.extra_value_0 = expected.extra_value_0
                            actual.extra_value_2 = expected.extra_value_2
                        self.assertEqual(bytes(actual), bytes(expected), f"Arithmetic descriptor changed at {index}")
                    self.assertEqual(graph.get_op("dispatch").task_num, 4 * n // 128)
                    self.assertIsNone(_row_credit_element_offset(current, spec), "Tile mode patched old row thresholds")

    def test_tile_counters_follow_the_complete_unchanged_diagnostic_tail(self) -> None:
        """The local tile table cannot alias a peer route prefix or any per-task counter line."""
        kwargs = dict(dispatch_mode="pull", routed_slots=65536, receive_capacity=65536,
                      hidden_size=5120, ep_size=4, num_experts=32, dtype=torch.bfloat16)
        original = _workspace_allocation_specs(**kwargs, forward_ingress_mode="peer_token")
        current = _workspace_allocation_specs(**kwargs, forward_ingress_mode="peer_tile")
        self.assertEqual(current[:-1], original[:-1], "Tile mode changed another symmetric buffer")
        old_rows = original[-1][1].shape[0]
        expected_counters = (4 * 65536 // 128) * 8
        self.assertEqual(current[-1][1].shape, (old_rows + expected_counters // 4, 4),
                         "The local readiness table is missing or overlaps existing storage")

    def test_tile_credit_reset_preserves_routes_statistics_and_adjacent_storage(self) -> None:
        """A fresh generation clears only its owned local counter suffix on the launch stream."""
        spec = MegaMoeSpec(128, 5120, 1792, 32, 8, None, 1024, 4, None, 0, 32,
                           dispatch_mode="pull", forward_ingress_mode="peer_tile")
        rows = _route_metadata_rows(spec.routed_slots, spec.num_experts, spec.ep_size, "peer_tile")
        prefix = _route_metadata_rows(spec.routed_slots, spec.num_experts, spec.ep_size, "peer_token")
        allocation = torch.arange(1, (rows + 1) * 4 + 1, dtype=torch.int32).reshape(rows + 1, 4)
        original = allocation.clone()
        _reset_forward_tile_credits(spec, allocation[:rows])
        torch.testing.assert_close(allocation[:prefix], original[:prefix])
        torch.testing.assert_close(allocation[prefix:rows], torch.zeros_like(allocation[prefix:rows]))
        torch.testing.assert_close(allocation[rows:], original[rows:])

    def test_tile_ready_rejects_unsupported_selector_and_dependency_contract(self) -> None:
        """A marked Up task cannot silently enter generic GMM without its input fence."""
        spec = MegaMoeSpec(128, 5120, 1792, 32, 8, None, 1024, 4, None, 0, 32,
                           dispatch_mode="pull", forward_ingress_mode="peer_tile")
        reference = _build_runtime_artifacts(spec)[1]
        for field, value in (("extra_value_1", 1), ("extra_value_1", 3),
                             ("extra_value_0", 0), ("extra_value_0", 4), ("extra_value_0", 17),
                             ("dependent_event", 1)):
            with self.subTest(field=field, value=value):
                config = type(reference).from_buffer_copy(bytes(reference))
                task = next(task for task in config.all_tasks[:config.task_num]
                            if task.task_type == TaskType.TASK_GROUPED_MATMUL
                            and task.outputs[0].input_position == 7)
                setattr(task, field, value)
                with self.assertRaisesRegex(ValueError, "tile-ready"):
                    validate_runtime_config(config, _build_task_values(spec), spec.num_cube_cores)

    def test_tile_ready_accepts_matching_supported_topk(self) -> None:
        """The public topology, token jobs and marked consumer share one packed-record geometry."""
        for top_k in (1, 2, 4, 8, 16):
            with self.subTest(top_k=top_k):
                spec = MegaMoeSpec(128, 5120, 1792, 32, top_k, None, 128 * top_k, 4, None, 0, 32,
                                   dispatch_mode="pull", forward_ingress_mode="peer_tile")
                config = _build_runtime_artifacts(spec)[1]
                marked = [task for task in config.all_tasks[:config.task_num]
                          if task.extra_value_2 == 0x4D54494C]
                self.assertTrue(marked, "No tile-aware consumer is reachable")
                self.assertEqual({task.extra_value_0 for task in marked}, {top_k}, "Top-K differs across consumers")
