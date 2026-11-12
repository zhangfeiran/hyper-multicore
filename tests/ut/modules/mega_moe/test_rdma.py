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
"""RDMA event ownership, restart layout and immutable-source contracts."""

import os
import struct
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_multicore._loader import NativeComponentUnavailableError
from hyper_multicore.modules.mega_moe import rdma
from hyper_multicore.modules.mega_moe.module import MegaMoeExperts, _MegaMoeExecutionResources
from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_multicore.modules.mega_moe.route import _permute_topk_input, prepare_topk_route
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.modules.mega_moe.workspace import _workspace_allocation_specs
from hyper_multicore.scheduler.config import EVENT_INVALID_ID, TaskType, mega_moe_event_capacity
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestRdmaTransport(unittest.TestCase):
    """Check communication contracts without creating an NPU process group."""

    # Exercise resource and metadata contracts without binding a native device.
    # pylint: disable=protected-access

    def test_prototype_native_is_rejected_before_peer_owner_execution(self) -> None:
        """Missing or incompatible native protocols must require a rebuild before any launch."""
        for version in (None, 0, 1, 2):
            namespace = SimpleNamespace() if version is None else SimpleNamespace(mega_moe_rdma_version=lambda: version)
            with (self.subTest(version=version), patch.object(torch.ops, "hyper_multicore", namespace),
                  patch.object(rdma.multicore_ops, "_load_native") as load):
                if version == 2:
                    rdma.require_native()
                else:
                    with self.assertRaisesRegex(NativeComponentUnavailableError, "rebuilt"):
                        rdma.require_native()
                load.assert_called_once_with()

    def test_growth_replaces_a_cached_plan_without_mutating_a_retained_plan(self) -> None:
        """Unchanged route counts must still use the enlarged physical source and receive reservations."""
        resources = _MegaMoeExecutionResources.__new__(_MegaMoeExecutionResources)
        resources.spec = MegaMoeSpec(128, 4, 8, 4, 2, None, 256, 2, None, 0, 1,
                                     dispatch_mode="pull", shmem_data_engine="roce")
        resources.home_spec = None
        resources.workspace = SimpleNamespace(capacity_floor=256)
        resources._token_capacity, resources._plans = 128, {}
        def factory(spec: MegaMoeSpec, _device: Any) -> SimpleNamespace:
            """Retain the exact plan snapshot without preparing native tilings."""
            return SimpleNamespace(spec=spec)

        with patch("hyper_multicore.modules.mega_moe.module.build_mega_moe_plan", side_effect=factory):
            retained = resources.plan_for_tokens(128, "cpu")
            resources._token_capacity, resources.workspace.capacity_floor = 512, 1024
            enlarged = resources.plan_for_tokens(128, "cpu")
            self.assertIsNot(enlarged, retained)
            self.assertEqual(enlarged.spec.routed_slots, 1024)
            self.assertEqual(enlarged.spec.receive_capacity, 1024)
            self.assertEqual(retained.spec.routed_slots, 256)
            self.assertIs(resources.plan_for_tokens(128, "cpu"), enlarged)

    def test_indexed_source_uses_group_raw_bound_and_keeps_the_original_storage(self) -> None:
        """Raw token bounds differ from assignment counts and can be larger on another source rank."""
        original = torch.randn(5, 4)
        token = torch.tensor([0, 0, 2, 2, 3, 4, 4, 4])
        carrier = original.as_strided((8, 4), (0, 1))
        ids = torch.tensor([[0], [1], [0], [1], [0], [1], [0], [1]], dtype=torch.int32)
        spec = MegaMoeSpec(8, 4, 8, 2, 1, None, 8, 1, None, 0, 1,
                           dispatch_mode="pull", shmem_data_engine="roce")
        route = prepare_topk_route(carrier, ids, torch.ones(8, 1), spec, None)
        executor = SimpleNamespace(top_k=1, _resource_group=SimpleNamespace(specification=vars(spec)))
        attached = MegaMoeExperts._attach_indexed_source(executor, route, carrier, (original, token), 6)
        self.assertIs(attached.metadata.original_source, original)
        self.assertEqual(attached.metadata.source_rows_bound, 6)
        self.assertEqual(attached.metadata.source_row_indices.tolist(), token[[0, 2, 4, 6, 1, 3, 5, 7]].tolist())

    def test_indexed_permutation_retains_source_and_exact_stable_order(self) -> None:
        """Metadata sorting must preserve duplicate order and integers beyond FP32's exact range."""
        hidden = torch.randn(3, 4)
        for experts, ids in ((8, [[7, 0], [7, 0], [1, 7]]),
                             (2**24 + 2, [[2**24 + 1, 2**24], [0, 2**24 + 1], [2**24, 0]])):
            with self.subTest(experts=experts):
                keys = torch.tensor(ids, dtype=torch.int64)
                source, inverse = _permute_topk_input(
                    hidden, keys, SimpleNamespace(shmem_data_engine="roce", num_experts=experts))
                expected_order = sorted(range(keys.numel()), key=lambda index: int(keys.flatten()[index]))
                observed_order = torch.empty_like(inverse, dtype=torch.int64)
                observed_order.scatter_(0, inverse.long(), torch.arange(inverse.numel()))
                self.assertIs(source, hidden)
                self.assertEqual(observed_order.tolist(), expected_order)

    def test_engine_is_explicit_and_validated(self) -> None:
        """A stale or misspelled engine must not silently choose another transport."""
        for value in ("mte", "mte_roce", "roce", "ROCE", "auto"):
            with self.subTest(engine=value), patch.dict(os.environ, {rdma.DATA_ENGINE_ENV: value}):
                if value in ("mte", "mte_roce", "roce"):
                    self.assertEqual(rdma.configured_engine(), value)
                else:
                    with self.assertRaisesRegex(ValueError, rdma.DATA_ENGINE_ENV):
                        rdma.configured_engine()

    def test_fixed_pull_counts_routes_before_claiming_receive_storage(self) -> None:
        """Indexed routing must leave heap growth possible until the full receive load is known."""
        spec = MegaMoeSpec(3, 4, 8, 2, 1, None, 3, 1, None, 0, 1,
                           dispatch_mode="pull", shmem_data_engine="roce")
        hidden = torch.randn(3, 4)
        ids = torch.tensor([[1], [0], [1]], dtype=torch.int32)
        workspace = SimpleNamespace(in_use=False, source_buffer=None, wait_for_reuse=Mock())
        route = prepare_topk_route(hidden, ids, torch.ones(3, 1), spec, None, workspace=workspace)
        self.assertIs(route.routed_tokens, hidden)
        self.assertEqual(route.maximum_received_slots, 3)
        self.assertFalse(workspace.in_use)
        workspace.wait_for_reuse.assert_called_once_with()

    def test_both_operands_and_return_source_are_symmetric(self) -> None:
        """Push and pull must reserve all RDMA operands in the same restart transaction."""
        for mode in ("push", "pull"):
            with self.subTest(mode=mode):
                allocations = dict(_workspace_allocation_specs(
                    dispatch_mode=mode, routed_slots=24576, receive_capacity=32768, hidden_size=6144,
                    ep_size=16, num_experts=128, dtype=torch.bfloat16, shmem_data_engine="roce"))
                self.assertEqual(allocations["source_buffer"].shape, (24576, 6144))
                self.assertEqual(allocations["expert_buffer"].shape, (32768, 6144))
                self.assertEqual(allocations["return_source_buffer"].shape, (32768, 6144))
                self.assertEqual(allocations["routed_buffer"].shape, (24576, 6144))
                self.assertEqual(allocations["forward_event_counters"].shape,
                                 allocations["backward_event_counters"].shape)

    def test_credit_reset_preserves_generations_and_dma_constants(self) -> None:
        """A repeat call clears only its progress, retaining ready epochs and immutable sources."""
        credits, progress, values, total = rdma.layout(16, 128, 24576)
        count = 128 * (24576 // 128) + 1
        workspace = SimpleNamespace(rdma_ep_size=16, rdma_num_experts=128, routed_slots=24576,
                                    rdma_dispatch_split=128, rdma_combine_split=128,
                                    rdma_value_bytes=count * 4)
        events = torch.full((total,), 17, dtype=torch.uint8)
        rdma.prepare_events(workspace, events, False)
        expected = torch.arange(count, dtype=torch.int32)
        self.assertTrue(torch.equal(events[values:values + count * 4].view(torch.int32), expected))
        events[credits:progress].fill_(99)
        events[progress:values].fill_(31)
        rdma.prepare_events(workspace, events, True)
        self.assertTrue(torch.all(events[:credits] == 17))
        self.assertTrue(torch.all(events[credits:values] == 0))
        self.assertTrue(torch.equal(events[values:values + count * 4].view(torch.int32), expected))
        suffix = rdma.runtime_suffix(SimpleNamespace(ep_size=16, num_experts=128, routed_slots=24576,
                                                    dispatch_split=128, combine_split=128,
                                                    shmem_data_engine="roce", local_num_tokens=24576, top_k=12), events)
        encoded = struct.unpack("<8Q", suffix)
        self.assertEqual(encoded[0], rdma.RDMA_RUNTIME_MAGIC)
        self.assertEqual(encoded[1:4], tuple(events.data_ptr() + value for value in (credits, progress, values)))
        self.assertEqual(encoded[4:], (count, 3, 0, 24576))

    def test_profile_budget_follows_fixed_peer_ownership(self) -> None:
        """Wait/compute/trigger records must fit when multiple peers share one dedicated AIV."""
        tasks = [SimpleNamespace(task_type=kind, task_index=peer * 8, task_split_num=32,
                                 dependent_event=dependency)
                 for kind, peer, dependency in (
                     (TaskType.TASK_SHMEM_GET_MEM, 0, 4),
                     (TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, 0, 4),
                     (TaskType.TASK_SHMEM_GET_MEM, 2, EVENT_INVALID_ID),
                     (TaskType.TASK_SHMEM_GET_MEM, 1, 4),
                     (TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, 3, EVENT_INVALID_ID),
                     (TaskType.TASK_SWI_GLU, 0, 4))]
        configuration = SimpleNamespace(all_tasks=tasks, vector_task_indices=list(range(len(tasks))),
                                        task_index_num=(0, len(tasks), 0))
        spec = SimpleNamespace(shmem_data_engine="roce", ep_size=4, num_cube_cores=2)
        self.assertEqual(rdma.communication_profile_records(spec, configuration), 7)
        spec.shmem_data_engine = "mte"
        self.assertEqual(rdma.communication_profile_records(spec, configuration), 0)

    def test_completion_includes_every_dedicated_communication_owner(self) -> None:
        """A worker fence must cover queue quiet before any rank can reuse source storage."""
        for mode in ("push", "pull"):
            spec = MegaMoeSpec(local_num_tokens=2048, hidden_size=6144, intermediate_size=2048,
                               num_experts=128, top_k=12, initial_capacity_factor=None,
                               receive_capacity=24576, ep_size=16, ep_group=None, rank_id=0,
                               num_cube_cores=24, dispatch_mode=mode, shmem_data_engine="roce",
                               _gradient_queue_order="original")
            _, forward, _, backward = _build_runtime_artifacts(spec)
            for runtime in (forward, backward):
                with self.subTest(mode=mode):
                    self.assertGreater(runtime.completion_event, runtime.ready_event)
                    self.assertEqual(runtime.all_event_num_triggers[runtime.completion_event], 64)
                    self.assertLess(runtime.completion_event + 8, mega_moe_event_capacity(128, 16))
                    self.assertEqual(runtime.protocol_version, 1)
