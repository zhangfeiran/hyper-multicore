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
"""MX projection-return queue coverage, complete fences and bounded FP32 storage."""

from dataclasses import replace
import os
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from hyper_multicore.replica.capacity import ExpertReplicaConfig
from hyper_multicore.replica.plan import ExpertExecutionPlan
from hyper_multicore.replica.signal_transport import (
    SignalReplicaTransport, signal_storage_bytes,
)
from hyper_multicore.modules.mega_moe.mxfp8_graph import build_mxfp8_runtime
from hyper_multicore.modules.mega_moe.mxfp8_gradients import lease_mx_gradient_inbox, mx_projection_events
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.scheduler.mxfp8 import MX_GMM_TASK_TYPE, MxGmmDescriptor
from tests.common.mark_utils import arg_mark


class TestMXFP8Gradients(unittest.TestCase):
    """Keep transport owners, mathematical work and output storage auditable."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_helpers_preserve_tasks_communication_and_complete_projection_fences(self) -> None:
        """Feature: MX projection-return helpers.
        Description: Compare EP4/64 B2 graphs with the ordinary P2P window queue.
        Expectation: Every task is retained, odd QP owners are exact and TN fences join all 32 Cubes.
        """
        for ep in (4, 64):
            spec = MegaMoeSpec(4096, 5120, 1792, ep * 8, 8, 1.25, 40960, ep, None, ep - 1, 32,
                               logical_num_experts=ep * 6, replica_slots_per_rank=2, compute_stages=6,
                               mx_vector_policy="window2", replica_transport="p2p")
            with self.subTest(ep=ep), mock.patch.dict(os.environ, {
                    "HYPER_PARALLEL_SHMEM_DATA_ENGINE": "mte_udma", "HYPER_PARALLEL_SHMEM_UDMA_QP_NUM": "16"}):
                old = build_mxfp8_runtime(spec, forward=False)
                current = build_mxfp8_runtime(replace(spec, replica_transport="shmem_signal_kernel_gradient"),
                                             forward=False)
                self.assertTrue(current.kernel_gradients)
                self.assertEqual(current.buffer_count, old.buffer_count)
                cfg, baseline = current.config, old.config
                self.assertEqual(list(cfg.cube_task_indices), list(baseline.cube_task_indices))
                count = cfg.task_index_num[1]
                self.assertEqual(sorted(cfg.vector_task_indices[:count]), sorted(baseline.vector_task_indices[:count]))
                for worker in range(64):
                    queue = cfg.vector_task_indices[cfg.mix_task_indices[worker]:cfg.mix_task_indices[worker + 1]]
                    previous = baseline.vector_task_indices[
                        baseline.mix_task_indices[worker]:baseline.mix_task_indices[worker + 1]]
                    if worker % 2:
                        self.assertEqual(list(queue), list(previous))
                    elif worker < 8:
                        self.assertEqual(list(queue), [])
                changed = 0
                for task in cfg.all_tasks[:cfg.task_num]:
                    if task.task_type != MX_GMM_TASK_TYPE:
                        continue
                    record = MxGmmDescriptor.from_buffer_copy(current.gmm_descriptors, task.extra_value_0 * 128)
                    expert = task.task_index // 32
                    if record.layout == 2 and expert >= 6:
                        changed += 1
                        self.assertIn(task.outputs[0].input_position, (6, 18))
                        self.assertEqual(record.y_offset, (expert - 6) * record.m * record.n * 4)
                self.assertEqual(changed, 2 * 2 * 32)
                for outputs in ((6, 37), (18, 38)):
                    events = mx_projection_events(cfg, 8, outputs)
                    self.assertEqual(len(events), 8)
                    self.assertEqual(len(set(events)), 8)
                    self.assertTrue(all(cfg.all_event_num_triggers[event] == 32 for event in events))
                cfg.all_event_num_triggers[events[0]] = 31
                with self.assertRaisesRegex(ValueError, "32 Cube producers"):
                    mx_projection_events(cfg, 8, (18, 38))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_gradient_lease_never_copies_or_retains_physical_weight_storage(self) -> None:
        """Feature: Exact MX weights with symmetric FP32-only return.
        Description: Borrow, poison and reacquire a bounded two-slot gradient inbox.
        Expectation: Fresh zero gradients, increasing epochs, no weight transfers and exception-safe release.
        """
        shapes = ((64, 128), (128, 64))
        weights = tuple(torch.empty((2, *shape), dtype=torch.float8_e4m3fn) for shape in shapes)
        storage = torch.empty(signal_storage_bytes(shapes, 2, 2, 2), dtype=torch.uint8)
        runtime = mock.MagicMock()
        provider = SignalReplicaTransport(runtime, storage, 2, 2)
        route = SimpleNamespace(plan=SimpleNamespace(config=SimpleNamespace(replica_slots_per_rank=2, ep_size=2)))
        with self.assertRaisesRegex(RuntimeError, "Consumer failure"):
            with lease_mx_gradient_inbox(provider, weights, route) as pool:
                first_epoch = provider.kernel_gradient_signals()[0]
                self.assertGreater(provider.kernel_gradient_signals()[0], first_epoch)
                for gradient, owner in zip(pool.gradients, weights):
                    self.assertEqual(gradient.dtype, torch.float32)
                    self.assertEqual(gradient.shape, owner.shape)
                    self.assertEqual(int(gradient.count_nonzero()), 0)
                    self.assertEqual(gradient.untyped_storage().data_ptr(), storage.data_ptr())
                    self.assertNotEqual(gradient.data_ptr(), owner.data_ptr())
                    gradient.fill_(3)
                raise RuntimeError("Consumer failure")
        with lease_mx_gradient_inbox(provider, weights, route) as pool:
            self.assertGreater(provider.kernel_gradient_signals()[0], first_epoch)
            self.assertTrue(all(int(value.count_nonzero()) == 0 for value in pool.gradients))
        self.assertFalse(provider._lock.locked())
        runtime.put.assert_not_called()
        runtime.get.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_exact_guest_quotas_preserve_overwritten_and_unread_storage(self) -> None:
        """Feature: Selective MX guest gradient initialization.
        Description: Reuse poisoned symmetric storage with nonempty, empty-assigned and inactive guest slots.
        Expectation: Only zero-row guests that will return gradients are cleared, without device count readback.
        """
        shapes = ((64, 128), (128, 64))
        weights = tuple(torch.empty((2, *shape), dtype=torch.float8_e4m3fn) for shape in shapes)
        storage = torch.empty(signal_storage_bytes(shapes, 2, 3, 2), dtype=torch.uint8)
        runtime = mock.MagicMock()
        provider = SignalReplicaTransport(runtime, storage, 2, 3)
        config = ExpertReplicaConfig(6, 3, 2)
        slots = ((0, 1, 2, 4), (2, 3, -1, -1), (4, 5, -1, -1))
        for rank, first, second in ((0, 7, 0), (0, 0, 9), (0, 5, 6), (0, 0, 0), (1, 7, 0)):
            with self.subTest(rank=rank, first=first, second=second):
                dispatch = ((5, 0, first, second) + (0,) * 8, (0,) * 12, (0,) * 12)
                logical = ((5, 0, first, 0, second, 0), (0,) * 6, (0,) * 6)
                plan = ExpertExecutionPlan(config, logical, slots, dispatch)
                plan.validate()
                route = SimpleNamespace(plan=plan, rank=rank)
                with lease_mx_gradient_inbox(provider, weights, route) as pool:
                    for gradient in pool.gradients:
                        gradient.fill_(float("nan"))
                with mock.patch.object(torch.Tensor, "cpu", side_effect=AssertionError("Readback is forbidden")):
                    with lease_mx_gradient_inbox(provider, weights, route) as pool:
                        for gradient in pool.gradients:
                            self.assertEqual(gradient.untyped_storage().data_ptr(), storage.data_ptr())
                            for slot in range(2):
                                if rank == 0 and (first, second)[slot] == 0:
                                    self.assertEqual(int(gradient[slot].count_nonzero()), 0)
                                else:
                                    self.assertTrue(torch.isnan(gradient[slot]).all().item())
                self.assertFalse(provider._lock.locked())
        runtime.put.assert_not_called()
        runtime.get.assert_not_called()
