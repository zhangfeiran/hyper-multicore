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
"""Projection weight readiness preserves exact bytes, queue ownership and lease joins."""

from dataclasses import replace
import os
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.distributed as dist

from hyper_multicore.replica.pool import ReplicaPool
from hyper_multicore.replica.signal_transport import (
    SignalReplicaTransport, signal_storage_bytes,
)
from hyper_multicore.modules.mega_moe.mxfp8_graph import build_mxfp8_runtime
from hyper_multicore.modules.mega_moe.mxfp8_module import MXFP8MegaMoeExperts
from hyper_multicore.modules.mega_moe.mxfp8_prefetch import MXProjectionPrefetch
from hyper_multicore.modules.mega_moe.mxfp8_weights import MXFP8PhysicalWeight
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.scheduler.mxfp8 import MX_GMM_TASK_TYPE
from tests.common.mark_utils import arg_mark


def _weight(rows: int, columns: int) -> MXFP8PhysicalWeight:
    data = torch.full((1, rows, columns), 31, dtype=torch.uint8)
    return MXFP8PhysicalWeight((1, rows, columns), data.view(torch.float8_e4m3fn),
                               torch.full((1, rows // 64, columns, 2), 17, dtype=torch.uint8),
                               (data + 7).view(torch.float8_e4m3fn),
                               torch.full((1, rows, columns // 64, 2), 29, dtype=torch.uint8))


class TestMXFP8Prefetch(unittest.TestCase):
    """Exercise supported launch graphs and mocked stream submission boundaries."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_only_guest_weight_reads_are_fenced_without_moving_tasks(self) -> None:
        """Feature: MX projection-ready readers.
        Description: Compare EP4 and EP64 prefetch graphs against the same fused-gradient graphs.
        Expectation: Only guest NN/NT reader tags change; TN, home work and all queues remain exact.
        """
        for ep in (4, 64):
            spec = MegaMoeSpec(4096, 5120, 1792, ep * 8, 8, 1.25, 40960, ep, None, 0, 32,
                               logical_num_experts=ep * 6, replica_slots_per_rank=2,
                               replica_transport="shmem_signal_kernel_gradient", compute_stages=6,
                               mx_vector_policy="window2")
            for forward in (True, False):
                with self.subTest(ep=ep, forward=forward):
                    baseline = build_mxfp8_runtime(spec, forward=forward)
                    current = build_mxfp8_runtime(spec, forward=forward, weight_prefetch=True)
                    self.assertEqual(current.buffer_count, baseline.buffer_count + 1)
                    self.assertTrue(current.weight_prefetch)
                    self.assertEqual(current.gmm_descriptors, baseline.gmm_descriptors)
                    self.assertEqual(current.quant_descriptors, baseline.quant_descriptors)
                    self.assertEqual(list(current.config.vector_task_indices),
                                     list(baseline.config.vector_task_indices))
                    self.assertEqual(list(current.config.cube_task_indices), list(baseline.config.cube_task_indices))
                    changed = 0
                    for index in range(current.config.task_num + 1):
                        task, old = current.config.all_tasks[index], baseline.config.all_tasks[index]
                        if task.extra_value_1:
                            changed += 1
                            self.assertEqual(task.task_type, MX_GMM_TASK_TYPE)
                            self.assertIn(task.outputs[0].input_position, (7, 8, 11, 12))
                            self.assertGreaterEqual(task.task_index // 32, 6)
                            self.assertEqual(task.extra_value_1, current.buffer_count)
                            old.extra_value_1 = task.extra_value_1
                        self.assertEqual(bytes(task), bytes(old))
                    self.assertEqual(changed, 2 * 2 * 32)
            home = build_mxfp8_runtime(replace(spec, replica_slots_per_rank=0, num_experts=ep * 6),
                                       forward=True, weight_prefetch=True)
            self.assertFalse(home.weight_prefetch)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_current_projection_bytes_publish_before_consumers_and_always_join(self) -> None:
        """Feature: Exact MX P2P prefetch.
        Description: Submit both phase orders with mocked HCCL and SDMA, then raise inside the consumer.
        Expectation: Current data/scale pointers, projection-order readiness and exception-safe final stream join.
        """
        for backward in (False, True):
            with self.subTest(backward=backward):
                weights = (_weight(64, 128), _weight(128, 64))
                fields = ("row_data", "row_scale") if backward else ("col_data", "col_scale")
                axes = tuple(getattr(weight, field) for weight in weights for field in fields)
                pool = ReplicaPool(tuple(value.view(torch.uint8).reshape(1, -1) for value in axes), 1)
                runtime, backend = mock.MagicMock(), mock.MagicMock()
                storage = torch.empty(signal_storage_bytes(((64, 128), (128, 64)), 1, 2, 2), dtype=torch.uint8)
                provider = SignalReplicaTransport(runtime, storage, 1, 2)
                transfers = (SimpleNamespace(owner_rank=0, owner_slot=0, target_rank=1, target_slot=1),
                             SimpleNamespace(owner_rank=1, owner_slot=0, target_rank=0, target_slot=1))
                config = SimpleNamespace(home_experts=1, ep_size=2, replica_slots_per_rank=1)
                route = SimpleNamespace(rank=0, group=object(),
                                        plan=SimpleNamespace(config=config, transfers=transfers))
                timeline, sends = [], []

                def exchange(operations: list) -> list:
                    """Record current source addresses and complete one mocked projection dependency."""
                    sends.extend(operation[1].data_ptr() for operation in operations if operation[0] is dist.isend)
                    request = mock.MagicMock()
                    request.wait.side_effect = lambda: timeline.append("wait_projection")
                    return [request]

                runtime.put.side_effect = lambda *args, **kwargs: timeline.append("publish_sdma")
                target = "hyper_multicore.modules.mega_moe.mxfp8_prefetch"
                with mock.patch(target + ".replica_pool", return_value=pool), \
                        mock.patch(target + "._global_peer_rank", side_effect=lambda group, peer: peer), \
                        mock.patch.object(torch, "npu", backend), mock.patch.object(torch.Tensor, "record_stream"), \
                        mock.patch.object(dist, "P2POp", side_effect=lambda *args: args), \
                        mock.patch.object(dist, "batch_isend_irecv", side_effect=exchange):
                    with self.assertRaisesRegex(RuntimeError, "Consumer failure"):
                        prefetch = MXProjectionPrefetch()
                        with prefetch.lease(weights, route, provider, backward=backward) as (guests, info):
                            self.assertEqual(timeline, ["wait_projection", "publish_sdma"] * 2)
                            order = (2, 3, 0, 1) if backward else (0, 1, 2, 3)
                            self.assertEqual(sends, [axes[index].data_ptr() for index in order])
                            self.assertEqual(len(guests), 4)
                            self.assertEqual(info.numel(), 40)
                            backend.current_stream.return_value.wait_event.assert_not_called()
                            raise RuntimeError("Consumer failure")
                    backend.current_stream.return_value.wait_event.assert_called_once()
                self.assertFalse(pool._lock.locked())
                self.assertTrue(all(call.kwargs["use_sdma"] for call in runtime.put.call_args_list))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_changing_active_slots_preserves_inactive_bytes_and_receives_fresh_axes(self) -> None:
        """Feature: MX inactive guest storage.
        Description: Alternate two guest slots and current forward/backward axes in one reusable pool.
        Expectation: Receives replace every active byte; inactive poison and prior generations stay untouched.
        """
        initial = (_weight(64, 128), _weight(128, 64))
        aliases = tuple(getattr(weight, field).view(torch.uint8).reshape(1, -1)
                        for weight in initial for field in ("col_data", "col_scale"))
        pool = ReplicaPool(aliases, 2)
        for guest in pool.weights:
            guest.fill_(255)
        runtime, backend = mock.MagicMock(), mock.MagicMock()
        storage = torch.empty(signal_storage_bytes(((64, 128), (128, 64)), 2, 2, 2), dtype=torch.uint8)
        provider = SignalReplicaTransport(runtime, storage, 2, 2)
        config = SimpleNamespace(home_experts=1, ep_size=2, replica_slots_per_rank=2)
        prefetch = MXProjectionPrefetch()
        for generation, backward in enumerate((False, True, False, True)):
            active = generation % 2
            with self.subTest(generation=generation, backward=backward, active=active):
                weights = (_weight(64, 128), _weight(128, 64))
                fields = ("row_data", "row_scale") if backward else ("col_data", "col_scale")
                axes = tuple(getattr(weight, field) for weight in weights for field in fields)
                for index, axis in enumerate(axes):
                    axis.view(torch.uint8).fill_(generation * 4 + index + 1)
                inactive = tuple(guest[1 - active].clone() for guest in pool.weights)
                transfers = (SimpleNamespace(owner_rank=1, owner_slot=0, target_rank=0,
                                             target_slot=1 + active),)
                route = SimpleNamespace(rank=0, group=object(),
                                        plan=SimpleNamespace(config=config, transfers=transfers))
                order = (2, 3, 0, 1) if backward else (0, 1, 2, 3)
                received = []

                def exchange(operations: list) -> list:
                    """Complete exact current projection receives before publishing readiness."""
                    for operation in operations:
                        self.assertIs(operation[0], dist.irecv)
                        index = order[len(received)]
                        operation[1].copy_(axes[index].view(torch.uint8).reshape(-1))
                        received.append(index)
                    return [mock.MagicMock()]

                target = "hyper_multicore.modules.mega_moe.mxfp8_prefetch"
                runtime.put.reset_mock()
                with mock.patch(target + ".replica_pool", return_value=pool), \
                        mock.patch(target + "._global_peer_rank", side_effect=lambda group, peer: peer), \
                        mock.patch.object(torch, "npu", backend), mock.patch.object(torch.Tensor, "record_stream"), \
                        mock.patch.object(dist, "P2POp", side_effect=lambda *args: args), \
                        mock.patch.object(dist, "batch_isend_irecv", side_effect=exchange):
                    with prefetch.lease(weights, route, provider, backward=backward) as (guests, _):
                        self.assertEqual(received, list(order))
                        for guest, axis, old in zip(guests, axes, inactive):
                            self.assertTrue(torch.equal(guest[active].view(torch.uint8),
                                                        axis[0].view(torch.uint8)))
                            self.assertTrue(torch.equal(guest[1 - active].view(torch.uint8).reshape(-1), old))
                        self.assertEqual(runtime.put.call_count, 2)
                        for call, matrix in zip(runtime.put.call_args_list, ((1, 0) if backward else (0, 1))):
                            self.assertEqual(call.args[0].data_ptr(),
                                             provider.projection_signals[matrix, active, :1].data_ptr())
                self.assertFalse(pool._lock.locked())
        self.assertEqual(backend.current_stream.return_value.wait_event.call_count, 4)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_prefetch_requires_the_owned_symmetric_inbox(self) -> None:
        """Feature: Prefetch configuration boundary.
        Description: Enable overlap without its symmetric readiness storage.
        Expectation: Reject the configuration before distributed allocation.
        """
        with self.assertRaisesRegex(ValueError, "symmetric FP32 gradient inbox"):
            MXFP8MegaMoeExperts(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                               num_experts=24, top_k=8, ep_size=4, replica_slots_per_rank=2,
                               overlap_weights=True)
        with mock.patch.dict(os.environ, {"HCCL_OP_EXPANSION_MODE": "AIV"}):
            with self.assertRaisesRegex(ValueError, "AICPU_CacheDisable"):
                MXFP8MegaMoeExperts(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                                   num_experts=24, top_k=8, ep_size=4, replica_slots_per_rank=2,
                                   replica_transport="shmem_signal_kernel_gradient", overlap_weights=True)
