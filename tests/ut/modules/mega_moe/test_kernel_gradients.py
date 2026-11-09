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
"""W2 metadata ordering, ownership and fallback tests without device execution."""

import struct
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_multicore.modules.mega_moe import plan as plan_module
from hyper_multicore.modules.mega_moe.backward.storage import (
    can_reuse_backward_dispatch,
    replica_w2_ready_events,
    replica_w13_ready_events,
)
from hyper_multicore.modules.mega_moe.kernel_gradients import prepare_kernel_gradient_return
from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.replica import build_expert_replica_plan, signal_transport


class TestKernelGradientReturn(unittest.TestCase):
    """Retain per-call completion storage and encode only a verified ready dependency."""

    def test_band_metadata_retains_flags_and_orders_peers_for_multiple_slots(self):
        """Each matrix uses compact physical bands, distinct epochs and invocation-owned flags."""
        placement = build_expert_replica_plan([[100, 80, 0, 0, 0, 0, 0, 0]] * 4, 2)
        plan = SimpleNamespace(
            replica_w2_events=(32, 64, 96, 128),
            replica_w13_events=(160, 192, 224, 256),
            spec=SimpleNamespace(num_cube_cores=20),
        )
        for rank in range(4):
            with self.subTest(rank=rank), patch.object(signal_transport, "_GRADIENT_BAND_BYTES", 4096):
                shapes = ((256, 64), (128, 64))
                size = signal_transport.signal_storage_bytes(shapes, 2, 4, 2, gradient_bands=True)
                provider = signal_transport.SignalReplicaTransport(
                    Mock(), torch.empty(size, dtype=torch.uint8), 2, 4, gradient_bands=True
                )
                provider._bind(tuple((torch.zeros(2, *shape, dtype=torch.bfloat16) for shape in shapes)))
                route = SimpleNamespace(plan=placement, rank=rank)
                owner13, owner2 = (torch.zeros(2, *shape) for shape in shapes)
                with provider._lock:
                    with patch.object(torch.npu, "current_stream"), patch.object(
                        torch.Tensor, "record_stream"
                    ) as record:
                        result = prepare_kernel_gradient_return(
                            plan,
                            route,
                            provider,
                            owner2,
                            provider.pool.gradients[1],
                            gradient_w13=owner13,
                            guest_w13=provider.pool.gradients[0],
                        )
                    self.assertEqual(record.call_count, 4)
                    self.assertEqual(len(result.resources), 2)
                    raw = bytes(result.metadata.tolist())
                    values = struct.unpack("<" + "Q" * (len(raw) // 8), raw)
                    first, second = (values[values[1] : values[2]], values[values[2] :])
                    self.assertEqual((first[0], second[0]), (7, 7))
                    self.assertEqual((first[3], second[3]), (1, 2))
                    self.assertEqual(first[14:18], (16, 128, 64, 8))
                    self.assertEqual(second[14:18], (16, 256, 64, 16))
                    for matrix, item, flags, events in zip(
                        (1, 0), (first, second), result.resources, (plan.replica_w2_events, plan.replica_w13_events)
                    ):
                        self.assertEqual(item[12], flags.data_ptr())
                        self.assertEqual(item[13], provider.kernel_gradient_band_signals(matrix, 20)[0].data_ptr())
                        self.assertEqual(int(flags.count_nonzero()), 0)
                        incoming = [t for t in placement.transfers if t.target_rank == rank]
                        owned = sorted(
                            (t for t in placement.transfers if t.owner_rank == rank), key=lambda t: t.target_rank
                        )
                        expected = []
                        for transfer in incoming:
                            expected.extend(
                                (transfer.owner_rank, transfer.target_slot - 2, events[transfer.target_slot])
                            )
                        for transfer in owned:
                            expected.extend(
                                (
                                    transfer.target_rank,
                                    transfer.target_slot - 2,
                                    transfer.owner_slot,
                                    events[transfer.owner_slot],
                                )
                            )
                        self.assertEqual(item[18:], tuple(expected))

    def test_band_fallback_uses_one_consistent_protocol_for_both_projections(self):
        """Unsupported tables select full-expert records before reserving their epochs."""
        placement = build_expert_replica_plan([[100, 0, 0, 0]] * 4, 1)
        route = SimpleNamespace(plan=placement, rank=0)
        plan = SimpleNamespace(
            replica_w2_events=(32, 64), replica_w13_events=(96, 128), spec=SimpleNamespace(num_cube_cores=33)
        )
        provider = SimpleNamespace(
            kernel_gradients=True,
            gradient_bands=True,
            kernel_gradient_band_signals=Mock(return_value=None),
            kernel_gradient_signals=Mock(side_effect=((7, 4096, 8192), (8, 4096, 8192))),
        )
        owner, guest = (torch.zeros(1, 4, 8), torch.zeros(1, 4, 8))
        with patch.object(torch.npu, "current_stream"), patch.object(torch.Tensor, "record_stream"):
            result = prepare_kernel_gradient_return(
                plan, route, provider, owner, guest, gradient_w13=owner, guest_w13=guest
            )
        raw = bytes(result.metadata.tolist())
        values = struct.unpack("<" + "Q" * (len(raw) // 8), raw)
        self.assertEqual((values[values[1]], values[values[2]]), (1, 1))
        self.assertEqual(result.resources, ())
        self.assertEqual(provider.kernel_gradient_band_signals.call_count, 2)
        self.assertEqual(provider.kernel_gradient_signals.call_count, 2)

    def test_metadata_matches_physical_slots_and_ordered_peers(self):
        """Every ready event belongs to its physical expert; records follow FP32 peer order."""
        placement = build_expert_replica_plan([[100, 0, 0, 0]] * 4, 1)
        plan = SimpleNamespace(replica_w2_events=(32, 64), spec=SimpleNamespace(num_cube_cores=20))
        retained = []
        for rank in range(4):
            route = SimpleNamespace(plan=placement, rank=rank)
            provider = SimpleNamespace(
                kernel_gradients=True, kernel_gradient_signals=Mock(return_value=(7, 4096, 8192))
            )
            gradient, guest = (torch.zeros(1, 4, 8), torch.zeros(1, 4, 8))
            with patch.object(torch.npu, "current_stream"), patch.object(torch.Tensor, "record_stream") as record:
                result = prepare_kernel_gradient_return(plan, route, provider, gradient, guest)
            retained.append(result)
            self.assertEqual(record.call_count, 2)
            provider.kernel_gradient_signals.assert_called_once_with()
            raw = bytes(result.metadata.tolist())
            values = struct.unpack("<" + "Q" * (len(raw) // 8), raw)
            incoming = [t for t in placement.transfers if t.target_rank == rank]
            owned = sorted((t for t in placement.transfers if t.owner_rank == rank), key=lambda t: t.target_rank)
            self.assertEqual(
                values[:12],
                (
                    1,
                    rank,
                    1,
                    7,
                    32,
                    gradient.data_ptr(),
                    guest.data_ptr(),
                    4096,
                    8192,
                    result.completion.data_ptr(),
                    len(incoming),
                    len(owned),
                ),
            )
            expected = []
            for transfer in incoming:
                expected.extend((transfer.owner_rank, transfer.target_slot - 1, 64))
            for transfer in owned:
                expected.extend((transfer.target_rank, transfer.target_slot - 1, transfer.owner_slot, 32))
            self.assertEqual(values[12:], tuple(expected))
            torch.testing.assert_close(result.completion, torch.zeros(20, 16, dtype=torch.int32))
        self.assertEqual(len({r.completion.data_ptr() for r in retained}), 4)

    def test_invalid_owner_buffer_fails_before_protocol_advance(self):
        """The fused path cannot silently cast or mutate an autograd-owned gradient."""
        placement = build_expert_replica_plan([[100, 0, 0, 0]] * 4, 1)
        route = SimpleNamespace(plan=placement, rank=0)
        plan = SimpleNamespace(replica_w2_events=(32, 64))
        provider = SimpleNamespace(kernel_gradients=True, kernel_gradient_signals=Mock())
        for gradient in (
            torch.ones(1, 4, 8, dtype=torch.bfloat16),
            torch.ones(1, 4, 8, requires_grad=True),
            torch.ones(1, 8, 4).transpose(1, 2),
        ):
            with self.assertRaisesRegex(ValueError, "detached contiguous FP32"):
                prepare_kernel_gradient_return(plan, route, provider, gradient, torch.ones(1, 4, 8))
        provider.kernel_gradient_signals.assert_not_called()

    def test_two_projections_use_distinct_epochs_and_completion_storage(self):
        """A W13 ready word must never be mistaken for an unfinished W2 epoch."""
        placement = build_expert_replica_plan([[100, 0, 0, 0]] * 4, 1)
        plan = SimpleNamespace(
            replica_w2_events=(32, 64), replica_w13_events=(96, 128), spec=SimpleNamespace(num_cube_cores=20)
        )
        route = SimpleNamespace(plan=placement, rank=0)
        provider = SimpleNamespace(
            kernel_gradients=True, kernel_gradient_signals=Mock(side_effect=((7, 4096, 8192), (8, 4096, 8192)))
        )
        owner, guest = (torch.zeros(1, 4, 8), torch.zeros(1, 4, 8))
        w13, guest13 = (torch.zeros(1, 8, 8), torch.zeros(1, 8, 8))
        with patch.object(torch.npu, "current_stream"), patch.object(torch.Tensor, "record_stream"):
            result = prepare_kernel_gradient_return(
                plan, route, provider, owner, guest, gradient_w13=w13, guest_w13=guest13
            )
        raw = bytes(result.metadata.tolist())
        values = struct.unpack("<" + "Q" * (len(raw) // 8), raw)
        self.assertEqual(result.matrices, (1, 0))
        self.assertEqual(values[:2], (2, 3))
        first, second = (values[values[1] : values[2]], values[values[2] :])
        self.assertEqual((first[3], second[3]), (7, 8))
        self.assertEqual((first[4], second[4]), (32, 64))
        self.assertEqual((first[5], second[5]), (owner.data_ptr(), w13.data_ptr()))
        self.assertNotEqual(first[9], second[9])
        self.assertEqual(second[15], 96)
        self.assertEqual(tuple(result.completion.shape), (2, 20, 16))

    def test_w13_schedule_proves_complete_expert_before_return(self):
        """Real generated push/pull queues retain receive reuse and per-Cube ordering."""
        for mode in ("push", "pull"):
            for rank in range(2):
                with self.subTest(mode=mode, rank=rank):
                    spec = MegaMoeSpec(
                        128,
                        128,
                        128,
                        12,
                        2,
                        1.25,
                        512,
                        2,
                        None,
                        rank,
                        20,
                        dispatch_mode=mode,
                        replica_slots_per_rank=2,
                        logical_num_experts=8,
                        replica_transport="shmem_signal_kernel_gradient",
                    )
                    _, _, _, config = _build_runtime_artifacts(spec)
                    self.assertTrue(can_reuse_backward_dispatch(config, 6, 20))
                    events = replica_w13_ready_events(config, 6, 20)
                    self.assertEqual(len(events), 6)
                    self.assertEqual(len(set(events)), 6)
                    indices = list(config.cube_task_indices[: config.task_index_num[0]])
                    for worker in range(20):
                        queue = [config.all_tasks[index] for index in indices[worker::20]]
                        for expert in range(6):
                            stages = [
                                task.outputs[0].input_position for task in queue if task.task_index // 20 == expert
                            ]
                            self.assertEqual(stages, [6, 8, 12, 18])
                    weight = next(
                        (
                            config.all_tasks[index]
                            for index in indices
                            if config.all_tasks[index].outputs[0].input_position == 18
                        )
                    )
                    weight.dependent_event = events[0]
                    self.assertFalse(replica_w13_ready_events(config, 6, 20))

    def test_no_replica_runtime_preserves_original_schedule(self):
        """Only enabled W13 return adds producer fences; no-copy queues stay unchanged."""
        for mode in ("push", "pull"):
            for transport in ("p2p", "shmem_signal_kernel_gradient", "shmem_signal_kernel_gradient_bands"):
                with self.subTest(mode=mode, transport=transport):
                    spec = MegaMoeSpec(
                        128,
                        128,
                        128,
                        12,
                        2,
                        1.25,
                        512,
                        2,
                        None,
                        0,
                        20,
                        dispatch_mode=mode,
                        replica_slots_per_rank=2,
                        logical_num_experts=8,
                        replica_transport=transport,
                    )
                    _, _, _, original = _build_runtime_artifacts(spec, overlap_w13=False)
                    with patch.object(plan_module.dist, "is_initialized", return_value=False):
                        plan = plan_module.build_mega_moe_plan(spec, torch.device("cpu"))
                    if transport == "p2p":
                        self.assertIsNone(plan.bwd_diagnostics_no_replica)
                        self.assertFalse(plan.replica_w13_events)
                        self.assertEqual(bytes(plan.bwd_diagnostics._runtime.host_config), bytes(original))
                    else:
                        self.assertEqual(bytes(plan.bwd_diagnostics_no_replica._runtime.host_config), bytes(original))
                        self.assertNotEqual(bytes(plan.bwd_diagnostics._runtime.host_config), bytes(original))
                        self.assertTrue(plan.replica_w13_events)
                        self.assertTrue(
                            can_reuse_backward_dispatch(plan.bwd_diagnostics_no_replica._runtime.host_config, 6, 20)
                        )
                        self.assertTrue(plan.bwd_runtime_no_replica_reuse_dispatch)
                        self.assertEqual(
                            replica_w2_ready_events(plan.bwd_diagnostics_no_replica._runtime.host_config, 6, 20),
                            plan.replica_w2_events,
                        )
