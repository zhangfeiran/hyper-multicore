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
"""Bounded scratch and ordered metadata for the optional MX UDMA gradient return."""

from dataclasses import replace
import os
import struct
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from hyper_multicore.replica import build_expert_replica_plan
from hyper_multicore.replica.signal_transport import SignalReplicaTransport, signal_storage_bytes
from hyper_multicore.modules.mega_moe.kernel_gradients import prepare_kernel_gradient_return
from hyper_multicore.modules.mega_moe.mxfp8_gradient_udma import (
    mx_gradient_peer_qps,
    mx_udma_gradient_scratch,
)
from hyper_multicore.modules.mega_moe.mxfp8_graph import build_mxfp8_runtime
from hyper_multicore.modules.mega_moe import mxfp8_module
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.scheduler.config import TaskType
from tests.common.mark_utils import arg_mark


def _provider(slots: int, *, qps: int = 32, engine: str = "mte_udma") -> SignalReplicaTransport:
    """Bind a poison-filled symmetric layout without any real device or process group.

    Args:
        slots: Resident guest budget.
        qps: Runtime QP count reported by the mocked initialized transport.
        engine: Runtime data engine used by the topology guard.
    """
    shapes = ((1024, 2048), (512, 1024))
    runtime = mock.MagicMock()
    runtime.debug_state.return_value = {"backend_soc_type": "Ascend950", "device_model": "ascend_950",
                                        "config": {"udma_qp_num": qps, "data_engine": engine}}
    storage = torch.full((signal_storage_bytes(shapes, slots, 4, 2),), 173, dtype=torch.uint8)
    provider = SignalReplicaTransport(runtime, storage, slots, 4)
    provider._bind(tuple(torch.empty((2, *shape), dtype=torch.bfloat16) for shape in shapes))
    return provider


class TestMXFP8GradientUdma(unittest.TestCase):
    """Keep guest return scratch separate from FP32 inboxes and token-owned QPs."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_runtime_guards_precede_symmetric_heap_construction(self) -> None:
        """Feature: Fail-closed optional MX gradient transport.
        Description: Try unsupported QP banks, engines and native capability receipts.
        Expectation: No unsupported configuration reaches symmetric allocation.
        """
        spec = MegaMoeSpec(4096, 5120, 1792, 32, 8, 1.25, 40960, 4, None, 0, 32,
                           logical_num_experts=24, replica_slots_per_rank=2, compute_stages=6,
                           mx_vector_policy="window2", mx_gradient_udma=True,
                           replica_transport="shmem_signal_kernel_gradient")
        for qps, engine, feature_error, expected in (
                (1, "mte_udma", None, "8, 16 or 32 QPs"),
                (32, "mte", None, "mte_udma"),
                (8, "mte_udma", RuntimeError("Missing gradient capability"), "Missing gradient capability"),
                (16, "mte_udma", RuntimeError("Missing gradient capability"), "Missing gradient capability"),
                (32, "mte_udma", RuntimeError("Missing gradient capability"), "Missing gradient capability")):
            with self.subTest(qps=qps, engine=engine, capability=feature_error), mock.patch.dict(
                    os.environ, {"HYPER_PARALLEL_SHMEM_UDMA_QP_NUM": str(qps),
                                 "HYPER_PARALLEL_SHMEM_DATA_ENGINE": engine}), mock.patch.object(
                    mxfp8_module, "bind_mega_moe_spec", return_value=spec), mock.patch.object(
                    mxfp8_module, "require_mxfp8_window_feature", side_effect=feature_error) as require, \
                    mock.patch.object(mxfp8_module, "_get_heap_manager") as heap:
                with self.assertRaisesRegex((ValueError, RuntimeError), expected):
                    mxfp8_module._MXFP8Resources({}, torch.empty(1), shared=False, active_specifications=())
                heap.assert_not_called()
                if feature_error:
                    require.assert_called_once_with("replica_udma_gradient_peer_qp")
                else:
                    require.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_constructor_defaults_and_invalid_switches_do_not_initialize_transport(self) -> None:
        """Feature: Opt-in MX gradient transport.
        Description: Construct default, enabled, invalid and incompatible MX executors on CPU.
        Expectation: The default remains disabled and invalid selections fail before resources exist.
        """
        options = {"local_num_tokens": 4096, "hidden_size": 5120, "intermediate_size": 1792,
                   "num_experts": 96, "top_k": 8, "ep_size": 16, "replica_slots_per_rank": 2}
        with mock.patch.dict(os.environ, {"HYPER_PARALLEL_MX_GRADIENT_UDMA": "0"}):
            module = mxfp8_module.MXFP8MegaMoeExperts(**options)
            self.addCleanup(module.close)
            self.assertFalse(module._resource_group.specification["mx_gradient_udma"])
        with mock.patch.dict(os.environ, {"HYPER_PARALLEL_MX_GRADIENT_UDMA": "1"}):
            with self.assertRaisesRegex(ValueError, "symmetric FP32"):
                mxfp8_module.MXFP8MegaMoeExperts(**options)
            module = mxfp8_module.MXFP8MegaMoeExperts(
                **options, replica_transport="shmem_signal_kernel_gradient")
            self.addCleanup(module.close)
            self.assertTrue(module._resource_group.specification["mx_gradient_udma"])
        with mock.patch.dict(os.environ, {"HYPER_PARALLEL_MX_GRADIENT_UDMA": "2"}):
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                mxfp8_module.MXFP8MegaMoeExperts(**options)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_gradient_qps_do_not_change_token_owners_for_all_target_budgets(self) -> None:
        """Feature: Concurrent token and FP32 gradient UDMA.
        Description: Build EP4/16/64 backward queues for B1/B2/B6 with the original QP16 bank.
        Expectation: Four unused queues per peer preserve every token queue and exclude signal QP0.
        """
        with mock.patch.dict(os.environ, {"HYPER_PARALLEL_SHMEM_DATA_ENGINE": "mte_udma",
                                          "HYPER_PARALLEL_SHMEM_UDMA_QP_NUM": "16"}):
            for ep in (4, 16, 64):
                for budget in (1, 2, 6):
                    with self.subTest(ep=ep, budget=budget):
                        spec = MegaMoeSpec(4096, 5120, 1792, ep * (6 + budget), 8, 1.25, 40960,
                                           ep, None, 0, 32, logical_num_experts=ep * 6,
                                           replica_slots_per_rank=budget, compute_stages=6,
                                           mx_vector_policy="window2", mx_gradient_udma=True,
                                           replica_transport="shmem_signal_kernel_gradient")
                        runtime = build_mxfp8_runtime(spec, forward=False)
                        baseline = build_mxfp8_runtime(replace(spec, mx_gradient_udma=False), forward=False)
                        self.assertEqual(list(runtime.config.vector_task_indices),
                                         list(baseline.config.vector_task_indices))
                        self.assertEqual(list(runtime.config.mix_task_indices),
                                         list(baseline.config.mix_task_indices))
                        gradient_qps = mx_gradient_peer_qps(runtime.config, spec)
                        owners = {}
                        cfg = runtime.config
                        for worker in range(1, 64, 2):
                            queue = cfg.vector_task_indices[
                                cfg.mix_task_indices[worker]:cfg.mix_task_indices[worker + 1]]
                            for task_id in queue:
                                task = cfg.all_tasks[task_id]
                                if task.task_type != TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
                                    continue
                                peer = task.task_index // (task.task_split_num // ep)
                                if peer == spec.rank_id:
                                    continue
                                qp = worker // 4
                                self.assertLess(qp, 16)
                                self.assertEqual(owners.setdefault((peer, qp), worker), worker)
                        self.assertTrue(owners)
                        for peer, row in enumerate(gradient_qps):
                            self.assertEqual(len(row), 4)
                            self.assertEqual(len(set(row)), 4)
                            for qp in row:
                                self.assertGreater(qp, 0)
                                self.assertLess(qp, 16)
                                self.assertNotIn((peer, qp), owners)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_no_free_peer_queue_is_rejected_before_any_device_submission(self) -> None:
        """Feature: Exclusive gradient QP admission.
        Description: Occupy every QP of one peer with a distinct existing token producer.
        Expectation: Fail instead of aliasing a token QP or silently changing its producer.
        """
        queues = [[] for _ in range(64)]
        tasks = []
        for qp in range(16):
            queues[4 * qp + 1] = [qp]
            tasks.append(SimpleNamespace(task_type=TaskType.TASK_SHMEM_PUT_MEM_SIGNAL,
                                         task_index=1, task_split_num=2))
        offsets = [0]
        for queue in queues:
            offsets.append(offsets[-1] + len(queue))
        config = SimpleNamespace(vector_task_indices=[task for queue in queues for task in queue],
                                 mix_task_indices=offsets, all_tasks=tasks)
        with mock.patch.dict(os.environ, {"HYPER_PARALLEL_SHMEM_DATA_ENGINE": "mte_udma",
                                          "HYPER_PARALLEL_SHMEM_UDMA_QP_NUM": "16"}):
            with self.assertRaisesRegex(ValueError, "peer=1, free=0"):
                mx_gradient_peer_qps(config, SimpleNamespace(ep_size=2, rank_id=0))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_eight_qps_keep_exclusive_helpers_for_each_replica_rank(self) -> None:
        """Feature: Smaller gradient QP banks.
        Description: Reserve helpers for all EP16/32 ranks and B1/B2/B6 task queues.
        Expectation: Preserve token queues and keep four distinct helpers outside their ownership and QP0.
        """
        with mock.patch.dict(os.environ, {"HYPER_PARALLEL_SHMEM_DATA_ENGINE": "mte_udma",
                                          "HYPER_PARALLEL_SHMEM_UDMA_QP_NUM": "8"}):
            for ep in (16, 32):
                for budget in (1, 2, 6):
                    for rank in range(ep):
                        with self.subTest(ep=ep, budget=budget, rank=rank):
                            spec = MegaMoeSpec(4096, 5120, 1792, ep * (6 + budget), 8, 1.25, 40960,
                                               ep, None, rank, 32, logical_num_experts=ep * 6,
                                               replica_slots_per_rank=budget, compute_stages=6,
                                               mx_vector_policy="window2", mx_gradient_udma=True,
                                               replica_transport="shmem_signal_kernel_gradient")
                            cfg = build_mxfp8_runtime(spec, forward=False).config
                            original = tuple(cfg.vector_task_indices)
                            gradient_qps = mx_gradient_peer_qps(cfg, spec)
                            self.assertEqual(tuple(cfg.vector_task_indices), original)
                            owners = set()
                            for worker in range(1, 64, 2):
                                queue = cfg.vector_task_indices[
                                    cfg.mix_task_indices[worker]:cfg.mix_task_indices[worker + 1]]
                                for task_id in queue:
                                    task = cfg.all_tasks[task_id]
                                    if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
                                        peer = task.task_index // (task.task_split_num // ep)
                                        if peer != rank:
                                            owners.add((peer, worker // 8))
                            for peer, row in enumerate(gradient_qps):
                                self.assertEqual(len(set(row)), 4)
                                for qp in row:
                                    self.assertGreater(qp, 0)
                                    self.assertLess(qp, 8)
                                    self.assertNotIn((peer, qp), owners)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_eight_qp_scratch_keeps_the_existing_fp32_prefix_layout(self) -> None:
        """Feature: Eight-QP gradient scratch admission.
        Description: Borrow a poisoned BF16 prefix with the smaller initialized bank.
        Expectation: Keep four helpers, reuse storage and leave both FP32 gradient destinations untouched.
        """
        provider = _provider(2, qps=8)
        originals = tuple(gradient.view(torch.uint8).clone() for gradient in provider.pool.gradients)
        layout = mx_udma_gradient_scratch(provider)
        self.assertEqual(layout.helpers, 4)
        self.assertEqual(layout.scratch.data_ptr(), provider.pool.weights[0].data_ptr())
        self.assertEqual(layout.scratch.dtype, torch.float32)
        self.assertEqual(layout.scratch.numel(), 8 * layout.chunk_elements)
        self.assertLessEqual(layout.scratch.numel() * 4, provider.pool.weights[0].numel() * 2)
        for original, gradient in zip(originals, provider.pool.gradients):
            self.assertTrue(torch.equal(original, gradient.view(torch.uint8)))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_scratch_borrows_only_the_unused_weight_prefix_for_b1_b2_b6(self) -> None:
        """Feature: MX UDMA gradient staging.
        Description: Construct all helper ping-pong banks inside the existing BF16 prefix.
        Expectation: No new payload allocation, disjoint banks and untouched FP32 destinations.
        """
        for slots in (1, 2, 6):
            with self.subTest(slots=slots):
                provider = _provider(slots)
                originals = tuple(gradient.view(torch.uint8).clone() for gradient in provider.pool.gradients)
                with mock.patch.object(torch.Tensor, "cpu", side_effect=AssertionError("Readback is forbidden")):
                    layout = mx_udma_gradient_scratch(provider)
                self.assertEqual(layout.helpers, 4)
                self.assertEqual(layout.qp_base, 16)
                self.assertEqual(layout.scratch.data_ptr(), provider.pool.weights[0].data_ptr())
                self.assertEqual(layout.scratch.dtype, torch.float32)
                self.assertEqual(layout.scratch.numel(), 8 * layout.chunk_elements)
                self.assertLessEqual(layout.chunk_elements * 4, 2 * 1024**2)
                self.assertLessEqual(layout.scratch.numel() * 4, provider.pool.weights[0].numel() * 2)
                self.assertEqual(layout.chunk_elements % 8, 0)
                layout.scratch.fill_(3)
                for original, gradient in zip(originals, provider.pool.gradients):
                    self.assertTrue(torch.equal(original, gradient.view(torch.uint8)))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_qp_guard_rejects_transport_without_exclusive_gradient_lanes(self) -> None:
        """Feature: Disjoint UDMA producers.
        Description: Try a token-only QP allocation or an MTE-only runtime.
        Expectation: Fail before any epoch or transfer can advance.
        """
        for qps, engine in ((1, "mte_udma"), (32, "mte")):
            with self.subTest(qps=qps, engine=engine):
                provider = _provider(1, qps=qps, engine=engine)
                with self.assertRaisesRegex(ValueError, "8, 16 or 32 QPs"):
                    mx_udma_gradient_scratch(provider)
                self.assertEqual(provider.epoch, 0)
                provider.runtime.put.assert_not_called()
                provider.runtime.get.assert_not_called()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_both_projection_records_keep_ready_ack_and_sorted_peer_order(self) -> None:
        """Feature: Ordered two-projection UDMA return metadata.
        Description: Extend the existing records with scratch/QP fields for a real replica plan.
        Expectation: Preserve source quotas, complete projection fences and distinct epochs.
        """
        placement = build_expert_replica_plan([[100, 0, 0, 0]] * 4, 1)
        provider = _provider(1)
        layout = mx_udma_gradient_scratch(provider)
        peer_qps = ((1, 2, 3, 4),) * 4
        layout = replace(layout, peer_qps=peer_qps)
        events = ((32, 64), (96, 128))
        plan = SimpleNamespace(replica_w2_events=events[0], replica_w13_events=events[1],
                               spec=SimpleNamespace(num_cube_cores=32))
        for rank in range(4):
            with self.subTest(rank=rank):
                route = SimpleNamespace(plan=placement, rank=rank)
                owner13 = torch.zeros_like(provider.pool.gradients[0])
                owner2 = torch.zeros_like(provider.pool.gradients[1])
                before_epoch = provider.epoch
                with provider._lock, mock.patch.object(
                        torch, "npu", SimpleNamespace(current_stream=mock.Mock()), create=True), \
                        mock.patch.object(torch.Tensor, "record_stream"):
                    result = prepare_kernel_gradient_return(
                        plan, route, provider, owner2, provider.pool.gradients[1],
                        gradient_w13=owner13, guest_w13=provider.pool.gradients[0], udma=layout)
                raw = bytes(result.metadata.tolist())
                words = struct.unpack("<" + "Q" * (len(raw) // 8), raw)
                records = (words[words[1]:words[2]], words[words[2]:])
                self.assertEqual(result.resources, (layout.scratch,))
                for index, (record, fences) in enumerate(zip(records, events)):
                    self.assertEqual(record[0], 9)
                    self.assertEqual(record[3], before_epoch + index + 1)
                    expected = []
                    for transfer in placement.transfers:
                        if transfer.target_rank == rank:
                            expected.extend((transfer.owner_rank, transfer.target_slot - 1,
                                             fences[transfer.target_slot]))
                    for transfer in sorted(placement.transfers, key=lambda item: item.target_rank):
                        if transfer.owner_rank == rank:
                            expected.extend((transfer.target_rank, transfer.target_slot - 1,
                                             transfer.owner_slot, fences[transfer.owner_slot]))
                    end = 16 + len(expected)
                    self.assertEqual(record[12:16], (layout.scratch.data_ptr(), layout.chunk_elements, 4, end))
                    self.assertEqual(record[16:end], tuple(expected))
                    self.assertEqual(record[end:], tuple(qp for row in peer_qps for qp in row))
