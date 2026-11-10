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
"""Validate original-row FP8 ingress storage and fail-closed mode selection."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.common.mark_utils import arg_mark
from hyper_multicore.modules.mega_moe.mxfp8_ingress import (
    borrow_row_planes, source_row_fp8_enabled, validate_source_row_buffers,
)
from hyper_multicore.modules.mega_moe.mxfp8_graph import build_mxfp8_runtime, with_source_row_ingress
from hyper_multicore.modules.mega_moe.mxfp8_execution import (
    _forward_graph_buffers, _forward_ingress_buffers,
)
from hyper_multicore.modules.mega_moe.mxfp8_ops import (
    launch_mxfp8_task_graph, serialize_mxfp8_launch,
)
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.scheduler.mxfp8 import MxLaunchDescriptor
from hyper_multicore.scheduler.runtime import serialize_runtime_config


class TestMXFP8Ingress(unittest.TestCase):
    """Exercise physical byte views without a device, registration or collective."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_registered_receive_storage_is_reused_without_overlap(self) -> None:
        """Feature: Registered receive-plane reuse.
        Description: Place two disjoint byte planes inside the existing BF16 allocation.
        Expectation: No additional backing storage is allocated and the unused suffix is untouched.
        """
        for capacity, width in ((1, 1024), (7, 5120), (65, 8192)):
            with self.subTest(capacity=capacity, width=width):
                storage = torch.empty(capacity, width, dtype=torch.bfloat16)
                raw = storage.view(torch.uint8).flatten()
                raw.fill_(0xA5)
                row, scale = borrow_row_planes(storage)
                self.assertEqual(row.dtype, torch.float8_e4m3fn)
                self.assertEqual(scale.dtype, torch.uint8)
                self.assertEqual(row.shape, (capacity, width))
                self.assertEqual(scale.shape, (capacity, width // 64, 2))
                self.assertEqual(row.data_ptr(), storage.data_ptr())
                self.assertEqual(scale.data_ptr() - row.data_ptr(), capacity * width)
                self.assertEqual(scale.untyped_storage().data_ptr(), storage.untyped_storage().data_ptr())
                row.view(torch.uint8).fill_(0x17)
                scale.fill_(0x29)
                end = capacity * width * 33 // 32
                self.assertTrue(torch.all(raw[end:] == 0xA5))
                self.assertTrue(torch.all(raw[:capacity * width] == 0x17))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_borrowed_storage_rejects_invalid_layouts(self) -> None:
        """Feature: Receive-plane bounds.
        Description: Supply wrong dtypes, feature alignment, dimensions and strides.
        Expectation: Invalid borrowed layouts fail before any device operation.
        """
        invalid = (torch.empty(2, 1024), torch.empty(2, 512, dtype=torch.bfloat16),
                   torch.empty(2, 1024, dtype=torch.bfloat16).t(),
                   torch.empty(0, 1024, dtype=torch.bfloat16), torch.empty(1024, dtype=torch.bfloat16))
        for value in invalid:
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                borrow_row_planes(value)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_mode_is_default_off_and_restricted_to_supported_original_forward(self) -> None:
        """Feature: Optional source-row quantization.
        Description: Resolve the flag with supported and unsupported topology or precision controls.
        Expectation: Default-off behavior is unchanged; unsupported opt-in combinations are rejected.
        """
        arguments = {"vector_policy": "window2", "row_only": True, "replica_slots": 0, "hidden_size": 5120}
        with patch.dict("os.environ", {"PANGU_MX_SOURCE_ROW_FP8": "0"}):
            self.assertFalse(source_row_fp8_enabled(**arguments))
        with patch.dict("os.environ", {"PANGU_MX_SOURCE_ROW_FP8": "1"}):
            self.assertTrue(source_row_fp8_enabled(**arguments))
            self.assertTrue(source_row_fp8_enabled(**{**arguments, "replica_slots": 2}))
            for changed in ({"vector_policy": "legacy_odd"}, {"row_only": False},
                            {"replica_slots": -1}, {"hidden_size": 512}, {"hidden_size": 9216}):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    source_row_fp8_enabled(**{**arguments, **changed})
        with patch.dict("os.environ", {"PANGU_MX_SOURCE_ROW_FP8": "yes"}):
            with self.assertRaises(ValueError):
                source_row_fp8_enabled(**arguments)

    @staticmethod
    def _bindings() -> tuple[list[torch.Tensor], SimpleNamespace]:
        storage = torch.empty(8, 1024, dtype=torch.bfloat16)
        row, scale = borrow_row_planes(storage)
        buffers = [torch.empty(1, dtype=torch.uint8) for _ in range(29)]
        buffers[0], buffers[16], buffers[17], buffers[18] = storage, row, scale, row
        buffers[2] = torch.empty(4, 1024, dtype=torch.float8_e4m3fn)
        buffers[27] = torch.empty(4, 16, 2, dtype=torch.uint8)
        buffers[28] = torch.arange(8, dtype=torch.int32)
        spec = SimpleNamespace(rank_id=0, ep_size=1, num_experts=2, hidden_size=1024, local_num_tokens=4,
                               num_cube_cores=32, top_k=2, mx_vector_policy="window2", replica_slots_per_rank=0)
        return buffers, spec

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_native_receive_bindings_use_full_storage_without_changing_logical_bounds(self) -> None:
        """Feature: CANN TensorList backing-storage descriptors.
        Description: Model the observed 400 MiB base versus 200 MiB row and 6.25 MiB scale storage mismatch.
        Expectation: The native call receives complete base aliases while the logical launch retains plane bounds.
        """
        buffers, spec = self._bindings()
        resource = torch.empty(64, dtype=torch.uint8)
        before = serialize_mxfp8_launch(buffers, resource, resource, spec, source_row_fp8=True)
        prefix = "hyper_multicore.modules.mega_moe.mxfp8_ops."
        with patch(prefix + "_load_mxfp8_native"), patch(prefix + "require_mxfp8_window_feature"), \
                patch.object(torch.ops.hyper_multicore, "mxfp8_task_graph", create=True) as native:
            launch_mxfp8_task_graph(buffers, resource, resource, resource, resource,
                                   vector_policy="window2", source_row_fp8=True)
        physical = native.call_args.args[0]
        for position in (16, 17, 18):
            with self.subTest(position=position):
                self.assertEqual(physical[position].numel(), buffers[0].numel() * 2)
                self.assertEqual(physical[position].data_ptr(), buffers[0].data_ptr())
                self.assertEqual(physical[position].untyped_storage().data_ptr(),
                                 buffers[0].untyped_storage().data_ptr())
        self.assertEqual(buffers[17].data_ptr() - buffers[0].data_ptr(), buffers[16].numel())
        self.assertEqual(serialize_mxfp8_launch(buffers, resource, resource, spec, source_row_fp8=True), before)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_launch_version_is_explicit_and_old_bytes_are_unchanged(self) -> None:
        """Feature: Separately versioned source-row launch.
        Description: Serialize the same bounds through old and source-row protocols.
        Expectation: Only the explicit version and protocol code change; all 592 bytes stay bounded.
        """
        buffers, spec = self._bindings()
        runtime, events = torch.empty(64, dtype=torch.uint8), torch.empty(64, dtype=torch.uint8)
        old = serialize_mxfp8_launch(buffers, runtime, events, spec)
        new = serialize_mxfp8_launch(buffers, runtime, events, spec, source_row_fp8=True)
        expected = MxLaunchDescriptor.from_buffer_copy(old)
        self.assertEqual((expected.version, expected.reserved), (6, 5))
        expected.version, expected.reserved = 14, 13
        self.assertEqual(bytes(expected), new)
        self.assertEqual(len(new), 592)
        validate_source_row_buffers(buffers, spec)
        buffers[17] = buffers[17].clone()
        with self.assertRaisesRegex(ValueError, "alias"):
            serialize_mxfp8_launch(buffers, runtime, events, spec, source_row_fp8=True)
        with self.assertRaisesRegex(ValueError, "guest"):
            serialize_mxfp8_launch(buffers, runtime, events, spec, source_row_fp8=True, kernel_gradients=True)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_original_source_uses_scalar_indices_and_recompute_keeps_both_axes(self) -> None:
        """Feature: Sender storage and retained-axis separation.
        Description: Bind original FP8 rows without expanded vectors, then bind the retained BF16 route.
        Expectation: Original source has T rows; the retained path uses unchanged independent column buffers.
        """
        buffers, spec = self._bindings()
        spec.intermediate_size, spec.local_experts = 128, 2
        hidden = torch.empty(4, 1024, dtype=torch.bfloat16)
        workspace = SimpleNamespace(expert_buffer=buffers[0])
        route = SimpleNamespace(metadata=SimpleNamespace(expert_capacity=8),
                                routed_tokens=None, permute_src_index=buffers[28])
        target = "hyper_multicore.modules.mega_moe.mxfp8_execution.quantize_source_rows"
        with patch(target, return_value=(buffers[2], buffers[27])) as quantize:
            source, axes, _, extra, routed_rows = _forward_ingress_buffers(
                spec, workspace, hidden, route, source_row_fp8=True, row_only=True)
            quantize.assert_called_once_with(hidden)
        self.assertIs(source, buffers[2])
        self.assertEqual(source.shape[0], 4)
        self.assertEqual(routed_rows, 8)
        self.assertEqual(axes[0].data_ptr(), workspace.expert_buffer.data_ptr())
        self.assertIs(extra[0], buffers[27])
        self.assertIs(extra[1], buffers[28])
        route.routed_tokens = torch.empty(8, 1024, dtype=torch.bfloat16)
        with patch(target, side_effect=AssertionError("Recompute source quantization is forbidden")):
            source, axes, _, extra, routed_rows = _forward_ingress_buffers(
                spec, workspace, hidden, route, source_row_fp8=False, row_only=False)
        self.assertIs(source, route.routed_tokens)
        self.assertEqual(extra, ())
        self.assertNotEqual(axes[0].data_ptr(), axes[2].data_ptr())
        with self.assertRaisesRegex(ValueError, "without expanded token"):
            _forward_ingress_buffers(spec, workspace, hidden, route, source_row_fp8=True, row_only=True)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_source_graph_preserves_all_transport_credits_queues_and_gmm_bytes(self) -> None:
        """Feature: Source-row task dependency conservation.
        Description: Replace only completed input-quant work with its existing event-producing no-op.
        Expectation: Every transport, event, queue, math descriptor and unused flag remains byte-exact.
        """
        with patch.dict("os.environ", {"HYPER_PARALLEL_SHMEM_DATA_ENGINE": "mte_udma",
                                       "HYPER_PARALLEL_SHMEM_UDMA_QP_NUM": "8", "PANGU_MX_GMM_TILE": "t6"}):
            for rank in (0, 15):
                for split in (512, 4096):
                    with self.subTest(rank=rank, split=split):
                        spec = MegaMoeSpec(4096, 5120, 1792, 96, 8, 1.25, 40960, 16, None, rank, 32,
                                           mx_vector_policy="window2", dispatch_split=split, combine_split=split,
                                           compute_stages=6)
                        old = build_mxfp8_runtime(spec, forward=True, swiglu_row_only=True, ready_depth=8)
                        unmodified = build_mxfp8_runtime(spec, forward=True, swiglu_row_only=True, ready_depth=8)
                        original_bytes = serialize_runtime_config(unmodified.config)
                        new = with_source_row_ingress(spec, unmodified)
                        self.assertEqual(serialize_runtime_config(unmodified.config), original_bytes)
                        count = 0
                        for task in old.config.all_tasks[:old.config.task_num]:
                            if task.task_type == 165:
                                task.task_type = 166
                                count += 1
                        self.assertGreater(count, 0)
                        before_terminators = sum(task.task_type == 0 for task in
                                                 unmodified.config.all_tasks[:unmodified.config.task_num])
                        after_terminators = sum(task.task_type == 0 for task in
                                                new.config.all_tasks[:new.config.task_num])
                        self.assertEqual(after_terminators, before_terminators)
                        self.assertEqual(new.buffer_count, 29)
                        self.assertEqual(new.gmm_descriptors, old.gmm_descriptors)
                        self.assertEqual(new.quant_descriptors, old.quant_descriptors)
                        self.assertEqual(serialize_runtime_config(new.config), serialize_runtime_config(old.config))
                        self.assertTrue(new.source_row_fp8)
                        with self.assertRaisesRegex(ValueError, "no-grad B0"):
                            with_source_row_ingress(spec, build_mxfp8_runtime(spec, forward=False))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_forward_bindings_keep_the_original_business_roles(self) -> None:
        """Feature: Stable forward buffer roles.
        Description: Bind the shared workspace, weight axes and metadata with an original source tensor.
        Expectation: The 27 ordinary business roles retain their data identities and element-offset units.
        """
        buffers, _ = self._bindings()
        workspace = SimpleNamespace(expert_buffer=buffers[0], routed_buffer=torch.empty(8, 1024))
        offsets = torch.tensor([0, 4], dtype=torch.int64)
        sizes = torch.tensor([4, 4], dtype=torch.int32)
        groups = torch.tensor([4, 8], dtype=torch.int64)
        metadata = SimpleNamespace(dispatch_target_off=offsets, dispatch_src_off=offsets, dispatch_size=sizes,
                                    combine_target_off=offsets, combine_src_off=offsets, combine_size=sizes,
                                    group_list=groups)
        weights = tuple(SimpleNamespace(col_data=torch.empty(3), col_scale=torch.empty(2)) for _ in range(2))
        activations = (torch.empty(8, 256), torch.empty(8, 1024))
        x = (buffers[16], buffers[17], buffers[16], torch.empty(1, dtype=torch.uint8))
        other = tuple(torch.empty(8, 128, dtype=torch.uint8) for _ in range(4))
        runtime = SimpleNamespace(gmm=torch.empty(64, dtype=torch.uint8), quant=torch.empty(64, dtype=torch.uint8))
        result = _forward_graph_buffers(workspace, metadata, buffers[2], weights, activations, (x, other), runtime)
        self.assertEqual(len(result), 27)
        for position, value in ((0, workspace.expert_buffer), (2, buffers[2]), (5, weights[0].col_data),
                                (6, groups), (9, weights[1].col_data), (12, workspace.routed_buffer),
                                (16, x[0]), (17, x[1]), (23, weights[0].col_scale),
                                (24, weights[1].col_scale), (25, runtime.gmm), (26, runtime.quant)):
            with self.subTest(position=position):
                self.assertIs(result[position], value)
        for position in (1, 3, 13, 14):
            self.assertTrue(torch.equal(result[position], offsets * 1024))
        for position in (4, 15):
            self.assertTrue(torch.equal(result[position], sizes * 1024))
