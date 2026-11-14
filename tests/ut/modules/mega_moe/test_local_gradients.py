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
"""Verify local BF16 ownership, storage, protocol and older-payload rejection."""

from contextlib import ExitStack
import os
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_multicore._loader import NativeComponentUnavailableError
from hyper_multicore.modules.mega_moe import local_gradients
from hyper_multicore.modules.mega_moe.configuration import execution_options
from hyper_multicore.modules.mega_moe.function import _allocate_backward_intermediates
from hyper_multicore.modules.mega_moe.kernel_gradients import prepare_kernel_gradient_return
from hyper_multicore.replica import build_expert_replica_plan


class TestLocalGradientBf16(unittest.TestCase):
    """Only ranks without remote additions may change their home storage type."""

    def setUp(self) -> None:
        """Use a real placement whose second rank lends no expert gradients."""
        placement = build_expert_replica_plan([[100, 0, 0, 0]] * 4, 1)
        self.route = SimpleNamespace(plan=placement, rank=1)
        self.plan = SimpleNamespace(replica_w2_events=(32, 64), replica_w13_events=(96, 128),
                                    spec=SimpleNamespace(num_cube_cores=2, _local_gradient_bf16=True))
        self.provider = SimpleNamespace(
            kernel_gradients=True, gradient_bands=True,
            kernel_gradient_band_signals=Mock(return_value=(torch.zeros(1, 2, 2, 16, dtype=torch.int32), 2, 2)),
            kernel_gradient_signals=Mock(side_effect=[(epoch, 4096, 8192) for epoch in range(1, 10)]),
        )
        self.weights = tuple(torch.empty(1, 4, 8, dtype=torch.bfloat16) for _ in range(2))
        self.guests = tuple(torch.empty(1, 4, 8, dtype=torch.float32) for _ in range(2))

    def _prepare(self):
        with ExitStack() as context:
            context.enter_context(patch.object(torch.npu, "current_stream"))
            context.enter_context(patch.object(torch.Tensor, "record_stream"))
            return prepare_kernel_gradient_return(self.plan, self.route, self.provider, self.weights[1],
                                                  self.guests[1], gradient_w13=self.weights[0],
                                                  guest_w13=self.guests[0])

    def test_real_placement_encodes_bf16_home_and_fp32_guest_without_owned_edges(self) -> None:
        """Both projections preserve producer fences, distinct epochs and the original edge layout."""
        self.assertTrue(local_gradients.use_local_gradient_bf16(self.plan, self.route, self.provider, self.weights))
        result = self._prepare()
        raw = bytes(result.metadata.tolist())
        packed = struct.unpack("<" + "Q" * (len(raw) // 8), raw)
        records = packed[packed[1]:packed[2]], packed[packed[2]:]
        self.assertEqual(result.matrices, (1, 0))
        self.assertEqual(len(result.resources), 2)
        self.assertEqual([item[3] for item in records], [1, 2])
        for index, item in enumerate(records):
            projection = 1 - index
            self.assertEqual(item[0], 11)
            self.assertEqual(item[10:12], (1, 0))
            self.assertEqual(item[5:7], (self.weights[projection].data_ptr(), self.guests[projection].data_ptr()))
            self.assertEqual(item[18:], (0, 0, (self.plan.replica_w2_events, self.plan.replica_w13_events)[index][1]))
        self.assertEqual(int(result.completion.count_nonzero()), 0)
        self.assertEqual(len({tensor.untyped_storage().data_ptr() for tensor in result.resources}), 1)

    def test_owned_or_incomplete_paths_fall_back_before_allocating_or_reserving_epochs(self) -> None:
        """Missing proof never admits a BF16 buffer to an FP32 merge or full-expert publisher."""
        variants = (
            (self.route, {"_local_gradient_bf16": False}, {}, self.weights),
            (None, {}, {}, self.weights),
            (SimpleNamespace(plan=self.route.plan, rank=0), {}, {}, self.weights),
            (self.route, {}, {"kernel_gradients": False}, self.weights),
            (self.route, {}, {"gradient_bands": False}, self.weights),
            (self.route, {}, {}, (self.weights[0].float(), self.weights[1])),
        )
        for route, spec_options, provider_options, weights in variants:
            with self.subTest(route=route, spec=spec_options, provider=provider_options):
                spec = SimpleNamespace(**{**vars(self.plan.spec), **spec_options})
                plan = SimpleNamespace(**{**vars(self.plan), "spec": spec})
                provider = SimpleNamespace(**{**vars(self.provider), **provider_options})
                self.assertFalse(local_gradients.use_local_gradient_bf16(plan, route, provider, weights))
        self.provider.kernel_gradient_band_signals.return_value = None
        self.assertFalse(local_gradients.use_local_gradient_bf16(self.plan, self.route, self.provider, self.weights))
        with self.assertRaisesRegex(ValueError, "both band projections"):
            self._prepare()
        self.provider.kernel_gradient_signals.assert_not_called()

    def test_mixed_owner_dtypes_and_remote_owned_edges_are_rejected(self) -> None:
        """Callers cannot bypass the role check by directly packing the lower-level metadata."""
        self.weights = self.weights[0].float(), self.weights[1]
        with self.assertRaisesRegex(ValueError, "both band projections"):
            self._prepare()
        self.weights = tuple(weight.bfloat16() for weight in self.weights)
        self.route.rank = 0
        with self.assertRaisesRegex(ValueError, "no owned remote"):
            self._prepare()
        self.provider.kernel_gradient_signals.assert_not_called()

    def test_cold_spans_and_complete_zero_gradients_keep_bf16_storage(self) -> None:
        """Poisoned allocator data cannot reach untouched cold expert gradients."""
        spec = SimpleNamespace(replica_slots_per_rank=2, intermediate_size=4, hidden_size=8, dispatch_mode="push")
        weights = torch.empty(3, 8, 8, dtype=torch.bfloat16), torch.empty(3, 4, 8, dtype=torch.bfloat16)
        allocate = torch.empty_like

        def _poison(tensor: torch.Tensor, **kwargs) -> torch.Tensor:
            return allocate(tensor, **kwargs).fill_(float("nan"))

        with patch.object(torch, "empty_like", side_effect=_poison):
            value = _allocate_backward_intermediates(spec, 4, torch.empty(4, 8, dtype=torch.bfloat16),
                                                     *weights, empty_experts=(0, 2), local_bf16=True)
        for gradient in (value.grad_weight1, value.grad_weight2):
            self.assertEqual(gradient.dtype, torch.bfloat16)
            self.assertEqual(int(gradient[[0, 2]].count_nonzero()), 0)
            self.assertTrue(bool(gradient[1].isnan().all()))
        zeros = _allocate_backward_intermediates(spec, 4, torch.empty(4, 8, dtype=torch.bfloat16),
                                                 *weights, empty_experts=(0, 1, 2), local_bf16=True)
        self.assertEqual(int(zeros.grad_weight1.count_nonzero() + zeros.grad_weight2.count_nonzero()), 0)

    def test_explicit_request_rejects_old_native_marker(self) -> None:
        """An older adapter must fail before a mixed-dtype runtime image is submitted."""
        with patch.object(local_gradients.ops, "_load_native"), patch.object(
            local_gradients, "getattr", return_value=None, create=True
        ):
            with self.assertRaisesRegex(NativeComponentUnavailableError, "matching rebuilt"):
                local_gradients.require_local_gradient_native()
        for version in (0, 1, 2):
            with self.subTest(version=version), patch.object(local_gradients.ops, "_load_native"), patch.object(
                local_gradients, "getattr", return_value=lambda: version, create=True
            ):
                if version == 1:
                    local_gradients.require_local_gradient_native()
                else:
                    with self.assertRaises(NativeComponentUnavailableError):
                        local_gradients.require_local_gradient_native()

    def test_auto_negotiates_new_payload_and_keeps_old_payload_fallback(self) -> None:
        """Default selection must never submit protocol11 to an older payload."""
        for version in (None, 0, 1, 2):
            marker = None if version is None else lambda: version
            with self.subTest(version=version), patch.object(local_gradients.ops, "_load_native"), patch.object(
                local_gradients, "getattr", return_value=marker, create=True
            ):
                self.assertEqual(local_gradients.resolve_local_gradient_bf16(None), version == 1)
                if version == 1:
                    self.assertTrue(local_gradients.resolve_local_gradient_bf16(True))
                else:
                    with self.assertRaises(NativeComponentUnavailableError):
                        local_gradients.resolve_local_gradient_bf16(True)
        with patch.object(local_gradients.ops, "_load_native") as load:
            self.assertFalse(local_gradients.resolve_local_gradient_bf16(False))
            load.assert_not_called()
            with self.assertRaisesRegex(ValueError, "boolean or None"):
                local_gradients.resolve_local_gradient_bf16("auto")

    def test_default_control_is_auto_and_explicit_disable_stays_disabled(self) -> None:
        """Native capability negotiation is independent of route alpha or tensor contents."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(execution_options("push", 6)["_local_gradient_bf16"])
            self.assertFalse(execution_options("push", 0)["_local_gradient_bf16"])
            os.environ["HYPER_MULTICORE_REPLICA_LOCAL_BF16"] = "0"
            self.assertFalse(execution_options("push", 6)["_local_gradient_bf16"])
            os.environ["HYPER_MULTICORE_REPLICA_LOCAL_BF16"] = "1"
            self.assertTrue(execution_options("push", 6)["_local_gradient_bf16"])
            self.assertFalse(execution_options("push", 0)["_local_gradient_bf16"])
            os.environ["HYPER_MULTICORE_REPLICA_LOCAL_BF16"] = "invalid"
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                execution_options("push", 6)
