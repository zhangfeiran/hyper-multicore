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
"""Verify compact owner partials and final output ownership independently of NPUs."""

from contextlib import ExitStack
import os
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_multicore._loader import NativeComponentUnavailableError
from hyper_multicore.modules.mega_moe import final_gradients
from hyper_multicore.modules.mega_moe.configuration import execution_options
from hyper_multicore.modules.mega_moe.kernel_gradients import prepare_kernel_gradient_return
from hyper_multicore.modules.mega_moe.local_gradients import use_local_gradient_bf16


class TestFinalGradients(unittest.TestCase):
    """The final edge owns BF16 output, including nonadjacent peer lists."""

    def setUp(self) -> None:
        """Keep two owned experts, one direct expert, and a cold owned local partial."""
        edges = [SimpleNamespace(owner_rank=0, owner_slot=owner, target_rank=peer, target_slot=3)
                 for peer, owner in [(3, 0), (1, 0), (2, 2)]]
        config = SimpleNamespace(home_experts=3, replica_slots_per_rank=1)
        placement = SimpleNamespace(config=config, transfers=tuple(edges), destination_counts=((0, 4, 8),))
        self.route = SimpleNamespace(plan=placement, rank=0)
        self.plan = SimpleNamespace(replica_w2_events=(32, 64, 96, 128),
                                    replica_w13_events=(160, 192, 224, 256),
                                    spec=SimpleNamespace(num_cube_cores=2, _local_gradient_bf16=True,
                                                         _final_gradient_bf16=True))
        self.weights = tuple(torch.empty(3, 4, 8, dtype=torch.bfloat16) for _ in range(2))
        self.guests = tuple(torch.empty(1, 4, 8, dtype=torch.float32) for _ in range(2))
        self.provider = SimpleNamespace(
            kernel_gradients=True, gradient_bands=True,
            kernel_gradient_band_signals=Mock(return_value=(torch.zeros(1, 2, 2, 16, dtype=torch.int32), 2, 2)),
            kernel_gradient_signals=Mock(side_effect=[(7, 4096, 8192), (8, 4096, 8192)]),
        )

    def _prepare(self):
        with ExitStack() as context:
            context.enter_context(patch.object(torch.npu, "current_stream"))
            record = context.enter_context(patch.object(torch.Tensor, "record_stream"))
            result = prepare_kernel_gradient_return(self.plan, self.route, self.provider, self.weights[1],
                                                   self.guests[1], gradient_w13=self.weights[0],
                                                   guest_w13=self.guests[0])
        return result, record.call_count

    def test_nonadjacent_last_peer_mapping_and_compact_cold_scratch(self) -> None:
        """Sorted edges 0,2,0 mark only the second and third edges as final."""
        layout = final_gradients.owner_gradient_layout(self.route)
        self.assertEqual(layout.home_to_scratch, (1, 0, 2))
        self.assertEqual(layout.last_edges, (0, 1, 1))
        self.assertEqual(layout.owners, (0, 2))
        scratch = final_gradients.allocate_owner_scratch(self.route, self.weights[0], layout)
        self.assertEqual(scratch.shape, (2, 4, 8))
        self.assertEqual(scratch.dtype, torch.float32)
        self.assertEqual(int(scratch[0].count_nonzero()), 0)
        self.assertNotEqual(scratch.untyped_storage().data_ptr(), self.weights[0].untyped_storage().data_ptr())

    def test_both_projection_records_have_explicit_distinct_scratch_and_lifetime(self) -> None:
        """Do not conflate hidden FP32 resources with the single int32 control allocation."""
        self.assertTrue(use_local_gradient_bf16(self.plan, self.route, self.provider, self.weights))
        result, streams = self._prepare()
        raw = bytes(result.metadata.tolist())
        values = struct.unpack("<" + "Q" * (len(raw) // 8), raw)
        records = values[values[1]:values[2]], values[values[2]:]
        self.assertEqual(streams, 6)
        self.assertEqual(len(result.resources), 2)
        self.assertEqual(len(result.scratch_resources), 2)
        for index, record in enumerate(records):
            with self.subTest(matrix=index):
                self.assertEqual(record[0], 13)
                self.assertEqual(record[10:12], (0, 3))
                self.assertEqual(record[18:24], (result.scratch_resources[index].data_ptr(), 3, 2, 36, 39, 42))
                self.assertEqual(record[24:36:4], (1, 2, 3))
                self.assertEqual(record[26:36:4], (0, 2, 0))
                self.assertEqual(record[36:], (1, 0, 2, 0, 1, 1))
                self.assertEqual(record[5], self.weights[1 - index].data_ptr())
        pointers = [buffer.data_ptr() for buffer in result.scratch_resources]
        self.assertEqual(len(set(pointers)), 2)
        self.assertEqual(len({buffer.untyped_storage().data_ptr() for buffer in result.resources}), 1)

    def test_missing_projection_or_guest_proof_cannot_allocate_a_partial_protocol(self) -> None:
        """Both projections must choose one consistent output contract before reserving epochs."""
        self.provider.kernel_gradient_band_signals.return_value = None
        with self.assertRaisesRegex(ValueError, "both band projections"):
            self._prepare()
        self.provider.kernel_gradient_signals.assert_not_called()
        self.route.plan.transfers[0].owner_slot = 3
        with self.assertRaisesRegex(ValueError, "outside the home"):
            final_gradients.owner_gradient_layout(self.route)

    def test_final_capability_is_separate_from_local_capability_and_default_is_auto(self) -> None:
        """An adapter which only supports protocol11 must reject protocol13 requests."""
        for version in (None, 0, 1, 2):
            marker = None if version is None else lambda: version
            with self.subTest(version=version), patch.object(final_gradients.ops, "_load_native"), patch.object(
                final_gradients, "getattr", return_value=marker, create=True
            ):
                if version == 1:
                    final_gradients.require_final_gradient_native()
                else:
                    with self.assertRaises(NativeComponentUnavailableError):
                        final_gradients.require_final_gradient_native()
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(execution_options("push", 6)["_final_gradient_bf16"])
            os.environ["HYPER_MULTICORE_REPLICA_FINAL_BF16"] = "1"
            self.assertTrue(execution_options("push", 6)["_final_gradient_bf16"])
            self.assertFalse(execution_options("push", 0)["_final_gradient_bf16"])
            os.environ["HYPER_MULTICORE_REPLICA_LOCAL_BF16"] = "0"
            with self.assertRaisesRegex(ValueError, "require local"):
                execution_options("push", 6)

    def test_auto_final_capability_falls_back_and_explicit_on_fails_closed(self) -> None:
        """Do not send protocol13 to absent, older or unknown future payload versions."""
        for version in (None, 0, 1, 2):
            marker = None if version is None else lambda: version
            with self.subTest(version=version), patch.object(final_gradients.ops, "_load_native"), patch.object(
                final_gradients, "getattr", return_value=marker, create=True
            ):
                self.assertEqual(final_gradients.resolve_final_gradient_bf16(None), version == 1)
                if version == 1:
                    self.assertTrue(final_gradients.resolve_final_gradient_bf16(True))
                else:
                    with self.assertRaises(NativeComponentUnavailableError):
                        final_gradients.resolve_final_gradient_bf16(True)
        with patch.object(final_gradients.ops, "_load_native") as load:
            self.assertFalse(final_gradients.resolve_final_gradient_bf16(False))
            load.assert_not_called()
        with self.assertRaisesRegex(ValueError, "boolean or None"):
            final_gradients.resolve_final_gradient_bf16("auto")

    def test_explicit_disable_and_zero_budget_preserve_the_local_output_contract(self) -> None:
        """Disabling the final store retains N3a, while disabling local output disables both."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(execution_options("push", 0)["_final_gradient_bf16"])
            os.environ["HYPER_MULTICORE_REPLICA_FINAL_BF16"] = "0"
            self.assertFalse(execution_options("push", 6)["_final_gradient_bf16"])
            self.assertIsNone(execution_options("push", 6)["_local_gradient_bf16"])
            del os.environ["HYPER_MULTICORE_REPLICA_FINAL_BF16"]
            os.environ["HYPER_MULTICORE_REPLICA_LOCAL_BF16"] = "0"
            self.assertFalse(execution_options("push", 6)["_final_gradient_bf16"])
            os.environ["HYPER_MULTICORE_REPLICA_FINAL_BF16"] = "invalid"
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                execution_options("push", 6)
