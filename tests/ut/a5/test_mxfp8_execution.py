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
"""Protect physical activation storage for row-only checkpoint forwards."""

import unittest
from types import SimpleNamespace

import torch

from tests.common.mark_utils import arg_mark

from hyper_multicore.modules.mega_moe.mxfp8_execution import (
    _forward_quantized_buffers, _quantized_buffers,
)


class TestMXFP8ActivationBuffers(unittest.TestCase):
    """Check real CPU storage bounds without native or distributed initialization."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_forward_axes_follow_the_same_backward_retention_contract(self) -> None:
        """Feature: Forward activation axis retention.
        Description: Exercise both activation families with and without backward retention.
        Expectation: Column data is shared only when unused; scales always remain independent.
        """
        spec = SimpleNamespace(hidden_size=256, intermediate_size=128, local_experts=8)
        for row_only in (False, True):
            with self.subTest(row_only=row_only):
                activations = _forward_quantized_buffers(spec, 128, "cpu", row_only=row_only)
                self.assertEqual([tuple(axis[0].shape) for axis in activations], [(128, 256), (128, 128)])
                for row, row_scale, column, column_scale in activations:
                    self.assertEqual(row.data_ptr() == column.data_ptr(), row_only)
                    self.assertNotEqual(row.data_ptr(), column_scale.data_ptr())
                    self.assertNotEqual(row_scale.data_ptr(), column_scale.data_ptr())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_dual_axis_storage_remains_independent(self) -> None:
        """Feature: Retained dual-axis activation storage.
        Description: Allocate ordinary row and column data with expert-padded scales.
        Expectation: Each binding has its own full-size storage.
        """
        buffers = _quantized_buffers(128, 256, 8, "cpu")
        self.assertEqual([tuple(value.shape) for value in buffers],
                         [(128, 256), (128, 4, 2), (128, 256), (10, 256, 2)])
        self.assertEqual(len({value.untyped_storage().data_ptr() for value in buffers}), 4)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_row_only_reuses_column_data_with_independent_scales(self) -> None:
        """Feature: Row-only unused column storage reuse.
        Description: Rebind column data while preserving a separate native scale binding.
        Expectation: All shapes and byte bounds remain valid with three allocations.
        """
        buffers = _quantized_buffers(128, 256, 8, "cpu", row_only=True)
        row, row_scale, column, column_scale = buffers
        self.assertEqual([tuple(value.shape) for value in buffers],
                         [(128, 256), (128, 4, 2), (128, 256), (10, 256, 2)])
        self.assertEqual(column.dtype, torch.float8_e4m3fn)
        self.assertEqual(column_scale.dtype, torch.uint8)
        self.assertTrue(all(value.is_contiguous() for value in buffers))
        self.assertEqual(row.data_ptr(), column.data_ptr())
        self.assertNotEqual(row.untyped_storage().data_ptr(), column_scale.untyped_storage().data_ptr())
        self.assertNotEqual(row.data_ptr(), row_scale.data_ptr())
        self.assertLessEqual(column_scale.numel(), column_scale.untyped_storage().nbytes())
        storages = {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes() for value in buffers}
        self.assertEqual(sum(storages.values()), 128 * 256 + 128 * 4 * 2 + 10 * 256 * 2)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_tiny_capacity_keeps_scale_binding_in_bounds(self) -> None:
        """Feature: Small-capacity packed scale storage.
        Description: Use fewer row bytes than expert-padded column scale bytes.
        Expectation: Scales receive an independent full-size allocation.
        """
        row, _, column, column_scale = _quantized_buffers(1, 64, 8, "cpu", row_only=True)
        self.assertEqual(row.data_ptr(), column.data_ptr())
        self.assertEqual(tuple(column_scale.shape), (9, 64, 2))
        self.assertNotEqual(row.untyped_storage().data_ptr(), column_scale.untyped_storage().data_ptr())
        self.assertLessEqual(column_scale.numel(), column_scale.untyped_storage().nbytes())
