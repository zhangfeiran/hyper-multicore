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
"""Verify exact native byte arguments without NPU initialization or persistent view caches."""

import ctypes
import unittest

import torch

from hyper_multicore.modules.mega_moe.mxfp8_ops import _byte_aliases
from tests.common.mark_utils import arg_mark


class TestMXFP8ByteAliases(unittest.TestCase):
    """Cover byte interpretation, physical offsets and changing root storage."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_aliases_match_current_physical_bytes_without_copying(self) -> None:
        """Feature: Exact MX native byte views.
        Description: Pass mixed dtypes, tensor ranks, empty buffers and offset slices.
        Expectation: Every output is a contiguous byte vector aliasing the exact original bytes.
        """
        floating = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        buffers = [floating, floating.to(torch.bfloat16), floating.to(torch.float16),
                   floating.to(torch.int64), floating.to(torch.int32),
                   torch.arange(48, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(2, 3, 8),
                   torch.arange(48, dtype=torch.uint8).reshape(2, 3, 8),
                   torch.arange(48, dtype=torch.int32)[3:27], torch.arange(48, dtype=torch.uint8)[5:37],
                   torch.empty(0, dtype=torch.float32), torch.tensor(17, dtype=torch.uint8)]
        aliases = _byte_aliases(buffers)
        self.assertEqual(len(aliases), len(buffers))
        for source, alias in zip(buffers, aliases):
            with self.subTest(dtype=source.dtype, shape=source.shape):
                size = source.numel() * source.element_size()
                self.assertEqual(alias.dtype, torch.uint8)
                self.assertEqual(alias.shape, (size,))
                self.assertTrue(alias.is_contiguous())
                self.assertEqual(alias.data_ptr(), source.data_ptr())
                self.assertEqual(alias.untyped_storage().data_ptr(), source.untyped_storage().data_ptr())
                self.assertEqual(bytes(alias.tolist()), ctypes.string_at(source.data_ptr(), size))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_new_root_storage_and_input_versions_are_preserved(self) -> None:
        """Feature: Fresh MX physical root views.
        Description: Replace the root tensor between calls and mutate a byte view of an integer slice.
        Expectation: No old storage is reused, no graph is retained, and alias writes reach the input.
        """
        first = torch.arange(24, dtype=torch.float32).requires_grad_()
        second = first.detach().clone().requires_grad_()
        version = second._version
        before, after = _byte_aliases([first])[0], _byte_aliases([second])[0]
        self.assertNotEqual(before.data_ptr(), after.data_ptr())
        self.assertEqual(after.data_ptr(), second.data_ptr())
        self.assertEqual(second._version, version)
        self.assertFalse(after.requires_grad)
        self.assertIsNone(after.grad_fn)
        integers = torch.arange(24, dtype=torch.int32)[3:11]
        alias = _byte_aliases([integers])[0]
        changed = int(alias[0]) ^ 0xFF
        alias[0] = changed
        self.assertEqual(ctypes.string_at(integers.data_ptr(), 1), bytes([changed]))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_noncontiguous_input_cannot_trigger_an_implicit_copy(self) -> None:
        """Feature: MX native contiguous argument contract.
        Description: Supply a transposed tensor that would require a materialized reshape.
        Expectation: The boundary rejects it instead of allocating copied physical weights.
        """
        source = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()
        with self.assertRaisesRegex(ValueError, "implicit physical weight copies"):
            _byte_aliases([source])
