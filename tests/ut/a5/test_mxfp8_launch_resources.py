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
"""Keep ordinary MX placeholders reusable without sharing profiler-owned storage."""

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from hyper_multicore.modules.mega_moe import mxfp8_execution
from hyper_multicore.modules.mega_moe.mxfp8_module import _DeviceRuntime
from tests.common.mark_utils import arg_mark


def _runtime() -> _DeviceRuntime:
    return _DeviceRuntime(torch.empty(64, dtype=torch.uint8), torch.empty(0, dtype=torch.uint8),
                          torch.empty(0, dtype=torch.uint8), 1,
                          SimpleNamespace(kernel_gradients=False, weight_prefetch=False), None)


class TestMXFP8LaunchResources(unittest.TestCase):
    """Validate buffer ownership at the ordinary and profiled launch boundary."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_ordinary_calls_reuse_one_zeroed_placeholder_per_runtime(self) -> None:
        """Feature: Ordinary MX diagnostic storage reuse.
        Description: Launch twice with one runtime and inspect another independent runtime.
        Expectation: Calls reuse a zeroed byte vector while independent runtimes retain private storage.
        """
        runtime, other = _runtime(), _runtime()
        plan = SimpleNamespace(spec=SimpleNamespace(mx_vector_policy="legacy_odd"))
        execution = mxfp8_execution.MXFP8Execution(SimpleNamespace(metadata_transfer=None))
        with mock.patch.object(mxfp8_execution, "mx_profile_capture_enabled", return_value=False), \
                mock.patch.object(mxfp8_execution, "prepare_mxfp8_launch"), \
                mock.patch.object(mxfp8_execution, "launch_mxfp8_task_graph") as launch:
            for _ in range(2):
                execution._launch([], runtime, torch.empty(4, dtype=torch.uint8), plan)
        first, second = (call.args[3] for call in launch.call_args_list)
        self.assertIs(first, second)
        self.assertEqual(first.dtype, torch.uint8)
        self.assertEqual(first.shape, (64,))
        self.assertEqual(first.device, runtime.runtime.device)
        self.assertTrue(torch.equal(first, torch.zeros(64, dtype=torch.uint8)))
        self.assertNotEqual(first.data_ptr(), other.normal_diagnostic.data_ptr())

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_capture_and_invalid_launch_do_not_materialize_the_normal_placeholder(self) -> None:
        """Feature: Diagnostic capture and capacity isolation.
        Description: Request a profiled launch, then submit undersized event storage.
        Expectation: Capture owns its buffer and invalid launches fail before ordinary allocation.
        """
        runtime = _runtime()
        plan = SimpleNamespace(spec=SimpleNamespace(mx_vector_policy="legacy_odd", mx_ready_depth=0))
        execution = mxfp8_execution.MXFP8Execution(SimpleNamespace(metadata_transfer=None))
        with mock.patch.object(mxfp8_execution, "mx_profile_capture_enabled", return_value=True), \
                mock.patch.object(mxfp8_execution, "profile_mx_launch") as capture:
            execution._launch([], runtime, torch.empty(4, dtype=torch.uint8), plan)
        capture.assert_called_once()
        self.assertNotIn("normal_diagnostic", runtime.__dict__)
        with self.assertRaisesRegex(ValueError, "event storage"):
            execution._launch([], runtime, torch.empty(3, dtype=torch.uint8), plan)
        self.assertNotIn("normal_diagnostic", runtime.__dict__)
