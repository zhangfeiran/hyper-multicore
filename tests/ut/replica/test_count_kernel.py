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
"""Hardware-independent tests for replica count backend selection."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, sentinel

import torch

from hyper_multicore.replica import count_kernel
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestCountKernel(unittest.TestCase):
    """Keep compatible A2/A3 devices on integer counts with safe fallbacks."""

    def setUp(self) -> None:
        """Isolate the cached device capability from other unit tests."""
        count_kernel._counts_available.cache_clear()  # pylint: disable=protected-access
        count_kernel._count_expert_limit.cache_clear()  # pylint: disable=protected-access
        count_kernel._loaded_queue_operations.cache_clear()  # pylint: disable=protected-access

    def tearDown(self) -> None:
        """Discard mocked device capabilities before the next test."""
        count_kernel._counts_available.cache_clear()  # pylint: disable=protected-access
        count_kernel._count_expert_limit.cache_clear()  # pylint: disable=protected-access
        count_kernel._loaded_queue_operations.cache_clear()  # pylint: disable=protected-access

    def test_supported_devices_launch_integer_counts(self) -> None:
        """Feature: Exact integer route counts.
        Description: Select a backend for installed A2, A3 and incompatible device binaries.
        Expectation: Compatible devices launch once; other architectures retain the Torch fallback.
        """
        for name, supported in (
            ("Ascend910B", True),
            ("Ascend910B2", True),
            ("Ascend910_93", True),
            ("Ascend910_9391", True),
            ("Ascend910A", False),
            ("Ascend950", False),
        ):
            ids = SimpleNamespace(device=Mock(type="npu"), dtype=torch.int64, ndim=2, shape=(128, 8))
            npu = SimpleNamespace(get_device_name=Mock(return_value=name))
            with self.subTest(device=name), patch.object(
                type(count_kernel._LIBRARY_PATH), "is_file", return_value=True
            ), patch.object(count_kernel.torch, "npu", npu, create=True), patch.object(
                count_kernel, "launch_route_counts", return_value=sentinel.payload
            ) as launch:
                result = count_kernel.try_count_payload(ids, 96)
                if supported:
                    self.assertIs(result, sentinel.payload)
                    launch.assert_called_once_with(ids, 96)
                else:
                    self.assertIsNone(result)
                    launch.assert_not_called()

    def test_missing_binary_does_not_query_or_launch_device(self) -> None:
        """Feature: Optional count binary.
        Description: Request count acceleration when the shared library is absent.
        Expectation: Return the fallback without querying hardware or launching a kernel.
        """
        ids = SimpleNamespace(device=Mock(type="npu"), dtype=torch.int64, ndim=2, shape=(128, 8))
        npu = SimpleNamespace(get_device_name=Mock())
        with patch.object(type(count_kernel._LIBRARY_PATH), "is_file", return_value=False), patch.object(
            count_kernel.torch, "npu", npu, create=True
        ), patch.object(count_kernel, "launch_route_counts") as launch:
            self.assertIsNone(count_kernel.try_count_payload(ids, 96))
        npu.get_device_name.assert_not_called()
        launch.assert_not_called()

    def test_unsupported_inputs_do_not_reach_device_selection(self) -> None:
        """Feature: Bounded count inputs.
        Description: Supply CPU, dtype, TopK and expert-count inputs outside the native contract.
        Expectation: Fall back before any device capability query or kernel launch.
        """
        cases = (
            ("cpu", torch.int64, (128, 8), 96),
            ("npu", torch.float32, (128, 8), 96),
            ("npu", torch.int64, (128, 9), 96),
            ("npu", torch.int64, (128, 8), 513),
            ("npu", torch.int64, (128, 8), 0),
        )
        for device, dtype, shape, experts in cases:
            ids = SimpleNamespace(device=Mock(type=device), dtype=dtype, ndim=2, shape=shape)
            with self.subTest(device=device, dtype=dtype, shape=shape, experts=experts), patch.object(
                count_kernel, "_counts_available"
            ) as available, patch.object(count_kernel, "launch_route_counts") as launch:
                self.assertIsNone(count_kernel.try_count_payload(ids, experts))
            available.assert_not_called()
            launch.assert_not_called()

    def test_extended_experts_respect_installed_binary_capacity(self) -> None:
        """Retain the Torch fallback for old binaries and accelerate E384/E512 with ABI 2."""
        ids = SimpleNamespace(device=Mock(type="npu"), dtype=torch.int64, ndim=2, shape=(4096, 6))
        for abi, limit in ((1, 256), (2, 512)):
            library = Mock()
            library.replica_counts_abi_version.return_value = abi
            with patch.object(count_kernel, "_count_library", return_value=library):
                count_kernel._count_expert_limit.cache_clear()  # pylint: disable=protected-access
                for experts in (257, 384, 512):
                    with self.subTest(abi=abi, experts=experts), patch.object(
                        count_kernel, "_counts_available", return_value=True
                    ), patch.object(count_kernel, "launch_route_counts", return_value=sentinel.payload) as launch:
                        result = count_kernel.try_count_payload(ids, experts)
                        if experts <= limit:
                            self.assertIs(result, sentinel.payload)
                            launch.assert_called_once_with(ids, experts)
                        else:
                            self.assertIsNone(result)
                            launch.assert_not_called()

    def test_direct_extended_launch_rejects_legacy_binary_before_device_work(self) -> None:
        """Explicit E384 launches fail before allocating or calling the smaller ABI 1 kernel."""
        ids = SimpleNamespace(device=Mock(type="npu"), dtype=torch.int64, ndim=2, shape=(4096, 6))
        library = Mock()
        library.replica_counts_abi_version.return_value = 1
        with patch.object(count_kernel, "_count_library", return_value=library):
            with self.assertRaisesRegex(ValueError, "Rebuild the optional replica count library"):
                count_kernel.launch_route_counts(ids, 384)
        library.launch_replica_counts.assert_not_called()


    def test_route_payload_accepts_current_int32_ids_without_casting(self) -> None:
        """The optional payload path consumes the router's original integer width."""
        for dtype in (torch.int32, torch.int64):
            ids = SimpleNamespace(device=Mock(type="npu"), dtype=dtype, ndim=2, shape=(4096, 8),
                                  numel=Mock(return_value=4096 * 8))
            with self.subTest(dtype=dtype), patch.object(count_kernel, "_counts_available", return_value=True), \
                    patch.object(count_kernel, "_count_expert_limit", return_value=512), \
                    patch.object(count_kernel, "launch_route_payload", return_value=sentinel.payload) as launch:
                self.assertIs(count_kernel.try_route_payload(ids, 192, 31), sentinel.payload)
                launch.assert_called_once_with(ids, 192, 31)

    def test_route_payload_cpu_and_older_optional_library_retain_fallback(self) -> None:
        """Unsupported inputs never query devices; ABI 2 without the new symbol still works."""
        ids = torch.zeros(2, 8, dtype=torch.int32)
        with patch.object(count_kernel, "_counts_available") as available:
            self.assertIsNone(count_kernel.try_route_payload(ids, 192, 31))
            available.assert_not_called()
        with patch.object(count_kernel, "_count_library", return_value=SimpleNamespace()):
            count_kernel._route_payload_launch.cache_clear()
            self.assertIsNone(count_kernel._route_payload_launch())
        count_kernel._route_payload_launch.cache_clear()

    def test_direct_payload_rejects_bad_metadata_before_device_work(self) -> None:
        """Reject unsupported public launch metadata without opening the optional library."""
        with patch.object(count_kernel, "_route_payload_launch") as launch:
            with self.assertRaisesRegex(ValueError, "NPU"):
                count_kernel.launch_route_payload(torch.zeros(2, 8, dtype=torch.int32), 192, 31)
            launch.assert_not_called()

    def test_standalone_counts_do_not_load_or_cache_an_unloaded_adapter(self) -> None:
        """The independent count API can observe a later adapter load without initializing it."""
        with patch.object(count_kernel, "native_ready", side_effect=(False, True)), patch.object(
            count_kernel, "_loaded_queue_operations", return_value=sentinel.operations
        ) as probe:
            self.assertIsNone(count_kernel._queue_operations())
            probe.assert_not_called()
            self.assertIs(count_kernel._queue_operations(), sentinel.operations)
            probe.assert_called_once_with()

    def test_older_adapter_and_current_queue_capability(self) -> None:
        """Only a declared queue ABI selects the new entrypoints; old adapters retain ctypes."""
        for version in (None, 0, 1, 2):
            namespace = SimpleNamespace()
            if version is not None:
                namespace.replica_count_queue_version = Mock(return_value=version)
            if version == 1:
                namespace.replica_route_payload = sentinel.payload_op
                namespace.replica_route_summary = sentinel.summary_op
            count_kernel._loaded_queue_operations.cache_clear()
            with self.subTest(version=version), patch.object(count_kernel.torch.ops, "hyper_multicore", namespace):
                actual = count_kernel._loaded_queue_operations()
                if version == 1:
                    expected = (sentinel.payload_op, sentinel.summary_op, str(count_kernel._LIBRARY_PATH))
                    self.assertEqual(actual, expected)
                else:
                    self.assertIsNone(actual)

    def test_queued_payload_retains_current_dtype_and_controls(self) -> None:
        """The queue entry receives current operands directly and never opens a raw stream."""
        ids = SimpleNamespace(device=Mock(type="npu"), dtype=torch.int32, ndim=2, shape=(4096, 8),
                              numel=Mock(return_value=32768))
        payload_op = Mock(return_value=sentinel.result)
        operations = count_kernel._QueueOperations(payload_op, sentinel.summary_op, "owned-count-library.so")
        with patch.object(count_kernel, "_route_payload_launch", return_value=sentinel.legacy_launch), patch.object(
            count_kernel, "_route_summary_launch", return_value=sentinel.summary_launch
        ), patch.object(count_kernel, "_queue_operations", return_value=operations):
            self.assertIs(count_kernel.launch_route_payload(ids, 192, 31), sentinel.result)
        payload_op.assert_called_once_with(ids, 192, 31, "owned-count-library.so")

    def test_queue_failure_propagates_without_a_second_launch(self) -> None:
        """A declared native queue failure must not silently relaunch the kernel through ctypes."""
        ids = SimpleNamespace(device=Mock(type="npu"), dtype=torch.int64, ndim=2, shape=(4096, 8),
                              numel=Mock(return_value=32768))
        legacy_launch = Mock()
        payload_op = Mock(side_effect=RuntimeError("queue launch failed"))
        operations = count_kernel._QueueOperations(payload_op, sentinel.summary, "counts.so")
        with patch.object(count_kernel, "_route_payload_launch", return_value=legacy_launch), patch.object(
            count_kernel, "_route_summary_launch", return_value=sentinel.summary_launch
        ), patch.object(count_kernel, "_queue_operations", return_value=operations):
            with self.assertRaisesRegex(RuntimeError, "queue launch failed"):
                count_kernel.launch_route_payload(ids, 192, 31)
        legacy_launch.assert_not_called()
