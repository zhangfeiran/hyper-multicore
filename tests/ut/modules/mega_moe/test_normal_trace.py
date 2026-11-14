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
"""Validate ordinary graph identity, bounded records and explicit trace ownership."""

from dataclasses import replace
from contextvars import Context, copy_context
import struct
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from typing import Any

import torch

from hyper_multicore.modules.mega_moe import normal_trace
from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_multicore.modules.mega_moe.saved_dispatch import build_saved_dispatch
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.scheduler.runtime import serialize_runtime_config, runtime_cube_worker_capacity
from hyper_multicore.replica.recompute import create_recompute_contexts


def _spec(budget=0, compact=False):
    return MegaMoeSpec(128, 128, 128, 4 * (2 + budget), 2, 1.25, 1024, 4, None, 0, 4,
                       replica_slots_per_rank=budget, logical_num_experts=8,
                       _compact_schedule=compact, _compact_communication=compact)


def _fake_records(config, capacity=256):
    """Model physical worker ownership independently of the native trace writer."""
    cubes = runtime_cube_worker_capacity(config.num_workers)
    stride = 64 + 64 * capacity
    data = bytearray((cubes + config.num_workers) * stride)
    workers = {**{(1, worker): [] for worker in range(config.num_workers // 2)},
               **{(2, worker): [] for worker in range(config.num_workers)}}
    for core, queue in ((1, config.cube_task_indices), (2, config.vector_task_indices)):
        for index, task_id in enumerate(queue[:config.task_index_num[core - 1]]):
            worker = index % (config.num_workers // 2)
            if core == 2:
                worker = worker if config.vector_ready_queue else 2 * worker + 1
            workers[core, worker].append(task_id)
    for (core, worker), tasks in workers.items():
        slot = worker if core == 1 else cubes + worker
        base = slot * stride
        normal_trace.CORE_HEADER.pack_into(data, base, normal_trace.TRACE_MAGIC, 100, 1000 + 10 * len(tasks),
                                           len(tasks), 0, capacity, core, worker, 1, 0, 0)
        for index, task_id in enumerate(tasks):
            task = config.all_tasks[task_id]
            start = 110 + index * 10
            normal_trace.TASK_RECORD.pack_into(
                data, base + 64 + index * 64, start, start + 1, start + 3, start + 4,
                task_id, task.task_type, task.task_index, task.dependent_event, task.trigger_event,
                task.inputs[0].input_position if task.num_inputs else 0xFFFFFFFF,
                task.task_split_num, task.task_split_value,
            )
    return data


class _DeviceTensor:
    """Model asynchronous tensor preparation and fail on premature device readback."""

    def __init__(self, value: torch.Tensor) -> None:
        """Wrap host-owned test bytes with a device interface that forbids early reads."""
        self.value = value
        self.device = SimpleNamespace(type="npu")
        self.readback_allowed = False
        self.recorded_streams = []

    def clone(self) -> "_DeviceTensor":
        """Keep the device-only clone independent of its source bytes."""
        return _DeviceTensor(self.value.clone())

    def detach(self) -> "_DeviceTensor":
        """Retain the device proxy without performing a transfer."""
        return self

    def cpu(self) -> torch.Tensor:
        """Reject host readback until the test explicitly finishes the native launch."""
        if not self.readback_allowed:
            raise AssertionError("premature device-to-host transfer")
        return self.value

    def record_stream(self, stream: Any) -> None:
        """Record only allocator lifetime, without synchronizing the producer."""
        self.recorded_streams.append(stream)

    def __getitem__(self, key: Any) -> "_DeviceTensor":
        """Keep a writable device-view proxy for the asynchronous header update."""
        return _DeviceTensor(self.value[key])


class TestNormalTrace(unittest.TestCase):
    """Fail closed on mismatched payloads and incomplete diagnostic evidence."""

    def test_tag_preserves_graph_queues_credits_and_suffix(self):
        """Both ordinary and saved schedules change only reserved observation words."""
        for budget, compact in ((0, False), (1, False), (6, True)):
            spec = _spec(budget, compact)
            config = _build_runtime_artifacts(spec)[1]
            for saved in (False, True):
                source = build_saved_dispatch(config, spec) if saved else config
                image = serialize_runtime_config(source) + b"replica-suffix-preserved"
                for capacity in (1, 128, 1024, 16384):
                    with self.subTest(budget=budget, compact=compact, saved=saved, capacity=capacity):
                        tagged = normal_trace.trace_runtime_image(image, capacity)
                        self.assertEqual(tagged[:48], image[:48])
                        self.assertEqual(tagged[56:], image[56:])
                        self.assertEqual(struct.unpack_from("<II", tagged, 48),
                                         (normal_trace.TRACE_TAG, capacity))
                        self.assertEqual(serialize_runtime_config(source), image[:-24])

    def test_invalid_capacity_and_conflicting_diagnostics(self):
        """Unbounded, ambiguous and graph-transforming requests are rejected."""
        image = serialize_runtime_config(_build_runtime_artifacts(_spec())[1])
        for value in (True, 0, -1, 16385, 1.5, "8"):
            with self.assertRaisesRegex(ValueError, "capacity"):
                normal_trace.trace_runtime_image(image, value)
        for offset, value in ((20, 1), (48, 1), (52, 8), (56, 1)):
            altered = bytearray(image)
            struct.pack_into("<I", altered, offset, value)
            with self.assertRaisesRegex(ValueError, "NORMAL"):
                normal_trace.trace_runtime_image(bytes(altered), 8)
        with self.assertRaisesRegex(ValueError, "header"):
            normal_trace.trace_runtime_image(b"short", 8)

    def test_decode_requires_complete_ordinary_task_coverage(self):
        """All scheduled tasks are present even when many task bodies are empty."""
        spec = replace(_spec(1, True), _saved_dispatch_lanes=3)
        config = build_saved_dispatch(_build_runtime_artifacts(spec)[1], spec)
        image = normal_trace.trace_runtime_image(serialize_runtime_config(config), 256)
        data = _fake_records(config)
        workers = normal_trace.decode_task_trace(bytes(data), image, 256)
        self.assertEqual(len(workers), config.num_workers * 3 // 2)
        self.assertEqual(sum(len(row["records"]) for row in workers), sum(config.task_index_num[:2]))
        first = next(row["records"][0] for row in workers if row["records"])
        self.assertEqual((first["wait_cycles"], first["compute_cycles"], first["publish_cycles"]), (1, 2, 1))

    def test_decode_rejects_overflow_corruption_and_missing_tasks(self):
        """Headers, descriptor identity, cycle order and terminal coverage are all required."""
        config = _build_runtime_artifacts(_spec())[1]
        image = normal_trace.trace_runtime_image(serialize_runtime_config(config), 256)
        source = _fake_records(config)
        for offset, fmt, value, message in (
            (28, "I", 1, "overflow"), (0, "Q", 0, "header"), (16, "Q", 0, "exit"),
            (64 + 8, "Q", 0, "nonmonotonic"), (64 + 36, "I", 9999, "descriptor"),
            (24, "I", 0, "missing"),
        ):
            with self.subTest(offset=offset):
                data = bytearray(source)
                struct.pack_into("<" + fmt, data, offset, value)
                with self.assertRaisesRegex(ValueError, message):
                    normal_trace.decode_task_trace(bytes(data), image, 256)
        with self.assertRaisesRegex(ValueError, "byte length"):
            normal_trace.decode_task_trace(bytes(source[:-1]), image, 256)

    def test_scope_lifetime_and_ordinary_wrapper(self):
        """Disabled launches retain exact operands; active captures own independent images."""
        config = _build_runtime_artifacts(_spec())[1]
        runtime = torch.tensor(bytearray(serialize_runtime_config(config)), dtype=torch.uint8)
        groups = torch.tensor([1, 3], dtype=torch.int64)
        parameters = (*([groups] * 20), runtime, torch.zeros(4096, dtype=torch.uint8))
        attributes = (0, 4, 8, 128, 128)
        with patch.object(normal_trace.multicore_ops, "mega_moe") as ordinary:
            normal_trace.launch_normal_forward(parameters, attributes)
            ordinary.assert_called_once_with(*parameters, *attributes)
        namespace = SimpleNamespace(mega_moe_normal_trace_version=lambda: 1)
        with patch.object(normal_trace.multicore_ops, "_load_native"), \
             patch.object(normal_trace.torch.ops, "hyper_multicore", namespace), \
             patch.object(normal_trace.multicore_ops, "mega_moe_with_diagnostic_buffer") as native:
            with normal_trace.capture_forward_tasks(128) as session:
                with self.assertRaisesRegex(ValueError, "nested"):
                    with normal_trace.capture_forward_tasks(128):
                        self.fail("nested trace was accepted")
                with self.assertRaisesRegex(ValueError, "PROFILE"):
                    normal_trace.require_normal_trace_mode(True)
                normal_trace.launch_normal_forward(parameters, attributes)
            self.assertEqual(len(session.captures), 1)
            capture = session.captures[0]
            self.assertNotEqual(capture.runtime_config.data_ptr(), runtime.data_ptr())
            self.assertNotEqual(capture.buffer.data_ptr(), parameters[-1].data_ptr())
            self.assertEqual(bytes(runtime.tolist()), serialize_runtime_config(config))
            self.assertIs(native.call_args.args[-6], capture.buffer)
        normal_trace.require_normal_trace_mode(True)

    def test_capability_and_exception_restore_scope(self):
        """A missing payload cannot silently return empty traces; failures reset the scope."""
        with patch.object(normal_trace.multicore_ops, "_load_native"):
            for version in (None, 0, 2):
                namespace = SimpleNamespace()
                if version is not None:
                    namespace.mega_moe_normal_trace_version = lambda value=version: value
                with patch.object(normal_trace.torch.ops, "hyper_multicore", namespace):
                    with self.assertRaisesRegex(RuntimeError, "capability"):
                        with normal_trace.capture_forward_tasks():
                            self.fail("incompatible payload accepted")
            namespace = SimpleNamespace(mega_moe_normal_trace_version=lambda: 1)
            with patch.object(normal_trace.torch.ops, "hyper_multicore", namespace):
                with self.assertRaisesRegex(RuntimeError, "deliberate"):
                    with normal_trace.capture_forward_tasks():
                        raise RuntimeError("deliberate")
                normal_trace.require_normal_trace_mode(True)

    def test_decode_waits_for_original_producer_stream(self):
        """Readback from another current stream must observe the original writer's completion."""
        config = _build_runtime_artifacts(_spec())[1]
        image = normal_trace.trace_runtime_image(serialize_runtime_config(config), 256)
        buffer = torch.tensor(_fake_records(config), dtype=torch.uint8)
        producer = SimpleNamespace(synchronize=Mock())
        capture = normal_trace.ForwardTaskCapture(buffer, torch.empty(0), image, torch.empty(0),
                                                  256, (0, 4, 8, 128, 128), producer)
        decoded = capture.decode()
        producer.synchronize.assert_called_once_with()
        self.assertEqual(sum(len(row["records"]) for row in decoded), sum(config.task_index_num[:2]))

    def test_checkpoint_replay_recovers_owner_in_a_different_python_context(self):
        """NPU autograd replay must not depend on inheriting the caller's ContextVar values."""
        forward, replay = create_recompute_contexts()
        namespace = SimpleNamespace(mega_moe_normal_trace_version=lambda: 1)
        with patch.object(normal_trace.multicore_ops, "_load_native"), \
             patch.object(normal_trace.torch.ops, "hyper_multicore", namespace):
            with normal_trace.capture_forward_tasks() as session:
                with forward:
                    self.assertIs(normal_trace._current_trace(), session)

                def recompute() -> None:
                    """Run replay with the invocation owner but without caller ContextVars."""
                    with replay:
                        self.assertIsNone(normal_trace._CAPTURE.get())
                        self.assertIs(normal_trace._current_trace(), session)
                        with self.assertRaisesRegex(ValueError, "PROFILE"):
                            normal_trace.require_normal_trace_mode(True)

                Context().run(recompute)
            self.assertFalse(session.active)

    def test_closed_scope_does_not_capture_later_checkpoint_replay(self):
        """Invocation ownership keeps routing explicit after the original trace scope closes."""
        forward, replay = create_recompute_contexts()
        namespace = SimpleNamespace(mega_moe_normal_trace_version=lambda: 1)
        with patch.object(normal_trace.multicore_ops, "_load_native"), \
             patch.object(normal_trace.torch.ops, "hyper_multicore", namespace):
            with normal_trace.capture_forward_tasks() as session:
                with forward:
                    self.assertIs(normal_trace._current_trace(), session)

            def recompute() -> None:
                """Replay after the original explicit diagnostic scope has closed."""
                with replay:
                    self.assertIsNone(normal_trace._current_trace())

            Context().run(recompute)

    def test_checkpoint_trace_ownership_does_not_leak_to_another_invocation(self):
        """An unrelated replay cannot pick up another checkpoint's active diagnostic owner."""
        forward, _ = create_recompute_contexts()
        _, unrelated_replay = create_recompute_contexts()
        namespace = SimpleNamespace(mega_moe_normal_trace_version=lambda: 1)
        with patch.object(normal_trace.multicore_ops, "_load_native"), \
             patch.object(normal_trace.torch.ops, "hyper_multicore", namespace):
            with normal_trace.capture_forward_tasks() as session:
                with forward:
                    self.assertIs(normal_trace._current_trace(), session)

                def recompute() -> None:
                    """Enter an unrelated invocation inside a fresh execution context."""
                    with unrelated_replay:
                        self.assertIsNone(normal_trace._current_trace())

                Context().run(recompute)

    def test_one_checkpoint_rejects_multiple_trace_scope_owners(self):
        """Replay cannot ambiguously associate separate diagnostic sessions with one invocation."""
        forward, _ = create_recompute_contexts()
        namespace = SimpleNamespace(mega_moe_normal_trace_version=lambda: 1)
        with patch.object(normal_trace.multicore_ops, "_load_native"), \
             patch.object(normal_trace.torch.ops, "hyper_multicore", namespace):
            with forward:
                with normal_trace.capture_forward_tasks() as first:
                    self.assertIs(normal_trace._current_trace(), first)
                with normal_trace.capture_forward_tasks():
                    with self.assertRaisesRegex(ValueError, "multiple"):
                        normal_trace._current_trace()

    def test_inherited_context_cannot_reactivate_a_closed_trace_scope(self):
        """An inherited ContextVar reference cannot bypass the explicit closed-scope boundary."""
        forward, replay = create_recompute_contexts()
        namespace = SimpleNamespace(mega_moe_normal_trace_version=lambda: 1)
        with patch.object(normal_trace.multicore_ops, "_load_native"), \
             patch.object(normal_trace.torch.ops, "hyper_multicore", namespace):
            with normal_trace.capture_forward_tasks() as session:
                with forward:
                    self.assertIs(normal_trace._current_trace(), session)
                    inherited = copy_context()

            def recompute() -> None:
                """Replay from a Python context that retained the now-inactive owner."""
                with replay:
                    self.assertIs(normal_trace._CAPTURE.get(), session)
                    self.assertIsNone(normal_trace._current_trace())
                    normal_trace.require_normal_trace_mode(True)

            inherited.run(recompute)

    def test_npu_preparation_never_reads_back_or_synchronizes_before_launch(self):
        """Observation setup must not give asynchronous replica communication a host-drained head start."""
        config = _build_runtime_artifacts(_spec())[1]
        image = serialize_runtime_config(config)
        runtime = _DeviceTensor(torch.tensor(bytearray(image), dtype=torch.uint8))
        groups = _DeviceTensor(torch.tensor([1, 3], dtype=torch.int64))
        parameters = (*([groups] * 20), runtime, torch.empty(0, dtype=torch.uint8))
        stream = SimpleNamespace(synchronize=Mock())
        zeros = torch.zeros

        def allocate(size: int, **_kwargs: Any) -> _DeviceTensor:
            """Allocate fake device bytes without using the patched factory recursively."""
            return _DeviceTensor(zeros(size, dtype=torch.uint8))

        def copy_bytes(data: bytes, destination: _DeviceTensor) -> None:
            """Represent the existing asynchronous eight-byte upload on device storage."""
            destination.value.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))

        with patch.object(normal_trace.torch, "npu", SimpleNamespace(current_stream=lambda _device: stream),
                          create=True), \
             patch.object(normal_trace.torch, "zeros", side_effect=allocate), \
             patch.object(normal_trace.torch, "tensor", side_effect=AssertionError("host tensor conversion")), \
             patch.object(normal_trace, "_copy_device_bytes", side_effect=copy_bytes) as upload:
            session = normal_trace.ForwardTaskTrace(16)
            capture = session.prepare(parameters, (0, 4, 8, 128, 128), num_workers=8)
        stream.synchronize.assert_not_called()
        self.assertIsNone(capture._runtime_image)
        self.assertEqual(bytes(runtime.value.tolist()), image)
        expected = normal_trace.trace_runtime_image(image, 16)
        self.assertEqual(bytes(capture.runtime_config.value.tolist()), expected)
        self.assertEqual(upload.call_args.args[0], struct.pack("<II", normal_trace.TRACE_TAG, 16))
        capture.runtime_config.readback_allowed = True
        self.assertEqual(capture.runtime_image, expected)
        stream.synchronize.assert_called_once_with()
