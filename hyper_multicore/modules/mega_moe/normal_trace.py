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
"""Private bounded forward trace retaining the ordinary task graph and ready queue."""

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import struct
from typing import Any

import torch

from hyper_multicore.scheduler.config import RuntimeConfigC
from hyper_multicore.scheduler.runtime import allocate_runtime_config, runtime_cube_worker_capacity
from hyper_multicore.modules.recompute import get_recompute_state
from hyper_multicore.modules.device_metadata import _copy_device_bytes
from hyper_multicore.torch import ops as multicore_ops


TRACE_TAG = 0x4E545231
TRACE_MAGIC = 0x3152434152544E
TRACE_VERSION = 1
MAX_CAPACITY = 16384
CORE_HEADER = struct.Struct("<QQQ6I2Q")
TASK_RECORD = struct.Struct("<4Q8I")
_CAPTURE: ContextVar[Any] = ContextVar("mega_moe_normal_forward_trace", default=None)
_CHECKPOINT_TRACE_KEY = object()


def _current_trace() -> Any:
    """Recover an active explicit owner when NPU checkpoint replay changes Python contexts."""
    session = _CAPTURE.get()
    if session is not None and not session.active:
        session = None
    state = get_recompute_state()
    if state is None or (session is None and not state.is_recomputing):
        return session
    registry = state.get_resource(_CHECKPOINT_TRACE_KEY, dict)
    if session is not None:
        owner = registry.setdefault("owner", session)
        if owner is not session:
            raise ValueError("one checkpoint invocation cannot belong to multiple normal trace scopes")
        return session
    owner = registry.get("owner")
    return owner if owner is not None and owner.active else None


def _check_capacity(capacity: int) -> None:
    if isinstance(capacity, bool) or not isinstance(capacity, int) or not 1 <= capacity <= MAX_CAPACITY:
        raise ValueError(f"normal task trace capacity must be an integer in [1, {MAX_CAPACITY}]")


def trace_runtime_image(image: bytes, capacity: int) -> bytes:
    """Clone a normal image, changing only two reserved observation header words.

    Args:
        image: Already split and guest-pruned runtime including any replica suffix.
        capacity: Maximum task records per worker, including empty tasks.

    Returns:
        Independent bytes whose descriptors, queues and event credits are unchanged.
    """
    _check_capacity(capacity)
    if len(image) < 64:
        raise ValueError("normal task trace requires a complete runtime header")
    header = RuntimeConfigC.from_buffer_copy(image)
    if header.diagnostic_mode != 0 or any(header._padding):  # pylint: disable=protected-access
        raise ValueError("normal task trace requires NORMAL mode and unused reserved header words")
    if header.num_workers < 2 or header.num_workers % 2:
        raise ValueError("normal task trace requires paired Cube/Vector workers")
    result = bytearray(image)
    struct.pack_into("<II", result, 48, TRACE_TAG, capacity)
    return bytes(result)


def _runtime(image: bytes) -> Any:
    """Read physical arrays using their serialized capacities, preserving the suffix."""
    header = RuntimeConfigC.from_buffer_copy(image)
    if (header.task_capacity > len(image) // 576 or header.event_capacity > len(image) // 20
            or header.num_workers < 2 or header.num_workers % 2):
        raise ValueError("normal task trace runtime capacities exceed supplied storage")
    config = allocate_runtime_config(header.task_capacity, header.event_capacity, 0, header.num_workers)
    if len(image) < type(config).dynamic_data.offset + 16:
        raise ValueError("normal task trace runtime is truncated")
    prefix = bytearray(bytes(config))
    prefix[:min(len(prefix), len(image))] = image[:len(prefix)]
    return type(config).from_buffer_copy(prefix)


def _expected_queues(config: Any) -> dict[int, Counter]:
    result = {}
    for core, queue in ((1, config.cube_task_indices), (2, config.vector_task_indices)):
        count = int(config.task_index_num[core - 1])
        if not 0 <= count <= config.task_capacity:
            raise ValueError("normal task trace queue count exceeds runtime storage")
        result[core] = Counter(int(value) for value in queue[:count])
    return result


def _decode_records(data: bytes, offset: int, count: int, config: Any, entry: int, exit_cycle: int) -> list[dict]:
    records = []
    previous = entry
    fields = ("task_id", "task_type", "task_index", "dependent_event", "trigger_event",
              "input_position", "split_num", "split_value")
    for index in range(count):
        values = TASK_RECORD.unpack_from(data, offset + index * TASK_RECORD.size)
        selected, observed, computed, published = values[:4]
        if not previous <= selected <= observed <= computed <= published <= exit_cycle:
            raise ValueError("normal task trace cycles are incomplete or nonmonotonic")
        row = dict(zip(fields, values[4:]))
        if row["task_id"] >= config.task_capacity:
            raise ValueError("normal task trace task ID exceeds storage")
        task = config.all_tasks[row["task_id"]]
        expected = (task.task_type, task.task_index, task.dependent_event, task.trigger_event,
                    task.inputs[0].input_position if task.num_inputs else 0xFFFFFFFF,
                    task.task_split_num, task.task_split_value)
        if tuple(values[5:]) != expected:
            raise ValueError("normal task trace descriptor differs from the selected ordinary graph")
        row.update(selected_cycle=selected, dependency_observed_cycle=observed,
                   compute_returned_cycle=computed, event_published_cycle=published,
                   wait_cycles=observed - selected, compute_cycles=computed - observed,
                   publish_cycles=published - computed, gap_before_cycles=selected - previous)
        records.append(row)
        previous = published
    return records


def decode_task_trace(data: bytes, image: bytes, capacity: int) -> list[dict]:
    """Validate every worker, reject overflow, and require complete scheduled task coverage.

    Args:
        data: Completed trace bytes, copied after the launch stream has completed.
        image: Exact tagged runtime passed to the native kernel.
        capacity: Expected per-worker record capacity.

    Returns:
        Per-worker local intervals. Cross-worker cycle alignment is not assumed.
    """
    _check_capacity(capacity)
    config = _runtime(image)
    if tuple(config._padding) != (TRACE_TAG, capacity, 0, 0) or config.diagnostic_mode != 0:  # pylint: disable=protected-access
        raise ValueError("normal task trace runtime tag or capacity differs")
    expected = _expected_queues(config)
    cubes = runtime_cube_worker_capacity(config.num_workers)
    stride = CORE_HEADER.size + capacity * TASK_RECORD.size
    if len(data) != (cubes + config.num_workers) * stride:
        raise ValueError("normal task trace buffer byte length differs")
    seen = {1: Counter(), 2: Counter()}
    workers = []
    slots = [(1, worker, worker) for worker in range(config.num_workers // 2)]
    slots += [(2, worker, cubes + worker) for worker in range(config.num_workers)]
    for core, worker, slot in slots:
        offset = slot * stride
        magic, entry, exit_cycle, count, dropped, actual_capacity, kind, identity, version, *reserved = \
            CORE_HEADER.unpack_from(data, offset)
        if (magic, actual_capacity, kind, identity, version, *reserved) != \
                (TRACE_MAGIC, capacity, core, worker, TRACE_VERSION, 0, 0):
            raise ValueError("normal task trace worker header differs or is incomplete")
        if dropped or count > capacity or exit_cycle < entry:
            raise ValueError("normal task trace overflow or incomplete worker exit")
        records = _decode_records(data, offset + CORE_HEADER.size, count, config, entry, exit_cycle)
        seen[core].update(row["task_id"] for row in records)
        workers.append({"core": core, "worker": worker, "entry_cycle": entry, "exit_cycle": exit_cycle,
                        "records": records})
    if seen != expected:
        raise ValueError("normal task trace has missing, duplicate or unexpected scheduled tasks")
    return workers


@dataclass
class ForwardTaskCapture:
    """Keep independent storage alive until explicit post-launch diagnostic decoding."""

    buffer: torch.Tensor
    runtime_config: torch.Tensor
    _runtime_image: bytes | None
    group_list: torch.Tensor
    capacity: int
    attributes: tuple[int, ...]
    producer_stream: Any = None

    @property
    def runtime_image(self) -> bytes:
        """Read the per-launch runtime only after its producer stream has completed."""
        if self._runtime_image is None:
            if self.producer_stream is not None:
                self.producer_stream.synchronize()
            self._runtime_image = self.runtime_config.detach().cpu().numpy().tobytes()
        return self._runtime_image

    def decode(self) -> list[dict]:
        """Wait for the producer stream before explicit diagnostic readback on any stream."""
        if self.producer_stream is not None:
            self.producer_stream.synchronize()
        data = self.buffer.detach().cpu().numpy().tobytes()
        return decode_task_trace(data, self.runtime_image, self.capacity)


@dataclass
class ForwardTaskTrace:
    """Collect explicitly scoped launches without changing the plan or its resource key."""

    capacity: int
    captures: list[ForwardTaskCapture] = field(default_factory=list)
    active: bool = field(default=False, init=False)

    def prepare(
        self, parameters: tuple, attributes: tuple[int, ...], *, num_workers: int | None = None
    ) -> ForwardTaskCapture:
        """Own tagged launch bytes and a disjoint zero-initialized diagnostic allocation."""
        runtime = parameters[-2]
        if runtime.device.type == "npu":
            if num_workers is None:
                raise ValueError("ordinary NPU tracing requires the prepared plan's worker count")
            workers = num_workers
            image = None
            tagged = runtime.clone()
            _copy_device_bytes(struct.pack("<II", TRACE_TAG, self.capacity), tagged[48:56])
        else:
            image = trace_runtime_image(runtime.detach().cpu().numpy().tobytes(), self.capacity)
            workers = struct.unpack_from("<I", image, 4)[0]
            tagged = torch.tensor(bytearray(image), dtype=torch.uint8, device=runtime.device)
        slots = runtime_cube_worker_capacity(workers) + workers
        if workers < 2 or workers % 2:
            raise ValueError("ordinary NPU tracing requires paired Cube/Vector workers")
        size = slots * (CORE_HEADER.size + self.capacity * TASK_RECORD.size)
        buffer = torch.zeros(size, dtype=torch.uint8, device=runtime.device)
        groups = parameters[6].detach().clone()
        capture = ForwardTaskCapture(buffer, tagged, image, groups, self.capacity, attributes)
        if runtime.device.type == "npu":
            stream = torch.npu.current_stream(runtime.device)
            capture.producer_stream = stream
            for tensor in (tagged, buffer, groups):
                tensor.record_stream(stream)
        self.captures.append(capture)
        return capture


@contextmanager
def capture_forward_tasks(capacity: int = 1024) -> Iterator[ForwardTaskTrace]:
    """Trace ordinary forwards in this context, preserving NORMAL mode and the selected image.

    Args:
        capacity: Maximum task records per physical worker. Overflow invalidates decoding.

    Yields:
        Owner of per-launch buffers; decoding explicitly synchronizes those buffers.
    """
    _check_capacity(capacity)
    if _current_trace() is not None:
        raise ValueError("normal task trace contexts cannot be nested")
    multicore_ops._load_native()  # pylint: disable=protected-access
    version = getattr(torch.ops.hyper_multicore, "mega_moe_normal_trace_version", None)
    if version is None or version() != TRACE_VERSION:
        raise RuntimeError("normal task trace requires matching native capability v1; rebuild the complete payload")
    session = ForwardTaskTrace(capacity)
    session.active = True
    token = _CAPTURE.set(session)
    try:
        yield session
    finally:
        session.active = False
        _CAPTURE.reset(token)


def launch_normal_forward(
    parameters: tuple, attributes: tuple[int, ...], *, num_workers: int | None = None
) -> None:
    """Use the original wrapper unless an explicit scoped ordinary trace is active."""
    session = _current_trace()
    if session is None:
        multicore_ops.mega_moe(*parameters, *attributes)
        return
    capture = session.prepare(parameters, attributes, num_workers=num_workers)
    multicore_ops.mega_moe_with_diagnostic_buffer(
        *parameters[:-2], capture.runtime_config, parameters[-1], capture.buffer, *attributes,
    )


def require_normal_trace_mode(diagnostic_active: bool) -> None:
    """Reject a conflicting graph-transforming diagnostic instead of silently tracing nothing."""
    if diagnostic_active and _current_trace() is not None:
        raise ValueError("normal task trace cannot run with PROFILE or TENSOR_DUMP diagnostics")
