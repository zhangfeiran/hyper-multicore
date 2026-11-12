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
"""Layout and runtime metadata for device-initiated MegaMoE RDMA communication."""

from __future__ import annotations

from collections.abc import Mapping
import os
import struct
from typing import Any

import torch

from hyper_multicore._loader import NativeComponentUnavailableError
from hyper_multicore.scheduler.config import (
    EVENT_INVALID_ID, TaskType, event_workspace_bytes, mega_moe_event_capacity,
)
from hyper_multicore.torch import ops as multicore_ops

RDMA_RUNTIME_MAGIC = 0x3152444D4153484D
RDMA_RUNTIME_BYTES = 64
DATA_ENGINE_ENV = "HYPER_MULTICORE_SHMEM_DATA_ENGINE"


def require_native() -> None:
    """Require peer-owned queues and registered remote addresses before any RDMA launch."""
    multicore_ops._load_native()  # pylint: disable=protected-access
    try:
        if torch.ops.hyper_multicore.mega_moe_rdma_version() != 2:
            raise RuntimeError("Unsupported MegaMoE RDMA address and peer-owner protocol")
    except (AttributeError, RuntimeError) as error:
        raise NativeComponentUnavailableError("MegaMoE RDMA requires a rebuilt multicore payload") from error


def configured_engine() -> str:
    """Validate the process-wide engine before resource compatibility is frozen."""
    engine = os.environ.get(DATA_ENGINE_ENV, "mte")
    if engine not in ("mte", "mte_roce", "roce"):
        raise ValueError(f"{DATA_ENGINE_ENV} must be mte, mte_roce or roce")
    return engine


def enabled(specification: Any) -> bool:
    """Return whether a mapping or bound specification uses the RDMA task scheduler."""
    engine = (specification.get("shmem_data_engine", "mte") if isinstance(specification, Mapping)
              else getattr(specification, "shmem_data_engine", "mte"))
    return engine in ("mte_roce", "roce")


def layout(ep_size: int, num_experts: int, routed_slots: int,
           dispatch_split: int = 128, combine_split: int = 128) -> tuple[int, int, int, int]:
    """Return credit, progress, immutable-value offsets and the complete event allocation in bytes."""
    credits = event_workspace_bytes(ep_size, num_experts)
    matrix_bytes = ep_size * mega_moe_event_capacity(num_experts, ep_size) * 4
    progress = credits + matrix_bytes
    # Each peer has an independent symmetric receive area for indexed GET metadata.
    index_bytes = ep_size * ((dispatch_split * 4 + 127) // 128 * 128)
    values = progress + matrix_bytes + index_bytes
    count = num_experts * ((routed_slots + min(dispatch_split, combine_split) - 1)
                           // min(dispatch_split, combine_split)) + 1
    total = values + count * 4
    return credits, progress, values, (total + 511) // 512 * 512


def workspace_layout(specification: Mapping[str, Any], routed_slots: int) -> tuple[int, int, int, int]:
    """Resolve the same physical event layout for reservation, allocation and restart."""
    return layout(specification["ep_size"], specification["num_experts"], routed_slots,
                  specification.get("dispatch_split", 128), specification.get("combine_split", 128))


def prepare_events(workspace: Any, events: torch.Tensor, initialized: bool) -> None:
    """Clear per-source credits while preserving immutable DMA sources across invocations."""
    credits, _, values, _ = layout(workspace.rdma_ep_size, workspace.rdma_num_experts,
                                  workspace.routed_slots, workspace.rdma_dispatch_split,
                                  workspace.rdma_combine_split)
    events[credits:values].zero_()
    if not initialized:
        count = (workspace.rdma_value_bytes // 4)
        events[values:values + count * 4].view(torch.int32).copy_(
            torch.arange(count, dtype=torch.int32, device=events.device))


def runtime_suffix(spec: Any, events: torch.Tensor, *, row_indices: torch.Tensor | None = None,
                   source_rows: int | None = None, stage_control_source: bool = True) -> bytes:
    """Describe symmetric counters and immutable notification operands for one leased kernel."""
    credits, progress, values, _ = layout(spec.ep_size, spec.num_experts, spec.routed_slots,
                                        spec.dispatch_split, spec.combine_split)
    count = spec.num_experts * ((spec.routed_slots + min(spec.dispatch_split, spec.combine_split) - 1)
                               // min(spec.dispatch_split, spec.combine_split)) + 1
    return struct.pack("<8Q", RDMA_RUNTIME_MAGIC, events.data_ptr() + credits,
                       events.data_ptr() + progress, events.data_ptr() + values, count,
                       int(spec.shmem_data_engine == "roce") | int(stage_control_source) << 1,
                       0 if row_indices is None else row_indices.data_ptr(),
                       max(1, spec.local_num_tokens if source_rows is None else source_rows))


def communication_profile_records(spec: Any, configuration: Any) -> int:
    """Budget the busiest dedicated AIV, independently of the ordinary odd-worker queue."""
    if not enabled(spec):
        return 0
    records = [0] * min(spec.ep_size, spec.num_cube_cores)
    for slot in range(configuration.task_index_num[1]):
        task = configuration.all_tasks[int(configuration.vector_task_indices[slot])]
        if task.task_type not in (TaskType.TASK_SHMEM_GET_MEM, TaskType.TASK_SHMEM_PUT_MEM_SIGNAL):
            continue
        peer = task.task_index // (task.task_split_num // spec.ep_size)
        count = 1 + int(task.dependent_event != EVENT_INVALID_ID)
        count += int(task.task_type != TaskType.TASK_SHMEM_PUT_MEM_SIGNAL)
        records[peer % len(records)] += count
    return max(records, default=0)
