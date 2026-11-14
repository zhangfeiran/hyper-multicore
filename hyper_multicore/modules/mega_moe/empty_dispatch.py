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
"""Batch empty-tile credits for supported Push communication tasks."""

from collections import defaultdict
import importlib

import torch

from hyper_multicore._loader import NativeComponentUnavailableError
from hyper_multicore.scheduler.config import RuntimeConfigC, TaskType


EMPTY_DISPATCH_MARKER = 0x40000000


def mark_empty_dispatch(config: RuntimeConfigC) -> None:
    """Keep all descriptors and thresholds, marking only Push dispatch PUTs."""
    for task_id in config.vector_task_indices[: config.task_index_num[1]]:
        task = config.all_tasks[task_id]
        if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL and task.inputs[3].input_position == 4:
            if task.extra_value_0:
                raise ValueError("Empty-dispatch signal marker conflicts with an existing task policy")
            task.extra_value_0 = EMPTY_DISPATCH_MARKER


def _combine_credit_is_exact(task, group):
    """Reject conflicting policies and noncanonical Combine descriptor offsets."""
    return not (
        task.extra_value_0
        or task.num_inputs != 4
        or any((task.inputs[index].base_ptr_offset != group for index in (0, 2, 3)))
    )


def _validate_combine_groups(groups, expected):
    """Require every peer-expert tile to share its exact dependency and completion."""
    if groups.keys() != expected:
        raise ValueError("Incomplete home Combine groups")
    for group in groups.values():
        tiles, dependency, trigger = group[0][2:]
        if sorted((item[1] for item in group)) != list(range(tiles)) or any(
            (item[2:] != (tiles, dependency, trigger) for item in group)
        ):
            raise ValueError("Combine batch spans different producer or completion events")


def mark_empty_combine(
    config: RuntimeConfigC, *, ep_size: int, local_experts: int, home_experts: int, size_position: int
) -> None:
    """Mark complete home Combine groups, which guest-prefix pruning never splits."""
    groups = defaultdict(list)
    experts = ep_size * local_experts
    for task_id in config.vector_task_indices[: config.task_index_num[1]]:
        task = config.all_tasks[task_id]
        if task.task_type != TaskType.TASK_SHMEM_PUT_MEM_SIGNAL or task.inputs[3].input_position != size_position:
            continue
        tiles, remainder = divmod(task.task_split_num, experts)
        if remainder or tiles <= 0:
            raise ValueError("Invalid Combine task geometry for empty credit batching")
        group, tile = divmod(task.task_index, tiles)
        if not 0 <= group < experts:
            raise ValueError("Invalid Combine peer-expert group")
        if group % local_experts >= home_experts:
            continue
        if not _combine_credit_is_exact(task, group):
            raise ValueError("Invalid Combine credit descriptor or conflicting task policy")
        groups[group].append((task_id, tile, tiles, task.dependent_event, task.trigger_event))
    expected = {peer * local_experts + slot for peer in range(ep_size) for slot in range(home_experts)}
    _validate_combine_groups(groups, expected)
    for group in groups.values():
        for task_id, *_ in group:
            config.all_tasks[task_id].extra_value_0 = EMPTY_DISPATCH_MARKER


def require_empty_dispatch_native() -> None:
    """Reject a payload that cannot interpret the empty-dispatch marker."""
    # The native module imports torch_npu; keep CPU graph construction independent.
    ops = importlib.import_module("hyper_multicore.torch.ops")
    ops._load_native()  # pylint: disable=protected-access
    try:
        if torch.ops.hyper_multicore.mega_moe_empty_dispatch_version() != 1:
            raise RuntimeError("Unsupported empty-dispatch signal protocol")
    except (AttributeError, RuntimeError) as error:
        raise NativeComponentUnavailableError("Empty-dispatch signals require a rebuilt multicore payload") from error
