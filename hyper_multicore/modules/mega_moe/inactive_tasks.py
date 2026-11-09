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
"""Runtime patches omitting unmapped guest work with matched remote signals."""

from __future__ import annotations

import struct
import weakref
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import lcm
from typing import Any

import torch

from hyper_multicore.replica._metadata import _copy_device_bytes
from hyper_multicore.replica.capacity import _integer
from hyper_multicore.scheduler.config import RuntimeConfigC, TaskType

UNMAPPED_GUEST_SUFFIX_SCHEDULE = "fixed_queue_global_guest_communication_suffix_v1"


def replica_runtime_schedule() -> str:
    """Identify the fixed, ordered replica execution schedule."""
    return UNMAPPED_GUEST_SUFFIX_SCHEDULE


@dataclass(frozen=True)
class GuestTaskPrefix:
    """Queue and trigger patches retaining all physical descriptors and strides."""

    home_experts: int
    active_experts: int
    local_experts: int
    patches: tuple[tuple[int, bytes], ...]
    dropped_cube: int
    dropped_vector: int
    communication_experts: int = 0
    ep_size: int = 0
    communication_variants: tuple[GuestTaskPrefix, ...] = ()


def _whole_round_suffixes(queue: list[int], experts: dict[int, int], active: int, workers: int):
    """Keep worker ownership by deleting only whole rounds from inactive runs."""
    kept, removed = [], []
    position = 0
    while position < len(queue):
        end = position
        while end < len(queue) and experts.get(queue[end], -1) >= active:
            end += 1
        if end == position:
            kept.append(queue[position])
            position += 1
            continue
        count = (end - position) // workers * workers
        removed.extend(queue[position : position + count])
        kept.extend(queue[position + count : end])
        position = end
    return kept, removed


def _encode(values: Sequence[int]) -> bytes:
    """Encode native little-endian int32 arrays without changing their storage size."""
    return struct.pack(f"<{len(values)}i", *values)


def _communication_tasks(config: RuntimeConfigC, queue: list[int], local: int, ep: int, mode: str):
    """Recognize a complete communication geometry before changing its signals."""
    experts, stages = {}, {}
    for task_id in queue:
        task = config.all_tasks[task_id]
        if task.task_type not in (TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskType.TASK_SHMEM_GET_MEM):
            continue
        position = int(task.inputs[3].input_position)
        if position not in (4, 15, 16) or task.task_split_num % (local * ep):
            return None
        expected = (
            TaskType.TASK_SHMEM_GET_MEM if position == 4 and mode == "pull" else TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
        )
        tiles = int(task.task_split_num) // (local * ep)
        if task.task_type != expected or tiles < 1:
            return None
        experts[task_id] = int(task.task_index) // tiles % local
        stages[task_id] = position, int(task.task_index) // tiles
    return experts, stages


def _communication_prefix(config, queue, *, local, ep, active, workers, mode):
    """Drop symmetric whole empty rounds and count their destination-side signals."""
    geometry = _communication_tasks(config, queue, local, ep, mode)
    if geometry is None:
        return queue, [], Counter()
    experts, stages = geometry
    kept, removed = _whole_round_suffixes(queue, experts, active, lcm(workers, ep))
    groups = Counter(stages[task_id] for task_id in removed)
    for stage, group in groups:
        if any(groups[stage, group % local + rank * local] != groups[stage, group] for rank in range(ep)):
            return queue, [], Counter()
    reductions = Counter()
    for task_id in removed:
        is_dispatch = stages[task_id][0] == 4
        reductions[int(config.all_tasks[task_id].trigger_event)] += ep if is_dispatch and mode == "push" else 1
    return kept, removed, reductions


def _compute_tasks(config, queues, local):
    """Capture both ordinary graph events and optional complete-expert producer fences."""
    experts, events = {}, {}
    for task_id in (*queues[0], *queues[1]):
        task = config.all_tasks[task_id]
        if task.task_type not in (TaskType.TASK_GROUPED_MATMUL, TaskType.TASK_SWI_GLU, TaskType.TASK_SWI_GLU_GRAD):
            continue
        if task.task_split_num < local or task.task_split_num % local:
            return None
        experts[task_id] = int(task.task_index) // (int(task.task_split_num) // local)
        events[task_id] = (int(task.trigger_event),)
        if task.task_type == TaskType.TASK_GROUPED_MATMUL and task.extra_value_1:
            events[task_id] += (int(task.extra_value_1),)
    return experts, events


def _compile_prefix(config, queues, compute, communication, *, home, local, active, common, ep, workers):
    """Compile one immutable image after proving every removed producer threshold."""
    vector, removed_communication, remote_reductions = communication[common]
    experts, events = compute
    selected = [_whole_round_suffixes(queue, experts, active, workers) for queue in (queues[0], vector)]
    reductions = remote_reductions + Counter(
        event for _, removed in selected for task in removed for event in events[task]
    )
    triggers = list(config.all_event_num_triggers)
    for event, count in reductions.items():
        if not 0 <= event < len(triggers) or triggers[event] < count:
            return None
        triggers[event] -= count
    counts = list(config.task_index_num)
    counts[:2] = [len(kept) for kept, _ in selected]
    encoded = [_encode([*kept, *original[len(kept) :]]) for original, (kept, _) in zip(queues, selected)]
    layout = type(config)
    patches = (
        (layout.all_event_num_triggers.offset, _encode(triggers)),
        (layout.task_index_num.offset, _encode(counts)),
        (layout.cube_task_indices.offset, encoded[0]),
        (layout.vector_task_indices.offset, encoded[1]),
    )
    return GuestTaskPrefix(
        home, active, local, patches, len(selected[0][1]), len(selected[1][1]) + len(removed_communication), common, ep
    )


def build_guest_task_prefixes(
    config: RuntimeConfigC, *, home_experts: int, ep_size: int = 0, dispatch_mode: str = "push"
) -> tuple[GuestTaskPrefix, ...]:
    """Precompile queue prefixes with unchanged worker ownership and matched signals.

    Args:
        config: Final physical runtime, including complete-expert producer fences.
        home_experts: Number of owned expert slots preceding the guest pool.
        ep_size: EP size for symmetric communication variants, or zero to retain communication.
        dispatch_mode: Native push or pull protocol.

    Returns:
        Immutable patches for each active prefix. Unknown schedules return no patches.
    """
    local = int(config.dynamic_data.dynamic_group_size)
    _integer(home_experts, "home_experts")
    _integer(ep_size, "ep_size", 0)
    if home_experts >= local or dispatch_mode not in ("push", "pull"):
        raise ValueError("Guest prefixes require a home prefix and push/pull dispatch")
    workers = int(config.num_workers) // 2
    if workers < 1 or config.num_workers % 2 or config.task_index_num[2]:
        return ()
    queues = [
        list(config.cube_task_indices[: config.task_index_num[0]]),
        list(config.vector_task_indices[: config.task_index_num[1]]),
    ]
    compute = _compute_tasks(config, queues, local)
    if compute is None:
        return ()
    communication = {local: (queues[1], [], Counter())}
    if ep_size:
        communication.update(
            (
                active,
                _communication_prefix(
                    config, queues[1], local=local, ep=ep_size, active=active, workers=workers, mode=dispatch_mode
                ),
            )
            for active in range(home_experts, local)
        )
    prefixes = []
    for active in range(home_experts, local + 1):
        variants = [
            _compile_prefix(
                config,
                queues,
                compute,
                communication,
                home=home_experts,
                local=local,
                active=active,
                common=common,
                ep=ep_size,
                workers=workers,
            )
            for common in range(active if ep_size else local, local + 1)
        ]
        if any(variant is None for variant in variants):
            return ()
        prefixes.append(replace(variants[-1], communication_variants=tuple(variants[:-1])))
    return tuple(prefixes)


def _mapping_prefix(row: Sequence[int], full: GuestTaskPrefix) -> int:
    """Validate one physical mapping and retain holes plus mapped zero-row guests."""
    if len(row) != full.local_experts:
        raise ValueError("Replica mapping does not match the physical runtime expert count")
    for value in row:
        _integer(value, "Replica logical ID", -1)
    return max(
        (slot + 1 for slot in range(full.home_experts, full.local_experts) if row[slot] >= 0), default=full.home_experts
    )


def select_guest_task_prefix(
    prefixes: tuple[GuestTaskPrefix, ...],
    slot_to_logical: Sequence[int] | None,
    *,
    all_slot_to_logical: Sequence[Sequence[int]] | None = None,
) -> GuestTaskPrefix | None:
    """Select only a proven unmapped suffix, using a group-wide communication prefix.

    Args:
        prefixes: Precompiled physical runtime images.
        slot_to_logical: Saved local physical mapping, or None to restore the full image.
        all_slot_to_logical: Immutable group plan used for matched remote signal counts.

    Returns:
        The matching local computation and common communication patch.
    """
    if not prefixes:
        return None
    full = prefixes[-1]
    if slot_to_logical is None:
        return full
    active = _mapping_prefix(slot_to_logical, full)
    selected = prefixes[active - full.home_experts]
    if all_slot_to_logical is None or not selected.ep_size:
        return selected
    if len(all_slot_to_logical) != selected.ep_size:
        raise ValueError("Global replica mapping does not match the physical EP geometry")
    common = max(_mapping_prefix(row, full) for row in all_slot_to_logical)
    if common < active:
        raise ValueError("Global replica mapping omits a locally mapped guest slot")
    return selected if common == full.local_experts else selected.communication_variants[common - active]


def apply_guest_task_prefix(
    runtime: torch.Tensor,
    prefix: GuestTaskPrefix | None,
    *,
    base: torch.Tensor,
    cache: dict[int, tuple[Any, Any]] | None = None,
) -> torch.Tensor:
    """Update only an exclusively leased runtime image, never the saved plan body.

    Args:
        runtime: Writable split runtime ordered by the existing workspace lease.
        prefix: Saved route's immutable queue variant, including full restoration.
        base: Original runtime identity whose body remains immutable.
        cache: Workspace-local weak identity and prefix of the last applied image.

    Returns:
        The same device image, with small current-stream metadata copies only when
        its body variant changes. No extra device runtime allocation is retained.
    """
    if prefix is None:
        return runtime
    if runtime is base:
        raise ValueError("Guest task patches require an exclusive runtime image")
    previous = None if cache is None else cache.get(id(base))
    identity = (prefix.active_experts, prefix.communication_experts, prefix.dropped_vector)
    if previous is not None and previous[0]() is runtime and (previous[1] == identity):
        return runtime
    for offset, payload in prefix.patches:
        if offset < 0 or offset + len(payload) > base.numel():
            raise ValueError("Guest task patch exceeds the physical runtime body")
    if cache is not None:
        cache.pop(id(base), None)
    for offset, payload in prefix.patches:
        _copy_device_bytes(payload, runtime[offset : offset + len(payload)])
    if cache is not None:
        cache[id(base)] = (weakref.ref(runtime), identity)
    return runtime
