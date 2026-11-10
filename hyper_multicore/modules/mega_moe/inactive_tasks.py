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

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import lcm
import struct
from typing import Any
import weakref

import torch

from hyper_multicore.replica._metadata import _copy_device_bytes
from hyper_multicore.scheduler.config import RuntimeConfigC, TaskType


UNMAPPED_GUEST_SUFFIX_SCHEDULE = 'fixed_queue_global_guest_communication_suffix_v1'


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


def _whole_round_suffixes(queue: list[int], experts: dict[int, int], active: int,
                          workers: int) -> tuple[list[int], list[int]]:
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
        removed.extend(queue[position:position + count])
        kept.extend(queue[position + count:end])
        position = end
    return kept, removed


def _encode(values: Sequence[int]) -> bytes:
    """Encode native little-endian int32 arrays without changing their storage size."""
    return struct.pack(f'<{len(values)}i', *values)


def _communication_prefix(config: RuntimeConfigC, queue: list[int], *, local: int, ep: int,
                          active: int, workers: int, mode: str) -> tuple[list[int], list[int], Counter]:
    """Drop globally identical empty tiles and count their destination-side signals."""
    experts, stages = {}, {}
    for task_id in queue:
        task = config.all_tasks[task_id]
        if task.task_type not in (TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskType.TASK_SHMEM_GET_MEM):
            continue
        if task.inputs[3].input_position not in (4, 15, 16) or task.task_split_num % (local * ep):
            return queue, [], Counter()
        expected = (TaskType.TASK_SHMEM_GET_MEM if task.inputs[3].input_position == 4 and mode == 'pull'
                    else TaskType.TASK_SHMEM_PUT_MEM_SIGNAL)
        if task.task_type != expected:
            return queue, [], Counter()
        tiles = int(task.task_split_num) // (local * ep)
        if tiles < 1:
            return queue, [], Counter()
        experts[task_id] = (int(task.task_index) // tiles) % local
        stages[task_id] = (int(task.inputs[3].input_position), int(task.task_index) // tiles)
    # RATR rotates peer order by sender rank. Whole peer rounds preserve equal
    # incoming signal counts, and whole worker rounds preserve core ownership.
    kept, removed = _whole_round_suffixes(queue, experts, active, lcm(workers, ep))
    groups = Counter(stages[task_id] for task_id in removed)
    # Every sender must omit the same tiles for each destination. Keep ordinary
    # communication if a future graph does not have that symmetric geometry.
    for stage, group in groups:
        if any(groups[stage, group % local + rank * local] != groups[stage, group] for rank in range(ep)):
            return queue, [], Counter()
    reductions = Counter()
    for task_id in removed:
        task = config.all_tasks[task_id]
        dispatch = stages[task_id][0] == 4
        reductions[int(task.trigger_event)] += ep if dispatch and mode == 'push' else 1
    return kept, removed, reductions


def build_guest_task_prefixes(config: RuntimeConfigC, *, home_experts: int, ep_size: int = 0,
                              dispatch_mode: str = 'push') -> tuple[GuestTaskPrefix, ...]:
    """Precompile bounded CPU patches for every possible mapped guest prefix.

    Local computation uses the rank's mapped prefix. Communication can additionally
    use a common EP-wide prefix; receiver thresholds omit exactly the removed
    remote signals. Partial rounds retain physical worker ownership.

    Args:
        config: Final physical runtime, including any independent producer fences.
        home_experts: Number of owned expert slots preceding the guest pool.
        ep_size: Enable symmetric communication variants for this EP size, or zero.
        dispatch_mode: Native dispatch protocol used by this physical runtime.

    Returns:
        One immutable patch per active prefix, including the complete restoration
        image. Unknown queue geometries retain ordinary execution with no patches.
    """
    local = int(config.dynamic_data.dynamic_group_size)
    if type(home_experts) is not int or not 0 < home_experts < local:
        raise ValueError('home_experts must be positive and smaller than the physical local expert count')
    if type(ep_size) is not int or ep_size < 0 or dispatch_mode not in ('push', 'pull'):
        raise ValueError('Communication prefixes require a nonnegative EP size and push/pull dispatch')
    workers = int(config.num_workers) // 2
    if workers < 1 or config.num_workers % 2 or config.task_index_num[2]:
        return ()
    queues = [list(config.cube_task_indices[:config.task_index_num[0]]),
              list(config.vector_task_indices[:config.task_index_num[1]])]
    experts = {}
    trigger_events = {}
    for task_id in (*queues[0], *queues[1]):
        task = config.all_tasks[task_id]
        if task.task_type not in (TaskType.TASK_GROUPED_MATMUL, TaskType.TASK_SWI_GLU, TaskType.TASK_SWI_GLU_GRAD):
            continue
        if task.inputs[0].tensor_type == 2:
            continue
        if task.task_split_num < local or task.task_split_num % local:
            return ()
        experts[task_id] = int(task.task_index) // (int(task.task_split_num) // local)
        trigger_events[task_id] = (int(task.trigger_event),)
        if task.task_type == TaskType.TASK_GROUPED_MATMUL and task.extra_value_1:
            trigger_events[task_id] += (int(task.extra_value_1),)
    communication = {local: (queues[1], [], Counter())}
    if ep_size:
        communication.update((active, _communication_prefix(
            config, queues[1], local=local, ep=ep_size, active=active, workers=workers, mode=dispatch_mode,
        )) for active in range(home_experts, local))
    layout = type(config)

    def compile_prefix(active: int, common: int) -> GuestTaskPrefix | None:
        vector, removed_communication, remote_reductions = communication[common]
        selected = [_whole_round_suffixes(queue, experts, active, workers) for queue in (queues[0], vector)]
        reductions = remote_reductions + Counter(
            event for _, removed in selected for task in removed for event in trigger_events[task])
        triggers = list(config.all_event_num_triggers)
        for event, count in reductions.items():
            if not 0 <= event < len(triggers) or triggers[event] < count:
                return None
            triggers[event] -= count
        counts = list(config.task_index_num)
        counts[:2] = [len(kept) for kept, _ in selected]
        queue_bytes = []
        for original, (kept, _) in zip(queues, selected):
            queue_bytes.append(_encode([*kept, *original[len(kept):]]))
        patches = (
            (layout.all_event_num_triggers.offset, _encode(triggers)),
            (layout.task_index_num.offset, _encode(counts)),
            (layout.cube_task_indices.offset, queue_bytes[0]),
            (layout.vector_task_indices.offset, queue_bytes[1]),
        )
        return GuestTaskPrefix(home_experts, active, local, patches, len(selected[0][1]),
                               len(selected[1][1]) + len(removed_communication), common, ep_size)

    prefixes = []
    for active in range(home_experts, local + 1):
        prefix = compile_prefix(active, local)
        variants = tuple(compile_prefix(active, common) for common in range(active, local)) if ep_size else ()
        if prefix is None or any(variant is None for variant in variants):
            return ()
        prefixes.append(GuestTaskPrefix(prefix.home_experts, prefix.active_experts, prefix.local_experts,
                                        prefix.patches, prefix.dropped_cube, prefix.dropped_vector,
                                        prefix.communication_experts, ep_size, variants))
    return tuple(prefixes)


def select_guest_task_prefix(prefixes: tuple[GuestTaskPrefix, ...],
                             slot_to_logical: Sequence[int] | None, *,
                             all_slot_to_logical: Sequence[Sequence[int]] | None = None) -> GuestTaskPrefix | None:
    """Retain holes and mapped zero-row guests; remove only a proven unmapped suffix.

    Args:
        prefixes: Physical runtime patches compiled before steady-state execution.
        slot_to_logical: The saved invocation's local physical mapping, or no pruning.
        all_slot_to_logical: The same immutable EP-wide plan on every participant.

    Returns:
        The smallest prefix containing every mapped slot. The full prefix restores
        a reused runtime when no immutable route mapping is available.
    """
    if not prefixes:
        return None
    full = prefixes[-1]
    if slot_to_logical is None:
        return full
    if len(slot_to_logical) != full.local_experts:
        raise ValueError('Replica mapping does not match the physical runtime expert count')
    if any(type(value) is not int or value < -1 for value in slot_to_logical):
        raise ValueError('Replica physical mapping must contain logical IDs or -1')
    active = full.home_experts
    for slot in range(full.home_experts, full.local_experts):
        if slot_to_logical[slot] >= 0:
            active = slot + 1
    selected = prefixes[active - full.home_experts]
    if all_slot_to_logical is None or not selected.ep_size:
        return selected
    if len(all_slot_to_logical) != selected.ep_size or any(
            len(row) != full.local_experts or any(type(value) is not int or value < -1 for value in row)
            for row in all_slot_to_logical):
        raise ValueError('Global replica mapping does not match the physical EP geometry')
    common = max((slot + 1 for row in all_slot_to_logical
                  for slot in range(full.home_experts, full.local_experts) if row[slot] >= 0),
                 default=full.home_experts)
    if common < active:
        raise ValueError('Global replica mapping omits a locally mapped guest slot')
    return selected if common == full.local_experts else selected.communication_variants[common - active]


def apply_guest_task_prefix(runtime: torch.Tensor, prefix: GuestTaskPrefix | None, *,
                            base: torch.Tensor, cache: dict[int, tuple[Any, Any]] | None = None) -> torch.Tensor:
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
        raise ValueError('Guest task patches require an exclusive runtime image')
    previous = None if cache is None else cache.get(id(base))
    identity = (prefix.active_experts, prefix.communication_experts)
    if previous is not None and previous[0]() is runtime and previous[1] == identity:
        return runtime
    for offset, payload in prefix.patches:
        if offset < 0 or offset + len(payload) > base.numel():
            raise ValueError('Guest task patch exceeds the physical runtime body')
    # A failed submission can leave a partially rewritten body; never advertise its old prefix.
    if cache is not None:
        cache.pop(id(base), None)
    for offset, payload in prefix.patches:
        _copy_device_bytes(payload, runtime[offset:offset + len(payload)])
    if cache is not None:
        cache[id(base)] = (weakref.ref(runtime), identity)
    return runtime
