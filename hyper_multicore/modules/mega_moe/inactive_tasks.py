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
from dataclasses import dataclass, replace
from math import lcm
import os
import struct
from typing import Any
import weakref

import torch

from hyper_multicore.replica._metadata import _copy_device_bytes
from hyper_multicore.scheduler.config import RuntimeConfigC, TaskType


UNMAPPED_GUEST_SUFFIX_SCHEDULE = "fixed_queue_global_guest_communication_suffix_v1"
RANK_LOCAL_GUEST_SUFFIX_SCHEDULE = "fixed_queue_rank_local_guest_communication_suffix_v1"


def replica_runtime_schedule() -> str:
    """Return the schedule selected before building an immutable runtime."""
    if os.environ.get("HYPER_MULTICORE_REPLICA_RANK_LOCAL_COMM") == "1":
        return RANK_LOCAL_GUEST_SUFFIX_SCHEDULE
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
    combine_event: int = -1
    dropped_combine: int = 0
    incoming_combine: int = 0


def _whole_round_suffixes(
    queue: list[int], experts: dict[int, int], active: int, workers: int
) -> tuple[list[int], list[int]]:
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


def _communication_task(config, task_id, local, ep, mode, dispatch):
    """Validate one directional peer-expert tile without changing its physical identity."""
    task = config.all_tasks[task_id]
    if task.task_type not in (TaskType.TASK_SHMEM_PUT_MEM_SIGNAL, TaskType.TASK_SHMEM_GET_MEM):
        return None
    if task.inputs[3].input_position not in (4, 15, 16) or task.task_split_num % (local * ep):
        return False
    if dispatch is not None and (task.inputs[3].input_position == 4) != dispatch:
        return None
    expected = (
        TaskType.TASK_SHMEM_GET_MEM
        if task.inputs[3].input_position == 4 and mode == "pull"
        else TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
    )
    if task.task_type != expected:
        return False
    tiles = int(task.task_split_num) // (local * ep)
    if tiles < 1:
        return False
    expert = int(task.task_index) // tiles % local
    stage = (int(task.inputs[3].input_position), int(task.task_index) // tiles)
    return (expert, stage)


def _communication_prefix(
    config: RuntimeConfigC,
    queue: list[int],
    *,
    local: int,
    ep: int,
    active: int,
    workers: int,
    mode: str,
    dispatch: bool | None = None,
) -> tuple[list[int], list[int], Counter]:
    """Drop whole empty rounds and count their destination-side signals."""
    experts, stages = ({}, {})
    for task_id in queue:
        parsed = _communication_task(config, task_id, local, ep, mode, dispatch)
        if parsed is False:
            return (queue, [], Counter())
        if parsed is None:
            continue
        experts[task_id], stages[task_id] = parsed
    kept, removed = _whole_round_suffixes(queue, experts, active, lcm(workers, ep))
    groups = Counter((stages[task_id] for task_id in removed))
    for stage, group in groups:
        if any((groups[stage, group % local + rank * local] != groups[stage, group] for rank in range(ep))):
            return (queue, [], Counter())
    reductions = Counter()
    for task_id in removed:
        task = config.all_tasks[task_id]
        is_dispatch = stages[task_id][0] == 4
        reductions[int(task.trigger_event)] += ep if is_dispatch and mode == "push" else 1
    return (kept, removed, reductions)


def _prefix_compute(config, queues, local):
    """Collect and validate one independent prefix input."""
    experts = {}
    trigger_events = {}
    for task_id in (*queues[0], *queues[1]):
        task = config.all_tasks[task_id]
        if task.task_type not in (TaskType.TASK_GROUPED_MATMUL, TaskType.TASK_SWI_GLU, TaskType.TASK_SWI_GLU_GRAD):
            continue
        if task.task_split_num < local or task.task_split_num % local:
            return None
        experts[task_id] = int(task.task_index) // (int(task.task_split_num) // local)
        trigger_events[task_id] = (int(task.trigger_event),)
        if task.task_type == TaskType.TASK_GROUPED_MATMUL and task.extra_value_1:
            trigger_events[task_id] += (int(task.extra_value_1),)
    return (experts, trigger_events)


def _prefix_communication(
    config, queues, local, home_experts, ep_size, dispatch_mode, workers, rank_local_communication
):
    """Collect and validate one independent prefix input."""
    communication = {local: (queues[1], [], Counter())}
    dispatches, combines = ({}, {})
    combine_event = -1
    if ep_size and rank_local_communication:
        events = {
            int(config.all_tasks[task_id].trigger_event)
            for task_id in queues[1]
            if config.all_tasks[task_id].task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
            and config.all_tasks[task_id].inputs[3].input_position in (15, 16)
        }
        if len(events) != 1:
            return None
        combine_event = events.pop()
        for prefixes, dispatch in ((dispatches, True), (combines, False)):
            prefixes[local] = communication[local]
            prefixes.update(
                (
                    (
                        active,
                        _communication_prefix(
                            config,
                            queues[1],
                            local=local,
                            ep=ep_size,
                            active=active,
                            workers=workers,
                            mode=dispatch_mode,
                            dispatch=dispatch,
                        ),
                    )
                    for active in range(home_experts, local)
                )
            )
    elif ep_size:
        communication.update(
            (
                (
                    active,
                    _communication_prefix(
                        config, queues[1], local=local, ep=ep_size, active=active, workers=workers, mode=dispatch_mode
                    ),
                )
                for active in range(home_experts, local)
            )
        )
    return (communication, dispatches, combines, combine_event)


@dataclass
class _PrefixCompiler:
    """Own compilation context without capturing mutable execution state."""

    config: RuntimeConfigC
    home_experts: int
    local: int
    ep_size: int
    dispatch_mode: str
    workers: int
    queues: list
    experts: dict
    trigger_events: dict
    communication: dict
    dispatches: dict
    combines: dict
    combine_event: int
    layout: type

    def compile(self, active: int, common: int, *, use_local: bool = False) -> GuestTaskPrefix | None:
        """Compile a bounded immutable image for one mapped prefix."""
        removed_combine = 0
        if use_local:
            dispatch = self.dispatches[common if self.dispatch_mode == "push" else active]
            combine = self.combines[active]
            removed_communication = set(dispatch[1]) | set(combine[1])
            vector = [task_id for task_id in self.queues[1] if task_id not in removed_communication]
            remote_reductions = dispatch[2] + combine[2]
            removed_combine = len(combine[1])
        else:
            vector, removed_communication, remote_reductions = self.communication[common]
        selected = [
            _whole_round_suffixes(queue, self.experts, active, self.workers) for queue in (self.queues[0], vector)
        ]
        reductions = remote_reductions + Counter(
            (event for _, removed in selected for task in removed for event in self.trigger_events[task])
        )
        triggers = list(self.config.all_event_num_triggers)
        for event, count in reductions.items():
            if not 0 <= event < len(triggers) or triggers[event] < count:
                return None
            triggers[event] -= count
        counts = list(self.config.task_index_num)
        counts[:2] = [len(kept) for kept, _ in selected]
        queue_bytes = []
        for original, (kept, _) in zip(self.queues, selected):
            queue_bytes.append(_encode([*kept, *original[len(kept) :]]))
        patches = (
            (self.layout.all_event_num_triggers.offset, _encode(triggers)),
            (self.layout.task_index_num.offset, _encode(counts)),
            (self.layout.cube_task_indices.offset, queue_bytes[0]),
            (self.layout.vector_task_indices.offset, queue_bytes[1]),
        )
        return GuestTaskPrefix(
            self.home_experts,
            active,
            self.local,
            patches,
            len(selected[0][1]),
            len(selected[1][1]) + len(removed_communication),
            common,
            self.ep_size,
            combine_event=self.combine_event,
            dropped_combine=removed_combine,
            incoming_combine=removed_combine,
        )


def _compile_all_prefixes(compiler, home_experts, local, ep_size, rank_local_communication):
    """Keep one independent preparation stage outside the execution orchestration."""
    prefixes = []
    for active in range(home_experts, local + 1):
        prefix = compiler.compile(active, local)
        if ep_size and rank_local_communication:
            variants = tuple((compiler.compile(active, common, use_local=True) for common in range(active, local + 1)))
        else:
            variants = tuple((compiler.compile(active, common) for common in range(active, local))) if ep_size else ()
        if prefix is None or any((variant is None for variant in variants)):
            return ()
        prefixes.append(replace(prefix, communication_variants=variants))
    return tuple(prefixes)


def build_guest_task_prefixes(
    config: RuntimeConfigC,
    *,
    home_experts: int,
    ep_size: int = 0,
    dispatch_mode: str = "push",
    rank_local_communication: bool = False,
) -> tuple[GuestTaskPrefix, ...]:
    """Precompile bounded CPU patches for every possible mapped guest prefix.

    Local computation uses the rank's mapped prefix. Communication can additionally
    use a common EP-wide prefix; receiver thresholds omit exactly the removed
    remote signals. Partial rounds retain physical worker ownership.

    Args:
        config: Final physical runtime, including any independent producer fences.
        home_experts: Number of owned expert slots preceding the guest pool.
        ep_size: Enable symmetric communication variants for this EP size, or zero.
        dispatch_mode: Native dispatch protocol used by this physical runtime.
        rank_local_communication: Use the local prefix for Pull dispatch and combine.

    Returns:
        One immutable patch per active prefix, including the complete restoration
        image. Unknown queue geometries retain ordinary execution with no patches.
    """
    local = int(config.dynamic_data.dynamic_group_size)
    if type(home_experts) not in (int,) or not 0 < home_experts < local:
        raise ValueError("home_experts must be positive and smaller than the physical local expert count")
    if type(ep_size) not in (int,) or ep_size < 0 or dispatch_mode not in ("push", "pull"):
        raise ValueError("Communication prefixes require a nonnegative EP size and push/pull dispatch")
    workers = int(config.num_workers) // 2
    if workers < 1 or config.num_workers % 2 or config.task_index_num[2]:
        return ()
    queues = [
        list(config.cube_task_indices[: config.task_index_num[0]]),
        list(config.vector_task_indices[: config.task_index_num[1]]),
    ]
    prepared = _prefix_compute(config, queues, local)
    if prepared is None:
        return ()
    experts, trigger_events = prepared
    prepared = _prefix_communication(
        config, queues, local, home_experts, ep_size, dispatch_mode, workers, rank_local_communication
    )
    if prepared is None:
        return ()
    communication, dispatches, combines, combine_event = prepared
    layout = type(config)
    compiler = _PrefixCompiler(
        config,
        home_experts,
        local,
        ep_size,
        dispatch_mode,
        workers,
        queues,
        experts,
        trigger_events,
        communication,
        dispatches,
        combines,
        combine_event,
        layout,
    )
    return _compile_all_prefixes(compiler, home_experts, local, ep_size, rank_local_communication)


def _mapping_prefix(mapping, full, *, global_mapping=False):
    """Validate exact physical slots before calculating the highest mapped suffix."""
    valid = len(mapping) == full.local_experts and all(type(value) in (int,) and value >= -1 for value in mapping)
    if not valid:
        if global_mapping:
            raise ValueError("Global replica mapping does not match the physical EP geometry")
        raise ValueError("Replica physical mapping must match the physical expert count and contain IDs or -1")
    return max(
        (slot + 1 for slot in range(full.home_experts, full.local_experts) if mapping[slot] >= 0),
        default=full.home_experts,
    )


def _incoming_combine_prefix(prefixes, selected, mapped, full):
    """Replace omitted local sends with exactly the receiver's missing peer contributions."""
    removed = sum(prefixes[count - full.home_experts].communication_variants[-1].dropped_combine for count in mapped)
    if removed % selected.ep_size:
        raise ValueError("Combine suffixes do not have symmetric per-peer signal counts")
    incoming = removed // selected.ep_size
    if incoming == selected.incoming_combine:
        return selected
    offset, original = selected.patches[0]
    triggers = bytearray(original)
    index = selected.combine_event * struct.calcsize("<i")
    needed = struct.unpack_from("<i", triggers, index)[0] + selected.dropped_combine - incoming
    if needed < 0:
        raise ValueError("Combine suffixes exceed the receiver completion threshold")
    struct.pack_into("<i", triggers, index, needed)
    return replace(selected, patches=((offset, bytes(triggers)), *selected.patches[1:]), incoming_combine=incoming)


def select_guest_task_prefix(
    prefixes: tuple[GuestTaskPrefix, ...],
    slot_to_logical: Sequence[int] | None,
    *,
    all_slot_to_logical: Sequence[Sequence[int]] | None = None,
) -> GuestTaskPrefix | None:
    """Select only an immutable unmapped suffix and exact EP-wide communication credits.

    Args:
        prefixes: Prepared physical-runtime patches.
        slot_to_logical: Current invocation's local physical mapping.
        all_slot_to_logical: The exact EP-wide immutable mapping, when available.

    Returns:
        A safe prefix, a full restoration image, or None for an unknown geometry.
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
    mapped = tuple(_mapping_prefix(row, full, global_mapping=True) for row in all_slot_to_logical)
    common = max(mapped)
    if common < active:
        raise ValueError("Global replica mapping omits a locally mapped guest slot")
    if selected.combine_event < 0:
        return selected if common == full.local_experts else selected.communication_variants[common - active]
    if not any(tuple(row) == tuple(slot_to_logical) for row in all_slot_to_logical):
        raise ValueError("Global replica mapping omits the local physical mapping")
    selected = selected.communication_variants[common - active]
    return _incoming_combine_prefix(prefixes, selected, mapped, full)


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
    identity = (prefix.active_experts, prefix.communication_experts, prefix.incoming_combine, prefix.dropped_vector)
    if previous is not None and previous[0]() is runtime and previous[1] == identity:
        return runtime
    for offset, payload in prefix.patches:
        if offset < 0 or offset + len(payload) > base.numel():
            raise ValueError("Guest task patch exceeds the physical runtime body")
    # A failed submission can leave a partially rewritten body; never advertise its old prefix.
    if cache is not None:
        cache.pop(id(base), None)
    for offset, payload in prefix.patches:
        _copy_device_bytes(payload, runtime[offset : offset + len(payload)])
    if cache is not None:
        cache[id(base)] = (weakref.ref(runtime), identity)
    return runtime
