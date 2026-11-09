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
"""Check the physical backward schedule before sharing transient storage."""

from collections import defaultdict
from typing import Optional

from hyper_multicore.scheduler.config import (
    ATOMIC_ADD_VALUE_LEN,
    EVENT_INVALID_ID,
    RuntimeConfigC,
    TaskDescC,
    TaskType,
)


_DISPATCH = 0
_WEIGHT2_GRAD = 6
_ACTIVATION_GRAD = 8
_SWIGLU_GRAD = 10
_INPUT_GRAD = 12
_WEIGHT13_GRAD = 18


def _worker_read_order(
    tasks: list[TaskDescC],
    num_experts: int,
    num_cores: int,
    *,
    act_first: bool = False,
) -> bool:
    """Require every Cube to finish both receive readers before writing dX."""
    readers = (_ACTIVATION_GRAD, _WEIGHT2_GRAD) if act_first else (_WEIGHT2_GRAD, _ACTIVATION_GRAD)
    for worker in range(num_cores):
        locations = {}
        for index, task in enumerate(tasks[worker::num_cores]):
            output = task.outputs[0].input_position
            if output not in (_WEIGHT2_GRAD, _ACTIVATION_GRAD, _INPUT_GRAD):
                continue
            key = (output, task.task_index // num_cores)
            if key in locations or task.task_type != TaskType.TASK_GROUPED_MATMUL:
                return False
            locations[key] = index
        if len(locations) != 3 * num_experts:
            return False
        for expert in range(num_experts):
            order = [locations.get((output, expert), -1) for output in (*readers, _INPUT_GRAD)]
            if not 0 <= order[0] < order[1] < order[2]:
                return False
    return True


def _expert_events_join_readers(
    cfg: RuntimeConfigC,
    cube_tasks: list[TaskDescC],
    vector_tasks: list[TaskDescC],
    num_experts: int,
    num_cores: int,
    *,
    act_first: bool = False,
) -> bool:
    """Require the existing event chain to join all readers of each expert."""
    acts = defaultdict(list)
    gates = defaultdict(list)
    swiglu = defaultdict(list)
    late_reads = defaultdict(list)
    for task in cube_tasks:
        expert = task.task_index // num_cores
        if task.outputs[0].input_position == _ACTIVATION_GRAD:
            acts[expert].append(task)
        elif task.outputs[0].input_position == _INPUT_GRAD:
            gates[expert].append(task)
        elif act_first and task.outputs[0].input_position == _WEIGHT2_GRAD:
            late_reads[expert].append(task)
    for task in vector_tasks:
        if task.task_type != TaskType.TASK_SWI_GLU_GRAD:
            continue
        if task.task_split_num < num_experts or task.task_split_num % num_experts:
            return False
        expert = task.task_index // (task.task_split_num // num_experts)
        swiglu[expert].append(task)
    producers = defaultdict(list)
    for task in cube_tasks + vector_tasks:
        producers[task.trigger_event].append(task)
        if task.task_type == TaskType.TASK_GROUPED_MATMUL and task.extra_value_1:
            producers[task.extra_value_1].append(task)
    return all(
        _expert_dependency_chain(
            cfg, acts[expert], gates[expert], swiglu[expert], num_cores, producers, late_reads[expert]
        )
        for expert in range(num_experts)
    )


def _event_joins_tasks(cfg: RuntimeConfigC, tasks: list[TaskDescC], producers: dict) -> bool:
    """Require a completion event to include exactly these nonempty tasks."""
    event = tasks[0].trigger_event
    return (
        len(producers[event]) == len(tasks)
        and cfg.all_event_num_triggers[event] == len(tasks)
        and all(task.trigger_event == event for task in tasks)
    )


def _expert_dependency_chain(
    cfg: RuntimeConfigC,
    acts: list[TaskDescC],
    gates: list[TaskDescC],
    swiglu: list[TaskDescC],
    num_cores: int,
    producers: dict,
    late_reads: list[TaskDescC],
) -> bool:
    """Join every expert reader before any task overwrites its receive rows."""
    if len(acts) != num_cores or len(gates) != num_cores or not swiglu:
        return False
    if not _event_joins_tasks(cfg, acts, producers) or not _event_joins_tasks(cfg, swiglu + late_reads, producers):
        return False
    act_event = acts[0].trigger_event
    swiglu_event = swiglu[0].trigger_event
    return all(
        task.dependent_event == act_event and task.outputs[0].input_position == _SWIGLU_GRAD for task in swiglu
    ) and all(task.dependent_event == swiglu_event for task in gates)


def can_reuse_backward_dispatch(
    cfg: RuntimeConfigC,
    num_experts: int,
    num_cores: int,
    *,
    act_first: bool = False,
) -> bool:
    """Check a validated runtime before reusing received dY storage for dX.

    Args:
        cfg: Validated, fully reordered backward runtime descriptors.
        num_experts: Number of local experts.
        num_cores: Number of active Cube workers.
        act_first: Prove the alternative join with W2 as a late receive reader.

    Returns:
        Whether every receive reader precedes dX through worker order and
        the existing full-expert completion events. Unknown readers or changed
        schedules disable reuse without adding dependencies to the hot path.
    """
    if cfg.task_index_num[2]:
        return False
    cube_tasks = [cfg.all_tasks[index] for index in cfg.cube_task_indices[: cfg.task_index_num[0]]]
    vector_tasks = [cfg.all_tasks[index] for index in cfg.vector_task_indices[: cfg.task_index_num[1]]]
    for task in cube_tasks + vector_tasks:
        reads_dispatch = any(task.inputs[index].input_position == _DISPATCH for index in range(task.num_inputs))
        if reads_dispatch and (
            task.task_type != TaskType.TASK_GROUPED_MATMUL
            or task.outputs[0].input_position not in (_WEIGHT2_GRAD, _ACTIVATION_GRAD)
        ):
            return False
    return _worker_read_order(cube_tasks, num_experts, num_cores, act_first=act_first) and _expert_events_join_readers(
        cfg, cube_tasks, vector_tasks, num_experts, num_cores, act_first=act_first
    )


def replica_w2_ready_events(cfg: RuntimeConfigC, num_experts: int, num_cores: int) -> tuple[int, ...]:
    """Use full ActGrad completion only when every Cube first finishes its W2Grad.

    Unknown schedules return no events so the adapter retains late gradient
    return. No additional dependency or counter is added to the Cube queue.
    """
    if not can_reuse_backward_dispatch(cfg, num_experts, num_cores):
        return ()
    events = [0] * num_experts
    for index in cfg.cube_task_indices[: cfg.task_index_num[0]]:
        task = cfg.all_tasks[index]
        if task.outputs[0].input_position == _ACTIVATION_GRAD:
            events[task.task_index // num_cores] = task.trigger_event
    return tuple(events)


def _worker_w13_order(selected, gates, weights, worker, num_experts, num_cores):
    """Validate one worker's complete W13 producer list after its dX prefix."""
    if any(
        (
            gate.outputs[0].input_position != _INPUT_GRAD or gate.task_index != expert * num_cores + worker
            for expert, gate in enumerate(gates)
        )
    ):
        return None
    order = []
    seen = set()
    for weight in selected[num_experts:]:
        expert = weight.task_index // num_cores
        if not 0 <= expert < num_experts or expert in seen:
            return None
        gate = gates[expert]
        if (
            weight.outputs[0].input_position != _WEIGHT13_GRAD
            or weight.task_index != gate.task_index
            or weight.task_type != TaskType.TASK_GROUPED_MATMUL
            or (weight.dependent_event != gate.trigger_event)
            or (weight.num_inputs != 3)
            or (tuple((weight.inputs[i].input_position for i in range(3))) != (17, 10, 19))
        ):
            return None
        order.append(expert)
        seen.add(expert)
        weights[expert].append(weight)
    return order


def _w13_producers(cfg: RuntimeConfigC, num_experts: int, num_cores: int) -> list[list[TaskDescC]]:
    """Recognize complete W13 groups after every worker's dX prefix."""
    if not can_reuse_backward_dispatch(cfg, num_experts, num_cores):
        return []
    tasks = [cfg.all_tasks[index] for index in cfg.cube_task_indices[: cfg.task_index_num[0]]]
    weights = [[] for _ in range(num_experts)]
    producer_order = None
    for worker in range(num_cores):
        selected = [
            task for task in tasks[worker::num_cores] if task.outputs[0].input_position in (_INPUT_GRAD, _WEIGHT13_GRAD)
        ]
        if len(selected) != 2 * num_experts:
            return []
        gates = selected[:num_experts]
        order = _worker_w13_order(selected, gates, weights, worker, num_experts, num_cores)
        if order is None:
            return []
        if producer_order is not None and order != producer_order:
            return []
        producer_order = order
    return weights


def _act_first_completion_is_exact(cfg, cube, vector, w13_groups, final):
    """Prove that only the expected producers reach termination."""
    return not (
        final == EVENT_INVALID_ID
        or any((task.dependent_event == final and task.task_type != TaskType.TASK_TERMINATE for task in cube + vector))
        or sum((task.trigger_event == final for task in cube + vector)) != cfg.all_event_num_triggers[final]
        or any((task.trigger_event != final for group in w13_groups for task in group))
    )


def _act_first_wave_is_exact(gate, w2, act, expert, num_cores, final):
    """Keep each W2 and ActGrad lane on its original expert and receive dependency."""
    return not (
        gate.outputs[0].input_position != _INPUT_GRAD
        or gate.task_index != expert * num_cores
        or any(
            (
                w.outputs[0].input_position != _WEIGHT2_GRAD
                or a.outputs[0].input_position != _ACTIVATION_GRAD
                or w.task_index != expert * num_cores + worker
                or (a.task_index != w.task_index)
                or (w.dependent_event != a.dependent_event)
                or (w.trigger_event != final)
                for worker, (w, a) in enumerate(zip(w2, act))
            )
        )
    )


def prioritize_activation_gradient(cfg: RuntimeConfigC, num_experts: int, num_cores: int) -> bool:
    """Start SwiGLU earlier without additional counters, atomics or receive storage.

    W2 and ActGrad depend on the same dispatch event. GateGrad must join
    W2 as well as SwiGLU before it can reuse the receive buffer. Every W2
    then reaches termination through GateGrad and W13, so its original
    termination increment can move to the existing per-expert GateGrad join.
    """
    w13_groups = _w13_producers(cfg, num_experts, num_cores)
    count = cfg.task_index_num[0]
    if not w13_groups or count != 4 * num_experts * num_cores:
        return False
    ids = list(cfg.cube_task_indices[:count])
    cube = [cfg.all_tasks[index] for index in ids]
    vector = [cfg.all_tasks[index] for index in cfg.vector_task_indices[: cfg.task_index_num[1]]]
    terms = [task for task in vector if task.task_type == TaskType.TASK_TERMINATE]
    if len(terms) != 1 or any((task.extra_value_1 for task in cube)):
        return False
    final = terms[0].dependent_event
    if not _act_first_completion_is_exact(cfg, cube, vector, w13_groups, final):
        return False
    joins = []
    for expert in range(num_experts):
        start = 2 * expert * num_cores
        middle, stop = (start + num_cores, start + 2 * num_cores)
        w2, act = (cube[start:middle], cube[middle:stop])
        gate = cube[(2 * num_experts + expert) * num_cores]
        if not _act_first_wave_is_exact(gate, w2, act, expert, num_cores, final):
            return False
        joins.append((gate.dependent_event, w2))
        ids[start:stop] = ids[middle:stop] + ids[start:middle]
    for event, tasks in joins:
        cfg.all_event_num_triggers[event] += num_cores
        for task in tasks:
            task.trigger_event = event
    cfg.all_event_num_triggers[final] -= num_experts * num_cores
    cfg.cube_task_indices[:count] = ids
    return True


def _guest_w13_indices(
    cfg: RuntimeConfigC, num_experts: int, num_cores: int, replica_slots: int
) -> Optional[list[int]]:
    """Permute complete W13 tail groups without moving a task to another worker."""
    count = cfg.task_index_num[0]
    prefix = count - num_experts * num_cores
    groups = defaultdict(list)
    for offset, index in enumerate(cfg.cube_task_indices[:count]):
        task = cfg.all_tasks[index]
        if (offset >= prefix) != (task.outputs[0].input_position == _WEIGHT13_GRAD):
            return None
        if offset >= prefix:
            groups[task.task_index // num_cores].append(index)
    home = num_experts - replica_slots
    reordered = [index for expert in (*range(home, num_experts), *range(home)) for index in groups[expert]]
    if any(cfg.all_tasks[index].task_index % num_cores != offset % num_cores for offset, index in enumerate(reordered)):
        return None
    return reordered


def _w1_join_producers_are_exact(cube, producers, num_experts, num_cores, join):
    """Require every join contribution to be the final complete GateGrad wave."""
    return not (
        len(producers) != num_cores
        or any(
            (
                task.outputs[0].input_position != _INPUT_GRAD or task.task_index // num_cores != num_experts - 1
                for task in producers
            )
        )
        or any((task.task_type == TaskType.TASK_GROUPED_MATMUL and task.extra_value_1 == join for task in cube))
    )


def _apply_w1_join(weights, join):
    """Retain one complete expert join on each worker before subsequent W13 tasks."""
    for expert, tasks in enumerate(weights):
        for task in tasks:
            task.dependent_event = join if expert == 0 else EVENT_INVALID_ID


def join_w1_phase_once(cfg: RuntimeConfigC, num_experts: int, num_cores: int, *, apply: bool = True) -> bool:
    """Use one full GateGrad completion fence per ordinary W1 Cube queue.

    The last GateGrad event joins the preceding GateGrad work by per-worker
    program order. A worker's subsequent W1 tasks inherit that first wait.
    Unrecognized schedules are left unchanged.
    Pass apply=False to prove the join without changing the fallback queue.
    """
    weights = _w13_producers(cfg, num_experts, num_cores)
    if not weights:
        return False
    cube = [cfg.all_tasks[i] for i in cfg.cube_task_indices[: cfg.task_index_num[0]]]
    ordered = [task for task in cube if task.outputs[0].input_position == _WEIGHT13_GRAD]
    if any((task.task_index != ordinal for ordinal, task in enumerate(ordered))):
        return False
    join = weights[-1][0].dependent_event
    if join == EVENT_INVALID_ID or cfg.all_event_num_triggers[join] != num_cores:
        return False
    vector = [cfg.all_tasks[i] for i in cfg.vector_task_indices[: cfg.task_index_num[1]]]
    producers = [task for task in cube + vector if task.trigger_event == join]
    if not _w1_join_producers_are_exact(cube, producers, num_experts, num_cores, join):
        return False
    if apply:
        _apply_w1_join(weights, join)
    return True


def prepare_w13_overlap_schedule(
    cfg: RuntimeConfigC, num_experts: int, num_cores: int, *, replica_slots: int = 0
) -> bool:
    """Publish W13 completion, optionally producing guest gradients before home gradients.

    The optional GMM producer event joins every Cube, including zero-row
    producers. The ordinary trigger still contributes to graph termination.
    Guest priority permutes only complete W13 groups after the unchanged dX
    prefix; task ownership, dependencies and FP32 reader order are preserved.
    Unknown schedules retain late return without modifying their descriptors.

    Args:
        cfg: Validated physical backward descriptors.
        num_experts: Local home and guest expert count.
        num_cores: Active Cube worker count.
        replica_slots: Guest slots at the end of the local expert namespace.

    Returns:
        Whether every W13 producer and the requested queue order were proven safe.
    """
    if not 0 <= replica_slots < num_experts:
        return False
    weights = _w13_producers(cfg, num_experts, num_cores)
    if not weights or any(
        cfg.all_tasks[index].extra_value_1 for index in cfg.cube_task_indices[: cfg.task_index_num[0]]
    ):
        return False
    highest = max(index for index, count in enumerate(cfg.all_event_num_triggers) if count)
    first = highest + ATOMIC_ADD_VALUE_LEN
    if first + num_experts - 1 + ATOMIC_ADD_VALUE_LEN > cfg.event_capacity:
        return False
    reordered = None
    if replica_slots:
        reordered = _guest_w13_indices(cfg, num_experts, num_cores, replica_slots)
        if reordered is None:
            return False
    for expert, group in enumerate(weights):
        event = first + expert
        cfg.all_event_num_triggers[event] = num_cores
        for task in group:
            task.extra_value_1 = event
    if reordered is not None:
        count = cfg.task_index_num[0]
        cfg.cube_task_indices[count - len(reordered) : count] = reordered
    return True


def replica_w13_ready_events(cfg: RuntimeConfigC, num_experts: int, num_cores: int) -> tuple[int, ...]:
    """Prove each independent event joins exactly the expert's W13 producers."""
    weights = _w13_producers(cfg, num_experts, num_cores)
    if not weights:
        return ()
    producers = defaultdict(list)
    for indices, count in (
        (cfg.cube_task_indices, cfg.task_index_num[0]),
        (cfg.vector_task_indices, cfg.task_index_num[1]),
    ):
        for index in indices[:count]:
            task = cfg.all_tasks[index]
            producers[task.trigger_event].append(task)
            if task.task_type == TaskType.TASK_GROUPED_MATMUL and task.extra_value_1:
                producers[task.extra_value_1].append(task)
    events = []
    for group in weights:
        event = group[0].extra_value_1
        if (
            not event
            or event + ATOMIC_ADD_VALUE_LEN > cfg.event_capacity
            or cfg.all_event_num_triggers[event] != num_cores
            or len(producers[event]) != num_cores
            or any(task.extra_value_1 != event for task in group)
        ):
            return ()
        events.append(event)
    return tuple(events)
