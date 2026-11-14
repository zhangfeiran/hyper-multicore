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
"""Save X beside GMM1, then reuse symmetric dispatch for GMM2 Y."""

import ctypes
from collections import defaultdict
from dataclasses import replace

import torch

from hyper_multicore.tasks.gmm import GroupedMatmulTaskMode
from hyper_multicore.scheduler.config import RuntimeConfigC, TaskDescC, TaskType
from hyper_multicore.scheduler.runtime import allocate_runtime_config
from hyper_multicore.torch import ops as multicore_ops

from .spec import MegaMoeSpec


def resolve_saved_dispatch(spec: MegaMoeSpec, device: torch.device | str) -> MegaMoeSpec:
    """Require aligned Push numerics and the independent buffer-role and bounded-lane protocol v3."""
    compatible = (
        spec._fuse_saved_dispatch and spec.dispatch_mode == "push" and not spec.batch_experts
        and not spec.apply_probs_before_fc2 and spec.swiglu_limit is None
        and not spec.clamp_after_activation and spec.hidden_size % 16 == 0
    )
    enabled = False
    if compatible and torch.device(device).type == "npu":
        multicore_ops._load_native()  # pylint: disable=protected-access
        version = getattr(torch.ops.hyper_multicore, "mega_moe_saved_dispatch_version", None)
        enabled = version is not None and version() == 3
    return replace(spec, _fuse_saved_dispatch=enabled)


def _grow_runtime(config, additional):
    """Preserve descriptor fields while reserving complete worker rounds."""
    result = allocate_runtime_config(config.task_num + 1 + additional, config.event_capacity,
                                     config.dynamic_data.dynamic_group_size, config.num_workers)
    for name, _ in RuntimeConfigC._fields_:
        if name != "task_capacity":
            setattr(result, name, getattr(config, name))
    for name, _ in type(config)._fields_:
        ctypes.memmove(ctypes.addressof(result) + getattr(type(result), name).offset,
                       ctypes.addressof(config) + getattr(type(config), name).offset,
                       min(ctypes.sizeof(getattr(config, name)), ctypes.sizeof(getattr(result, name))))
    return result


def _expert_tasks(config, local, kind, output):
    """Require each standard per-expert descriptor exactly once in its queue."""
    queue = config.cube_task_indices if kind == TaskType.TASK_GROUPED_MATMUL else config.vector_task_indices
    count = config.task_index_num[0 if kind == TaskType.TASK_GROUPED_MATMUL else 1]
    groups = defaultdict(list)
    for task_id in queue[:count]:
        task = config.all_tasks[task_id]
        if task.task_type != kind or task.outputs[0].input_position != output:
            continue
        lanes, remainder = divmod(task.task_split_num, local)
        if remainder or lanes <= 0 or not 0 <= task.task_index < task.task_split_num:
            raise ValueError("Invalid saved-dispatch expert geometry")
        groups[task.task_index // lanes].append(task_id)
    if set(groups) != set(range(local)):
        raise ValueError("Incomplete saved-dispatch expert groups")
    for ids in groups.values():
        task = config.all_tasks[ids[0]]
        lanes = task.task_split_num // local
        if sorted(config.all_tasks[index].task_index % lanes for index in ids) != list(range(lanes)):
            raise ValueError("Incomplete or duplicated saved-dispatch expert lanes")
    return groups


def _uniform_stage(config, ids, ports):
    """Prove a full expert-stage fence, not a partial tile producer."""
    tasks = [config.all_tasks[index] for index in ids]
    signature = (tasks[0].dependent_event, tasks[0].trigger_event)
    for task in tasks:
        if ((task.dependent_event, task.trigger_event) != signature
                or tuple(task.inputs[index].input_position for index in range(task.num_inputs)) != ports):
            raise ValueError("Noncanonical saved-dispatch producer fence or ports")
    if config.all_event_num_triggers[signature[1]] != len(ids):
        raise ValueError("Unproven saved-dispatch producer threshold")
    return signature


def _prove_expert(config, first, activation, second):
    """GMM2 may overwrite X only after every GMM1 reader and save has finished."""
    ready, read_done = _uniform_stage(config, first, (0, 5, 6))
    act_ready, act_done = _uniform_stage(config, activation, (7,))
    write_ready, write_done = _uniform_stage(config, second, (8, 9, 10))
    if (act_ready, write_ready) != (read_done, act_done):
        raise ValueError("Broken saved-dispatch read-before-write chain")
    for index in (*first, *second):
        task = config.all_tasks[index]
        if (task.extra_value_1 not in (GroupedMatmulTaskMode.PER_EXPERT, GroupedMatmulTaskMode.SINGLE_GROUP)
                or task.inputs[0].dynamic_shape != 1
                or task.outputs[0].dynamic_shape != 1 or task.outputs[0].data_type != 2):
            raise ValueError("Saved-dispatch requires canonical per-expert GMM producers")
    return ready, read_done, write_done


def _combine_group(task, spec, completion):
    """Validate canonical transport ports before redirecting the Y source."""
    tiles, remainder = divmod(task.task_split_num, spec.num_experts)
    if remainder or tiles <= 0:
        raise ValueError("Invalid saved-dispatch Combine geometry")
    group, tile = divmod(task.task_index, tiles)
    ports = tuple(task.inputs[index].input_position for index in range(task.num_inputs))
    if (ports != (13, 11, 14, 15) or task.num_outputs != 1 or task.outputs[0].input_position != 12
            or task.inputs[1].base_ptr_offset or task.inputs[1].data_type != 2
            or tuple(task.inputs[index].base_ptr_offset for index in (0, 2, 3)) != (group,) * 3):
        raise ValueError("Noncanonical saved-dispatch Combine ports")
    if not 0 <= group < spec.num_experts or task.dependent_event != completion[group % spec.local_experts][2]:
        raise ValueError("Unproven saved-dispatch Combine reader completion")
    return group, tile, tiles


def _prove_combine(config, spec, completion):
    """Return Y senders after proving complete peer-expert tile coverage."""
    groups = defaultdict(list)
    for task_id in config.vector_task_indices[:config.task_index_num[1]]:
        task = config.all_tasks[task_id]
        if task.task_type != TaskType.TASK_SHMEM_PUT_MEM_SIGNAL or task.inputs[3].input_position != 15:
            continue
        group, tile, tiles = _combine_group(task, spec, completion)
        groups[group].append((task_id, tile, tiles))
    if set(groups) != set(range(spec.num_experts)):
        raise ValueError("Incomplete saved-dispatch Combine groups")
    for group in groups.values():
        if sorted(item[1] for item in group) != list(range(group[0][2])):
            raise ValueError("Incomplete saved-dispatch Combine tiles")
    return [item[0] for group in groups.values() for item in group]


def _prove_storage(config, first, second, combines, hidden):
    """Reject extra X readers or Y writers outside the proven stage chain."""
    expected_first = {index for ids in first.values() for index in ids}
    expected_second = {index for ids in second.values() for index in ids}
    readers, writers, y_readers = set(), set(), set()
    queued = (*config.cube_task_indices[:config.task_index_num[0]],
              *config.vector_task_indices[:config.task_index_num[1]])
    for index in queued:
        task = config.all_tasks[index]
        inputs = {task.inputs[port].input_position for port in range(task.num_inputs)}
        outputs = {task.outputs[port].input_position for port in range(task.num_outputs)}
        if 0 in inputs:
            readers.add(index)
        if 11 in inputs:
            y_readers.add(index)
        if 11 in outputs:
            writers.add(index)
    if (readers, writers, y_readers) != (expected_first, expected_second, set(combines)):
        raise ValueError("Unproven saved-dispatch storage readers or writers")
    if any(config.all_tasks[index].outputs[0].dim[1] != hidden for index in expected_second):
        raise ValueError("Incompatible saved-dispatch Y row width")


def _save_task(spec, expert, lane, lanes, events):
    """Describe one disjoint, dynamically bounded local X copy."""
    task = TaskDescC()
    task.task_type = TaskType.TASK_SAVE_DISPATCH
    task.task_index = expert * lanes + lane
    task.task_split_num = spec.local_experts * lanes
    task.task_split_value = min(spec._saved_dispatch_lanes or lanes, lanes)
    task.dependent_event, task.trigger_event = events[:2]
    task.num_inputs, task.num_outputs = 2, 1
    task.inputs[0].input_position, task.inputs[0].data_type = 0, 2
    task.inputs[1].input_position, task.inputs[1].data_type = 6, 8
    task.outputs[0].input_position, task.outputs[0].data_type = 11, 2
    task.extra_value_2 = spec._vector_event_refresh_us
    return task


def build_saved_dispatch(config: RuntimeConfigC, spec: MegaMoeSpec) -> RuntimeConfigC:
    """Build a separate proven image; diagnostics retain the unchanged original.

    Each complete worker round copies one expert's X in parallel with GMM1.
    Its credits join the original GMM1 fence, delaying SwiGLU until X is owned.
    GMM2 then writes symmetric pos0 and Combine reads Y there. Nonterminal task IDs
    and their worker residues remain unchanged, including guest prefix pruning.
    """
    local, lanes = spec.local_experts, 2 * spec.num_cube_cores
    if (config.num_workers, config.dynamic_data.dynamic_group_size) != (lanes, local):
        raise ValueError("Saved-dispatch runtime disagrees with the shape specification")
    first = _expert_tasks(config, local, TaskType.TASK_GROUPED_MATMUL, 7)
    activation = _expert_tasks(config, local, TaskType.TASK_SWI_GLU, 8)
    second = _expert_tasks(config, local, TaskType.TASK_GROUPED_MATMUL, 11)
    events = [_prove_expert(config, first[expert], activation[expert], second[expert]) for expert in range(local)]
    combines = _prove_combine(config, spec, events)
    _prove_storage(config, first, second, combines, spec.hidden_size)
    result = _grow_runtime(config, local * lanes)
    starts = {ids[0]: expert for expert, ids in activation.items()}
    queue, next_id = [], config.task_num
    for task_id in config.vector_task_indices[:config.task_index_num[1]]:
        if task_id in starts:
            expert = starts[task_id]
            for lane in range(lanes):
                result.all_tasks[next_id] = _save_task(spec, expert, lane, lanes, events[expert])
                queue.append(next_id)
                next_id += 1
            result.all_event_num_triggers[events[expert][1]] += lanes
        queue.append(config.task_num + local * lanes if task_id == config.task_num else task_id)
    for index, task_id in enumerate(queue):
        result.vector_task_indices[index] = task_id
    result.task_num += local * lanes
    result.all_tasks[result.task_num] = config.all_tasks[config.task_num]
    result.task_index_num[1] = len(queue)
    for ids in second.values():
        for task_id in ids:
            result.all_tasks[task_id].outputs[0].input_position = 0
    for task_id in combines:
        result.all_tasks[task_id].inputs[1].input_position = 0
    return result


def saved_dispatch_images(config: RuntimeConfigC, spec: MegaMoeSpec) -> tuple[RuntimeConfigC, RuntimeConfigC]:
    """Use exact serialized capacities for both independent prefix-patch layouts."""
    original = _grow_runtime(config, 0)
    return original, build_saved_dispatch(original, spec)
