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
"""Reorder W1 weight-gradient tasks without moving expert storage or communication."""

import importlib

import torch

from hyper_multicore._loader import NativeComponentUnavailableError
from hyper_multicore.scheduler.config import RuntimeConfigC, TaskSplitValue, TaskType
from hyper_multicore.scheduler.graph import ComputeGraph

__all__ = []


def reverse_w1_gradient_queue(config: RuntimeConfigC, graph: ComputeGraph, topology: TaskSplitValue) -> None:
    """Reverse the W1 expert groups at their existing Cube queue positions.

    Args:
        config: Ordinary backward runtime without physical guest experts.
        graph: Propagated operators in descriptor construction order.
        topology: The topology used to construct the runtime.

    Note:
        Descriptor identities, events, per-core lane assignments and all other
        task positions are retained. The plan cache applies this once during
        preparation; there is no per-forward queue rewrite or device readback.
    """
    home = topology.single_rank_expert_num
    start = 0
    for operation in graph.topological_sort():
        if operation.name == "w1_grad":
            if home < 1 or operation.task_num % home:
                raise ValueError("W1 queue permutation requires complete expert task groups")
            lanes = operation.task_num // home
            for index in range(config.task_index_num[0]):
                task = config.cube_task_indices[index]
                if start <= task < start + operation.task_num:
                    expert, lane = divmod(task - start, lanes)
                    config.cube_task_indices[index] = start + (home - 1 - expert) * lanes + lane
            return
        start += operation.task_num


ADAPTIVE_GRADIENT_QUEUE = "device_w1_shape_auto"
ACT_FIRST_GRADIENT_PROJECTIONS = {
    "device_active_lanes_act_first": (6, 18),
    "device_active_w1_lanes_act_first": (18,),
}
ACTIVE_GRADIENT_PROJECTIONS = {
    "device_active_lanes": (6, 18),
    "device_active_w1_lanes": (18,),
    **ACT_FIRST_GRADIENT_PROJECTIONS,
}
_DEVICE_POLICIES = {
    "device_w1_asc": 1,
    "device_w1_desc": 2,
    "device_w1_asc_join": 3,
    "device_late_auto": 4,
    "device_lane_rotate": 5,
    "device_w1_asc_join_once": 6,
    "device_w1_asc_join_rotate": 7,
    ADAPTIVE_GRADIENT_QUEUE: 8,
}
_DEVICE_LANE_POLICY = _DEVICE_POLICIES["device_lane_rotate"]
_DEVICE_SHAPE_POLICY = _DEVICE_POLICIES[ADAPTIVE_GRADIENT_QUEUE]


def has_adaptive_gradient_native(minimum: int = 2) -> bool:
    """Query the optional marker ABI once per eligible plan, allowing old payloads."""
    adapter = importlib.import_module("hyper_multicore.torch.ops")
    adapter._load_native()  # pylint: disable=protected-access
    version = getattr(torch.ops.hyper_multicore, "mega_moe_adaptive_gradient_version", None)
    return version is not None and version() >= minimum


def _active_wave_is_exact(wave, position, expert, cores):
    """Only mark canonical complete expert waves with unchanged matrix operands."""
    return not any(
        (
            task.task_type != TaskType.TASK_GROUPED_MATMUL
            or task.num_inputs != 3
            or task.num_outputs != 1
            or (task.outputs[0].input_position != position)
            or (task.task_index != expert * cores + lane_id)
            or (task.inputs[0].transpose_flag != 1)
            or (task.inputs[2].input_position != 19)
            or task.extra_value_0
            for lane_id, task in enumerate(wave)
        )
    )


def _wave_has_gradient(wave, projections):
    """Ignore complete nongradient waves without weakening operand validation."""
    return any(task.num_outputs and task.outputs[0].input_position in projections for task in wave)


def mark_active_gradient_lanes(
    config: RuntimeConfigC, experts: int, cores: int, *, projections: tuple[int, ...] = (6, 18)
) -> bool:
    """Mark complete W1/W2 waves with the same expert sequence on every Cube.

    The lane ordinal is counted inside each worker and advances only on an
    actual GEMM. Prove whole-expert waves before writing any descriptor.
    """
    queue = config.cube_task_indices[: config.task_index_num[0]]
    marked = []
    seen: dict[int, set[int]] = {position: set() for position in projections}
    for offset in range(0, len(queue), cores):
        wave = [config.all_tasks[index] for index in queue[offset : offset + cores]]
        if not _wave_has_gradient(wave, seen):
            continue
        position = wave[0].outputs[0].input_position
        expert, lane = divmod(wave[0].task_index, cores)
        if position not in seen or lane or expert >= experts or (expert in seen[position]) or (len(wave) != cores):
            return False
        if not _active_wave_is_exact(wave, position, expert, cores):
            return False
        seen[position].add(expert)
        marked.extend(wave)
    if any((len(phase) != experts for phase in seen.values())):
        return False
    for task in marked:
        task.extra_value_0 = 1519386624
    return True


def _mark_lane_groups(config, spans, policy):
    """Mark only the requested gradient phases and preserve the original fallback."""
    if policy in (_DEVICE_LANE_POLICY, 7, _DEVICE_SHAPE_POLICY):
        stages = ("w2_grad", "w1_grad") if policy == _DEVICE_LANE_POLICY else ("w2_grad",)
        marker = _DEVICE_SHAPE_POLICY if policy == _DEVICE_SHAPE_POLICY else _DEVICE_LANE_POLICY
        for name in stages:
            offset, count = spans[name]
            for index in range(offset, offset + count):
                config.all_tasks[index].extra_value_0 = 1509949440 | marker << 20
        if policy == _DEVICE_LANE_POLICY:
            return True
    return False


def mark_device_gradient_queue(config: RuntimeConfigC, graph: ComputeGraph, mode: str) -> None:
    """Annotate optional native scheduling while retaining the original host queue.

    Args:
        config: Ordinary backward runtime without physical guest experts.
        graph: Propagated graph in descriptor construction order.
        mode: One of the private device policy names.
    """
    policy = _DEVICE_POLICIES[mode]
    spans = {}
    start = 0
    for operation in graph.topological_sort():
        if operation.name in ("w2_grad", "gate_grad", "w1_grad"):
            spans[operation.name] = (start, operation.task_num)
        start += operation.task_num
    if _mark_lane_groups(config, spans, policy):
        return
    gate_start, gate_count = spans["gate_grad"]
    event = config.all_tasks[gate_start + gate_count - 1].trigger_event
    if event >= 1 << 20:
        raise ValueError("Device gradient policies require a completion event below 2**20")
    for name, (offset, count) in spans.items():
        if name == "w2_grad" or (name == "gate_grad" and policy != 4):
            continue
        other = spans["w1_grad" if name == "gate_grad" else "gate_grad"][0]
        for index in range(offset, offset + count):
            task = config.all_tasks[index]
            if task.inputs[2].dim[0] > 64:
                raise ValueError("Device gradient policies support at most 64 local experts")
            task.extra_value_0 = 1509949440 | policy << 20 | event
            task.tiling_data_offset = other


def require_device_gradient_native(mode: str = "device_w1_asc", *, physical: bool = False) -> None:
    """Fail before launch when an older native payload cannot interpret policy markers."""
    if mode == ADAPTIVE_GRADIENT_QUEUE or mode in ACTIVE_GRADIENT_PROJECTIONS:
        version = 2 if mode == ADAPTIVE_GRADIENT_QUEUE else (4 if physical else 3)
        if not has_adaptive_gradient_native(version):
            raise NativeComponentUnavailableError(
                "Device gradient scheduling requires a matching rebuilt multicore native payload"
            )
        return
    adapter = importlib.import_module("hyper_multicore.torch.ops")
    adapter._load_native()  # pylint: disable=protected-access
    try:
        required = {"device_lane_rotate": 2, "device_w1_asc_join_once": 3, "device_w1_asc_join_rotate": 3}.get(mode, 1)
        if torch.ops.hyper_multicore.mega_moe_schedule_version() < required:
            raise RuntimeError("Unsupported native gradient schedule version")
    except (AttributeError, RuntimeError) as error:
        raise NativeComponentUnavailableError(
            "Device gradient scheduling requires a matching rebuilt multicore native payload"
        ) from error
