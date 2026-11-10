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
"""Local shared-expert graph nodes and whole-round queue placements."""

from copy import deepcopy
from dataclasses import dataclass

from hyper_multicore.scheduler.config import (
    ATOMIC_ADD_VALUE_LEN, EVENT_INVALID_ID, RuntimeConfigC, TaskAiCoreType, mega_moe_event_capacity,
    TaskDescC, TaskSplitValue, TaskType,
)
from hyper_multicore.scheduler.graph import ComputeGraph, OperatorNode, OpType, SplitSpec, TensorSpec
from hyper_multicore.scheduler.scheduler import revise_gmm_task_queue_bwd, revise_task_queue
from hyper_multicore.tasks.utils import make_tensor_desc


GMM_TILING_BYTES = 2016
GROUP_LIST_BYTES = 8
FORWARD_POINTER_SLOT = 26
FORWARD_RUNTIME_SLOT = 27
BACKWARD_POINTER_SLOT = 33
BACKWARD_RUNTIME_SLOT = 34


@dataclass(frozen=True)
class SharedFillConfig:
    """Fill fixed local tensors without advancing routed expert event numbers."""

    pointer_slot: int
    runtime_slot: int
    tiling_offset: int
    dependency: int
    trigger: int

    def fill(self, cfg: RuntimeConfigC, op: OperatorNode, tsv: TaskSplitValue) -> None:
        """Append a complete shared stage to the graph's task namespace.

        Args:
            cfg: Cold RuntimeConfig receiving descriptors and queue entries.
            op: Shared graph operator with an assigned task count.
            tsv: Running descriptor and queue offsets.
        """
        cube = op.op_type == OpType.GMM
        queue = cfg.cube_task_indices if cube else cfg.vector_task_indices
        queue_start = tsv.pre_cube_task_num if cube else tsv.pre_vector_task_num
        task_type = {OpType.GMM: TaskType.TASK_GROUPED_MATMUL,
                     OpType.SWIGLU: TaskType.TASK_SWI_GLU,
                     OpType.SWIGLU_GRAD: TaskType.TASK_SWI_GLU_GRAD}[op.op_type]
        for index in range(op.task_num):
            task = TaskDescC()
            task.task_type = task_type
            task.task_aicore_type = TaskAiCoreType.TASK_AICORE_CUBE if cube else TaskAiCoreType.TASK_AICORE_VECTOR
            task.num_inputs, task.num_outputs = len(op.inputs), len(op.outputs)
            for target, specs in ((task.inputs, op.inputs), (task.outputs, op.outputs)):
                for pos, spec in enumerate(specs):
                    group_list = spec.name == "shared_group_list"
                    target[pos] = make_tensor_desc(
                        tensor_type=0 if group_list else 2,
                        input_position=self.runtime_slot if group_list else self.pointer_slot,
                        dynamic_dim=0 if group_list else spec.param_position,
                        base_ptr_offset=0 if cube else index * op.split_value * spec.shape[1],
                        data_type=spec.dtype_size, transpose_flag=int(spec.transpose), shape=spec.shape,
                    )
            task.dependent_event, task.trigger_event = self.dependency, self.trigger
            task.task_index, task.task_split_num, task.task_split_value = index, op.task_num, op.split_value
            task.tiling_data_position, task.tiling_data_offset = self.runtime_slot, self.tiling_offset
            cfg.all_tasks[tsv.pre_task_num + index] = task
            queue[queue_start + index] = tsv.pre_task_num + index
        cfg.all_event_num_triggers[self.trigger] = op.task_num
        cfg.task_index_num[0 if cube else 1] += op.task_num
        tsv.pre_task_num += op.task_num
        if cube:
            tsv.pre_cube_task_num += op.task_num
        else:
            tsv.pre_vector_task_num += op.task_num


def add_shared_graph(
    graph: ComputeGraph, tsv: TaskSplitValue, hidden: int, intermediate: int, cores: int, *, backward: bool,
) -> None:
    """Add an independent, local SwiGLU branch with caller-owned dense weights.

    Args:
        graph: Routed graph receiving the additional independent branch.
        tsv: Token and expert topology.
        hidden: Input/output feature width.
        intermediate: Shared SwiGLU feature width.
        cores: Cube worker count.
        backward: Whether to construct shared dX and dW instead of forward.
    """
    if (mega_moe_event_capacity(tsv.all_expert_num, tsv.ep, shared_expert=True)
            != mega_moe_event_capacity(tsv.all_expert_num, tsv.ep)):
        raise ValueError("Shared fusion exceeds the existing symmetric event capacity")
    tsv.shared_expert = True
    tokens = tsv.per_expert_seq_to_other
    if tokens % 128 or intermediate <= 0:
        raise ValueError("Shared branch requires positive intermediate size and token rows divisible by 128")
    pointer_slot, runtime_slot = ((BACKWARD_POINTER_SLOT, BACKWARD_RUNTIME_SLOT) if backward
                                  else (FORWARD_POINTER_SLOT, FORWARD_RUNTIME_SLOT))
    event_base = tsv.all_event_num + 3 + 2 * ATOMIC_ADD_VALUE_LEN
    stride = GMM_TILING_BYTES * cores
    group = TensorSpec("shared_group_list", [1], runtime_slot, dtype_size=8)

    def tensor(name: str, shape: list[int], index: int, transpose: bool = False) -> TensorSpec:
        """Describe one caller-owned local shared tensor.

        Args:
            name: Graph tensor name.
            shape: Physical tensor shape.
            index: Address-table entry owned by the caller.
            transpose: Whether the GMM consumes a transposed view.
        """
        return TensorSpec(name, shape, index, transpose=transpose)

    def add(name: str, kind: OpType, inputs: list[TensorSpec], output: TensorSpec,
            tiling: int, dependency: int, trigger: int) -> OperatorNode:
        """Append a complete local stage with explicit device event dependencies.

        Args:
            name: Stage name in the shared graph.
            kind: Compute operation type.
            inputs: Caller-owned source tensors.
            output: Caller-owned destination tensor.
            tiling: Byte offset within the shared runtime buffer.
            dependency: Required producer event or EVENT_INVALID_ID.
            trigger: Completion event published by this stage.
        """
        cube = kind == OpType.GMM
        count = cores if cube else tokens // 128
        op = OperatorNode(
            name=name, diagnostic_name=name, op_type=kind,
            inputs=inputs + [group] if cube else inputs, outputs=[output],
            param_positions=[], split_value=tokens if cube else 128,
            split_spec=SplitSpec(split_inputs=None, split_output_dims=[0], task_num_fn=lambda _: count),
            tiling_position=runtime_slot,
            fill_config=SharedFillConfig(pointer_slot, runtime_slot, tiling, dependency, trigger),
        )
        graph.add_op(op)
        return op

    if not backward:
        x = tensor("shared_x", [tokens, hidden], 0)
        w1 = tensor("shared_w1", [1, 2 * intermediate, hidden], 1, True)
        w2 = tensor("shared_w2", [1, hidden, intermediate], 2, True)
        up = tensor("shared_up", [tokens, 2 * intermediate], 3)
        act = tensor("shared_activation", [tokens, intermediate], 4)
        out = tensor("shared_output", [tokens, hidden], 5)
        first = add("shared_up_proj", OpType.GMM, [x, w1], up, GROUP_LIST_BYTES,
                    EVENT_INVALID_ID, event_base)
        middle = add("shared_swiglu", OpType.SWIGLU, [up], act, GROUP_LIST_BYTES + 2 * stride,
                     event_base, event_base + 1)
        last = add("shared_down_proj", OpType.GMM, [act, w2], out, GROUP_LIST_BYTES + stride,
                   event_base + 1, event_base + 2)
        graph.add_edge(first, middle).add_edge(middle, last)
        return

    dy = tensor("shared_dy", [tokens, hidden], 0)
    act = tensor("shared_activation", [tokens, intermediate], 1, True)
    dw2 = tensor("shared_dw2", [1, intermediate, hidden], 2)
    w2 = tensor("shared_w2", [1, hidden, intermediate], 3)
    dact = tensor("shared_dact", [tokens, intermediate], 4)
    up = tensor("shared_up", [tokens, 2 * intermediate], 5)
    dup = tensor("shared_dup", [tokens, 2 * intermediate], 6)
    w1 = tensor("shared_w1", [1, 2 * intermediate, hidden], 7)
    dx = tensor("shared_dx", [tokens, hidden], 8)
    x = tensor("shared_x", [tokens, hidden], 9, True)
    dw1 = tensor("shared_dw1", [1, hidden, 2 * intermediate], 10)
    first = None
    if cores != 32:
        first = add("shared_act_grad", OpType.GMM, [dy, w2], dact, GROUP_LIST_BYTES,
                    EVENT_INVALID_ID, event_base)
    add("shared_w2_grad", OpType.GMM, [act, dy], dw2, GROUP_LIST_BYTES + 3 * stride,
        EVENT_INVALID_ID, event_base + 1)
    middle = None
    if first is not None:
        middle = add("shared_swiglu_grad", OpType.SWIGLU_GRAD, [dact, up], dup, GROUP_LIST_BYTES + 4 * stride,
                     event_base, event_base + 2)
    # A5 native producers preserve BF16 activation rounding before shared weight reductions.
    dependency = event_base + 2 if middle is not None else EVENT_INVALID_ID
    gate = add("shared_gate_grad", OpType.GMM, [dup, w1], dx, GROUP_LIST_BYTES + stride,
               dependency, event_base + 3)
    weight = add("shared_w1_grad", OpType.GMM, [x, dup], dw1, GROUP_LIST_BYTES + 2 * stride,
                 dependency, event_base + 4)
    if first is not None and middle is not None:
        graph.add_edge(first, middle).add_edge(middle, gate).add_edge(middle, weight)



def _task_ids_by_stage(graph: ComputeGraph) -> dict[str, list[int]]:
    """Return each graph stage's contiguous cold descriptor identifiers.

    Args:
        graph: Graph whose task counts have already been propagated.

    Returns:
        Task identifiers keyed by stage name in topological descriptor order.
    """
    stages = {}
    offset = 0
    for operator in graph.topological_sort():
        stages[operator.name] = list(range(offset, offset + operator.task_num))
        offset += operator.task_num
    return stages


def reorder_shared_tasks(
    cfg: RuntimeConfigC, graph: ComputeGraph, tsv: TaskSplitValue, cores: int,
    *, backward: bool, placement: str,
) -> None:
    """Place whole shared Cube rounds without changing routed task dependencies.

    Args:
        cfg: Runtime whose queues are reordered in place.
        graph: Graph with assigned task counts and profiling order.
        tsv: Local rank and expert topology.
        cores: Participating Cube workers.
        backward: Select the gradient-stage placement.
        placement: Internal experiment policy: idle or front.
    """
    if placement not in ("idle", "front"):
        raise ValueError("shared placement must be idle or front")
    stages = _task_ids_by_stage(graph)
    shared_ids = {index for name, ids in stages.items() if name.startswith("shared_") for index in ids}
    cube = [index for index in cfg.cube_task_indices[:cfg.task_index_num[0]] if index not in shared_ids]
    vector = [index for index in cfg.vector_task_indices[:cfg.task_index_num[1]] if index not in shared_ids]
    cfg.cube_task_indices[:len(cube)] = cube
    cfg.vector_task_indices[:len(vector)] = vector
    swiglu_name = "swiglu_grad" if backward else "swiglu"
    revise_task_queue(cfg, tsv, len(stages["dispatch"]), len(stages[swiglu_name]), len(stages["combine"]))
    if backward:
        revise_gmm_task_queue_bwd(cfg, tsv, len(stages["act_grad"]), cores)
    cube = list(cfg.cube_task_indices[:len(cube)])
    vector = list(cfg.vector_task_indices[:len(vector)])
    if backward:
        first = stages.get("shared_act_grad", []) + stages["shared_w2_grad"]
        last = stages["shared_gate_grad"] + stages["shared_w1_grad"]
        activation = stages.get("shared_swiglu_grad", [])
    else:
        first, last, activation = (stages[name] for name in ("shared_up_proj", "shared_down_proj", "shared_swiglu"))
    if placement == "front":
        cube = first + last + cube
        vector = vector[:-1] + activation + vector[-1:]
    else:
        if backward:
            # Hide shared dActivation in the initial dispatch wait. Finish its
            # vector work before inserting dependent Cube rounds at the phase boundary.
            phase_end = 2 * tsv.single_rank_expert_num * cores
            cube = (stages.get("shared_act_grad", []) + cube[:phase_end] + stages["shared_w2_grad"]
                    + last + cube[phase_end:])
            activation_start = len(stages["dispatch"])
        else:
            # Routed Cube work ends before Combine. Insert shared SwiGLU late
            # enough for the shared projection to finish without blocking early sends.
            cube += first + last
            combine_prefix = (3 * len(stages["combine"]) // (4 * cores)) * cores
            activation_start = len(stages["dispatch"]) + len(stages[swiglu_name]) + combine_prefix
        # Even AIV workers consume this suffix; odd workers keep the routed queue.
        activation_start = len(vector) - 1
        vector = vector[:activation_start] + activation + vector[activation_start:]
    cfg.cube_task_indices[:len(cube)] = cube
    cfg.vector_task_indices[:len(vector)] = vector



def _prioritize_shared_forward_rounds(
    config: RuntimeConfigC,
    graph: ComputeGraph,
    *,
    num_cube_cores: int,
    include_down: bool = False,
) -> None:
    """Move complete independent shared forward rounds while preserving all task contracts.

    Args:
        config: Cold forward RuntimeConfig; no device image may have been published yet.
        graph: The same graph used to build the task descriptors.
        num_cube_cores: Complete Cube round width.
        include_down: Move shared Down immediately after Up; otherwise retain its original relative position.

    Raises:
        ValueError: If shared forward rounds are absent, incomplete, or not uniquely present in the Cube queue.
    """
    if isinstance(num_cube_cores, bool) or not isinstance(num_cube_cores, int) or num_cube_cores <= 0:
        raise ValueError("num_cube_cores must be a positive integer")
    stages = _task_ids_by_stage(graph)
    selected = ("shared_up_proj", "shared_down_proj") if include_down else ("shared_up_proj",)
    rounds = []
    queue = list(config.cube_task_indices[:config.task_index_num[0]])
    for name in selected:
        identifiers = stages.get(name, [])
        if len(identifiers) != num_cube_cores:
            raise ValueError(f"{name} must contain one complete Cube round")
        for identifier in identifiers:
            task = config.all_tasks[identifier]
            if (task.task_type, task.inputs[0].tensor_type) != (TaskType.TASK_GROUPED_MATMUL, 2):
                raise ValueError(f"{name} contains a non-shared GMM task")
            if queue.count(identifier) != 1:
                raise ValueError(f"{name} must appear exactly once in the Cube queue")
        if sorted(config.all_tasks[index].task_index for index in identifiers) != list(range(num_cube_cores)):
            raise ValueError(f"{name} has incomplete logical Cube ownership")
        rounds.extend(identifiers)
    moved = set(rounds)
    reordered = rounds + [index for index in queue if index not in moved]
    config.cube_task_indices[:len(reordered)] = reordered


def _shared_l1_view(*_args: object, **_kwargs: object) -> None:
    """Reject opaque-pointer dumps before a diagnostic kernel can read a wrong view."""
    raise NotImplementedError("Shared pointer-table L1 dumps are unsupported; use cycle profiling")


def bind_shared_diagnostic_graph(graph: ComputeGraph, _config: RuntimeConfigC) -> ComputeGraph:
    """Describe indirect native input ports without mutating the execution graph.

    Args:
        graph: Original routed/shared scheduling graph.
        _config: Private diagnostic configuration with unchanged task descriptors.

    Returns:
        A private graph with literal native pointer-table/runtime port bindings.
    """
    diagnostic = deepcopy(graph)
    for op in diagnostic.topological_sort():
        if not op.name.startswith("shared_"):
            continue
        for spec in op.inputs + op.outputs:
            spec.param_position = (op.fill_config.runtime_slot if spec.name == "shared_group_list"
                                   else op.fill_config.pointer_slot)
        op.param_positions = [spec.param_position for spec in op.inputs + op.outputs]
        op.diagnostic_view_resolver = _shared_l1_view
    return diagnostic
