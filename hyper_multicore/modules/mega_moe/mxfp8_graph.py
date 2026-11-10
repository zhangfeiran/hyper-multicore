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
"""Independent MXFP8 expert task graphs sharing the established BF16 transport."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from typing import Any

from hyper_multicore.scheduler.config import (
    ATOMIC_ADD_VALUE_LEN,
    EVENT_INVALID_ID,
    MIN_EVENT_CAPACITY,
    TaskAiCoreType,
    TaskDescC,
    TaskSplitValue,
    TaskType,
    configure_ready_handshake,
    init_task_split_value,
    validate_runtime_config,
)
from hyper_multicore.scheduler.graph import ComputeGraph, OperatorNode, OpType, SplitSpec, TensorSpec
from hyper_multicore.scheduler.mxfp8 import (
    MX_GMM_TASK_TYPE,
    MX_GMM_OUTPUT_FP32,
    make_mx_gmm_descriptor,
    serialize_mx_gmm_descriptors,
)
from hyper_multicore.scheduler.mxfp8_quant import (
    MX_QUANT_TASK,
    MX_SWIGLU_TASK,
    MX_SWIGLU_GRAD_TASK,
    MX_SWIGLU_ROW_TASK,
    MX_QUANT_ROW_TASK,
    MxQuantDescriptor,
    serialize_mx_quant_descriptors,
)
from hyper_multicore.scheduler.runtime import allocate_runtime_config
from hyper_multicore.scheduler.scheduler import revise_task_queue
from hyper_multicore.scheduler.expert_window import reorder_mx_expert_windows
from hyper_multicore.scheduler.all_transport import balance_all_transport
from hyper_multicore.scheduler.vector_roles import configure_vector_roles, configure_rule_stages
from hyper_multicore.scheduler.dispatch_quant import DispatchQuantStage, configure_dispatch_quant_queue
from hyper_multicore.scheduler.planned_queue import PLANNED_QUEUE_TAG, configure_planned_queue
from hyper_multicore.scheduler.window_queue import configure_window_queue
from hyper_multicore.shmem._runtime import configured_udma_qp_count
from hyper_multicore.tasks.alltoall import AllToAllFillConfig, AllToAllType
from hyper_multicore.tasks.utils import (
    add_dynamic_data,
    add_terminate,
    advance_tsv_cube,
    advance_tsv_vector,
    make_tensor_desc,
)

from .spec import MegaMoeSpec
from .mxfp8_ingress import MX_SOURCE_ROW_BUFFER_COUNT, MX_SOURCE_ROW_READY_TASK
from .mxfp8_gradients import configure_mx_gradient_helpers


_MX_GMM_PREZEROED_OUTPUT = 2
MX_PROJECTION_READY_MASK = 0x0000FFFF
MX_READY_DEPTH8 = 0x00010000
MX_READY_DEPTH1 = 0x00020000


@dataclass
class _MXTaskValues(TaskSplitValue):
    compute_stages: int = 4

    @property
    def all_event_num(self) -> int:
        """Reserve per-expert completion events for every compute stage."""
        return 1 + self.all_expert_num + self.compute_stages * self.single_rank_expert_num


@dataclass(frozen=True)
class MXFP8Runtime:
    graph: ComputeGraph
    config: Any
    gmm_descriptors: bytes
    quant_descriptors: bytes
    buffer_count: int
    swiglu_row_only: bool = False
    transport_schedule: str = "legacy_odd"
    kernel_gradients: bool = False
    weight_prefetch: bool = False
    source_row_fp8: bool = False


def _fill_compute_stage(
    cfg: Any, op: OperatorNode, tsv: TaskSplitValue, kind: int, workers: int, first: bool, indices: list[int]
) -> None:
    cube = kind == MX_GMM_TASK_TYPE
    for index, record_index in enumerate(indices):
        expert = index // workers
        task = TaskDescC()
        task.task_type = kind
        task.task_aicore_type = TaskAiCoreType.TASK_AICORE_CUBE if cube else TaskAiCoreType.TASK_AICORE_VECTOR
        task.num_inputs, task.num_outputs = len(op.inputs), len(op.outputs)
        for slot, spec in enumerate(op.inputs):
            task.inputs[slot] = make_tensor_desc(
                input_position=spec.param_position, data_type=spec.dtype_size, shape=spec.shape
            )
        for slot, spec in enumerate(op.outputs):
            task.outputs[slot] = make_tensor_desc(
                input_position=spec.param_position, data_type=spec.dtype_size, shape=spec.shape
            )
        rank_offset = tsv.rank_id * tsv.single_rank_expert_num if first else 0
        task.dependent_event = tsv.pre_pre_event_num + rank_offset + expert + 1
        task.trigger_event = tsv.pre_event_num + expert + 1
        cfg.all_event_num_triggers[task.trigger_event] = workers
        task.task_index, task.task_split_num = index, len(indices)
        task.task_split_value, task.extra_value_0 = 1, record_index
        task.tiling_data_position = 0xFFFFFFFF
        task_index = tsv.pre_task_num + index
        cfg.all_tasks[task_index] = task
        queue = cfg.cube_task_indices if cube else cfg.vector_task_indices
        offset = tsv.pre_cube_task_num if cube else tsv.pre_vector_task_num
        queue[offset + index] = task_index
    cfg.task_index_num[0 if cube else 1] += len(indices)
    advance = advance_tsv_cube if cube else advance_tsv_vector
    advance(tsv, len(indices), tsv.single_rank_expert_num)


@dataclass
class _GmmStage:
    layout: str
    m: int
    n: int
    k: int
    x_scale: int
    w_scale: int
    workers: int
    records: list
    home_experts: int = 0
    guest_data_position: int = 0
    guest_scale_position: int = 0

    def fill(self, cfg: Any, op: OperatorNode, tsv: TaskSplitValue) -> None:
        """Populate one task per expert and worker with byte-accurate descriptors.

        Args:
            cfg: Mutable host runtime image being assembled.
            op: Scheduled operator and its buffer specifications.
            tsv: Current task offsets and per-expert split values.
        """
        indices = []
        task_start = tsv.pre_task_num
        for expert in range(tsv.single_rank_expert_num):
            guest = self.home_experts > 0 and expert >= self.home_experts
            weight_slot = expert - self.home_experts if guest else expert
            for worker in range(self.workers):
                indices.append(len(self.records))
                self.records.append(
                    make_mx_gmm_descriptor(
                        layout=self.layout,
                        m=self.m,
                        n=self.n,
                        k=self.k,
                        x_scale_position=self.x_scale,
                        w_scale_position=self.guest_scale_position if guest and self.layout != "TN" else self.w_scale,
                        cube_count=self.workers,
                        cube_index=worker,
                        group_list_offset=expert * 8,
                        w_offset=0 if self.layout == "TN" else weight_slot * self.n * self.k,
                        w_scale_offset=0 if self.layout == "TN" else weight_slot * self.n * self.k // 32,
                        y_offset=weight_slot * self.m * self.n * 4 if self.layout == "TN" else 0,
                        output_dtype=MX_GMM_OUTPUT_FP32 if self.layout == "TN" else 0,
                    )
                )
                if self.layout == "TN":
                    self.records[-1].flags |= _MX_GMM_PREZEROED_OUTPUT
        _fill_compute_stage(cfg, op, tsv, MX_GMM_TASK_TYPE, self.workers, False, indices)
        if self.home_experts and self.layout != "TN":
            for index in range(self.home_experts * self.workers, len(indices)):
                task = cfg.all_tasks[task_start + index]
                task.inputs[1].input_position = self.guest_data_position


@dataclass
class _QuantStage:
    kind: int
    columns: int
    workers: int
    first: bool
    records: list

    def fill(self, cfg: Any, op: OperatorNode, tsv: TaskSplitValue) -> None:
        """Populate one task per expert and worker with byte-accurate descriptors.

        Args:
            cfg: Mutable host runtime image being assembled.
            op: Scheduled operator and its buffer specifications.
            tsv: Current task offsets and per-expert split values.
        """
        indices = []
        for expert in range(tsv.single_rank_expert_num):
            for worker in range(self.workers):
                indices.append(len(self.records))
                self.records.append(
                    MxQuantDescriptor(
                        columns=self.columns,
                        expert_index=expert,
                        worker_index=worker,
                        worker_count=self.workers,
                        task_type=self.kind,
                    )
                )
        _fill_compute_stage(cfg, op, tsv, self.kind, self.workers, self.first, indices)


def _add_compute(
    graph: ComputeGraph,
    name: str,
    specs: dict[int, TensorSpec],
    inputs: list[int],
    outputs: list[int],
    fill: Any,
    workers: int,
    op_type: OpType = OpType.GMM,
) -> None:
    graph.add_op(
        OperatorNode(
            name=name,
            op_type=op_type,
            inputs=[specs[index] for index in inputs],
            outputs=[specs[index] for index in outputs],
            param_positions=inputs + outputs,
            split_value=1,
            split_spec=SplitSpec(
                split_inputs=None,
                split_output_dims=[0] * len(outputs),
                task_num_fn=lambda tsv: tsv.single_rank_expert_num * workers,
            ),
            tiling_position=-1,
            fill_config=fill,
        )
    )


def _add_transport(
    graph: ComputeGraph, specs: dict[int, TensorSpec], *, forward: bool, dispatch_split: int, combine_split: int
) -> None:
    source, target, size, destination = (13, 14, 15, 12) if forward else (14, 15, 16, 13)
    payload = 11 if forward else 12
    for name, inputs, output, split, kind, group in (
        ("dispatch", [1, 2, 3, 4], 0, dispatch_split, AllToAllType.DISPATCH, "dispatch"),
        ("combine", [source, payload, target, size], destination, combine_split, AllToAllType.COMBINE, "combine"),
    ):
        fill = AllToAllFillConfig(moe_type=kind, advance="vector", event_group=1)
        graph.add_op(
            OperatorNode(
                name=name,
                op_type=OpType.ALLTOALL,
                inputs=[specs[index] for index in inputs],
                outputs=[specs[output]],
                param_positions=inputs + [output],
                split_value=split,
                split_spec=SplitSpec(
                    split_inputs=None,
                    split_output_dims=[0],
                    task_num_fn=lambda tsv, tile=split: tsv.all_expert_num * (tsv.seq_size // tile),
                ),
                tiling_position=-1,
                fill_config=fill,
            )
        )
        if group == "dispatch":
            fill.event_group = specs[1].shape[0]


def _buffer_specs(spec: MegaMoeSpec, forward: bool) -> dict[int, TensorSpec]:
    h, width, rows, experts = spec.hidden_size, spec.intermediate_size, spec.routed_slots, spec.local_experts
    count = 27 if forward else 33
    result = {index: TensorSpec(f"buffer_{index}", [rows, h], index, dtype_size=1) for index in range(count)}
    ordinary = (
        {0: h, 2: h, 7: 2 * width, 11: h, 12: h}
        if forward
        else {0: h, 2: h, 6: h, 8: width, 9: 2 * width, 12: h, 13: h, 18: 2 * width}
    )
    for index, columns in ordinary.items():
        result[index].shape, result[index].dtype_size = [rows, columns], 2
        if not forward and index in (6, 18):
            result[index].dtype_size = 4
    metadata = [1, 3, 13, 14] if forward else [1, 3, 14, 15]
    sizes = [4, 15] if forward else [4, 16]
    for index in metadata + sizes:
        result[index].shape = [spec.num_experts]
        result[index].dtype_size = 4 if index in sizes else 8
    for index in ([6, 10] if forward else [19]):
        result[index].shape, result[index].dtype_size = [experts], 8
    # Only the transport worker uses legacy dynamic tensor offsets. MX tasks
    # resolve token and packed-scale ranges directly from the live group list.
    result[0].is_dynamic = True
    payload = 11 if forward else 12
    result[payload].is_dynamic, result[payload].tensor_type = True, 1
    return result


def _apply_mx_poll_policy(cfg: Any, policy: str) -> None:
    """Use queue roles because transport descriptor core tags do not define scheduling."""
    if policy not in ("0", "fast"):
        raise ValueError("mx_poll_policy must be exactly 0 or fast.")
    if policy == "0":
        return
    seen = set()
    for indices, count, interval in zip(
        (cfg.cube_task_indices, cfg.vector_task_indices), cfg.task_index_num[:2], (10, 30)
    ):
        for task_id in indices[:count]:
            if task_id in seen:
                raise ValueError(f"MX poll policy requires unique queue membership: task {task_id}.")
            seen.add(task_id)
            task = cfg.all_tasks[task_id]
            if task.dependent_event != EVENT_INVALID_ID:
                task.extra_value_2 = interval


def _validate_mx_options(spec: MegaMoeSpec, forward: bool, row_only: bool, expert_window: int) -> bool:
    """Validate precision and queues, returning the independent backward dependency policy."""
    b0_backward_dag = _b0_backward_dag_enabled(spec)
    if row_only and not forward:
        raise ValueError("Row-only SwiGLU is valid only for a forward without saved backward state.")
    if spec.dispatch_mode != "push" or spec.swiglu_limit is not None:
        raise ValueError("MXFP8 task graphs currently require push transport and unclamped SwiGLU.")
    if configured_udma_qp_count() < 16 and spec.mx_vector_policy != "window2":
        raise ValueError("QP1/QP8 MXFP8 requires the owner-validated window2 queue")
    roles = spec.mx_vector_policy in ("roles", "roles_ready", "roles_rule", "planned", "window2")
    if roles and (spec.num_cube_cores != 32 or (row_only and spec.mx_vector_policy != "window2") or expert_window
                  or spec.dispatch_split != spec.combine_split):
        raise ValueError("Vector roles require Cube32, dual-axis, equal transport splits and no expert window.")
    return b0_backward_dag or spec.mx_vector_policy == "roles_ready"


def _b0_backward_dag_enabled(spec: MegaMoeSpec) -> bool:
    """Use independent home-only gradients while retaining the active replica return protocol."""
    option = os.environ.get("HYPER_PARALLEL_MX_B0_BACKWARD_DAG", "0")
    if option not in ("0", "1"):
        raise ValueError("HYPER_PARALLEL_MX_B0_BACKWARD_DAG must be exactly 0 or 1")
    if option == "1" and spec.mx_vector_policy != "window2":
        raise ValueError("B0 backward DAG requires window2")
    return option == "1" and not spec.replica_slots_per_rank


def _validate_dispatch_quant(spec: MegaMoeSpec, row_only: bool, expert_window: int, event_base: int) -> None:
    """Bound the optional dispatch-quant prototype by its existing event reservation."""
    if row_only:
        raise ValueError("Dispatch quant prototype requires the rebuilt dual-axis entry; row-only is not built.")
    if expert_window or spec.mx_vector_policy not in ("roles", "roles_rule", "planned") or spec.dispatch_split % 64:
        raise ValueError(
            "Dispatch quant requires roles, roles_rule or planned, no expert window and a 64-row aligned split."
        )
    fragments = spec.local_num_tokens // spec.dispatch_split
    if event_base + spec.local_experts * (fragments + 1) + ATOMIC_ADD_VALUE_LEN > MIN_EVENT_CAPACITY:
        raise ValueError("Dispatch quant exceeds the existing event budget; SHMEM expansion is forbidden.")


def _revise_mx_task_queue(cfg: Any, values: TaskSplitValue, graph: ComputeGraph, order: list[str]) -> None:
    """Count vector mathematics separately from dispatch and combine queue work."""
    middle_vectors = sum(
        graph.get_op(name).task_num
        for name in order
        if name not in ("dispatch", "combine") and graph.get_op(name).op_type == OpType.SWIGLU
    )
    revise_task_queue(cfg, values, graph.get_op("dispatch").task_num,
                      middle_vectors, graph.get_op("combine").task_num)


def _configure_mx_vector_policy(cfg: Any, graph: ComputeGraph, spec: MegaMoeSpec, forward: bool,
                                transport_schedule: str) -> str:
    """Apply the selected existing queue policy without changing its event contract."""
    b0_backward_dag = _b0_backward_dag_enabled(spec) and not forward
    _apply_mx_poll_policy(cfg, spec.mx_poll_policy)
    if spec.mx_vector_policy in ("roles", "roles_ready", "roles_rule", "planned", "window2"):
        configure_vector_roles(cfg, graph, ready=spec.mx_vector_policy == "roles_ready" or b0_backward_dag,
                               forward=forward)
        transport_schedule = spec.mx_vector_policy + ": odd communication, even mathematics, no padding"
    if spec.mx_vector_policy == "roles_rule":
        configure_rule_stages(cfg, graph, forward=forward)
        transport_schedule = "roles_rule: explicit ready heads, fair dispatch/combine"
    if spec.mx_vector_policy == "planned":
        configure_planned_queue(cfg, spec, forward=forward)
        transport_schedule = "planned CSR: fixed odd communication, arbitrary AIV mathematics"
    if spec.mx_vector_policy == "window2":
        configure_window_queue(cfg, graph, spec, forward=forward, activation_first=b0_backward_dag)
        if _mx_gradient_return(spec, forward):
            configure_mx_gradient_helpers(cfg)
        transport_schedule = "window2 CSR: odd communication, even math, two-expert Cube lookahead"
    return transport_schedule


def _mx_gradient_return(spec: MegaMoeSpec, forward: bool) -> bool:
    """Keep the dedicated projection-return protocol behind its supported queue."""
    enabled = bool(not forward and spec.replica_slots_per_rank
                   and spec.replica_transport == "shmem_signal_kernel_gradient")
    if enabled and spec.mx_vector_policy != "window2":
        raise ValueError("MX kernel gradient return requires window2 queues")
    return enabled


def _mx_guest_positions(spec: MegaMoeSpec, forward: bool, name: str, layout: str,
                        kernel_gradients: bool) -> dict[str, int]:
    """Select separate current guest weight axes or symmetric FP32 destinations."""
    if not spec.replica_slots_per_rank:
        return {}
    home = spec.local_experts - spec.replica_slots_per_rank
    if layout == "TN":
        return {"home_experts": home} if kernel_gradients else {}
    matrix = 0 if name in ("up", "dx") else 1
    position = (27 if forward else 33) + 2 * matrix
    return {"home_experts": home, "guest_data_position": position, "guest_scale_position": position + 1}


def _tag_mx_weight_readers(config: Any, spec: MegaMoeSpec, ready_buffer: int) -> None:
    """Fence only guest matrix reads; preserve home/TN tasks and every producer event."""
    if spec.replica_transport != "shmem_signal_kernel_gradient" or spec.mx_vector_policy != "window2":
        raise ValueError("MX projection prefetch requires window2 and the symmetric gradient inbox")
    home = spec.local_experts - spec.replica_slots_per_rank
    for task in config.all_tasks[:config.task_num]:
        if (task.task_type == MX_GMM_TASK_TYPE and task.outputs[0].input_position in (7, 8, 11, 12)
                and task.inputs[1].input_position in (27, 29, 33, 35)):
            if task.task_index // spec.num_cube_cores < home or task.extra_value_1:
                raise ValueError("MX prefetch reader tags must refer only to guest weight matrices")
            task.extra_value_1 = ready_buffer + 1


def _finish_mx_runtime(graph: ComputeGraph, cfg: Any, gmms: list, quants: list, base_buffers: int,
                        spec: MegaMoeSpec, row_only: bool, schedule: str,
                        gradients: bool, prefetch: bool) -> MXFP8Runtime:
    """Bind reader flags and byte-buffer bounds to the exact completed native protocol."""
    ready_buffer = base_buffers + 4 * bool(spec.replica_slots_per_rank)
    if prefetch:
        _tag_mx_weight_readers(cfg, spec, ready_buffer)
    return MXFP8Runtime(graph, cfg, serialize_mx_gmm_descriptors(gmms), serialize_mx_quant_descriptors(quants),
                        ready_buffer + prefetch, row_only, schedule, gradients, prefetch)


def _validate_mx_ready_depth(spec: MegaMoeSpec, depth: int) -> None:
    """Reject invalid lookahead before allocating or scheduling a native runtime image."""
    if depth not in (0, 1, 8) or (depth and spec.mx_vector_policy != "window2"):
        raise ValueError("Ready depth must be 0, 1 or 8 on window2.")


def _configure_mx_ready_depth(config: Any, depth: int) -> None:
    """Keep the projection-ready buffer code while marking bounded math lookahead."""
    if not depth:
        return
    if config.task_index_num[0] // 32 > 64 or any(
            config.mix_task_indices[worker + 1] - config.mix_task_indices[worker] > 64
            for worker in range(0, 64, 2)):
        raise ValueError("Ready math queues support at most 64 tasks per worker.")
    flag = MX_READY_DEPTH8 if depth == 8 else MX_READY_DEPTH1
    for task in config.all_tasks[:config.task_num]:
        if task.task_type in (MX_GMM_TASK_TYPE, MX_QUANT_TASK, MX_SWIGLU_TASK,
                              MX_SWIGLU_GRAD_TASK, MX_SWIGLU_ROW_TASK, MX_QUANT_ROW_TASK):
            if task.extra_value_1 & ~MX_PROJECTION_READY_MASK:
                raise ValueError("MX task flags overlap the projection-ready buffer code.")
            task.extra_value_1 |= flag


def with_source_row_ingress(spec: MegaMoeSpec, runtime: MXFP8Runtime) -> MXFP8Runtime:
    """Keep original producer credits while replacing completed input quantization with a barrier.

    Args:
        spec: Bound B0 window2 topology.
        runtime: Completed row-only forward graph whose configuration remains unchanged.

    Returns:
        An independent configuration with source-row byte bindings and the original event protocol.
    """
    if (not runtime.swiglu_row_only or runtime.buffer_count != 27 or spec.replica_slots_per_rank
            or spec.mx_vector_policy != "window2" or runtime.kernel_gradients or runtime.weight_prefetch):
        raise ValueError("Source-row FP8 requires a no-grad B0 window2 forward without guest helpers.")
    config = type(runtime.config).from_buffer_copy(bytes(runtime.config))
    for task in config.all_tasks[:config.task_num]:
        if task.task_type == MX_QUANT_ROW_TASK:
            task.task_type = MX_SOURCE_ROW_READY_TASK
    return replace(runtime, config=config, buffer_count=MX_SOURCE_ROW_BUFFER_COUNT, source_row_fp8=True)


def _token_mte_transfer_tasks(runtime: MXFP8Runtime) -> set[int]:
    """Select ordinary token transfers while leaving source-row dispatch unmarked."""
    selected, offset = set(), 0
    for operator in runtime.graph.topological_sort():
        if operator.name == "combine" or (operator.name == "dispatch" and not runtime.source_row_fp8):
            selected.update(range(offset, offset + operator.task_num))
        offset += operator.task_num
    if not selected or any(runtime.config.all_tasks[index].task_type != TaskType.TASK_SHMEM_PUT_MEM_SIGNAL
                           or runtime.config.all_tasks[index].extra_value_0 for index in selected):
        raise ValueError("Token MTE scratch requires the complete unmarked ordinary transfer stages.")
    return selected


def _validate_token_mte_owners(config: Any, selected: set[int]) -> None:
    """Keep private vector staging exclusive to the original communication workers."""
    offsets = config.mix_task_indices[:65]
    owners = [(worker, index) for worker in range(64)
              for index in config.vector_task_indices[offsets[worker]:offsets[worker + 1]] if index in selected]
    if (len(owners) != len(selected) or {index for _, index in owners} != selected
            or any(worker % 2 == 0 for worker, _ in owners)):
        raise ValueError("Each ordinary token transfer must retain exactly one odd-worker owner.")


def with_token_mte_scratch(runtime: MXFP8Runtime, spec: MegaMoeSpec, scratch_bytes: int) -> MXFP8Runtime:
    """Select private mapped-MTE staging without changing indexed ingress or queues.

    Args:
        runtime: Completed phase graph before uploading its immutable runtime image.
        spec: Bound expert topology with owner-validated window queues.
        scratch_bytes: SDK staging size or a supported private staging size.

    Returns:
        The original SDK runtime or an independent configuration with marked ordinary token puts.
    """
    if scratch_bytes not in (16384, 32768, 65536):
        raise ValueError("Token MTE scratch must be 16384, 32768 or 65536 bytes.")
    if scratch_bytes == 16384:
        return runtime
    config = runtime.config
    if (spec.mx_vector_policy != "window2" or spec.num_cube_cores != 32 or config.num_workers != 64
            or config.task_index_num[3] != PLANNED_QUEUE_TAG):
        raise ValueError("Private token MTE scratch requires the 32-Cube window2 queues.")
    selected = _token_mte_transfer_tasks(runtime)
    _validate_token_mte_owners(config, selected)
    marked = type(config).from_buffer_copy(bytes(config))
    marker = 0x4D585500 | (1 if scratch_bytes == 32768 else 2)
    for index in selected:
        marked.all_tasks[index].extra_value_0 = marker
    return replace(runtime, config=marked)


def build_mxfp8_runtime(
    spec: MegaMoeSpec,
    *,
    forward: bool,
    swiglu_row_only: bool = False,
    expert_window: int = 0,
    dispatch_quant: bool = False,
    weight_prefetch: bool = False,
    ready_depth: int = 0,
) -> MXFP8Runtime:
    """Serialize phase-specific MX tasks with per-expert producer/consumer events.

    Args:
        spec: Validated physical expert topology and current capacity.
    """
    independent_backward = _validate_mx_options(spec, forward, swiglu_row_only, expert_window)
    _validate_mx_ready_depth(spec, ready_depth)
    kernel_gradients = _mx_gradient_return(spec, forward)
    weight_prefetch = bool(weight_prefetch and spec.replica_slots_per_rank)
    values = _MXTaskValues(
        tp=1,
        ep=spec.ep_size,
        seq_size=spec.local_num_tokens,
        all_expert_num=spec.num_experts,
        top_k=spec.top_k,
        compute_stages=4 if forward else 6,
    )
    specs, graph, gmms, quants = _buffer_specs(spec, forward), ComputeGraph(), [], []
    _add_transport(graph, specs, forward=forward, dispatch_split=spec.dispatch_split, combine_split=spec.combine_split)
    h, width, capacity, cubes = spec.hidden_size, spec.intermediate_size, spec.receive_capacity, spec.num_cube_cores
    vectors = 2 * cubes
    glist, gtable, qtable = (6, 25, 26) if forward else (19, 31, 32)
    dispatch_quant_stage = None
    fragments = spec.local_num_tokens // spec.dispatch_split
    event_base = (values.all_event_num + 3 + 2 * ATOMIC_ADD_VALUE_LEN + 15) // 16 * 16
    if dispatch_quant:
        _validate_dispatch_quant(spec, swiglu_row_only, expert_window, event_base)

    def quant(name: str, kind: int, columns: int, inputs: list[int], outputs: list[int], first: bool = False) -> None:
        """Add an activation quantization stage and its producer dependencies.

        Args:
            name: Stable stage name used by the compute graph.
            kind: Native quantization task kind.
            columns: Feature width of the quantized activation.
            inputs: Native business-buffer input positions.
            outputs: Native business-buffer output positions.
            first: Whether the stage consumes freshly dispatched activations.
        """
        nonlocal dispatch_quant_stage
        if first and dispatch_quant:
            dispatch_quant_stage = DispatchQuantStage(
                columns,
                vectors,
                fragments,
                spec.dispatch_split,
                event_base,
                spec.ep_size,
                spec.local_experts,
                14 if forward else 15,
                15 if forward else 16,
                quants,
                _fill_compute_stage,
            )
            _add_compute(
                graph,
                name,
                specs,
                inputs + [glist, qtable],
                outputs,
                dispatch_quant_stage,
                fragments * 4 + vectors,
                OpType.SWIGLU,
            )
            return
        _add_compute(
            graph,
            name,
            specs,
            inputs + [glist, qtable],
            outputs,
            _QuantStage(kind, columns, vectors, first, quants),
            vectors,
            OpType.SWIGLU,
        )

    def gmm(
        name: str, layout: str, dims: tuple[int, int, int], inputs: list[int], output: int, scales: tuple[int, int]
    ) -> None:
        """Add one grouped projection using phase-specific physical weight axes.

        Args:
            name: Stable stage name used by the compute graph.
            layout: Projection matrix layout, NN, NT or TN.
            dims: Logical M, N and K dimensions.
            inputs: Native business-buffer input positions.
            output: Native business-buffer output position.
            scales: Input and weight scale buffer positions.
        """
        guest_positions = _mx_guest_positions(spec, forward, name, layout, kernel_gradients)
        _add_compute(
            graph,
            name,
            specs,
            inputs + [glist, gtable],
            [output],
            _GmmStage(layout, *dims, *scales, cubes, gmms, **guest_positions),
            cubes,
        )

    if forward:
        x_kind, x_outputs = {False: (MX_QUANT_TASK, [16, 17, 18, 19]),
                             True: (MX_QUANT_ROW_TASK, [16, 17])}[swiglu_row_only]
        quant("quant_x", x_kind, h, [0], x_outputs, first=True)
        gmm("up", "NN", (capacity, 2 * width, h), [16, 5], 7, (17, 23))
        quant(
            "swiglu_quant",
            MX_SWIGLU_ROW_TASK if swiglu_row_only else MX_SWIGLU_TASK,
            width,
            [7],
            [8, 20] if swiglu_row_only else [8, 20, 21, 22],
        )
        gmm("down", "NN", (capacity, h, width), [8, 9], 11, (20, 24))
        order = ["dispatch", "quant_x", "up", "swiglu_quant", "down", "combine"]
    else:
        quant("quant_dy", MX_QUANT_TASK, h, [0], [20, 21, 22, 23], first=True)
        gmm("dw2", "TN", (width, h, capacity), [5, 22], 6, (24, 23))
        gmm("da", "NT", (capacity, width, h), [20, 7], 8, (21, 25))
        quant("swiglu_grad_quant", MX_SWIGLU_GRAD_TASK, 2 * width, [9, 8], [10, 26, 27, 28])
        gmm("dw1", "TN", (h, 2 * width, capacity), [17, 27], 18, (30, 28))
        gmm("dx", "NT", (capacity, h, 2 * width), [10, 11], 12, (26, 29))
        order = ["dispatch", "quant_dy", "dw2", "da", "swiglu_grad_quant", "dw1", "dx", "combine"]
    edges = list(zip(order, order[1:]))
    # Weight gradients and activation gradients only share read-only quantized inputs.
    # Combine consumes DX; every Cube worker still drains its complete gradient queue before native launch completion.
    if independent_backward and not forward:
        edges = [
            ("dispatch", "quant_dy"),
            ("quant_dy", "dw2"),
            ("quant_dy", "da"),
            ("da", "swiglu_grad_quant"),
            ("swiglu_grad_quant", "dw1"),
            ("swiglu_grad_quant", "dx"),
            ("dx", "combine"),
        ]
    for left, right in edges:
        graph.add_edge(left, right)
    graph.propagate_splits(values)
    tasks = 1 + sum(op.task_num for op in graph.topological_sort())
    required_events = values.all_event_num + 3 + 2 * ATOMIC_ADD_VALUE_LEN
    event_capacity = max(MIN_EVENT_CAPACITY, (required_events + 15) // 16 * 16)
    cfg = allocate_runtime_config(tasks, event_capacity, spec.local_experts, num_workers=2 * cubes)
    init_task_split_value(values)
    values.rank_id = spec.rank_id
    for op in graph.topological_sort():
        op.fill_config.fill(cfg, op, values)
    combine = graph.get_op("combine")
    add_terminate(cfg, values, combine.task_num)
    _revise_mx_task_queue(cfg, values, graph, order)
    reorder_mx_expert_windows(cfg, graph, spec.local_experts, expert_window, forward)
    transport_schedule = balance_all_transport(cfg, graph, spec)
    if dispatch_quant_stage is not None:
        configure_dispatch_quant_queue(cfg, graph, dispatch_quant_stage, spec.rank_id)
        transport_schedule = "dispatch_quant: source-fragment waves, no padding"
    add_dynamic_data(cfg, values, glist)
    cfg.task_num = sum(op.task_num for op in graph.topological_sort())
    cfg.atomic_add_values[0] = 1
    configure_ready_handshake(cfg, values)
    validate_runtime_config(cfg, values, cubes)
    transport_schedule = _configure_mx_vector_policy(cfg, graph, spec, forward, transport_schedule)
    runtime = _finish_mx_runtime(graph, cfg, gmms, quants, len(specs), spec,
                                swiglu_row_only, transport_schedule, kernel_gradients, weight_prefetch)
    _configure_mx_ready_depth(cfg, ready_depth)
    return runtime
