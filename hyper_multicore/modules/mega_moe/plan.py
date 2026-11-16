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
"""Build the static graph and tiling plan for managed MegaMoe execution."""

from __future__ import annotations

from copy import copy, deepcopy
import struct
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from hyper_multicore.replica.signal_transport import (
    KERNEL_GRADIENT_TRANSPORT_MODES,
)

from hyper_multicore.diagnostics import MegaKernelDiagnostics, prepare_mega_kernel_diagnostics
from hyper_multicore.modules.mega_moe.backward.gen_runtime_data import (
    build_config_for_rank as build_backward_config,
)
from hyper_multicore.modules.mega_moe.backward.graph import (
    build_backward_graph,
)
from hyper_multicore.modules.mega_moe.backward.storage import (
    can_reuse_backward_dispatch,
    join_w1_phase_once,
    prepare_w13_overlap_schedule,
    prioritize_activation_gradient,
    replica_w2_ready_events,
    replica_w13_ready_events,
)
from hyper_multicore.modules.mega_moe.backward.tiling_tables import (
    get_act_grad_tiling_bytes,
    get_gate_grad_tiling_bytes,
    get_swiglu_grad_tiling_bytes,
    get_w1_grad_tiling_bytes,
    get_w2_grad_tiling_bytes,
)
from hyper_multicore.modules.mega_moe.forward.gen_runtime_data import (
    build_config_for_rank as build_forward_config,
)
from hyper_multicore.modules.mega_moe.forward.graph import (
    build_forward_graph,
)
from hyper_multicore.modules.mega_moe.forward.tiling_tables import (
    get_down_proj_tiling_bytes,
    get_swiglu_tiling_bytes,
    get_up_proj_tiling_bytes,
)
from hyper_multicore.scheduler.config import (
    FAST_DEPENDENCY_POLL_INTERVAL_US, A5_GMM_KERNEL_TAG, TaskAiCoreType, TaskType, validate_runtime_config,
    TaskSplitValue,
)
from hyper_multicore.scheduler.runtime import serialize_runtime_config
from hyper_multicore.scheduler.graph import ComputeGraph, OperatorNode, OpType, SplitSpec, TensorSpec
from hyper_multicore.torch import ops as multicore_ops

from .diagnostics import _mega_moe_owner_options, _adapt_mega_moe_diagnostic_graph
from .probability import (
    PROBABILITY_POSITION_FORWARD, PROBABILITY_POSITION_BACKWARD,
)

from .schedule_policy import (
    MegaMoeSchedulePolicy, compact_graph,
)

from .gradient_queue import (
    ACT_FIRST_GRADIENT_PROJECTIONS,
    ACTIVE_GRADIENT_PROJECTIONS,
    ADAPTIVE_GRADIENT_QUEUE,
    has_adaptive_gradient_native,
    mark_active_gradient_lanes,
    mark_device_gradient_queue,
    require_device_gradient_native,
    reverse_w1_gradient_queue,
)
from .empty_dispatch import mark_empty_combine, mark_empty_dispatch, require_empty_dispatch_native
from .event_refresh import apply_event_refresh
from .saved_dispatch import saved_dispatch_images, resolve_saved_dispatch
from .local_gradients import resolve_local_gradient_bf16
from .final_gradients import resolve_final_gradient_bf16
from .inactive_tasks import RANK_LOCAL_GUEST_SUFFIX_SCHEDULE, GuestTaskPrefix, build_guest_task_prefixes
from .inactive_tasks import replica_runtime_schedule
from .replica_task_options import validate_replica_task_options
from .spec import MegaMoeSpec, _route_metadata_rows, _supports_indexed_backward_pair
from hyper_multicore.shmem._runtime import configured_udma_qp_count
from .bf16_gmm_tiling import retile_a5_bf16
from .shared import shared_tiling_bytes
from .shared_schedule import bind_shared_diagnostic_graph
from .tensor_dump import attach_mega_moe_l1_view_resolvers
from .udma_order import validate_udma_qp_owners


@dataclass(frozen=True)
class MegaMoePlan:
    """Rank-local schedules and tiling tensors for forward and backward."""

    spec: MegaMoeSpec
    forward_graph: ComputeGraph
    fwd_diagnostics: MegaKernelDiagnostics
    up_proj_tiling: Any
    swiglu_tiling: Any
    down_proj_tiling: Any
    backward_graph: ComputeGraph
    bwd_diagnostics: MegaKernelDiagnostics
    act_grad_tiling: Any
    gate_grad_tiling: Any
    w1_grad_tiling: Any
    w2_grad_tiling: Any
    swiglu_grad_tiling: Any
    reuse_backward_dispatch: bool = False
    weight_grad_dtype: torch.dtype | None = None

    replica_w2_events: tuple[int, ...] = ()
    replica_w13_events: tuple[int, ...] = ()
    bwd_diagnostics_no_replica: MegaKernelDiagnostics | None = None
    bwd_runtime_no_replica_reuse_dispatch: bool = False
    fwd_task_prefixes: tuple[GuestTaskPrefix, ...] = ()
    bwd_task_prefixes: tuple[GuestTaskPrefix, ...] = ()
    bwd_no_replica_task_prefixes: tuple[GuestTaskPrefix, ...] = ()
    _recorded_streams: set = field(default_factory=set, repr=False, compare=False)

    fwd_saved_dispatch_config: Any = None
    fwd_saved_task_prefixes: tuple[GuestTaskPrefix, ...] = ()
    shared_forward_runtime: Any = None
    shared_backward_runtime: Any = None
    ingress_diagnostics: MegaKernelDiagnostics | None = None
    row_credit_element_offset: int | None = None

    @property
    def fwd_runtime_config(self) -> Any:
        """Return the original ordinary forward RuntimeConfig."""
        return self.fwd_diagnostics.runtime_config

    @property
    def bwd_runtime_config(self) -> Any:
        """Return the original ordinary backward RuntimeConfig."""
        return self.bwd_diagnostics.runtime_config


    def record_stream(self, runtime_config: Any, *, stream: Any = None) -> None:
        """Register immutable plan storage once per stream, preserving allocator lifetime."""
        if not isinstance(runtime_config, torch.Tensor) or runtime_config.device.type != "npu":
            return
        if stream is None:
            stream = torch.npu.current_stream(runtime_config.device)
        key = (runtime_config.device, stream.npu_stream)
        if key not in self._recorded_streams:
            tensors = [value for value in vars(self).values() if isinstance(value, torch.Tensor)]
            tensors.extend((self.fwd_runtime_config, self.bwd_runtime_config))
            for tensor in tensors:
                if isinstance(tensor, torch.Tensor) and tensor.device.type == "npu":
                    tensor.record_stream(stream)
            self._recorded_streams.add(key)
        if runtime_config is not self.fwd_runtime_config and runtime_config is not self.bwd_runtime_config:
            # Diagnostic buffers are per invocation and cannot share the ordinary cache.
            runtime_config.record_stream(stream)


# Keep the existing free-function call sites without another Python wrapper call.
record_plan_stream = MegaMoePlan.record_stream




def _tensor_from_bytes(data: bytes, device: Any) -> torch.Tensor:
    """Copy serialized runtime data into a device uint8 tensor."""
    array = np.frombuffer(bytearray(data), dtype=np.uint8).copy()
    return torch.from_numpy(array).to(device=device, dtype=torch.uint8)


def _gmm_tensor_from_bytes(data: bytes, spec: MegaMoeSpec, device: Any) -> torch.Tensor:
    """Apply the qualified 32-Cube geometry without changing A3 tilings."""
    if spec.num_cube_cores == 32:
        data = retile_a5_bf16(data, spec.num_cube_cores)
    return _tensor_from_bytes(data, device)


def _resize_swiglu_tiling(data: bytes, num_cube_cores: int) -> bytes:
    """Resize the baseline 24-core repeated struct for the active workers."""
    baseline_workers = 49
    if len(data) % baseline_workers:
        raise RuntimeError("invalid baseline SwiGLU tiling byte length.")
    struct_size = len(data) // baseline_workers
    return data[:struct_size] * (2 * num_cube_cores + 1)


def _build_task_values(spec: MegaMoeSpec) -> TaskSplitValue:
    """Translate rank-local token shape into legacy scheduler coordinates.

    Args:
        spec: Validated local-token and expert topology specification.

    Returns:
        Scheduler values whose single partition is the rank-local input.
    """
    return TaskSplitValue(
        tp=1,
        ep=spec.ep_size,
        seq_size=spec.plan_tokens,
        all_expert_num=spec.num_experts,
        top_k=spec.top_k,
        dispatch_mode=spec.dispatch_mode,
        communication_rows=spec._communication_rows,
    )


def _resolve_a5_execution_spec(spec: MegaMoeSpec) -> MegaMoeSpec:
    """Apply the shared-forward numerical policy to the actual plan rows."""
    if (spec.num_cube_cores == 32 and spec.shared_intermediate_size
            and spec.plan_tokens < 512 and spec.forward_shared_expert):
        return replace(spec, forward_shared_expert=False)
    return spec


def _apply_gradient_order(spec, task_values, backward_data, backward_graph):
    """Keep original queue geometry when no explicit gradient policy is selected."""
    adaptive = spec._gradient_queue_order == ADAPTIVE_GRADIENT_QUEUE
    join_once = adaptive or spec._gradient_queue_order in (
        "w1_join_once",
        "device_w1_asc_join_once",
        "device_w1_asc_join_rotate",
    )
    if join_once and (not spec.replica_slots_per_rank):
        if not join_w1_phase_once(backward_data, spec.local_experts, spec.num_cube_cores, apply=not adaptive):
            raise ValueError("W1 phase join requires the canonical owner-only backward queue")
    if spec._gradient_queue_order.startswith("device_") and (not spec.replica_slots_per_rank):
        mark_device_gradient_queue(backward_data, backward_graph, spec._gradient_queue_order)
    if spec._gradient_queue_order == "w1_reverse" and (not spec.replica_slots_per_rank):
        reverse_w1_gradient_queue(backward_data, backward_graph, task_values)


def _apply_execution_policies(spec, task_values, forward_data, backward_data, backward_graph, overlap_w13):
    """Apply validated execution options after constructing the ordinary immutable graph."""
    _apply_gradient_order(spec, task_values, backward_data, backward_graph)
    if overlap_w13 and spec.replica_slots_per_rank and (spec.replica_transport in KERNEL_GRADIENT_TRANSPORT_MODES):
        prepare_w13_overlap_schedule(
            backward_data, spec.local_experts, spec.num_cube_cores, replica_slots=spec.replica_slots_per_rank
        )
    if spec._cube_event_refresh_us or spec._vector_event_refresh_us:
        for runtime in (forward_data, backward_data):
            apply_event_refresh(runtime, spec._cube_event_refresh_us, spec._vector_event_refresh_us)
    if spec._batch_empty_dispatch and spec.dispatch_mode == "push" and (not spec.replica_slots_per_rank):
        mark_empty_dispatch(forward_data)
        mark_empty_dispatch(backward_data)
    if spec._batch_empty_combine and spec.dispatch_mode == "push":
        options = {
            "ep_size": spec.ep_size,
            "local_experts": spec.local_experts,
            "home_experts": spec.local_experts - spec.replica_slots_per_rank,
        }
        mark_empty_combine(forward_data, size_position=15, **options)
        mark_empty_combine(backward_data, size_position=16, **options)



def _build_backward_runtime_graph(spec, task_values, weight_grad_dtype, compact):
    """Lower backward operands and numerical widths before runtime queue construction."""
    backward_graph = build_backward_graph(
        task_values,
        dispatch_sv=spec.dispatch_split,
        swiglu_sv=spec.swiglu_split,
        combine_sv=spec.combine_split,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_cube_cores=spec.num_cube_cores,
        swiglu_limit=spec.swiglu_limit,
        clamp_after_activation=spec.clamp_after_activation,
        apply_probs_before_fc2=spec.apply_probs_before_fc2,
        expert_weight_layout=spec.expert_weight_layout,
        shared_intermediate_size=spec.shared_intermediate_size if spec.backward_shared_expert else 0,
        indexed_backward=spec.backward_ingress_mode == "indexed",
    )
    if spec.replica_slots_per_rank:
        for operation in backward_graph.topological_sort():
            for tensor_spec in (*operation.inputs, *operation.outputs):
                if tensor_spec.param_position in (6, 18):
                    tensor_spec.dtype_size = 4
    if weight_grad_dtype == torch.float32:
        for name in ("w1_grad", "w2_grad"):
            backward_graph.get_op(name).outputs[0].dtype_size = 4
    if compact:
        compact_graph(backward_graph, task_values, 2 * spec.num_cube_cores, spec.nominal_schedule)
    backward_graph.propagate_splits(task_values)
    attach_mega_moe_l1_view_resolvers(backward_graph)
    return backward_graph


def _compact_schedule_requested(spec: MegaMoeSpec) -> bool:
    """Retain current owner compaction and explicitly qualified physical schedules."""
    return spec._compact_schedule or (
        not spec.replica_slots_per_rank and not spec._fuse_saved_dispatch
        and spec.dispatch_mode == "push" and spec.intermediate_size < 2048
        and spec._gradient_queue_order == "original" and not (spec._batch_empty_dispatch or spec._batch_empty_combine))


def _build_runtime_artifacts(
    spec: MegaMoeSpec, weight_grad_dtype: torch.dtype | None = None, *, overlap_w13: bool = True,
) -> tuple[Any, Any, Any, Any]:
    """Build forward/backward graphs and their serialized RuntimeConfig objects."""
    if spec.batch_experts and spec.replica_slots_per_rank:
        raise ValueError("Batched expert tasks require owner-only execution")
    task_values = _build_task_values(spec)
    forward_graph = build_forward_graph(
        task_values,
        dispatch_sv=spec.dispatch_split,
        swiglu_sv=spec.swiglu_split,
        combine_sv=spec.combine_split,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_cube_cores=spec.num_cube_cores,
        swiglu_limit=spec.swiglu_limit,
        clamp_after_activation=spec.clamp_after_activation,
        apply_probs_before_fc2=spec.apply_probs_before_fc2,
        expert_weight_layout=spec.expert_weight_layout,
        shared_intermediate_size=spec.shared_intermediate_size if spec.forward_shared_expert else 0,
    )
    compact_requested = (_compact_schedule_requested(spec) and not spec.shared_intermediate_size
                         and spec.forward_ingress_mode == "legacy" and spec.backward_ingress_mode == "legacy")
    compact = compact_requested and compact_graph(
        forward_graph, task_values, 2 * spec.num_cube_cores, spec.nominal_schedule)
    forward_graph.propagate_splits(task_values)
    attach_mega_moe_l1_view_resolvers(forward_graph)
    forward_data = build_forward_config(
        forward_graph,
        task_values,
        spec.rank_id,
        spec.num_cube_cores,
        shared_placement=spec.shared_placement,
    )
    if spec.forward_ingress_mode != "legacy":
        if spec.num_cube_cores != 32 or spec.dispatch_mode != "pull" or spec.max_local_num_tokens is not None:
            raise ValueError("Indexed ingress requires the static Ascend 950 pull plan")
        for task in forward_data.all_tasks[:forward_data.task_num]:
            if task.task_type == TaskType.TASK_SHMEM_GET_MEM:
                task.extra_value_0 = 16
                task.extra_value_1 = 3 if spec.forward_ingress_mode == "peer_dedup" else 1
                task.inputs[1].dim[3] = spec.top_k
            elif task.task_type == TaskType.TASK_GROUPED_MATMUL and task.inputs[0].tensor_type != 2:
                # Blaze does not use the legacy GMM workspace tensor at port 16.
                task.extra_value_2 = A5_GMM_KERNEL_TAG | 2
    if spec.forward_ingress_mode == "peer_overlap":
        _configure_owner_prefix(forward_data)
    elif spec.forward_ingress_mode in ("peer_token", "peer_tile"):
        _configure_token_prefix(forward_graph, forward_data, spec)
        if spec.forward_ingress_mode == "peer_tile":
            for task in forward_data.all_tasks[:forward_data.task_num]:
                if task.task_type == TaskType.TASK_GROUPED_MATMUL and task.outputs[0].input_position == 7:
                    task.dependent_event = 0xFFFFFFFF
                    task.extra_value_0 = spec.top_k
                    task.extra_value_2 = 0x4D54494C
    if spec.forward_ingress_mode != "legacy":
        validate_runtime_config(forward_data, task_values, spec.num_cube_cores)
        _attach_route_metadata_input(forward_graph, spec)
    task_values.shared_expert = False
    backward_graph = _build_backward_runtime_graph(spec, task_values, weight_grad_dtype, compact)
    backward_data = build_backward_config(
        backward_graph,
        task_values,
        spec.rank_id,
        spec.num_cube_cores,
        shared_placement=spec.shared_placement,
    )
    policy = MegaMoeSchedulePolicy(
        vector_ready_queue=bool(compact),
        pipeline_vector=bool(compact) and not spec.batch_experts,
        pipeline_cube=bool(compact) and not spec.batch_experts,
        batch_experts=spec.batch_experts,
        dependency_poll_interval=FAST_DEPENDENCY_POLL_INTERVAL_US,
    )
    policy.apply(forward_data, forward_graph, task_values, spec.num_cube_cores)
    # Guest W13 producers require every dX task to precede weight gradients.
    replace(policy, pipeline_cube=policy.pipeline_cube and not spec.replica_slots_per_rank).apply(
        backward_data, backward_graph, task_values, spec.num_cube_cores)
    if compact and not spec.batch_experts:
        forward_data.cube_single_group = 1
    _apply_execution_policies(spec, task_values, forward_data, backward_data, backward_graph, overlap_w13)
    if spec.replica_slots_per_rank:
        backward_data.task_validator = validate_replica_task_options
    if spec.backward_ingress_mode == "indexed":
        if spec.num_cube_cores != 32 or not _supports_indexed_backward_pair(
                spec.forward_ingress_mode, spec.dispatch_mode, spec.replica_slots_per_rank):
            raise ValueError("Indexed backward requires an Ascend 950 indexed or B0 peer pull plan")
        for task in backward_data.all_tasks[:backward_data.task_num]:
            if task.task_type == TaskType.TASK_SHMEM_GET_MEM:
                task.extra_value_0, task.extra_value_1 = 25, 8
                task.inputs[1].dim[3] = spec.top_k
            elif task.task_type == TaskType.TASK_GROUPED_MATMUL and task.inputs[0].tensor_type != 2:
                task.extra_value_2 = A5_GMM_KERNEL_TAG | 3
        validate_runtime_config(backward_data, task_values, spec.num_cube_cores)
    if spec.shared_intermediate_size:
        clamp = struct.unpack("<I", struct.pack("<f", spec.swiglu_limit or 0.0))[0]
        for config in (forward_data, backward_data):
            for task in config.all_tasks[:config.task_num]:
                if task.inputs[0].tensor_type == 2 and task.task_type in (TaskType.TASK_SWI_GLU,
                                                                       TaskType.TASK_SWI_GLU_GRAD):
                    task.extra_value_0 = clamp
    if spec.num_cube_cores == 32:
        for direction, config in enumerate((forward_data, backward_data)):
            for task in config.all_tasks[:config.task_num]:
                if (task.task_type == TaskType.TASK_GROUPED_MATMUL and task.inputs[0].tensor_type != 2
                        and task.extra_value_2 != 0x4D54494C):
                    task.extra_value_2 = A5_GMM_KERNEL_TAG | (2 if direction == 0 else 3)
                elif direction == 1 and task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
                    task.extra_value_1 |= 1
    if spec.num_cube_cores == 32:
        qp_count = configured_udma_qp_count()
        if qp_count not in (16, 32):
            raise ValueError("BF16 MegaMoE requires 16 or 32 UDMA QPs")
        if qp_count != 32:
            for config in (forward_data, backward_data):
                validate_udma_qp_owners(config, ep_size=spec.ep_size, num_experts=spec.num_experts,
                                        rank_id=spec.rank_id, qp_count=qp_count)
    return forward_graph, forward_data, backward_graph, backward_data


def _resolve_compact_schedule(spec, device, automatic):
    """Retain explicit research schedules and older adapters on their existing path."""
    compatible = (automatic and spec.dispatch_mode == "push" and spec.expert_weight_layout == "EIO"
                  and spec._gradient_queue_order in ("original", "device_active_w1_lanes")
                  and spec.intermediate_size < 2048
                  and not (spec._batch_empty_dispatch or spec._batch_empty_combine))
    if not compatible or torch.device(device).type != "npu":
        return spec
    multicore_ops._load_native()  # pylint: disable=protected-access
    version = getattr(torch.ops.hyper_multicore, "mega_moe_compact_version", None)
    return replace(spec, _compact_schedule=version is not None and version() >= 1)



def _prepare_diagnostics(
    spec: MegaMoeSpec,
    forward_graph: ComputeGraph,
    forward_config: Any,
    backward_graph: ComputeGraph,
    backward_config: Any,
    device: Any,
) -> tuple[MegaKernelDiagnostics, MegaKernelDiagnostics]:
    """Bind forward/backward plans to the public diagnostics integration API."""
    topology = _build_task_values(spec)
    topology.rank_id = spec.rank_id
    options = {
        "kernel_name": "mega_moe",
        "device": device,
        "rank": dist.get_rank() if dist.is_initialized() else spec.rank_id,
        **_mega_moe_owner_options(
            topology, num_cube_cores=spec.num_cube_cores, compact_queues=bool(forward_config.vector_ready_queue)
        ),
    }
    if spec.shared_intermediate_size and (spec.apply_probs_before_fc2 or spec.batch_experts):
        def adapt_combined(graph, config):
            return _adapt_mega_moe_diagnostic_graph(bind_shared_diagnostic_graph(graph, config), config)
        options["graph_adapter"] = adapt_combined
    elif spec.shared_intermediate_size:
        options["graph_adapter"] = bind_shared_diagnostic_graph
    elif spec.apply_probs_before_fc2 or spec.batch_experts:
        options["graph_adapter"] = _adapt_mega_moe_diagnostic_graph
    copies = (spec.ep_size - 1) * spec.replica_slots_per_rank
    groups = min(4, spec.num_cube_cores)
    bands_per_worker = (64 + groups - 1) // groups
    gradient_records = (
        2 * copies * (5 * bands_per_worker + 3) + 3
        if copies and spec.replica_transport in KERNEL_GRADIENT_TRANSPORT_MODES
        else 0
    )
    producer_records = 2 * spec.local_experts * 64 if gradient_records else 0
    weight_records = 2 * spec.replica_slots_per_rank
    return (
        prepare_mega_kernel_diagnostics(
            forward_graph, forward_config,
            parameter_count=max(PROBABILITY_POSITION_FORWARD + 1 if spec.apply_probs_before_fc2 else 16,
                                17 if spec.forward_ingress_mode != "legacy" else 16,
                                28 if spec.shared_intermediate_size and spec.forward_shared_expert else 16),
            extra_aic_records=weight_records, **options
        ),
        prepare_mega_kernel_diagnostics(
            backward_graph,
            backward_config,
            parameter_count=max(PROBABILITY_POSITION_BACKWARD + 1 if spec.apply_probs_before_fc2 else 20,
                                26 if spec.backward_ingress_mode == "indexed" else 20,
                                35 if spec.shared_intermediate_size and spec.backward_shared_expert else 20),
            extra_aiv_records=gradient_records,
            extra_aic_records=producer_records + weight_records,
            **options,
        ),
    )


def _soc_name(device: torch.device) -> str:
    """Read the actual device family without importing calibration policy."""
    return str(torch.npu.get_device_name(device)) if device.type == "npu" else ""


def _resolve_gradient_queue(spec: MegaMoeSpec, device: Any) -> MegaMoeSpec:
    """Resolve automatic scheduling using only the runtime ABI and projection."""
    order = spec._gradient_queue_order
    if order in ("auto", "auto_w1"):
        abi = 4 if spec.replica_slots_per_rank else 3
        supported = torch.device(device).type == "npu" and has_adaptive_gradient_native(abi)
        if order == "auto":
            candidate = "device_active_lanes" if spec.replica_slots_per_rank else "device_active_lanes_act_first"
        else:
            candidate = "device_active_w1_lanes"
        return replace(spec, _gradient_queue_order=candidate if supported else "original")
    if order in ACTIVE_GRADIENT_PROJECTIONS:
        require_device_gradient_native(order, physical=bool(spec.replica_slots_per_rank))
        return spec
    if order.startswith("device_") and not spec.replica_slots_per_rank:
        require_device_gradient_native(order)
    return spec


def _qualify_gradient_order(spec, baseline, backward_config, projections, automatic):
    """Prove complete gradient waves and receive-reader completion before reordering."""
    act_first = spec._gradient_queue_order in ACT_FIRST_GRADIENT_PROJECTIONS
    if projections and (
        not mark_active_gradient_lanes(
            backward_config, spec.local_experts, spec.num_cube_cores, projections=projections
        )
    ):
        if act_first and (not automatic):
            raise ValueError("ActGrad priority requires complete gradient tile waves")
        act_first = False
        spec = baseline
    if act_first and (
        spec.replica_slots_per_rank
        or not prioritize_activation_gradient(backward_config, spec.local_experts, spec.num_cube_cores)
    ):
        if not automatic:
            raise ValueError("ActGrad priority requires a canonical owner-only backward graph")
        act_first = False
        spec = replace(spec, _gradient_queue_order="device_active_lanes")
    original_reuse = can_reuse_backward_dispatch(
        backward_config, spec.local_experts, spec.num_cube_cores, act_first=act_first
    )
    if act_first and (not original_reuse):
        raise ValueError("ActGrad priority failed to prove dispatch buffer reuse")
    return (spec, act_first, original_reuse)


def _physical_prefix_options(spec):
    """Keep one independent preparation stage outside the execution orchestration."""
    prefix_options = {
        "home_experts": spec.local_experts - spec.replica_slots_per_rank,
        "ep_size": spec.ep_size,
        "dispatch_mode": spec.dispatch_mode,
        "rank_local_communication": spec.replica_slots_per_rank > 0
        and replica_runtime_schedule() == RANK_LOCAL_GUEST_SUFFIX_SCHEDULE,
    }
    return prefix_options


def _saved_dispatch_resources(spec, config, device):
    """Keep normal diagnostics and fused buffer roles in independent images."""
    if not spec._fuse_saved_dispatch:
        return config, {}
    original, saved = saved_dispatch_images(config, spec)
    return original, {
        "fwd_saved_dispatch_config": _tensor_from_bytes(serialize_runtime_config(saved), device),
        "fwd_saved_task_prefixes": (
            build_guest_task_prefixes(saved, **_physical_prefix_options(spec)) if spec.replica_slots_per_rank else ()
        ),
    }


def _plan_gmm_options(spec: MegaMoeSpec, forward_config: Any) -> tuple[dict, dict]:
    """Keep forward single-group tiling separate from backward expert indexing."""
    gmm_options = {
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "num_groups": 1 if forward_config.cube_single_group else spec.local_experts,
        "num_cube_cores": spec.num_cube_cores,
    }
    backward_gmm_options = gmm_options.copy()
    backward_gmm_options["num_groups"] = spec.local_experts
    return gmm_options, backward_gmm_options


def _configure_owner_prefix(config: Any) -> None:
    """Put every independent owner GET before every compute-dependent vector task."""
    queued = [int(config.vector_task_indices[i]) for i in range(config.task_index_num[1])]
    owners = [i for i in queued if config.all_tasks[i].task_type == TaskType.TASK_SHMEM_GET_MEM]
    owner_set = set(owners)
    for index in owners:
        task = config.all_tasks[index]
        task.task_type = TaskType.TASK_MOE_OWNER_GET
        task.extra_value_1 = 4
        task.dependent_event = 0xFFFFFFFF
    for position, index in enumerate(owners + [i for i in queued if i not in owner_set]):
        config.vector_task_indices[position] = index


def _attach_route_metadata_input(graph: ComputeGraph, spec: MegaMoeSpec) -> None:
    """Expose the complete port-16 binding without adding a Device task."""
    rows = _route_metadata_rows(spec.routed_slots, spec.num_experts, spec.ep_size, spec.forward_ingress_mode)
    metadata = TensorSpec("route_metadata_workspace", [rows, 16], param_position=16, dtype_size=1)
    node = OperatorNode(
        name="route_metadata", op_type=OpType.AIV_PIPELINE, inputs=[metadata], outputs=[],
        param_positions=[16], split_value=1,
        split_spec=SplitSpec(split_inputs=None, task_num_fn=lambda _: 0),
        tiling_position=-1, fill_config=None,
    )
    graph.add_op(node)
    if not (spec.shared_intermediate_size and spec.forward_shared_expert):
        graph.add_edge(node, graph.get_op("dispatch"))
    # A new predecessor would reorder the independent shared branch after task
    # IDs have been assigned. This zero-task node only describes a Host binding.
    source = graph.get_op("dispatch").inputs[1]
    source.shape = [spec.local_num_tokens, spec.hidden_size]


def _configure_token_prefix(graph: ComputeGraph, config: Any, spec: MegaMoeSpec) -> None:
    """Compact canonical descriptors while preserving every math and Combine queue."""
    dispatch = graph.get_op("dispatch")
    old_dispatch_count = dispatch.task_num
    token_jobs = spec.ep_size * (spec.local_num_tokens // spec.dispatch_split)
    if token_jobs > old_dispatch_count or token_jobs == 0:
        raise RuntimeError("Token dispatch exceeds the canonical descriptor reservation")
    if any(bytes(event).strip(b"\0") for event in config.all_events):
        raise RuntimeError("Token compaction cannot remap auxiliary EventDesc task references")
    original = [type(task).from_buffer_copy(bytes(task)) for task in config.all_tasks[:config.task_num + 1]]
    queues = [list(indices[:count]) for indices, count in zip(
        (config.cube_task_indices, config.vector_task_indices, config.mix_task_indices), config.task_index_num[:3])]
    event_base = original[0].trigger_event
    for index in range(token_jobs):
        task = original[index]
        task.task_type = TaskType.TASK_MOE_TOKEN_GET
        task.task_aicore_type = TaskAiCoreType.TASK_AICORE_VECTOR
        task.task_index = index
        task.task_split_num = token_jobs
        task.task_split_value = spec.dispatch_split
        task.trigger_event = event_base
        task.dependent_event = 0xFFFFFFFF
        task.extra_value_1 = 6 if spec.forward_ingress_mode == "peer_tile" else 5
        config.all_tasks[index] = task
    for index, task in enumerate(original[old_dispatch_count:], start=token_jobs):
        config.all_tasks[index] = task
    shift = old_dispatch_count - token_jobs
    for kind, (indices, queued) in enumerate(zip(
            (config.cube_task_indices, config.vector_task_indices, config.mix_task_indices), queues)):
        retained = [int(index) - shift for index in queued if index >= old_dispatch_count]
        if kind == 1:
            retained = list(range(token_jobs)) + retained
        config.task_index_num[kind] = len(retained)
        for position, index in enumerate(retained):
            indices[position] = index
    config.task_num -= shift
    dispatch.task_num = token_jobs
    dispatch.diagnostic_name = "DispatchToken"
    dispatch.split_spec = replace(dispatch.split_spec, task_num_fn=lambda topology: (
        topology.ep * (topology.per_expert_seq_to_other // spec.dispatch_split)))


def _ingress_graph(forward_graph: ComputeGraph) -> ComputeGraph:
    """Describe only the tasks actually scheduled by the independent prepass."""
    graph = ComputeGraph()
    metadata = copy(forward_graph.get_op("route_metadata"))
    metadata.predecessors = []
    metadata.successors = []
    graph.add_op(metadata)
    dispatch = copy(forward_graph.get_op("dispatch"))
    dispatch.predecessors = []
    dispatch.successors = []
    dispatch.diagnostic_name = "DispatchDedup"
    graph.add_op(dispatch).add_edge(metadata, dispatch)
    return graph


def _row_credit_element_offset(config: Any, spec: MegaMoeSpec) -> int | None:
    """Resolve the device INT32 threshold slice from actual Up-projection dependencies."""
    if spec.forward_ingress_mode not in ("peer_overlap", "peer_token"):
        return None
    start = int(config.all_tasks[0].trigger_event)
    for task in config.all_tasks[:config.task_num]:
        if task.task_type == TaskType.TASK_GROUPED_MATMUL and task.outputs[0].input_position == 7:
            expert = task.task_index // spec.num_cube_cores
            if task.dependent_event != start + expert:
                raise RuntimeError("Owner row-credit events do not match the Up projection")
    return type(config).all_event_num_triggers.offset // 4 + start


def _build_ingress_config(forward_config: Any) -> Any:
    """Queue only independent owner GETs; rely on outer all-worker/peer completion."""
    config = type(forward_config).from_buffer_copy(bytes(forward_config))
    task_ids = [int(config.vector_task_indices[index]) for index in range(config.task_index_num[1])]
    gets = [index for index in task_ids if config.all_tasks[index].task_type == TaskType.TASK_SHMEM_GET_MEM]
    config.task_num = len(gets)
    config.task_index_num[0] = 0
    config.task_index_num[1] = len(gets)
    config.all_event_num_triggers[0] = len(gets)
    for position, index in enumerate(gets):
        config.vector_task_indices[position] = index
        task = config.all_tasks[index]
        task.dependent_event = 0xFFFFFFFF
        # Dummy 0 is never waited on. No per-expert credit is published here.
        task.trigger_event = 0
        task.extra_value_1 = 2
    return config


def build_mega_moe_plan(
    spec: MegaMoeSpec, device: Any, *, weight_grad_dtype: torch.dtype | None = None,
) -> MegaMoePlan:
    """Build trimmed dense forward/backward runtime images without fusion slots.

    Args:
        spec: Validated local-token and expert topology specification.
        device: NPU device receiving serialized descriptors and tiling tensors.
        weight_grad_dtype: Keep partial expert weight gradients in FP32.

    Returns:
        Rank-local forward and backward runtime resources.
    """
    spec = _resolve_a5_execution_spec(spec)
    if spec.apply_probs_before_fc2 and (spec.forward_ingress_mode != "legacy" or spec.backward_ingress_mode == "indexed"
                                         or spec.shared_intermediate_size):
        raise ValueError("Pre-FC2 probability input is not yet qualified with indexed routing or shared experts")
    if weight_grad_dtype not in (None, torch.float32):
        raise ValueError("Partial expert gradients support only the native dtype or FP32")
    if spec._batch_empty_combine and (not _soc_name(torch.device(device)).startswith("Ascend910_93")):
        raise ValueError("Empty Combine credit batching currently requires Ascend910C")
    if spec._batch_empty_dispatch or spec._batch_empty_combine:
        require_empty_dispatch_native()
    spec = replace(spec, _final_gradient_bf16=resolve_final_gradient_bf16(spec._final_gradient_bf16))
    spec = replace(spec, _local_gradient_bf16=resolve_local_gradient_bf16(
        True if spec._final_gradient_bf16 else spec._local_gradient_bf16))
    automatic = spec._gradient_queue_order in ("auto", "auto_w1")
    spec = _resolve_gradient_queue(spec, device)
    spec = _resolve_compact_schedule(spec, device, automatic)
    spec = resolve_saved_dispatch(spec, device)
    projections = ACTIVE_GRADIENT_PROJECTIONS.get(spec._gradient_queue_order)
    baseline = replace(spec, _gradient_queue_order="original") if projections else spec
    forward_graph, forward_config, backward_graph, backward_config = _build_runtime_artifacts(
        baseline, weight_grad_dtype, overlap_w13=False
    )
    spec, act_first, original_reuse = _qualify_gradient_order(spec, baseline, backward_config, projections, automatic)
    original_backward_config = backward_config
    backward_config, fallback = _prepare_overlap_return(
        spec, forward_graph, forward_config, backward_graph, backward_config, device
    )
    forward_config, saved_options = _saved_dispatch_resources(spec, forward_config, device)
    fwd_diagnostics, bwd_diagnostics = _prepare_diagnostics(
        spec, forward_graph, forward_config, backward_graph, backward_config, device
    )
    ingress_diagnostics = None
    if spec.forward_ingress_mode == "peer_dedup":
        topology = _build_task_values(spec)
        topology.rank_id = spec.rank_id
        ingress_diagnostics = prepare_mega_kernel_diagnostics(
            _ingress_graph(forward_graph), _build_ingress_config(forward_config), parameter_count=17,
            kernel_name="mega_moe_ingress", device=device,
            rank=dist.get_rank() if dist.is_initialized() else spec.rank_id,
            **_mega_moe_owner_options(topology, num_cube_cores=spec.num_cube_cores),
        )
    gmm_options, backward_gmm_options = _plan_gmm_options(spec, forward_config)
    prefix_options = _physical_prefix_options(spec)
    return MegaMoePlan(
        spec=spec,
        weight_grad_dtype=weight_grad_dtype,
        ingress_diagnostics=ingress_diagnostics,
        shared_forward_runtime=(_tensor_from_bytes(shared_tiling_bytes(spec, backward=False), device)
                                if spec.shared_intermediate_size and spec.forward_shared_expert else None),
        shared_backward_runtime=(_tensor_from_bytes(shared_tiling_bytes(spec, backward=True), device)
                                 if spec.shared_intermediate_size and spec.backward_shared_expert else None),
        row_credit_element_offset=_row_credit_element_offset(forward_config, spec),
        **saved_options,
        fwd_task_prefixes=(
            build_guest_task_prefixes(forward_config, **prefix_options) if spec.replica_slots_per_rank else ()
        ),
        bwd_task_prefixes=(
            build_guest_task_prefixes(backward_config, **prefix_options) if spec.replica_slots_per_rank else ()
        ),
        bwd_no_replica_task_prefixes=(
            build_guest_task_prefixes(original_backward_config, **prefix_options) if fallback is not None else ()
        ),
        replica_w13_events=replica_w13_ready_events(backward_config, spec.local_experts, spec.num_cube_cores),
        replica_w2_events=replica_w2_ready_events(backward_config, spec.local_experts, spec.num_cube_cores),
        reuse_backward_dispatch=can_reuse_backward_dispatch(
            backward_config, spec.local_experts, spec.num_cube_cores, act_first=act_first
        ),
        bwd_runtime_no_replica_reuse_dispatch=original_reuse if fallback else False,
        forward_graph=forward_graph,
        fwd_diagnostics=fwd_diagnostics,
        up_proj_tiling=_gmm_tensor_from_bytes(
            get_up_proj_tiling_bytes(forward_graph.get_op("up_proj").split_value, **gmm_options), spec, device
        ),
        swiglu_tiling=_tensor_from_bytes(
            _resize_swiglu_tiling(
                get_swiglu_tiling_bytes(
                    forward_graph.get_op("swiglu").split_value, intermediate_size=spec.intermediate_size
                ),
                spec.num_cube_cores,
            ),
            device,
        ),
        down_proj_tiling=_gmm_tensor_from_bytes(
            get_down_proj_tiling_bytes(forward_graph.get_op("down_proj").split_value, **gmm_options),
            spec, device,
        ),
        backward_graph=backward_graph,
        bwd_diagnostics=bwd_diagnostics,
        bwd_diagnostics_no_replica=fallback,
        act_grad_tiling=_gmm_tensor_from_bytes(
            get_act_grad_tiling_bytes(backward_graph.get_op("act_grad").split_value, **backward_gmm_options),
            spec, device,
        ),
        gate_grad_tiling=_gmm_tensor_from_bytes(
            get_gate_grad_tiling_bytes(backward_graph.get_op("gate_grad").split_value, **backward_gmm_options),
            spec, device,
        ),
        w1_grad_tiling=_gmm_tensor_from_bytes(
            get_w1_grad_tiling_bytes(backward_graph.get_op("w1_grad").split_value,
                                    expert_weight_layout=spec.expert_weight_layout, **backward_gmm_options),
            spec, device,
        ),
        w2_grad_tiling=_gmm_tensor_from_bytes(
            get_w2_grad_tiling_bytes(backward_graph.get_op("w2_grad").split_value,
                                    expert_weight_layout=spec.expert_weight_layout, **backward_gmm_options),
            spec, device,
        ),
        swiglu_grad_tiling=_tensor_from_bytes(
            _resize_swiglu_tiling(
                get_swiglu_grad_tiling_bytes(
                    backward_graph.get_op("swiglu_grad").split_value, intermediate_size=spec.intermediate_size
                ),
                spec.num_cube_cores,
            ),
            device,
        ),
    )


def _prepare_overlap_return(spec, forward_graph, forward_config, backward_graph, backward_config, device):
    """Keep an original diagnostic fallback unless both gradient event sets agree."""
    fallback = None
    if spec.replica_slots_per_rank and spec.replica_transport in KERNEL_GRADIENT_TRANSPORT_MODES:
        overlap_config = deepcopy(backward_config)
        if prepare_w13_overlap_schedule(
            overlap_config, spec.local_experts, spec.num_cube_cores
        ) and replica_w2_ready_events(
            overlap_config, spec.local_experts, spec.num_cube_cores
        ) == replica_w2_ready_events(
            backward_config, spec.local_experts, spec.num_cube_cores
        ):
            _, fallback = _prepare_diagnostics(
                spec, forward_graph, forward_config, backward_graph, backward_config, device
            )
            backward_config = overlap_config
    return backward_config, fallback
