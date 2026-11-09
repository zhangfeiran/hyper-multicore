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

from copy import deepcopy
from dataclasses import dataclass, replace
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
from hyper_multicore.scheduler.config import TaskSplitValue
from hyper_multicore.scheduler.graph import ComputeGraph

from .diagnostics import _mega_moe_owner_options
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
from .inactive_tasks import RANK_LOCAL_GUEST_SUFFIX_SCHEDULE, GuestTaskPrefix, build_guest_task_prefixes
from .inactive_tasks import replica_runtime_schedule
from .spec import MegaMoeSpec
from .tensor_dump import attach_mega_moe_l1_view_resolvers


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
    replica_w2_events: tuple[int, ...] = ()
    replica_w13_events: tuple[int, ...] = ()
    bwd_diagnostics_no_replica: MegaKernelDiagnostics | None = None
    bwd_runtime_no_replica_reuse_dispatch: bool = False
    fwd_task_prefixes: tuple[GuestTaskPrefix, ...] = ()
    bwd_task_prefixes: tuple[GuestTaskPrefix, ...] = ()
    bwd_no_replica_task_prefixes: tuple[GuestTaskPrefix, ...] = ()

    @property
    def fwd_runtime_config(self) -> Any:
        """Return the original ordinary forward RuntimeConfig."""
        return self.fwd_diagnostics.runtime_config

    @property
    def bwd_runtime_config(self) -> Any:
        """Return the original ordinary backward RuntimeConfig."""
        return self.bwd_diagnostics.runtime_config


def record_plan_stream(plan: MegaMoePlan, runtime_config: Any) -> None:
    """Protect cached tiling and runtime storage from cross-stream eviction."""
    tensors = [value for value in vars(plan).values() if isinstance(value, torch.Tensor)]
    tensors.append(runtime_config)
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.device.type == "npu":
            tensor.record_stream(torch.npu.current_stream(tensor.device))


def _tensor_from_bytes(data: bytes, device: Any) -> torch.Tensor:
    """Copy serialized runtime data into a device uint8 tensor."""
    array = np.frombuffer(bytearray(data), dtype=np.uint8).copy()
    return torch.from_numpy(array).to(device=device, dtype=torch.uint8)


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
        expert_rows_bound=spec.schedule_expert_rows,
        source_expert_rows_bound=spec.schedule_source_expert_rows,
    )


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


def _build_runtime_artifacts(spec: MegaMoeSpec, *, overlap_w13: bool = True) -> tuple[Any, Any, Any, Any]:
    """Build forward/backward graphs and their serialized RuntimeConfig objects."""
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
        swiglu_hf_bf16=spec.swiglu_hf_bf16,
        expert_weight_layout=spec.expert_weight_layout,
    )
    forward_graph.propagate_splits(task_values)
    attach_mega_moe_l1_view_resolvers(forward_graph)
    forward_data = build_forward_config(forward_graph, task_values, spec.rank_id, spec.num_cube_cores)
    backward_graph = build_backward_graph(
        task_values,
        dispatch_sv=spec.dispatch_split,
        swiglu_sv=spec.swiglu_split,
        combine_sv=spec.combine_split,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_cube_cores=spec.num_cube_cores,
        swiglu_limit=spec.swiglu_limit,
        swiglu_hf_bf16=spec.swiglu_hf_bf16,
        expert_weight_layout=spec.expert_weight_layout,
    )
    if spec.replica_slots_per_rank:
        for operation in backward_graph.topological_sort():
            for tensor_spec in (*operation.inputs, *operation.outputs):
                if tensor_spec.param_position in (6, 18):
                    tensor_spec.dtype_size = 4
    backward_graph.propagate_splits(task_values)
    attach_mega_moe_l1_view_resolvers(backward_graph)
    backward_data = build_backward_config(backward_graph, task_values, spec.rank_id, spec.num_cube_cores)
    _apply_execution_policies(spec, task_values, forward_data, backward_data, backward_graph, overlap_w13)
    return (forward_graph, forward_data, backward_graph, backward_data)


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
        **_mega_moe_owner_options(topology, num_cube_cores=spec.num_cube_cores),
    }
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
            forward_graph, forward_config, parameter_count=16, extra_aic_records=weight_records, **options
        ),
        prepare_mega_kernel_diagnostics(
            backward_graph,
            backward_config,
            parameter_count=20,
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


def build_mega_moe_plan(spec: MegaMoeSpec, device: Any) -> MegaMoePlan:
    """Build trimmed dense forward/backward runtime images without fusion slots.

    Args:
        spec: Validated local-token and expert topology specification.
        device: NPU device receiving serialized descriptors and tiling tensors.

    Returns:
        Rank-local forward and backward runtime resources.
    """
    if spec._batch_empty_combine and (not _soc_name(torch.device(device)).startswith("Ascend910_93")):
        raise ValueError("Empty Combine credit batching currently requires Ascend910C")
    if spec._batch_empty_dispatch or spec._batch_empty_combine:
        require_empty_dispatch_native()
    automatic = spec._gradient_queue_order in ("auto", "auto_w1")
    spec = _resolve_gradient_queue(spec, device)
    projections = ACTIVE_GRADIENT_PROJECTIONS.get(spec._gradient_queue_order)
    baseline = replace(spec, _gradient_queue_order="original") if projections else spec
    forward_graph, forward_config, backward_graph, backward_config = _build_runtime_artifacts(
        baseline, overlap_w13=False
    )
    spec, act_first, original_reuse = _qualify_gradient_order(spec, baseline, backward_config, projections, automatic)
    original_backward_config = backward_config
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
    fwd_diagnostics, bwd_diagnostics = _prepare_diagnostics(
        spec, forward_graph, forward_config, backward_graph, backward_config, device
    )
    gmm_options = {
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "num_groups": spec.local_experts,
        "num_cube_cores": spec.num_cube_cores,
    }
    prefix_options = _physical_prefix_options(spec)
    return MegaMoePlan(
        spec=spec,
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
        up_proj_tiling=_tensor_from_bytes(
            get_up_proj_tiling_bytes(forward_graph.get_op("up_proj").split_value, **gmm_options), device
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
        down_proj_tiling=_tensor_from_bytes(
            get_down_proj_tiling_bytes(forward_graph.get_op("down_proj").split_value, **gmm_options), device
        ),
        backward_graph=backward_graph,
        bwd_diagnostics=bwd_diagnostics,
        bwd_diagnostics_no_replica=fallback,
        act_grad_tiling=_tensor_from_bytes(
            get_act_grad_tiling_bytes(backward_graph.get_op("act_grad").split_value, **gmm_options), device
        ),
        gate_grad_tiling=_tensor_from_bytes(
            get_gate_grad_tiling_bytes(backward_graph.get_op("gate_grad").split_value, **gmm_options), device
        ),
        w1_grad_tiling=_tensor_from_bytes(
            get_w1_grad_tiling_bytes(backward_graph.get_op("w1_grad").split_value, **gmm_options), device
        ),
        w2_grad_tiling=_tensor_from_bytes(
            get_w2_grad_tiling_bytes(backward_graph.get_op("w2_grad").split_value, **gmm_options), device
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
