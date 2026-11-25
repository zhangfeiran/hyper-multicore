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

from collections import Counter
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from hyper_multicore.diagnostics import MegaKernelDiagnostics, prepare_mega_kernel_diagnostics
from hyper_multicore.modules.mega_moe.backends.bf16.backward.gen_runtime_data import (
    build_config_for_rank as build_backward_config,
)
from hyper_multicore.modules.mega_moe.backends.bf16.backward.graph import (
    build_backward_graph,
)
from hyper_multicore.modules.mega_moe.backends.bf16.backward.storage import can_reuse_backward_dispatch
from hyper_multicore.modules.mega_moe.backends.bf16.backward.tiling_tables import (
    get_act_grad_tiling_bytes,
    get_gate_grad_tiling_bytes,
    get_w1_grad_tiling_bytes,
    get_w2_grad_tiling_bytes,
)
from hyper_multicore.modules.mega_moe.backends.bf16.forward.gen_runtime_data import (
    build_config_for_rank as build_forward_config,
)
from hyper_multicore.modules.mega_moe.backends.bf16.forward.graph import (
    build_forward_graph,
)
from hyper_multicore.modules.mega_moe.backends.bf16.forward.tiling_tables import (
    get_down_proj_tiling_bytes,
    get_up_proj_tiling_bytes,
)
from hyper_multicore.modules.mega_moe.schedule import configure_expert_queues, defer_local_w13
from hyper_multicore.scheduler.config import (
    FAST_DEPENDENCY_POLL_INTERVAL_US,
    TaskSplitValue,
    TaskType,
)
from hyper_multicore.scheduler.graph import ComputeGraph
from hyper_multicore.scheduler.runtime import copy_runtime_config, cube_task_index_offset, serialize_runtime_config
from hyper_multicore.tasks.gmm import GroupedMatmulTaskMode

from ...diagnostics import _mega_moe_owner_options, _adapt_mega_moe_diagnostic_graph
from ...probability import (
    PROBABILITY_POSITION_FORWARD, PROBABILITY_POSITION_BACKWARD,
)

from ...schedule_policy import (
    MegaMoeSchedulePolicy, compact_graph,
)
from ...spec import MegaMoeSpec, _COMMUNICATION_SPLIT, _align_capacity
from ...tensor_dump import attach_mega_moe_l1_view_resolvers
from .activation import prepare_activation_tiling


@dataclass(frozen=True)
class MegaMoePlan:
    """Rank-local schedules and tiling tensors for forward and backward."""

    spec: MegaMoeSpec
    forward_graph: ComputeGraph
    fwd_diagnostics: MegaKernelDiagnostics
    event_counter_bytes: int
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
    _recorded_streams: set = field(default_factory=set, repr=False, compare=False)
    _w13_return_queues: dict[tuple[int, tuple[int, ...]], tuple[int, torch.Tensor] | None] = field(
        default_factory=dict, repr=False, compare=False,
    )

    @property
    def fwd_runtime_config(self) -> Any:
        """Return the original ordinary forward RuntimeConfig."""
        return self.fwd_diagnostics.runtime_config

    @property
    def bwd_runtime_config(self) -> Any:
        """Return the original ordinary backward RuntimeConfig."""
        return self.bwd_diagnostics.runtime_config

    @cached_property
    def fenced_forward_runtime(self) -> torch.Tensor:
        """Remove the forward ready wait after the count exchange has fenced it."""
        config = copy_runtime_config(self.fwd_diagnostics._runtime._host_config)
        if not config.ready_event or config.completion_event:
            raise RuntimeError("Count fencing requires an ordinary Push ready event")
        config.ready_event = 0
        return _tensor_from_bytes(serialize_runtime_config(config), self.fwd_runtime_config.device)

    @cached_property
    def _w13_cube_waves(self):
        config = self.bwd_diagnostics._runtime._host_config
        cubes = self.spec.num_cube_cores
        experts = self.spec.local_experts
        count = config.task_index_num[0]
        start = count - experts * cubes
        if config.queue_layout != 2 or start < 0 or start % cubes:
            return None
        for lane, queue in enumerate(config.worker_queues[:cubes]):
            if (queue.begin, queue.end, queue.stride, queue.ready_depth) != (lane, count, cubes, 0):
                return None
        prefix_events = Counter(config.all_tasks[task].trigger_event for task in config.cube_task_indices[:start]
                                if config.all_tasks[task].outputs[0].input_position == 12)
        waves = []
        for expert in range(experts):
            begin = start + expert * cubes
            wave = config.cube_task_indices[begin:begin + cubes]
            for lane, task_id in enumerate(wave):
                task = config.all_tasks[task_id]
                dependency = task.dependent_event
                if (task.task_type != TaskType.TASK_GROUPED_MATMUL or task.outputs[0].input_position != 18
                        or task.extra_value_1 != GroupedMatmulTaskMode.SINGLE_GROUP
                        or task.task_index != expert * cubes + lane or dependency not in prefix_events
                        or prefix_events[dependency] != config.all_event_num_triggers[dependency]):
                    return None
            waves.append(wave)
        return cube_task_index_offset(config) + start * 4, waves

    def w13_return_queue(self, home: int, owners: tuple[int, ...]) -> tuple[int, torch.Tensor] | None:
        """Move full export/merge waves before local-only W13 after GateGrad."""
        key = home, owners
        if self.spec.architecture == 3510:
            if key not in self._w13_return_queues:
                if len(self._w13_return_queues) >= 32:
                    del self._w13_return_queues[next(iter(self._w13_return_queues))]
                config = self.bwd_diagnostics._runtime._host_config
                revised = defer_local_w13(config, self.spec.local_experts, home, owners)
                if revised is not None:
                    offset, queue = revised
                    self._w13_return_queues[key] = offset, _tensor_from_bytes(queue, self.bwd_runtime_config.device)
                else:
                    self._w13_return_queues[key] = None
            return self._w13_return_queues.get(key)
        wave_layout = self._w13_cube_waves
        if wave_layout is None or len(owners) == home:
            return None
        offset, waves = wave_layout
        if not 0 < home <= len(waves):
            return None
        order = (*owners, *range(home, len(waves)), *(expert for expert in range(home) if expert not in owners))
        if order == tuple(range(len(waves))):
            return None
        if key not in self._w13_return_queues:
            queue = np.asarray([waves[expert] for expert in order], dtype="<i4").tobytes()
            self._w13_return_queues[key] = offset, _tensor_from_bytes(queue, self.bwd_runtime_config.device)
        return self._w13_return_queues[key]

    def record_stream(self, runtime_config: Any, *, stream: Any = None) -> None:
        """Register immutable plan storage once per stream, preserving allocator lifetime."""
        if not isinstance(runtime_config, torch.Tensor) or runtime_config.device.type != "npu":
            return
        if stream is None:
            stream = torch.npu.current_stream(runtime_config.device)
        # Reading npu_stream can drain the PTA queue while Python holds the GIL.
        key = (runtime_config.device, stream.stream_id)
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


def _build_task_values(spec: MegaMoeSpec) -> TaskSplitValue:
    """Translate rank-local token shape into scheduler task coordinates.

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
    )


def _uses_explicit_queues(spec: MegaMoeSpec) -> bool:
    """Validate the device-specific queue and lowering contract."""
    explicit_queues = spec.architecture == 3510
    if explicit_queues and (spec.batch_experts or spec.nominal_schedule or spec.lowering == "compact"):
        raise ValueError("arch35 requires dense per-expert lowering")
    if not explicit_queues and spec.schedule_policy != "auto":
        raise ValueError("Named MegaMoE schedule policies require arch35")
    return explicit_queues


def _build_runtime_artifacts(
    spec: MegaMoeSpec, weight_grad_dtype: torch.dtype | None = None, *,
    replica_pipeline: bool = False, reserve_merge_workers: bool = False,
) -> tuple[Any, Any, Any, Any]:
    """Build forward/backward graphs and their serialized RuntimeConfig objects."""
    explicit_queues = _uses_explicit_queues(spec)
    task_values = _build_task_values(spec)
    dispatch_split, combine_split = spec.dispatch_split, spec.combine_split
    if (explicit_queues and spec.auto_tokens and spec.lowering == "auto"
            and spec.dispatch_mode == "pull" and spec.transport_engine == "mte"):
        # Coarsen default descriptors when EP already supplies communication
        # concurrency. Keep at least four fragments at large source lengths.
        fragments = max(4, 2 * spec.num_cube_cores // spec.ep_size)
        rows = (spec.plan_tokens + fragments - 1) // fragments
        split = min(_align_capacity(rows), 8 * _COMMUNICATION_SPLIT)
        if dispatch_split == _COMMUNICATION_SPLIT:
            dispatch_split = split
        if combine_split == _COMMUNICATION_SPLIT:
            combine_split = split
    forward_graph = build_forward_graph(
        task_values,
        dispatch_sv=dispatch_split,
        swiglu_sv=spec.swiglu_split,
        combine_sv=combine_split,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_cube_cores=spec.num_cube_cores,
        activation_parameters=spec.activation_parameters,
        apply_probs_before_fc2=spec.apply_probs_before_fc2,
        expert_weight_layout=spec.expert_weight_layout,
    )
    compact_dispatch = spec.dispatch_mode == "push" or spec.compact_pull
    compact = (not explicit_queues and spec.lowering != "dense" and compact_dispatch
               and spec.intermediate_size < 2048
               and compact_graph(forward_graph, task_values, 2 * spec.num_cube_cores, spec.nominal_schedule))
    if not compact and (spec.lowering == "compact" or spec.nominal_schedule):
        raise ValueError("The requested compact lowering is unavailable for this MegaMoE configuration")
    forward_graph.propagate_splits(task_values)
    attach_mega_moe_l1_view_resolvers(forward_graph)
    forward_data = build_forward_config(
        forward_graph,
        task_values,
        spec.rank_id,
        spec.num_cube_cores,
    )
    backward_graph = build_backward_graph(
        task_values,
        dispatch_sv=dispatch_split,
        swiglu_sv=spec.swiglu_split,
        combine_sv=combine_split,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_cube_cores=spec.num_cube_cores,
        activation_parameters=spec.activation_parameters,
        apply_probs_before_fc2=spec.apply_probs_before_fc2,
        expert_weight_layout=spec.expert_weight_layout,
    )
    if weight_grad_dtype == torch.float32:
        for name in ("w1_grad", "w2_grad"):
            backward_graph.get_op(name).outputs[0].dtype_size = 4
    if compact:
        compact_graph(backward_graph, task_values, 2 * spec.num_cube_cores, spec.nominal_schedule)
    backward_graph.propagate_splits(task_values)
    attach_mega_moe_l1_view_resolvers(backward_graph)
    backward_data = build_backward_config(
        backward_graph,
        task_values,
        spec.rank_id,
        spec.num_cube_cores,
    )
    if explicit_queues:
        forward_data = configure_expert_queues(
            forward_data, forward_graph, spec, forward=True, bf16_replica_helpers=replica_pipeline)
        backward_data = configure_expert_queues(
            backward_data, backward_graph, spec, forward=False, bf16_replica_helpers=replica_pipeline)
        return forward_graph, forward_data, backward_graph, backward_data
    if not compact and spec.ready_depth not in (None, 0):
        raise ValueError("Ready lookahead on arch22 requires compact lowering")
    policy = MegaMoeSchedulePolicy(
        vector_ready_queue=bool(compact),
        vector_ready_depth=256 if spec.ready_depth is None else spec.ready_depth,
        pipeline_vector=bool(compact) and not spec.batch_experts,
        pipeline_cube=bool(compact) and not spec.batch_experts,
        batch_experts=spec.batch_experts,
        dependency_poll_interval=FAST_DEPENDENCY_POLL_INTERVAL_US,
    )
    # Complete backward projection waves in builder order for replica gradient returns.
    backward_policy = replace(policy, pipeline_cube=False) if replica_pipeline else policy
    for graph, config, schedule, forward in (
        (forward_graph, forward_data, policy, True), (backward_graph, backward_data, backward_policy, False),
    ):
        schedule.apply(config, graph, task_values, spec.num_cube_cores, forward=forward)
        if replica_pipeline and not schedule.pipeline_cube:
            for task_id in config.cube_task_indices[:config.task_index_num[0]]:
                config.all_tasks[task_id].extra_value_1 = GroupedMatmulTaskMode.SINGLE_GROUP
    if compact and reserve_merge_workers:
        end = backward_data.task_index_num[1]
        for worker, queue in enumerate(backward_data.worker_queues[spec.num_cube_cores:]):
            queue.begin = min(worker // 2, end) if worker % 2 else end
            queue.stride = spec.num_cube_cores
    return forward_graph, forward_data, backward_graph, backward_data


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
        "graph_adapter": _adapt_mega_moe_diagnostic_graph,
        "device": device,
        "rank": dist.get_rank() if dist.is_initialized() else spec.rank_id,
        **_mega_moe_owner_options(topology, num_cube_cores=spec.num_cube_cores),
    }
    return (
        prepare_mega_kernel_diagnostics(
            forward_graph, forward_config,
            parameter_count=PROBABILITY_POSITION_FORWARD + 1 if spec.apply_probs_before_fc2 else 16, **options),
        prepare_mega_kernel_diagnostics(
            backward_graph, backward_config,
            parameter_count=PROBABILITY_POSITION_BACKWARD + 1 if spec.apply_probs_before_fc2 else 20, **options),
    )


def build_mega_moe_plan(
    spec: MegaMoeSpec, device: Any, *, weight_grad_dtype: torch.dtype | None = None,
    replica_pipeline: bool = False, reserve_merge_workers: bool = False,
) -> MegaMoePlan:
    """Build trimmed dense forward/backward runtime images without fusion slots.

    Args:
        spec: Validated local-token and expert topology specification.
        device: NPU device receiving serialized descriptors and tiling tensors.
        weight_grad_dtype: Keep partial expert weight gradients in FP32.
        replica_pipeline: Use one GMM group per split weight destination.
        reserve_merge_workers: Give the even AIV lanes to owner gradient consumers.

    Returns:
        Rank-local forward and backward runtime resources.
    """
    if weight_grad_dtype not in (None, torch.float32):
        raise ValueError("Partial expert gradients support only the native dtype or FP32")
    if replica_pipeline and (spec.architecture not in (2201, 3510) or spec.batch_experts):
        raise ValueError("BF16 replica pipeline requires per-expert GMMs")
    if replica_pipeline and spec.architecture == 3510 and (
        weight_grad_dtype != torch.float32 or (spec.dispatch_mode == "pull" and spec.transport_engine != "udma")
    ):
        raise ValueError("Ascend950 replica pipeline requires FP32 partial gradients and UDMA for Pull")
    if reserve_merge_workers and not replica_pipeline:
        raise ValueError("Replica merge workers require a replica pipeline")
    forward_graph, forward_config, backward_graph, backward_config = _build_runtime_artifacts(
        spec, weight_grad_dtype, replica_pipeline=replica_pipeline, reserve_merge_workers=reserve_merge_workers)
    fwd_diagnostics, bwd_diagnostics = _prepare_diagnostics(
        spec, forward_graph, forward_config, backward_graph, backward_config, device,
    )
    gmm_options = {
        "hidden_size": spec.hidden_size,
        "intermediate_size": spec.intermediate_size,
        "num_groups": (1 if replica_pipeline or (
                       not spec.batch_experts and hasattr(forward_graph.get_op("dispatch"), "compact_lanes"))
                       else spec.local_experts),
        "num_cube_cores": spec.num_cube_cores,
    }
    return MegaMoePlan(
        spec=spec,
        event_counter_bytes=forward_config.event_capacity * 4,
        weight_grad_dtype=weight_grad_dtype,
        reuse_backward_dispatch=can_reuse_backward_dispatch(
            backward_config, spec.local_experts, spec.num_cube_cores,
        ),
        forward_graph=forward_graph,
        fwd_diagnostics=fwd_diagnostics,
        up_proj_tiling=_tensor_from_bytes(
            get_up_proj_tiling_bytes(forward_graph.get_op("up_proj").split_value, **gmm_options),
            device,
        ),
        swiglu_tiling=_tensor_from_bytes(
            prepare_activation_tiling(
                spec, backward=False, rows_per_task=forward_graph.get_op("activation").split_value,
            ),
            device,
        ),
        down_proj_tiling=_tensor_from_bytes(
            get_down_proj_tiling_bytes(forward_graph.get_op("down_proj").split_value, **gmm_options),
            device,
        ),
        backward_graph=backward_graph,
        bwd_diagnostics=bwd_diagnostics,
        act_grad_tiling=_tensor_from_bytes(
            get_act_grad_tiling_bytes(backward_graph.get_op("act_grad").split_value, **gmm_options),
            device,
        ),
        gate_grad_tiling=_tensor_from_bytes(
            get_gate_grad_tiling_bytes(backward_graph.get_op("gate_grad").split_value, **gmm_options),
            device,
        ),
        w1_grad_tiling=_tensor_from_bytes(
            get_w1_grad_tiling_bytes(backward_graph.get_op("w1_grad").split_value,
                                    expert_weight_layout=spec.expert_weight_layout, **gmm_options),
            device,
        ),
        w2_grad_tiling=_tensor_from_bytes(
            get_w2_grad_tiling_bytes(backward_graph.get_op("w2_grad").split_value,
                                    expert_weight_layout=spec.expert_weight_layout, **gmm_options),
            device,
        ),
        swiglu_grad_tiling=_tensor_from_bytes(
            prepare_activation_tiling(
                spec, backward=True, rows_per_task=backward_graph.get_op("activation").split_value,
            ),
            device,
        ),
    )
