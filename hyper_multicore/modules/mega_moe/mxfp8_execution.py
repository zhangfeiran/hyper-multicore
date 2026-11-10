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
"""Orchestrate native MX expert graphs without retaining gathered weight views."""

from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import contextmanager
from collections.abc import Iterator
from functools import partial
from typing import Any, Callable

import torch
from hyper_multicore._loader import load_torch_npu

from hyper_multicore.torch import ops as multicore_ops
from hyper_multicore.replica.plan import ExpertExecutionPlan

from .mxfp8_function import MXFP8ForwardState
from .mxfp8_gradients import lease_mx_gradient_return
from .mxfp8_diagnostics import current_mx_profile_scope, mx_profile_capture_enabled, profile_mx_launch
from .mxfp8_ops import launch_mxfp8_task_graph, prepare_mxfp8_launch, serialize_mxfp8_launch
from .mxfp8_weights import MXFP8PhysicalWeight
from .mxfp8_ingress import borrow_row_planes, quantize_source_rows
from .mxfp8_replica import lease_replica_weights, reduce_replica_gradients
from .route import prepare_topk_route


@dataclass(frozen=True)
class _ExecutionMetadata:
    plan: Any
    hidden_shape: tuple[int, ...]
    source_rows: int
    probs_dtype: torch.dtype


def _quantized_buffers(
    rows: int, columns: int, experts: int, device: Any, *, row_only: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Allocate live axes and independent scale storage for native byte bindings.

    Args:
        rows: Current token workspace capacity.
        columns: Activation width in elements.
        experts: Physical local experts, including guest slots.
        device: Device receiving activation storage.
        row_only: Whether the graph neither reads nor writes column outputs.

    Returns:
        Row data and scales followed by column data and packed column scales.
    """
    row = torch.empty((rows, columns), dtype=torch.float8_e4m3fn, device=device)
    row_scale = torch.empty((rows, columns // 64, 2), dtype=torch.uint8, device=device)
    column = row if row_only else torch.empty((rows, columns), dtype=torch.float8_e4m3fn, device=device)
    scale_shape = ((rows + 63) // 64 + experts, columns, 2)
    # Native TensorList scale bindings require their own byte-shaped storage.
    column_scale = torch.empty(scale_shape, dtype=torch.uint8, device=device)
    return row, row_scale, column, column_scale


def _use_row_only(spec: Any, retain_backward: bool) -> bool:
    return spec.mx_swiglu_row_only and not retain_backward


def _forward_quantized_buffers(
    spec: Any, rows: int, device: Any, *, row_only: bool,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    """Apply the same retained-axis contract to both forward activations."""
    return tuple(
        _quantized_buffers(rows, width, spec.local_experts, device, row_only=row_only)
        for width in (spec.hidden_size, spec.intermediate_size)
    )


def _forward_ingress_buffers(spec: Any, workspace: Any, hidden: torch.Tensor, route: Any, *,
                              source_row_fp8: bool, row_only: bool) -> tuple[Any, ...]:
    """Prepare original source planes or the existing routed source under the same workspace lease."""
    capacity, device = route.metadata.expert_capacity, hidden.device
    if not source_row_fp8:
        x_buffers, a_buffers = _forward_quantized_buffers(spec, capacity, device, row_only=row_only)
        return route.routed_tokens, x_buffers, a_buffers, (), route.routed_tokens.shape[0]
    if route.routed_tokens is not None or route.permute_src_index is None:
        raise ValueError("Source-row FP8 ingress requires scalar route indices without expanded token storage.")
    source, source_scale = quantize_source_rows(hidden)
    row, scale = borrow_row_planes(workspace.expert_buffer)
    x_buffers = (row, scale, row, torch.empty(1, dtype=torch.uint8, device=device))
    a_buffers = _quantized_buffers(capacity, spec.intermediate_size, spec.local_experts, device, row_only=True)
    return source, x_buffers, a_buffers, (source_scale, route.permute_src_index), hidden.shape[0] * spec.top_k


def _forward_graph_buffers(workspace: Any, metadata: Any, source: torch.Tensor,
                             weights: tuple[Any, Any], activations: tuple[torch.Tensor, torch.Tensor],
                             quantized: tuple[tuple[torch.Tensor, ...], ...], runtime: Any) -> list[torch.Tensor]:
    """Bind the forward business ABI without changing activation or weight ownership."""
    weight1, weight2 = weights
    hidden_size = source.shape[1]
    up, down = activations
    (x_row, x_scale, x_col, x_col_scale), (a_row, a_scale, a_col, a_col_scale) = quantized
    return [
        workspace.expert_buffer,
        metadata.dispatch_target_off * hidden_size,
        source,
        metadata.dispatch_src_off * hidden_size,
        metadata.dispatch_size * hidden_size,
        weight1.col_data,
        metadata.group_list,
        up,
        a_row,
        weight2.col_data,
        metadata.group_list,
        down,
        workspace.routed_buffer,
        metadata.combine_target_off * hidden_size,
        metadata.combine_src_off * hidden_size,
        metadata.combine_size * hidden_size,
        x_row,
        x_scale,
        x_col,
        x_col_scale,
        a_scale,
        a_col,
        a_col_scale,
        weight1.col_scale,
        weight2.col_scale,
        runtime.gmm,
        runtime.quant,
    ]


def _source_row_launches(host: Any, *functions: Callable) -> tuple[Callable, ...]:
    """Bind the explicit source protocol only to eligible original-forward calls."""
    if not getattr(host, "source_row_fp8", False):
        return functions
    return tuple(partial(function, source_row_fp8=True) for function in functions)


def _empty_expert_ranges(counts: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Return maximal empty expert intervals without reading device state.

    Args:
        counts: Exact CPU row totals in physical expert order.

    Returns:
        Consecutive half-open ranges containing only zero-row experts.
    """
    ranges, start = [], None
    for expert in range(len(counts) + 1):
        if expert < len(counts) and counts[expert] == 0:
            if start is None:
                start = expert
        elif start is not None:
            ranges.append((start, expert))
            start = None
    return tuple(ranges)


def _unpermute_mx_output(expert_output: torch.Tensor, mapping: torch.Tensor, probs: torch.Tensor,
                         routed_buffer: torch.Tensor, require_owned: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore tokens while keeping a no-clone checkpoint result independent of workspace reuse."""
    probabilities = probs.float().contiguous()
    output = load_torch_npu().npu_moe_token_unpermute(expert_output, mapping, probs=probabilities)
    if require_owned and output.untyped_storage().data_ptr() == routed_buffer.untyped_storage().data_ptr():
        raise RuntimeError("No-grad MX unpermute output must not alias shared routed storage.")
    return output, probabilities


class MXFP8Execution:
    """Own one autograd invocation's orchestration while borrowing serial resources."""

    def __init__(
        self,
        resources: Any,
        *,
        nograd_no_clone: bool = False,
        replica_route: Any = None,
        counts: torch.Tensor | None = None,
    ) -> None:
        """Bind validated invocation resources without retaining gathered weight views."""
        self.resources = resources
        self.replica_route = replica_route
        self.counts = counts
        self.nograd_no_clone = nograd_no_clone
        self._profile_scope = None

    def _launch(
        self,
        buffers: list[torch.Tensor],
        runtime: Any,
        events: torch.Tensor,
        plan: Any,
        direction: str = "forward",
        *,
        swiglu_row_only: bool = False,
    ) -> None:
        if events.numel() < runtime.event_capacity * 4:
            raise ValueError("MXFP8 workspace event storage is smaller than the current graph event capacity.")
        launch_graph = (
            partial(launch_mxfp8_task_graph, vector_policy=plan.spec.mx_vector_policy)
            if plan.spec.mx_vector_policy != "legacy_odd"
            else launch_mxfp8_task_graph
        )
        if swiglu_row_only:
            launch_graph = partial(launch_graph, swiglu_row_only=True)
        prepare_launch, serialize_launch = prepare_mxfp8_launch, serialize_mxfp8_launch
        prepare_launch, serialize_launch, launch_graph = _source_row_launches(
            runtime.host, prepare_launch, serialize_launch, launch_graph)
        if runtime.host.kernel_gradients:
            prepare_launch = partial(prepare_launch, kernel_gradients=True)
            serialize_launch = partial(serialize_launch, kernel_gradients=True)
        if runtime.host.weight_prefetch:
            prepare_launch = partial(prepare_launch, weight_prefetch=True)
            serialize_launch = partial(serialize_launch, weight_prefetch=True)
        if mx_profile_capture_enabled():
            scope = current_mx_profile_scope()
            if direction != "forward" and scope is None:
                scope = self._profile_scope
            if plan.spec.mx_ready_depth:
                scope = dict(scope or {}, ready_depth=str(plan.spec.mx_ready_depth),
                             ready_scan_wait="unrecorded_scheduler_gap")
            if direction == "forward":
                self._profile_scope = scope
            if scope is not None and direction == "backward":
                scope = {**scope, "phase": "backward"}
            profile_mx_launch(
                buffers, runtime, events, plan.spec, direction, prepare_launch, launch_graph, scope=scope
            )
            return
        diagnostic = runtime.normal_diagnostic
        if getattr(self.resources, "metadata_transfer", None) is None:
            launch = prepare_launch(buffers, runtime.runtime, events, plan.spec)
            launch_graph(buffers, runtime.runtime, events, diagnostic, launch)
        else:
            content = serialize_launch(buffers, runtime.runtime, events, plan.spec)
            self.resources.metadata_transfer.launch(
                content,
                runtime.runtime.device,
                lambda launch: launch_graph(buffers, runtime.runtime, events, diagnostic, launch),
            )

    def forward(
        self,
        hidden: torch.Tensor,
        ids: torch.Tensor,
        probs: torch.Tensor,
        weight1: MXFP8PhysicalWeight,
        weight2: MXFP8PhysicalWeight,
        retain_backward: bool,
    ) -> MXFP8ForwardState:
        """Execute the forward graph with current column weights and owned saved activations.

        Args:
            hidden: Original token-major activation tensor.
            ids: Final physical expert IDs for each selected route.
            probs: Original differentiable Router probabilities.
            weight1: Current first-projection physical weight axes.
            weight2: Current second-projection physical weight axes.
            retain_backward: Whether to retain owned activations for a later backward.
        """
        resources, workspace = self.resources, self.resources.workspace
        home_only = self.replica_route is not None and self.replica_route.home_only
        spec = resources.home_spec if home_only else resources.spec
        flat = hidden.reshape(-1, spec.hidden_size)
        row_only = _use_row_only(spec, retain_backward)
        source_row_fp8 = bool(getattr(resources, "source_row_fp8", False) and row_only
                              and not spec.replica_slots_per_rank)
        route_spec = replace(spec, forward_ingress_mode="indexed") if source_row_fp8 else spec
        route = prepare_topk_route(
            flat, ids, probs, route_spec, self.counts, workspace=workspace, replica_route=self.replica_route
        )
        resources.ensure_capacity(route.maximum_received_slots)
        workspace.ensure(resources.spec, hidden.dtype, hidden.device)
        plan = resources.plan_for_capacity(workspace.expert_capacity, hidden.device,
                                           home_only=home_only, swiglu_row_only=row_only, source_row_fp8=source_row_fp8)
        resources.last_execution_plan = plan
        metadata = route.metadata
        capacity, device = metadata.expert_capacity, hidden.device
        dispatch_source, x_buffers, a_buffers, source_bindings, routed_rows = _forward_ingress_buffers(
            spec, workspace, flat, route, source_row_fp8=source_row_fp8, row_only=row_only)
        x_col, x_col_scale = x_buffers[2:]
        a_col, a_col_scale = a_buffers[2:]
        up = torch.empty((capacity, 2 * spec.intermediate_size), dtype=torch.bfloat16, device=device)
        down = torch.empty((capacity, spec.hidden_size), dtype=torch.bfloat16, device=device)
        workspace.claim()
        try:
            buffers = _forward_graph_buffers(
                workspace, metadata, dispatch_source, (weight1, weight2), (up, down),
                (x_buffers, a_buffers), plan.forward)
            events = workspace.prepare_event_counters(forward=True)
            with self._lease_weights((weight1, weight2), plan.forward, backward=False) as (guests, ready):
                buffers.extend(guests)
                buffers.extend(ready)
                buffers.extend(source_bindings)
                self._launch(buffers, plan.forward, events, plan)
            expert_output = workspace.routed_buffer[:routed_rows]
            if retain_backward or not self.nograd_no_clone:
                expert_output = expert_output.clone()
            output, probabilities = _unpermute_mx_output(
                expert_output, route.unpermute_mapping, probs, workspace.routed_buffer,
                self.nograd_no_clone and not retain_backward)
            saved = (
                (
                    x_col, x_col_scale,
                    a_col,
                    a_col_scale,
                    up,
                    expert_output,
                    probabilities,
                    route.unpermute_mapping,
                    metadata.group_list,
                    metadata.dispatch_target_off,
                    metadata.dispatch_src_off,
                    metadata.dispatch_size,
                    metadata.combine_target_off,
                    metadata.combine_src_off,
                    metadata.combine_size,
                )
                if retain_backward
                else ()
            )
            return MXFP8ForwardState(
                output.reshape_as(hidden),
                saved,
                _ExecutionMetadata(plan, tuple(hidden.shape), routed_rows, probs.dtype),
                workspace,
            )
        finally:
            workspace.release()

    def backward(
        self,
        gradient: torch.Tensor,
        saved: tuple[torch.Tensor, ...],
        metadata: _ExecutionMetadata,
        weight1: MXFP8PhysicalWeight,
        weight2: MXFP8PhysicalWeight,
    ) -> tuple[torch.Tensor, ...]:
        """Resolve current dual-axis weights and return logical owner gradients.

        Args:
            gradient: Gradient of the owned token-major output.
            saved: Owned forward activations and exact route metadata.
            metadata: Invocation plan, shape, probability dtype and source-row count.
            weight1: Current first-projection physical weight axes.
            weight2: Current second-projection physical weight axes.
        """
        workspace, plan = self.resources.workspace, metadata.plan
        spec, device = plan.spec, gradient.device
        (
            x_col,
            x_col_scale,
            a_col,
            a_col_scale,
            up,
            expert_output,
            probs,
            mapping,
            group_list,
            dispatch_target,
            dispatch_source,
            dispatch_size,
            combine_target,
            combine_source,
            combine_size,
        ) = saved
        capacity = x_col.shape[0]
        dy_row, dy_scale, dy_col, dy_col_scale = _quantized_buffers(
            capacity, spec.hidden_size, spec.local_experts, device
        )
        du_row, du_scale, du_col, du_col_scale = _quantized_buffers(
            capacity, 2 * spec.intermediate_size, spec.local_experts, device
        )
        source = torch.empty((metadata.source_rows, spec.hidden_size), dtype=torch.bfloat16, device=device)
        grad_probs = torch.empty_like(probs)
        da = torch.empty((capacity, spec.intermediate_size), dtype=torch.bfloat16, device=device)
        dx = torch.empty((capacity, spec.hidden_size), dtype=torch.bfloat16, device=device)
        dw1, dw2 = self._weight_gradient_buffers(plan, (weight1, weight2), device, route=self.replica_route)
        workspace.claim()
        try:
            multicore_ops.mega_moe_unpermute_grad_out(
                expert_output,
                gradient.reshape(-1, spec.hidden_size).contiguous(),
                mapping,
                probs,
                source,
                grad_probs,
            )
            buffers = [
                workspace.expert_buffer,
                dispatch_target * spec.hidden_size,
                source,
                dispatch_source * spec.hidden_size,
                dispatch_size * spec.hidden_size,
                a_col,
                dw2,
                weight2.row_data,
                da,
                up,
                du_row,
                weight1.row_data,
                dx,
                workspace.routed_buffer,
                combine_target * spec.hidden_size,
                combine_source * spec.hidden_size,
                combine_size * spec.hidden_size,
                x_col,
                dw1,
                group_list,
                dy_row,
                dy_scale,
                dy_col,
                dy_col_scale,
                a_col_scale,
                weight2.row_scale,
                du_scale,
                du_col,
                du_col_scale,
                weight1.row_scale,
                x_col_scale,
                plan.backward.gmm,
                plan.backward.quant,
            ]
            events = workspace.prepare_event_counters(forward=False)
            self._launch_backward(buffers, events, plan, (dw1, dw2), (weight1, weight2))
            grad_hidden = multicore_ops.moe_token_permute_grad(
                workspace.routed_buffer[: metadata.source_rows],
                mapping,
                gradient.numel() // spec.hidden_size,
                spec.top_k,
            )
            if not plan.backward.host.kernel_gradients:
                dw1, dw2 = reduce_replica_gradients((dw1, dw2), self.replica_route)
            return grad_hidden.reshape(metadata.hidden_shape), grad_probs.to(metadata.probs_dtype), dw1, dw2
        finally:
            workspace.release()

    @staticmethod
    def _weight_gradient_buffers(plan: Any, weights: tuple[MXFP8PhysicalWeight, ...],
                                 device: Any, *, route: Any = None) -> tuple[torch.Tensor, ...]:
        """Clear skipped matrices while complete-K TN tasks overwrite nonempty outputs.

        Args:
            plan: Current graph and owner/guest output geometry.
            weights: Current physical roots supplying matrix dimensions.
            device: Device receiving fresh FP32 partials.
            route: Optional exact CPU replica quotas; absent quotas retain full clearing.

        Returns:
            Fresh first and second projection partials, ready for their TN producers.
        """
        owners = plan.spec.local_experts
        if plan.backward.host.kernel_gradients:
            owners -= plan.spec.replica_slots_per_rank
        cpu_plan = route is not None and isinstance(route.plan, ExpertExecutionPlan)
        counts = route.plan.destination_counts[route.rank][:owners] if cpu_plan else None
        if counts is not None and len(counts) != owners:
            raise ValueError("MX gradient output geometry exceeds the exact replica destination quotas")
        empty = None if counts is None else tuple(expert for expert, rows in enumerate(counts) if rows == 0)
        clear_all = empty is None or len(empty) == owners
        allocate = torch.zeros if clear_all else torch.empty
        gradients = tuple(allocate((owners, *weight.logical_shape[1:]), dtype=torch.float32, device=device)
                          for weight in weights)
        if not clear_all:
            # Exact CPU quotas avoid another readback. Empty owners must start at zero before guest additions.
            for start, end in _empty_expert_ranges(counts):
                for gradient in gradients:
                    gradient[start:end].zero_()
        return gradients

    def _launch_backward(self, buffers: list[torch.Tensor], events: torch.Tensor, plan: Any,
                          gradients: tuple[torch.Tensor, ...], weights: tuple[MXFP8PhysicalWeight, ...]) -> None:
        """Keep both physical weights and symmetric guest gradients live through the final native join."""
        with self._lease_weights(weights, plan.backward, backward=True) as (guests, ready):
            buffers.extend(guests)
            with lease_mx_gradient_return(self.resources, plan, self.replica_route, gradients,
                                          weights) as runtime:
                buffers.extend(ready)
                self._launch(buffers, runtime, events, plan, direction="backward")

    @contextmanager
    def _lease_weights(self, weights: tuple[MXFP8PhysicalWeight, ...], runtime: Any,
                        *, backward: bool) -> Iterator[tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]]:
        """Keep exact guest bytes and projection-ready storage live until native consumers join."""
        if runtime.host.weight_prefetch:
            with self.resources.weight_prefetch.lease(weights, self.replica_route,
                                                      self.resources.workspace.replica_provider,
                                                      backward=backward) as (guests, information):
                yield guests, (information,)
        else:
            with lease_replica_weights(weights, self.replica_route, backward=backward) as guests:
                yield guests, ()
