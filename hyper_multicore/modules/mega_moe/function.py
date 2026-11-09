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
"""Differentiable autograd function for the low-level MegaMoe ABI."""

from __future__ import annotations

import struct
from contextlib import ExitStack
from dataclasses import dataclass, replace
from typing import Any

import torch

from hyper_multicore._loader import load_torch_npu
from hyper_multicore.replica._metadata import _async_device_bytes, _copy_device_bytes
from hyper_multicore.replica.recompute import get_recompute_state
from hyper_multicore.replica.routing import ReplicaRoute
from hyper_multicore.replica.transport import prefetch_weights, return_gradients
from hyper_multicore.torch import ops as multicore_ops

from .inactive_tasks import GuestTaskPrefix, apply_guest_task_prefix, select_guest_task_prefix
from .kernel_gradients import prepare_kernel_gradient_return
from .plan import MegaMoePlan, record_plan_stream
from .route import PreparedTopKRoute, RouteMetadata
from .workspace import MegaMoeWorkspace


@dataclass
class _ReplicaWeightGeneration:
    """Share the latest weight generation with the original checkpoint context."""

    value: object | None


class _RecomputedReplicaWeights:
    """Match weight generations in checkpoint invocation order without retaining tensors."""

    def __init__(self) -> None:
        """Initialize an ordered registry for one checkpoint invocation."""
        self.entries: list[tuple[Any, _ReplicaWeightGeneration]] = []
        self.cursor = 0

    def clear(self) -> None:
        """Release registry references while backward retains its generation holder."""
        self.entries.clear()
        self.cursor = 0

    def remember(self, generation: object | None, route: Any, *, recomputing: bool) -> _ReplicaWeightGeneration:
        """Refresh only the matching invocation and globally identical placement."""
        placement = (route.plan.config, route.plan.transfers)
        if not recomputing:
            holder = _ReplicaWeightGeneration(generation)
            self.entries.append((placement, holder))
            return holder
        if self.cursor >= len(self.entries):
            return _ReplicaWeightGeneration(generation)
        expected, holder = self.entries[self.cursor]
        self.cursor += 1
        holder.value = generation if placement == expected else None
        return holder


def _remember_replica_generation(generation: object | None, route: Any, workspace: Any) -> object | None:
    """Connect replayed weights to backward while preserving transport residency checks."""
    state = get_recompute_state()
    if state is None or route is None:
        return generation
    cache = state.get_resource((_RecomputedReplicaWeights, id(workspace)), _RecomputedReplicaWeights)
    return cache.remember(generation, route, recomputing=state.is_recomputing)


@dataclass(frozen=True)
class _ForwardIntermediates:
    """Forward tensors owned by one autograd invocation."""

    up_proj: Any
    activation: Any
    down_proj: Any


@dataclass(frozen=True)
class _SavedBackwardState:
    """Forward tensors and route metadata restored by backward."""

    dispatch: Any
    up_proj: Any
    activation: Any
    weight1: Any
    weight2: Any
    group_list: Any
    dispatch_src_off: Any
    dispatch_target_off: Any
    dispatch_size: Any
    combine_src_off: Any
    combine_target_off: Any
    combine_size: Any


@dataclass(frozen=True)
class _BackwardIntermediates:
    """Backward tensors owned by one autograd invocation."""

    grad_weight1: Any
    grad_weight2: Any
    act_grad: Any
    swiglu_grad: Any
    gate_dx: Any


@dataclass(frozen=True)
class _ForwardExecution:
    """Borrowed workspace and event state for one forward launch."""

    dispatch: Any
    source: Any
    combine: Any
    event_counters: Any
    gmm_workspace: Any
    intermediates: _ForwardIntermediates
    runtime_cache: dict[int, tuple[Any, bytes, torch.Tensor]] | None = None
    task_prefix: GuestTaskPrefix | None = None
    prefix_cache: dict[int, tuple[Any, tuple[int, int, int, int]]] | None = None


@dataclass(frozen=True)
class _BackwardExecution:
    """Borrowed workspace and event state for one backward launch."""

    dispatch: Any
    grad_x: Any
    event_counters: Any
    gmm_workspace: Any
    swiglu_workspace: Any
    intermediates: _BackwardIntermediates
    diagnostics: Any = None
    runtime_cache: dict[int, tuple[Any, bytes, torch.Tensor]] | None = None
    task_prefix: GuestTaskPrefix | None = None
    prefix_cache: dict[int, tuple[Any, tuple[int, int, int, int]]] | None = None


def _workspace_tensor(tensor: Any | None, name: str) -> Any:
    """Return an initialized workspace tensor or raise an invariant error."""
    if tensor is None:
        raise RuntimeError(f"MegaMoe workspace {name} is not initialized.")
    return tensor


def _dispatch_and_source(spec: Any, workspace: Any, rows: Any, capacity: int) -> tuple[Any, Any]:
    """Select transport storage while retaining a source-ready stream dependency."""
    if getattr(spec, "dispatch_mode", "push") == "push":
        return _workspace_tensor(workspace.expert_buffer, "expert_buffer"), rows
    source = _workspace_tensor(workspace.source_buffer, "source_buffer")[: rows.shape[0]]
    if source.data_ptr() != rows.data_ptr():
        source.copy_(rows)
    dispatch = torch.empty((capacity, spec.hidden_size), dtype=rows.dtype, device=rows.device)
    return dispatch, source


def _stage_backward_source(ctx: Any, grad_output: Any, permutation_inputs: tuple[Any, ...]) -> tuple[Any, Any]:
    """Write pull dY into SHMEM before expert scratch allocation."""
    if getattr(ctx.plan.spec, "dispatch_mode", "push") == "push":
        return grad_output, None
    source = _workspace_tensor(ctx.workspace.source_buffer, "source_buffer")[: ctx.source_rows]
    if not ctx.has_unpermute:
        source.copy_(grad_output)
        return source, None
    mapping, expert_output, probs = permutation_inputs
    grad_probs = torch.empty_like(probs)
    if ctx.source_rows:
        multicore_ops.mega_moe_unpermute_grad_out(expert_output, grad_output, mapping, probs, source, grad_probs)
    return source, grad_probs.to(ctx.topk_dtype) if ctx.needs_input_grad[7] else None


def _allocate_forward_intermediates(
    spec: Any,
    capacity: int,
    routed_tokens: Any,
    dispatch: Any,
) -> _ForwardIntermediates:
    """Allocate outputs whose received rows are overwritten before use."""
    # GMM and SwiGLU cover the received prefix; an empty rank reads no rows.
    up_proj = torch.empty(
        (capacity, spec.intermediate_size * 2),
        dtype=routed_tokens.dtype,
        device=routed_tokens.device,
    )
    activation = torch.empty(
        (capacity, spec.intermediate_size),
        dtype=routed_tokens.dtype,
        device=routed_tokens.device,
    )
    return _ForwardIntermediates(up_proj, activation, torch.empty_like(dispatch[:capacity]))


def _allocate_backward_intermediates(
    spec: Any,
    capacity: int,
    grad_output: Any,
    weight1: Any,
    weight2: Any,
    reusable_dispatch: Any | None = None,
    empty_experts: tuple[int, ...] | None = None,
) -> _BackwardIntermediates:
    """Allocate overwritten activations and zero-safe expert gradients."""
    gradient_dtype = torch.float32 if spec.replica_slots_per_rank else weight2.dtype
    clear_all = empty_experts is None or len(empty_experts) == weight1.shape[0]
    allocate = torch.zeros_like if clear_all else torch.empty_like
    grad_weight1 = allocate(weight1, dtype=gradient_dtype)
    grad_weight2 = allocate(weight2, dtype=gradient_dtype)
    if not clear_all:
        for expert in empty_experts:
            grad_weight1[expert].zero_()
            grad_weight2[expert].zero_()
    act_grad = torch.empty((capacity, spec.intermediate_size), dtype=grad_output.dtype, device=grad_output.device)
    swiglu_grad = torch.empty(
        (capacity, spec.intermediate_size * 2), dtype=grad_output.dtype, device=grad_output.device
    )
    gate_dx = (
        reusable_dispatch
        if reusable_dispatch is not None
        else torch.empty((capacity, spec.hidden_size), dtype=grad_output.dtype, device=grad_output.device)
    )
    return _BackwardIntermediates(grad_weight1, grad_weight2, act_grad, swiglu_grad, gate_dx)


def _native_weight_operand(weight: torch.Tensor, plan: MegaMoePlan) -> torch.Tensor:
    """Prepare one expert operand without copying native EOI storage."""
    if plan.spec.expert_weight_layout == "EOI":
        return weight
    return weight.contiguous()


def _saved_backward_state(saved_tensors: tuple[Any, ...]) -> _SavedBackwardState:
    """Restore named kernel state from the flat autograd tensor tuple."""
    return _SavedBackwardState(*saved_tensors[:12])


def _prepare_forward_execution(
    workspace: MegaMoeWorkspace, plan: MegaMoePlan, routed_tokens: Any, capacity: int
) -> _ForwardExecution:
    """Resolve workspace, event counters, and intermediate tensors for forward."""
    dispatch, source = _dispatch_and_source(plan.spec, workspace, routed_tokens, capacity)
    combine = _workspace_tensor(workspace.routed_buffer, "routed_buffer")
    return _ForwardExecution(
        runtime_cache=workspace.replica_runtime_images,
        prefix_cache=workspace.replica_task_prefixes,
        dispatch=dispatch,
        source=source,
        combine=combine,
        event_counters=workspace.prepare_event_counters(forward=True),
        gmm_workspace=_workspace_tensor(workspace.gmm_workspace, "gmm_workspace"),
        intermediates=_allocate_forward_intermediates(plan.spec, capacity, routed_tokens, dispatch),
    )


def _prepare_backward_execution(
    workspace: MegaMoeWorkspace,
    plan: MegaMoePlan,
    saved: _SavedBackwardState,
    grad_output: Any,
    *,
    has_replica_transfers: bool = True,
    overlap_w13: bool = True,
    empty_experts: tuple[int, ...] | None = None,
) -> _BackwardExecution:
    """Resolve workspace, event counters, and intermediate tensors for backward."""
    dispatch = (
        torch.empty_like(saved.dispatch)
        if plan.spec.dispatch_mode == "pull"
        else _workspace_tensor(workspace.expert_buffer, "expert_buffer")
    )
    grad_x = _workspace_tensor(workspace.routed_buffer, "routed_buffer")
    diagnostics = plan.bwd_diagnostics
    reuse_dispatch = plan.reuse_backward_dispatch
    if not has_replica_transfers or not overlap_w13:
        fallback = getattr(plan, "bwd_diagnostics_no_replica", None)
        if fallback is not None:
            diagnostics = fallback
            reuse_dispatch = getattr(plan, "bwd_runtime_no_replica_reuse_dispatch", False)
    return _BackwardExecution(
        runtime_cache=workspace.replica_runtime_images,
        prefix_cache=workspace.replica_task_prefixes,
        dispatch=dispatch,
        grad_x=grad_x,
        diagnostics=diagnostics,
        event_counters=workspace.prepare_event_counters(forward=False),
        gmm_workspace=_workspace_tensor(workspace.gmm_workspace, "gmm_workspace"),
        swiglu_workspace=_workspace_tensor(workspace.swiglu_grad_workspace, "swiglu_grad_workspace"),
        intermediates=_allocate_backward_intermediates(
            plan.spec,
            saved.dispatch.shape[0],
            grad_output,
            saved.weight1,
            saved.weight2,
            dispatch[: saved.dispatch.shape[0]] if reuse_dispatch else None,
            empty_experts=empty_experts,
        ),
    )


def _restore_input_gradient(ctx: Any, grad_x: Any, permutation_inputs: tuple[Any, ...]) -> Any:
    """Return an owned input gradient while the workspace lease is held."""
    if not ctx.needs_input_grad[0]:
        return None
    grad_x = grad_x[: ctx.source_rows]
    if not ctx.has_permutation:
        return grad_x.clone()
    (unpermute_mapping,) = permutation_inputs
    spec = ctx.plan.spec
    if ctx.token_rows == 0:
        return grad_x[:0].clone()
    # Consume the shared gradient before release records completion.
    # The permutation gradient owns its reduced [T, H] output.
    return multicore_ops.moe_token_permute_grad(grad_x, unpermute_mapping, ctx.token_rows, spec.top_k)


def _save_forward_state(
    ctx: Any,
    plan: MegaMoePlan,
    workspace: MegaMoeWorkspace,
    saved_dispatch: Any,
    up_proj: Any,
    activation: Any,
    weight1: Any,
    weight2: Any,
    metadata: RouteMetadata,
    permutation_inputs: tuple[Any, ...] = (),
) -> None:
    """Save owned forward tensors and route state for backward."""
    ctx.plan = plan
    ctx.workspace = workspace
    ctx.empty_owner_experts = getattr(metadata, "empty_owner_experts", None)
    ctx.save_for_backward(
        saved_dispatch,
        up_proj,
        activation,
        weight1,
        weight2,
        metadata.group_list,
        metadata.dispatch_src_off,
        metadata.dispatch_target_off,
        metadata.dispatch_size,
        metadata.combine_src_off,
        metadata.combine_target_off,
        metadata.combine_size,
        *permutation_inputs,
    )
    workspace.track_graph(ctx)


def _release_completed_graph(ctx: Any, workspace: MegaMoeWorkspace) -> None:
    """Keep retained graphs alive; older engines conservatively wait for context GC."""
    # The engine's retention flag avoids unpacking saved tensors or guessing from output references.
    keep_graph = getattr(
        torch._C._autograd, "_get_current_graph_task_keep_graph", None
    )  # pylint: disable=protected-access
    if keep_graph is not None and not keep_graph():
        workspace.release_graph(ctx)


def _split_runtime(
    base: torch.Tensor,
    pool: Any,
    home: int,
    *,
    backward: bool = False,
    gradient_return: torch.Tensor | None = None,
    home_only: bool = False,
    cache: dict[int, tuple[Any, bytes, torch.Tensor]] | None = None,
    task_prefix: GuestTaskPrefix | None = None,
    prefix_cache: dict[int, tuple[Any, tuple[int, int, int, int]]] | None = None,
) -> torch.Tensor:
    """Append split addresses after profiler preparation, retaining the base ABI."""
    if gradient_return is not None and (not backward or getattr(pool, "projection_ready", None) is None):
        raise ValueError("Kernel gradient metadata requires a projection-ready backward lease")
    if pool is None and (not home_only):
        return base
    if pool is None:
        projection_ready = None
        values = (home, 0, 0, 0, 0)
    else:
        pointers = (0, 0) if not backward else tuple((value.data_ptr() for value in pool.gradients))
        values = (home, *(value.data_ptr() for value in pool.weights), *pointers)
        projection_ready = getattr(pool, "projection_ready", None)
    if projection_ready is not None:
        bases, epoch = projection_ready
        if len(bases) != 2:
            raise ValueError("Multicore requires two projection ready addresses")
        data = (
            struct.pack("<II9Q", 1397774164, 5, *values, *bases, epoch, gradient_return.data_ptr())
            if gradient_return is not None
            else struct.pack("<II8Q", 1397774164, 4, *values, *bases, epoch)
        )
    else:
        data = struct.pack("<II5Q", 1397774164, 2, *values)
    runtime = _runtime_image(base, data, cache, static=projection_ready is None)
    return apply_guest_task_prefix(runtime, task_prefix, base=base, cache=prefix_cache)


def _runtime_image(
    base: torch.Tensor, data: bytes, cache: dict[int, tuple[Any, bytes, torch.Tensor]] | None, *, static: bool
) -> torch.Tensor:
    """Reuse images and update dynamic suffixes inside the serial workspace lease."""
    previous = None if cache is None else cache.get(id(base))
    if (
        previous is not None
        and previous[0] is base
        and (previous[1] == data or (not static and len(previous[1]) == len(data)))
    ):
        result = previous[2]
        if previous[1] != data:
            _copy_device_bytes(data, result[base.numel() :])
            cache[id(base)] = (base, data, result)
        result.record_stream(torch.npu.current_stream(base.device))
        return result
    metadata = _async_device_bytes(data, base.device)
    result = torch.cat((base, metadata))
    result.record_stream(torch.npu.current_stream(base.device))
    if cache is not None:
        cache[id(base)] = (base, data, result)
    return result


def _launch_forward_kernel(
    plan: MegaMoePlan,
    metadata: RouteMetadata,
    routed_tokens: Any,
    weight1: Any,
    weight2: Any,
    execution: _ForwardExecution,
    pool: Any = None,
) -> None:
    """Launch the internal forward ABI with prepared buffers and metadata."""
    spec = plan.spec
    kernel_parameters = (
        execution.dispatch,
        metadata.dispatch_target_off * spec.hidden_size,
        _native_source(routed_tokens),
        metadata.dispatch_src_off * spec.hidden_size,
        metadata.dispatch_size * spec.hidden_size,
        weight1,
        metadata.group_list,
        execution.intermediates.up_proj,
        execution.intermediates.activation,
        weight2,
        metadata.group_list,
        execution.intermediates.down_proj,
        execution.combine,
        metadata.combine_target_off * spec.hidden_size,
        metadata.combine_src_off * spec.hidden_size,
        metadata.combine_size * spec.hidden_size,
    )
    if not plan.fwd_diagnostics.is_active:
        record_plan_stream(plan, plan.fwd_runtime_config)
        multicore_ops.mega_moe(
            *kernel_parameters,
            execution.gmm_workspace,
            plan.up_proj_tiling,
            plan.swiglu_tiling,
            plan.down_proj_tiling,
            _split_runtime(
                plan.fwd_runtime_config,
                pool,
                weight1.shape[0],
                home_only=pool is None and bool(spec.replica_slots_per_rank),
                cache=execution.runtime_cache,
                task_prefix=execution.task_prefix,
                prefix_cache=execution.prefix_cache,
            ),
            execution.event_counters,
            spec.rank_id,
            spec.ep_size,
            spec.num_experts,
            spec.hidden_size,
            getattr(spec, "plan_tokens", spec.local_num_tokens),
        )
        return
    with plan.fwd_diagnostics.launch(
        kernel_parameters, direction="forward", fallback_event_counters=execution.event_counters
    ) as diagnostic_call:
        record_plan_stream(plan, diagnostic_call.runtime_config)
        multicore_ops.mega_moe_with_diagnostic_buffer(
            *kernel_parameters,
            execution.gmm_workspace,
            plan.up_proj_tiling,
            plan.swiglu_tiling,
            plan.down_proj_tiling,
            _split_runtime(
                diagnostic_call.runtime_config,
                pool,
                weight1.shape[0],
                home_only=pool is None and bool(spec.replica_slots_per_rank),
                task_prefix=None if diagnostic_call.preserves_task_queues else execution.task_prefix,
            ),
            diagnostic_call.event_counters,
            diagnostic_call.diagnostic_buffer,
            spec.rank_id,
            spec.ep_size,
            spec.num_experts,
            spec.hidden_size,
            getattr(spec, "plan_tokens", spec.local_num_tokens),
        )


def _launch_backward_kernel(
    plan: MegaMoePlan,
    saved: _SavedBackwardState,
    grad_output: Any,
    execution: _BackwardExecution,
    pool: Any = None,
    gradient_return: torch.Tensor | None = None,
) -> None:
    """Launch the internal backward ABI with prepared buffers and metadata."""
    spec = plan.spec
    diagnostics = execution.diagnostics or plan.bwd_diagnostics
    gradients = execution.intermediates
    kernel_parameters = (
        execution.dispatch,
        saved.dispatch_target_off * spec.hidden_size,
        _native_source(grad_output),
        saved.dispatch_src_off * spec.hidden_size,
        saved.dispatch_size * spec.hidden_size,
        saved.activation,
        gradients.grad_weight2,
        saved.weight2,
        gradients.act_grad,
        saved.up_proj,
        gradients.swiglu_grad,
        saved.weight1,
        gradients.gate_dx,
        execution.grad_x,
        saved.combine_target_off * spec.hidden_size,
        saved.combine_src_off * spec.hidden_size,
        saved.combine_size * spec.hidden_size,
        saved.dispatch,
        gradients.grad_weight1,
        saved.group_list,
    )
    if not diagnostics.is_active:
        record_plan_stream(plan, diagnostics.runtime_config)
        multicore_ops.mega_moe_grad(
            *kernel_parameters,
            plan.act_grad_tiling,
            plan.gate_grad_tiling,
            plan.w1_grad_tiling,
            plan.w2_grad_tiling,
            plan.swiglu_grad_tiling,
            execution.gmm_workspace,
            execution.swiglu_workspace,
            _split_runtime(
                diagnostics.runtime_config,
                pool,
                saved.weight1.shape[0],
                backward=True,
                gradient_return=gradient_return,
                home_only=pool is None and bool(spec.replica_slots_per_rank),
                cache=execution.runtime_cache,
                task_prefix=execution.task_prefix,
                prefix_cache=execution.prefix_cache,
            ),
            execution.event_counters,
            spec.rank_id,
            spec.ep_size,
            spec.num_experts,
            spec.hidden_size,
            getattr(spec, "plan_tokens", spec.local_num_tokens),
        )
        return
    with diagnostics.launch(
        kernel_parameters, direction="backward", fallback_event_counters=execution.event_counters
    ) as diagnostic_call:
        record_plan_stream(plan, diagnostic_call.runtime_config)
        multicore_ops.mega_moe_grad_with_diagnostic_buffer(
            *kernel_parameters,
            plan.act_grad_tiling,
            plan.gate_grad_tiling,
            plan.w1_grad_tiling,
            plan.w2_grad_tiling,
            plan.swiglu_grad_tiling,
            execution.gmm_workspace,
            execution.swiglu_workspace,
            _split_runtime(
                diagnostic_call.runtime_config,
                pool,
                saved.weight1.shape[0],
                backward=True,
                gradient_return=gradient_return,
                home_only=pool is None and bool(spec.replica_slots_per_rank),
                task_prefix=None if diagnostic_call.preserves_task_queues else execution.task_prefix,
            ),
            diagnostic_call.event_counters,
            diagnostic_call.diagnostic_buffer,
            spec.rank_id,
            spec.ep_size,
            spec.num_experts,
            spec.hidden_size,
            getattr(spec, "plan_tokens", spec.local_num_tokens),
        )


def _native_source(tensor: torch.Tensor) -> torch.Tensor:
    """Keep a non-null ABI pointer and preserve empty symmetric source storage."""
    if tensor.shape[0] == 0:
        row_elements = tensor.shape[1]
        required_bytes = (tensor.storage_offset() + row_elements) * tensor.element_size()
        if tensor.untyped_storage().nbytes() >= required_bytes:
            # Pull derives remote addresses from this local symmetric base even on empty ranks.
            return tensor.as_strided((1, row_elements), (row_elements, 1))
        return tensor.new_empty((1, tensor.shape[1]))
    return tensor.contiguous()


def _has_replica_transfers(route: ReplicaRoute | None) -> bool:
    """Distinguish a guest execution from a reserved but unused physical budget."""
    return route is not None and bool(route.plan.transfers)


def _claim_forward_workspace(workspace, spec, rows, claimed):
    """Validate ownership before any consumer can borrow directional buffers."""
    if claimed:
        if not workspace.in_use:
            raise RuntimeError("MegaMoe forward requires the caller-held workspace lease.")
        return
    if not spec.auto_tokens:
        workspace.ensure(spec, rows.dtype, rows.device)
    workspace.claim()


def _with_task_prefix(execution, route, prefixes):
    """Use an invocation's immutable mapping while preserving physical strides."""
    if not prefixes:
        return execution
    slots = None if route is None else route.plan.slot_to_logical[route.rank]
    prefix = select_guest_task_prefix(
        prefixes, slots, all_slot_to_logical=None if route is None else route.plan.slot_to_logical
    )
    return replace(execution, task_prefix=prefix)


def _backward_pool(ctx, saved, provider, leases, has_transfers):
    """Lease guest weights only after Torch has validated saved parameter versions."""
    if not has_transfers:
        return None
    generation = ctx.replica_weight_generation
    if isinstance(generation, _ReplicaWeightGeneration):
        generation = generation.value
    return leases.enter_context(
        prefetch_weights(
            (saved.weight1, saved.weight2),
            ctx.replica_route,
            backward=True,
            provider=provider,
            overlap=True,
            reuse_generation=generation,
            _overwrite_gradients=getattr(provider, "kernel_gradients", False),
        )
    )


def _empty_owner_experts(ctx, saved, has_transfers):
    """Clear every owner partial that a zero-row expert leaves unwritten."""
    if not has_transfers:
        return getattr(ctx, "empty_owner_experts", None)
    counts = ctx.replica_route.plan.destination_counts[ctx.replica_route.rank][: saved.weight1.shape[0]]
    return tuple(expert for expert, rows in enumerate(counts) if not rows)


def _prepare_gradient_return(ctx, execution, provider, pool, enabled, w13):
    """Select proven projection fences or retain the ordinary late-return path."""
    if not enabled:
        return None
    return prepare_kernel_gradient_return(
        ctx.plan,
        ctx.replica_route,
        provider,
        execution.intermediates.grad_weight2,
        None if pool is None else pool.gradients[1],
        gradient_w13=execution.intermediates.grad_weight1 if w13 else None,
        guest_w13=None if pool is None else pool.gradients[0],
    )


def _complete_gradient_return(ctx, execution, provider, pool, early_return, has_transfers):
    """Consume every FP32 partial exactly once, including the projection fallback."""
    first, second = execution.intermediates.grad_weight1, execution.intermediates.grad_weight2
    if provider is not None and hasattr(provider, "last_kernel_gradient_matrices"):
        provider.last_kernel_gradient_matrices = () if early_return is None else early_return.matrices
    if has_transfers:
        if early_return is None:
            first, second = return_gradients((first, second), ctx.replica_route, pool.gradients, provider, consume=True)
        elif 0 not in early_return.matrices:
            (first,) = return_gradients((first,), ctx.replica_route, pool.gradients[:1], provider, consume=True)
    return replace(execution, dispatch=None, intermediates=_BackwardIntermediates(first, second, None, None, None))


class _MegaMoeFunction(torch.autograd.Function):
    """Autograd bridge for the in-place forward/backward custom ops."""

    @staticmethod
    def forward(
        ctx: Any,
        routed_tokens: Any,
        weight1: Any,
        weight2: Any,
        route: RouteMetadata,
        plan: MegaMoePlan,
        workspace: MegaMoeWorkspace,
        permutation: tuple[Any, Any, Any] | None,
        topk_weights: torch.Tensor | None,
        workspace_claimed: bool,
    ) -> Any:
        """Launch the legacy forward op and save owned backward inputs.

        Args:
            ctx: Autograd context for saved backward state.
            routed_tokens: Expert-major token rows.
            weight1: Local gate and up-projection weights.
            weight2: Local down-projection weights.
            route: Route offsets, sizes and group metadata.
            plan: Shape-specific forward and backward descriptors.
            workspace: Reusable directional execution buffers.
            permutation: Optional routed rows, expert IDs and inverse mapping
                when the first tensor argument contains original token rows.
            topk_weights: Optional Router weights for pull source-output gradients.
            workspace_claimed: Whether the caller owns the forward workspace lease.

        Returns:
            Owned expert-major output rows.
        """
        spec = plan.spec
        if topk_weights is not None and (spec.dispatch_mode != "pull" or permutation is None):
            raise ValueError("Integrated output unpermutation requires pull dispatch and an input permutation.")
        metadata = route
        ctx.replica_route = getattr(route, "replica_route", None)
        home_weights = (weight1, weight2)
        permutation_inputs = ()
        if permutation is not None:
            routed_tokens, _, unpermute_mapping = permutation
            if ctx.needs_input_grad[0] or topk_weights is not None:
                permutation_inputs = (unpermute_mapping,)
        ctx.source_rows = routed_tokens.shape[0]
        ctx.token_rows = ctx.source_rows // spec.top_k
        ctx.has_permutation = permutation is not None
        ctx.has_unpermute = topk_weights is not None
        _claim_forward_workspace(workspace, spec, routed_tokens, workspace_claimed)
        has_replica_transfers = _has_replica_transfers(ctx.replica_route)
        execution = None
        leases = ExitStack()
        try:
            provider = workspace.replica_provider
            pool = (
                None
                if not has_replica_transfers
                else leases.enter_context(
                    prefetch_weights(home_weights, ctx.replica_route, provider=provider, overlap=True)
                )
            )
            generation = None if pool is None else getattr(pool, "weight_generation", None)
            ctx.replica_weight_generation = _remember_replica_generation(generation, ctx.replica_route, workspace)
            capacity = metadata.expert_capacity
            execution = _prepare_forward_execution(workspace, plan, routed_tokens, capacity)
            execution = _with_task_prefix(execution, ctx.replica_route, plan.fwd_task_prefixes)
            _launch_forward_kernel(plan, metadata, execution.source, weight1, weight2, execution, pool)
            output = execution.combine[: ctx.source_rows].clone()
            if spec.dispatch_mode == "pull":
                saved_dispatch = execution.dispatch
            else:
                execution.intermediates.down_proj.copy_(execution.dispatch[:capacity])
                saved_dispatch = execution.intermediates.down_proj
            if ctx.has_unpermute:
                probs = topk_weights.float().contiguous()
                ctx.topk_dtype = topk_weights.dtype
                permutation_inputs += (output, probs)
                output = (
                    load_torch_npu().npu_moe_token_unpermute(output, unpermute_mapping, probs=probs)
                    if ctx.source_rows
                    else output[:0]
                )
            _save_forward_state(
                ctx,
                plan,
                workspace,
                saved_dispatch,
                execution.intermediates.up_proj,
                execution.intermediates.activation,
                home_weights[0],
                home_weights[1],
                metadata,
                permutation_inputs,
            )
            return output
        finally:
            try:
                leases.close()
            finally:
                if not workspace_claimed:
                    workspace.release()

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> tuple[Any, ...]:
        """Launch the legacy backward op into zero-safe gradient buffers.

        Args:
            ctx: Autograd context populated by ``forward``.
            grad_output: Expert-major output gradient rows.

        Returns:
            Gradients aligned with the differentiable forward inputs.
        """
        plan = ctx.plan
        workspace = ctx.workspace
        saved_tensors = ctx.saved_tensors
        ctx.maybe_clear_saved_tensors()
        saved = _saved_backward_state(saved_tensors)
        permutation_inputs = saved_tensors[12:]
        del saved_tensors
        workspace.claim()
        has_replica_transfers = _has_replica_transfers(ctx.replica_route)
        execution = None
        leases = ExitStack()
        try:
            provider = workspace.replica_provider
            pool = _backward_pool(ctx, saved, provider, leases, has_replica_transfers)
            source, grad_topk_weights = _stage_backward_source(ctx, grad_output, permutation_inputs)
            permutation_inputs = permutation_inputs[:1]
            kernel_return = has_replica_transfers and getattr(provider, "kernel_gradients", False)
            kernel_w13 = kernel_return and bool(plan.replica_w13_events)
            execution = _prepare_backward_execution(
                workspace,
                plan,
                saved,
                grad_output,
                has_replica_transfers=has_replica_transfers,
                overlap_w13=kernel_w13,
                empty_experts=_empty_owner_experts(ctx, saved, has_replica_transfers),
            )
            prefixes = (
                plan.bwd_task_prefixes
                if execution.diagnostics is plan.bwd_diagnostics
                else plan.bwd_no_replica_task_prefixes
            )
            execution = _with_task_prefix(execution, ctx.replica_route, prefixes)
            early_return = _prepare_gradient_return(ctx, execution, provider, pool, kernel_return, kernel_w13)
            _launch_backward_kernel(
                plan, saved, source, execution, pool, None if early_return is None else early_return.metadata
            )
            execution = _complete_gradient_return(ctx, execution, provider, pool, early_return, has_replica_transfers)
            grad_input = _restore_input_gradient(ctx, execution.grad_x, permutation_inputs)
            grad_weight1 = execution.intermediates.grad_weight1
            grad_weight2 = execution.intermediates.grad_weight2
            _release_completed_graph(ctx, workspace)
            return (grad_input, grad_weight1, grad_weight2, None, None, None, None, grad_topk_weights, None)
        finally:
            try:
                leases.close()
            finally:
                workspace.release()


def execute_mega_moe(
    routed_tokens: Any,
    weight1: Any,
    weight2: Any,
    route: RouteMetadata,
    plan: MegaMoePlan,
    workspace: MegaMoeWorkspace,
) -> Any:
    """Execute expert-major rows through the differentiable legacy kernels.

    Args:
        routed_tokens: Expert-major token rows.
        weight1: Local gate and up-projection weights in the plan-selected layout.
        weight2: Local down-projection weights in the plan-selected layout.
        route: Route offsets, sizes and group metadata.
        plan: Shape-specific forward and backward descriptors.
        workspace: Reusable directional execution buffers.

    Returns:
        Expert-major output rows with independent storage.
    """
    # Historical input-output views are materialized outside Function.forward so
    # gradients reach the caller's original weights. Output-input storage is native.
    return _MegaMoeFunction.apply(
        routed_tokens,
        _native_weight_operand(weight1, plan),
        _native_weight_operand(weight2, plan),
        route,
        plan,
        workspace,
        None,
        None,
        False,
    )


def execute_mega_moe_with_permutation(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    route: PreparedTopKRoute,
    plan: MegaMoePlan,
    workspace: MegaMoeWorkspace,
    *,
    topk_weights: torch.Tensor | None = None,
    workspace_claimed: bool = False,
) -> torch.Tensor:
    """Include input permutation backward within the workspace lease.

    Args:
        hidden_states: Original flattened token rows.
        topk_ids: Expert IDs used to prepare the routed rows.
        weight1: Local gate and up-projection weights in the plan-selected layout.
        weight2: Local down-projection weights in the plan-selected layout.
        route: Rows and metadata prepared without recording autograd operations.
        plan: Shape-specific native descriptors.
        workspace: Reusable communication buffers.
        topk_weights: Optional Router probabilities for integrated pull output unpermutation.
        workspace_claimed: Borrow a caller-held forward lease without releasing it.
            Backward always acquires its own lease on the original workspace.

    Returns:
        Owned expert-major rows, or token-major rows when topk_weights is provided,
        differentiable with respect to input tokens, expert weights and Router probabilities.
    """
    return _MegaMoeFunction.apply(
        hidden_states,
        _native_weight_operand(weight1, plan),
        _native_weight_operand(weight2, plan),
        route.metadata,
        plan,
        workspace,
        (route.routed_tokens, topk_ids, route.unpermute_mapping),
        topk_weights,
        workspace_claimed,
    )
