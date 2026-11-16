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

import os
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

from .normal_trace import launch_normal_forward, require_normal_trace_mode
from .probability import PROBABILITY_PLANES, probability_stride

from .inactive_tasks import GuestTaskPrefix, apply_guest_task_prefix, select_guest_task_prefix
from .kernel_gradients import prepare_kernel_gradient_return
from .local_gradients import use_local_gradient_bf16
from .plan import MegaMoePlan, record_plan_stream
from .route import _peer_owner_flags, PreparedTopKRoute, RouteMetadata, restore_topk_output
from .spec import MegaMoeSpec, _route_metadata_rows
from .workspace import MegaMoeWorkspace
from .shared import pointer_table, shared_forward_buffers, shared_backward_buffers
from .indexed_backward import publish_backward_route


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
            # A later retained-graph replay has no original registry to refresh.
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
    shared_ptrs: Any = None


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
    shared_ptrs: Any = None


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


def _kernel_workspace(workspace: MegaMoeWorkspace, spec: Any) -> torch.Tensor:
    """Use the reserved native scratch port for declared probability planes."""
    if getattr(spec, "apply_probs_before_fc2", False) is True:
        return workspace.probabilities(spec).raw
    return _workspace_tensor(workspace.gmm_workspace, "gmm_workspace")


def _stage_probabilities(
    workspace: MegaMoeWorkspace, mapping: torch.Tensor, probs: torch.Tensor, spec: Any,
) -> None:
    """Publish source probabilities in the current expert-major permutation."""
    if probs.numel():
        # Preserve local input bounds: the upstream scatter tiling depends on this shape.
        load_torch_npu().npu_scatter_nd_update_(
            workspace.probabilities(spec).slice(probs.numel()),
            mapping.view(-1, 1), probs.reshape(-1))


def _restore_unweighted_output(rows: torch.Tensor, mapping: torch.Tensor, top_k: int) -> torch.Tensor:
    """The transpose of token expansion sums Top-K rows without weighting again."""
    return multicore_ops.moe_token_permute_grad(rows, mapping, rows.shape[0] // top_k, top_k)


def _stage_backward_source(ctx: Any, grad_output: Any, permutation_inputs: tuple[Any, ...]) -> tuple[Any, Any]:
    """Stage backward tokens and probabilities before expert scratch allocation."""
    if getattr(ctx.plan.spec, "apply_probs_before_fc2", False) is True:
        mapping, probs = permutation_inputs
        source = (ctx.workspace.source_buffer[:ctx.source_rows]
                  if ctx.plan.spec.dispatch_mode == "pull" else grad_output.new_empty(
                      (ctx.source_rows, ctx.plan.spec.hidden_size)))
        if ctx.source_rows:
            # Consume the saved exact permutation instead of sorting expert IDs again.
            multicore_ops.moe_token_permute_from_mapping_out(
                grad_output.contiguous(), mapping, probs, source,
                ctx.workspace.probabilities(ctx.plan.spec).slice(ctx.source_rows))
        return source, None
    if getattr(ctx.plan.spec, "dispatch_mode", "push") == "push":
        return grad_output, None
    source = _workspace_tensor(ctx.workspace.source_buffer, "source_buffer")[: ctx.source_rows]
    if not ctx.has_unpermute:
        source.copy_(grad_output)
        return source, None
    if getattr(ctx.plan.spec, "backward_ingress_mode", "legacy") == "indexed":
        mapping, expert_output, probs, sorted_slots = permutation_inputs
        return publish_backward_route(ctx.workspace, grad_output, mapping, expert_output, probs, sorted_slots,
                                      needs_probability_grad=ctx.needs_input_grad[7])
    mapping, expert_output, probs = permutation_inputs
    grad_probs = torch.empty_like(probs)
    if ctx.source_rows:
        multicore_ops.mega_moe_unpermute_grad_out(expert_output, grad_output, mapping, probs, source, grad_probs)
    return source, grad_probs if ctx.needs_input_grad[7] else None


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


def _empty_expert_spans(experts: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Coalesce adjacent cold matrices without touching a live expert's output."""
    spans = []
    for expert in experts:
        if spans and spans[-1][1] == expert:
            spans[-1] = spans[-1][0], expert + 1
        else:
            spans.append((expert, expert + 1))
    return tuple(spans)


def _allocate_backward_intermediates(
    spec: Any,
    capacity: int,
    grad_output: Any,
    weight1: Any,
    weight2: Any,
    reusable_dispatch: Any | None = None,
    *, weight_grad_dtype: torch.dtype | None = None,
    group_list: torch.Tensor | None = None,
    all_experts_active: bool = False,
    empty_experts: tuple[int, ...] | None = None,
    local_bf16: bool = False,
) -> _BackwardIntermediates:
    """Allocate overwritten activations and zero-safe expert gradients."""
    if getattr(spec, "replica_slots_per_rank", 0) or empty_experts is not None:
        gradient_dtype = (torch.float32 if getattr(spec, "replica_slots_per_rank", 0) and not local_bf16
                          else weight_grad_dtype or weight2.dtype)
        clear_all = empty_experts is None or len(empty_experts) == weight1.shape[0]
        allocate = torch.zeros_like if clear_all else torch.empty_like
        grad_weight1 = allocate(weight1, dtype=gradient_dtype)
        grad_weight2 = allocate(weight2, dtype=gradient_dtype)
        if not clear_all:
            for first, last in _empty_expert_spans(empty_experts):
                grad_weight1[first:last].zero_()
                grad_weight2[first:last].zero_()
    elif group_list is None:
        # Without count metadata, cold rows cannot be identified; initialize every expert.
        grad_weight1 = torch.zeros_like(weight1, dtype=weight_grad_dtype)
        grad_weight2 = torch.zeros_like(weight2, dtype=weight_grad_dtype)
    else:
        grad_weight1 = torch.empty_like(weight1, dtype=weight_grad_dtype, memory_format=torch.contiguous_format)
        grad_weight2 = torch.empty_like(weight2, dtype=weight_grad_dtype, memory_format=torch.contiguous_format)
        if not all_experts_active:
            multicore_ops.zero_empty_expert_gradients(grad_weight1, grad_weight2, group_list)
    act_grad = torch.empty(
        (capacity, spec.intermediate_size),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    swiglu_grad = torch.empty(
        (capacity, spec.intermediate_size * 2), dtype=grad_output.dtype, device=grad_output.device
    )
    gate_dx = (
        reusable_dispatch
        if reusable_dispatch is not None
        else torch.empty((capacity, spec.hidden_size), dtype=grad_output.dtype, device=grad_output.device)
    )
    return _BackwardIntermediates(grad_weight1, grad_weight2, act_grad, swiglu_grad, gate_dx)


def _restore_weight_gradient_layout(spec: Any, gradient: torch.Tensor) -> torch.Tensor:
    """Return the gradient already written in the caller-selected layout."""
    # The native gradient tiling writes the caller's EIO or EOI layout directly.
    return gradient


def _empty_backward_experts(ctx: Any, weight: torch.Tensor) -> tuple[int, ...] | None:
    """Reuse exact host counts for A5 BF16 complete-K gradient overwrites."""
    route = ctx.replica_route
    if route is not None and route.plan.transfers:
        return tuple(expert for expert, rows in enumerate(
            route.plan.destination_counts[route.rank][:weight.shape[0]]) if not rows)
    spec = ctx.plan.spec
    if (getattr(spec, "num_cube_cores", 0) != 32 or spec.replica_slots_per_rank
            or spec.expert_weight_layout != "EIO" or weight.dtype != torch.bfloat16):
        return None
    # Small dynamic routes already own these CPU counts. Unknown or device-only
    # routes keep full initialization instead of adding another readback.
    return getattr(ctx, "empty_experts", None)


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
        gmm_workspace=_kernel_workspace(workspace, plan.spec),
        intermediates=_allocate_forward_intermediates(
            plan.spec,
            capacity,
            routed_tokens,
            dispatch,
        ),
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
    all_experts_active: bool = False,

    local_bf16: bool = False,
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
        gmm_workspace=(_workspace_tensor(workspace.backward_route_metadata, "backward_route_metadata").view(torch.uint8)
                       if getattr(plan.spec, "backward_ingress_mode", "legacy") == "indexed" else
                       _kernel_workspace(workspace, plan.spec)),
        swiglu_workspace=_workspace_tensor(
            workspace.swiglu_grad_workspace,
            "swiglu_grad_workspace",
        ),
        intermediates=_allocate_backward_intermediates(
            plan.spec,
            saved.dispatch.shape[0],
            grad_output,
            saved.weight1,
            saved.weight2,
            dispatch[:saved.dispatch.shape[0]] if reuse_dispatch else None,
            weight_grad_dtype=plan.weight_grad_dtype,
            group_list=saved.group_list,
            all_experts_active=all_experts_active,
            empty_experts=empty_experts,
            local_bf16=local_bf16,
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


def _restore_forward_output(
    ctx: Any, spec: MegaMoeSpec, execution: _ForwardExecution,
    topk_weights: torch.Tensor | None, unpermute_mapping: torch.Tensor | None,
    permute_src_index: torch.Tensor | None, permutation_inputs: tuple[Any, ...],
    save_backward: bool,
) -> tuple[Any, Any, tuple[Any, ...]]:
    """Consume borrowed receive rows before releasing the forward workspace.

    Args:
        ctx: Autograd context containing per-call permutation requirements.
        spec: Current transport and ingress configuration.
        execution: Borrowed receive storage and owned expert intermediates.
        topk_weights: Optional Router probabilities for integrated unpermutation.
        unpermute_mapping: Inverse mapping for token-major output restoration.
        permute_src_index: Source slots retained by indexed backward ingress.
        permutation_inputs: Existing per-call permutation state.
        save_backward: Whether backward will read expert-major output rows.

    Returns:
        Owned final output, saved dispatch storage, and backward permutation state.
    """
    output = execution.combine[:ctx.source_rows]
    if save_backward or not ctx.has_unpermute or not ctx.source_rows:
        output = output.clone()
    if spec.dispatch_mode == "pull" or not save_backward:
        saved_dispatch = execution.dispatch
    else:
        # Combine consumed down_proj before the symmetric dispatch storage is reused.
        execution.intermediates.down_proj.copy_(execution.dispatch[:execution.intermediates.down_proj.shape[0]])
        saved_dispatch = execution.intermediates.down_proj
    if ctx.has_unpermute:
        probs = topk_weights.float().contiguous()
        ctx.topk_dtype = topk_weights.dtype
        permutation_inputs += (output, probs)
        if getattr(spec, "backward_ingress_mode", "legacy") == "indexed":
            permutation_inputs += (permute_src_index,)
        output = (load_torch_npu().npu_moe_token_unpermute(output, unpermute_mapping, probs=probs)
                  if ctx.source_rows else output[:0])
    return output, saved_dispatch, permutation_inputs


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
    permutation_inputs: tuple[Any, ...],
    output: Any,
    topk_weights: torch.Tensor | None,
    *, sorted_slots: torch.Tensor | None = None, extra_saved: tuple[Any, ...] = (),
) -> Any:
    """Restore the owned output and save its backward state in the existing call."""
    if ctx.has_unpermute:
        spec = plan.spec
        if not getattr(spec, "apply_probs_before_fc2", False) is True:
            probs = topk_weights.float().contiguous()
            permutation_inputs += (output, probs)
            if sorted_slots is not None:
                permutation_inputs += (sorted_slots,)
        ctx.topk_dtype = topk_weights.dtype
        if getattr(spec, "apply_probs_before_fc2", False) is True:
            output = _restore_unweighted_output(output, permutation_inputs[0], spec.top_k)
        else:
            output = (load_torch_npu().npu_moe_token_unpermute(output, permutation_inputs[0], probs=probs)
                      if ctx.source_rows else output[:0])
    ctx.plan = plan
    ctx.all_experts_active = metadata.all_experts_active
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
        *extra_saved,
    )
    workspace.track_graph(ctx)
    return output


_CAPTURED_WEIGHT_GRADS: tuple[torch.Tensor, torch.Tensor] | None = None


def _release_completed_graph(ctx: Any, workspace: MegaMoeWorkspace) -> None:
    """Keep retained graphs alive; older engines conservatively wait for context GC."""
    # The engine's retention flag avoids unpacking saved tensors or guessing from output references.
    keep_graph = getattr(
        torch._C._autograd, "_get_current_graph_task_keep_graph", None
    )  # pylint: disable=protected-access
    if keep_graph is not None and not keep_graph():
        workspace.release_graph(ctx)


def _diagnostic_parameters(
    plan: MegaMoePlan, parameters: tuple[Any, ...], workspace: torch.Tensor,
    tilings: tuple[Any, ...] = (),
) -> tuple[Any, ...]:
    """Expose the existing scalar scratch at its real native ABI position."""
    if not getattr(plan.spec, "apply_probs_before_fc2", False) is True:
        return parameters
    stride = probability_stride(plan.spec)
    planes = workspace.view(torch.float32).reshape(-1)[:PROBABILITY_PLANES * stride].view(
        PROBABILITY_PLANES, stride)
    return (*parameters, *tilings, planes)

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
def _patch_forward_row_thresholds(plan: MegaMoePlan, metadata: RouteMetadata, runtime_config: Any) -> None:
    """Update the actual ordinary/profile/dump image under the same workspace lease."""
    offset = getattr(plan, "row_credit_element_offset", None)
    if offset is None:
        return
    if metadata.expert_row_counts is None:
        raise RuntimeError("Owner overlap requires per-expert receive row counts")
    runtime_config.view(torch.int32)[offset:offset + plan.spec.local_experts].copy_(metadata.expert_row_counts)


def _reset_forward_tile_credits(spec: MegaMoeSpec, storage: torch.Tensor) -> None:
    """Reset only the local readiness table while the caller owns the forward lease."""
    if spec.forward_ingress_mode != "peer_tile":
        return
    prefix_rows = _route_metadata_rows(spec.routed_slots, spec.num_experts, spec.ep_size, "peer_token")
    storage.view(-1)[prefix_rows * 4:].zero_()



def _saved_dispatch_launch(plan):
    """Capture one coherent runtime/prefix choice for this diagnostic context."""
    enabled = getattr(plan.spec, "_fuse_saved_dispatch", False) is True and not plan.fwd_diagnostics.is_active
    return enabled, plan.fwd_saved_task_prefixes if enabled else plan.fwd_task_prefixes


def _launch_forward_kernel(
    plan: MegaMoePlan,
    metadata: RouteMetadata,
    routed_tokens: Any,
    weight1: Any,
    weight2: Any,
    execution: _ForwardExecution,
    pool: Any = None,
    *, stream: Any = None, save_dispatch: bool = True,
) -> None:
    """Launch the internal forward ABI with prepared buffers and metadata."""
    spec = plan.spec
    require_normal_trace_mode(plan.fwd_diagnostics.is_active)
    kernel_parameters = (
        execution.dispatch,
        metadata.dispatch_target_off,
        _native_source(routed_tokens),
        metadata.dispatch_src_off,
        metadata.dispatch_size,
        weight1,
        metadata.group_list,
        execution.intermediates.up_proj,
        execution.intermediates.activation,
        weight2,
        metadata.group_list,
        execution.intermediates.down_proj,
        execution.combine,
        metadata.combine_target_off,
        metadata.combine_src_off,
        metadata.combine_size,
    )
    if not plan.fwd_diagnostics.is_active:
        runtime_config = (plan.fwd_saved_dispatch_config
                          if save_dispatch and getattr(spec, "_fuse_saved_dispatch", False) is True else plan.fwd_runtime_config)
        _patch_forward_row_thresholds(plan, metadata, runtime_config)
        record_plan_stream(plan, runtime_config, stream=stream)
        launch_normal_forward((
            *kernel_parameters,
            execution.gmm_workspace,
            plan.up_proj_tiling,
            plan.swiglu_tiling,
            plan.down_proj_tiling,
            _split_runtime(
                runtime_config,
                pool,
                weight1.shape[0],
                home_only=pool is None and bool(spec.replica_slots_per_rank),
                cache=execution.runtime_cache,
                task_prefix=execution.task_prefix,
                prefix_cache=execution.prefix_cache,
            ),
            execution.event_counters,
        ), (
            spec.rank_id,
            spec.ep_size,
            spec.num_experts,
            spec.hidden_size,
            getattr(spec, "plan_tokens", spec.local_num_tokens),
        ), num_workers=2 * spec.num_cube_cores,
           **({"shared_ptrs": execution.shared_ptrs, "shared_runtime": plan.shared_forward_runtime}
              if execution.shared_ptrs is not None else {}))
        return
    diagnostic_parameters = _diagnostic_parameters(plan, kernel_parameters, execution.gmm_workspace)
    if execution.shared_ptrs is not None:
        diagnostic_parameters = (*kernel_parameters, execution.gmm_workspace, plan.up_proj_tiling,
                                 plan.swiglu_tiling, plan.down_proj_tiling, plan.fwd_runtime_config,
                                 None, None, None, execution.event_counters, None,
                                 execution.shared_ptrs, plan.shared_forward_runtime)
    with plan.fwd_diagnostics.launch(
        diagnostic_parameters,
        direction="forward",
        fallback_event_counters=execution.event_counters,
    ) as diagnostic_call:
        _patch_forward_row_thresholds(plan, metadata, diagnostic_call.runtime_config)
        record_plan_stream(plan, diagnostic_call.runtime_config, stream=stream)
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
            **({"shared_ptrs": execution.shared_ptrs, "shared_runtime": plan.shared_forward_runtime}
               if execution.shared_ptrs is not None else {}),
        )


def _launch_backward_kernel(
    plan: MegaMoePlan,
    saved: _SavedBackwardState,
    grad_output: Any,
    execution: _BackwardExecution,
    pool: Any = None,
    gradient_return: torch.Tensor | None = None,
    *, stream: Any = None,
) -> None:
    """Launch the internal backward ABI with prepared buffers and metadata."""
    spec = plan.spec
    diagnostics = execution.diagnostics or plan.bwd_diagnostics
    if getattr(spec, "backward_ingress_mode", "legacy") == "indexed":
        multicore_ops.require_indexed_backward_support()
    gradients = execution.intermediates
    kernel_parameters = (
        execution.dispatch,
        saved.dispatch_target_off,
        _native_source(grad_output),
        saved.dispatch_src_off,
        saved.dispatch_size,
        saved.activation,
        gradients.grad_weight2,
        saved.weight2,
        gradients.act_grad,
        saved.up_proj,
        gradients.swiglu_grad,
        saved.weight1,
        gradients.gate_dx,
        execution.grad_x,
        saved.combine_target_off,
        saved.combine_src_off,
        saved.combine_size,
        saved.dispatch,
        gradients.grad_weight1,
        saved.group_list,
    )
    if not diagnostics.is_active:
        record_plan_stream(plan, diagnostics.runtime_config, stream=stream)
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
            **({"shared_ptrs": execution.shared_ptrs, "shared_runtime": plan.shared_backward_runtime}
               if execution.shared_ptrs is not None else {}),
        )
        return

    diagnostic_parameters = _diagnostic_parameters(plan, kernel_parameters, execution.gmm_workspace, (
        plan.act_grad_tiling, plan.gate_grad_tiling, plan.w1_grad_tiling,
        plan.w2_grad_tiling, plan.swiglu_grad_tiling))
    if execution.shared_ptrs is not None:
        diagnostic_parameters = (*kernel_parameters, plan.act_grad_tiling, plan.gate_grad_tiling,
                                 plan.w1_grad_tiling, plan.w2_grad_tiling, plan.swiglu_grad_tiling,
                                 execution.gmm_workspace, execution.swiglu_workspace, plan.bwd_runtime_config,
                                 None, None, None, execution.event_counters, None,
                                 execution.shared_ptrs, plan.shared_backward_runtime)
    with diagnostics.launch(
        diagnostic_parameters,
        direction="backward",
        fallback_event_counters=execution.event_counters,
    ) as diagnostic_call:
        record_plan_stream(plan, diagnostic_call.runtime_config, stream=stream)
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
            **({"shared_ptrs": execution.shared_ptrs, "shared_runtime": plan.shared_backward_runtime}
               if execution.shared_ptrs is not None else {}),
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
        known = getattr(ctx, "empty_owner_experts", None)
        return known if known is not None else _empty_backward_experts(ctx, saved.weight1)
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


def _validate_integrated_output(spec: Any, permutation: Any, topk_weights: Any) -> None:
    """Require the mapping and probabilities consumed by native integrated output."""
    if topk_weights is not None and (
        permutation is None or (spec.dispatch_mode != "pull" and not getattr(spec, "apply_probs_before_fc2", False) is True)
    ):
        raise ValueError("Integrated output unpermutation requires pull dispatch or pre-FC2 weighting, "
                         "and an input permutation.")
    if getattr(spec, "apply_probs_before_fc2", False) is True and topk_weights is None:
        raise ValueError("Pre-FC2 weighting requires the original Top-K probabilities and permutation.")



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
        permutation: tuple[Any, ...] | None,
        topk_weights: torch.Tensor | None,
        workspace_claimed: bool,
        shared_weight1: torch.Tensor | None,
        shared_weight2: torch.Tensor | None,
        save_backward: bool,
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
            shared_weight1: Optional current shared gate/up weight.
            shared_weight2: Optional current shared down-projection weight.
            save_backward: Whether the caller's active autograd graph needs saved intermediates.

        Returns:
            Owned token-major rows with integrated unpermutation, otherwise expert-major rows.
        """
        spec = plan.spec
        _validate_integrated_output(spec, permutation, topk_weights)
        metadata = route.for_kernel(spec.hidden_size)
        ctx.replica_route = getattr(route, "replica_route", None)
        ctx.has_shared = shared_weight1 is not None
        ctx.shared_backward = ctx.has_shared and getattr(spec, "backward_shared_expert", False)
        shared_input = routed_tokens
        ctx.empty_experts = getattr(metadata, "empty_experts", None)
        home_weights = (weight1, weight2)
        permutation_inputs = ()
        indexed_ingress = getattr(spec, "forward_ingress_mode", "legacy") != "legacy"
        permute_src_index = None
        unpermute_mapping = None
        if permutation is not None:
            prepared_rows, _, unpermute_mapping = permutation[:3]
            if indexed_ingress:
                if len(permutation) < 4 or permutation[3] is None or prepared_rows is not None:
                    raise ValueError("Indexed ingress requires scalar route indices and original token rows")
                permute_src_index = permutation[3]
            else:
                routed_tokens = prepared_rows
            if ctx.needs_input_grad[0] or topk_weights is not None:
                permutation_inputs = (unpermute_mapping,)
        elif indexed_ingress:
            raise ValueError("Indexed ingress requires a prepared Top-K route")
        ctx.source_rows = routed_tokens.shape[0] * (spec.top_k if indexed_ingress else 1)
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
            execution = _prepare_forward_execution(
                workspace,
                plan,
                routed_tokens,
                capacity,
            )
            saved_in_kernel, prefixes = _saved_dispatch_launch(plan)
            saved_in_kernel = saved_in_kernel and save_backward
            if not save_backward:
                prefixes = plan.fwd_task_prefixes
            execution = _with_task_prefix(execution, ctx.replica_route, prefixes)
            if getattr(spec, "apply_probs_before_fc2", False) is True:
                probs = topk_weights.float().contiguous()
                _stage_probabilities(workspace, unpermute_mapping, probs, spec)
                permutation_inputs += (probs,)
            shared_saved = ()
            shared_output = None
            if ctx.has_shared:
                if getattr(spec, "forward_shared_expert", True):
                    shared_up, shared_activation, shared_output = shared_forward_buffers(
                        shared_input, spec.shared_intermediate_size)
                    execution = replace(execution, shared_ptrs=pointer_table(
                        (shared_input, shared_weight1, shared_weight2, shared_up, shared_activation,
                         shared_output, workspace.gmm_workspace), idle_vectors=True))
                else:
                    shared_up, shared_activation, shared_output = _native_shared_forward(
                        shared_input, shared_weight1, shared_weight2, spec.swiglu_limit)
                shared_saved = (shared_input, shared_weight1, shared_weight2, shared_up, shared_activation)
            if indexed_ingress:
                # Port 16 is unused by Blaze F2. Keep its read-only metadata in
                # symmetric storage until the all-rank source-read completion.
                metadata_storage = _workspace_tensor(workspace.route_metadata, "route_metadata")
                _reset_forward_tile_credits(spec, metadata_storage)
                metadata_buffer = metadata_storage[:ctx.source_rows]
                if spec.forward_ingress_mode in ("peer_token", "peer_tile"):
                    # Packed records are indexed by original slot. The gap up to
                    # NK*16 bytes preserves the common diagnostic-tail address.
                    packed = torch.stack((permutation[1].reshape(-1).to(torch.int32), permutation[4]), dim=1)
                    metadata_storage.view(-1)[:ctx.source_rows * 2].view(-1, 2).copy_(packed)
                else:
                    metadata_buffer[:, 0].copy_(permute_src_index)
                    if spec.forward_ingress_mode in ("peer_dedup", "peer_overlap"):
                        metadata_buffer[:, 2].copy_(permutation[1].reshape(-1))
                        metadata_buffer[:, 3].copy_(permutation[4])
                        if spec.forward_ingress_mode == "peer_overlap":
                            metadata_buffer[:, 1].copy_(_peer_owner_flags(
                                permutation[1], permute_src_index, spec.local_experts))
                execution = replace(execution, gmm_workspace=metadata_storage.view(torch.uint8))
            if getattr(plan, "ingress_diagnostics", None) is not None:
                ingress_plan = replace(plan, fwd_diagnostics=plan.ingress_diagnostics, ingress_diagnostics=None)
                _launch_forward_kernel(ingress_plan, metadata, execution.source, weight1, weight2, execution)
                # Same-stream reset follows the complete ingress kernel. Keep
                # persistent ready generations for the following full graph.
                execution = replace(execution, event_counters=workspace.prepare_event_counters(forward=True))
            _launch_forward_kernel(
                plan,
                metadata,
                execution.source,
                weight1,
                weight2,
                execution, pool, stream=workspace.lease_stream, save_dispatch=save_backward)
            output = execution.combine[:ctx.source_rows]
            if (getattr(spec, "apply_probs_before_fc2", False) is not True
                    and (save_backward or not ctx.has_unpermute or not ctx.source_rows)):
                output = output.clone()
            if spec.dispatch_mode == "pull":
                saved_dispatch = execution.dispatch
            else:
                if save_backward and not saved_in_kernel:
                    execution.intermediates.down_proj.copy_(execution.dispatch[:capacity])
                saved_dispatch = execution.intermediates.down_proj
            if save_backward:
                output = _save_forward_state(
                    ctx, plan, workspace, saved_dispatch, execution.intermediates.up_proj,
                    execution.intermediates.activation, weight1, weight2, metadata, permutation_inputs,
                    output, topk_weights,
                    sorted_slots=permute_src_index if getattr(spec, "backward_ingress_mode", "legacy") == "indexed" else None,
                    extra_saved=shared_saved if ctx.shared_backward else (),
                )
            elif ctx.has_unpermute:
                if getattr(spec, "apply_probs_before_fc2", False) is True:
                    output = _restore_unweighted_output(output, unpermute_mapping, spec.top_k)
                else:
                    probs = topk_weights.float().contiguous()
                    output = (load_torch_npu().npu_moe_token_unpermute(output, unpermute_mapping, probs=probs)
                              if ctx.source_rows else output[:0])
            if ctx.has_shared:
                ctx.routed_output_shape = tuple(output.shape)
                if ctx.shared_backward:
                    ctx.mark_non_differentiable(shared_up, shared_activation)
                else:
                    ctx.mark_non_differentiable(shared_output, shared_up, shared_activation)
                ctx.set_materialize_grads(False)
                return output, shared_output, shared_up, shared_activation
            return output
        finally:
            try:
                leases.close()
            finally:
                if not workspace_claimed:
                    workspace.release(stream=workspace.lease_stream)

    @staticmethod
    def backward(
        ctx: Any, grad_output: Any, grad_shared_output: Any = None,
        _grad_shared_up: Any = None, _grad_shared_activation: Any = None,
    ) -> tuple[Any, ...]:
        """Launch the legacy backward op into zero-safe gradient buffers.

        Args:
            ctx: Autograd context populated by ``forward``.
            grad_output: Expert-major output gradient rows.
            grad_shared_output: Token-major shared output gradient when shared backward is fused.
            _grad_shared_up: Unused non-differentiable shared preactivation.
            _grad_shared_activation: Unused non-differentiable shared activation.

        Returns:
            Gradients aligned with the differentiable forward inputs.
        """
        plan = ctx.plan
        workspace = ctx.workspace
        saved_tensors = ctx.saved_tensors
        ctx.maybe_clear_saved_tensors()
        saved = _saved_backward_state(saved_tensors)
        generation = ctx.replica_weight_generation
        if isinstance(generation, _ReplicaWeightGeneration):
            generation = generation.value
        shared_backward = getattr(ctx, "shared_backward", False) is True
        shared_saved = saved_tensors[-5:] if shared_backward else ()
        permutation_inputs = saved_tensors[12:-5] if shared_backward else saved_tensors[12:]
        del saved_tensors
        workspace.claim()
        has_replica_transfers = _has_replica_transfers(ctx.replica_route)
        execution = None
        leases = ExitStack()
        try:
            if grad_output is None:
                grad_output = saved.weight1.new_zeros(ctx.routed_output_shape)
            provider = workspace.replica_provider
            pool = _backward_pool(ctx, saved, provider, leases, has_replica_transfers)
            source, grad_topk_weights = _stage_backward_source(ctx, grad_output, permutation_inputs)
            saved_probabilities = permutation_inputs[-1] if getattr(plan.spec, "apply_probs_before_fc2", False) is True else None
            permutation_inputs = permutation_inputs[:1]
            kernel_return = has_replica_transfers and getattr(provider, "kernel_gradients", False)
            kernel_w13 = kernel_return and bool(plan.replica_w13_events)
            execution = _prepare_backward_execution(
                workspace,
                plan,
                saved,
                grad_output,
                all_experts_active=ctx.all_experts_active,

                has_replica_transfers=has_replica_transfers,
                overlap_w13=kernel_w13,
                empty_experts=_empty_owner_experts(ctx, saved, has_replica_transfers),
                local_bf16=use_local_gradient_bf16(plan, ctx.replica_route, provider, (saved.weight1, saved.weight2)),
            )
            prefixes = (
                plan.bwd_task_prefixes
                if execution.diagnostics is plan.bwd_diagnostics
                else plan.bwd_no_replica_task_prefixes
            )
            execution = _with_task_prefix(execution, ctx.replica_route, prefixes)
            shared_tensors = ()
            if shared_backward:
                if grad_shared_output is None:
                    grad_shared_output = torch.zeros_like(shared_saved[0])
                shared_tensors, pointers = shared_backward_buffers(
                    grad_shared_output, shared_saved, idle_vectors=plan.spec.num_cube_cores != 32,
                    native_activation_grad=plan.spec.num_cube_cores == 32, swiglu_limit=plan.spec.swiglu_limit,
                )
                execution = replace(execution, shared_ptrs=pointers)
            early_return = _prepare_gradient_return(ctx, execution, provider, pool, kernel_return, kernel_w13)
            _launch_backward_kernel(
                plan,
                saved,
                source,
                execution, pool, None if early_return is None else early_return.metadata,
                stream=workspace.lease_stream)
            execution = _complete_gradient_return(
                ctx, execution, provider, pool, early_return, has_replica_transfers)
            if getattr(plan.spec, "apply_probs_before_fc2", False) is True and ctx.needs_input_grad[7]:
                grad_topk_weights = (
                    workspace.probabilities(plan.spec).slice(ctx.source_rows, gradient=True).index_select(
                    0, permutation_inputs[0]).reshape_as(saved_probabilities).to(ctx.topk_dtype)
                    if ctx.source_rows else saved_probabilities.new_empty(
                        saved_probabilities.shape, dtype=ctx.topk_dtype))
            # Release ordinary scratch before allocating the owned token gradient;
            # SHMEM remains leased until the current-stream work is recorded.
            grad_weight1 = execution.intermediates.grad_weight1
            grad_weight2 = execution.intermediates.grad_weight2
            if os.environ.get("HP_MEGA_MOE_CAPTURE_DW") == "1":
                global _CAPTURED_WEIGHT_GRADS  # pylint: disable=global-statement
                _CAPTURED_WEIGHT_GRADS = (grad_weight1.detach().clone(),
                                          grad_weight2.detach().clone())
            grad_x = execution.grad_x
            del execution
            grad_input = _restore_input_gradient(ctx, grad_x, permutation_inputs)
            if grad_topk_weights is not None:
                grad_topk_weights = grad_topk_weights.to(ctx.topk_dtype)
            shared_dw1 = shared_dw2 = None
            if shared_backward:
                if grad_input is not None:
                    grad_input = grad_input + shared_tensors[8]
                shared_dw1 = shared_tensors[10].t() if ctx.needs_input_grad[9] else None
                shared_dw2 = shared_tensors[2].t() if ctx.needs_input_grad[10] else None
            _release_completed_graph(ctx, workspace)
            return (
                grad_input,
                grad_weight1,
                grad_weight2,
                None,
                None,
                None,
                None,
                grad_topk_weights,
                None,
                shared_dw1,
                shared_dw2,
                None,
            )
        finally:
            try:
                leases.close()
            finally:
                workspace.release(stream=workspace.lease_stream)


def _native_shared_forward(
    source: torch.Tensor, first: torch.Tensor, second: torch.Tensor, limit: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute native shared intermediates once without staging copies or backward recomputation.

    Args:
        source: Dense token-major input.
        first: Current shared gate/up weight.
        second: Current shared down-projection weight.
        limit: Optional clamp applied to the SwiGLU branches.

    Returns:
        Owned preactivation, activation and output in native dense layouts.
    """
    # Pylint cannot infer the callable signature of Torch's C linear binding.
    up = torch.nn.functional.linear(source, first)  # pylint: disable=not-callable
    activation_input = up
    if limit is not None and limit > 0:
        gate, branch = up.chunk(2, dim=-1)
        activation_input = torch.cat((gate.clamp(max=limit), branch.clamp(-limit, limit)), dim=-1)
    activation = load_torch_npu().npu_swiglu(activation_input, dim=-1)
    return up, activation, torch.nn.functional.linear(activation, second)  # pylint: disable=not-callable


class _NativeSharedBackward(torch.autograd.Function):
    """Keep native shared gradients independent of the routed kernel's completion."""

    @staticmethod
    def forward(
        ctx: Any, source: torch.Tensor, first: torch.Tensor, second: torch.Tensor,
        up: torch.Tensor, activation: torch.Tensor, output: torch.Tensor, limit: float | None,
    ) -> torch.Tensor:
        """Attach native backward to the shared forward result and owned intermediates.

        Args:
            ctx: Autograd state retaining this invocation's tensors.
            source: Original token rows before TopK expansion.
            first: Current dense gate/up weight.
            second: Current dense down-projection weight.
            up: Owned gate/up result produced by the fused forward.
            activation: Owned shared activation produced by the fused forward.
            output: Owned shared output produced by the fused forward.
            limit: Optional SwiGLU branch clamp, matching the native MLP.

        Returns:
            The shared output with an independent native backward node.
        """
        ctx.save_for_backward(source, first, second, up, activation)
        ctx.limit = limit
        return output

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[Any, ...]:
        """Use the same dense layouts and activation derivative as native shared MLP.

        Args:
            ctx: Invocation-specific saved tensors and gradient requirements.
            gradient: Token-major shared output gradient.

        Returns:
            Input and weight gradients followed by non-differentiable state slots.
        """
        source, first, second, up, activation = ctx.saved_tensors
        grad_activation = torch.mm(gradient, second)
        activation_input = up
        if ctx.limit is not None and ctx.limit > 0:
            gate, branch = up.chunk(2, dim=-1)
            activation_input = torch.cat((gate.clamp(max=ctx.limit), branch.clamp(-ctx.limit, ctx.limit)), dim=-1)
        grad_up = torch.ops.npu.npu_swiglu_backward(grad_activation, activation_input, -1)
        if ctx.limit is not None and ctx.limit > 0:
            grad_gate, grad_branch = grad_up.chunk(2, dim=-1)
            grad_up = torch.cat((torch.where(gate <= ctx.limit, grad_gate, 0),
                                 torch.where((branch >= -ctx.limit) & (branch <= ctx.limit), grad_branch, 0)), dim=-1)
        grad_source = torch.mm(grad_up, first) if ctx.needs_input_grad[0] else None
        grad_first = torch.mm(grad_up.t(), source) if ctx.needs_input_grad[1] else None
        grad_second = torch.mm(gradient.t(), activation) if ctx.needs_input_grad[2] else None
        return grad_source, grad_first, grad_second, None, None, None, None


def _requires_backward(*tensors: torch.Tensor | None) -> bool:
    """Inspect autograd outside the custom Function's always-disabled forward scope.

    Args:
        tensors: Differentiable caller-owned input and parameter tensors.

    Returns:
        Whether the current call can construct a backward graph.
    """
    return torch.is_grad_enabled() and any(tensor is not None and tensor.requires_grad for tensor in tensors)


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
        None,
        None,
        _requires_backward(routed_tokens, weight1, weight2),
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
    shared_weight1: torch.Tensor | None = None,
    shared_weight2: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        shared_weight1: Optional caller-owned dense shared gate/up weight.
        shared_weight2: Optional caller-owned dense shared down-projection weight.

    Returns:
        Owned expert-major rows, or token-major rows when topk_weights is provided,
        differentiable with respect to input tokens, expert weights and Router probabilities.
    """
    # Shared descriptors bind dense rows; keep the layout copy in the caller's autograd graph.
    if shared_weight1 is not None:
        hidden_states = hidden_states.contiguous()
    forward_shared = getattr(plan.spec, "forward_shared_expert", True)
    backward_shared = getattr(plan.spec, "backward_shared_expert", False)
    fused_shared = shared_weight1 is not None and (forward_shared or backward_shared)
    result = _MegaMoeFunction.apply(
        hidden_states,
        _native_weight_operand(weight1, plan),
        _native_weight_operand(weight2, plan),
        route.metadata,
        plan,
        workspace,
        (route.routed_tokens, topk_ids, route.unpermute_mapping, getattr(route, "permute_src_index", None),
         getattr(route, "peer_row_ordinals", None)),
        topk_weights,
        workspace_claimed,
        (shared_weight1 if backward_shared else shared_weight1.detach()) if fused_shared else None,
        (shared_weight2 if backward_shared else shared_weight2.detach()) if fused_shared else None,
        _requires_backward(hidden_states, weight1, weight2, topk_weights,
                           shared_weight1 if backward_shared else None,
                           shared_weight2 if backward_shared else None),
    )
    if shared_weight1 is None:
        return result
    if not fused_shared:
        return result, _native_shared_forward(hidden_states, shared_weight1, shared_weight2, plan.spec.swiglu_limit)[2]
    routed_output, shared_output, shared_up, shared_activation = result
    if backward_shared:
        return routed_output, shared_output
    shared_output = _NativeSharedBackward.apply(
        hidden_states, shared_weight1, shared_weight2, shared_up, shared_activation,
        shared_output, plan.spec.swiglu_limit,
    )
    return routed_output, shared_output


def _execute_mega_moe_without_autograd(
    topk_weights: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    route: PreparedTopKRoute,
    plan: MegaMoePlan,
    workspace: MegaMoeWorkspace,
    *, workspace_claimed: bool = False,
) -> torch.Tensor:
    """Restore owned token outputs before releasing inference-only scratch.

    Initial reentrant-checkpoint forwards need neither a backward dispatch copy
    nor an owned expert-major intermediate. Consume both SHMEM views on their
    leased stream; only the unpermuted, independently owned output escapes.
    """
    if torch.is_grad_enabled():
        raise RuntimeError("The no-autograd path requires a no_grad context")
    if workspace_claimed:
        if not workspace.in_use:
            raise RuntimeError("Inference requires the caller-held workspace lease")
    else:
        if not plan.spec.auto_tokens:
            workspace.ensure(plan.spec, route.routed_tokens.dtype, route.routed_tokens.device)
        workspace.claim()
    leases = ExitStack()
    try:
        replica_route = getattr(route.metadata, "replica_route", None)
        pool = (leases.enter_context(prefetch_weights(
            (weight1, weight2), replica_route, provider=workspace.replica_provider, overlap=True))
            if _has_replica_transfers(replica_route) else None)
        execution = _prepare_forward_execution(
            workspace, plan, route.routed_tokens, route.metadata.expert_capacity)
        execution = _with_task_prefix(execution, replica_route, plan.fwd_task_prefixes)
        if getattr(plan.spec, "apply_probs_before_fc2", False) is True:
            _stage_probabilities(workspace, route.unpermute_mapping, topk_weights.float().contiguous(), plan.spec)
        _launch_forward_kernel(
            plan, route.metadata.for_kernel(plan.spec.hidden_size), execution.source,
            _native_weight_operand(weight1, plan),
            _native_weight_operand(weight2, plan), execution, pool, stream=workspace.lease_stream,
            save_dispatch=False)
        rows = route.routed_tokens.shape[0]
        if not rows:
            return route.routed_tokens.new_empty((0, plan.spec.hidden_size))
        if getattr(plan.spec, "apply_probs_before_fc2", False) is True:
            return _restore_unweighted_output(execution.combine[:rows], route.unpermute_mapping, plan.spec.top_k)
        return restore_topk_output(
            execution.combine[:rows], route.unpermute_mapping, topk_weights)
    finally:
        try:
            leases.close()
        finally:
            if not workspace_claimed:
                workspace.release(stream=workspace.lease_stream)
