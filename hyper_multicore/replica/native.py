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
"""Native grouped-MoE adapter using the shared expert placement contract."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from hyper_multicore.replica.communication import differentiable_all_to_all_single

from .capacity import ExpertReplicaConfig
from .cost import ExpertReplicaCostModel
from .device import build_device_expert_replica_plan, validate_planner_backend
from .routing import ReplicaRoute, _gather_replica_counts, _plan_replica_route, stable_expert_order
from .transport import prefetch_weights, return_gradients


def _npu_ops() -> Any:
    """Load the optional NPU backend only when native NPU experts execute."""
    import torch_npu  # pylint: disable=import-outside-toplevel,unused-import
    # These operators are registered dynamically by the optional extension.
    return torch.ops.npu


def _gmm(inputs: torch.Tensor, weights: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Use native grouped matmul for BF16 forward and input gradients."""
    return _npu_ops().npu_grouped_matmul(
        x=[inputs], weight=[weights], bias=[], group_list=groups,
        split_item=3, group_type=0, group_list_type=0,
    )[0]


def _weight_gradient(inputs: torch.Tensor, gradients: torch.Tensor, route: ReplicaRoute,
                     home: torch.Tensor, guest: torch.Tensor) -> None:
    """Compute FP32 expert partials on native kernels that lack mixed-output GMM.

    CANN's non-quantized grouped-matmul contract ties output dtype to the input
    dtype. Ordinary FP32 matmul avoids an irreversible BF16 rounding of each
    replica partial. Boundaries come from the host plan, with no device readback.
    """
    counts = route.plan.destination_counts[route.rank]
    inputs, gradients = inputs.float(), gradients.float()
    offset = 0
    for slot, count in enumerate(counts):
        if count:
            output = home[slot] if slot < home.shape[0] else guest[slot - home.shape[0]]
            torch.mm(inputs[offset:offset + count].T, gradients[offset:offset + count], out=output)
        offset += count


def _split_gmm(inputs: torch.Tensor, home: torch.Tensor, guest: torch.Tensor,
               groups: torch.Tensor, route: ReplicaRoute, *, transpose: bool = False) -> torch.Tensor:
    """Run home and guest segments without concatenating expert matrices."""
    count = route.plan.config.home_experts
    home_rows = sum(route.plan.destination_counts[route.rank][:count])
    if transpose:
        home, guest = home.transpose(-1, -2), guest.transpose(-1, -2)
    parts = []
    if home_rows:
        parts.append(_gmm(inputs[:home_rows], home, groups[:count]))
    if inputs.shape[0] > home_rows:
        parts.append(_gmm(inputs[home_rows:], guest, groups[count:] - home_rows))
    return torch.cat(parts) if len(parts) > 1 else parts[0]


def _replica_swiglu(up: torch.Tensor, limit: float) -> torch.Tensor:
    """Evaluate optional clamp and SwiGLU in FP32, restoring the input dtype."""
    gate, value = up.float().chunk(2, dim=-1)
    if limit > 0:
        gate, value = gate.clamp(max=limit), value.clamp(-limit, limit)
    return (F.silu(gate) * value).to(up.dtype)


def _replica_swiglu_backward(gradient: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """Match Torch clamp derivatives, including both inclusive boundary values."""
    gate, value = up.float().chunk(2, dim=-1)
    clipped_gate = gate.clamp(max=limit) if limit > 0 else gate
    clipped_value = value.clamp(-limit, limit) if limit > 0 else value
    sigmoid = clipped_gate.sigmoid()
    grad_gate = gradient.float() * clipped_value * sigmoid * (1 + clipped_gate * (1 - sigmoid))
    grad_value = gradient.float() * F.silu(clipped_gate)
    if limit > 0:
        grad_gate = grad_gate * (gate <= limit)
        grad_value = grad_value * ((value >= -limit) & (value <= limit))
    return torch.cat((grad_gate, grad_value), dim=-1).to(up.dtype)


class _NativeReplicaExperts(torch.autograd.Function):
    """Return gradients to original parameter owners inside the expert backward."""

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor,
                counts: torch.Tensor, route: ReplicaRoute, swiglu_limit: float | None) -> torch.Tensor:
        """Prefetch this invocation's weights and execute native grouped SwiGLU."""
        groups = counts.to(torch.int64).cumsum(0)
        with prefetch_weights((weight1, weight2), route) as pool:
            if inputs.shape[0]:
                up = _split_gmm(inputs, weight1, pool.weights[0], groups, route)
                activation = (_npu_ops().npu_swiglu(up) if swiglu_limit is None
                              else _replica_swiglu(up, swiglu_limit))
                output = _split_gmm(activation, weight2, pool.weights[1], groups, route)
            else:
                up = inputs.new_empty((0, weight1.shape[-1]))
                activation = inputs.new_empty((0, weight2.shape[1]))
                output = inputs.clone()
        ctx.route = route
        ctx.swiglu_limit = swiglu_limit
        ctx.save_for_backward(inputs, weight1, weight2, groups, up, activation)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        """Re-prefetch without retaining guest snapshots and sum FP32 partials."""
        inputs, weight1, weight2, groups, up, activation = ctx.saved_tensors
        route = ctx.route
        with prefetch_weights((weight1, weight2), route, backward=True) as pool:
            grad1 = torch.zeros_like(weight1, dtype=torch.float32)
            grad2 = torch.zeros_like(weight2, dtype=torch.float32)
            if inputs.shape[0]:
                grad_output = grad_output.contiguous()
                grad_activation = _split_gmm(grad_output, weight2, pool.weights[1], groups, route,
                                             transpose=True)
                grad_up = (_npu_ops().npu_swiglu_backward(grad_activation, up) if ctx.swiglu_limit is None
                           else _replica_swiglu_backward(grad_activation, up, ctx.swiglu_limit))
                grad_input = _split_gmm(grad_up, weight1, pool.weights[0], groups, route, transpose=True)
                _weight_gradient(inputs, grad_up, route, grad1, pool.gradients[0])
                _weight_gradient(activation, grad_output, route, grad2, pool.gradients[1])
            else:
                grad_input = grad_output.clone()
            grad1, grad2 = return_gradients((grad1, grad2), route, pool.gradients, consume=True)
        return grad_input, grad1, grad2, None, None, None


def native_replica_experts(inputs: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor,
                           counts: torch.Tensor, route: ReplicaRoute, *,
                           swiglu_limit: float | None = None) -> torch.Tensor:
    """Execute packed weights [H,D,2I], [H,I,D] with native NPU kernels.

    Args:
        inputs: BF16 expert-major NPU token rows.
        weight1: Original owner's packed gate/up matrices.
        weight2: Original owner's down matrices.
        counts: Physical expert row counts.
        route: Invocation-owned placement and source quotas.
        swiglu_limit: Use FP32 SwiGLU when set; positive values also clamp gate/up.
            None preserves the ordinary native NPU SwiGLU implementation.

    Returns:
        Expert-major BF16 outputs with gradients returned to original owners.
    """
    if inputs.device.type != "npu" or inputs.dtype != torch.bfloat16:
        raise ValueError("native hot replicas currently require BF16 NPU inputs")
    if swiglu_limit is not None and (
            isinstance(swiglu_limit, bool) or not isinstance(swiglu_limit, (int, float))
            or not math.isfinite(swiglu_limit)):
        raise ValueError("swiglu_limit must be a finite number or None")
    return _NativeReplicaExperts.apply(inputs, weight1, weight2, counts, route, swiglu_limit)


@dataclass(frozen=True)
class NativeReplicaDispatch:
    """Invocation-owned all-to-all inverses and physical expert route."""

    route: ReplicaRoute
    send_splits: list[int]
    recv_splits: list[int]
    send_order: torch.Tensor
    recv_order: torch.Tensor


def dispatch_native_replicas(inputs: tuple, config: ExpertReplicaConfig,
                             group: object,
                             *, minimum_replica_rows: int = 0,
                             planner_backend: str = "cpu",
                             minimum_load_reduction: float = 0.0,
                             minimum_saved_rows: int = 0,
                             _rank_nodes: tuple[int, ...] = (),
                             _cost_model: ExpertReplicaCostModel | None = None) -> tuple[tuple, NativeReplicaDispatch]:
    """Dispatch existing native expert-major inputs through shared replicas.

    Args:
        inputs: Native expert-major values, logical counts and optional scores.
        config: Logical expert topology and per-rank replica budget.
        group: EP process group.
        minimum_replica_rows: Minimum useful row count for a guest copy.
        planner_backend: CPU or device placement implementation.
        minimum_load_reduction: Required receive-peak reduction for optional copies.
        minimum_saved_rows: Required absolute receive-peak reduction for optional copies.

    Returns:
        Dispatched inputs and invocation-owned inverse metadata for combine.
    """
    validate_planner_backend(planner_backend, minimum_replica_rows, _cost_model)
    values, counts = inputs[:2]
    rank = dist.get_rank(group)
    payload = counts.to(torch.int64).contiguous()
    gathered = _gather_replica_counts(payload, config.ep_size, group)
    width = config.slots_per_rank
    device_plan = None
    if planner_backend == "device":
        device_plan = build_device_expert_replica_plan(gathered, config,
                                                       minimum_replica_rows=minimum_replica_rows,
                                                       minimum_load_reduction=minimum_load_reduction,
                                                       minimum_saved_rows=minimum_saved_rows)
        plan = device_plan.host_summary()
        slots, repeats = device_plan.source_runs(rank)
        send_splits = list(plan.rank_splits[rank])
        recv_splits = [row[rank] for row in plan.rank_splits]
        device_counts = device_plan.dispatch_counts
        recv_counts = device_counts[:, rank * width:(rank + 1) * width].contiguous()
    else:
        host = gathered.cpu().tolist()
        plan, _ = _plan_replica_route(host, config, group, rank, adaptive=bool(config.replica_slots_per_rank),
            minimum_replica_rows=minimum_replica_rows, cost_model=_cost_model,
            minimum_load_reduction=minimum_load_reduction, minimum_saved_rows=minimum_saved_rows,
            _rank_nodes=_rank_nodes)
        runs = sorted((expert, slot, plan.dispatch_counts[rank][slot])
                      for slot, expert in enumerate(plan.physical_to_logical) if expert >= 0)
        slots = torch.tensor([slot for _, slot, _ in runs], device=values.device)
        repeats = torch.tensor([count for _, _, count in runs], device=values.device)
        send_splits = [sum(plan.dispatch_counts[rank][peer * width:(peer + 1) * width])
                       for peer in range(config.ep_size)]
        received = [row[rank * width:(rank + 1) * width] for row in plan.dispatch_counts]
        recv_splits = [sum(row) for row in received]
        recv_counts = torch.tensor(received, dtype=torch.int64, device=values.device)
        device_counts = torch.tensor(plan.dispatch_counts, device=values.device)
    physical_ids = torch.repeat_interleave(slots, repeats, output_size=values.shape[0])
    send_order = stable_expert_order(physical_ids, config.physical_experts)
    recv_ids = torch.arange(width, device=values.device).repeat(config.ep_size)
    recv_order = stable_expert_order(torch.repeat_interleave(
        recv_ids, recv_counts.flatten(), output_size=sum(recv_splits)), width)
    route = ReplicaRoute(plan, physical_ids, device_counts, rank, group, device_plan=device_plan)
    state = NativeReplicaDispatch(route, send_splits, recv_splits, send_order, recv_order)
    routed = differentiable_all_to_all_single(values[send_order], send_splits, recv_splits, group)[recv_order]
    result = (routed, recv_counts.sum(0))
    if len(inputs) > 2 and inputs[2] is not None:
        scores = differentiable_all_to_all_single(inputs[2][send_order], send_splits, recv_splits, group)[recv_order]
        result += (scores,)
    return result, state


def combine_native_replicas(output: torch.Tensor, state: NativeReplicaDispatch) -> torch.Tensor:
    """Restore original logical expert-major rows before model unpermutation.

    Args:
        output: Local physical expert output rows.
        state: This invocation's inverse token exchange and ordering metadata.

    Returns:
        Source rank outputs in their original logical expert-major order.
    """
    rank_major = torch.empty_like(output)
    rank_major[state.recv_order] = output
    combined = differentiable_all_to_all_single(
        rank_major, state.recv_splits, state.send_splits, state.route.group)
    result = torch.empty_like(combined)
    result[state.send_order] = combined
    return result
