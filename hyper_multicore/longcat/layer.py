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
"""Framework-independent LongCat identity and MegaMoE expert integration."""

import os
from typing import Any

import torch
from torch import nn

from hyper_multicore import MegaMoeExperts
from hyper_multicore.longcat._routing import route_experts


def _validate_configuration(values: tuple, zero_experts: int, swiglu_precision: str,
                            routing_backend: str, gate_up_padded: bool) -> None:
    """Check model dimensions and options before constructing the executor."""
    _, _, num_experts, top_k, ep_size = values
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("LongCat dimensions, real experts, top_k and ep_size must be positive integers")
    if isinstance(zero_experts, bool) or not isinstance(zero_experts, int) or zero_experts < 0:
        raise ValueError("zero_experts must be a nonnegative integer")
    if num_experts % ep_size or zero_experts % ep_size or top_k > num_experts + zero_experts:
        raise ValueError("LongCat expert counts must divide EP and top_k must fit the expert count")
    if swiglu_precision not in ("fused", "hf") or routing_backend not in ("torch", "triton"):
        raise ValueError("Unsupported LongCat SwiGLU precision or routing backend")
    if not isinstance(gate_up_padded, bool):
        raise TypeError("gate_up_padded must be boolean")


class LongcatFlashExperts(nn.Module):
    """Execute LongCat real and identity experts with caller-owned EOI weights.

    The caller retains parameter ownership, router, optimizer and EP sharding.
    IDs enumerate real experts first, then identity experts. Expert IDs and
    probabilities retain the original TopK shape; zero experts have no GEMM.
    """

    def __init__(self, *, hidden_size: int, intermediate_size: int, num_experts: int,
                 zero_experts: int, top_k: int, ep_size: int = 1, ep_group: Any = None,
                 dispatch_mode: str = "push", swiglu_precision: str = "fused",
                 communication_backend: str = "shmem",
                 routing_backend: str = "torch", replica_slots_per_rank: int = 0,
                 gate_up_padded: bool = False, initial_token_capacity: int | None = None,
                 replica_min_rows: int = 1024, replica_transport: str | None = None) -> None:
        """Configure real experts and optional whole-expert replicas.

        Args:
            hidden_size: Original hidden width.
            intermediate_size: Original per-expert feed-forward width.
            num_experts: Global number of parameterized experts.
            zero_experts: Global number of identity experts.
            top_k: Number of original router selections per token.
            ep_size: Number of ranks in the explicit expert group.
            ep_group: Expert group; required for distributed execution.
            dispatch_mode: MegaMoE push or pull transport.
            communication_backend: Fused SHMEM MTE/RoCE communication or HCCL routing with local MegaMoE compute.
            swiglu_precision: Fused arithmetic or HF BF16 rounding.
            routing_backend: Portable Torch or optional Ascend Triton routing.
            replica_slots_per_rank: Number of whole-expert guest slots per rank.
            gate_up_padded: Caller storage includes trailing identity rows per rank.
            initial_token_capacity: Optional compact-assignment reservation.
            replica_min_rows: Soft minimum rows for a guest expert.
            replica_transport: Guest weight and gradient transport; RDMA defaults to independent HCCL P2P.
        """
        super().__init__()
        values = (hidden_size, intermediate_size, num_experts, top_k, ep_size)
        _validate_configuration(values, zero_experts, swiglu_precision, routing_backend, gate_up_padded)
        self.hidden_size, self.intermediate_size = hidden_size, intermediate_size
        self.num_experts, self.zero_experts, self.top_k = num_experts, zero_experts, top_k
        self.ep_size, self.ep_group = ep_size, ep_group
        self.routing_backend, self.gate_up_padded = routing_backend, gate_up_padded
        self.replica_slots_per_rank = replica_slots_per_rank
        options = {} if initial_token_capacity is None else {"initial_token_capacity": initial_token_capacity}
        engine = os.environ.get("HYPER_MULTICORE_SHMEM_DATA_ENGINE", "mte")
        if replica_transport is not None:
            options["replica_transport"] = replica_transport
        elif engine != "mte" and replica_slots_per_rank:
            options["replica_transport"] = "p2p"
        self.executor = MegaMoeExperts(
            hidden_size=hidden_size, intermediate_size=intermediate_size, num_experts=num_experts, top_k=1,
            ep_group=ep_group, ep_size=ep_size, create_parameters=False, expert_weight_layout="EOI",
            swiglu_hf_bf16=swiglu_precision == "hf", dispatch_mode=dispatch_mode,
            communication_backend=communication_backend,
            single_expert_tasks=True, route_task_bounds=True, replica_slots_per_rank=replica_slots_per_rank,
            replica_granularity="expert" if replica_slots_per_rank else "token", replica_min_rows=replica_min_rows,
            gate_up_expert_padding=zero_experts if gate_up_padded and not replica_slots_per_rank else 0, **options,
        )

    def _weights(self, expert_weights: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate owner-local EOI storage and exclude unused rows for replicas."""
        if not isinstance(expert_weights, tuple) or len(expert_weights) != 2:
            raise TypeError("expert_weights must be a (gate_up, down) tuple")
        gate, down = expert_weights
        local_real = self.num_experts // self.ep_size
        local_zero = self.zero_experts // self.ep_size if self.gate_up_padded else 0
        if tuple(gate.shape) != (local_real + local_zero, 2 * self.intermediate_size, self.hidden_size):
            raise ValueError("gate_up shape does not match owner-local EOI storage")
        if tuple(down.shape) != (local_real, self.hidden_size, self.intermediate_size):
            raise ValueError("down shape does not match owner-local EOI storage")
        if self.replica_slots_per_rank and self.gate_up_padded:
            gate = gate[:local_real]
        return gate, down

    def forward(self, hidden: torch.Tensor, expert_ids: torch.Tensor, router_weights: torch.Tensor,
                *, expert_weights: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Apply real and identity experts while preserving router-weight gradients.

        Args:
            hidden: BF16 NPU states ending in the configured hidden width.
            expert_ids: Global real/identity IDs of shape [flattened_tokens, top_k].
            router_weights: Differentiable FP32 probabilities with the same shape.
            expert_weights: Current owner-local EOI Gate/Up and Down tensors.

        Returns:
            States with the same shape as ``hidden``.
        """
        if hidden.ndim < 2 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("LongCat hidden width differs from the configured hidden_size")
        weights = self._weights(expert_weights)
        return route_experts(self.executor, hidden, expert_ids, router_weights, self.num_experts,
                             self.zero_experts, self.top_k, self.ep_group, weights, self.routing_backend)

    def close(self) -> None:
        """Release MegaMoE execution resources after the final backward."""
        self.executor.close()
