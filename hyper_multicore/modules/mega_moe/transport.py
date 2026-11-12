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
"""HCCL routing with local MegaMoE computation for EP groups without full MTE connectivity."""

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.distributed as dist

from hyper_multicore._loader import load_torch_npu
from hyper_multicore.replica.communication import differentiable_all_to_all_single


_LOCAL_COMPUTE_GROUP: tuple[Any, Any] | None = None


def _local_compute_group() -> Any:
    """Reuse one singleton communicator until the enclosing distributed world changes."""
    global _LOCAL_COMPUTE_GROUP  # pylint: disable=global-statement
    if not dist.is_initialized():
        return None
    world = dist.group.WORLD
    if _LOCAL_COMPUTE_GROUP is None or _LOCAL_COMPUTE_GROUP[0] is not world:
        group = dist.new_group(ranks=[dist.get_rank()], backend="hccl", use_local_synchronization=True)
        _LOCAL_COMPUTE_GROUP = world, group
    return _LOCAL_COMPUTE_GROUP[1]


def require_mte_peers(accessible: Sequence[bool], ep_size: int) -> None:
    """Reject an unreachable MTE peer before submitting the fused communication kernel.

    Args:
        accessible: Actual SDK reachability in Root PE order, including self.
        ep_size: Expected Root size.
    """
    if len(accessible) != ep_size:
        raise RuntimeError("MegaMoE SHMEM topology does not match the expert group")
    missing = [peer for peer, reachable in enumerate(accessible) if not reachable]
    if missing:
        raise RuntimeError(
            f"MegaMoE requires MTE connectivity to every EP peer; unreachable Root PEs: {missing}. "
            "Select HYPER_MULTICORE_SHMEM_DATA_ENGINE=mte_roce before constructing modules if RoCE is available. "
            'Use communication_backend="hccl" to keep local MegaMoE compute with HCCL dispatch/combine, '
            "or place each SHMEM EP group inside an MTE domain. PCIe peer access alone is insufficient."
        )


def _permute(hidden: torch.Tensor, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the empty-source autograd edge without invoking an unsupported empty permute."""
    if not hidden.shape[0]:
        return hidden, ids.new_empty((0,), dtype=torch.int32)
    return load_torch_npu().npu_moe_token_permute(hidden, ids.to(torch.int32).contiguous())


class HcclExpertTransport:
    """Use autograd All-to-All around a single-PE, fused MegaMoE executor.

    Cross-PE data and gradients use HCCL. The local executor initializes SHMEM
    with one PE, so its internal copies and completion counters only address self.
    It retains the same GMM, SwiGLU, weight layout, and dW kernels.
    """

    def __init__(self, group: Any, num_experts: int, factory: Callable[[Any], Any]) -> None:
        """Bind the outer expert group and create a local compute executor.

        Args:
            group: Original, explicit EP process group.
            num_experts: Global parameterized expert count.
            factory: Constructor for a parameterless single-PE MegaMoE module.
        """
        self.group = group
        self.ep_size = dist.get_world_size(group) if dist.is_initialized() else 1
        self.local_experts = num_experts // self.ep_size
        self.num_experts = num_experts
        self.compute_group = _local_compute_group()
        self.compute = factory(self.compute_group)

    def __call__(self, hidden: torch.Tensor, ids: torch.Tensor, probabilities: torch.Tensor,
                 weights: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Dispatch routed rows, execute local fused experts, and combine with router weights.

        Args:
            hidden: Flattened BF16 token states.
            ids: Global expert IDs in TopK order.
            probabilities: Original differentiable router probabilities.
            weights: Caller-owned expert matrices, including optional unused trailing Gate rows.

        Returns:
            Flattened token outputs with activation, router, and expert gradients.
        """
        counts = torch.bincount(ids.reshape(-1).long(), minlength=self.num_experts)
        mask = int(torch.is_grad_enabled()) | int(hidden.requires_grad) << 1
        mask |= int(weights[0].requires_grad) << 2 | int(weights[1].requires_grad) << 3
        mask |= int(probabilities.requires_grad) << 4
        wire = torch.cat((counts.view(self.ep_size, -1), counts.new_full((self.ep_size, 1), mask)), dim=1)
        received_wire = torch.empty_like(wire)
        if self.ep_size > 1:
            dist.all_to_all_single(received_wire, wire, group=self.group)
        else:
            received_wire.copy_(wire)
        received_counts = received_wire[:, :-1].contiguous().view(-1)
        header = torch.stack((counts.view(self.ep_size, -1).sum(1),
                              received_counts.view(self.ep_size, -1).sum(1), received_wire[:, -1])).tolist()
        send_sizes, receive_sizes, masks = header
        if any(peer_mask != mask for peer_mask in masks):
            raise RuntimeError("HCCL MegaMoE autograd configuration must match on all EP ranks")
        source, mapping = _permute(hidden, ids)
        received = (differentiable_all_to_all_single(source, send_sizes, receive_sizes, self.group)
                    if self.ep_size > 1 else source)
        local_ids = torch.arange(self.local_experts, device=hidden.device).repeat(self.ep_size)
        local_ids = local_ids.repeat_interleave(received_counts).reshape(-1, 1)
        local_weights = weights[0][:self.local_experts], weights[1]
        result = self.compute(received, local_ids, received.new_ones(local_ids.shape, dtype=torch.float32),
                              expert_weights=local_weights)
        returned = (differentiable_all_to_all_single(result, receive_sizes, send_sizes, self.group)
                    if self.ep_size > 1 else result)
        if not hidden.shape[0]:
            return returned[:0] + probabilities.sum().to(returned.dtype) * 0
        return load_torch_npu().npu_moe_token_unpermute(returned, mapping,
                                                      probs=probabilities.float().contiguous())

    def close(self) -> None:
        """Close the local fused executor after the final backward.

        The compute group remains registered until distributed shutdown because
        serial executors can share the first group's SHMEM Heap manager.
        """
        self.compute.close()
