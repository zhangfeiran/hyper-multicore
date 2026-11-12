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
"""Route real LongCat assignments and combine them with identity experts."""

from typing import Any

import torch
import torch.distributed as dist

from hyper_multicore.longcat._assignments import masked_expert_forward
from hyper_multicore.longcat._torch_routing import OrderedInputGather


def _validate(hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, top_k: int) -> None:
    """Validate input metadata before entering any distributed collective."""
    if hidden.ndim != 2 or ids.ndim != 2 or ids.shape != weights.shape:
        raise ValueError("LongCat hidden and routes must be [T,H] and matching [T,K] tensors")
    if ids.shape != (hidden.shape[0], top_k):
        raise ValueError("LongCat routes must match the flattened token count and configured top_k")
    if ids.dtype not in (torch.int32, torch.int64) or not weights.is_floating_point():
        raise TypeError("LongCat expert IDs must be integer and router weights floating point")
    if ids.device != hidden.device or weights.device != hidden.device:
        raise ValueError("LongCat inputs must reside on the same device")


def _identity(hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, real: int,
              initial: torch.Tensor, inputs: tuple) -> torch.Tensor:
    """Add identity contributions in expert order with per-slot BF16 casts."""
    result = initial
    for slot in range(ids.shape[1]):
        probability = weights[:, slot] * (ids[:, slot] >= real)
        result = result + (inputs[slot] * probability[:, None]).to(hidden.dtype)
    return result


def _uses_indexed_source(executor: Any) -> bool:
    """Keep metadata carriers inside the fused SHMEM consumer that understands them."""
    return (getattr(executor, "communication_backend", "shmem") == "shmem"
            and getattr(executor, "shmem_data_engine", "mte") in ("mte_roce", "roce"))


def _torch_forward(executor: Any, hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
                   real: int, total: int, group: Any, expert_weights: tuple) -> torch.Tensor:
    """Preserve ordered LongCat arithmetic using ordinary Torch operators."""
    error = ((ids < 0) | (ids >= total)).any()
    order = (ids.float() if total <= 2**24 else ids).argsort(dim=-1)
    ids, weights = ids.gather(1, order), weights.gather(1, order)
    indexed = _uses_indexed_source(executor)
    gather = OrderedInputGather(ids.shape[1], indexed=indexed)
    result = masked_expert_forward(executor, hidden, ids, weights, ids < real, expert_weights=expert_weights,
                                   ep_group=group, route_error=error, input_gather=gather)
    return _identity(hidden, ids, weights, real, result, gather.identity_inputs)


def _triton_forward(executor: Any, hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
                    real: int, total: int, group: Any, expert_weights: tuple) -> torch.Tensor:
    """Use optional Ascend Triton kernels for ordered BF16 routing arithmetic."""
    if hidden.device.type != "npu" or hidden.dtype != torch.bfloat16 or weights.dtype != torch.float32:
        raise ValueError("Triton routing requires NPU BF16 states and FP32 router weights")
    if total > 2**24:
        raise ValueError("Triton routing requires exactly float32-representable expert IDs")
    from hyper_multicore.longcat._route_kernels import OrderedCombine, OrderedGather  # pylint: disable=import-outside-toplevel

    hidden = hidden.contiguous()
    order = ids.float().argsort(dim=-1)
    ids, weights = ids.gather(1, order), weights.gather(1, order)
    token, slot = (ids < real).nonzero(as_tuple=True)
    token, slot = token.contiguous(), slot.contiguous()
    invalid = ((ids < 0) | (ids >= total)).any().to(torch.int64)
    agreement = torch.stack((invalid.new_tensor(token.numel()), invalid))
    if group is not None:
        dist.all_reduce(agreement, group=group)
    count, error = agreement.tolist()
    if error:
        raise ValueError("LongCat route contains an invalid expert ID")
    inverse = torch.full(ids.shape, -1, device=ids.device, dtype=torch.int32)
    if token.numel():
        positions = token * ids.shape[1] + slot
        inverse.view(-1).index_copy_(0, positions, torch.arange(token.numel(), device=ids.device, dtype=torch.int32))
    indexed = _uses_indexed_source(executor)
    states, identity = OrderedGather.apply(hidden, token, inverse, indexed)
    if count:
        compact_ids = ids[token, slot].reshape(-1, 1).contiguous()
        options = {"_indexed_source": (hidden, token)} if indexed else {}
        unweighted = executor(states, compact_ids, weights.new_ones((token.numel(), 1)),
                              expert_weights=expert_weights, **options)
    else:
        unweighted = states
    return OrderedCombine.apply(identity, unweighted, weights, inverse, token, slot)


def route_experts(executor: Any, hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
                  real: int, zero: int, top_k: int, group: Any, expert_weights: tuple,
                  backend: str) -> torch.Tensor:
    """Execute compact real routes and ordered local identities with one executor."""
    original_shape = hidden.shape
    if hidden.ndim < 2:
        raise ValueError("LongCat hidden states must include token and hidden dimensions")
    flat = hidden.reshape(-1, original_shape[-1])
    _validate(flat, ids, weights, top_k)
    if backend == "torch":
        result = _torch_forward(executor, flat, ids, weights, real, real + zero, group, expert_weights)
    elif backend == "triton":
        result = _triton_forward(executor, flat, ids, weights, real, real + zero, group, expert_weights)
    else:
        raise ValueError("LongCat routing backend must be torch or triton")
    return result.reshape(original_shape)
