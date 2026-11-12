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
"""Preserve LongCat's expert-ordered input-gradient accumulation."""

from typing import Any

# Native tensor APIs are the model adapter contract.
import torch


class _OrderedExpertInputs(torch.autograd.Function):
    """Expose identity aliases without expanding them, and reduce slot gradients in order."""

    @staticmethod
    def forward(ctx: Any, hidden: torch.Tensor, token: torch.Tensor,
                slot: torch.Tensor, top_k: int, indexed: bool) -> tuple:
        """Gather real rows and retain one storage-sharing identity view per slot.

        Args:
            ctx: Autograd context retaining the source route.
            hidden: Original token matrix.
            token: Source token for each compact real assignment.
            slot: Sorted expert slot for each compact real assignment.
            top_k: Number of original router slots.
            indexed: Emit an assignment-shaped gradient carrier for an indexed executor.
        """
        ctx.save_for_backward(token, slot)
        ctx.source_shape = hidden.shape
        ctx.top_k = top_k
        # The carrier preserves assignment-shaped gradients; only the indexed executor may consume its values.
        real = (hidden.as_strided((token.numel(), hidden.shape[1]), (0, 1)) if indexed
                else hidden.index_select(0, token))
        return (real,) + tuple(hidden.view_as(hidden) for _ in range(top_k))

    @staticmethod
    def backward(ctx: Any, real_gradient: torch.Tensor, *identity_gradients: torch.Tensor) -> tuple:
        """Accumulate descending expert IDs with the source BF16 cast boundaries.

        Args:
            ctx: Retained source route and shape.
            real_gradient: One gradient for each compact real assignment.
            *identity_gradients: Separate identity input gradients for the original slots.
        """
        token, slot = ctx.saved_tensors
        gradient = real_gradient.new_zeros(ctx.source_shape)
        for position in range(ctx.top_k - 1, -1, -1):
            selected = slot == position
            slot_tokens = token[selected]
            contribution = identity_gradients[position]
            if slot_tokens.numel():
                contribution = contribution.index_copy(0, slot_tokens, real_gradient[selected])
            gradient = gradient + contribution
        return gradient, None, None, None, None


class OrderedInputGather:
    """Bind the real gather and identity aliases to one forward invocation."""

    def __init__(self, top_k: int, *, indexed: bool = False) -> None:
        """Retain the current source slot count.

        Args:
            top_k: Number of router slots in this invocation.
            indexed: Keep original activation storage and pass explicit row indices to the executor.
        """
        self.top_k = top_k
        self.indexed = indexed
        self.identity_inputs = ()

    def __call__(self, hidden: torch.Tensor, token: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        """Return compact real rows while retaining the identity views.

        Args:
            hidden: Original token matrix.
            token: Source token for each compact real assignment.
            slot: Sorted expert slot for each compact real assignment.
        """
        real, *identity = _OrderedExpertInputs.apply(hidden, token, slot, self.top_k, self.indexed)
        self.identity_inputs = tuple(identity)
        return real
