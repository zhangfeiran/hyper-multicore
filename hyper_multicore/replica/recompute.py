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
"""Checkpoint compatibility facade sharing generic invocation ownership."""
from typing import Any, Callable
from torch.utils.checkpoint import checkpoint
from hyper_multicore.modules.recompute import (
    RecomputeState, create_recompute_contexts, get_recompute_state, is_recomputing,
    _RecomputeInvocation, _RecomputeContext,
)

__all__ = ["RecomputeState", "create_recompute_contexts", "get_recompute_state", "is_recomputing",
           "replica_checkpoint"]


def replica_checkpoint(function: Callable[..., Any], *args: Any, use_reentrant: bool = False, **kwargs: Any) -> Any:
    """Checkpoint a consumer with a shared replica invocation in both Torch modes.

    Args:
        function: Callable containing one or more replica expert invocations.
        *args: Positional inputs to the callable.
        use_reentrant: Select Torch's reentrant or nonreentrant checkpoint behavior.
        **kwargs: Torch checkpoint controls and nonreentrant callable arguments.

    Returns:
        The checkpointed callable's result.

    Raises:
        ValueError: An external context_fn would replace the invocation contract.
    """
    if "context_fn" in kwargs:
        raise ValueError("replica_checkpoint owns context_fn; compose with create_recompute_contexts instead")
    preserve = kwargs.get("preserve_rng_state", True)
    if not use_reentrant:
        contexts = create_recompute_contexts(preserve)
        return checkpoint(function, *args, use_reentrant=False, context_fn=lambda: contexts, **kwargs)
    invocation = _RecomputeInvocation()
    called = False

    def scoped(*values: Any) -> Any:
        """Associate each reentrant replay with the original explicit invocation."""
        nonlocal called
        state = RecomputeState(invocation, recomputing=called, preserve_rng_state=preserve)
        called = True
        with _RecomputeContext(state):
            return function(*values)

    return checkpoint(scoped, *args, use_reentrant=True, **kwargs)
