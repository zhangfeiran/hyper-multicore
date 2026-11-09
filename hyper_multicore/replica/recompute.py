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
"""Dynamic execution state shared by activation recomputation features."""

import contextvars
from typing import Any, Callable, Dict, Optional, Tuple

from torch.utils.checkpoint import checkpoint


class _RecomputeInvocation:
    """Own resources whose lifetime is one checkpoint invocation."""

    def __init__(self) -> None:
        """Initialize an empty invocation resource registry."""
        self.identity = object()
        self.resources: Dict[object, Any] = {}

    def get_resource(self, key: object, factory: Callable[[], Any]) -> Any:
        """Get or create one invocation-local resource."""
        if key not in self.resources:
            self.resources[key] = factory()
        return self.resources[key]

    def clear(self) -> None:
        """Release invocation resources, including partially consumed caches."""
        for resource in self.resources.values():
            clear = getattr(resource, "clear", None)
            if clear is not None:
                clear()
        self.resources.clear()


class RecomputeState:
    """Describe the current checkpoint invocation and execution phase."""

    def __init__(
        self,
        invocation: _RecomputeInvocation,
        recomputing: bool,
        preserve_rng_state: bool = True,
    ) -> None:
        """Initialize state for one phase of a checkpoint invocation."""
        self._invocation = invocation
        self.is_recomputing = recomputing
        self.preserve_rng_state = preserve_rng_state

    @property
    def invocation_id(self) -> object:
        """Return an identity that is stable across forward and recomputation."""
        return self._invocation.identity

    def get_resource(self, key: object, factory: Callable[[], Any]) -> Any:
        """Get an invocation-local resource shared by both execution phases."""
        return self._invocation.get_resource(key, factory)

    def _clear_resources(self) -> None:
        """Release all resources owned by this checkpoint invocation."""
        self._invocation.clear()


_CURRENT_RECOMPUTE_STATE: contextvars.ContextVar[Optional[RecomputeState]] = contextvars.ContextVar(
    "hyper_multicore_replica_recompute_state",
    default=None,
)


class _RecomputeContext:
    """Install one checkpoint execution phase in the current dynamic scope."""

    def __init__(self, state: RecomputeState) -> None:
        """Initialize a context for the supplied execution state."""
        self._state = state
        self._token = None

    def __enter__(self) -> "_RecomputeContext":
        """Expose this phase as the current recompute state."""
        self._token = _CURRENT_RECOMPUTE_STATE.set(self._state)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        """Restore the outer state and clear resources after recomputation."""
        if self._token is not None:
            _CURRENT_RECOMPUTE_STATE.reset(self._token)
        if self._state.is_recomputing or exc_type is not None:
            self._state._clear_resources()  # pylint: disable=protected-access
        return False


def get_recompute_state() -> Optional[RecomputeState]:
    """Return the current checkpoint execution state, if one is active."""
    return _CURRENT_RECOMPUTE_STATE.get()


def is_recomputing() -> bool:
    """Return whether the current dynamic scope is replaying a checkpoint."""
    state = get_recompute_state()
    return state is not None and state.is_recomputing


def create_recompute_contexts(
    preserve_rng_state: bool = True,
) -> Tuple[_RecomputeContext, _RecomputeContext]:
    """Create forward and recompute contexts for one checkpoint invocation."""
    invocation = _RecomputeInvocation()
    return (
        _RecomputeContext(RecomputeState(invocation, recomputing=False, preserve_rng_state=preserve_rng_state)),
        _RecomputeContext(RecomputeState(invocation, recomputing=True, preserve_rng_state=preserve_rng_state)),
    )


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
