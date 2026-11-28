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
"""Checkpoint-owned routing and replica state for BF16 Push and Pull."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from types import TracebackType
from typing import Any, TYPE_CHECKING

import torch

from hyper_multicore.modules.mega_moe.route import (
    _FORWARD_PLAN_REPLAY_MASK, _INVALID_CHECKPOINT_VERSION_MASK, _INVALID_WEIGHT_MASK,
    _TENSOR_DUMP_MASK, PreparedTopKRoute, _validate_supplied_counts, _validate_topk_inputs,
)


if TYPE_CHECKING:
    from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec


@dataclass
class ReplayRecord:
    """Hold the forward placement and invocation metadata without saving Router tensors."""

    owner: Any
    executor: object | None = None
    route: PreparedTopKRoute | None = None
    route_inputs: tuple[torch.Size, torch.dtype, torch.device, bool, bool] | None = None
    stream: Any = None
    autograd_mask: int = 0


@dataclass
class _CheckpointScope:
    records: list[ReplayRecord]
    replay: bool
    same_parameters: bool
    non_reentrant: bool
    cursor: int = 0


_ACTIVE: ContextVar[_CheckpointScope | None] = ContextVar("hmc_hot_replica_checkpoint", default=None)
_REPLAY_SCOPES: ContextVar[tuple[AbstractContextManager, ...]] = ContextVar("hmc_replay_scopes", default=())


class _ReplayContext(AbstractContextManager):
    """Start a new replay cursor each time Torch re-enters its recompute context."""

    def __init__(self, tape: ReplayCheckpoint) -> None:
        """Keep the original invocation's tape alive for each recomputation."""
        self.tape = tape

    def __enter__(self) -> None:
        """Enter a fresh replay scope when Torch starts recomputation."""
        scope = self.tape.scope(True, non_reentrant=True)
        scope.__enter__()
        _REPLAY_SCOPES.set((*_REPLAY_SCOPES.get(), scope))

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None,
    ) -> bool | None:
        """Restore the previous scope on completion or error."""
        scopes = _REPLAY_SCOPES.get()
        _REPLAY_SCOPES.set(scopes[:-1])
        return scopes[-1].__exit__(exc_type, exc, traceback)


class ReplayCheckpoint:
    """Record the original placement of each expert call for one checkpoint."""

    def __init__(self, parameter_version: Callable[[], Any]) -> None:
        """Register the caller's logical parameter version lookup."""
        self.parameter_version = parameter_version
        self.records: list[ReplayRecord] = []
        self.forward_version = None
        self.started = False

    def contexts(self) -> tuple[AbstractContextManager, AbstractContextManager]:
        """Provide the context_fn for one non-reentrant torch checkpoint call."""
        return self.scope(False, non_reentrant=True), _ReplayContext(self)

    @contextmanager
    def scope(self, replay: bool, *, non_reentrant: bool = False) -> Iterator[None]:
        """Enter the tape before running any operator of the checkpointed block."""
        version = self.parameter_version()
        if not replay:
            if self.started:
                raise RuntimeError("Create a separate wrap_checkpoint wrapper or ReplayCheckpoint for each forward")
            self.started = True
            self.forward_version = version
        state = _CheckpointScope(self.records, replay, version is not None and version == self.forward_version,
                                 non_reentrant)
        token = _ACTIVE.set(state)
        try:
            yield
            if replay and state.cursor != len(state.records):
                raise RuntimeError("Checkpoint replay did not reach all of its recorded forward expert calls")
        finally:
            _ACTIVE.reset(token)


def wrap_checkpoint(
    function: Callable[..., Any], parameter_version: Callable[[], Any], *, prepare_replay: bool = False,
) -> Callable[..., Any]:
    """Wrap one reentrant checkpoint call with its forward expert plans.

    The parameter version and ordered Top-K routes must match in the original
    forward and recompute. The legacy prepare_replay argument is accepted
    for compatibility.
    """
    del prepare_replay
    tape = ReplayCheckpoint(parameter_version)

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        """Enter the checkpoint scope for this invocation."""
        with tape.scope(torch.is_grad_enabled()):
            return function(*args, **kwargs)

    return wrapped


def is_checkpointing() -> bool:
    """Report whether route selection is in a checkpoint call."""
    return _ACTIVE.get() is not None


def is_checkpoint_replaying() -> bool:
    """Report whether the active checkpoint is recomputing its forward."""
    scope = _ACTIVE.get()
    return scope is not None and scope.replay


def route_record(owner: Any, executor: object | None = None) -> tuple[ReplayRecord | None, bool]:
    """Match the current expert call by checkpoint position and executor identity."""
    scope = _ACTIVE.get()
    if scope is None:
        return None, False
    if not scope.replay:
        record = ReplayRecord(owner, executor)
        scope.records.append(record)
        return record, False
    if not scope.same_parameters:
        raise RuntimeError("HotReplica checkpoint replay requires the original logical parameter version")
    if scope.cursor >= len(scope.records):
        raise RuntimeError("Checkpoint replay reached an expert call absent from the original forward")
    record = scope.records[scope.cursor]
    if record.owner is not owner or record.executor is not executor:
        raise RuntimeError("Checkpoint replay did not reach the recorded forward expert executor and resources")
    scope.cursor += 1
    return record, True


def forward_plan_replay_mask(
    record: ReplayRecord, hidden: torch.Tensor, ids: torch.Tensor, probs: torch.Tensor,
    counts: torch.Tensor | None, spec: MegaMoeSpec, mask: int, *, scheduled: bool,
) -> int:
    """Validate the replay contract using Host metadata and the original forward exchange."""
    scope = _ACTIVE.get()
    assert scope is not None
    mask |= _FORWARD_PLAN_REPLAY_MASK
    if scope.replay and mask & _INVALID_WEIGHT_MASK:
        raise ValueError("expert weights do not match the recorded forward layout")
    inputs = (ids.shape, ids.dtype, ids.device, counts is not None, scheduled)
    stream = torch.npu.current_stream(hidden.device)
    if not scope.replay:
        record.route_inputs, record.stream, record.autograd_mask = inputs, stream, mask
        return mask if scope.same_parameters else mask | _INVALID_CHECKPOINT_VERSION_MASK

    _validate_topk_inputs(hidden, ids, probs)
    if counts is not None:
        _validate_supplied_counts(counts, spec.num_experts, hidden.device)
    # Reentrant recompute enables autograd after its no-grad original forward.
    mode_mask = -1 if scope.non_reentrant else _TENSOR_DUMP_MASK
    if (record.route is None or record.route_inputs != inputs or record.stream != stream
            or (record.autograd_mask ^ mask) & mode_mask):
        raise RuntimeError("Recorded forward expert plan does not match the replay invocation")
    return mask
