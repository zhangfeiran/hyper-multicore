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
"""Compact backward route publication and independent route-weight gradient jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from hyper_multicore.scheduler.config import (
    EVENT_INVALID_ID, RuntimeConfigC, TaskAiCoreType, TaskDescC, TaskSplitValue, TaskType,
)
from hyper_multicore.scheduler.graph import OperatorNode
from hyper_multicore.tasks.task_base import FillConfig
from hyper_multicore.tasks.utils import advance_tsv_vector_only, make_tensor_desc


# Preserve the complete legacy scratch prefix before the local pointer header.
BACKWARD_HEADER_OFFSET = 512
BACKWARD_METADATA_PREFIX_ROWS = 40
ROUTE_GRAD_TOKEN_SPLIT = 16


def reject_indirect_gradient_dump(*_args: object, **_kwargs: object) -> None:
    """Reject raw L1 views whose gradient target is held in the local pointer header."""
    raise NotImplementedError("Indexed backward route-gradient L1 dumps require indirect pointer resolution")


def publish_backward_route(
    workspace: Any, gradient: torch.Tensor, mapping: torch.Tensor,
    expert_output: torch.Tensor, probabilities: torch.Tensor, sorted_slots: torch.Tensor,
    *, needs_probability_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Publish original dY and scalar routes while retaining owned dP output storage.

    Args:
        workspace: Claimed symmetric workspace with compact source and backward metadata.
        gradient: Original token-major output gradient.
        mapping: Original route slot to expert-major output row.
        expert_output: Owned unweighted forward output snapshot.
        probabilities: Owned float32 route probabilities.
        sorted_slots: Expert-major row to original route slot.
        needs_probability_grad: Whether the caller needs route-weight gradients.

    Returns:
        Symmetric original dY rows and the optional owned float32 dP target.
    """
    source = workspace.source_buffer[:gradient.shape[0]]
    source.copy_(gradient)
    metadata = workspace.backward_route_metadata
    rows = metadata[BACKWARD_METADATA_PREFIX_ROWS:BACKWARD_METADATA_PREFIX_ROWS + sorted_slots.numel()]
    rows[:, 0].copy_(sorted_slots)
    ordered = probabilities.reshape(-1).index_select(0, sorted_slots.to(torch.int64))
    rows[:, 1].copy_(ordered.view(torch.int32))
    rows[:, 2].copy_(mapping)
    grad_probabilities = torch.empty_like(probabilities) if needs_probability_grad else None
    addresses = torch.tensor(
        [int(needs_probability_grad), expert_output.data_ptr(),
         grad_probabilities.data_ptr() if grad_probabilities is not None else 0],
        dtype=torch.int64, pin_memory=metadata.device.type != 'cpu',
    )
    header = addresses.to(device=metadata.device, non_blocking=True).view(torch.uint8)
    metadata.view(torch.uint8).reshape(-1)[BACKWARD_HEADER_OFFSET:BACKWARD_HEADER_OFFSET + header.numel()].copy_(header)
    return source, grad_probabilities


@dataclass(frozen=True)
class RouteWeightGradFillConfig(FillConfig):
    """Append source-local dot products without changing routed event counters."""

    def fill(self, cfg: RuntimeConfigC, op: OperatorNode, tsv: TaskSplitValue) -> None:
        """Serialize token blocks and leave all expert joins and termination thresholds intact."""
        for index in range(op.task_num):
            task = TaskDescC()
            task.task_type = TaskType.TASK_MOE_ROUTE_GRAD
            task.task_aicore_type = TaskAiCoreType.TASK_AICORE_VECTOR
            task.num_inputs, task.num_outputs = 2, 1
            task.dependent_event = EVENT_INVALID_ID
            # Slot zero has no backward consumer or threshold. The generic
            # worker requires a valid trigger even for independent local jobs.
            task.trigger_event = 0
            for position, tensor in enumerate(op.inputs):
                task.inputs[position] = make_tensor_desc(
                    input_position=tensor.param_position, data_type=tensor.dtype_size, shape=tensor.shape,
                )
            task.outputs[0] = make_tensor_desc(input_position=25, data_type=4, shape=op.outputs[0].shape)
            task.task_index = index
            task.task_split_num = op.task_num
            task.task_split_value = op.split_value
            cfg.all_tasks[tsv.pre_task_num + index] = task
            cfg.vector_task_indices[tsv.pre_vector_task_num + index] = tsv.pre_task_num + index
        cfg.task_index_num[1] += op.task_num
        advance_tsv_vector_only(tsv, op.task_num)
