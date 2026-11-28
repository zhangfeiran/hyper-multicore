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
"""BF16 autograd launch and restoration over the common invocation lifecycle."""
from typing import Any, TYPE_CHECKING
import torch

from hyper_multicore.modules.hot_replica.shmem_transfer import ReplicaExecutionMode
from hyper_multicore.modules.hot_replica.checkpoint import is_checkpointing

from ...route import PreparedTopKRoute, restore_topk_output

from .autograd import execute_mega_moe_with_permutation, _execute_mega_moe_without_autograd

if TYPE_CHECKING:
    from .resources import BF16Resources


def forward(
    module: Any, hidden_states: torch.Tensor, hidden_flat: torch.Tensor,
    topk_ids: torch.Tensor, topk_weights: torch.Tensor,
    weights: tuple[torch.Tensor, torch.Tensor], resources: "BF16Resources",
    route: PreparedTopKRoute,
) -> torch.Tensor:
    """Launch BF16 experts over the route prepared by the frontend."""
    del module
    pull = resources.spec.dispatch_mode == "pull"
    before_fc2 = resources.spec.apply_probs_before_fc2
    workspace = resources.workspace
    workspace_claimed = pull or getattr(route, "workspace_claimed", False)
    binding = route.weight_binding
    plan_options = {}
    if binding is not None:
        binding, plan_options = _prepare_replica_binding(binding, route.tensor_dump, resources.spec.num_experts)
    plan = resources.plan_for_tokens(route.metadata.plan_tokens, hidden_flat.device, **plan_options)
    options = {} if binding is None else {"weight_binding": binding}
    record = route.checkpoint_record
    if not torch.is_grad_enabled():
        output = _execute_mega_moe_without_autograd(
            topk_weights, *weights, route, plan, workspace,
            workspace_claimed=workspace_claimed, **options)
        if record is not None and binding is not route.weight_binding:
            # Both bank layouts save the same owned guest records. The
            # original binding still selects its own replay protocol.
            route.weight_binding.snapshot = binding.snapshot
    else:
        integrated_unpermute = (pull or before_fc2 or (
            getattr(workspace, "reuse_bf16_push_saved", False) is True and not route.tensor_dump and is_checkpointing()
        ))
        output = execute_mega_moe_with_permutation(
            hidden_flat, topk_ids, *weights, route, plan, workspace,
            topk_weights=topk_weights if integrated_unpermute else None,
            workspace_claimed=workspace_claimed, **options)
        if not integrated_unpermute:
            output = restore_topk_output(output, route.unpermute_mapping, topk_weights)
    if binding is not None:
        resources.placement.record_submission(binding.mode)
    return output.reshape_as(hidden_states)


def _prepare_replica_binding(binding, tensor_dump, num_experts):
    """Match the plan to the execution slots, gradient precision and diagnostic mode."""
    options = {}
    route_experts = binding.placement.counts_by_source.shape[1]
    if route_experts != num_experts:
        options["route_experts"] = route_experts
    if binding.mode is ReplicaExecutionMode.FP32_BAND:
        options["weight_grad_dtype"] = torch.float32
    if binding.fused and tensor_dump:
        binding = binding.for_tensor_dump()
    if binding.fused:
        options.update(replica_pipeline=True, reserve_merge_workers=binding.owns_returns)
    return binding, options
