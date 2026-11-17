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
"""Model-facing Torch MegaMoe expert module."""

from __future__ import annotations

__all__ = ["MegaMoeExperts", "A5MegaMoeExperts"]

import hashlib
import math
import os
import struct
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Literal

import torch
import torch.distributed as dist

from hyper_multicore.replica.admission import (
    reduction_basis_points,
    resolve_replica_admission,
)
from hyper_multicore.replica.capacity import (
    ExpertReplicaConfig,
    _integer,
)
from hyper_multicore.replica.policy import (
    _calibration_available, _cost_model_enabled, _critical_path_available, _critical_path_identity,
    _runtime_cost_model, _soc_name,
)
from hyper_multicore.replica.recompute import get_recompute_state
from hyper_multicore.replica.critical_path import execution_fingerprint
from hyper_multicore.replica.calibration_catalog import ENVIRONMENT_DEFAULTS
from hyper_multicore._loader import get_multicore_paths
from hyper_multicore.replica.topology import _local_node_identity, _node_ids
from hyper_multicore.replica.device import (
    validate_planner_backend,
)
from hyper_multicore.replica.routing import (
    ReplicaRoute, _GatheredReplicaCounts, _RejectedQuotaMemo, prepare_replica_route,
)
from hyper_multicore.replica._metadata import _ReplicaCountHeaderCache
from hyper_multicore.replica.signal_transport import (
    KERNEL_GRADIENT_TRANSPORT_MODES,
)
from hyper_multicore import _collective, shmem
from hyper_multicore.modules.mega_moe.numerics import MegaMoeNumericsOptions
from hyper_multicore.experimental import MegaMoeTuningOptions
from hyper_multicore.replica.admission import optional_peak_gain_is_sufficient
from hyper_multicore.replica.routing import _has_home_recompute_hint, _HostReplicaCounts, _record_home_decision
from hyper_multicore.scheduler.config import mega_moe_event_capacity
from hyper_multicore.shmem._runtime import configured_udma_qp_count

from ..module import MulticoreModule
from .function import execute_mega_moe_with_permutation, _execute_mega_moe_without_autograd
from .heap_manager import _get_heap_manager, _workspace_layout
from .inactive_tasks import UNMAPPED_GUEST_SUFFIX_SCHEDULE
from .plan import build_mega_moe_plan
from .route import _INVALID_WEIGHT_MASK, prepare_topk_route, restore_topk_output

from .configuration import execution_options as _execution_options
from .configuration import qualify_topology_schedule, summary_admission
from .inactive_tasks import replica_runtime_schedule
from .spec import (
    MegaMoeSpec,
    _COMMUNICATION_SPLIT,
    _align_capacity,
    _resolve_capacity_factors,
    _supports_indexed_backward_pair,
    bind_mega_moe_spec,
    initial_receive_capacity,
)
from .workspace import MegaMoeWorkspace


class _InvalidExpertWeights(ValueError):
    """Expert metadata that must be rejected consistently across EP ranks."""


class _InvalidExpertWeightType(TypeError):
    """Expert dtype that must be rejected consistently across EP ranks."""


ExpertWeightLayout = Literal["EIO", "EOI"]
_EXPERT_WEIGHT_LAYOUTS = frozenset(("EIO", "EOI"))
_MAX_A5_ROW_PLANS = 50
_CHECKPOINT_GRAD_MODE_KEY = object()


def _checkpoint_contract() -> str:
    """Remember original call-boundary grad mode across one checkpoint invocation."""
    state = get_recompute_state()
    if state is None:
        return 'none'
    original_grad = state.get_resource(
        _CHECKPOINT_GRAD_MODE_KEY, lambda: None if state.is_recomputing else torch.is_grad_enabled())
    if original_grad is None:
        return 'unknown'
    return 'nonreentrant' if original_grad else 'reentrant'


@lru_cache(maxsize=1)
def _critical_native_fingerprint() -> str:
    """Hash the deployed adapter and vendor kernels once, outside steady-state scoring."""
    vendor, adapter = get_multicore_paths()
    digest = hashlib.sha256(adapter.read_bytes())
    for path in sorted(vendor.rglob('*')):
        if path.is_file() and path.suffix in ('.so', '.o', '.json'):
            digest.update(str(path.relative_to(vendor)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _critical_execution_signature(module: Any, resources: Any, *, execution_tokens: int | None = None) -> str:
    """Bind calibration to the actual static execution and normalized EP topology."""
    metadata = {name: getattr(module, name) for name in
                ('hidden_size', 'intermediate_size', 'ep_size', 'num_experts', 'top_k',
                 'replica_transport', 'replica_slots_per_rank')}
    metadata['dispatch_mode'] = module._dispatch_mode
    metadata.update(weight_layout=resources.spec.expert_weight_layout, swiglu_limit=resources.spec.swiglu_limit,
                    dispatch_split=resources.spec.dispatch_split, combine_split=resources.spec.combine_split,
                    rank_nodes=resources.rank_nodes, checkpointed=get_recompute_state() is not None,
                    checkpoint_contract=_checkpoint_contract(),
                    native_payload=_critical_native_fingerprint(), policy_abi='dynamic_tokens_v4',
                    cube_cores=getattr(resources.spec, 'num_cube_cores', None),
                    global_world_size=dist.get_world_size() if dist.is_initialized() else module.ep_size,
                    template='replica_ready_ack_v1', weight_dtype='bfloat16', partial_dtype='float32')
    if module.replica_slots_per_rank:
        metadata['runtime_schedule'] = UNMAPPED_GUEST_SUFFIX_SCHEDULE
    metadata['runtime_environment'] = {name: os.environ.get(name, default)
                                       for name, default in ENVIRONMENT_DEFAULTS.items()}
    if execution_tokens is not None:
        _integer(execution_tokens, 'Execution token domain', 0)
        metadata['maximum_source_tokens'] = execution_tokens
    return execution_fingerprint(metadata)


def _validate_swiglu_limit(swiglu_limit: float | None) -> None:
    """Validate a positive clamp value that survives float32 serialization."""
    if swiglu_limit is None:
        return
    valid_type = isinstance(swiglu_limit, (int, float)) and not isinstance(swiglu_limit, bool)
    try:
        encoded_limit = struct.unpack("<f", struct.pack("<f", float(swiglu_limit)))[0] if valid_type else 0.0
        valid_value = valid_type and math.isfinite(swiglu_limit) and math.isfinite(encoded_limit) and encoded_limit > 0
    except (OverflowError, TypeError, ValueError, struct.error):
        valid_value = False
    if not valid_value:
        raise ValueError(
            "swiglu_limit must be None or a finite positive float32-representable number, " f"got {swiglu_limit!r}."
        )


def _create_mega_moe_parameters(
    local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    expert_weight_layout: ExpertWeightLayout,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    """Create independently initialized local expert parameters."""
    if expert_weight_layout == "EOI":
        gate_up_shape = (local_experts, intermediate_size * 2, hidden_size)
        down_shape = (local_experts, hidden_size, intermediate_size)
    else:
        gate_up_shape = (local_experts, hidden_size, intermediate_size * 2)
        down_shape = (local_experts, intermediate_size, hidden_size)
    gate_up_weight = torch.nn.Parameter(
        torch.empty(gate_up_shape)
    )
    down_weight = torch.nn.Parameter(
        torch.empty(down_shape)
    )
    for expert_index in range(local_experts):
        torch.nn.init.xavier_uniform_(gate_up_weight[expert_index])
        torch.nn.init.xavier_uniform_(down_weight[expert_index])
    return gate_up_weight, down_weight


def _resolve_ep_size(ep_group: Any | None) -> int:
    """Infer size from an explicit group without silently choosing WORLD."""
    if ep_group is not None:
        if not dist.is_initialized():
            raise ValueError("Initialize torch.distributed before passing ep_group")
        return dist.get_world_size(ep_group)
    if dist.is_initialized() and dist.get_world_size() > 1:
        raise ValueError("Pass ep_group explicitly for distributed MegaMoeExperts")
    return 1


def _required_replica_inbox_slots(route: ReplicaRoute) -> int:
    """Derive an identical matrix prefix from the agreed plan on all EP ranks."""
    if route.home_only:
        return 0
    plan = route.plan
    home = plan.config.home_experts
    return max((transfer.target_slot - home + 1 for transfer in plan.transfers), default=0)


@dataclass
class _ForwardRouteState:
    """Share selected physical storage needs with the automatic Heap callback."""

    inbox_slots: int = 0
    home_only: bool = False




class _MegaMoeExecutionResources:
    """Own one shape-bound plan and workspace in the shared SHMEM lifecycle."""

    def __init__(
        self,
        specification: dict[str, Any],
        tensor: torch.Tensor,
        *,
        expert_weight_layout: str | None = None,
        shared: bool,
        active_specifications: tuple[Mapping[str, Any], ...],
    ) -> None:
        """Bind resources once to the first NPU tensor."""
        expert_weight_layout = expert_weight_layout or specification.get("expert_weight_layout", "EIO")
        bound_specification = {**specification, "expert_weight_layout": expert_weight_layout}
        self.spec = bind_mega_moe_spec(bound_specification, tensor)
        self._token_capacity = self.spec.local_num_tokens
        self._plans = {}
        calibrated = any(item.get('_replica_cost_model_enabled') and _calibration_available(
            item['dispatch_mode'], item['hidden_size'], item['intermediate_size'], item['ep_size'],
            item['replica_transport'], item['logical_num_experts'], item['top_k'])
            for item in active_specifications)
        self.soc_name = _soc_name(torch.device(tensor.device)) if calibrated else ''
        self.rank_nodes = (0,) * self.spec.ep_size
        if dist.is_initialized():
            self._validate_distributed_configuration(active_specifications)
        self.spec = qualify_topology_schedule(self.spec, self.rank_nodes)
        reservations = tuple(
            (item, _workspace_layout(
                {**item, "expert_weight_layout": expert_weight_layout} if item is specification else item,
                tensor.dtype,
            ))
            for item in active_specifications
        )
        self.heap_manager = _get_heap_manager(self.spec.ep_group, tensor)
        added_specifications = self.heap_manager.reserve(
            reservations,
            allow_resize=any(
                (
                    item.get("dispatch_mode", "push") == "push" or item.get("auto_tokens", False)
                    for item in active_specifications
                )
            ),
        )
        attached = False
        self.plan = None
        self.workspace = None
        try:
            attached = self.heap_manager.attach_runtime()

            def construct() -> None:
                """Build and bind local resources before converging construction results."""
                self.plan = build_mega_moe_plan(self.spec, tensor.device) if not self.spec.dynamic_tokens else None
                self.home_spec = None
                self.home_plan = None
                if self.spec.replica_slots_per_rank:
                    home_spec = replace(self.spec, num_experts=self.spec.logical_num_experts, replica_slots_per_rank=0)
                    if mega_moe_event_capacity(home_spec.num_experts, home_spec.ep_size) == mega_moe_event_capacity(
                        self.spec.num_experts, self.spec.ep_size
                    ):
                        self.home_spec = home_spec
                        if self.plan is not None:
                            self.home_plan = build_mega_moe_plan(home_spec, tensor.device)
                self.last_execution_plan = self.plan
                self.workspace = MegaMoeWorkspace(shared=shared)
                self.workspace.capacity_floor = self.heap_manager.bind(self.workspace, self.spec, specification)
                self.workspace.heap_manager = self.heap_manager
                self.spec = self.heap_manager.workspace_spec(self.workspace)
                self._token_capacity = self.spec.local_num_tokens

            _collective.collective_call(
                construct, stage="MegaMoe resource construction", group=self.heap_manager.root_group
            )
        except Exception:
            self.heap_manager.rollback_reservations(added_specifications, self.workspace)
            if attached and (not self.heap_manager.workspaces):
                self.heap_manager.detach_runtime()
            raise
        self._closed = False
        self._workspace_closed = False

    def ensure_capacity(self, maximum_received_slots: int) -> None:
        """Submit a larger layout, falling back to minimum capacity within an explicit budget.

        Args:
            maximum_received_slots: Maximum destination load from the root count exchange.
        """
        self.heap_manager.ensure_capacity(self.workspace, maximum_received_slots)

    def lifecycle_signature(self) -> tuple[Any, ...]:
        """Identify allocations independently of process-local group addresses."""
        return (
            "mega_moe", self.spec.local_num_tokens, self.spec.hidden_size,
            self.spec.intermediate_size, self.spec.num_experts, self.spec.top_k,
            self.spec.receive_capacity, self.spec.ep_size, self.spec.dispatch_mode,
            self.spec.expert_weight_layout, self.spec.apply_probs_before_fc2,
            self.spec.initial_capacity_factor, self.spec.capacity_growth_factor,
            self.spec.dispatch_split, self.spec.combine_split, self.spec.max_local_num_tokens, self.spec.auto_tokens,
            self.spec.forward_ingress_mode, self.spec.backward_ingress_mode,
            self.spec.shared_intermediate_size, self.spec.shared_placement,
            self.spec.forward_shared_expert, self.spec.backward_shared_expert,
            self.workspace.udma_qp_count,
        )

    def prepare_auto_workspace(
        self, tokens: int, received: int, tensor: torch.Tensor, *, replica_slots: int = 0
    ) -> None:
        """Commit physical capacity before taking a pull lease and publishing source rows.

        Args:
            tokens: Aligned group-maximum actual input length.
            received: Group-maximum received route rows.
            tensor: Input supplying device and dtype.
            replica_slots: EP-agreed guest matrix prefix for this invocation.
        """
        physical_spec = self.heap_manager.ensure_token_capacity(
            self.workspace, tokens, received, replica_slots=replica_slots
        )
        self.workspace.ensure(physical_spec, tensor.dtype, tensor.device)
        self._token_capacity = physical_spec.local_num_tokens
        if self.spec.dispatch_mode == "pull":
            self.workspace.claim()

    def receive_capacity_limit(self, tokens: int) -> int:
        """Query the allocator's common Push growth limit before selecting a route."""
        return self.heap_manager.receive_capacity_limit(self.workspace, tokens)

    def can_close(self) -> bool:
        """Report whether neither an active call nor a backward graph needs buffers."""
        return self._workspace_closed or self.workspace.can_close()

    def _validate_distributed_configuration(self, specifications: tuple[Mapping[str, Any], ...]) -> None:
        """Reject different allocation manifests before acquiring symmetric memory."""
        if self.spec.ep_size == 1:
            return
        fields = (
            "local_num_tokens",
            "max_local_num_tokens",
            "hidden_size",
            "intermediate_size",
            "num_experts",
            "top_k",
            "initial_capacity_factor",
            "capacity_growth_factor",
            "dispatch_mode",
            "dispatch_split",
            "combine_split",
            "swiglu_limit",
            "clamp_after_activation",
            "apply_probs_before_fc2",
            "expert_weight_layout",
            "ep_size",
            "auto_tokens",
            "_host_replica_metadata",
            "_replica_inbox_slots",
            "_gradient_queue_order",
            "_cube_event_refresh_us",
            "_vector_event_refresh_us",
            "_batch_empty_dispatch",
            "_batch_empty_combine",
            "_compact_communication",
            "_selective_owner_gradient_init",
            "_fuse_saved_dispatch",
            "_saved_dispatch_lanes",
            "_local_gradient_bf16",
            "_final_gradient_bf16",
            "_summary_home_admission",
            "replica_slots_per_rank",
            "replica_transport",
            "_replica_cost_model_enabled", "_replica_critical_path_identity",
            "forward_ingress_mode", "backward_ingress_mode", "shared_intermediate_size",
            "shared_placement", "forward_shared_expert", "backward_shared_expert",
            "compute_stages", "mx_vector_policy", "mx_poll_policy", "mx_swiglu_row_only", "mx_ready_depth",
            "mx_peer_qp0",
        )
        manifest = (
            "dynamic_tokens_v4",
            "bfloat16",
            self.spec.expert_weight_layout,
            tuple((tuple((spec.get(key) for key in fields)) for spec in specifications)),
            getattr(self, "soc_name", ""),
        )
        manifest += (replica_runtime_schedule(), configured_udma_qp_count())
        if any(spec.get("_replica_critical_path_identity") for spec in specifications):
            manifest += (_critical_native_fingerprint(),)
        manifest += (_local_node_identity(),)
        manifests = [None] * self.spec.ep_size
        dist.all_gather_object(manifests, manifest, group=self.spec.ep_group)
        if any((value[:-1] != manifests[0][:-1] for value in manifests)):
            raise ValueError("MegaMoe resource configurations differ across EP ranks")
        self.rank_nodes = _node_ids([value[-1] for value in manifests])

    def plan_for_tokens(self, tokens: int, device: Any, *, weight_grad_dtype: torch.dtype | None = None,
                        batch_experts: bool = False, nominal_schedule: bool = False,
                        home_only: bool = False, maximum_edge_rows: int | None = None) -> Any:
        """Reuse one covering plan per geometric bucket within committed capacity.

        Args:
            tokens: EP group maximum input rows, already aligned to 128.
            device: Execution device used for the parameters and workspace.
            weight_grad_dtype: Optional FP32 native output for partial expert weight
                gradients. Autograd and the caller control the final gradient dtype.
            batch_experts: Internal override for batched expert tasks.
            nominal_schedule: Internal override for compact lane selection.

            home_only: Use the original expert topology without guest slots.
            maximum_edge_rows: Validated EP-wide bound on a source-to-expert segment.

        Returns:
            Immutable plan covering the requested rows without enlarging storage.
        """
        spec = self.home_spec if home_only else self.spec
        if spec is None:
            raise ValueError("MegaMoe home-only execution requires compatible persistent event offsets")
        if tokens > self._token_capacity:
            raise RuntimeError(
                "MegaMoe schedule exceeds committed token capacity: "
                f"requested={tokens}, capacity={self._token_capacity}"
            )
        bucket = 1 << (max(128, tokens) - 1).bit_length()
        exact_rows = spec.forward_ingress_mode != "legacy" or bool(spec.shared_intermediate_size)
        key = ("a5_rows", tokens) if exact_rows else bucket
        if weight_grad_dtype is not None:
            key = (key, weight_grad_dtype)
        if home_only:
            key = (key, "home_only")
        if batch_experts:
            key = (key, "batched_experts")
        if nominal_schedule:
            key = (key, "nominal_schedule")
        communication_rows = None
        if spec._compact_communication and maximum_edge_rows is not None:
            if not 0 <= maximum_edge_rows <= tokens:
                raise ValueError("MegaMoe route exceeds the schedule's per-edge token capacity")
            edge_bucket = 1 << (max(_COMMUNICATION_SPLIT, maximum_edge_rows) - 1).bit_length()
            if edge_bucket < tokens:
                communication_rows = edge_bucket
                key = (key, edge_bucket)
        plan = self._plans.get(key)
        if plan is None or plan.spec.plan_tokens < tokens:
            schedule_tokens = min(bucket, self._token_capacity)
            options = {} if weight_grad_dtype is None else {"weight_grad_dtype": weight_grad_dtype}
            execution_spec = replace(spec, local_num_tokens=tokens, schedule_tokens=tokens) if exact_rows else spec
            plan = build_mega_moe_plan(
                replace(execution_spec, schedule_tokens=tokens if exact_rows else schedule_tokens,
                        _communication_rows=communication_rows, batch_experts=batch_experts,
                        nominal_schedule=nominal_schedule), device, **options)
            # INT32 route bounds allow at most 25 buckets starting at 128 rows.
            # A growing partial bucket replaces its plan; backward keeps the old object.
            self._plans[key] = plan
            if exact_rows and len(self._plans) > _MAX_A5_ROW_PLANS:
                self._plans.pop(next(iter(self._plans)))
        return plan

    def close(self) -> None:
        """Release the workspace and leave the shared SHMEM lifecycle."""
        if self._closed:
            return
        if not self._workspace_closed:
            self.heap_manager.close_workspace(self.workspace)
            self._workspace_closed = True
        self._plans.clear()
        self._closed = True


class A5MegaMoeExperts(MulticoreModule):
    """Execute Router-selected local experts with the Torch MegaMoe kernel.

    The caller owns expert parameters and passes their current local tensors to
    each forward. Serial layers may share a workspace before their first forward.
    """

    _expose_legacy_options = True

    def _resource_signature(self, tensor: Any) -> Any:
        """Compare static configuration without process-local group identities."""
        configuration = tuple(sorted((name, value) for name, value in self._resource_group.specification.items()
                                     if name != "ep_group"))
        return (super()._resource_signature(tensor), configuration,
                self._resource_group.shared, self._pending_layout)

    def _register_runtime_release(self, manager: Any, resources: Any) -> None:
        """Register the Heap Manager's sole Runtime reference for final cleanup."""
        manager.runtime_release = resources.heap_manager.detach_runtime

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        ep_group: Any | None = None,
        numerics_options: MegaMoeNumericsOptions | None = None,
        tuning_options: MegaMoeTuningOptions | None = None,
        swiglu_limit: float | None = None,
        ep_size: int | None = None,
        create_parameters: bool = True,
        expert_weight_layout: ExpertWeightLayout | None = None,
        dispatch_mode: str | None = None,
        initial_capacity_factor: float | None = None,
        capacity_growth_factor: float | None = None,
        dispatch_split: int | None = None,
        combine_split: int | None = None,
        replica_slots_per_rank: int | None = None,
        replica_transport: str = "shmem_signal_kernel_gradient_bands",
        replica_min_rows: int = 1024,
        replica_planner: str = "cpu",
        replica_target_load: int | None = None,
        replica_minimum_load_reduction: float | None = None,
        forward_ingress_mode: str = "legacy",
        backward_ingress_mode: str = "legacy",
        shared_intermediate_size: int = 0,
        shared_placement: str = "idle",
        forward_shared_expert: bool = True,
        backward_shared_expert: bool = False,
    ) -> None:
        """Initialize a parameterless lazy execution owner.

        Args:
            hidden_size: Input and output hidden dimension.
            intermediate_size: SwiGLU intermediate dimension per expert.
            num_experts: Global routed-expert count.
            top_k: Experts selected for every token.
            ep_group: Explicit EP group; expert ownership follows group-local rank order.
                Disjoint PP/DP groups bootstrap independently. Only one ordered EP membership
                can be active in SHMEM per process. Its size determines the local expert count.
                May be omitted for a standalone single-rank process.
            numerics_options: Stable activation and probability semantics, imported
                from ``hyper_multicore``. None selects ordinary SwiGLU and after-FC2
                weighting. Options specify the clamp limit and its position relative
                to SiLU, plus the probability position relative to FC2. The before-FC2
                path rounds activations to BF16 both before and after FP32 weighting.
            tuning_options: Public experimental tuning controls, imported from
                ``hyper_multicore.experimental``. None selects the default policy.
                Options are validated and copied into the instance configuration;
                they do not change the chosen probability or clamp semantics.

        Numerical choices and tuning options are fixed for this instance.
        Construct another executor to change them.
        """
        if numerics_options is not None and not isinstance(numerics_options, MegaMoeNumericsOptions):
            raise TypeError("numerics_options must be MegaMoeNumericsOptions or None")
        if tuning_options is not None and not isinstance(tuning_options, MegaMoeTuningOptions):
            raise TypeError("tuning_options must be MegaMoeTuningOptions or None")
        numerics = MegaMoeNumericsOptions() if numerics_options is None else numerics_options
        tuning = MegaMoeTuningOptions() if tuning_options is None else tuning_options
        for name, value in (("apply_probs_before_fc2", numerics.apply_probs_before_fc2),
                            ("clamp_after_silu", numerics.clamp_after_silu)):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")
        if swiglu_limit is not None:
            if numerics_options is not None and numerics.swiglu_limit != swiglu_limit:
                raise ValueError("swiglu_limit conflicts with numerics_options")
            numerics = replace(numerics, swiglu_limit=swiglu_limit)
        direct = {"dispatch_mode": dispatch_mode, "initial_capacity_factor": initial_capacity_factor,
                  "capacity_growth_factor": capacity_growth_factor, "dispatch_split": dispatch_split,
                  "combine_split": combine_split, "replica_slots_per_rank": replica_slots_per_rank}
        direct = {name: value for name, value in direct.items() if value is not None}
        if tuning_options is not None and direct:
            raise ValueError("Pass tuning_options or direct legacy tuning arguments, not both")
        if direct:
            tuning = replace(tuning, **direct)
        actual_ep_size = _resolve_ep_size(ep_group)
        if ep_size is not None:
            if type(ep_size) is not int or ep_size <= 0:
                raise ValueError("ep_size must be a positive integer or None")
            if ep_group is not None and ep_size != actual_ep_size:
                raise ValueError("ep_size must match the inferred EP group size")
        else:
            ep_size = actual_ep_size
        replica_slots_per_rank = tuning.replica_slots_per_rank
        dispatch_mode = tuning.dispatch_mode
        replica_config = ExpertReplicaConfig(num_experts, ep_size, replica_slots_per_rank)
        if replica_transport not in ("p2p", *KERNEL_GRADIENT_TRANSPORT_MODES):
            raise ValueError("Unsupported replica_transport; expected p2p or a kernel-gradient transport")
        _integer(replica_min_rows, "replica_min_rows", 0)
        if replica_target_load is not None:
            _integer(replica_target_load, "replica_target_load", 1)
        validate_planner_backend(replica_planner, replica_min_rows, None)
        self._cost_model_enabled = (_cost_model_enabled()
                                    if replica_slots_per_rank and replica_planner == 'cpu' else False)
        self.replica_planner = replica_planner
        self.replica_min_rows = replica_min_rows
        self.replica_target_load = replica_target_load
        if replica_minimum_load_reduction is not None:
            reduction_basis_points(replica_minimum_load_reduction)
        self.replica_minimum_load_reduction = replica_minimum_load_reduction
        self.replica_transport = replica_transport
        if (isinstance(shared_intermediate_size, bool) or not isinstance(shared_intermediate_size, int)
                or shared_intermediate_size < 0):
            raise ValueError("shared_intermediate_size must be a nonnegative integer")
        if shared_placement not in ("idle", "front"):
            raise ValueError("shared_placement must be idle or front")
        if not isinstance(forward_shared_expert, bool) or not isinstance(backward_shared_expert, bool):
            raise ValueError("forward_shared_expert and backward_shared_expert must be booleans")
        if backward_ingress_mode not in ("legacy", "indexed"):
            raise ValueError("backward_ingress_mode must be legacy or indexed")
        if backward_ingress_mode == "indexed":
            if not _supports_indexed_backward_pair(forward_ingress_mode, dispatch_mode, replica_slots_per_rank):
                raise ValueError("Indexed backward requires indexed or B0 peer token/tile forward with static pull")
            if hidden_size > 8192 or hidden_size % 32:
                raise ValueError("Indexed backward ingress requires hidden_size divisible by 32 and at most 8192")
            if top_k > 32:
                raise ValueError("Indexed backward ingress currently supports top_k at most 32")
        if forward_ingress_mode not in ("legacy", "indexed", "peer_dedup", "peer_overlap", "peer_token", "peer_tile"):
            raise ValueError("forward_ingress_mode must be legacy, indexed, peer_dedup, "
                             "peer_overlap, peer_token or peer_tile")
        if forward_ingress_mode != "legacy":
            if dispatch_mode != "pull":
                raise ValueError("Indexed forward ingress requires static pull dispatch")
            if forward_ingress_mode in ("peer_dedup", "peer_overlap", "peer_token", "peer_tile") and top_k > 16:
                raise ValueError("Peer dedup ingress currently supports top_k at most 16")
            if forward_ingress_mode in ("peer_overlap", "peer_token", "peer_tile") and num_experts // ep_size > 64:
                raise ValueError("Peer overlap supports at most 64 local experts")
        self._summary_home_admission = summary_admission(replica_slots_per_rank)
        self._summary_home_bypasses = 0
        specification = {
            "local_num_tokens": _COMMUNICATION_SPLIT,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": replica_config.physical_experts,
            "logical_num_experts": num_experts,
            "replica_slots_per_rank": replica_slots_per_rank,
            "replica_transport": self.replica_transport,
            "top_k": top_k,
            "swiglu_limit": numerics.swiglu_limit,
            "clamp_after_activation": numerics.clamp_after_silu,
            "apply_probs_before_fc2": numerics.apply_probs_before_fc2,
            "ep_size": ep_size,
            "ep_group": ep_group,
            "dispatch_mode": tuning.dispatch_mode,
            "initial_capacity_factor": tuning.initial_capacity_factor,
            "capacity_growth_factor": tuning.capacity_growth_factor,
            "dispatch_split": tuning.dispatch_split,
            "combine_split": tuning.combine_split,
            "_host_replica_metadata": True,
            "_summary_home_admission": self._summary_home_admission,
            "forward_ingress_mode": forward_ingress_mode,
            "backward_ingress_mode": backward_ingress_mode,
            "shared_intermediate_size": shared_intermediate_size,
            "shared_placement": shared_placement,
            "forward_shared_expert": forward_shared_expert,
            "backward_shared_expert": backward_shared_expert,
        }
        specification.update(_execution_options(tuning.dispatch_mode, replica_slots_per_rank))
        specification["auto_tokens"] = True
        if replica_slots_per_rank:
            specification['_replica_cost_model_enabled'] = self._cost_model_enabled
            if self._cost_model_enabled and _critical_path_available(
                    dispatch_mode, hidden_size, intermediate_size, ep_size,
                    replica_transport, num_experts, top_k):
                specification['_replica_critical_path_identity'] = _critical_path_identity(
                    dispatch_mode, hidden_size, intermediate_size, ep_size,
                    replica_transport, num_experts, top_k)
        if expert_weight_layout is not None or create_parameters:
            specification["expert_weight_layout"] = expert_weight_layout or "EIO"
        specification = self._resolve_specification(specification)
        self._initialize_execution_specification(specification, replica_config, create_parameters)

    def _initialize_execution_specification(
        self, specification: dict[str, Any], replica_config: ExpertReplicaConfig, create_parameters: bool,
    ) -> None:
        """Register the validated BF16 or MX owner before allocating any parameters."""
        ep_group = specification["ep_group"]
        compatibility_key = tuple(
            id(value) if name == "ep_group" else value for name, value in specification.items())
        super().__init__(
            resource_specification=specification,
            resource_compatibility_key=(compatibility_key, None),
            resource_scope_key=("mega_moe", shmem.root_members(ep_group) if dist.is_initialized() else id(ep_group)),
        )
        self._base_compatibility_key = compatibility_key
        self._pending_layout = None
        self.replica_config = replica_config
        self._rejected_quota_memo = _RejectedQuotaMemo() if replica_config.replica_slots_per_rank else None
        self.replica_slots_per_rank = replica_config.replica_slots_per_rank
        self._initialize_module_state(specification, create_parameters=create_parameters)

    def _initialize_module_state(
        self, specification: Mapping[str, Any], *, create_parameters: bool = False,
    ) -> None:
        """Bind configuration and optionally create local expert parameters."""
        self.hidden_size = specification["hidden_size"]
        self.intermediate_size = specification["intermediate_size"]
        self.num_experts = self.replica_config.num_experts
        self.top_k = specification["top_k"]
        self._dispatch_mode = specification["dispatch_mode"]
        if self._expose_legacy_options:
            self.expert_weight_layout = specification.get("expert_weight_layout")
            self.initial_capacity_factor = specification["initial_capacity_factor"]
            self.capacity_growth_factor = specification["capacity_growth_factor"]
            self.dispatch_mode = specification["dispatch_mode"]
            self.dispatch_split = specification["dispatch_split"]
            self.combine_split = specification["combine_split"]
        self.shared_intermediate_size = specification.get("shared_intermediate_size", 0)
        self.ep_size = specification["ep_size"]
        self.local_experts = self.num_experts // self.ep_size
        self._ep_group = specification["ep_group"]
        local, hidden, intermediate = self.local_experts, self.hidden_size, self.intermediate_size
        self._weight_shapes = {
            "EIO": ((local, hidden, intermediate * 2), (local, intermediate, hidden)),
            "EOI": ((local, intermediate * 2, hidden), (local, hidden, intermediate)),
        }
        if create_parameters:
            self.gate_up_weight, self.down_weight = _create_mega_moe_parameters(
                local, hidden, intermediate, specification.get("expert_weight_layout") or "EIO")
        else:
            self.register_parameter("gate_up_weight", None)
            self.register_parameter("down_weight", None)

    @property
    def swiglu_limit(self) -> float | None:
        """Return the fixed clamp limit, or None for ordinary SwiGLU."""
        return self._resource_group.specification["swiglu_limit"]

    @property
    def clamp_after_silu(self) -> bool:
        """Report whether the gate clamp follows SiLU, before multiplication by up."""
        return self._resource_group.specification["clamp_after_activation"]

    @property
    def apply_probs_before_fc2(self) -> bool:
        """Report the fixed position of routing-probability multiplication."""
        return self._resource_group.specification["apply_probs_before_fc2"]

    @classmethod
    def _resolve_specification(cls, specification: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize configuration before registering an execution owner."""
        expert_weight_layout = specification.get("expert_weight_layout")
        if expert_weight_layout is not None and expert_weight_layout not in _EXPERT_WEIGHT_LAYOUTS:
            raise ValueError("expert_weight_layout must be 'EIO', 'EOI' or None")
        if (cls._expose_legacy_options and expert_weight_layout == "EOI"
                and specification.get("replica_slots_per_rank", 0)):
            raise ValueError("Hot replicas require expert_weight_layout='EIO'; EOI is supported with B0")
        initial_capacity_factor, capacity_growth_factor = _resolve_capacity_factors(
            specification["dispatch_mode"],
            specification["initial_capacity_factor"],
            specification["capacity_growth_factor"],
        )
        cls._validate_topology(
            local_num_tokens=specification["local_num_tokens"],
            hidden_size=specification["hidden_size"],
            intermediate_size=specification["intermediate_size"],
            num_experts=specification.get("logical_num_experts", specification["num_experts"]),
            top_k=specification["top_k"],
            swiglu_limit=specification["swiglu_limit"],
            ep_size=specification["ep_size"],
            dispatch_split=specification["dispatch_split"],
            combine_split=specification["combine_split"],
            auto_tokens=specification.get("auto_tokens", False),
        )
        swiglu_limit = specification["swiglu_limit"]
        return {
            **specification,
            "initial_capacity_factor": initial_capacity_factor,
            "capacity_growth_factor": capacity_growth_factor,
            "swiglu_limit": float(swiglu_limit) if swiglu_limit is not None else None,
        }

    @staticmethod
    def _validate_topology(
        *,
        local_num_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        swiglu_limit: float | None,
        ep_size: int,
        dispatch_split: int,
        combine_split: int,
        auto_tokens: bool = False,
    ) -> None:
        """Validate static shape and topology values before allocation."""
        values = {
            "local_num_tokens": local_num_tokens,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": num_experts,
            "top_k": top_k,
            "ep_size": ep_size,
        }
        for name, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if top_k > num_experts:
            raise ValueError(f"top_k ({top_k}) cannot exceed num_experts ({num_experts}).")
        if num_experts % ep_size:
            raise ValueError(f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size}).")
        _validate_swiglu_limit(swiglu_limit)
        for name, split in (("dispatch_split", dispatch_split), ("combine_split", combine_split)):
            if not isinstance(split, int) or isinstance(split, bool) or split <= 0 or split % _COMMUNICATION_SPLIT:
                raise ValueError(
                    f"{name} must be a positive integer multiple of {_COMMUNICATION_SPLIT}, got {split!r}."
                )
            if not auto_tokens and local_num_tokens % split:
                raise ValueError(f"local_num_tokens ({local_num_tokens}) must be divisible by {name} ({split}).")

    def _root_group(self) -> Any:
        """Keep lifecycle coordination within this module's EP group."""
        return self._ep_group

    def _infer_weight_layout(self, shape1: tuple[int, ...], shape2: tuple[int, ...]) -> str:
        """Classify the first pair using both logical shapes, including square matrices."""
        actual = (shape1, shape2)
        if actual == self._weight_shapes["EIO"]:
            return "EIO"
        if actual == self._weight_shapes["EOI"]:
            return "EOI"
        raise _InvalidExpertWeights(
            f"expert_weights must have EIO shapes {self._weight_shapes['EIO']} or "
            f"EOI shapes {self._weight_shapes['EOI']}, got {actual}.")

    def _validate_tensors(self, hidden_states: torch.Tensor, expert_weights: tuple) -> None:
        """Validate activation and parameter metadata before acquiring resources."""
        if hidden_states.dtype != torch.bfloat16 or not getattr(hidden_states, "is_npu", False):
            raise TypeError("MegaMoeExperts requires BF16 NPU hidden states.")
        try:
            weight1, weight2 = expert_weights
            shape1, shape2 = tuple(weight1.shape), tuple(weight2.shape)
        except (AttributeError, TypeError, ValueError) as error:
            raise _InvalidExpertWeights("expert_weights must contain two tensors") from error
        resources = self._resource_group.resources
        declared = self._resource_group.specification.get("expert_weight_layout") if resources is None else None
        layout = (declared or self._infer_weight_layout(shape1, shape2)) if resources is None else resources.spec.expert_weight_layout
        expected_weight1, expected_weight2 = self._weight_shapes[layout]
        if shape1 != expected_weight1:
            raise _InvalidExpertWeights(f"gate_up_weight must have shape {expected_weight1}, got {shape1}.")
        if shape2 != expected_weight2:
            raise _InvalidExpertWeights(f"down_weight must have shape {expected_weight2}, got {shape2}.")
        if (
            weight1.device != hidden_states.device
            or weight2.device != hidden_states.device
        ):
            raise _InvalidExpertWeights(
                "expert weights and hidden states must be on the same NPU."
            )
        if weight1.dtype != hidden_states.dtype or weight2.dtype != hidden_states.dtype:
            raise _InvalidExpertWeightType("expert weights and hidden states must all use BF16.")
        self._validate_layout_storage(layout, weight1, weight2)
        if resources is None:
            self._pending_layout = layout
            self._resource_group.compatibility_key = (self._base_compatibility_key, layout)

    def _validate_layout_storage(self, layout: str, weight1: torch.Tensor, weight2: torch.Tensor) -> None:
        """Require dense EOI weights for zero-copy expert and replica execution."""
        if layout == "EOI" and (
            not weight1.is_contiguous() or not weight2.is_contiguous()
        ):
            raise _InvalidExpertWeights("EOI expert weights must be contiguous for zero-copy execution.")

    def _validate_forward_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        tokens_per_expert: torch.Tensor | None,
        expert_weights: tuple,
    ) -> torch.Tensor:
        """Validate input shapes and return flattened local token states."""
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "hidden_states must have at least two dimensions and configured "
                f"hidden size {self.hidden_size}, got {tuple(hidden_states.shape)}."
            )
        hidden_flat = hidden_states.reshape(-1, self.hidden_size)
        route_shape = (hidden_flat.shape[0], self.top_k)
        if tuple(topk_ids.shape) != route_shape:
            raise ValueError(f"topk_ids must have shape {route_shape}, got {tuple(topk_ids.shape)}.")
        if tuple(topk_weights.shape) != route_shape:
            raise ValueError(f"topk_weights must have shape {route_shape}, " f"got {tuple(topk_weights.shape)}.")
        if tokens_per_expert is not None and tuple(tokens_per_expert.shape) != (self.num_experts,):
            raise ValueError(
                f"tokens_per_expert must have shape ({self.num_experts},), " f"got {tuple(tokens_per_expert.shape)}."
            )
        self._validate_tensors(hidden_flat, expert_weights)
        return hidden_flat

    def _prepare_forward_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        tokens_per_expert: torch.Tensor | None,
        expert_weights: tuple,
    ) -> tuple[torch.Tensor, bool]:
        """Defer rank-local weight errors until the binding or route collective."""
        try:
            return self._validate_forward_inputs(
                hidden_states, topk_ids, topk_weights, tokens_per_expert, expert_weights,
            ), False
        except (_InvalidExpertWeights, _InvalidExpertWeightType):
            if self.ep_size == 1 or not dist.is_initialized():
                raise
            # Every rank must reach the existing binding/route collective before
            # rejecting a rank-local weight error; otherwise peers may hang.
            if self._resource_group.resources is None:
                self._pending_layout = None
                self._resource_group.compatibility_key = (self._base_compatibility_key, None)
            return hidden_states.reshape(-1, self.hidden_size), True

    def _summary_keeps_home(
        self, resources: _MegaMoeExecutionResources, summary: list[list[int]], loads: list[int],
        admission: int | None = None,
    ) -> bool:
        """Keep an already feasible home layout if even perfect balance cannot be admitted."""
        admission = self._summary_home_admission if admission is None else min(self._summary_home_admission, admission)
        if not admission:
            return False
        peak = max(loads)
        if peak > 2**31 - 1:
            return False
        lower_bound = (sum(loads) + len(loads) - 1) // len(loads)
        if optional_peak_gain_is_sufficient(peak, lower_bound, admission, 0):
            return False
        if resources.spec.dispatch_mode == "push":
            planned_tokens = max(128, _align_capacity(max((row[1] for row in summary))))
            if planned_tokens > resources._token_capacity or peak > resources.workspace.capacity_floor:
                return False
        state = get_recompute_state()
        if state is not None and state.is_recomputing and (not _has_home_recompute_hint()):
            return False
        _record_home_decision()
        self._summary_home_bypasses += 1
        return True

    def _replica_route_policy(self, autograd_mask: int) -> tuple[float, int]:
        """Price complete training projections separately from no-gradient routes."""
        training = bool(autograd_mask & 1) or get_recompute_state() is not None
        projections = 2 * bool(autograd_mask & 4) + bool(autograd_mask & 8) if training else 0
        reduction = 0.25 if projections == 3 else resolve_replica_admission(None)[0]
        return reduction, self.intermediate_size * projections // 6

    def _prepare_replica_route(
        self, topk_ids: torch.Tensor, resources: _MegaMoeExecutionResources, hidden: torch.Tensor,
        autograd_mask: int, counts: torch.Tensor | None, gathered: _GatheredReplicaCounts | None = None,
        *, count_header_cache: _ReplicaCountHeaderCache | None = None, complete_experts: bool = False,
    ) -> ReplicaRoute:
        """Resolve a placement using current source counts and available calibrated policy."""
        reduction, row_credit = self._replica_route_policy(autograd_mask)
        selection = {"execution_schedule": UNMAPPED_GUEST_SUFFIX_SCHEDULE} if self.replica_slots_per_rank else {}
        if self._cost_model_enabled and _critical_path_available(
                self._dispatch_mode, self.hidden_size, self.intermediate_size, self.ep_size,
                self.replica_transport, self.num_experts, self.top_k):
            maximum = max(row[1] for row in gathered.source_summary) if gathered is not None else None
            selection["execution_signature"] = _critical_execution_signature(self, resources, execution_tokens=maximum)
        cost_model = _runtime_cost_model(self._cost_model_enabled, self._dispatch_mode, self.hidden_size,
            self.intermediate_size, self.ep_size, self.replica_transport, self.num_experts, self.top_k,
            hidden.device, **selection)
        if self.replica_minimum_load_reduction is not None:
            reduction = self.replica_minimum_load_reduction
        elif cost_model is not None:
            reduction = 0.0
        admission, saved_rows = resolve_replica_admission(reduction)
        if cost_model is not None:
            row_credit = 0
        mode = getattr(resources.spec, "dispatch_mode", self._dispatch_mode)
        target_load = self.replica_target_load
        if self._expose_legacy_options and target_load is None and mode == "push" and not resources.spec.auto_tokens:
            target_load = initial_receive_capacity(self._resource_group.specification)
        return prepare_replica_route(
            topk_ids, self.replica_config, self._ep_group,
            target_load=target_load, minimum_replica_rows=self.replica_min_rows,
            receive_capacity=(initial_receive_capacity(self._resource_group.specification)
                              if self._expose_legacy_options and mode == "push" and not resources.spec.auto_tokens
                              else None),
            minimum_load_reduction=admission, minimum_saved_rows=saved_rows,
            planner_backend=self.replica_planner, _cost_model=cost_model,
            _replica_transfer_row_credit=row_credit, _rank_nodes=resources.rank_nodes,
            compact_home=resources.home_spec is not None,
            dynamic_header=(int(hidden.shape[0] * max(self.hidden_size, self.top_k) > 2**31 - 1),
                            hidden.shape[0], autograd_mask),
            _auto_capacity_factor=(resources.spec.initial_capacity_factor
                                   if resources.spec.auto_tokens and mode == "push" else None),
            _capacity_budget=resources.receive_capacity_limit if mode == "push" else None,
            _trusted_counts=counts, _gathered_counts=gathered,
            _route_cache=getattr(resources.workspace, "replica_route_cache", None),
            _rejected_quota_memo=self._rejected_quota_memo if cost_model is not None or complete_experts else None,
            _count_header_cache=count_header_cache,
            _complete_experts=complete_experts,
        )

    def _replica_route_selector(self, resources, hidden, ids, trusted, mask, state):
        """Select replicas only after current group counts and controls are validated."""
        adaptive = getattr(resources.spec, "_replica_inbox_slots", None) is not None
        admission = reduction_basis_points(self._replica_route_policy(mask)[0])

        def select(
            counts: torch.Tensor, summary: list[list[int]], host_counts: _HostReplicaCounts | None
        ) -> tuple[MegaMoeSpec, ReplicaRoute] | None:
            """Preserve home call ordinals even when the full planner is bypassed."""
            loads = [row[6] for row in summary]
            if min(loads) == max(loads):
                _record_home_decision()
                return None
            if self._summary_keeps_home(resources, summary, loads, admission):
                return None
            gathered = _GatheredReplicaCounts(counts, summary, host_counts)
            proposal = self._prepare_replica_route(ids, resources, hidden, mask, trusted, gathered)
            if adaptive:
                state.inbox_slots = _required_replica_inbox_slots(proposal)
            return resources.home_spec if proposal.home_only else resources.spec, proposal

        return select

    def _prepare_forward_route(self, resources, hidden, ids, probabilities, counts, options, state):
        """Resolve physical experts while keeping B0's count and permutation path intact."""
        replica_route = None
        spec = resources.spec
        integrated = bool(self.replica_slots_per_rank and spec.auto_tokens and resources.home_spec is not None)
        if self.replica_slots_per_rank:
            if integrated:
                spec = resources.home_spec
                options["_select_replica_route"] = self._replica_route_selector(
                    resources, hidden, ids, counts, options["autograd_mask"], state
                )
            else:
                replica_route = self._prepare_replica_route(ids, resources, hidden, options["autograd_mask"], counts)
                if getattr(resources.spec, "_replica_inbox_slots", None) is not None:
                    state.inbox_slots = _required_replica_inbox_slots(replica_route)
                if replica_route.home_only:
                    state.home_only = True
                    spec = resources.home_spec
                    if not resources.spec.auto_tokens:
                        resources.workspace.ensure(resources.spec, hidden.dtype, hidden.device)
                ids = replica_route.physical_ids
                counts = replica_route.counts_by_source[resources.spec.rank_id]
        route = prepare_topk_route(
            hidden,
            ids,
            probabilities,
            spec,
            counts,
            workspace=resources.workspace,
            replica_route=replica_route,
            **options,
        )
        resources.last_empty_owner_experts = getattr(route.metadata, "empty_owner_experts", None)
        if integrated:
            replica_route = route.metadata.replica_route
            state.home_only = replica_route is None
            if replica_route is not None:
                ids = replica_route.physical_ids
        return route, ids

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        expert_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
        tokens_per_expert: torch.Tensor | None = None,
        shared_gate_up_weight: torch.Tensor | None = None,
        shared_down_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route token states through the configured experts.

        Args:
            hidden_states: Tensor ending in ``hidden_size``.
            topk_ids: Global expert IDs shaped ``[tokens, top_k]`` for the current input.
            topk_weights: Router weights with the same shape as ``topk_ids``.
            expert_weights: Current local, unsharded gate/up and down weights owned by
                the outer module: EIO ``([E,H,2I], [E,I,H])`` or EOI
                ``([E,2I,H], [E,H,I])``. Layout binds on first use and stays fixed.
                EIO copies retain autograd; EOI weights must be contiguous.
            tokens_per_expert: Trusted exact global-expert histogram; skips histogram recomputation.

        Note:
            Returns the input shape and dtype. EOI noncontiguity raises ValueError;
            graph capture raises RuntimeError. Both transports require eager execution;
            trusted counts must match ``topk_ids`` without rebuilding the histogram.
        """
        if self._resource_group.specification.get("forward_ingress_mode", "legacy") != "legacy":
            if topk_ids.numel() > 2**24:
                raise ValueError("Indexed scalar route sorting requires at most 2**24 route slots")
        weights = (self.gate_up_weight, self.down_weight) if expert_weights is None else expert_weights
        hidden_flat, weight_error = self._prepare_forward_inputs(
            hidden_states, topk_ids, topk_weights, tokens_per_expert, weights,
        )
        shared_weights = (shared_gate_up_weight, shared_down_weight)
        if self.shared_intermediate_size:
            expected_shapes = ((2 * self.shared_intermediate_size, self.hidden_size),
                               (self.hidden_size, self.shared_intermediate_size))
            for weight, expected in zip(shared_weights, expected_shapes):
                if weight is None or tuple(weight.shape) != expected:
                    raise ValueError(f"Shared weight must have shape {expected}")
                if (weight.device != hidden_flat.device or weight.dtype != hidden_flat.dtype
                        or not weight.is_contiguous()):
                    raise ValueError("Shared weights must be contiguous and match the input device and dtype")
        elif any(weight is not None for weight in shared_weights):
            raise ValueError("Shared weights require shared_intermediate_size at construction")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("MegaMoe push heap growth and dynamic tokens do not support graph capture")
        resources = self._get_execution_resources(hidden_flat)
        _checkpoint_contract()
        route_options = {"autograd_mask": _INVALID_WEIGHT_MASK if weight_error else (
            int(torch.is_grad_enabled()) | (int(hidden_flat.requires_grad) << 1)
            | (int(weights[0].requires_grad) << 2) | (int(weights[1].requires_grad) << 3)
            | (int(topk_weights.requires_grad) << 4)
        )}
        pull = getattr(resources.spec, "dispatch_mode", self._dispatch_mode) == "pull"
        before_fc2 = getattr(resources.spec, "apply_probs_before_fc2", False)
        automatic = resources.spec.auto_tokens
        claimed = False
        route_state = _ForwardRouteState()
        adaptive_inbox = getattr(resources.spec, "_replica_inbox_slots", None) is not None

        def prepare_workspace(tokens: int, received: int) -> None:
            """Grow before publishing pull source rows into symmetric memory.

            Args:
                tokens: Current local source-token count.
                received: Current routed receive-row count.
            """
            nonlocal claimed
            if adaptive_inbox:
                resources.prepare_auto_workspace(tokens, received, hidden_flat, replica_slots=route_state.inbox_slots)
            else:
                resources.prepare_auto_workspace(tokens, received, hidden_flat)
            claimed = pull

        if automatic:
            route_options["prepare_workspace"] = prepare_workspace
        elif pull:
            resources.workspace.ensure(resources.spec, hidden_flat.dtype, hidden_flat.device)
            resources.workspace.claim()
            claimed = True
        try:
            with torch.no_grad():
                route, topk_ids = self._prepare_forward_route(
                    resources, hidden_flat, topk_ids, topk_weights, tokens_per_expert, route_options, route_state
                )
            if not pull and (not automatic):
                resources.ensure_capacity(route.maximum_received_slots)
            execution_plan = resources.plan_for_tokens(
                route.metadata.plan_tokens, hidden_flat.device, home_only=route_state.home_only,
                maximum_edge_rows=getattr(route.metadata, "maximum_edge_rows", None),
            )
            resources.last_execution_plan = execution_plan
            if not torch.is_grad_enabled() and not self.shared_intermediate_size and route.routed_tokens is not None:
                return _execute_mega_moe_without_autograd(
                    topk_weights, *weights, route, execution_plan, resources.workspace,
                    workspace_claimed=pull).reshape_as(hidden_states)
            expert_output = execute_mega_moe_with_permutation(
                hidden_flat,
                topk_ids,
                weights[0],
                weights[1],
                route,
                execution_plan,
                resources.workspace,
                topk_weights=topk_weights if pull or before_fc2 else None,
                workspace_claimed=pull,
                **({"shared_weight1": shared_gate_up_weight, "shared_weight2": shared_down_weight}
                   if self.shared_intermediate_size else {}),
            )
        finally:
            if claimed:
                resources.workspace.release()
        shared_output = None
        if self.shared_intermediate_size:
            expert_output, shared_output = expert_output
        if pull or before_fc2:
            output = expert_output.reshape_as(hidden_states)
        else:
            output = restore_topk_output(expert_output, route.unpermute_mapping, topk_weights)
            output = output.reshape(hidden_states.shape)
        return output if shared_output is None else output + shared_output.reshape_as(hidden_states)

    def _create_execution_resources(
        self,
        tensor: torch.Tensor,
        *,
        shared: bool,
        active_specifications: tuple[Mapping[str, Any], ...],
    ) -> _MegaMoeExecutionResources:
        """Create the plan and workspace owned by this resource group."""
        return _MegaMoeExecutionResources(
            self._resource_group.specification,
            tensor,
            expert_weight_layout=self._require_pending_layout(),
            shared=shared,
            active_specifications=active_specifications,
        )

    def _require_pending_layout(self) -> str:
        """Reject an invalid first weight pair after rank-wide binding agreement."""
        if self._pending_layout is None:
            raise ValueError("expert weights do not match a supported layout on all EP ranks")
        return self._pending_layout


class MegaMoeExperts(A5MegaMoeExperts):
    """Target-branch executor with caller-owned expert weights and typed options."""

    _expose_legacy_options = False

    def __init__(self, *, hidden_size: int, intermediate_size: int, num_experts: int, top_k: int,
                 ep_group: Any | None = None, numerics_options: MegaMoeNumericsOptions | None = None,
                 tuning_options: MegaMoeTuningOptions | None = None) -> None:
        super().__init__(hidden_size=hidden_size, intermediate_size=intermediate_size, num_experts=num_experts,
                         top_k=top_k, ep_group=ep_group, numerics_options=numerics_options,
                         tuning_options=tuning_options, create_parameters=False)

    def forward(self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor, *,
                expert_weights: tuple[torch.Tensor, torch.Tensor],
                tokens_per_expert: torch.Tensor | None = None) -> torch.Tensor:
        return super().forward(hidden_states, topk_ids, topk_weights, expert_weights=expert_weights,
                               tokens_per_expert=tokens_per_expert)
