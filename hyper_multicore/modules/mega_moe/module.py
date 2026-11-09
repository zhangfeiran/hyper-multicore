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

__all__ = ["MegaMoeExperts"]
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.distributed as dist

from hyper_multicore import _collective, shmem
from hyper_multicore.modules.mega_moe.numerics import MegaMoeNumericsOptions
from hyper_multicore.experimental import MegaMoeTuningOptions

from hyper_multicore.replica.admission import (
    optional_peak_gain_is_sufficient,
    reduction_basis_points,
    resolve_replica_admission,
)
from hyper_multicore.replica.capacity import ExpertReplicaConfig, _integer
from hyper_multicore.replica.recompute import get_recompute_state
from hyper_multicore.replica.signal_transport import KERNEL_GRADIENT_TRANSPORT_MODES
from hyper_multicore.replica.routing import (
    ReplicaRoute,
    _GatheredReplicaCounts,
    _has_home_recompute_hint,
    _HostReplicaCounts,
    _record_home_decision,
    prepare_replica_route,
)
from hyper_multicore.shmem._topology import _local_node_identity, _node_ids
from hyper_multicore.scheduler.config import mega_moe_event_capacity

from ..module import MulticoreModule
from .function import execute_mega_moe_with_permutation, _execute_mega_moe_without_autograd
from .heap_manager import _get_heap_manager, _workspace_layout
from .plan import build_mega_moe_plan
from .route import _INVALID_WEIGHT_MASK, RouteSchedule, prepare_topk_route, restore_topk_output
from .configuration import execution_options as _execution_options
from .configuration import qualify_topology_schedule, summary_admission
from .inactive_tasks import replica_runtime_schedule
from .spec import (
    _COMMUNICATION_SPLIT,
    MegaMoeSpec,
    _align_capacity,
    _resolve_capacity_factors,
    bind_mega_moe_spec,
)
from .workspace import MegaMoeWorkspace


class _InvalidExpertWeights(ValueError):
    """Expert metadata that must be rejected consistently across EP ranks."""


class _InvalidExpertWeightType(TypeError):
    """Expert dtype that must be rejected consistently across EP ranks."""


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
        expert_weight_layout: str,
        shared: bool,
        active_specifications: tuple[Mapping[str, Any], ...],
    ) -> None:
        """Bind resources once to the first NPU tensor."""
        bound_specification = {**specification, "expert_weight_layout": expert_weight_layout}
        self.spec = bind_mega_moe_spec(bound_specification, tensor)
        self._token_capacity = self.spec.local_num_tokens
        self._plans = {}
        self.rank_nodes = (0,) * self.spec.ep_size
        if dist.is_initialized():
            self._validate_distributed_configuration(active_specifications)
        self.spec = qualify_topology_schedule(self.spec, self.rank_nodes)
        reservations = tuple(
            (item, _workspace_layout({**item, "expert_weight_layout": expert_weight_layout}, tensor.dtype))
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
                    home_spec = replace(self.spec, num_experts=self.spec.logical_num_experts,
                                        replica_slots_per_rank=0, route_task_bounds=False)
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
            "swiglu_hf_bf16",
            "replica_granularity",
            "gate_up_expert_padding",
            "replica_min_rows",
            "replica_minimum_load_reduction",
            "single_expert_tasks",
            "route_task_bounds",
        )
        manifest = (
            "dynamic_tokens_v4",
            "bfloat16",
            self.spec.expert_weight_layout,
            tuple((tuple((spec.get(key) for key in fields)) for spec in specifications)),
            getattr(self, "soc_name", ""),
        )
        manifest += (replica_runtime_schedule(), _local_node_identity())
        manifests = [None] * self.spec.ep_size
        dist.all_gather_object(manifests, manifest, group=self.spec.ep_group)
        if any((value[:-1] != manifests[0][:-1] for value in manifests)):
            raise ValueError("MegaMoe resource configurations differ across EP ranks")
        self.rank_nodes = _node_ids([value[-1] for value in manifests])

    def plan_for_tokens(self, tokens: int, device: Any, *, weight_grad_dtype: torch.dtype | None = None,
                        batch_experts: bool = False, nominal_schedule: bool = False,
                        home_only: bool = False, maximum_edge_rows: int | None = None,
                        schedule: RouteSchedule | None = None) -> Any:
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
            schedule: Optional exact expert bounds from the validated route.

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
        key = bucket if weight_grad_dtype is None else (bucket, weight_grad_dtype)
        if home_only:
            key = (key, "home_only")
        if batch_experts:
            key = (key, "batched_experts")
        if nominal_schedule:
            key = (key, "nominal_schedule")
        schedule_tokens = min(bucket, self._token_capacity)
        bounds = {}
        if schedule is not None:
            if schedule.expert_rows > schedule_tokens * spec.ep_size or schedule.source_expert_rows > schedule_tokens:
                raise RuntimeError("MegaMoe route counts exceed supported per-expert task bounds")
            expert = min(schedule_tokens * spec.ep_size, 1 << (max(128, schedule.expert_rows) - 1).bit_length())
            source = min(schedule_tokens, 1 << (max(128, schedule.source_expert_rows) - 1).bit_length())
            key = (key, expert, source)
            bounds = {"schedule_expert_rows": expert, "schedule_source_expert_rows": source}
        communication_rows = None
        if spec._compact_communication and maximum_edge_rows is not None:
            if not 0 <= maximum_edge_rows <= tokens:
                raise ValueError("MegaMoe route exceeds the schedule's per-edge token capacity")
            edge_bucket = 1 << (max(_COMMUNICATION_SPLIT, maximum_edge_rows) - 1).bit_length()
            source_bound = bounds.get("schedule_source_expert_rows", schedule_tokens)
            if edge_bucket > source_bound:
                raise RuntimeError("MegaMoe communication bound exceeds the expert route bound")
            if edge_bucket < source_bound:
                communication_rows = edge_bucket
                key = (key, edge_bucket)
        plan = self._plans.get(key)
        if plan is None or plan.spec.plan_tokens < tokens:
            options = {} if weight_grad_dtype is None else {"weight_grad_dtype": weight_grad_dtype}
            plan = build_mega_moe_plan(
                replace(spec, schedule_tokens=schedule_tokens, _communication_rows=communication_rows,
                        batch_experts=batch_experts,
                        nominal_schedule=nominal_schedule, **bounds), device, **options)
            # INT32 route bounds allow at most 25 buckets starting at 128 rows.
            # A growing partial bucket replaces its plan; backward keeps the old object.
            self._plans[key] = plan
        if schedule is not None or communication_rows is not None:
            self._plans[key] = self._plans.pop(key)
            while len(self._plans) > 64:
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


class MegaMoeExperts(MulticoreModule):
    """Execute Router-selected local experts with the Torch MegaMoe kernel.

    The caller owns expert parameters and passes their current local tensors to
    each forward. Serial layers may share a workspace before their first forward.
    """

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
        ep_size = _resolve_ep_size(ep_group)
        replica_slots_per_rank = tuning.replica_slots_per_rank
        replica_granularity = tuning.replica_granularity
        single_expert_tasks = tuning.single_expert_tasks
        route_task_bounds = tuning.route_task_bounds
        replica_transport = tuning.replica_transport
        replica_min_rows = tuning.replica_min_rows
        replica_minimum_load_reduction = tuning.replica_minimum_load_reduction
        initial_token_capacity = tuning.initial_token_capacity
        gate_up_expert_padding = tuning.gate_up_expert_padding
        if replica_granularity not in ("token", "expert"):
            raise ValueError("replica_granularity must be 'token' or 'expert'")
        if not isinstance(single_expert_tasks, bool) or not isinstance(route_task_bounds, bool):
            raise ValueError("single_expert_tasks and route_task_bounds must be boolean")
        if replica_transport not in ("p2p", *KERNEL_GRADIENT_TRANSPORT_MODES):
            raise ValueError("Unsupported replica_transport")
        _integer(replica_min_rows, "replica_min_rows", 0)
        resolve_replica_admission(replica_minimum_load_reduction)
        _integer(initial_token_capacity, "initial_token_capacity", 1)
        _integer(gate_up_expert_padding, "gate_up_expert_padding", 0)
        if gate_up_expert_padding and (replica_slots_per_rank or gate_up_expert_padding % ep_size):
            raise ValueError("gate_up_expert_padding requires B0 and divisibility by EP")
        replica_config = ExpertReplicaConfig(num_experts, ep_size, replica_slots_per_rank)
        self.replica_min_rows = replica_min_rows
        self.replica_minimum_load_reduction = replica_minimum_load_reduction
        self.replica_transport = replica_transport
        self._summary_home_admission = summary_admission(replica_slots_per_rank, replica_minimum_load_reduction)
        self._summary_home_bypasses = 0
        specification = {
            "local_num_tokens": _align_capacity(initial_token_capacity),
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": replica_config.physical_experts,
            "logical_num_experts": num_experts,
            "replica_slots_per_rank": replica_slots_per_rank,
            "replica_transport": self.replica_transport,
            "replica_granularity": replica_granularity,
            "gate_up_expert_padding": gate_up_expert_padding,
            "replica_min_rows": replica_min_rows,
            "replica_minimum_load_reduction": replica_minimum_load_reduction,
            "top_k": top_k,
            "swiglu_limit": numerics.swiglu_limit,
            "clamp_after_activation": numerics.clamp_after_silu,
            "apply_probs_before_fc2": numerics.apply_probs_before_fc2,
            "swiglu_hf_bf16": numerics.swiglu_hf_bf16,
            "single_expert_tasks": single_expert_tasks,
            "route_task_bounds": route_task_bounds,
            "ep_size": ep_size,
            "ep_group": ep_group,
            "dispatch_mode": tuning.dispatch_mode,
            "initial_capacity_factor": tuning.initial_capacity_factor,
            "capacity_growth_factor": tuning.capacity_growth_factor,
            "dispatch_split": tuning.dispatch_split,
            "combine_split": tuning.combine_split,
            "_host_replica_metadata": True,
            "_summary_home_admission": self._summary_home_admission,
        }
        specification.update(_execution_options(tuning.dispatch_mode, replica_slots_per_rank))
        specification["auto_tokens"] = True
        specification = self._resolve_specification(specification)
        compatibility_key = tuple((id(value) if name == "ep_group" else value for name, value in specification.items()))
        super().__init__(
            resource_specification=specification,
            resource_compatibility_key=(compatibility_key, None),
            resource_scope_key=("mega_moe", shmem.root_members(ep_group) if dist.is_initialized() else id(ep_group)),
        )
        self._base_compatibility_key = compatibility_key
        self._pending_layout = None
        self.replica_config = replica_config
        self.replica_slots_per_rank = replica_slots_per_rank
        self.replica_granularity = replica_granularity
        self.gate_up_expert_padding = gate_up_expert_padding
        self.initial_token_capacity = specification["local_num_tokens"]
        self._initialize_module_state(specification)

    def _initialize_module_state(
        self, specification: Mapping[str, Any],
    ) -> None:
        """Bind normalized configuration and precompute valid weight shapes."""
        self.hidden_size = specification["hidden_size"]
        self.intermediate_size = specification["intermediate_size"]
        self.num_experts = self.replica_config.num_experts
        self.top_k = specification["top_k"]
        self.single_expert_tasks = specification.get("single_expert_tasks", False)
        self.route_task_bounds = specification.get("route_task_bounds", False)
        self.ep_size = specification["ep_size"]
        self.local_experts = self.num_experts // self.ep_size
        self._ep_group = specification["ep_group"]
        local, hidden, intermediate = self.local_experts, self.hidden_size, self.intermediate_size
        self._weight_shapes = {
            "EIO": ((local, hidden, intermediate * 2), (local, intermediate, hidden)),
            "EOI": ((local + self.gate_up_expert_padding // self.ep_size, intermediate * 2, hidden),
                    (local, hidden, intermediate)),
        }

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
        hf_bf16 = specification.get("swiglu_hf_bf16", False)
        if not isinstance(hf_bf16, bool) or (hf_bf16 and specification["swiglu_limit"] is not None):
            raise ValueError("swiglu_hf_bf16 must be boolean and cannot be combined with swiglu_limit")
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
        if resources is None:
            layout = self._infer_weight_layout(shape1, shape2)
        else:
            layout = resources.spec.expert_weight_layout
            expected_weight1, expected_weight2 = self._weight_shapes[layout]
            if shape1 != expected_weight1:
                raise _InvalidExpertWeights(
                    f"gate_up_weight must have shape {expected_weight1}, got {shape1}.")
            if shape2 != expected_weight2:
                raise _InvalidExpertWeights(
                    f"down_weight must have shape {expected_weight2}, got {shape2}.")
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
            if self.ep_size == 1:
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
        self,
        topk_ids: torch.Tensor,
        resources: _MegaMoeExecutionResources,
        hidden: torch.Tensor,
        autograd_mask: int,
        counts: torch.Tensor | None,
        gathered: _GatheredReplicaCounts | None = None,
    ) -> ReplicaRoute:
        """Apply only the shared no-table heuristic under the current Heap allowance."""
        reduction, row_credit = self._replica_route_policy(autograd_mask)
        return prepare_replica_route(
            topk_ids,
            self.replica_config,
            self._ep_group,
            minimum_replica_rows=self.replica_min_rows,
            minimum_load_reduction=reduction,
            _replica_transfer_row_credit=row_credit,
            _rank_nodes=resources.rank_nodes,
            compact_home=resources.home_spec is not None,
            dynamic_header=(
                int(hidden.shape[0] * max(self.hidden_size, self.top_k) > 2**31 - 1),
                hidden.shape[0],
                autograd_mask,
            ),
            _auto_capacity_factor=(resources.spec.initial_capacity_factor
                                   if resources.spec.dispatch_mode == "push" else None),
            _capacity_budget=resources.receive_capacity_limit if resources.spec.dispatch_mode == "push" else None,
            _trusted_counts=counts,
            _gathered_counts=gathered,
            _route_cache=getattr(resources.workspace, "replica_route_cache", None),
            _whole_experts=self.replica_granularity == "expert",
        )

    def _replica_route_selector(self, resources, hidden, ids, trusted, mask, state):
        """Select replicas only after current group counts and controls are validated."""
        adaptive = resources.spec._replica_inbox_slots is not None
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
                if resources.spec._replica_inbox_slots is not None:
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
        resources.last_empty_owner_experts = route.metadata.empty_owner_experts
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
        expert_weights: tuple[torch.Tensor, torch.Tensor],
        tokens_per_expert: torch.Tensor | None = None,
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
        weights = expert_weights
        hidden_flat, weight_error = self._prepare_forward_inputs(
            hidden_states, topk_ids, topk_weights, tokens_per_expert, weights,
        )
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("MegaMoe push heap growth and dynamic tokens do not support graph capture")
        resources = self._get_execution_resources(hidden_flat)
        route_options = {"autograd_mask": _INVALID_WEIGHT_MASK if weight_error else (
            int(torch.is_grad_enabled()) | (int(hidden_flat.requires_grad) << 1)
            | (int(weights[0].requires_grad) << 2) | (int(weights[1].requires_grad) << 3)
            | (int(topk_weights.requires_grad) << 4)
        )}
        pull = resources.spec.dispatch_mode == "pull"
        before_fc2 = resources.spec.apply_probs_before_fc2
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
                maximum_edge_rows=route.metadata.maximum_edge_rows,
                schedule=getattr(route.metadata, "schedule", None),
            )
            resources.last_execution_plan = execution_plan
            if not torch.is_grad_enabled():
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
            )
        finally:
            if claimed:
                resources.workspace.release()
        if pull or before_fc2:
            return expert_output.reshape_as(hidden_states)
        output = restore_topk_output(expert_output, route.unpermute_mapping, topk_weights)
        return output.reshape(hidden_states.shape)

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
