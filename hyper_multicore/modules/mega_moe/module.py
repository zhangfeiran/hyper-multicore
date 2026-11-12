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
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch
import torch.distributed as dist

from hyper_multicore import _collective, shmem
from hyper_multicore.replica.admission import (
    optional_peak_gain_is_sufficient,
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
from hyper_multicore.replica.topology import _local_node_identity, _node_ids
from hyper_multicore.scheduler.config import mega_moe_event_capacity

from ..module import MulticoreModule
from .transport import HcclExpertTransport, require_mte_peers
from . import rdma
from .function import execute_mega_moe_with_permutation
from .heap_manager import _get_heap_manager, _workspace_layout
from .plan import build_mega_moe_plan
from .route import RouteSchedule, prepare_topk_route, restore_topk_output
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

ExpertWeightLayout = Literal["EIO", "EOI"]
_EXPERT_WEIGHT_LAYOUTS = frozenset(("EIO", "EOI"))
_HCCL_OWNERS: weakref.WeakSet[MegaMoeExperts] = weakref.WeakSet()


def _validate_execution_controls(communication_backend: str, replica_slots_per_rank: int,
                                 replica_transport: str, replica_granularity: str,
                                 single_expert_tasks: bool, route_task_bounds: bool) -> str:
    """Validate mutually compatible communication and scheduling choices once."""
    if communication_backend not in ("shmem", "hccl"):
        raise ValueError("communication_backend must be 'shmem' or 'hccl'")
    if communication_backend == "hccl" and replica_slots_per_rank:
        raise ValueError("HCCL MegaMoE communication currently requires replica_slots_per_rank=0")
    if rdma.configured_engine() != "mte" and replica_slots_per_rank and replica_transport != "p2p":
        raise ValueError("RDMA expert replicas currently require replica_transport='p2p'")
    if replica_granularity not in ("token", "expert"):
        raise ValueError("replica_granularity must be 'token' or 'expert'")
    if not isinstance(single_expert_tasks, bool):
        raise ValueError("single_expert_tasks must be boolean")
    if not isinstance(route_task_bounds, bool):
        raise ValueError("route_task_bounds must be boolean")
    if replica_transport not in ("p2p", *KERNEL_GRADIENT_TRANSPORT_MODES):
        raise ValueError("Unsupported replica_transport; expected p2p or a kernel-gradient transport")
    return rdma.configured_engine()


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
    local_experts: int, hidden_size: int, intermediate_size: int, expert_weight_layout: ExpertWeightLayout
) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    """Create independently initialized local expert parameters."""
    if expert_weight_layout == "EOI":
        gate_up_shape = (local_experts, intermediate_size * 2, hidden_size)
        down_shape = (local_experts, hidden_size, intermediate_size)
    else:
        gate_up_shape = (local_experts, hidden_size, intermediate_size * 2)
        down_shape = (local_experts, intermediate_size, hidden_size)
    gate_up_weight = torch.nn.Parameter(torch.empty(gate_up_shape))
    down_weight = torch.nn.Parameter(torch.empty(down_shape))
    gate_up_weight._replica_owned_parameter = True
    down_weight._replica_owned_parameter = True
    for expert_index in range(local_experts):
        torch.nn.init.xavier_uniform_(gate_up_weight[expert_index])
        torch.nn.init.xavier_uniform_(down_weight[expert_index])
    return (gate_up_weight, down_weight)


def _resolve_ep_size(ep_size: int | None, ep_group: Any | None) -> int:
    """Infer size from an explicit group without silently choosing WORLD."""
    if ep_size is not None and (not isinstance(ep_size, int) or isinstance(ep_size, bool) or ep_size <= 0):
        raise ValueError("ep_size must be a positive integer or None")
    if ep_group is not None:
        if not dist.is_initialized():
            raise ValueError("Initialize torch.distributed before passing ep_group")
        group_size = dist.get_world_size(ep_group)
        if ep_size is not None and ep_size != group_size:
            raise ValueError(f"ep_size={ep_size} does not match ep_group size={group_size}")
        return group_size
    if dist.is_initialized() and dist.get_world_size() > 1:
        raise ValueError("Pass ep_group explicitly for distributed MegaMoeExperts")
    return 1 if ep_size is None else ep_size


def _required_replica_inbox_slots(route: ReplicaRoute) -> int:
    """Derive an identical matrix prefix from the agreed plan on all EP ranks."""
    if route.home_only:
        return 0
    plan = route.plan
    home = plan.config.home_experts
    return max((transfer.target_slot - home + 1 for transfer in plan.transfers), default=0)


def _schedule_plan_bounds(schedule: RouteSchedule | None, schedule_tokens: int, ep_size: int,
                          bucket: int, home_only: bool) -> tuple[Any, dict]:
    """Build a bounded plan key without changing its reserved storage shape."""
    key = (bucket, True) if home_only else bucket
    if schedule is None:
        return key, {}
    if schedule.expert_rows > schedule_tokens * ep_size or schedule.source_expert_rows > schedule_tokens:
        raise RuntimeError("MegaMoe route counts exceed supported per-expert task bounds")
    expert = min(schedule_tokens * ep_size, 1 << (max(128, schedule.expert_rows) - 1).bit_length())
    source = min(schedule_tokens, 1 << (max(128, schedule.source_expert_rows) - 1).bit_length())
    return (bucket, home_only, expert, source), {
        "schedule_expert_rows": expert, "schedule_source_expert_rows": source,
    }


def _communication_plan_rows(spec: MegaMoeSpec, maximum_edge_rows: int | None,
                             tokens: int, source_bound: int) -> int | None:
    """Use the agreed communication bound only if it tightens the source schedule."""
    if not spec._compact_communication or maximum_edge_rows is None:
        return None
    if not 0 <= maximum_edge_rows <= tokens:
        raise ValueError("MegaMoe route exceeds the schedule's per-edge token capacity")
    edge_bucket = 1 << (max(_COMMUNICATION_SPLIT, maximum_edge_rows) - 1).bit_length()
    if maximum_edge_rows > source_bound:
        raise RuntimeError("MegaMoe communication bound exceeds the expert route bound")
    return edge_bucket if edge_bucket < source_bound else None


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
        shared: bool,
        active_specifications: tuple[Mapping[str, Any], ...],
    ) -> None:
        """Bind resources once to the first NPU tensor."""
        self.spec = bind_mega_moe_spec(specification, tensor)
        self._token_capacity = self.spec.local_num_tokens
        self._plans = {}
        self.rank_nodes = (0,) * self.spec.ep_size
        if dist.is_initialized():
            self._validate_distributed_configuration(active_specifications)
        self.spec = qualify_topology_schedule(self.spec, self.rank_nodes)
        reservations = tuple(((item, _workspace_layout(item, tensor.dtype)) for item in active_specifications))
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
                if rdma.enabled(self.spec):
                    rdma.require_native()
                    masks = shmem.peer_transport_masks()
                    missing = [peer for peer, mask in enumerate(masks) if peer != self.spec.rank_id
                               and not (mask & 2 or self.spec.shmem_data_engine == "mte_roce" and mask & 1)]
                    if len(masks) != self.spec.ep_size or missing:
                        raise RuntimeError(f"SHMEM peers lack the requested MTE/RoCE transport: {missing}")
                    if shmem.debug_state()["config"]["data_engine"] != self.spec.shmem_data_engine:
                        raise RuntimeError("SHMEM data engine changed after module construction")
                else:
                    require_mte_peers(shmem.peer_mte_accessibility(), self.spec.ep_size)
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
            "mega_moe",
            self.spec.local_num_tokens,
            self.spec.hidden_size,
            self.spec.intermediate_size,
            self.spec.num_experts,
            self.spec.top_k,
            self.spec.receive_capacity,
            self.spec.ep_size,
            self.spec.dispatch_mode,
            self.spec.expert_weight_layout,
            self.spec.initial_capacity_factor,
            self.spec.capacity_growth_factor,
            self.spec.dispatch_split,
            self.spec.combine_split,
            self.spec.max_local_num_tokens,
            self.spec.auto_tokens,
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
            "shmem_data_engine",
        )
        manifest = (
            "dynamic_tokens_v4",
            "bfloat16",
            tuple((tuple((spec.get(key) for key in fields)) for spec in specifications)),
            getattr(self, "soc_name", ""),
        )
        manifest += (replica_runtime_schedule(), _local_node_identity())
        manifests = [None] * self.spec.ep_size
        dist.all_gather_object(manifests, manifest, group=self.spec.ep_group)
        if any((value[:-1] != manifests[0][:-1] for value in manifests)):
            raise ValueError("MegaMoe resource configurations differ across EP ranks")
        self.rank_nodes = _node_ids([value[-1] for value in manifests])

    def plan_for_tokens(
        self, tokens: int, device: Any, *, home_only: bool = False, maximum_edge_rows: int | None = None,
        schedule: RouteSchedule | None = None,
    ) -> Any:
        """Reuse one covering plan per geometric bucket within committed capacity.

        Args:
            tokens: EP group maximum input rows, already aligned to 128.
            device: Execution device used for the parameters and workspace.
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
        if rdma.enabled(spec):
            spec = replace(spec, local_num_tokens=self._token_capacity,
                           receive_capacity=self.workspace.capacity_floor)
        bucket = 1 << (max(128, tokens) - 1).bit_length()
        schedule_tokens = min(bucket, self._token_capacity)
        key, bounds = _schedule_plan_bounds(schedule, schedule_tokens, spec.ep_size, bucket, home_only)
        if rdma.enabled(spec):
            key = (key, spec.local_num_tokens, spec.receive_capacity)
        source_bound = bounds.get("schedule_source_expert_rows", schedule_tokens)
        communication_rows = _communication_plan_rows(spec, maximum_edge_rows, tokens, source_bound)
        if communication_rows is not None:
            key = (key, communication_rows)
        plan = self._plans.get(key)
        if plan is None or plan.spec.plan_tokens < tokens:
            plan = build_mega_moe_plan(
                replace(spec, schedule_tokens=schedule_tokens, _communication_rows=communication_rows, **bounds),
                device,
            )
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

    Serial model layers can call :meth:`share_execution_resources` before
    their first forward to share one automatically growing workspace while keeping
    independent parameters and optimizer state.
    """

    @classmethod
    def from_module(
        cls, module: torch.nn.Module, *, top_k: int, ep_group: Any | None = None, **execution_options: Any
    ) -> MegaMoeExperts:
        """Convert materialized local GroupedExperts SwiGLU weights before optimizer construction.

        Args:
            module: Module exposing w1/w3 [local_E, I, H], w2 [local_E, H, I].
                DTensor local EP shards are accepted; FSDP weights must be materialized first.
            top_k: Experts selected by the existing router for each token.
            ep_group: Explicit EP group owning these local experts, or None for a standalone rank.
            **execution_options: Advanced MegaMoe options, including num_experts and expert_weight_layout.

        Returns:
            Experts with copied, converted Parameters. Optimizer state is not migrated.
        """
        values = []
        for name in ("w1", "w3", "w2"):
            value = getattr(module, name, None)
            if hasattr(value, "to_local"):
                value = value.to_local()
            if not isinstance(value, torch.Tensor) or value.ndim != 3:
                raise ValueError(f"source {name} must be a materialized local rank-three expert tensor")
            if value.is_meta:
                raise ValueError(f"source {name} is meta; materialize and initialize/load weights before from_module")
            values.append(value)
        gate, up, down = (values[0], values[1], values[2])
        if (
            up.shape != gate.shape
            or down.shape != (gate.shape[0], gate.shape[2], gate.shape[1])
            or any((value.device != gate.device or value.dtype != gate.dtype for value in values))
            or (gate.requires_grad != up.requires_grad)
        ):
            raise ValueError("source SwiGLU expert shapes, dtypes, devices and gate/up gradient policies must match")
        ep_size = _resolve_ep_size(execution_options.get("ep_size"), ep_group)
        experts = execution_options.pop("num_experts", getattr(module, "num_experts", gate.shape[0] * ep_size))
        if experts != gate.shape[0] * ep_size:
            raise ValueError("source local expert count does not match num_experts and ep_group")
        result = cls(
            hidden_size=gate.shape[2],
            intermediate_size=gate.shape[1],
            num_experts=experts,
            top_k=top_k,
            ep_group=ep_group,
            create_parameters=False,
            **execution_options,
        )
        gate_up = torch.cat((gate.detach(), up.detach()), dim=1)
        down_data = down.detach()
        if result.expert_weight_layout == "EIO":
            gate_up = gate_up.transpose(1, 2)
            down_data = down_data.transpose(1, 2)
        result.gate_up_weight = torch.nn.Parameter(gate_up.contiguous(), requires_grad=gate.requires_grad)
        result.down_weight = torch.nn.Parameter(
            down_data.clone(memory_format=torch.contiguous_format), requires_grad=down.requires_grad
        )
        result.gate_up_weight._replica_owned_parameter = True
        result.down_weight._replica_owned_parameter = True
        return result

    @staticmethod
    def _validate_unbound_groups(members: tuple, first_group: Any) -> None:
        """Keep HCCL routing owners immutable after their first execution."""
        if any(hasattr(member._resource_group, "hccl_executor") for member in members):
            raise RuntimeError("execution resources must be shared before first use")
        MulticoreModule._validate_unbound_groups(members, first_group)

    def _resource_signature(self, tensor: Any) -> Any:
        """Compare static configuration without process-local group identities."""
        configuration = tuple(
            sorted((name, value) for name, value in self._resource_group.specification.items() if name != "ep_group")
        )
        return (super()._resource_signature(tensor), configuration, self._resource_group.shared)

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
        swiglu_limit: float | None = None,
        swiglu_hf_bf16: bool = False,
        single_expert_tasks: bool = False,
        route_task_bounds: bool = False,
        ep_size: int | None = None,
        ep_group: Any | None = None,
        create_parameters: bool = True,
        expert_weight_layout: ExpertWeightLayout = "EIO",
        dispatch_mode: str = "push",
        communication_backend: str = "shmem",
        initial_capacity_factor: float | None = None,
        initial_token_capacity: int = _COMMUNICATION_SPLIT,
        capacity_growth_factor: float | None = None,
        dispatch_split: int = _COMMUNICATION_SPLIT,
        combine_split: int = _COMMUNICATION_SPLIT,
        replica_slots_per_rank: int = 0,
        replica_granularity: str = "token",
        gate_up_expert_padding: int = 0,
        replica_transport: str = "shmem_signal_kernel_gradient_bands",
        replica_min_rows: int = 1024,
        replica_minimum_load_reduction: float | None = None,
    ) -> None:
        """Initialize local expert parameters and a lazy execution owner.

        Args:
            hidden_size: Input and output hidden dimension.
            intermediate_size: SwiGLU intermediate dimension per expert.
            num_experts: Global routed-expert count.
            top_k: Experts selected for every token.
            initial_capacity_factor: Push initial receive rows as a multiple of local routed rows.
                Defaults to 1.25 for push; omitted for pull. Must be finite and at least 1.0.
            initial_token_capacity: Initial source rows, rounded to a communication tile; larger inputs grow.
            capacity_growth_factor: Push capacity multiplier on overflow, defaulting to 1.25.
                Must be finite and at least 1.0; 1.0 grows only to the current route demand.
                The resulting capacity is capped by the lossless route bound. Pull rejects explicit factors.
            replica_slots_per_rank: Maximum physical guest slots on each EP rank. Zero
                keeps owner routing. Positive values use load and topology heuristics,
                with projection-ready weights and FP32 gradient return. Optional
                replicas require at least a 40 percent peak-row reduction; replicas
                needed to fit the current receive budget bypass this admission gate.
            replica_granularity: ``"token"`` may split an expert between ranks;
                ``"expert"`` keeps its reduction whole for reference parity.
            replica_transport: Replica transfer protocol, defaulting to SHMEM gradient bands.
            replica_min_rows: Soft minimum assignment count per guest expert.
            replica_minimum_load_reduction: Optional minimum improvement in peak destination load.
            gate_up_expert_padding: Global unrouted trailing rows in external EOI Gate/Up storage.
                Requires no replicas and a row count divisible by EP; trailing gradients are zero.
            dispatch_mode: Dispatch transport, either "push" (default) or "pull".
                Construct separate modules to switch modes; sharing requires equal modes.
            communication_backend: ``"shmem"`` retains fused device communication, using
                MTE or RoCE according to HYPER_MULTICORE_SHMEM_DATA_ENGINE. ``"hccl"`` uses differentiable HCCL routing
                around single-PE MegaMoE compute; no remote MTE operations are submitted.
                HCCL currently requires replica_slots_per_rank=0.
            dispatch_split: Dispatch tile rows in forward and backward, default 128.
                Must be a positive multiple of 128.
            combine_split: Combine tile rows in forward and backward, with the same
                constraints and default as dispatch_split. Sharing requires equal splits.
            swiglu_limit: Optional positive, finite float32-representable clamp
                limit for SwiGLU. The gate branch uses ``min(gate, limit)`` and
                the up branch is clamped to ``[-limit, limit]``. ``None``
                preserves the legacy unclamped path.
            swiglu_hf_bf16: Retain HF's BF16 activation and gradient rounding boundaries.
                It cannot be combined with swiglu_limit and defaults to fused arithmetic.
            single_expert_tasks: Address owner matrices as one-expert GMM groups.
            route_task_bounds: Bound automatic task plans by exact group route counts.
            ep_size: Optional compatibility check; normally inferred from explicit ep_group.
            create_parameters: Allocate owned weights; False requires explicit expert_weights each forward.
            expert_weight_layout: Logical expert-weight axis order. ``"EIO"`` means
                ``[experts, in_features, out_features]`` and preserves the historical layout. ``"EOI"`` means
                ``[experts, out_features, in_features]`` and accepts Torch linear storage directly. ``EOI`` weights
                must use dense contiguous storage for zero-copy execution.
            ep_group: Explicit EP group; expert ownership follows group-local rank order.
                Disjoint PP/DP groups bootstrap independently. Only one ordered EP membership
                can be active in SHMEM per process.
        """
        self.shmem_data_engine = _validate_execution_controls(
            communication_backend, replica_slots_per_rank, replica_transport, replica_granularity,
            single_expert_tasks, route_task_bounds)
        self.communication_backend = communication_backend
        self._communication_checked = False
        _integer(replica_min_rows, "replica_min_rows", 0)
        resolve_replica_admission(replica_minimum_load_reduction)
        ep_size = _resolve_ep_size(ep_size, ep_group)
        _integer(initial_token_capacity, "initial_token_capacity", 1)
        _integer(gate_up_expert_padding, "gate_up_expert_padding", 0)
        if gate_up_expert_padding and (
            create_parameters or replica_slots_per_rank or gate_up_expert_padding % ep_size
            or expert_weight_layout != "EOI"
        ):
            raise ValueError("gate_up_expert_padding requires external non-replica EOI weights and divisibility by EP")
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
            "initial_capacity_factor": initial_capacity_factor,
            "swiglu_limit": swiglu_limit,
            "swiglu_hf_bf16": swiglu_hf_bf16,
            "single_expert_tasks": single_expert_tasks,
            "route_task_bounds": route_task_bounds,
            "ep_size": ep_size,
            "ep_group": ep_group,
            "expert_weight_layout": expert_weight_layout,
            "dispatch_mode": dispatch_mode,
            "communication_backend": communication_backend,
            "shmem_data_engine": self.shmem_data_engine,
            "capacity_growth_factor": capacity_growth_factor,
            "dispatch_split": dispatch_split,
            "combine_split": combine_split,
            "_host_replica_metadata": True,
            "_summary_home_admission": self._summary_home_admission,
        }
        specification.update(_execution_options(dispatch_mode, replica_slots_per_rank))
        specification["auto_tokens"] = True
        specification = self._resolve_specification(specification)
        compatibility_key = tuple((id(value) if name == "ep_group" else value for name, value in specification.items()))
        super().__init__(
            resource_specification=specification,
            resource_compatibility_key=compatibility_key,
            resource_scope_key=("mega_moe", shmem.root_members(ep_group) if dist.is_initialized() else id(ep_group)),
        )
        self.replica_config = replica_config
        self.replica_slots_per_rank = replica_slots_per_rank
        self.replica_granularity = replica_granularity
        self.gate_up_expert_padding = gate_up_expert_padding
        self.initial_token_capacity = specification["local_num_tokens"]
        self._initialize_module_state(specification, create_parameters=create_parameters)
        if self.communication_backend == "hccl":
            _HCCL_OWNERS.add(self)

    def _initialize_module_state(self, specification: Mapping[str, Any], *, create_parameters: bool) -> None:
        """Bind normalized configuration and initialize expert parameters."""
        self.hidden_size = specification["hidden_size"]
        self.intermediate_size = specification["intermediate_size"]
        self.num_experts = self.replica_config.num_experts
        self.top_k = specification["top_k"]
        self.expert_weight_layout = specification["expert_weight_layout"]
        self.initial_capacity_factor = specification["initial_capacity_factor"]
        self.capacity_growth_factor = specification["capacity_growth_factor"]
        self.dispatch_mode = specification["dispatch_mode"]
        self.dispatch_split = specification["dispatch_split"]
        self.combine_split = specification["combine_split"]
        self.swiglu_limit = specification["swiglu_limit"]
        self.swiglu_hf_bf16 = specification.get("swiglu_hf_bf16", False)
        self.single_expert_tasks = specification.get("single_expert_tasks", False)
        self.route_task_bounds = specification.get("route_task_bounds", False)
        self.ep_size = specification["ep_size"]
        self.local_experts = self.num_experts // self.ep_size
        self._ep_group = specification["ep_group"]
        if create_parameters:
            self.gate_up_weight, self.down_weight = _create_mega_moe_parameters(
                self.local_experts, self.hidden_size, self.intermediate_size, self.expert_weight_layout
            )
        else:
            self.register_parameter("gate_up_weight", None)
            self.register_parameter("down_weight", None)

    @classmethod
    def _resolve_specification(cls, specification: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize configuration before registering an execution owner."""
        expert_weight_layout = specification["expert_weight_layout"]
        hf_bf16 = specification.get("swiglu_hf_bf16", False)
        if not isinstance(hf_bf16, bool) or (hf_bf16 and specification["swiglu_limit"] is not None):
            raise ValueError("swiglu_hf_bf16 must be boolean and cannot be combined with swiglu_limit")
        if expert_weight_layout not in _EXPERT_WEIGHT_LAYOUTS:
            raise ValueError(
                "expert_weight_layout must be 'EIO' ([experts, in_features, out_features]) or "
                f"EOI ([experts, out_features, in_features]), got {expert_weight_layout!r}."
            )
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

    def _validate_tensors(self, hidden_states: torch.Tensor, expert_weights: tuple) -> None:
        """Validate activation and parameter metadata before acquiring resources."""
        if hidden_states.dtype != torch.bfloat16 or not getattr(hidden_states, "is_npu", False):
            raise TypeError("MegaMoeExperts requires BF16 NPU hidden states.")
        weight1, weight2 = expert_weights
        if weight1 is None or weight2 is None:
            raise ValueError("Parameterless MegaMoe requires explicit expert_weights")
        storage_experts = self.local_experts + self.gate_up_expert_padding // self.ep_size
        if self.expert_weight_layout == "EOI":
            expected_weight1 = (storage_experts, self.intermediate_size * 2, self.hidden_size)
            expected_weight2 = (self.local_experts, self.hidden_size, self.intermediate_size)
        else:
            expected_weight1 = (storage_experts, self.hidden_size, self.intermediate_size * 2)
            expected_weight2 = (self.local_experts, self.intermediate_size, self.hidden_size)
        if tuple(weight1.shape) != expected_weight1:
            raise ValueError(f"gate_up_weight must have shape {expected_weight1}, got {tuple(weight1.shape)}.")
        if tuple(weight2.shape) != expected_weight2:
            raise ValueError(f"down_weight must have shape {expected_weight2}, got {tuple(weight2.shape)}.")
        if weight1.device != hidden_states.device or weight2.device != hidden_states.device:
            raise ValueError("expert weights and hidden states must be on the same NPU.")
        if weight1.dtype != hidden_states.dtype or weight2.dtype != hidden_states.dtype:
            raise TypeError("expert weights and hidden states must all use BF16.")
        if self.expert_weight_layout == "EOI" and (not weight1.is_contiguous() or not weight2.is_contiguous()):
            raise ValueError("EOI expert weights must be contiguous for zero-copy execution.")

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

    def _summary_keeps_home(
        self, resources: _MegaMoeExecutionResources, summary: list[list[int]], loads: list[int]
    ) -> bool:
        """Keep an already feasible home layout if even perfect balance cannot be admitted."""
        admission = self._summary_home_admission
        if not admission:
            return False
        peak = max(loads)
        if peak > 2**31 - 1:
            return False
        lower_bound = (sum(loads) + len(loads) - 1) // len(loads)
        if optional_peak_gain_is_sufficient(peak, lower_bound, admission, 0):
            return False
        if self.dispatch_mode == "push":
            planned_tokens = max(128, _align_capacity(max((row[1] for row in summary))))
            if planned_tokens > resources._token_capacity or peak > resources.workspace.capacity_floor:
                return False
        state = get_recompute_state()
        if state is not None and state.is_recomputing and (not _has_home_recompute_hint()):
            return False
        _record_home_decision()
        self._summary_home_bypasses += 1
        return True

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
        training = bool(autograd_mask & 1) or get_recompute_state() is not None
        projections = 2 * bool(autograd_mask & 4) + bool(autograd_mask & 8) if training else 0
        return prepare_replica_route(
            topk_ids,
            self.replica_config,
            self._ep_group,
            minimum_replica_rows=self.replica_min_rows,
            minimum_load_reduction=resolve_replica_admission(self.replica_minimum_load_reduction)[0],
            _replica_transfer_row_credit=self.intermediate_size * projections // 6,
            _rank_nodes=resources.rank_nodes,
            compact_home=resources.home_spec is not None,
            dynamic_header=(
                int(hidden.shape[0] * max(self.hidden_size, self.top_k) > 2**31 - 1),
                hidden.shape[0],
                autograd_mask,
            ),
            _auto_capacity_factor=self.initial_capacity_factor if self.dispatch_mode == "push" else None,
            _capacity_budget=resources.receive_capacity_limit if self.dispatch_mode == "push" else None,
            _trusted_counts=counts,
            _gathered_counts=gathered,
            _whole_experts=self.replica_granularity == "expert",
        )

    def _replica_route_selector(self, resources, hidden, ids, trusted, mask, state):
        """Select replicas only after current group counts and controls are validated."""
        adaptive = resources.spec._replica_inbox_slots is not None

        def select(
            counts: torch.Tensor, summary: list[list[int]], host_counts: _HostReplicaCounts | None
        ) -> tuple[MegaMoeSpec, ReplicaRoute] | None:
            """Preserve home call ordinals even when the full planner is bypassed."""
            loads = [row[6] for row in summary]
            if min(loads) == max(loads):
                _record_home_decision()
                return None
            if self._summary_keeps_home(resources, summary, loads):
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
        tokens_per_expert: torch.Tensor | None = None,
        expert_weights: tuple[torch.Tensor, torch.Tensor] | None = None,
        _indexed_source: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Route token states through the configured experts.

        Args:
            hidden_states: Tensor ending in ``hidden_size``.
            topk_ids: Global expert IDs shaped ``[tokens, top_k]`` for the current input.
            topk_weights: Router weights with the same shape as ``topk_ids``.
            expert_weights: Current unsharded weights owned by an outer module.
                Noncontiguous EIO weights are copied with autograd tracking, never cached
                by this executor. EOI weights must be contiguous for zero-copy execution.
            _indexed_source: Internal original source and compact assignment row map for RDMA adapters.
            tokens_per_expert: Optional trusted exact global-expert histogram.
                Supplying Router-produced counts skips histogram recomputation.

        Returns:
            Tensor matching the shape and dtype of ``hidden_states``.

        Raises:
            ValueError: EOI expert weights are noncontiguous.

        Note:
            ``tokens_per_expert`` must describe the current ``topk_ids`` exactly.
            The steady-state path validates metadata but deliberately does not
            rebuild and compare the histogram.
        """
        weights = (self.gate_up_weight, self.down_weight) if expert_weights is None else expert_weights
        hidden_flat = self._validate_forward_inputs(hidden_states, topk_ids, topk_weights, tokens_per_expert, weights)
        self._check_forward_runtime()
        if self.communication_backend == "hccl":
            executor = self._hccl_transport()
            return executor(hidden_flat, topk_ids, topk_weights, weights).reshape_as(hidden_states)
        raw_capacity = self._indexed_source_capacity(hidden_flat, _indexed_source)
        resources = self._get_execution_resources(hidden_flat)
        route_options = {
            "autograd_mask": int(torch.is_grad_enabled())
            | int(hidden_flat.requires_grad) << 1
            | int(weights[0].requires_grad) << 2
            | int(weights[1].requires_grad) << 3
            | int(topk_weights.requires_grad) << 4
        }
        pull = self.dispatch_mode == "pull"
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
            tokens = max(tokens, raw_capacity)
            if adaptive_inbox:
                resources.prepare_auto_workspace(tokens, received, hidden_flat, replica_slots=route_state.inbox_slots)
            else:
                resources.prepare_auto_workspace(tokens, received, hidden_flat)
            claimed = pull

        if automatic:
            route_options["prepare_workspace"] = prepare_workspace
        elif pull and not rdma.enabled(resources.spec):
            resources.workspace.ensure(resources.spec, hidden_flat.dtype, hidden_flat.device)
            resources.workspace.claim()
            claimed = True
        try:
            with torch.no_grad():
                route, topk_ids = self._prepare_forward_route(
                    resources, hidden_flat, topk_ids, topk_weights, tokens_per_expert, route_options, route_state
                )
            route = self._attach_indexed_source(route, hidden_flat, _indexed_source, raw_capacity)
            if not automatic and (not pull or rdma.enabled(resources.spec)):
                resources.ensure_capacity(route.maximum_received_slots)
            self._validate_received_capacity(resources, route.maximum_received_slots)
            execution_plan = resources.plan_for_tokens(
                route.metadata.plan_tokens, hidden_flat.device, home_only=route_state.home_only,
                maximum_edge_rows=route.metadata.maximum_edge_rows,
                schedule=route.metadata.schedule,
            )
            resources.last_execution_plan = execution_plan
            expert_output = execute_mega_moe_with_permutation(
                hidden_flat,
                topk_ids,
                weights[0],
                weights[1],
                route,
                execution_plan,
                resources.workspace,
                topk_weights=topk_weights if pull else None,
                workspace_claimed=pull,
            )
        finally:
            if claimed:
                resources.workspace.release()
        if pull:
            return expert_output.reshape_as(hidden_states)
        output = restore_topk_output(expert_output, route.unpermute_mapping, topk_weights)
        return output.reshape(hidden_states.shape)

    def _check_forward_runtime(self) -> None:
        """Converge transport choices before any rank enters a different execution path."""
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("MegaMoe push heap growth and dynamic tokens do not support graph capture")
        if self._resource_closed or self._resource_group.closed or self._resource_group.closing:
            raise RuntimeError("cannot execute a closed or closing multicore module")
        if not self._communication_checked:
            signature = (self.communication_backend, self.hidden_size, self.intermediate_size, self.num_experts,
                         self.top_k, self.expert_weight_layout, self.swiglu_hf_bf16, self.swiglu_limit,
                         self.shmem_data_engine)
            peers = _collective.exchange(signature, group=self._ep_group, failure_prefix="MegaMoE transport agreement")
            if any(peer != signature for peer in peers):
                raise RuntimeError("MegaMoE communication and compute configuration must match on all EP ranks")
            self._communication_checked = True

    def _indexed_source_capacity(self, hidden_flat: torch.Tensor,
                                 indexed_source: tuple[torch.Tensor, torch.Tensor] | None) -> int:
        """Keep raw source capacity common even when a rank has no real assignments."""
        if indexed_source is not None and not rdma.enabled(self._resource_group.specification):
            raise ValueError("Indexed sources require an RDMA execution specification")
        original_rows = hidden_flat.shape[0] if indexed_source is None else indexed_source[0].shape[0]
        raw_capacity = original_rows
        if indexed_source is not None:
            self._validate_indexed_source(hidden_flat, indexed_source)
            capacity = torch.tensor([original_rows], dtype=torch.int64, device=hidden_flat.device)
            if self.ep_size > 1:
                dist.all_reduce(capacity, op=dist.ReduceOp.MAX, group=self._ep_group)
            raw_capacity = int(capacity.item())
        return raw_capacity

    @staticmethod
    def _validate_received_capacity(resources: Any, maximum_received_slots: int) -> None:
        """Reject a group-wide overflow before any rank launches a symmetric receive."""
        if rdma.enabled(resources.spec) and maximum_received_slots > resources.workspace.capacity_floor:
            raise RuntimeError("RDMA expert receive capacity does not cover the agreed route load")

    def _validate_indexed_source(self, hidden_flat: torch.Tensor,
                                 indexed_source: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Reject metadata carriers that do not alias the original contiguous activation."""
        source, indices = indexed_source
        if (source.ndim != 2 or source.shape[1] != self.hidden_size or source.dtype != hidden_flat.dtype
                or source.device != hidden_flat.device or indices.ndim != 1
                or indices.numel() != hidden_flat.shape[0] or indices.device != source.device
                or indices.dtype not in (torch.int32, torch.int64) or not source.is_contiguous()
                or hidden_flat.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()):
            raise ValueError("Invalid indexed original source or assignment row metadata")

    def _attach_indexed_source(self, route: Any, hidden_flat: torch.Tensor,
                               indexed_source: tuple[torch.Tensor, torch.Tensor] | None,
                               source_rows_bound: int | None = None) -> Any:
        """Attach original source row metadata without materializing assignment activations."""
        if rdma.enabled(self._resource_group.specification):
            inverse = route.unpermute_mapping.long()
            assignment_order = torch.empty_like(inverse)
            assignment_order.scatter_(0, inverse, torch.arange(inverse.numel(), device=inverse.device))
            source_indices = torch.div(assignment_order, self.top_k, rounding_mode="floor").long()
            source = hidden_flat
            if indexed_source is not None:
                source = indexed_source[0]
                source_indices = indexed_source[1].index_select(0, source_indices)
            route = replace(route, metadata=replace(route.metadata, original_source=source,
                                                   source_row_indices=source_indices.to(torch.int32),
                                                   source_rows_bound=(source_rows_bound if indexed_source is not None
                                                                      else None)))
        return route

    def _hccl_transport(self) -> HcclExpertTransport:
        """Register local layers before the first native bind reserves the shared Heap."""
        if hasattr(self._resource_group, "hccl_executor"):
            return self._resource_group.hccl_executor
        owners = sorted((owner for owner in _HCCL_OWNERS if not owner._resource_closed
                         and owner._resource_group.scope_key == self._resource_group.scope_key),
                        key=lambda owner: owner._resource_group.identifier)
        # Later layers may execute while earlier autograd contexts still need the Heap.
        # Registering every local executor now preserves the existing aggregate reservation policy.
        for owner in owners:
            group = owner._resource_group
            if not hasattr(group, "hccl_executor"):
                group.hccl_executor = HcclExpertTransport(owner._ep_group, owner.num_experts, owner._local_executor)
        return self._resource_group.hccl_executor

    def _local_executor(self, group: Any) -> MegaMoeExperts:
        """Retain arithmetic and layout while binding native communication to self."""
        specification = self._resource_group.specification
        options = {key: specification[key] for key in (
            "swiglu_limit", "swiglu_hf_bf16", "single_expert_tasks", "route_task_bounds",
            "expert_weight_layout", "dispatch_mode", "initial_capacity_factor", "capacity_growth_factor",
            "dispatch_split", "combine_split",
        )}
        return MegaMoeExperts(hidden_size=self.hidden_size, intermediate_size=self.intermediate_size,
                              num_experts=self.local_experts, top_k=1, ep_group=group, ep_size=1,
                              create_parameters=False, initial_token_capacity=self.initial_token_capacity, **options)

    def close(self) -> None:
        """Release transport-local compute after the last shared owner finishes."""
        group = self._resource_group
        if hasattr(group, "hccl_executor") and group.members == {self._resource_member_token}:
            group.hccl_executor.close()
        super().close()

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
            shared=shared,
            active_specifications=active_specifications,
        )
