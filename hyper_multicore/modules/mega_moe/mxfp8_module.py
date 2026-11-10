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
"""Ascend 950 MXFP8 task graphs with bounded projected expert replicas."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any
from functools import cached_property
import os

import torch
import torch.distributed as dist

from hyper_multicore.replica.recompute import get_recompute_state
from hyper_multicore.replica.capacity import ExpertReplicaConfig
from hyper_multicore.replica._metadata import _ReplicaCountHeaderCache
from hyper_multicore import _collective
from hyper_multicore.shmem._runtime import configured_data_engine, configured_udma_qp_count
from hyper_multicore.scheduler.runtime import serialize_runtime_config
from hyper_multicore.scheduler.config import mega_moe_event_capacity

from .heap_manager import _get_heap_manager, _workspace_layout
from .module import MegaMoeExperts, _MegaMoeExecutionResources, _checkpoint_contract
from .mxfp8_execution import MXFP8Execution
from .mxfp8_diagnostics import prepare_mx_diagnostics
from .mxfp8_function import execute_mxfp8_experts
from .mxfp8_graph import build_mxfp8_runtime, with_source_row_ingress, with_token_mte_scratch
from .mxfp8_gradient_udma import mx_gradient_peer_qps, mx_udma_gradient_enabled
from .mxfp8_ingress import source_row_fp8_enabled
from .mxfp8_gradients import mx_projection_events
from .mxfp8_prefetch import MXProjectionPrefetch
from .mxfp8_ops import mxfp8_window_feature_available, require_mxfp8_window_feature
from .mxfp8_weights import MXFP8ExpertWeightBinding
from .spec import bind_mega_moe_spec
from .workspace import MegaMoeWorkspace


def _device_bytes(data: bytes, device: Any) -> torch.Tensor:
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device=device)


@dataclass(frozen=True)
class _DeviceRuntime:
    runtime: torch.Tensor
    gmm: torch.Tensor
    quant: torch.Tensor
    event_capacity: int
    host: Any
    spec: Any
    gradient_udma_qps: tuple[tuple[int, ...], ...] = ()

    @cached_property
    def normal_diagnostic(self) -> torch.Tensor:
        """Keep one zeroed diagnostic placeholder for ordinary launches of this runtime."""
        return torch.zeros(64, dtype=torch.uint8, device=self.runtime.device)

    @cached_property
    def diagnostics(self) -> Any:
        """Materialize diagnostic metadata only for an explicit capture."""
        return prepare_mx_diagnostics(self.host, self.spec, self.runtime.device)


@dataclass(frozen=True)
class _MXFP8Plan:
    spec: Any
    forward: _DeviceRuntime
    backward: _DeviceRuntime

    @cached_property
    def replica_w2_events(self) -> tuple[int, ...]:
        """Require every Cube producer before publishing second-projection gradients."""
        return mx_projection_events(self.backward.host.config, self.spec.local_experts, (6,))

    @cached_property
    def replica_w13_events(self) -> tuple[int, ...]:
        """Require every Cube producer before publishing first-projection gradients."""
        return mx_projection_events(self.backward.host.config, self.spec.local_experts, (18,))


def _device_runtime(host: Any, device: Any, spec: Any) -> _DeviceRuntime:
    gradient_qps = mx_gradient_peer_qps(host.config, spec) if spec.mx_gradient_udma and host.kernel_gradients else ()
    return _DeviceRuntime(
        _device_bytes(serialize_runtime_config(host.config), device),
        _device_bytes(host.gmm_descriptors, device),
        _device_bytes(host.quant_descriptors, device),
        host.config.event_capacity,
        host,
        spec,
        gradient_qps,
    )


def _mx_home_spec(spec: Any) -> Any:
    """Keep home-only graphs within the reserved physical ready-counter layout."""
    if not spec.replica_slots_per_rank:
        return None
    home = replace(spec, num_experts=spec.logical_num_experts, replica_slots_per_rank=0)
    if mega_moe_event_capacity(home.num_experts, home.ep_size, compute_stages=6) != mega_moe_event_capacity(
        spec.num_experts, spec.ep_size, compute_stages=6
    ):
        return None
    return home


class _MXFP8Resources(_MegaMoeExecutionResources):
    """Retain the existing heap restart protocol without building BF16 graphs."""

    def __init__(
        self,
        specification: dict[str, Any],
        tensor: torch.Tensor,
        *,
        shared: bool,
        active_specifications: tuple[Mapping[str, Any], ...],
    ) -> None:
        """Bind validated invocation resources without retaining gathered weight views."""
        self.spec = bind_mega_moe_spec(specification, tensor)
        self.source_row_fp8 = specification.get("mx_source_row_fp8", False)
        self.token_mte_scratch_bytes = 16384
        self.weight_prefetch = MXProjectionPrefetch() if specification.get("mx_weight_prefetch", False) else None
        self.replica_headers = _ReplicaCountHeaderCache() if self.spec.replica_slots_per_rank else None
        if self.spec.num_cube_cores != 32:
            raise ValueError("MXFP8 MegaMoE requires the Ascend 950 32-Cube kernel.")
        self.soc_name = ""
        self.rank_nodes = (0,) * self.spec.ep_size
        self._token_capacity = self.spec.local_num_tokens
        self._plans = OrderedDict()
        self.home_spec = _mx_home_spec(self.spec)
        self.home_plan = self.last_execution_plan = None
        if dist.is_initialized():
            self._validate_distributed_configuration(active_specifications)

        def validate_features() -> None:
            """Check the selected binary before creating symmetric resources."""
            if self.spec.mx_ready_depth:
                require_mxfp8_window_feature("ready_math")
            if self.spec.mx_swiglu_row_only:
                require_mxfp8_window_feature("nograd_row_only")
                require_mxfp8_window_feature("nograd_x_row_only")
            if self.spec.mx_gradient_udma:
                if configured_udma_qp_count() not in (8, 16, 32):
                    raise ValueError("MX UDMA gradient helpers require 8, 16 or 32 QPs")
                if configured_data_engine() != "mte_udma":
                    raise ValueError("MX UDMA gradient helpers require mte_udma")
                require_mxfp8_window_feature("replica_udma_gradient_peer_qp")
            if self.source_row_fp8:
                require_mxfp8_window_feature("mx_source_row_fp8")
                require_mxfp8_window_feature("mx_source_row_ready_event")
                require_mxfp8_window_feature("mx_source_row_base_bindings")
                if mxfp8_window_feature_available("mx_token_mte_scratch_v1"):
                    self.token_mte_scratch_bytes = 65536

        if (self.spec.mx_ready_depth or self.spec.mx_swiglu_row_only or self.spec.mx_gradient_udma
                or self.source_row_fp8):
            _collective.collective_call(validate_features, stage="MXFP8 window capabilities", group=self.spec.ep_group)
        self.heap_manager = _get_heap_manager(self.spec.ep_group, tensor)
        reservations = tuple((item, _workspace_layout(item, tensor.dtype)) for item in active_specifications)
        added = self.heap_manager.reserve(reservations, allow_resize=True)
        attached = False
        self.plan = self.workspace = None
        try:
            attached = self.heap_manager.attach_runtime()

            def construct() -> None:
                """Bind the reserved symmetric workspace after collective construction agreement."""
                self.workspace = MegaMoeWorkspace(shared=shared)
                self.workspace.capacity_floor = self.heap_manager.bind(self.workspace, self.spec, specification)
                self.workspace.heap_manager = self.heap_manager

            _collective.collective_call(
                construct, stage="MXFP8 resource construction", group=self.heap_manager.root_group
            )
        except Exception:
            self.heap_manager.rollback_reservations(added, self.workspace)
            if attached and not self.heap_manager.workspaces:
                self.heap_manager.detach_runtime()
            raise
        self._closed = self._workspace_closed = False

    def lifecycle_signature(self) -> tuple[Any, ...]:
        """Identify MX precision and scheduling in collective resource lifecycle checks."""
        return (
            *super().lifecycle_signature(),
            "mxfp8_v1",
            self.spec.mx_vector_policy,
            self.spec.mx_poll_policy,
            self.spec.mx_swiglu_row_only,
            self.spec.mx_ready_depth,
            self.spec.compute_stages,
            self.weight_prefetch is not None,
            self.spec.mx_gradient_udma,
            self.source_row_fp8,
        )

    def _validate_distributed_configuration(self, specifications: tuple[Mapping[str, Any], ...]) -> None:
        """Agree on projection readiness before any rank can allocate or launch its protocol."""
        super()._validate_distributed_configuration(specifications)
        if self.spec.ep_size == 1:
            return
        flags = tuple((bool(spec.get("mx_weight_prefetch", False)), bool(spec.get("mx_gradient_udma", False)),
                       bool(spec.get("mx_source_row_fp8", False)))
                      for spec in specifications)
        gathered = [None] * self.spec.ep_size
        dist.all_gather_object(gathered, flags, group=self.spec.ep_group)
        if any(value != flags for value in gathered):
            raise ValueError(
                "MXFP8 projection-prefetch or gradient-transport configuration differs across expert ranks")

    def plan_for_capacity(
        self, capacity: int, device: Any, *, home_only: bool = False, swiglu_row_only: bool = False,
        source_row_fp8: bool = False,
    ) -> _MXFP8Plan:
        """Cache pointer-free MX plans against the current symmetric receive bound.

        Args:
            capacity: Current symmetric receive-row bound.
            device: Device owning execution and diagnostic storage.
        """
        key = (capacity, home_only, swiglu_row_only) + (("source_row_fp8",) if source_row_fp8 else ())
        selected = self.home_spec if home_only else self.spec
        if selected is None:
            raise ValueError("MX home-only graph does not fit the physical event layout.")
        if key not in self._plans:
            spec = replace(selected, receive_capacity=capacity)
            forward = build_mxfp8_runtime(
                spec, forward=True, swiglu_row_only=swiglu_row_only, ready_depth=spec.mx_ready_depth,
                weight_prefetch=self.weight_prefetch is not None)
            if source_row_fp8:
                forward = with_source_row_ingress(spec, forward)
            forward = with_token_mte_scratch(forward, spec, self.token_mte_scratch_bytes)
            backward = build_mxfp8_runtime(
                spec, forward=False, ready_depth=spec.mx_ready_depth,
                weight_prefetch=self.weight_prefetch is not None)
            backward = with_token_mte_scratch(backward, spec, self.token_mte_scratch_bytes)
            self._plans[key] = _MXFP8Plan(
                spec,
                _device_runtime(forward, device, spec),
                _device_runtime(backward, device, spec),
            )
        self._plans.move_to_end(key)
        while len(self._plans) > 4:
            self._plans.popitem(last=False)
        return self._plans[key]

    def close(self) -> None:
        """Release immutable control uploads after closing the execution workspace."""
        super().close()
        if self.replica_headers is not None:
            self.replica_headers.clear()


def _mx_window_options(vector_policy: str) -> tuple[int, bool]:
    """Parse optional window settings before creating shared runtime resources."""
    depth = os.environ.get("PANGU_MX_READY_DEPTH", "0")
    row_only = os.environ.get("PANGU_MX_SWIGLU_ROW_ONLY", "0")
    if depth not in ("0", "1", "8"):
        raise ValueError("PANGU_MX_READY_DEPTH must be 0, 1 or 8.")
    if row_only not in ("0", "1"):
        raise ValueError("PANGU_MX_SWIGLU_ROW_ONLY must be 0 or 1.")
    if vector_policy != "window2" and (depth != "0" or row_only == "1"):
        raise ValueError("MXFP8 ready selection and no-grad row-only require window2.")
    return int(depth), row_only == "1"


class MXFP8MegaMoeExperts(MegaMoeExperts):
    """Execute MXFP8 experts while keeping FSDP root Parameters phase-current.

    Args:
        local_num_tokens: Fixed source rows per rank, divisible by transport splits.
        hidden_size: Input width, divisible by 64.
        intermediate_size: Routed expert width, divisible by 64.
        num_experts: Logical global expert count, excluding guest slots.
        top_k: Number of routes selected by the existing Router.
        ep_size: Expert parallel group size.
        ep_group: Raw Torch expert parallel group.
        replica_slots_per_rank: Bounded guest budget. B2 is the recommended candidate.
        replica_min_rows: Minimum useful guest rows for the existing CPU planner.
        initial_capacity_factor: Initial push receive allocation; never a worst-case reservation.
        capacity_growth_factor: Geometric growth factor, bounded by actual route demand.
        dispatch_split: Source transport rows per task.
        combine_split: Return transport rows per task.
        vector_policy: Adopted two-expert lookahead or the legacy control schedule.
        nograd_no_clone: Avoid the intermediate clone for checkpoint forwards.
        replica_transport: Exact weight P2P with optional fused FP32 projection-gradient return.
    """

    def __init__(
        self,
        *,
        local_num_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        ep_size: int = 1,
        ep_group: Any = None,
        replica_slots_per_rank: int = 0,
        replica_min_rows: int = 1024,
        initial_capacity_factor: float | None = None,
        capacity_growth_factor: float | None = None,
        dispatch_split: int = 512,
        combine_split: int = 512,
        vector_policy: str = "window2",
        nograd_no_clone: bool = True,
        replica_transport: str = "p2p",
        overlap_weights: bool = False,
    ) -> None:
        """Bind validated invocation resources without retaining gathered weight views."""
        if hidden_size % 64 or intermediate_size % 64:
            raise ValueError("MXFP8 hidden and intermediate dimensions must be divisible by 64.")
        if vector_policy not in ("window2", "legacy_odd"):
            raise ValueError("MXFP8 vector_policy must be window2 or legacy_odd.")
        if not isinstance(nograd_no_clone, bool):
            raise ValueError("nograd_no_clone must be a boolean.")
        if replica_transport not in ("p2p", "shmem_signal_kernel_gradient"):
            raise ValueError("MXFP8 replica transport must be p2p or shmem_signal_kernel_gradient.")
        if replica_transport != "p2p" and vector_policy != "window2":
            raise ValueError("MXFP8 kernel gradient return requires window2 queues.")
        if not isinstance(overlap_weights, bool):
            raise ValueError("MXFP8 overlap_weights must be a boolean.")
        if overlap_weights and replica_transport != "shmem_signal_kernel_gradient":
            raise ValueError("MXFP8 projection prefetch requires the symmetric FP32 gradient inbox.")
        if overlap_weights and os.environ.get("HCCL_OP_EXPANSION_MODE") != "AICPU_CacheDisable":
            raise ValueError("MXFP8 projection prefetch requires HCCL_OP_EXPANSION_MODE=AICPU_CacheDisable.")
        udma_gradients = mx_udma_gradient_enabled(replica_slots_per_rank, replica_transport)
        depth, row_only = _mx_window_options(vector_policy)
        source_row_fp8 = source_row_fp8_enabled(vector_policy=vector_policy, row_only=row_only,
                                               replica_slots=replica_slots_per_rank, hidden_size=hidden_size)
        config = ExpertReplicaConfig(num_experts, ep_size, replica_slots_per_rank)
        self.replica_planner = "cpu"
        self.replica_min_rows = replica_min_rows
        self.replica_target_load = None
        self.replica_minimum_load_reduction = None
        self.replica_transport = replica_transport
        self._cost_model_enabled = False
        specification = {
            "local_num_tokens": local_num_tokens,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_experts": config.physical_experts,
            "logical_num_experts": num_experts,
            "top_k": top_k,
            "ep_size": ep_size,
            "ep_group": ep_group,
            "expert_weight_layout": "EIO",
            "swiglu_limit": None,
            "dispatch_mode": "push",
            "initial_capacity_factor": initial_capacity_factor,
            "capacity_growth_factor": capacity_growth_factor,
            "dispatch_split": dispatch_split,
            "combine_split": combine_split,
            "replica_slots_per_rank": replica_slots_per_rank,
            "replica_transport": replica_transport,
            "mx_weight_prefetch": overlap_weights,
            "mx_gradient_udma": udma_gradients,
            "auto_tokens": False,
            "compute_stages": 6,
            "mx_vector_policy": vector_policy,
            "mx_swiglu_row_only": row_only,
            "mx_source_row_fp8": source_row_fp8,
            "mx_ready_depth": depth,
            "mx_poll_policy": "0",
            "_replica_cost_model_enabled": False,
        }
        specification = self._resolve_specification(specification)
        self._initialize_execution_specification(specification, config, False)
        self.nograd_no_clone = nograd_no_clone

    def _create_execution_resources(
        self, tensor: torch.Tensor, *, shared: bool, active_specifications: tuple[Mapping[str, Any], ...]
    ) -> _MXFP8Resources:
        return _MXFP8Resources(
            self._resource_group.specification, tensor, shared=shared, active_specifications=active_specifications
        )

    def _validate_tensors(self, hidden_states: torch.Tensor, expert_weights: tuple) -> None:
        if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_npu:
            raise TypeError("MXFP8 MegaMoE requires BF16 NPU hidden states.")
        for root, shape in zip(expert_weights, self._weight_shapes):
            MXFP8ExpertWeightBinding(root, shape)

    @property
    def _weight_shapes(self) -> tuple[tuple[int, int, int], ...]:
        return (
            (self.local_experts, self.hidden_size, 2 * self.intermediate_size),
            (self.local_experts, self.intermediate_size, self.hidden_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        expert_weights: tuple[torch.Tensor, torch.Tensor],
        tokens_per_expert: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Use final-route counts and resolve both physical weight axes per phase.

        Args:
            hidden_states: Original token activations with the configured hidden width.
            topk_ids: Final logical expert IDs produced by the Router.
            topk_weights: Differentiable probabilities for the same selected routes.
        """
        flat = self._validate_forward_inputs(hidden_states, topk_ids, topk_weights, tokens_per_expert, expert_weights)
        if flat.shape[0] != self._resource_group.specification["local_num_tokens"]:
            raise ValueError("MXFP8 source rows differ from the configured fixed token count.")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("MXFP8 push transport does not support graph capture.")
        resources = self._get_execution_resources(flat)
        _checkpoint_contract()
        replica_route = None
        if self.replica_slots_per_rank:
            mask = (
                int(torch.is_grad_enabled())
                | (int(flat.requires_grad) << 1)
                | (int(expert_weights[0].requires_grad) << 2)
                | (int(expert_weights[1].requires_grad) << 3)
                | (int(topk_weights.requires_grad) << 4)
            )
            with torch.no_grad():
                replica_route = self._prepare_replica_route(
                    topk_ids, resources, flat, mask, tokens_per_expert,
                    count_header_cache=resources.replica_headers,
                    complete_experts=bool(mask & 1) or get_recompute_state() is not None,
                )
                topk_ids = replica_route.physical_ids
                tokens_per_expert = replica_route.counts_by_source[resources.spec.rank_id]
        execution = MXFP8Execution(
            resources, nograd_no_clone=self.nograd_no_clone, replica_route=replica_route, counts=tokens_per_expert
        )
        return execute_mxfp8_experts(
            hidden_states, *expert_weights, topk_ids, topk_weights, self._weight_shapes, execution
        )
