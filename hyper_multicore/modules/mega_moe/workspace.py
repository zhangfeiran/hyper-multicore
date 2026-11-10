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
"""Owned directional workspace for managed Torch MegaMoe execution."""

from __future__ import annotations

import math
import threading
import weakref
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch

from hyper_multicore.replica.signal_transport import (
    GRADIENT_BAND_TRANSPORT,
    KERNEL_GRADIENT_TRANSPORT_MODES,
    SignalReplicaTransport,
    signal_storage_bytes,
)
from hyper_multicore import _collective, shmem
from hyper_multicore.scheduler.config import (
    MIN_EVENT_CAPACITY,
    event_workspace_bytes,
    mega_moe_event_capacity,
)
from hyper_multicore.shmem import _lifecycle
from hyper_multicore.shmem._runtime import configured_udma_qp_count

from .spec import MegaMoeSpec, initial_receive_capacity, _route_metadata_rows, _supports_indexed_backward_pair
from .indexed_backward import BACKWARD_METADATA_PREFIX_ROWS
from .udma_order import SUBMISSION_ORDER, validate_transport

# The composed Cube-only GMM and SwiGLU-grad kernels do not use these legacy
# tensor arguments. Keep non-empty ABI placeholders; native tiling reserves
# the separate CANN library workspace needed by the enclosing operator.
_GMM_WORKSPACE_BYTES = 512
_SWIGLU_GRAD_WORKSPACE_BYTES = 512
_WORKSPACE_ALIGNMENT = 512
_SYMMETRIC_FIELDS = (
    "source_buffer", "expert_buffer", "routed_buffer", "forward_event_counters", "backward_event_counters",
    "replica_inbox", "route_metadata", "backward_route_metadata",
)


def _workspace_allocation_specs(
    *,
    dispatch_mode: str,
    routed_slots: int,
    receive_capacity: int,
    hidden_size: int,
    ep_size: int,
    num_experts: int,
    dtype: torch.dtype,
    replica_inbox_bytes: int = 0,
    shared_expert: bool = False,
    compute_stages: int = 3,
    forward_ingress_mode: str = "legacy",
    backward_ingress_mode: str = "legacy",
    top_k: int = 1,
) -> tuple[tuple[str, _lifecycle._AllocationSpec], ...]:  # pylint: disable=protected-access
    """Return the single ordered physical Allocation description for a workspace."""
    data_field = "source_buffer" if dispatch_mode == "pull" else "expert_buffer"
    data_rows = routed_slots if dispatch_mode == "pull" else receive_capacity
    if backward_ingress_mode == "indexed":
        if not _supports_indexed_backward_pair(forward_ingress_mode, dispatch_mode) or top_k <= 0:
            raise ValueError("Compact backward allocations require original-row indexed or peer pull ingress")
        data_rows = routed_slots // top_k
    event_bytes = event_workspace_bytes(
        ep_size, num_experts, shared_expert=shared_expert, compute_stages=compute_stages)
    allocation_spec = _lifecycle._AllocationSpec  # pylint: disable=protected-access
    allocations = (
        (data_field, allocation_spec((data_rows, hidden_size), dtype, _WORKSPACE_ALIGNMENT)),
        ("routed_buffer", allocation_spec((routed_slots, hidden_size), dtype, _WORKSPACE_ALIGNMENT)),
        ("forward_event_counters", allocation_spec((event_bytes,), torch.uint8, _WORKSPACE_ALIGNMENT)),
        ("backward_event_counters", allocation_spec((event_bytes,), torch.uint8, _WORKSPACE_ALIGNMENT)),
    )
    if replica_inbox_bytes:
        allocations += (("replica_inbox", allocation_spec(
            (replica_inbox_bytes,), torch.uint8, _WORKSPACE_ALIGNMENT)),)
    if forward_ingress_mode != "legacy":
        # The tail holds one private 128-byte cache line per diagnostic task.
        # Peers never read it; scalar counter stores cannot share cache lines.
        metadata_rows = _route_metadata_rows(routed_slots, num_experts, ep_size, forward_ingress_mode)
        allocations += (("route_metadata", allocation_spec(
            (metadata_rows, 4), torch.int32, _WORKSPACE_ALIGNMENT)),)
    if backward_ingress_mode == "indexed":
        allocations += (("backward_route_metadata", allocation_spec(
            (BACKWARD_METADATA_PREFIX_ROWS + routed_slots, 4), torch.int32, _WORKSPACE_ALIGNMENT)),)
    return allocations


def _replica_inbox_bytes(specification: Mapping[str, Any]) -> int:
    """Return the replica payload tracked by SHMEM restart transactions."""
    budget = specification.get("replica_slots_per_rank", 0)
    mode = specification.get("replica_transport", GRADIENT_BAND_TRANSPORT)
    if not budget or mode == "p2p":
        return 0
    if mode not in KERNEL_GRADIENT_TRANSPORT_MODES:
        raise ValueError("Unsupported replica_transport; expected p2p or a kernel-gradient transport")
    hidden, intermediate = specification["hidden_size"], specification["intermediate_size"]
    return signal_storage_bytes(((hidden, 2 * intermediate), (intermediate, hidden)), budget,
                                specification["ep_size"], 2, gradient_bands=mode == GRADIENT_BAND_TRANSPORT)


def _spec_workspace_bytes(
    specification: Mapping[str, Any], dtype: torch.dtype, *, receive_capacity: int | None = None,
) -> int:
    """Return planned symmetric bytes for one execution-resource group."""
    local_tokens = specification["local_num_tokens"]
    routed_slots = local_tokens * specification["top_k"]
    capacity = initial_receive_capacity(specification)
    if receive_capacity is not None:
        capacity = receive_capacity
    if specification.get("dispatch_mode", "push") == "pull":
        capacity = routed_slots
    allocations = _workspace_allocation_specs(
        dispatch_mode=specification.get("dispatch_mode", "push"),
        routed_slots=routed_slots,
        receive_capacity=capacity,
        hidden_size=specification["hidden_size"],
        ep_size=specification["ep_size"],
        num_experts=specification["num_experts"],
        dtype=dtype,
        replica_inbox_bytes=_replica_inbox_bytes(specification),
        shared_expert=bool(specification.get("shared_intermediate_size", 0)),
        compute_stages=specification.get("compute_stages", 3),
        forward_ingress_mode=specification.get("forward_ingress_mode", "legacy"),
        backward_ingress_mode=specification.get("backward_ingress_mode", "legacy"),
        top_k=specification["top_k"],
    )
    return sum(
        spec.dtype.itemsize * math.prod(spec.shape)
        + spec.alignment_bytes - 1
        for _, spec in allocations
    )


@dataclass
class MegaMoeWorkspace:
    """Buffers owned by one standalone or explicitly shared resource group."""

    shared: bool
    dtype: Any | None = None
    device: Any | None = None
    expert_capacity: int = 0
    routed_slots: int = 0
    dispatch_mode: str = "push"
    source_buffer: Any | None = None
    expert_buffer: Any | None = None
    routed_buffer: Any | None = None
    route_metadata: Any | None = None
    backward_route_metadata: Any | None = None
    backward_ingress_mode: str = "legacy"
    forward_event_counters: Any | None = None
    backward_event_counters: Any | None = None
    replica_inbox: Any | None = None
    replica_provider: Any | None = None
    gmm_workspace: Any | None = None
    swiglu_grad_workspace: Any | None = None
    completion_event: Any | None = None
    replica_runtime_images: dict[int, tuple[Any, bytes, Any]] = field(default_factory=dict, repr=False)
    replica_task_prefixes: dict[int, tuple[Any, tuple[int, int]]] = field(default_factory=dict, repr=False)
    in_use: bool = False
    closing: bool = False
    used: bool = False
    event_counter_bytes: int = MIN_EVENT_CAPACITY * 4
    forward_ready_initialized: bool = False
    backward_ready_initialized: bool = False
    capacity_floor: int = 0
    heap_manager: Any | None = field(default=None, repr=False)
    route_cache: Any | None = field(default=None, repr=False)
    route_header_cache: Any | None = field(default=None, repr=False)
    route_cache_probe_after: int = field(default=0, repr=False)
    lock: Any = field(default_factory=threading.Lock, repr=False)
    graphs: Any = field(default_factory=weakref.WeakSet, repr=False)
    uses_udma: bool = field(init=False, repr=False)
    udma_qp_count: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Bind process-wide QP ownership to this workspace's data engine."""
        self.uses_udma = validate_transport()
        self.udma_qp_count = configured_udma_qp_count()

    def describe_restart_allocations(
        self,
        spec: MegaMoeSpec,
        dtype: torch.dtype,
        receive_capacity: int,
    ) -> tuple[_lifecycle._AllocationReplacement, ...]:  # pylint: disable=protected-access
        """Project materialized symmetric fields into ordered restart replacements.

        Args:
            spec: Validated physical expert topology and current capacity.
            dtype: Activation dtype used by the symmetric workspace.
            receive_capacity: Committed receive-row capacity for the replacement workspace.
        """
        if self.completion_event is None:
            return ()
        targets = _workspace_allocation_specs(
            dispatch_mode=spec.dispatch_mode,
            forward_ingress_mode=getattr(spec, "forward_ingress_mode", "legacy"),
            backward_ingress_mode=getattr(spec, "backward_ingress_mode", "legacy"),
            top_k=spec.top_k,
            routed_slots=spec.routed_slots,
            receive_capacity=receive_capacity,
            hidden_size=spec.hidden_size,
            ep_size=spec.ep_size,
            num_experts=spec.num_experts,
            dtype=dtype,
            replica_inbox_bytes=_replica_inbox_bytes(vars(spec)),
            shared_expert=bool(getattr(spec, "shared_intermediate_size", 0)),
            compute_stages=getattr(spec, "compute_stages", 3),
        )
        replacement = _lifecycle._AllocationReplacement  # pylint: disable=protected-access
        return tuple(replacement(getattr(self, field_name), target) for field_name, target in targets)

    def validate_restart_allocations(
        self,
        spec: MegaMoeSpec,
        dtype: torch.dtype,
        current_capacity: int,
        replacements: tuple[_lifecycle._AllocationReplacement, ...],  # pylint: disable=protected-access
    ) -> bool:
        """Validate the read-only workspace state captured for restart preflight.

        Args:
            spec: Validated physical expert topology and current capacity.
            dtype: Activation dtype used by the symmetric workspace.
            current_capacity: Receive-row capacity recorded in the current generation.
            replacements: Ordered symmetric allocation replacements captured during preflight.
        """
        if self.in_use:
            raise RuntimeError("active workspace lease prevents heap growth")
        if self.closing:
            raise RuntimeError("closing workspace prevents heap growth")
        materialized = self.completion_event is not None
        if not materialized:
            if replacements:
                raise RuntimeError("lazy workspace must not describe restart Allocations")
            return False
        expected = _workspace_allocation_specs(
            dispatch_mode=spec.dispatch_mode,
            forward_ingress_mode=getattr(spec, "forward_ingress_mode", "legacy"),
            backward_ingress_mode=getattr(spec, "backward_ingress_mode", "legacy"),
            top_k=spec.top_k,
            routed_slots=spec.routed_slots,
            receive_capacity=current_capacity,
            hidden_size=spec.hidden_size,
            ep_size=spec.ep_size,
            num_experts=spec.num_experts,
            dtype=dtype,
            replica_inbox_bytes=_replica_inbox_bytes(vars(spec)),
            shared_expert=bool(getattr(spec, "shared_intermediate_size", 0)),
            compute_stages=getattr(spec, "compute_stages", 3),
        )
        if len(replacements) != len(expected):
            raise RuntimeError("materialized workspace has an incomplete restart Allocation mapping")
        for (field_name, current_spec), replacement in zip(expected, replacements):
            tensor = getattr(self, field_name)
            if replacement.current is not tensor or tensor is None:
                raise RuntimeError("workspace restart Allocation mapping changed during preflight")
            if tuple(tensor.shape) != current_spec.shape or tensor.dtype != current_spec.dtype:
                raise RuntimeError("workspace symmetric Allocation does not match its committed layout")
        return True

    def bind_restarted_allocations(
        self,
        spec: MegaMoeSpec,
        dtype: torch.dtype,
        device: Any,
        receive_capacity: int,
        tensors: tuple[torch.Tensor, ...],
    ) -> None:
        """Bind validated replacement Tensors without allocating or running collectives.

        Args:
            spec: Validated physical expert topology and current capacity.
            dtype: Activation dtype used by the symmetric workspace.
            device: Device owning execution and diagnostic storage.
            receive_capacity: Committed receive-row capacity for the replacement workspace.
            tensors: New symmetric allocation views returned by the restart transaction.
        """
        targets = _workspace_allocation_specs(
            dispatch_mode=spec.dispatch_mode,
            forward_ingress_mode=getattr(spec, "forward_ingress_mode", "legacy"),
            backward_ingress_mode=getattr(spec, "backward_ingress_mode", "legacy"),
            top_k=spec.top_k,
            routed_slots=spec.routed_slots,
            receive_capacity=receive_capacity,
            hidden_size=spec.hidden_size,
            ep_size=spec.ep_size,
            num_experts=spec.num_experts,
            dtype=dtype,
            replica_inbox_bytes=_replica_inbox_bytes(vars(spec)),
            shared_expert=bool(getattr(spec, "shared_intermediate_size", 0)),
            compute_stages=getattr(spec, "compute_stages", 3),
        )
        self.source_buffer = None
        self.expert_buffer = None
        for tensor, (field_name, _) in zip(tensors, targets):
            setattr(self, field_name, tensor)
        self.dispatch_mode = spec.dispatch_mode
        self.backward_ingress_mode = getattr(spec, "backward_ingress_mode", "legacy")
        self.dtype = dtype
        self.device = device
        self.expert_capacity = receive_capacity
        self.routed_slots = spec.routed_slots
        self.event_counter_bytes = mega_moe_event_capacity(
            spec.num_experts, spec.ep_size, shared_expert=bool(getattr(spec, "shared_intermediate_size", 0)),
            compute_stages=getattr(spec, "compute_stages", 3),
        ) * 4
        self.forward_ready_initialized = False
        self.backward_ready_initialized = False
        for _, _, image in self.replica_runtime_images.values():
            image.untyped_storage().resize_(0)
        self.replica_runtime_images.clear()
        self.replica_task_prefixes.clear()
        self.replica_provider = None
        if self.replica_inbox is not None:
            self._allocate_replica_storage(spec)

    def validate_restarted_allocations(
        self,
        spec: MegaMoeSpec,
        dtype: torch.dtype,
        receive_capacity: int,
        tensors: tuple[torch.Tensor, ...],
    ) -> None:
        """Validate a complete replacement slice without modifying workspace state.

        Args:
            spec: Validated physical expert topology and current capacity.
            dtype: Activation dtype used by the symmetric workspace.
            receive_capacity: Committed receive-row capacity for the replacement workspace.
            tensors: New symmetric allocation views returned by the restart transaction.
        """
        targets = _workspace_allocation_specs(
            dispatch_mode=spec.dispatch_mode,
            forward_ingress_mode=getattr(spec, "forward_ingress_mode", "legacy"),
            backward_ingress_mode=getattr(spec, "backward_ingress_mode", "legacy"),
            top_k=spec.top_k,
            routed_slots=spec.routed_slots,
            receive_capacity=receive_capacity,
            hidden_size=spec.hidden_size,
            ep_size=spec.ep_size,
            num_experts=spec.num_experts,
            dtype=dtype,
            replica_inbox_bytes=_replica_inbox_bytes(vars(spec)),
            shared_expert=bool(getattr(spec, "shared_intermediate_size", 0)),
            compute_stages=getattr(spec, "compute_stages", 3),
        )
        if len(tensors) != len(targets):
            raise RuntimeError("SHMEM restart returned an unexpected Allocation count")
        for tensor, (_, target) in zip(tensors, targets):
            if tuple(tensor.shape) != target.shape or tensor.dtype != target.dtype:
                raise RuntimeError("SHMEM restart returned an unexpected Allocation specification")

    def track_graph(self, context: Any) -> None:
        """Retain a weak lease while this context can still run backward.

        Args:
            context: Autograd context that may still access workspace buffers.
        """
        self.graphs.add(context)

    def can_close(self) -> bool:
        """Check active calls and weak graph leases without unpacking saved tensors."""
        return not self.in_use and not self.graphs

    def release_graph(self, context: Any) -> None:
        """Retire this context's lease after successful non-retained backward.

        Args:
            context: Autograd context whose final backward completed.
        """
        self.graphs.discard(context)

    def ensure(self, spec: MegaMoeSpec, dtype: Any, device: Any) -> None:
        """Allocate the fixed configured route capacity once.

        Args:
            spec: First-forward-bound shape and capacity specification.
            dtype: Element dtype for symmetric data buffers.
            device: Device that owns local and symmetric buffers.
        """
        with self._access():
            self._ensure_buffers(spec, dtype, device)

    def _access(self) -> Any:
        """Exclude submissions during a root Runtime restart."""
        return self.heap_manager.access() if self.heap_manager is not None else nullcontext()

    def _ensure_buffers(self, spec: MegaMoeSpec, dtype: Any, device: Any) -> None:
        """Allocate this workspace at its current monotonic capacity."""
        if self.closing:
            raise RuntimeError("cannot allocate a closing MegaMoe workspace.")
        requested_capacity = max(spec.receive_capacity, self.capacity_floor)
        if self.expert_buffer is not None or self.source_buffer is not None:
            compatible = (
                self.dispatch_mode == spec.dispatch_mode
                and self.backward_ingress_mode == getattr(spec, "backward_ingress_mode", "legacy")
                and self.dtype == dtype
                and self.device == device
                and self.routed_slots == spec.routed_slots
                and self.expert_capacity == requested_capacity
            )
            if not compatible:
                raise ValueError("MegaMoe workspace cannot change device, dtype, routed shape, or capacity.")
            return

        allocations = _workspace_allocation_specs(
            dispatch_mode=spec.dispatch_mode,
            forward_ingress_mode=getattr(spec, "forward_ingress_mode", "legacy"),
            backward_ingress_mode=getattr(spec, "backward_ingress_mode", "legacy"),
            top_k=spec.top_k,
            routed_slots=spec.routed_slots,
            receive_capacity=requested_capacity,
            hidden_size=spec.hidden_size,
            ep_size=spec.ep_size,
            num_experts=spec.num_experts,
            dtype=dtype,
            replica_inbox_bytes=_replica_inbox_bytes(vars(spec)),
            shared_expert=bool(getattr(spec, "shared_intermediate_size", 0)),
            compute_stages=getattr(spec, "compute_stages", 3),
        )
        self.dispatch_mode = spec.dispatch_mode
        self.backward_ingress_mode = getattr(spec, "backward_ingress_mode", "legacy")
        self.dtype = dtype
        self.device = device
        self.expert_capacity = requested_capacity
        self.routed_slots = spec.routed_slots
        self.event_counter_bytes = mega_moe_event_capacity(
            spec.num_experts, spec.ep_size, shared_expert=bool(getattr(spec, "shared_intermediate_size", 0)),
            compute_stages=getattr(spec, "compute_stages", 3),
        ) * 4
        try:
            for field_name, allocation in allocations:
                setattr(self, field_name, shmem.empty(
                    allocation.shape,
                    dtype=allocation.dtype,
                    alignment=allocation.alignment_bytes,
                ))
            if spec.replica_slots_per_rank and spec.replica_transport != "p2p":
                self._allocate_replica_storage(spec)
            self.gmm_workspace = torch.empty(
                (_GMM_WORKSPACE_BYTES,),
                dtype=torch.uint8,
                device=device,
            )
            self.swiglu_grad_workspace = torch.empty(
                (_SWIGLU_GRAD_WORKSPACE_BYTES,),
                dtype=torch.uint8,
                device=device,
            )
            self.completion_event = torch.npu.Event()
        except Exception:
            self._free_symmetric_tensors()
            self._free_local_tensors()
            raise

    def _allocate_replica_storage(self, spec: MegaMoeSpec) -> None:
        """Recreate provider state with the new symmetric heap generation."""
        shapes = ((spec.hidden_size, 2 * spec.intermediate_size), (spec.intermediate_size, spec.hidden_size))
        size = signal_storage_bytes(shapes, spec.replica_slots_per_rank, spec.ep_size, 2,
                                    gradient_bands=spec.replica_transport == GRADIENT_BAND_TRANSPORT)
        if self.replica_inbox is None:
            self.replica_inbox = shmem.empty((size,), dtype=torch.uint8, alignment=_WORKSPACE_ALIGNMENT)
        self.replica_provider = SignalReplicaTransport(shmem, self.replica_inbox,
                                                      spec.replica_slots_per_rank, spec.ep_size,
                                                      gradient_bands=spec.replica_transport == GRADIENT_BAND_TRANSPORT)

    def prepare_event_counters(self, *, forward: bool) -> Any:
        """Reset per-call counters while preserving the ready generation."""
        if forward:
            field_name = "forward_event_counters"
            events = self.forward_event_counters
            ready_initialized = self.forward_ready_initialized
        else:
            field_name = "backward_event_counters"
            events = self.backward_event_counters
            ready_initialized = self.backward_ready_initialized
        if events is None:
            raise RuntimeError(f"MegaMoe workspace {field_name} is not initialized.")
        events[:self.event_counter_bytes].zero_()
        if events.numel() == self.event_counter_bytes or ready_initialized:
            return events

        # Initialize the persistent generation before any peer can signal it.
        events[self.event_counter_bytes:].zero_()
        shmem.host_barrier()
        if forward:
            self.forward_ready_initialized = True
        else:
            self.backward_ready_initialized = True
        return events

    def wait_for_reuse(self) -> None:
        """Order count exchange after the previous workspace lease."""
        with self._access(), self.lock:
            if self.used:
                torch.npu.current_stream(self.device).wait_event(self.completion_event)

    def claim(self) -> None:
        """Claim the serial workspace and order it after the previous stream."""
        with self._access(), self.lock:
            if configured_udma_qp_count() != self.udma_qp_count:
                raise RuntimeError("UDMA QP count differs from the bound workspace")
            if self.closing:
                raise RuntimeError("cannot claim a closing MegaMoe workspace.")
            if self.in_use:
                raise RuntimeError("MegaMoe execution resources do not support concurrent calls.")
            if self.used:
                torch.npu.current_stream().wait_event(self.completion_event)
            if self.uses_udma:
                SUBMISSION_ORDER.claim(torch.npu.current_stream())
            self.in_use = True

    def release(self) -> None:
        """Record completion ordering and release the serial workspace lease."""
        with self._access(), self.lock:
            if not self.in_use:
                raise RuntimeError("MegaMoe workspace was released without an active lease.")
            self.completion_event.record(torch.npu.current_stream())
            if self.uses_udma:
                SUBMISSION_ORDER.release(torch.npu.current_stream(), self.completion_event)
            self.in_use = False
            self.used = True

    def _free_symmetric_tensors(self) -> None:
        """Free each unique SHMEM allocation and invalidate its tensor view."""
        if self.closing:
            remaining = tuple(getattr(self, name) is not None for name in _SYMMETRIC_FIELDS)
            peers = _collective.exchange(
                remaining,
                group=self._root_group(),
                failure_prefix="MegaMoe symmetric workspace release failed while converging rank results",
            )
            if any(peer != remaining for peer in peers):
                raise RuntimeError("partial symmetric teardown differs across ranks; restart the process")
        self.replica_provider = None
        for field_name in _SYMMETRIC_FIELDS:
            tensor = getattr(self, field_name)
            if tensor is None:
                continue
            shmem.free(tensor)
            setattr(self, field_name, None)
        self.expert_capacity = 0
        self.routed_slots = 0
        self.forward_ready_initialized = False
        self.backward_ready_initialized = False

    def _free_local_tensors(self) -> None:
        """Release local-only workspaces after all queued kernels complete."""
        for _, _, image in self.replica_runtime_images.values():
            image.untyped_storage().resize_(0)
        self.replica_runtime_images.clear()
        self.replica_task_prefixes.clear()
        for field_name in ("gmm_workspace", "swiglu_grad_workspace"):
            tensor = getattr(self, field_name)
            if tensor is None:
                continue
            tensor.untyped_storage().resize_(0)
            setattr(self, field_name, None)

    def _root_group(self) -> Any:
        """Require an owning manager for collective teardown without a default-group fallback."""
        if self.heap_manager is None:
            raise RuntimeError("MegaMoe workspace collective teardown requires an owning HeapManager")
        return self.heap_manager.root_group

    def close(self) -> None:
        """Synchronize and release physical buffers on the owning HeapManager's Root."""
        with self._access(), self.lock:
            if self.in_use:
                raise RuntimeError("cannot close MegaMoe workspace during an active call.")
            if not self.can_close():
                raise RuntimeError("cannot close MegaMoe workspace while backward graphs still need its buffers.")
            empty = all(getattr(self, name) is None for name in (
                *_SYMMETRIC_FIELDS, "gmm_workspace", "swiglu_grad_workspace", "completion_event",
            )) and not self.replica_runtime_images
            root_group = self._root_group()
            self.closing = True
            self.route_cache = None
            self.route_header_cache = None
            self.route_cache_probe_after = 0
        # An empty rank must still participate when peers retain buffers after a failure.
        peers = _collective.exchange(
            empty, group=root_group, failure_prefix="MegaMoe workspace close failed while converging rank results",
        )
        if all(peers):
            self.closing = True
            return
        _collective.collective_call(
            lambda: None if empty else torch.npu.synchronize(self.device),
            stage="MegaMoe workspace synchronization",
            group=root_group,
        )
        # Workspace teardown requires each collective barrier to complete before
        # the following free or local Tensor release.
        shmem.host_barrier()
        _collective.collective_call(
            self._free_symmetric_tensors,
            stage="MegaMoe symmetric workspace release",
            group=root_group,
        )
        shmem.host_barrier()
        _collective.collective_call(
            self._free_local_tensors,
            stage="MegaMoe local workspace release",
            group=root_group,
        )
        self.dtype = None
        self.device = None
        self.completion_event = None
        self.used = False
        self.closing = True
