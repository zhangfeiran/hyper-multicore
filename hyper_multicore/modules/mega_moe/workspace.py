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

from hyper_multicore import _collective, shmem
from hyper_multicore.replica.signal_transport import (
    GRADIENT_BAND_TRANSPORT,
    KERNEL_GRADIENT_TRANSPORT_MODES,
    SignalReplicaTransport,
    signal_storage_bytes,
)
from hyper_multicore.scheduler.config import MIN_EVENT_CAPACITY, event_workspace_bytes, mega_moe_event_capacity
from hyper_multicore.shmem import _lifecycle

from .spec import MegaMoeSpec, initial_receive_capacity
from . import rdma

_GMM_WORKSPACE_BYTES = 512
_SWIGLU_GRAD_WORKSPACE_BYTES = 512
_WORKSPACE_ALIGNMENT = 512
_SYMMETRIC_FIELDS = (
    "source_buffer",
    "source_indices",
    "expert_buffer",
    "return_source_buffer",
    "routed_buffer",
    "forward_event_counters",
    "backward_event_counters",
    "replica_inbox",
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
    shmem_data_engine: str = "mte",
    dispatch_split: int = 128,
    combine_split: int = 128,
) -> tuple[tuple[str, _lifecycle._AllocationSpec], ...]:
    """Return the single ordered physical Allocation description for a workspace."""
    data_field = "source_buffer" if dispatch_mode == "pull" else "expert_buffer"
    data_rows = routed_slots if dispatch_mode == "pull" else receive_capacity
    event_bytes = (event_workspace_bytes(ep_size, num_experts) if shmem_data_engine == "mte"
                   else rdma.layout(ep_size, num_experts, routed_slots, dispatch_split, combine_split)[3])
    allocation_spec = _lifecycle._AllocationSpec
    allocations = (
        (data_field, allocation_spec((data_rows, hidden_size), dtype, _WORKSPACE_ALIGNMENT)),
        ("routed_buffer", allocation_spec((routed_slots, hidden_size), dtype, _WORKSPACE_ALIGNMENT)),
        ("forward_event_counters", allocation_spec((event_bytes,), torch.uint8, _WORKSPACE_ALIGNMENT)),
        ("backward_event_counters", allocation_spec((event_bytes,), torch.uint8, _WORKSPACE_ALIGNMENT)),
    )
    if shmem_data_engine != "mte":
        additional = "expert_buffer" if dispatch_mode == "pull" else "source_buffer"
        additional_rows = receive_capacity if dispatch_mode == "pull" else routed_slots
        allocations += (
            (additional, allocation_spec((additional_rows, hidden_size), dtype, _WORKSPACE_ALIGNMENT)),
            ("source_indices", allocation_spec((routed_slots,), torch.int32, _WORKSPACE_ALIGNMENT)),
            ("return_source_buffer", allocation_spec((receive_capacity, hidden_size), dtype, _WORKSPACE_ALIGNMENT)),
        )
    if replica_inbox_bytes:
        allocations += (("replica_inbox", allocation_spec((replica_inbox_bytes,), torch.uint8, _WORKSPACE_ALIGNMENT)),)
    return allocations


def _replica_matrix_shapes(hidden: int, intermediate: int, layout: str) -> tuple[tuple[int, int], ...]:
    """Match guest buffers and gradient bands to the physical expert weight layout."""
    if layout == "EOI":
        return ((2 * intermediate, hidden), (hidden, intermediate))
    if layout == "EIO":
        return ((hidden, 2 * intermediate), (intermediate, hidden))
    raise ValueError(f"Unsupported expert weight layout: {layout!r}")


def _replica_inbox_bytes(specification: Mapping[str, Any]) -> int:
    """Return the replica payload tracked by SHMEM restart transactions."""
    budget = specification.get("replica_slots_per_rank", 0)
    mode = specification.get("replica_transport", GRADIENT_BAND_TRANSPORT)
    if not budget or mode == "p2p":
        return 0
    if mode not in KERNEL_GRADIENT_TRANSPORT_MODES:
        raise ValueError("Unsupported replica_transport; expected p2p or a kernel-gradient transport")
    hidden, intermediate = (specification["hidden_size"], specification["intermediate_size"])
    return signal_storage_bytes(
        _replica_matrix_shapes(hidden, intermediate, specification.get("expert_weight_layout", "EIO")),
        budget,
        specification["ep_size"],
        2,
        gradient_bands=mode == GRADIENT_BAND_TRANSPORT,
        storage_slots=specification.get("_replica_inbox_slots"),
    )


def _spec_workspace_bytes(
    specification: Mapping[str, Any], dtype: torch.dtype, *, receive_capacity: int | None = None
) -> int:
    """Return planned symmetric bytes for one execution-resource group."""
    local_tokens = specification["local_num_tokens"]
    routed_slots = local_tokens * specification["top_k"]
    capacity = initial_receive_capacity(specification)
    if receive_capacity is not None:
        capacity = receive_capacity
    if specification.get("dispatch_mode", "push") == "pull" and not rdma.enabled(specification):
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
        shmem_data_engine=specification.get("shmem_data_engine", "mte"),
        dispatch_split=specification.get("dispatch_split", 128),
        combine_split=specification.get("combine_split", 128),
    )
    return sum((spec.dtype.itemsize * math.prod(spec.shape) + spec.alignment_bytes - 1 for _, spec in allocations))


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
    source_indices: Any | None = None
    expert_buffer: Any | None = None
    return_source_buffer: Any | None = None
    routed_buffer: Any | None = None
    forward_event_counters: Any | None = None
    backward_event_counters: Any | None = None
    replica_inbox: Any | None = None
    replica_provider: Any | None = None
    gmm_workspace: Any | None = None
    swiglu_grad_workspace: Any | None = None
    completion_event: Any | None = None
    replica_runtime_images: dict[int, tuple[Any, bytes, Any]] = field(default_factory=dict, repr=False)
    replica_task_prefixes: dict[int, tuple[Any, tuple[int, int, int, int]]] = field(default_factory=dict, repr=False)
    in_use: bool = False
    closing: bool = False
    used: bool = False
    event_counter_bytes: int = MIN_EVENT_CAPACITY * 4
    rdma_enabled: bool = False
    rdma_ep_size: int = 1
    rdma_num_experts: int = 1
    rdma_dispatch_split: int = 128
    rdma_combine_split: int = 128
    rdma_value_bytes: int = 0
    forward_ready_initialized: bool = False
    backward_ready_initialized: bool = False
    capacity_floor: int = 0
    heap_manager: Any | None = field(default=None, repr=False)
    route_cache: Any | None = field(default=None, repr=False)
    route_header_cache: Any | None = field(default=None, repr=False)
    route_cache_probe_after: int = field(default=0, repr=False)
    lock: Any = field(default_factory=threading.Lock, repr=False)
    graphs: Any = field(default_factory=weakref.WeakSet, repr=False)

    def describe_restart_allocations(
        self, spec: MegaMoeSpec, dtype: torch.dtype, receive_capacity: int
    ) -> tuple[_lifecycle._AllocationReplacement, ...]:
        """Project materialized symmetric fields into ordered restart replacements."""
        if self.completion_event is None:
            return ()
        targets = _workspace_allocation_specs(
            dispatch_mode=spec.dispatch_mode,
            routed_slots=spec.routed_slots,
            receive_capacity=receive_capacity,
            hidden_size=spec.hidden_size,
            ep_size=spec.ep_size,
            num_experts=spec.num_experts,
            dtype=dtype,
            replica_inbox_bytes=_replica_inbox_bytes(vars(spec)),
            shmem_data_engine=getattr(spec, "shmem_data_engine", "mte"),
            dispatch_split=getattr(spec, "dispatch_split", 128),
            combine_split=getattr(spec, "combine_split", 128),
        )
        replacement = _lifecycle._AllocationReplacement
        return tuple((replacement(getattr(self, field_name), target) for field_name, target in targets))

    def validate_restart_allocations(
        self,
        spec: MegaMoeSpec,
        dtype: torch.dtype,
        current_capacity: int,
        replacements: tuple[_lifecycle._AllocationReplacement, ...],
    ) -> bool:
        """Validate the read-only workspace state captured for restart preflight."""
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
            routed_slots=spec.routed_slots,
            receive_capacity=current_capacity,
            hidden_size=spec.hidden_size,
            ep_size=spec.ep_size,
            num_experts=spec.num_experts,
            dtype=dtype,
            replica_inbox_bytes=_replica_inbox_bytes(vars(spec)),
            shmem_data_engine=getattr(spec, "shmem_data_engine", "mte"),
            dispatch_split=getattr(spec, "dispatch_split", 128),
            combine_split=getattr(spec, "combine_split", 128),
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
        """Bind validated replacement Tensors without allocating or running collectives."""
        targets = _workspace_allocation_specs(
            dispatch_mode=spec.dispatch_mode,
            routed_slots=spec.routed_slots,
            receive_capacity=receive_capacity,
            hidden_size=spec.hidden_size,
            ep_size=spec.ep_size,
            num_experts=spec.num_experts,
            dtype=dtype,
            replica_inbox_bytes=_replica_inbox_bytes(vars(spec)),
            shmem_data_engine=getattr(spec, "shmem_data_engine", "mte"),
            dispatch_split=getattr(spec, "dispatch_split", 128),
            combine_split=getattr(spec, "combine_split", 128),
        )
        self.source_buffer = None
        self.expert_buffer = None
        self.return_source_buffer = None
        for tensor, (field_name, _) in zip(tensors, targets):
            setattr(self, field_name, tensor)
        self.rdma_enabled = rdma.enabled(spec)
        self.rdma_ep_size = spec.ep_size
        self.rdma_num_experts = spec.num_experts
        self.rdma_dispatch_split = getattr(spec, "dispatch_split", 128)
        self.rdma_combine_split = getattr(spec, "combine_split", 128)
        self.rdma_value_bytes = (spec.num_experts * ((spec.routed_slots
                                 + min(self.rdma_dispatch_split, self.rdma_combine_split) - 1)
                                 // min(self.rdma_dispatch_split, self.rdma_combine_split)) + 1) * 4
        self.dispatch_mode = spec.dispatch_mode
        self.dtype = dtype
        self.device = device
        self.expert_capacity = receive_capacity
        self.routed_slots = spec.routed_slots
        self.event_counter_bytes = mega_moe_event_capacity(spec.num_experts, spec.ep_size) * 4
        self.forward_ready_initialized = False
        self.backward_ready_initialized = False
        for _, _, image in self.replica_runtime_images.values():
            image.untyped_storage().resize_(0)
        self.route_cache = None
        self.route_header_cache = None
        self.route_cache_probe_after = 0
        self.replica_runtime_images.clear()
        self.replica_task_prefixes.clear()
        self.replica_provider = None
        if self.replica_inbox is not None:
            self._allocate_replica_storage(spec)

    def validate_restarted_allocations(
        self, spec: MegaMoeSpec, dtype: torch.dtype, receive_capacity: int, tensors: tuple[torch.Tensor, ...]
    ) -> None:
        """Validate a complete replacement slice without modifying workspace state."""
        targets = _workspace_allocation_specs(
            dispatch_mode=spec.dispatch_mode,
            routed_slots=spec.routed_slots,
            receive_capacity=receive_capacity,
            hidden_size=spec.hidden_size,
            ep_size=spec.ep_size,
            num_experts=spec.num_experts,
            dtype=dtype,
            replica_inbox_bytes=_replica_inbox_bytes(vars(spec)),
            shmem_data_engine=getattr(spec, "shmem_data_engine", "mte"),
            dispatch_split=getattr(spec, "dispatch_split", 128),
            combine_split=getattr(spec, "combine_split", 128),
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
                and self.dtype == dtype
                and (self.device == device)
                and (self.routed_slots == spec.routed_slots)
                and (self.expert_capacity == requested_capacity)
            )
            if not compatible:
                raise ValueError("MegaMoe workspace cannot change device, dtype, routed shape, or capacity.")
            return
        allocations = _workspace_allocation_specs(
            dispatch_mode=spec.dispatch_mode,
            routed_slots=spec.routed_slots,
            receive_capacity=requested_capacity,
            hidden_size=spec.hidden_size,
            ep_size=spec.ep_size,
            num_experts=spec.num_experts,
            dtype=dtype,
            replica_inbox_bytes=_replica_inbox_bytes(vars(spec)),
            shmem_data_engine=getattr(spec, "shmem_data_engine", "mte"),
            dispatch_split=getattr(spec, "dispatch_split", 128),
            combine_split=getattr(spec, "combine_split", 128),
        )
        self.rdma_enabled = rdma.enabled(spec)
        self.rdma_ep_size = spec.ep_size
        self.rdma_num_experts = spec.num_experts
        self.rdma_dispatch_split = getattr(spec, "dispatch_split", 128)
        self.rdma_combine_split = getattr(spec, "combine_split", 128)
        self.rdma_value_bytes = (spec.num_experts * ((spec.routed_slots
                                 + min(self.rdma_dispatch_split, self.rdma_combine_split) - 1)
                                 // min(self.rdma_dispatch_split, self.rdma_combine_split)) + 1) * 4
        self.dispatch_mode = spec.dispatch_mode
        self.dtype = dtype
        self.device = device
        self.expert_capacity = requested_capacity
        self.routed_slots = spec.routed_slots
        self.event_counter_bytes = mega_moe_event_capacity(spec.num_experts, spec.ep_size) * 4
        try:
            for field_name, allocation in allocations:
                setattr(
                    self,
                    field_name,
                    shmem.empty(allocation.shape, dtype=allocation.dtype, alignment=allocation.alignment_bytes),
                )
            if spec.replica_slots_per_rank and spec.replica_transport != "p2p":
                self._allocate_replica_storage(spec)
            self.gmm_workspace = torch.empty((_GMM_WORKSPACE_BYTES,), dtype=torch.uint8, device=device)
            self.swiglu_grad_workspace = torch.empty((_SWIGLU_GRAD_WORKSPACE_BYTES,), dtype=torch.uint8, device=device)
            self.completion_event = torch.npu.Event()
        except Exception:
            self._free_symmetric_tensors()
            self._free_local_tensors()
            raise

    def _allocate_replica_storage(self, spec: MegaMoeSpec) -> None:
        """Recreate provider state with the new symmetric heap generation."""
        shapes = _replica_matrix_shapes(spec.hidden_size, spec.intermediate_size, spec.expert_weight_layout)
        bands = spec.replica_transport == GRADIENT_BAND_TRANSPORT
        slots = vars(spec).get("_replica_inbox_slots")
        storage_options = {} if slots is None else {"storage_slots": slots}
        size = signal_storage_bytes(
            shapes, spec.replica_slots_per_rank, spec.ep_size, 2, gradient_bands=bands, **storage_options
        )
        if self.replica_inbox is None:
            self.replica_inbox = shmem.empty((size,), dtype=torch.uint8, alignment=_WORKSPACE_ALIGNMENT)
        self.replica_provider = SignalReplicaTransport(
            shmem,
            self.replica_inbox,
            spec.replica_slots_per_rank,
            spec.ep_size,
            gradient_bands=bands,
            **storage_options,
        )

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
        if self.rdma_enabled:
            if not ready_initialized:
                events[self.event_counter_bytes:].zero_()
            rdma.prepare_events(self, events, ready_initialized)
            if not ready_initialized:
                shmem.host_barrier()
                if forward:
                    self.forward_ready_initialized = True
                else:
                    self.backward_ready_initialized = True
            return events
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
            if self.closing:
                raise RuntimeError("cannot claim a closing MegaMoe workspace.")
            if self.in_use:
                raise RuntimeError("MegaMoe execution resources do not support concurrent calls.")
            if self.used:
                torch.npu.current_stream().wait_event(self.completion_event)
            self.in_use = True

    def release(self) -> None:
        """Record completion ordering and release the serial workspace lease."""
        with self._access(), self.lock:
            if not self.in_use:
                raise RuntimeError("MegaMoe workspace was released without an active lease.")
            self.completion_event.record(torch.npu.current_stream())
            self.in_use = False
            self.used = True

    def _free_symmetric_tensors(self) -> None:
        """Free each unique SHMEM allocation and invalidate its tensor view."""
        if self.closing:
            remaining = tuple((getattr(self, name) is not None for name in _SYMMETRIC_FIELDS))
            peers = _collective.exchange(
                remaining,
                group=self._root_group(),
                failure_prefix="MegaMoe symmetric workspace release failed while converging rank results",
            )
            if any((peer != remaining for peer in peers)):
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
        self.route_cache = None
        self.route_header_cache = None
        self.route_cache_probe_after = 0
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
            empty = all(
                (
                    getattr(self, name) is None
                    for name in (*_SYMMETRIC_FIELDS, "gmm_workspace", "swiglu_grad_workspace", "completion_event")
                )
            ) and (not self.replica_runtime_images)
            root_group = self._root_group()
            self.closing = True
        peers = _collective.exchange(
            empty, group=root_group, failure_prefix="MegaMoe workspace close failed while converging rank results"
        )
        if all(peers):
            self._free_local_tensors()
            self.closing = True
            return
        _collective.collective_call(
            lambda: None if empty else torch.npu.synchronize(self.device),
            stage="MegaMoe workspace synchronization",
            group=root_group,
        )
        shmem.host_barrier()
        _collective.collective_call(
            self._free_symmetric_tensors, stage="MegaMoe symmetric workspace release", group=root_group
        )
        shmem.host_barrier()
        _collective.collective_call(self._free_local_tensors, stage="MegaMoe local workspace release", group=root_group)
        self.dtype = None
        self.device = None
        self.completion_event = None
        self.used = False
        self.closing = True
