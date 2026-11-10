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
"""Prepare trusted Router output for managed Torch MegaMoe execution."""

from __future__ import annotations

from array import array
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from itertools import accumulate
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.distributed as dist

from hyper_multicore._loader import load_torch_npu
from hyper_multicore.replica.recompute import get_recompute_state
from hyper_multicore.replica.routing import (
    ReplicaRoute,
    _HostReplicaCounts,
    _hot_replay_readback_enabled,
    _replica_count_payload,
)
from hyper_multicore.torch import ops as multicore_ops

from .spec import MegaMoeSpec, _align_capacity

if TYPE_CHECKING:
    from .workspace import MegaMoeWorkspace


# Avoid large Host transfers and Python boxing when the gathered count table grows.
_HOST_ROUTE_MAX_COUNTS = 2048
# Preserve the smaller descriptor budget when counts change on every call.
_HOST_DESCRIPTOR_MAX_COUNTS = 1024
_ROUTE_CACHE_REPROBE_STEPS = 8
_METADATA_ALIGNMENT = 512
_PULL_STAGING_MAX_BYTES = 64 * 1024 * 1024
ReplicaRouteSelector = Callable[
    [torch.Tensor, list[list[int]], _HostReplicaCounts | None], tuple[MegaMoeSpec, ReplicaRoute] | None
]


@dataclass(frozen=True)
class RouteMetadata:
    """Exact offsets and counts consumed by the low-level native ABI."""

    dispatch_src_off: torch.Tensor
    dispatch_target_off: torch.Tensor
    dispatch_size: torch.Tensor
    combine_src_off: torch.Tensor
    combine_target_off: torch.Tensor
    combine_size: torch.Tensor
    group_list: torch.Tensor
    expert_capacity: int
    local_num_tokens: int | None = None
    plan_tokens: int | None = None
    replica_route: ReplicaRoute | None = None
    empty_owner_experts: tuple[int, ...] | None = None
    maximum_edge_rows: int | None = None


@dataclass(frozen=True)
class PreparedTopKRoute:
    """Permuted payload and exact metadata for one Router decision."""

    routed_tokens: torch.Tensor
    unpermute_mapping: torch.Tensor
    tokens_per_expert: torch.Tensor
    received_counts: torch.Tensor
    metadata: RouteMetadata
    maximum_received_slots: int = 0


@dataclass(frozen=True)
class _HostRouteCache:
    """One validated wire table and immutable route descriptors for a workspace."""

    signature: tuple[Any, ...]
    raw: list[list[int]]
    summary: tuple[int, int, int]
    metadata: RouteMetadata | None
    stream: Any = None
    ready: Any = None
    maximum_edge_rows: int | None = None

    def wait(self, device: Any) -> None:
        """Order a different consumer stream after the descriptor upload.

        Args:
            device: Device whose current stream consumes the cached descriptors.
        """
        if self.ready is not None and self.metadata is not None:
            current = torch.npu.current_stream(device)
            if current != self.stream:
                current.wait_event(self.ready)
                for tensor in vars(self.metadata).values():
                    if isinstance(tensor, torch.Tensor):
                        tensor.record_stream(current)


@dataclass(frozen=True)
class _RouteHeaderCache:
    """One immutable control header, retained across identical local invocations."""

    key: tuple[Any, ...]
    tensor: torch.Tensor | None
    stream: Any = None
    ready: Any = None

    def wait(self, device: Any) -> None:
        """Preserve upload ordering and allocation lifetime on another stream.

        Args:
            device: Device whose current stream consumes the cached header.
        """
        if self.ready is not None and self.tensor is not None:
            current = torch.npu.current_stream(device)
            if current != self.stream:
                current.wait_event(self.ready)
                self.tensor.record_stream(current)


@dataclass
class _RouteEvidence:
    """Scheduling facts from the same validated count exchange as the route."""

    empty_owner_experts: tuple[int, ...] | None = None
    maximum_edge_rows: int | None = None


def _validate_topk_inputs(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Validate route tensor metadata without synchronizing tensor contents."""
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"topk_ids must be int32 or int64, got {topk_ids.dtype}.")
    if not topk_weights.is_floating_point():
        raise TypeError(f"topk_weights must be floating point, got {topk_weights.dtype}.")
    if topk_ids.device != hidden_states.device or topk_weights.device != hidden_states.device:
        raise ValueError("hidden_states, topk_ids and topk_weights must be on one device.")
    return topk_ids.reshape(-1)


def _resolve_counts(
    flat_ids: torch.Tensor,
    supplied_counts: torch.Tensor | None,
    spec: MegaMoeSpec,
) -> torch.Tensor:
    """Use trusted Router counts or compute them only when omitted."""
    if supplied_counts is None:
        return torch.bincount(flat_ids.to(torch.int64), minlength=spec.num_experts).to(torch.int32).contiguous()
    if tuple(supplied_counts.shape) != (spec.num_experts,):
        raise ValueError(
            f"tokens_per_expert must have shape ({spec.num_experts},), " f"got {tuple(supplied_counts.shape)}."
        )
    if supplied_counts.device != flat_ids.device:
        raise ValueError("tokens_per_expert must be on the route tensor device.")
    if supplied_counts.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"tokens_per_expert must be int32 or int64, got {supplied_counts.dtype}.")
    if supplied_counts.dtype == torch.int32 and supplied_counts.is_contiguous():
        return supplied_counts
    return supplied_counts.to(torch.int32).contiguous()


def _start_count_gather(
    counts: torch.Tensor,
    spec: MegaMoeSpec,
) -> tuple[torch.Tensor, Any | None]:
    """Start the rank-major INT32 count gather used by route construction."""
    if spec.ep_size == 1:
        return counts.reshape(1, spec.num_experts), None
    gathered_counts = torch.empty(
        spec.ep_size * spec.num_experts,
        dtype=torch.int32,
        device=counts.device,
    )
    work = dist.all_gather_into_tensor(
        gathered_counts,
        counts,
        group=spec.ep_group,
        async_op=True,
    )
    return gathered_counts.reshape(spec.ep_size, spec.num_experts), work


def _finish_count_gather(
    counts_by_source: torch.Tensor,
    work: Any | None,
    spec: MegaMoeSpec,
) -> torch.Tensor:
    """Wait for gathered counts before reading route metadata."""
    if work is not None:
        work.wait()
    local_start = spec.rank_id * spec.local_experts
    local_end = local_start + spec.local_experts
    return counts_by_source[:, local_start:local_end].contiguous()


def _expert_capacity(
    counts_by_source: torch.Tensor,
    spec: MegaMoeSpec,
) -> tuple[int, int]:
    """Validate the global bound and size rank-local computation tensors."""
    destination_loads = counts_by_source.reshape(
        spec.ep_size,
        spec.ep_size,
        spec.local_experts,
    ).sum(dim=(0, 2), dtype=torch.int32)
    # One host transfer serves both the coordinated overflow check and local
    # allocation; the existing gathered counts require no extra collective.
    loads = destination_loads.tolist()
    # Keep a non-null ABI argument on ranks whose experts receive no tokens.
    # Source outputs and symmetric communication buffers retain their own sizes.
    return max(1, loads[spec.rank_id]), max(loads)


def _permute_topk_input(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand hidden rows with the native NPU MoE permutation operator."""
    if hidden_states.shape[0] == 0:
        return hidden_states, torch.empty(0, dtype=torch.int32, device=hidden_states.device)
    routed_tokens, unpermute_mapping = load_torch_npu().npu_moe_token_permute(
        hidden_states,
        topk_ids,
    )
    return (
        routed_tokens,
        unpermute_mapping.reshape(-1).to(torch.int32).contiguous(),
    )


def _permute_topk_input_out(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the inverse mapping owned while writing rows into leased SHMEM."""
    mapping = torch.empty(topk_ids.numel(), dtype=torch.int32, device=topk_ids.device)
    multicore_ops.moe_token_permute_out(hidden_states.contiguous(), topk_ids.contiguous(), output, mapping)
    return output, mapping


def _compute_route_metadata(
    counts: torch.Tensor,
    counts_by_source: torch.Tensor,
    received_counts: torch.Tensor,
    spec: MegaMoeSpec,
    expert_capacity: int,
) -> RouteMetadata:
    """Build exact element offsets without padding or route truncation."""
    counts_i32 = counts.to(dtype=torch.int32).contiguous()
    source_prefix = counts_by_source.cumsum(dim=1, dtype=torch.int64) - counts_by_source
    destination_layout = (
        counts_by_source.reshape(
            spec.ep_size,
            spec.ep_size,
            spec.local_experts,
        )
        .permute(1, 2, 0)
        .contiguous()
    )
    destination_flat = destination_layout.reshape(spec.ep_size, -1)
    destination_prefix = (destination_flat.cumsum(dim=1, dtype=torch.int64) - destination_flat).reshape(
        spec.ep_size, spec.local_experts, spec.ep_size
    )
    local_destination_prefix = destination_prefix[spec.rank_id]
    local_destination_counts = destination_layout[spec.rank_id]
    local_start = spec.rank_id * spec.local_experts
    local_end = local_start + spec.local_experts
    return RouteMetadata(
        dispatch_src_off=(
            source_prefix[:, local_start:local_end].reshape(-1).to(torch.int64)
            if spec.dispatch_mode == "pull"
            else source_prefix[spec.rank_id].to(torch.int64)
        ),
        dispatch_target_off=(
            local_destination_prefix.T.reshape(-1).to(torch.int64)
            if spec.dispatch_mode == "pull"
            else destination_prefix[:, :, spec.rank_id].reshape(-1).to(torch.int64)
        ),
        dispatch_size=(received_counts.reshape(-1).to(torch.int32) if spec.dispatch_mode == "pull" else counts_i32),
        combine_src_off=local_destination_prefix.T.reshape(-1).to(torch.int64),
        combine_target_off=(source_prefix[:, local_start:local_end].reshape(-1).to(torch.int64)),
        combine_size=received_counts.reshape(-1).to(torch.int32),
        group_list=(local_destination_prefix[:, -1] + local_destination_counts[:, -1]).to(torch.int64),
        expert_capacity=expert_capacity,
    )


def _upload_route_tensor(tensor: torch.Tensor, device: Any) -> torch.Tensor:
    """Keep Host metadata alive through the pinned allocator during asynchronous upload."""
    if torch.device(device).type == "npu":
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)


def _host_route_metadata(
    raw: list[list[int]],
    spec: MegaMoeSpec,
    capacity: int,
    device: Any,
) -> RouteMetadata:
    """Pack the native INT64 offsets and INT32 route sizes in one retained allocation."""
    counts = [row[3:] for row in raw]
    local, rank = spec.local_experts, spec.rank_id
    start = rank * local
    prefix, destinations, destination_prefix = _host_route_prefixes(counts, spec)
    source_offsets = [
        prefix[source][expert] for source in range(spec.ep_size) for expert in range(start, start + local)
    ]
    target_offsets = [
        destination_prefix[rank][expert * spec.ep_size + source]
        for source in range(spec.ep_size)
        for expert in range(local)
    ]
    received = [counts[source][expert] for source in range(spec.ep_size) for expert in range(start, start + local)]
    if spec.dispatch_mode == "pull":
        dispatch_source, dispatch_target, dispatch_sizes = source_offsets, target_offsets, received
    else:
        dispatch_source = prefix[rank]
        dispatch_target = [
            destination_prefix[target][expert * spec.ep_size + rank]
            for target in range(spec.ep_size)
            for expert in range(local)
        ]
        dispatch_sizes = counts[rank]
    groups = [
        destination_prefix[rank][(expert + 1) * spec.ep_size - 1] + destinations[rank][(expert + 1) * spec.ep_size - 1]
        for expert in range(local)
    ]
    wide = (dispatch_source, dispatch_target, target_offsets, source_offsets, groups)
    return _pack_host_route(wide, (dispatch_sizes, received), capacity, device)


def _host_route_prefixes(
    counts: list[list[int]],
    spec: MegaMoeSpec,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Scan each source and each destination independently in native layout order."""
    local = spec.local_experts
    prefix = [list(accumulate(row, initial=0))[:-1] for row in counts]
    destinations = [
        [
            counts[source][expert]
            for expert in range(target * local, (target + 1) * local)
            for source in range(spec.ep_size)
        ]
        for target in range(spec.ep_size)
    ]
    destination_prefix = [list(accumulate(row, initial=0))[:-1] for row in destinations]
    return prefix, destinations, destination_prefix


def _pack_host_route(
    wide: tuple[list[int] | np.ndarray, ...],
    size_fields: tuple[list[int] | np.ndarray, ...],
    capacity: int,
    device: Any,
) -> RouteMetadata:
    """Transfer aligned descriptor fields together and retain storage through their views."""
    fields = [*wide, *size_fields]
    dtypes = [torch.int64] * len(wide) + [torch.int32] * 2
    storage = bytearray()
    slices = []
    for values, dtype in zip(fields, dtypes):
        if isinstance(values, np.ndarray):
            encoded = values.astype(np.int64 if dtype == torch.int64 else np.int32, copy=False).tobytes()
        else:
            encoded = array("q" if dtype == torch.int64 else "i", values).tobytes()
        start = len(storage)
        storage.extend(encoded)
        slices.append((start, len(storage), dtype))
        storage.extend(bytes(-len(storage) % _METADATA_ALIGNMENT))
    buffer = _upload_route_tensor(torch.frombuffer(storage, dtype=torch.uint8), device)
    tensors = [buffer[start:end].view(dtype) for start, end, dtype in slices]
    offsets, sizes = (tensors[:5], tensors[5:])
    return RouteMetadata(
        dispatch_src_off=offsets[0],
        dispatch_target_off=offsets[1],
        dispatch_size=sizes[0],
        combine_src_off=offsets[2],
        combine_target_off=offsets[3],
        combine_size=sizes[1],
        group_list=offsets[4],
        expert_capacity=capacity,
    )


def prepare_topk_route(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    spec: MegaMoeSpec,
    tokens_per_expert: torch.Tensor | None,
    workspace: MegaMoeWorkspace | None = None,
    *,
    autograd_mask: int = 0,
    prepare_workspace: Callable[[int, int], None] | None = None,
    replica_route: ReplicaRoute | None = None,
    _select_replica_route: ReplicaRouteSelector | None = None,
) -> PreparedTopKRoute:
    """Overlap Top-K permutation with count exchange and build route metadata.

    Args:
        hidden_states: Flattened token states to route.
        topk_ids: Global experts selected for every token.
        topk_weights: Router weights paired with ``topk_ids``.
        spec: Bound shape and expert-parallel specification.
        tokens_per_expert: Optional trusted Router histogram.
        workspace: Optional workspace ordered before the count exchange. Pull
            requires a lease acquired by the caller or the automatic-capacity callback.
        autograd_mask: Group-consistent gradient participation flags in dynamic mode.
        prepare_workspace: Automatic-capacity callback, run after count agreement and before pull permutation.
        replica_route: Previously agreed replica route, or None for owner routing.
        _select_replica_route: Internal policy invoked after exact group count validation.

    Returns:
        Permuted tokens and exact native route metadata.
    """
    flat_ids = _validate_topk_inputs(hidden_states, topk_ids, topk_weights)
    if spec.dynamic_tokens and (replica_route is not None or _select_replica_route is not None):
        return _prepare_replica_dynamic_route(
            hidden_states,
            topk_ids,
            spec,
            tokens_per_expert,
            workspace,
            autograd_mask,
            replica_route,
            prepare_workspace,
            _select_replica_route,
        )
    if spec.dynamic_tokens:
        return _prepare_dynamic_route(
            hidden_states, topk_ids, flat_ids, spec, tokens_per_expert, workspace, autograd_mask, prepare_workspace
        )
    counts = _resolve_counts(flat_ids, tokens_per_expert, spec)
    permutation_output = None
    if workspace is not None:
        if spec.dispatch_mode == "pull":
            if not workspace.in_use or workspace.source_buffer is None:
                raise RuntimeError("Pull permutation requires an initialized, claimed workspace.")
            permutation_output = workspace.source_buffer
        else:
            workspace.wait_for_reuse()
    counts_by_source, count_work = _start_count_gather(counts, spec)
    if permutation_output is None:
        routed_tokens, unpermute_mapping = _permute_topk_input(hidden_states, topk_ids)
    else:
        routed_tokens, unpermute_mapping = _permute_topk_input_out(hidden_states, topk_ids, permutation_output)
    received_counts = _finish_count_gather(counts_by_source, count_work, spec)
    expert_capacity, maximum_received_slots = _expert_capacity(counts_by_source, spec)
    return PreparedTopKRoute(
        routed_tokens=routed_tokens,
        unpermute_mapping=unpermute_mapping,
        tokens_per_expert=counts,
        received_counts=received_counts,
        maximum_received_slots=maximum_received_slots,
        metadata=_compute_route_metadata(
            counts,
            counts_by_source,
            received_counts,
            spec,
            expert_capacity,
        ),
    )


def _valid_route_counts(row: list[int], top_k: int) -> bool:
    """Validate one source histogram against its actual distinct Top-K rows."""
    return row[3] == row[1] * top_k and row[4] >= 0 and row[5] <= row[1]


def _validate_dynamic_sizes(rows: list[list[int]], spec: MegaMoeSpec) -> None:
    """Keep native size and legacy bound rejection ahead of route publication."""
    flexible = spec.auto_tokens
    if flexible and any(row[0] or row[1] < 0 for row in rows):
        raise RuntimeError("MegaMoe dynamic route exceeds native INT32 size limits")
    if not flexible and any(row[0] or not 0 <= row[1] <= spec.max_local_num_tokens for row in rows):
        raise RuntimeError(
            f"MegaMoe token capacity overflow: limit={spec.max_local_num_tokens}, " f"actual={[row[1] for row in rows]}"
        )


def _validate_dynamic_rows(rows: list[list[int]], spec: MegaMoeSpec) -> None:
    """Validate every rank before any automatic allocation or SHMEM write."""
    _validate_dynamic_sizes(rows, spec)
    if any(row[2] != rows[0][2] for row in rows):
        raise RuntimeError("MegaMoe autograd participation differs across EP ranks")
    if any(not _valid_route_counts(row, spec.top_k) for row in rows):
        raise RuntimeError("MegaMoe counts must describe the actual distinct Top-K routes")


def _owner_empty_experts(
    spec: MegaMoeSpec, agreement: list[list[int]] | None, host_rows: list[list[int]] | None
) -> tuple[int, ...] | None:
    """Elide full initialization only when every owner gradient is overwritten."""
    if not spec._selective_owner_gradient_init or spec.replica_slots_per_rank:
        return None
    if agreement is not None and any(row[4] > 0 for row in agreement):
        # One source covers every logical expert, so every group has positive K.
        return ()
    if host_rows is None:
        return None
    first = 3 + spec.rank_id * spec.local_experts
    if all(any(row[first + expert] for row in host_rows) for expert in range(spec.local_experts)):
        return ()
    return None


def _maximum_edge_rows(spec: MegaMoeSpec, rows: Sequence[Sequence[int]]) -> int | None:
    """Bound physical communication by the largest agreed logical source/expert count."""
    if not spec.auto_tokens or not spec._compact_communication:
        return None
    # A replica can split an expert's source count across slots, but cannot merge
    # the counts of different logical experts into the same slot.
    return max(row[5] for row in rows)


def _record_route_evidence(
    evidence: _RouteEvidence | None, spec: MegaMoeSpec, rows: list[list[int]], raw: list[list[int]] | None = None
) -> None:
    """Retain scheduling and owner-coverage evidence without reading counts again."""
    if evidence is not None:
        evidence.empty_owner_experts = _owner_empty_experts(spec, rows if raw is None else None, raw)
        evidence.maximum_edge_rows = _maximum_edge_rows(spec, rows)


def _dynamic_summary(
    payload: torch.Tensor,
    spec: MegaMoeSpec,
    evidence: _RouteEvidence | None = None,
) -> tuple[int, int, int]:
    """Use one Host summary for shape, participation, receive limits, and schedule."""
    rows = _device_route_rows(payload, spec)
    summary = _summarize_dynamic_rows(rows, spec)
    _record_route_evidence(evidence, spec, rows)
    return summary


def _summarize_dynamic_rows(
    rows: list[list[int]], spec: MegaMoeSpec, *, defer_receive_limit: bool = False
) -> tuple[int, int, int]:
    """Keep overflow and participation checks common to both metadata policies."""
    _validate_dynamic_rows(rows, spec)
    maximum = max(row[6] for row in rows)
    maximum_tokens = max(row[1] for row in rows) if spec.auto_tokens else spec.local_num_tokens
    maximum_capacity = _align_capacity(spec.ep_size * maximum_tokens * spec.top_k)
    if maximum > maximum_capacity:
        raise RuntimeError(
            f"MegaMoe receive capacity overflow: configured_capacity={maximum_capacity}, actual_maximum={maximum}"
        )
    plan_tokens = max(128, _align_capacity(max(row[1] for row in rows)))
    if plan_tokens * max(spec.hidden_size, spec.top_k) > 2**31 - 1 or (maximum > 2**31 - 1 and not defer_receive_limit):
        raise RuntimeError("MegaMoe dynamic route exceeds native INT32 size limits")
    return max(1, rows[spec.rank_id][6]), plan_tokens, maximum


def _read_dynamic_route(
    payload: torch.Tensor,
    spec: MegaMoeSpec,
    evidence: _RouteEvidence | None = None,
) -> tuple[tuple[int, int, int], list[list[int]] | None]:
    """Reuse the required readback for small descriptors; reduce large tables on device."""
    if spec.ep_size * spec.num_experts > _HOST_ROUTE_MAX_COUNTS:
        return _dynamic_summary(payload, spec, evidence), None
    raw = payload.tolist()
    return _summarize_host_route(raw, spec, evidence), raw


def _summarize_host_route(
    raw: list[list[int]], spec: MegaMoeSpec, evidence: _RouteEvidence | None = None
) -> tuple[int, int, int]:
    """Validate a bounded Host table before allocating or publishing source rows."""
    counts = [row[3:] for row in raw]
    local = spec.local_experts
    loads = [
        sum(sum(values[start : start + local]) for values in counts) for start in range(0, spec.num_experts, local)
    ]
    rows = [row[:3] + [sum(values), min(values), max(values), load] for row, values, load in zip(raw, counts, loads)]
    summary = _summarize_dynamic_rows(rows, spec)
    _record_route_evidence(evidence, spec, rows, raw)
    return summary


def _route_cache_signature(spec: MegaMoeSpec, device: Any) -> tuple[Any, ...]:
    """Keep descriptor reuse specific to the route topology and size contract."""
    return (
        spec.ep_size,
        id(spec.ep_group),
        spec.rank_id,
        spec.num_experts,
        spec.top_k,
        spec.hidden_size,
        spec.dispatch_mode,
        spec.auto_tokens,
        spec.max_local_num_tokens,
        spec._compact_communication,
        torch.device(device),
    )


def _read_cached_route(
    payload: torch.Tensor,
    spec: MegaMoeSpec,
    workspace: MegaMoeWorkspace | None,
    device: Any,
    evidence: _RouteEvidence | None = None,
) -> tuple[tuple[int, int, int], list[list[int]] | None, RouteMetadata | None]:
    """Reuse only an identical, previously validated all-rank wire table."""
    if workspace is None or not spec.auto_tokens or spec.ep_size * spec.num_experts > _HOST_ROUTE_MAX_COUNTS:
        summary, raw = _read_dynamic_route(payload, spec, evidence)
        return summary, raw, None
    cached = getattr(workspace, "route_cache", None)
    same_topology = isinstance(cached, _HostRouteCache) and cached.signature == _route_cache_signature(spec, device)
    remaining = getattr(workspace, "route_cache_probe_after", 0)
    if same_topology and remaining > 0:
        workspace.route_cache_probe_after = remaining - 1
        return _dynamic_summary(payload, spec, evidence), None, None
    workspace.route_cache_probe_after = 0
    raw = payload.tolist()
    if same_topology and cached.raw == raw:
        if evidence is not None:
            evidence.empty_owner_experts = _owner_empty_experts(spec, None, raw)
            evidence.maximum_edge_rows = cached.maximum_edge_rows
        cached.wait(device)
        return cached.summary, raw, cached.metadata
    summary = _summarize_host_route(raw, spec, evidence)
    if same_topology and spec.ep_size * spec.num_experts > _HOST_DESCRIPTOR_MAX_COUNTS:
        workspace.route_cache_probe_after = _ROUTE_CACHE_REPROBE_STEPS
    return summary, raw, None


def _cache_host_route(
    workspace: MegaMoeWorkspace | None,
    spec: MegaMoeSpec,
    raw: list[list[int]],
    summary: tuple[int, int, int],
    metadata: RouteMetadata,
    device: Any,
    *,
    maximum_edge_rows: int | None = None,
) -> None:
    """Retain descriptors only after confirming a repeated validated wire table."""
    if workspace is None or not spec.auto_tokens:
        return
    signature = _route_cache_signature(spec, device)
    previous = getattr(workspace, "route_cache", None)
    repeated = isinstance(previous, _HostRouteCache) and previous.signature == signature and previous.raw == raw
    retained = metadata if repeated else None
    stream, ready = None, None
    if repeated and torch.device(device).type == "npu":
        stream = torch.npu.current_stream(device)
        ready = torch.npu.Event()
        ready.record(stream)
    workspace.route_cache = _HostRouteCache(signature, raw, summary, retained, stream, ready, maximum_edge_rows)


def _dynamic_route_metadata(
    counts: torch.Tensor,
    gathered: torch.Tensor,
    host_rows: list[list[int]] | None,
    spec: MegaMoeSpec,
    capacity: int,
) -> RouteMetadata:
    """Keep wider cache misses on device instead of rebuilding large Host descriptors."""
    if host_rows is not None and spec.ep_size * spec.num_experts <= _HOST_DESCRIPTOR_MAX_COUNTS:
        return _host_route_metadata(host_rows, spec, capacity, counts.device)
    counts_by_source = gathered[:, 3:].contiguous()
    start = spec.rank_id * spec.local_experts
    received_counts = counts_by_source[:, start : start + spec.local_experts].contiguous()
    # Retained dispatch sizes must not alias a caller-owned Router count buffer.
    local_counts = counts_by_source[spec.rank_id] if spec.auto_tokens else counts
    return _compute_route_metadata(local_counts, counts_by_source, received_counts, spec, capacity)


def _permute_dynamic_pull(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    spec: MegaMoeSpec,
    workspace: MegaMoeWorkspace | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take the source view only after automatic growth and lease acquisition."""
    if workspace is None or not workspace.in_use or workspace.source_buffer is None:
        raise RuntimeError("Pull permutation requires an initialized, claimed workspace.")
    tokens = hidden_states.shape[0]
    output = workspace.source_buffer[: tokens * spec.top_k]
    return _permute_topk_input_out(hidden_states, topk_ids, output) if tokens else (output, topk_ids.reshape(-1))


def _can_stage_pull(hidden_states: torch.Tensor, spec: MegaMoeSpec) -> bool:
    """Bound ordinary-memory permutation staging while capacity agreement is pending."""
    return (
        spec.auto_tokens
        and spec.dispatch_mode == "pull"
        and hidden_states.device.type == "npu"
        and hidden_states.numel() * spec.top_k * hidden_states.element_size() <= _PULL_STAGING_MAX_BYTES
    )


def _stage_dynamic_permutation(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    spec: MegaMoeSpec,
    invalid: bool,
    workspace: MegaMoeWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Stage only valid automatic routes whose ordinary storage survives heap growth."""
    if invalid or not spec.auto_tokens:
        return None
    if (
        spec.dispatch_mode == "push"
        or _can_stage_pull(hidden_states, spec)
        and not getattr(workspace, "route_cache_probe_after", 0)
    ):
        return _permute_topk_input(hidden_states, topk_ids)
    return None


def _publish_staged_pull(
    permuted: tuple[torch.Tensor, torch.Tensor],
    hidden_states: torch.Tensor,
    spec: MegaMoeSpec,
    workspace: MegaMoeWorkspace | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Publish staged rows only after every rank validated and committed its workspace."""
    if workspace is None or not workspace.in_use or workspace.source_buffer is None:
        raise RuntimeError("Pull publication requires an initialized, claimed workspace.")
    output = workspace.source_buffer[: hidden_states.shape[0] * spec.top_k]
    output.copy_(permuted[0])
    return output, permuted[1]


def _dynamic_header(
    values: tuple[int, int, int],
    dtype: torch.dtype,
    device: Any,
    workspace: MegaMoeWorkspace | None,
) -> torch.Tensor:
    """Avoid repeated Host allocation and upload of an unchanged control header."""
    key = (values, dtype, torch.device(device))
    cached = getattr(workspace, "route_header_cache", None)
    repeated = isinstance(cached, _RouteHeaderCache) and cached.key == key
    if repeated and cached.tensor is not None:
        cached.wait(device)
        return cached.tensor
    header = _upload_route_tensor(torch.tensor(values, dtype=dtype), device)
    if workspace is not None:
        stream, ready = None, None
        if repeated and torch.device(device).type == "npu":
            stream = torch.npu.current_stream(device)
            ready = torch.npu.Event()
            ready.record(stream)
        workspace.route_header_cache = _RouteHeaderCache(key, header if repeated else None, stream, ready)
    return header


def _dynamic_route_payload(
    hidden_states: torch.Tensor,
    flat_ids: torch.Tensor,
    spec: MegaMoeSpec,
    supplied_counts: torch.Tensor | None,
    autograd_mask: int,
    workspace: MegaMoeWorkspace | None = None,
    *,
    validate_replica_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Pack one group-consistent wire format, including coordinated overflow flags."""
    tokens = hidden_states.shape[0]
    flexible = spec.auto_tokens
    invalid = tokens * max(spec.hidden_size, spec.top_k) > 2**31 - 1 if flexible else tokens > spec.max_local_num_tokens
    counts = (
        torch.zeros(spec.num_experts, dtype=torch.int32, device=hidden_states.device)
        if invalid
        else (
            torch.empty(spec.num_experts, dtype=torch.int32, device=flat_ids.device)
            if validate_replica_ids and supplied_counts is None
            else _resolve_counts(flat_ids, supplied_counts, spec)
        )
    )
    if validate_replica_ids and supplied_counts is None and not invalid:
        checked = _replica_count_payload(flat_ids.reshape(tokens, spec.top_k).to(torch.int64), spec.num_experts)
        counts = checked[: spec.num_experts].to(torch.int32)
        counts[:1].masked_fill_(checked[-1:].bool(), -1)
    payload_dtype = torch.int32 if flexible else torch.int64
    # An invalid source must still join the same collective before all ranks reject it.
    header_tokens = min(tokens, 2**31 - 1) if flexible else tokens
    header = _dynamic_header(
        (int(invalid), header_tokens, autograd_mask),
        payload_dtype,
        hidden_states.device,
        workspace if flexible else None,
    )
    payload = torch.cat((header, counts.to(payload_dtype)))
    return counts, payload, invalid


def _start_dynamic_gather(payload: torch.Tensor, spec: MegaMoeSpec) -> tuple[torch.Tensor, Any | None]:
    """Match receive storage to the dynamic count wire dtype."""
    if spec.ep_size == 1:
        gathered, work = payload.unsqueeze(0), None
    else:
        gathered = torch.empty((spec.ep_size, payload.numel()), dtype=payload.dtype, device=payload.device)
        work = dist.all_gather_into_tensor(gathered.reshape(-1), payload, group=spec.ep_group, async_op=True)
    return gathered, work


def _prepare_dynamic_route(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    flat_ids: torch.Tensor,
    spec: MegaMoeSpec,
    supplied_counts: torch.Tensor | None,
    workspace: MegaMoeWorkspace | None,
    autograd_mask: int,
    prepare_workspace: Callable[[int, int], None] | None = None,
) -> PreparedTopKRoute:
    """Exchange variable-length metadata without adding a token-count collective."""
    tokens = hidden_states.shape[0]
    counts, payload, invalid = _dynamic_route_payload(
        hidden_states, flat_ids, spec, supplied_counts, autograd_mask, workspace
    )
    if workspace is not None and spec.dispatch_mode != "pull":
        workspace.wait_for_reuse()
    gathered, work = _start_dynamic_gather(payload, spec)
    # Ordinary storage remains valid if count agreement triggers a SHMEM restart.
    permuted = _stage_dynamic_permutation(hidden_states, topk_ids, spec, invalid, workspace)
    summary, host_rows, metadata = None, None, None
    evidence = _RouteEvidence()
    if spec.auto_tokens:
        if work is not None:
            work.wait()
            work = None
        summary, host_rows, metadata = _read_cached_route(
            gathered, spec, workspace, hidden_states.device, evidence
        )
        if prepare_workspace is None:
            raise RuntimeError("Automatic MegaMoe routes require workspace capacity preparation")
        prepare_workspace(summary[1], summary[2])
    if invalid:
        permuted = None
    elif spec.dispatch_mode == "pull":
        permuted = (
            _permute_dynamic_pull(hidden_states, topk_ids, spec, workspace)
            if permuted is None
            else _publish_staged_pull(permuted, hidden_states, spec, workspace)
        )
    elif permuted is None:
        permuted = _permute_topk_input(hidden_states, topk_ids)
    if work is not None:
        work.wait()
    if summary is None:
        summary, host_rows = _read_dynamic_route(gathered, spec, evidence)
    capacity, plan_tokens, maximum = summary
    if metadata is None:
        metadata = _dynamic_route_metadata(counts, gathered, host_rows, spec, capacity)
        if host_rows is not None:
            _cache_host_route(
                workspace, spec, host_rows, summary, metadata, hidden_states.device,
                maximum_edge_rows=evidence.maximum_edge_rows,
            )
    received_counts = metadata.combine_size.reshape(spec.ep_size, spec.local_experts)
    return PreparedTopKRoute(
        routed_tokens=permuted[0],
        unpermute_mapping=permuted[1],
        tokens_per_expert=counts,
        received_counts=received_counts,
        maximum_received_slots=maximum,
        metadata=replace(
            metadata,
            local_num_tokens=tokens,
            plan_tokens=plan_tokens,
            empty_owner_experts=evidence.empty_owner_experts,
            maximum_edge_rows=evidence.maximum_edge_rows,
        ),
    )


def restore_topk_output(
    expert_output: torch.Tensor,
    unpermute_mapping: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Restore token order and apply Router weights with the native NPU op.

    Args:
        expert_output: Expert outputs in the expanded routing order.
        unpermute_mapping: Inverse mapping retained by the source permutation.
        topk_weights: Router weights for the original token rows.

    Returns:
        Weighted expert outputs in the original token order.
    """
    if topk_weights.shape[0] == 0:
        # Keep both the expert backward and the empty routing-weight gradient connected.
        return expert_output[:0] + topk_weights.sum().to(expert_output.dtype)
    return load_torch_npu().npu_moe_token_unpermute(
        expert_output,
        unpermute_mapping,
        probs=topk_weights.float().contiguous(),
    )


def _numpy_route_metadata(counts: np.ndarray, spec: MegaMoeSpec, capacity: int, device: Any) -> RouteMetadata:
    """Pack rank-local descriptor fields without boxing each count or offset."""
    local, rank = (spec.local_experts, spec.rank_id)
    start = rank * local
    received = counts[:, start : start + local]
    expert_major = received.T.reshape(-1)
    local_prefix = expert_major.cumsum(dtype=np.int64)
    groups = local_prefix[spec.ep_size - 1 :: spec.ep_size]
    combine_src = (local_prefix - expert_major).reshape(local, spec.ep_size).T.reshape(-1)
    preceding_destinations = counts[:, :start].sum(axis=1, dtype=np.int64)
    combine_dst = (preceding_destinations[:, None] + received.cumsum(axis=1, dtype=np.int64) - received).reshape(-1)
    combine_size = received.reshape(-1)
    if spec.dispatch_mode == "pull":
        dispatch_src, dispatch_dst, dispatch_size = (combine_dst, combine_src, combine_size)
    else:
        dispatch_size = counts[rank]
        dispatch_src = dispatch_size.cumsum(dtype=np.int64) - dispatch_size
        preceding_sources = counts[:rank].sum(axis=0, dtype=np.int64)
        destination_totals = (preceding_sources + counts[rank:].sum(axis=0, dtype=np.int64)).reshape(-1, local)
        prior_experts = destination_totals.cumsum(axis=1, dtype=np.int64) - destination_totals
        dispatch_dst = preceding_sources + prior_experts.reshape(-1)
    wide = (dispatch_src, dispatch_dst, combine_src, combine_dst, groups)
    return _pack_host_route(wide, (dispatch_size, combine_size), capacity, device)


def _device_route_rows(payload: torch.Tensor, spec: MegaMoeSpec) -> list[list[int]]:
    """Read the compact agreement once for validation and replica admission."""
    counts = payload[:, 3:]
    loads = counts.reshape(spec.ep_size, spec.ep_size, spec.local_experts).sum(dim=(0, 2))
    summary = torch.cat(
        (
            payload[:, :3],
            counts.sum(1, keepdim=True),
            counts.amin(1, keepdim=True),
            counts.amax(1, keepdim=True),
            loads[:, None],
        ),
        dim=1,
    )
    return summary.tolist()


def _read_replica_dynamic_route(
    payload: torch.Tensor, spec: MegaMoeSpec, *, defer_receive_limit: bool = False
) -> tuple[tuple[int, int, int], list[list[int]] | None, list[list[int]]]:
    """Reuse the required readback for small descriptors; reduce large tables on device."""
    if spec.ep_size * spec.num_experts > _HOST_ROUTE_MAX_COUNTS:
        rows = _device_route_rows(payload, spec)
        return (_summarize_dynamic_rows(rows, spec, defer_receive_limit=defer_receive_limit), None, rows)
    raw = payload.tolist()
    counts = [row[3:] for row in raw]
    local = spec.local_experts
    loads = [
        sum((sum(values[start : start + local]) for values in counts)) for start in range(0, spec.num_experts, local)
    ]
    rows = [row[:3] + [sum(values), min(values), max(values), load] for row, values, load in zip(raw, counts, loads)]
    return (_summarize_dynamic_rows(rows, spec, defer_receive_limit=defer_receive_limit), raw, rows)


def _read_hot_replay_route(
    payload: torch.Tensor, spec: MegaMoeSpec
) -> tuple[tuple[int, int, int], np.ndarray, list[list[int]]]:
    """Read fresh counts once for both validation and the exact checkpoint guard."""
    host = payload.cpu().numpy()
    counts = host[:, 3:]
    loads = counts.reshape(spec.ep_size, spec.ep_size, spec.local_experts).sum(axis=(0, 2), dtype=np.int64)
    rows = np.column_stack(
        (host[:, :3], counts.sum(axis=1, dtype=np.int64), counts.min(axis=1), counts.max(axis=1), loads)
    ).tolist()
    return _summarize_dynamic_rows(rows, spec, defer_receive_limit=True), counts, rows


def _replica_readback_history(spec: MegaMoeSpec) -> tuple[deque | None, bool]:
    """Record each checkpoint route, including home decisions, in invocation order."""
    if spec.ep_size * spec.num_experts <= _HOST_ROUTE_MAX_COUNTS or not _hot_replay_readback_enabled():
        return None, False
    state = get_recompute_state()
    if state is None:
        return None, False
    return state.get_resource(_replica_readback_history, deque), state.is_recomputing


def _agree_replica_dynamic_route(
    gathered: torch.Tensor, spec: MegaMoeSpec, select: ReplicaRouteSelector
) -> tuple[tuple[int, int, int], list[list[int]] | None, list[list[int]], tuple[MegaMoeSpec, ReplicaRoute] | None]:
    """Use the optional hot replay readback without changing current admission."""
    history, replaying = _replica_readback_history(spec)
    hot = False
    if replaying and history:
        saved_spec, used_replicas = history.popleft()
        hot = saved_spec is spec and used_replicas
    if hot:
        summary, host_counts, agreement = _read_hot_replay_route(gathered, spec)
        raw = None
    else:
        summary, raw, agreement = _read_replica_dynamic_route(gathered, spec, defer_receive_limit=True)
        host_counts = None if raw is None else [row[3:] for row in raw]
    proposal = select(gathered[:, 3:], agreement, host_counts)
    if history is not None and not replaying:
        history.append((spec, proposal is not None and not proposal[1].home_only))
    return summary, raw, agreement, proposal


@dataclass(frozen=True)
class _ReplicaAgreement:
    """Validated source agreement and the selected current physical route."""

    spec: MegaMoeSpec
    ids: torch.Tensor
    counts: torch.Tensor
    table: torch.Tensor
    summary: tuple[int, int, int]
    raw: list[list[int]] | None = None
    route: ReplicaRoute | None = None
    source_summary: list[list[int]] | None = None
    maximum_edge_rows: int | None = None


def _physical_agreement(spec: MegaMoeSpec, ids: torch.Tensor, route: ReplicaRoute) -> _ReplicaAgreement:
    """Reuse the same count exchange after selecting physical guest slots."""
    if route.dynamic_summary is None:
        raise ValueError("A physical replica route requires the current source controls")
    summary = _summarize_dynamic_rows(route.dynamic_summary, spec)
    return _ReplicaAgreement(
        spec, ids, route.counts_by_source[spec.rank_id], route.counts_by_source, summary, route=route,
        maximum_edge_rows=_maximum_edge_rows(spec, route.dynamic_summary),
    )


def _replica_agreement(hidden_states, ids, spec, supplied_counts, workspace, mask, select) -> _ReplicaAgreement:
    """Validate all ranks before choosing weights, allocating a Heap, or publishing rows."""
    if workspace is not None and spec.dispatch_mode != "pull":
        workspace.wait_for_reuse()
    counts, payload, _ = _dynamic_route_payload(
        hidden_states, ids.reshape(-1), spec, supplied_counts, mask, validate_replica_ids=select is not None
    )
    gathered, work = _start_dynamic_gather(payload, spec)
    if work is not None:
        work.wait()
    if select is None:
        summary, raw, agreement = _read_replica_dynamic_route(gathered, spec, defer_receive_limit=True)
    else:
        summary, raw, agreement, proposal = _agree_replica_dynamic_route(gathered, spec, select)
        if proposal is not None and not proposal[1].home_only:
            selected, route = proposal
            return _physical_agreement(selected, route.physical_ids, route)
    if summary[2] > 2**31 - 1:
        raise RuntimeError("MegaMoe selected home route exceeds native INT32 receive limits")
    return _ReplicaAgreement(
        spec, ids, counts, gathered[:, 3:].contiguous(), summary, raw, source_summary=agreement,
        maximum_edge_rows=_maximum_edge_rows(spec, agreement),
    )


def _replica_metadata(agreed: _ReplicaAgreement, capacity: int, device: Any) -> RouteMetadata:
    """Pack exact physical counts once, preserving owned metadata and native strides."""
    route, spec = agreed.route, agreed.spec
    if spec._host_replica_metadata and route is not None and route._host_counts is not None:
        table = np.frombuffer(route.plan._dispatch_counts_bytes, dtype="<i4")
        return _numpy_route_metadata(table.reshape(spec.ep_size, spec.num_experts), spec, capacity, device)
    if agreed.raw is not None and spec.ep_size * spec.num_experts <= _HOST_DESCRIPTOR_MAX_COUNTS:
        return _host_route_metadata(agreed.raw, spec, capacity, device)
    start = spec.rank_id * spec.local_experts
    received = agreed.table[:, start : start + spec.local_experts].contiguous()
    return _compute_route_metadata(agreed.counts, agreed.table, received, spec, capacity)


def _prepare_replica_dynamic_route(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    spec: MegaMoeSpec,
    supplied_counts: torch.Tensor | None,
    workspace: MegaMoeWorkspace | None,
    autograd_mask: int,
    replica_route: ReplicaRoute | None = None,
    prepare_workspace: Callable[[int, int], None] | None = None,
    select_replica_route: ReplicaRouteSelector | None = None,
) -> PreparedTopKRoute:
    """Prepare an automatic replica route with one validated group count exchange."""
    if not spec.auto_tokens or prepare_workspace is None:
        raise ValueError("Replica execution requires automatic token capacity preparation")
    if replica_route is None:
        agreed = _replica_agreement(
            hidden_states, topk_ids, spec, supplied_counts, workspace, autograd_mask, select_replica_route
        )
    else:
        agreed = _physical_agreement(spec, topk_ids, replica_route)
    capacity, plan_tokens, maximum = agreed.summary
    prepare_workspace(plan_tokens, maximum)
    permuted = (
        _permute_dynamic_pull(hidden_states, agreed.ids, agreed.spec, workspace)
        if agreed.spec.dispatch_mode == "pull"
        else _permute_topk_input(hidden_states, agreed.ids)
    )
    metadata = _replica_metadata(agreed, capacity, hidden_states.device)
    route = agreed.route
    return PreparedTopKRoute(
        routed_tokens=permuted[0],
        unpermute_mapping=permuted[1],
        tokens_per_expert=agreed.counts,
        received_counts=metadata.combine_size.reshape(spec.ep_size, agreed.spec.local_experts),
        maximum_received_slots=maximum,
        metadata=replace(
            metadata,
            local_num_tokens=hidden_states.shape[0],
            plan_tokens=plan_tokens,
            replica_route=None if route is not None and route.home_only else route,
            empty_owner_experts=_owner_empty_experts(agreed.spec, agreed.source_summary, agreed.raw),
            maximum_edge_rows=agreed.maximum_edge_rows,
        ),
    )
