# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind aligned Ascend950 routes to the shared MegaMoe MXFP8 executor."""

from collections import OrderedDict
from dataclasses import dataclass, replace

import numpy as np
import torch

from ..mega_moe import route
from ..mega_moe.heap_manager import _HeapCapacityError
from ._planner.aligned import materialize_aligned_moves
from ._planner.capacity import ExpertReplicaConfig
from ._planner.receive import SourceQuota, receive_limit, select_replica_plan
from .mega_moe_adapter import _get_mega_bank, _release_mega_bank
from .mx_gradient import MXGradientPool
from .mx_prefetch import MXWeightPrefetch
from .placement import _LoweredPlacement
from .planner import HotReplicaPlanner, replica_slot_budget


_TRAINING_MASK = 1 << 7
_SNAPSHOT_MAX_BYTES = 256 * 1024
_GEOMETRY_CACHE_ENTRIES = 8
_GEOMETRY_CACHE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class MXBindings:
    """Bind the invocation's exact quotas to serial weight and FP32 return storage."""

    placement: _LoweredPlacement
    prefetch: MXWeightPrefetch
    gradients: MXGradientPool


@dataclass(frozen=True)
class _MXCountSnapshot:
    """Source counts and expert totals from the same validated route."""

    counts: np.ndarray
    expert_totals: np.ndarray


@dataclass(frozen=True)
class _MXGeometry:
    """Exact device route descriptors and host transfer ownership."""

    sorted_ids: torch.Tensor
    counts: torch.Tensor
    copies: torch.Tensor
    owners: tuple
    loads: tuple
    rows: tuple
    metadata: route.RouteMetadata

    def storage_bytes(self):
        """Count backing storage once for offset views and the packed quota input."""
        views = (self.metadata,) if self.metadata._elements is None else (self.metadata, self.metadata._elements[1])
        storages = {}
        for tensor in (self.sorted_ids, self.counts, self.copies,
                       *(value for view in views for value in vars(view).values() if isinstance(value, torch.Tensor))):
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
        return sum(storages.values())


def _snapshot_summary(gathered, spec, source_rows):
    """Use the same exact readback for count agreement and an active B2 plan."""
    wire = gathered.cpu().numpy()
    counts = wire[:, 3:]
    expert_totals = counts.sum(0, dtype=np.int64)
    expert_loads = expert_totals.reshape(spec.ep_size, spec.local_experts)
    rows = np.column_stack((wire[:, :3], counts.sum(1, dtype=np.int64), counts.min(1), counts.max(1),
                            expert_loads.sum(1), expert_loads.min(1))).tolist()
    result = route._summarize_dynamic_rows(rows, spec, all_experts_active=rows[spec.rank_id][7] > 0)
    source_rows.extend(row[3] for row in rows)
    return result, _MXCountSnapshot(counts, expert_totals)


def _source_indices(hidden, ids, spec):
    """Match the native permutation order using a single aligned payload tile."""
    if not ids.numel():
        return ids.new_empty(0, dtype=torch.int32), None
    (_, indices), counts = route._permute_and_count(hidden.new_empty((hidden.shape[0], 16)), ids, spec)
    return indices, counts


def _render_geometry(plan, spec, tokens, plan_tokens, device):
    """Materialize the selected aligned quotas in the common Push descriptor."""
    rank, config = spec.rank_id, plan.config
    home, stride = config.home_experts, config.slots_per_rank
    borrowed = plan.compact_borrowed_width
    width = home + borrowed
    owners = tuple(row[home:width] for row in plan.slot_to_logical)
    runs = plan.source_runs_for(rank)
    slots = np.fromiter((slot // stride * width + slot % stride for slot, _ in runs),
                        dtype=np.int32, count=len(runs))
    lengths = np.fromiter((rows for _, rows in runs), dtype=np.int64, count=len(runs))
    if lengths.sum() != tokens * spec.top_k:
        raise ValueError("MX placement quotas do not cover the current source")
    host_counts = np.frombuffer(plan._dispatch_counts_bytes, dtype=np.int32).reshape(config.ep_size, config.ep_size, stride)
    host_counts = host_counts[:, :, :width].reshape(-1)
    host_copies = np.asarray(owners, dtype=np.int32).reshape(-1)
    packed = np.concatenate((host_counts, host_copies, np.repeat(slots, lengths)))
    uploaded = route._upload_route_tensor(torch.from_numpy(packed), device)
    count_end = host_counts.size
    copy_end = count_end + host_copies.size
    counts = uploaded[:count_end].view(config.ep_size, config.ep_size * width)
    copies = uploaded[count_end:copy_end].view(config.ep_size, borrowed)
    sorted_ids = uploaded[copy_end:]
    loads = plan.destination_loads
    virtual_spec = replace(spec, num_experts=spec.ep_size * width)
    metadata = route._compute_route_metadata(counts[rank], counts, None, virtual_spec, max(1, loads[rank]))
    metadata = replace(metadata, local_num_tokens=tokens, plan_tokens=plan_tokens)
    return _MXGeometry(sorted_ids, counts, copies, owners, loads, plan.destination_counts[rank][:width], metadata)


class MXPlacementResources:
    """Select MXFP8 routes and own their lazy prefetch and FP32 return banks."""

    def __init__(self, parent, *, replica_slots_per_rank=None, dedicated_bank=False, cross_node_cost=None,
                 receive_capacity_percent=None, rank_nodes=()):
        self.parent = parent
        self.pool = None
        self.dedicated_bank = dedicated_bank
        self.planner = HotReplicaPlanner(replica_slots_per_rank=replica_slots_per_rank,
                                         cross_node_cost=cross_node_cost,
                                         receive_capacity_percent=receive_capacity_percent)
        spec = parent.spec
        slots = replica_slot_budget(spec.local_experts, replica_slots_per_rank)
        self.config = ExpertReplicaConfig(spec.num_experts, spec.ep_size, slots)
        self.nodes = rank_nodes or (0,) * spec.ep_size
        self._geometry_cache = OrderedDict()
        self._geometry_cache_bytes = 0
        self._previous_active = False
        self.statistics = dict(invocations=0, placements=0, mx_full_route_readbacks=0,
                               mx_packed_route_readbacks=0, mx_packed_home_routes=0, mx_geometry_hits=0,
                               mx_geometry_misses=0, mx_geometry_cache_peak_bytes=0, w13_prioritized_calls=0)

    def _read_route(self, gathered, hidden, stream, source_rows, *, use_snapshot):
        spec, workspace = self.parent.spec, self.parent.workspace
        if use_snapshot:
            if self._previous_active:
                summary, snapshot = _snapshot_summary(gathered, spec, source_rows)
                self.statistics["mx_full_route_readbacks"] += 1
                return summary, None, None, snapshot
            packed = route._read_packed_dynamic_route(gathered, spec, hidden.shape[0], source_rows)
            if packed is not None:
                self.statistics["mx_packed_route_readbacks"] += 1
                return packed[0], None, packed[1], None
        summary, raw, metadata = route._read_cached_route(gathered, spec, workspace, hidden.device,
                                                          source_rows, stream=stream)
        return summary, raw, metadata, None

    def prepare_route(self, hidden, ids, probs, supplied_counts, mask, *, training):
        """Gather logical counts once and materialize only the route selected for this invocation."""
        parent = self.parent
        spec, workspace = parent.spec, parent.workspace
        flat_ids = route._validate_topk_inputs(hidden, ids, probs)
        stream = torch.npu.current_stream(hidden.device)
        inverse = None
        router_counts = supplied_counts is not None
        if not router_counts and hidden.shape[0] * max(spec.hidden_size, spec.top_k) <= 2**31 - 1:
            inverse, supplied_counts = _source_indices(hidden, ids, spec)
        # Checkpoint forward and recomputation use the same training policy.
        mask |= _TRAINING_MASK if training else 0
        counts, payload, _ = route._dynamic_route_payload(
            hidden, flat_ids, spec, supplied_counts, mask, workspace, stream=stream)
        workspace.wait_for_reuse(stream=stream)
        gathered, work = route._start_dynamic_gather(payload, spec)
        if work is not None:
            work.wait()
        source_rows = []
        use_snapshot = (training and spec.auto_tokens and spec.ep_size * spec.num_experts > route._HOST_ROUTE_MAX_COUNTS
                        and gathered.numel() * gathered.element_size() <= _SNAPSHOT_MAX_BYTES)
        summary, raw, metadata, snapshot = self._read_route(gathered, hidden, stream, source_rows,
                                                            use_snapshot=use_snapshot)
        self.statistics["invocations"] += 1
        geometry = self._select(hidden, gathered, summary, raw, source_rows, mask, stream,
                                snapshot=snapshot, use_snapshot=use_snapshot)
        if geometry is not None:
            try:
                placed = self._prepare_virtual(hidden, ids, gathered, geometry, summary.plan_tokens, inverse)
            except _HeapCapacityError:
                limit = receive_limit(self.planner.receive_capacity_percent, sum(source_rows), spec.ep_size)
                if limit is not None and summary.maximum > limit:
                    raise
            else:
                self._previous_active = True
                self.statistics["placements"] += 1
                return placed
        self._previous_active = False
        if metadata is not None and use_snapshot:
            self.statistics["mx_packed_home_routes"] += 1
        parent.prepare_auto_workspace(summary.plan_tokens, summary.maximum, hidden)
        permuted = route._permute_topk_input(hidden, ids)
        return route._finalize_dynamic_route(
            permuted, counts, summary, raw, metadata, None, gathered, spec, workspace,
            hidden.device, hidden.shape[0])

    def _select(self, hidden, gathered, summary, raw, source_rows, mask, stream, *, snapshot=None, use_snapshot=False):
        spec = self.parent.spec
        tokens, peak = summary.plan_tokens, summary.maximum
        prices, limit, keep_home = self.planner._policy(  # pylint: disable=protected-access
            spec.hidden_size, spec.intermediate_size, spec.num_cube_cores, bool(mask & _TRAINING_MASK),
            spec.transport_engine, peak, sum(source_rows), spec.ep_size, None,
            architecture=spec.architecture, precision="mxfp8", rank_nodes=self.nodes,
        )
        if keep_home:
            return None
        totals = None
        if snapshot is not None:
            counts, totals = snapshot.counts, snapshot.expert_totals.tolist()
        elif raw is not None:
            counts = np.asarray(raw, dtype=np.int32)[:, 3:]
        else:
            counts = gathered.cpu().numpy()[:, 3:]
            if use_snapshot:
                self.statistics["mx_full_route_readbacks"] += 1
        key = (counts.tobytes(), prices, limit, tokens, hidden.shape[0], stream)
        cache = self._geometry_cache
        if key in cache:
            cache.move_to_end(key)
            self.statistics["mx_geometry_hits"] += 1
            return cache[key][0]
        plan = select_replica_plan(
            counts, self.config, quantum=64, limit=limit, nodes=self.nodes, totals=totals,
            prices=prices, source_quota=SourceQuota.RANK_ORDER, materialize=materialize_aligned_moves,
            cross_node_cost=self.planner.cross_node_cost,
        )
        geometry = None if plan is None else _render_geometry(plan, spec, hidden.shape[0], tokens, hidden.device)
        self._retain_geometry(key, geometry, entries=_GEOMETRY_CACHE_ENTRIES if use_snapshot else 1)
        self.statistics["mx_geometry_misses"] += 1
        return geometry

    def _retain_geometry(self, key, geometry, *, entries):
        size = len(key[0]) + (geometry.storage_bytes() if geometry is not None else 0)
        if size > _GEOMETRY_CACHE_BYTES:
            return
        cache = self._geometry_cache
        while len(cache) >= entries or self._geometry_cache_bytes + size > _GEOMETRY_CACHE_BYTES:
            _, (_, released) = cache.popitem(last=False)
            self._geometry_cache_bytes -= released
        cache[key] = geometry, size
        self._geometry_cache_bytes += size
        self.statistics["mx_geometry_cache_peak_bytes"] = max(self.statistics["mx_geometry_cache_peak_bytes"],
                                                               self._geometry_cache_bytes)

    def can_close(self):
        """Check whether saved graphs still reference the FP32 return bank."""
        return self.pool is None or self.pool.can_close()

    def close(self):
        """Release the bank and cached route metadata."""
        _release_mega_bank(self.parent, self)
        self._geometry_cache.clear()
        self._geometry_cache_bytes = 0

    def _prepare_virtual(self, hidden, ids, gathered, geometry, tokens, inverse):
        parent = self.parent
        spec = parent.spec
        if inverse is None:
            inverse, _ = _source_indices(hidden, ids, spec)
        virtual_ids = geometry.sorted_ids.index_select(0, inverse.flatten()).view_as(ids)
        peak, local = max(geometry.loads), geometry.loads[spec.rank_id]
        slots = geometry.copies.shape[1]
        placement = _LoweredPlacement(
            virtual_ids=virtual_ids, counts_by_source=geometry.counts, copies=geometry.copies,
            borrowed_capacity=slots, receive_capacity=peak, local_receive_capacity=local,
            published_slots=slots, copies_by_rank=geometry.owners, owner_rows=geometry.rows,
            route_metadata=geometry.metadata,
        )
        if self.pool is None:
            self._allocate_banks(hidden, tokens, peak)
        parent.prepare_auto_workspace(tokens, peak, hidden)
        permuted = route._permute_topk_input(hidden, virtual_ids)
        prepared = route._finalize_dynamic_route(
            permuted, geometry.counts[spec.rank_id], route._RouteSummary(max(1, local), tokens, peak), None,
            geometry.metadata, None, gathered, parent._route_spec(geometry.counts.shape[1]),
            parent.workspace, hidden.device, hidden.shape[0])
        return replace(prepared, weight_binding=MXBindings(placement, self.pool.prefetch, self.pool))

    def _allocate_banks(self, hidden, tokens, received):
        """Reserve the first placed route's capacity together with its FP32 bank."""
        parent = self.parent
        manager = parent.heap_manager

        with manager.access():
            if self.pool is not None:
                return
            self.pool = _get_mega_bank(hidden, parent, self, tokens=tokens, received=received,
                                       mx_guest_slots=self.config.replica_slots_per_rank)
