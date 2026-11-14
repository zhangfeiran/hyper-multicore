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
"""Resolve execution accelerators independently of replica cost calibration."""

import os
from dataclasses import replace

from hyper_multicore.replica.admission import reduction_basis_points, resolve_replica_admission

from .event_refresh import event_refresh_intervals
from .gradient_queue import ACTIVE_GRADIENT_PROJECTIONS, ADAPTIVE_GRADIENT_QUEUE
from .spec import MegaMoeSpec


def qualify_topology_schedule(spec: MegaMoeSpec, rank_nodes: tuple[int, ...]) -> MegaMoeSpec:
    """Preserve the source cross-node W2 assignment for automatic scheduling.

    Args:
        spec: Shape specification after rank manifests have agreed.
        rank_nodes: Canonical node identity for every EP rank.

    Returns:
        Automatic W1-only scheduling across nodes, or the original specification.
    """
    if len(set(rank_nodes)) > 1 and spec._gradient_queue_order == "auto":
        return replace(spec, _gradient_queue_order="auto_w1")
    return spec


def summary_admission(budget: int) -> int:
    """Resolve the default summary shortcut independently of planner execution.

    Args:
        budget: Maximum guest slots per rank.

    Returns:
        Integer admission basis points, or zero when explicitly disabled.
    """
    enabled = os.environ.get("HYPER_MULTICORE_MOE_SUMMARY_HOME_ADMISSION", "1") == "1"
    return reduction_basis_points(resolve_replica_admission(None)[0]) if budget and enabled else 0


def execution_options(dispatch_mode: str, budget: int) -> dict:
    """Retain source defaults and explicit research controls without a cost policy.

    Args:
        dispatch_mode: Push or Pull activation transport.
        budget: Maximum guest slots per execution rank.

    Returns:
        Immutable-spec fields participating in resource and group agreement.
    """
    order = os.environ.get("HYPER_MULTICORE_MOE_GRADIENT_QUEUE_ORDER", "auto")
    supported = (
        "auto",
        "auto_w1",
        "original",
        "w1_reverse",
        "w1_join_once",
        "device_w1_asc",
        "device_w1_desc",
        "device_w1_asc_join",
        "device_late_auto",
        "device_lane_rotate",
        "device_w1_asc_join_once",
        "device_w1_asc_join_rotate",
        ADAPTIVE_GRADIENT_QUEUE,
        *ACTIVE_GRADIENT_PROJECTIONS,
    )
    if order not in supported:
        raise ValueError("Unsupported HYPER_MULTICORE_MOE_GRADIENT_QUEUE_ORDER")
    fused_save = os.environ.get("HYPER_MULTICORE_MOE_FUSED_DISPATCH_SAVE")
    if fused_save not in (None, "0", "1"):
        raise ValueError("HYPER_MULTICORE_MOE_FUSED_DISPATCH_SAVE must be 0 or 1")
    copy_lanes = os.environ.get("HYPER_MULTICORE_MOE_SAVE_DISPATCH_LANES", "0")
    try:
        copy_lanes = int(copy_lanes)
    except ValueError as error:
        raise ValueError("HYPER_MULTICORE_MOE_SAVE_DISPATCH_LANES must be an integer in [0, 128]") from error
    if not 0 <= copy_lanes <= 128:
        raise ValueError("HYPER_MULTICORE_MOE_SAVE_DISPATCH_LANES must be in [0, 128]")
    gradient_outputs = _gradient_output_options(budget)
    cube, vector = event_refresh_intervals()
    result = {
        "_gradient_queue_order": order,
        "_fuse_saved_dispatch": dispatch_mode == "push" and fused_save == "1",
        "_saved_dispatch_lanes": copy_lanes,
        "_cube_event_refresh_us": cube,
        "_vector_event_refresh_us": vector,
        "_batch_empty_dispatch": dispatch_mode == "push"
        and not budget
        and os.environ.get("HYPER_MULTICORE_MOE_BATCH_EMPTY_DISPATCH") == "1",
        "_batch_empty_combine": dispatch_mode == "push"
        and os.environ.get("HYPER_MULTICORE_MOE_BATCH_EMPTY_COMBINE") == "1",
        "_compact_communication": dispatch_mode == "push"
        and os.environ.get("HYPER_MULTICORE_MOE_COMPACT_COMMUNICATION") == "1",
        "_selective_owner_gradient_init": os.environ.get("HYPER_MULTICORE_MOE_OWNER_GRADIENT_INIT") != "full",
        "_host_replica_metadata": os.environ.get("HYPER_MULTICORE_MOE_ROUTE_METADATA") != "device",
    }
    result.update(gradient_outputs)
    if (
        budget > 1
        and os.environ.get("HYPER_MULTICORE_SHMEM_HEAP_SIZE") is None
        and os.environ.get("HYPER_MULTICORE_MOE_ADAPTIVE_REPLICA_INBOX") == "1"
    ):
        result["_replica_inbox_slots"] = 1
    return result


def _gradient_output_options(budget: int) -> dict:
    """Validate both output requests before exposing a coherent native capability choice."""
    local_gradient = os.environ.get("HYPER_MULTICORE_REPLICA_LOCAL_BF16")
    if local_gradient not in (None, "0", "1"):
        raise ValueError("HYPER_MULTICORE_REPLICA_LOCAL_BF16 must be 0 or 1")
    final_gradient = os.environ.get("HYPER_MULTICORE_REPLICA_FINAL_BF16")
    if final_gradient not in (None, "0", "1"):
        raise ValueError("HYPER_MULTICORE_REPLICA_FINAL_BF16 must be 0 or 1")
    if final_gradient == "1" and local_gradient == "0":
        raise ValueError("Final BF16 gradients require local BF16 output support")
    return {
        "_local_gradient_bf16": None if budget and local_gradient is None else bool(budget) and local_gradient == "1",
        "_final_gradient_bf16": (
            None if budget and final_gradient is None and local_gradient != "0"
            else bool(budget) and final_gradient == "1"
        ),
    }
