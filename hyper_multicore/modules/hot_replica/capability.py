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
"""Validate the actual input device and selected HotReplica artifacts before allocation."""

from pathlib import Path
import os
import socket
from weakref import WeakKeyDictionary

import torch
from hyper_multicore import _collective
from hyper_multicore._loader import load_torch_npu, query_npu_arch, validate_native_compatibility
from hyper_multicore._native_manifest import A5_BF16_KERNEL_PIPELINE_VERSION
from ._planner.native_host import load_replica_planner
from .native import load_native_extension


_MTE_EDGES = WeakKeyDictionary()


def _local_mte_peers(tensor, rank, ranks):
    """Query the same runtime state that the A5 device prefetch uses."""
    bridge = load_native_extension("_hot_replica_shared_bridge")
    probe = getattr(bridge, "mte_peers", None)
    if probe is None:
        return (False,) * ranks
    peers = torch.empty(ranks, dtype=torch.int32, device=tensor.device)
    probe(peers, rank)
    return tuple(value == 1 for value in peers.tolist())


def a5_mte_edges(tensor, manager):
    """Agree on bidirectional replica edges once per attached heap epoch."""
    epoch = manager.epoch
    cached = _MTE_EDGES.get(manager)
    if cached is not None and cached[0] == epoch:
        return cached[1]
    group, ranks = manager.root_group, len(manager.members)
    rank = torch.distributed.get_rank(group)
    failure, peers = None, ()
    try:
        peers = _local_mte_peers(tensor, rank, ranks)
    except Exception as error:  # pylint: disable=broad-exception-caught
        failure = error
    report = None if failure is None else _collective.failure_report(failure)
    results = _collective.exchange((peers, report), group=group, failure_prefix="A5 HotReplica MTE probe")
    rows, failures = zip(*results)
    if any(failures):
        details = _collective.format_collective_failures(failures)
        raise RuntimeError(f"A5 HotReplica MTE probe failed on {details}") from failure
    if len(rows) != ranks or any(len(row) != ranks for row in rows):
        raise RuntimeError("A5 HotReplica MTE probe does not match its SHMEM Root")
    edges = tuple(tuple(a == b or rows[a][b] and rows[b][a] for b in range(ranks)) for a in range(ranks))
    # The unrestricted planner can keep using its native search.
    edges = () if all(map(all, edges)) else edges
    _MTE_EDGES[manager] = epoch, edges
    return edges


def unreachable_replica_edge(placement, home, edges):
    """Return the first owner/guest pair lacking the required capabilities."""
    if edges:
        for guest, owners in enumerate(placement.copies_by_rank):
            for expert in owners:
                if expert >= 0 and not edges[expert // home][guest]:
                    return expert // home, guest
    return None


def _require_a3(tensor, *, dispatch_mode: str = "push") -> bool:
    """Reject devices sharing arch22 unless their actual SOC is Atlas A3."""
    if tensor.device.type != "npu" or tensor.dtype != torch.bfloat16:
        raise ValueError("HotReplica requires BF16 input on an Atlas A3 NPU")
    load_torch_npu()
    index = tensor.device.index
    if index is None:
        index = torch.npu.current_device()
    name = torch.npu.get_device_name(index).lower()
    if query_npu_arch(index) != 2201 or not any(part in name for part in ("910_93", "910c")):
        raise ValueError(f"HotReplica supports Atlas A3 only, input device {index} is {name}")
    selected = validate_native_compatibility()
    artifacts = selected["variant"].get("hot_replica")
    if not artifacts or artifacts.get("soc") != "ascend910_93":
        raise RuntimeError("The selected Torch/PTA adapter has no Atlas A3 HotReplica artifacts")
    for key in ("bridge", "kernels"):
        if not (selected["root"] / artifacts[key]).is_file():
            raise RuntimeError(f"Selected HotReplica artifact is missing: {key}")
    if "planner" in artifacts:
        load_replica_planner(selected["root"] / artifacts["planner"])
    build_info = selected["build_info"]
    if dispatch_mode == "pull" and not build_info.get("a3_bf16_pull_runtime_rows", False):
        raise RuntimeError("BF16 HotReplica Pull on Ascend910_93 requires an A3 wheel with runtime-row GET support")
    return bool(build_info.get("a3_replica_bf16_partials", False))


def _require_a5_bf16(tensor):
    """Validate the selected Ascend950 BF16 bank and its device pipeline."""
    if tensor.device.type != "npu" or tensor.dtype != torch.bfloat16:
        raise ValueError("BF16 HotReplica requires BF16 input on Ascend950")
    selected = validate_native_compatibility()
    artifacts = selected["variant"].get("hot_replica_bf16_arch35")
    if not artifacts or artifacts.get("soc") != "ascend950":
        raise RuntimeError("The selected Torch/PTA adapter has no Ascend950 BF16 HotReplica artifacts")
    for key in ("bridge", "kernels"):
        if not (selected["root"] / artifacts[key]).is_file():
            raise RuntimeError(f"Selected Ascend950 BF16 HotReplica artifact is missing: {key}")
    if "planner" in artifacts:
        load_replica_planner(selected["root"] / artifacts["planner"])
    return artifacts.get("kernel_pipeline") == A5_BF16_KERNEL_PIPELINE_VERSION


def _require_mx(tensor, cube_cores):
    """Validate aligned-planner artifacts and sparse weight transport before SHMEM."""
    if tensor.device.type != "npu" or tensor.dtype != torch.bfloat16:
        raise ValueError("MXFP8 HotReplica requires BF16 input on Ascend950")
    load_torch_npu()
    index = tensor.device.index
    if index is None:
        index = torch.npu.current_device()
    if query_npu_arch(index) != 3510:
        raise ValueError("MXFP8 HotReplica requires Ascend950")
    from ..mega_moe.schedule import MX_GRADIENT_CUBE_CORES  # pylint: disable=import-outside-toplevel

    if cube_cores != MX_GRADIENT_CUBE_CORES:
        raise ValueError(f"MXFP8 HotReplica requires {MX_GRADIENT_CUBE_CORES} Cube cores, got {cube_cores}")
    if os.environ.get("HCCL_OP_EXPANSION_MODE") != "AICPU_CacheDisable":
        raise ValueError("MXFP8 HotReplica prefetch requires HCCL_OP_EXPANSION_MODE=AICPU_CacheDisable")
    selected = validate_native_compatibility()
    artifacts = selected["variant"].get("hot_replica_bf16_arch35", {})
    if "planner" not in artifacts:
        raise RuntimeError("The selected Torch/PTA adapter has no Ascend950 HotReplica planner")
    load_replica_planner(selected["root"] / artifacts["planner"])


def require_capability(
    tensor: torch.Tensor, group: torch.distributed.ProcessGroup, *,
    dispatch_mode: str = "push", architecture: int = 2201, mx_cube_cores: int | None = None,
) -> tuple[bool, tuple[int, ...]]:
    """Agree on the device artifacts and node labels before planning any route."""
    local_error = None
    fused_support = False
    try:
        if mx_cube_cores is not None:
            _require_mx(tensor, mx_cube_cores)
        elif architecture == 3510:
            fused_support = _require_a5_bf16(tensor)
        else:
            fused_support = _require_a3(tensor, dispatch_mode=dispatch_mode)
    except Exception as error:  # pylint: disable=broad-exception-caught
        local_error = error
    try:
        node = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        node = ""
    failure = None if local_error is None else _collective.failure_report(local_error)
    reports = _collective.exchange(
        (fused_support, node or socket.gethostname(), failure), group=group,
        failure_prefix="HotReplica capability failed while converging ranks",
    )
    capabilities, nodes, failures = zip(*reports)
    if any(failure is not None for failure in failures):
        details = _collective.format_collective_failures(failures)
        raise RuntimeError(f"HotReplica capability failed on {details}") from local_error
    labels = {node: rank for rank, node in enumerate(dict.fromkeys(nodes))}
    return all(capabilities), tuple(labels[node] for node in nodes)
