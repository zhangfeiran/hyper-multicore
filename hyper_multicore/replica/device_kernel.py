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
"""Independent Ascend AIV launch boundary for shared expert replica planning."""

from __future__ import annotations

import ctypes
from functools import lru_cache

import torch

from hyper_multicore._loader import _component_root

from .capacity import ExpertReplicaConfig, _integer


@lru_cache(maxsize=1)
def _planner_library() -> ctypes.CDLL:
    """Load only the planner library, with no multicore or SHMEM initialization."""
    path = _component_root() / "replica/libhyper_multicore_replica_planner.so"
    if not path.is_file():
        raise RuntimeError("The shared device planner is unavailable in this native payload")
    library = ctypes.CDLL(str(path))
    try:
        version = library.planner_abi_version
    except AttributeError as error:
        raise RuntimeError("Rebuild the shared device planner: missing ABI version") from error
    version.argtypes = []
    version.restype = ctypes.c_int
    if version() != 4:
        raise RuntimeError("Rebuild the shared device planner: incompatible ABI version")
    library.launch_planner.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int] * 3 + [ctypes.c_int64] * 5)
    library.launch_planner.restype = ctypes.c_int
    return library


def _validate_device(counts: torch.Tensor, config: ExpertReplicaConfig) -> None:
    """Keep topology arithmetic and the single-AIV scratch within their bounds."""
    if counts.device.type != "npu":
        raise ValueError("The fused device replica planner requires an Ascend NPU")
    scratch = 5 * config.ep_size * config.num_experts + 3 * config.num_experts + 4 * config.ep_size
    if scratch > 23040 or config.ep_size * config.physical_experts > torch.iinfo(torch.int32).max:
        raise ValueError("Replica topology exceeds the fused device planner scratch or index capacity")


def launch_device_planner(counts: torch.Tensor, config: ExpertReplicaConfig,
                          capacity: int | None, target: int | None,
                          minimum_rows: int, admission_basis_points: int = 0,
                          minimum_saved_rows: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch one task with invocation-owned buffers on the current NPU stream.

    Callers supply producer dependencies for counts. Stream recording keeps all
    launch arguments alive even if counts originated on another stream. The
    returned buffers have no dependency on a mutable graph replay workspace.

    Args:
        counts: Exact logical source histograms.
        config: Expert topology and per-rank slot budget.
        capacity: Hard receive bound, or None for the inferred capacity.
        target: Soft desired receive rows.
        minimum_rows: Soft minimum rows served by each guest copy.
        admission_basis_points: Required fractional peak reduction in integer basis points.
        minimum_saved_rows: Required absolute peak reduction for optional copies.

    Returns:
        Control and dispatch tensors retained by the caller.
    """
    _validate_device(counts, config)
    if (isinstance(admission_basis_points, bool) or not isinstance(admission_basis_points, int)
            or not 0 <= admission_basis_points <= 10000):
        raise ValueError("Admission basis points must be an integer between zero and 10000")
    _integer(minimum_saved_rows, "minimum_saved_rows", 0)
    if minimum_saved_rows > torch.iinfo(torch.int64).max:
        raise ValueError("minimum_saved_rows exceeds the device integer range")
    library = _planner_library()
    with torch.npu.device(counts.device):
        counts = counts.to(torch.int64).contiguous()
        ranks, width = config.ep_size, config.slots_per_rank
        control = torch.empty(1 + 2 * ranks * width + ranks * ranks + (ranks + 1) * ranks * width,
                              dtype=torch.int64, device=counts.device)
        dispatch = torch.empty((ranks, ranks * width), dtype=torch.int32, device=counts.device)
        stream = torch.npu.current_stream(counts.device)
        for tensor in (counts, control, dispatch):
            tensor.record_stream(stream)
        status = library.launch_planner(stream.npu_stream, counts.data_ptr(), control.data_ptr(), dispatch.data_ptr(),
                                        ranks, config.num_experts, config.replica_slots_per_rank,
                                        -1 if capacity is None else capacity, target or 0, minimum_rows,
                                        admission_basis_points, minimum_saved_rows)
        if status:
            raise RuntimeError(f"Replica planner launch failed with ACL error {status}")
    return control, dispatch
