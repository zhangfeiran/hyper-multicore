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
"""Independent Ascend launch boundary for exact route validation and counting."""

import ctypes
from functools import lru_cache
from typing import Optional

import torch

from hyper_multicore._loader import _component_root

_BLOCKS = 32
_MAXIMUM_EXPERTS = 512
_MAXIMUM_TOP_K = 8
_STRIDE_ALIGNMENT = 64
_DEVICE_PREFIXES = ("Ascend910B", "Ascend910_93")
_LIBRARY_PATH = _component_root() / "replica/libhyper_multicore_replica_counts.so"


@lru_cache(maxsize=8)
def _counts_available(device: torch.device) -> bool:
    """Check the installed binary and target architecture once per device."""
    path = _LIBRARY_PATH
    return path.is_file() and torch.npu.get_device_name(device).startswith(_DEVICE_PREFIXES)


@lru_cache(maxsize=1)
def _count_library() -> ctypes.CDLL:
    """Load the standalone count binary without multicore or SHMEM initialization."""
    path = _LIBRARY_PATH
    if not path.is_file():
        raise RuntimeError("Build the shared replica counts library with the repository build.sh")
    library = ctypes.CDLL(str(path))
    try:
        version = library.replica_counts_abi_version
        launch = library.launch_replica_counts
    except AttributeError as error:
        raise RuntimeError("Rebuild the shared replica counts library: missing launch ABI") from error
    version.argtypes = []
    version.restype = ctypes.c_int
    if version() not in (1, 2):
        raise RuntimeError("Rebuild the shared replica counts library: incompatible ABI version")
    launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] + [ctypes.c_int] * 3
    launch.restype = ctypes.c_int
    return library


@lru_cache(maxsize=1)
def _count_expert_limit() -> int:
    """Keep older optional binaries on their original expert-count bound."""
    return 512 if _count_library().replica_counts_abi_version() == 2 else 256


def _validate_input(ids: torch.Tensor, num_experts: int) -> None:
    if (
        ids.device.type != "npu"
        or ids.dtype != torch.int64
        or ids.ndim != 2
        or not 1 <= ids.shape[1] <= _MAXIMUM_TOP_K
        or not 1 <= num_experts <= _MAXIMUM_EXPERTS
    ):
        raise ValueError("Fused route counts require an NPU int64 matrix, TopK 1 to 8, and 1 to 512 experts")


def try_count_payload(ids: torch.Tensor, num_experts: int) -> Optional[torch.Tensor]:
    """Use the installed A2/A3 counter for supported inputs, or return None.

    Args:
        ids: Logical expert matrix; this function never initializes a non-NPU backend.
        num_experts: Logical expert count.

    Returns:
        Owned int64 counts and validation flag, or None for an unsupported or unbuilt counter.
    """
    if (
        ids.device.type != "npu"
        or ids.dtype != torch.int64
        or ids.ndim != 2
        or not 1 <= ids.shape[1] <= _MAXIMUM_TOP_K
        or not 1 <= num_experts <= _MAXIMUM_EXPERTS
    ):
        return None
    if not _counts_available(ids.device):
        return None
    if num_experts > 256 and num_experts > _count_expert_limit():
        return None
    return launch_route_counts(ids, num_experts)


def launch_route_counts(ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Launch exact integer partial counting and merging on the current stream.

    Args:
        ids: NPU int64 matrix of per-token logical expert IDs.
        num_experts: Logical experts, in the supported range 1 to 512 with count ABI 2.

    Returns:
        Invocation-owned int64 histogram followed by the collective validation flag.
    """
    _validate_input(ids, num_experts)
    library = _count_library()
    if num_experts > _count_expert_limit():
        raise ValueError("Rebuild the optional replica count library for more than 256 experts")
    with torch.npu.device(ids.device):
        source = ids.contiguous()
        stride = (num_experts + _STRIDE_ALIGNMENT) // _STRIDE_ALIGNMENT * _STRIDE_ALIGNMENT
        partials = torch.empty((_BLOCKS, stride), dtype=torch.int64, device=ids.device)
        result = torch.empty(num_experts + 1, dtype=torch.int64, device=ids.device)
        stream = torch.npu.current_stream(ids.device)
        for tensor in (source, partials, result):
            tensor.record_stream(stream)
        status = library.launch_replica_counts(
            stream.npu_stream,
            source.data_ptr(),
            partials.data_ptr(),
            result.data_ptr(),
            source.shape[0],
            source.shape[1],
            num_experts,
            stride,
        )
        if status:
            raise RuntimeError(f"Replica counts launch failed with ACL error {status}")
    return result
