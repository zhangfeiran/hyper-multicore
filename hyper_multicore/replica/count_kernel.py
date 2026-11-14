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
from typing import Callable, NamedTuple, Optional

import torch

from hyper_multicore._loader import _component_root, native_ready

_BLOCKS = 32
_MAXIMUM_EXPERTS = 512
_MAXIMUM_TOP_K = 8
_STRIDE_ALIGNMENT = 64
_DEVICE_PREFIXES = ("Ascend910B", "Ascend910_93")
_LIBRARY_PATH = _component_root() / "replica/libhyper_multicore_replica_counts.so"


class _QueueOperations(NamedTuple):
    """Bind both queued operations to the exact optional count binary."""

    payload: Callable[..., torch.Tensor]
    summary: Callable[..., torch.Tensor]
    library_path: str


@lru_cache(maxsize=1)
def _loaded_queue_operations() -> Optional[_QueueOperations]:
    """Probe only a loaded adapter, retaining support for older payloads."""
    namespace = torch.ops.hyper_multicore
    try:
        version = namespace.replica_count_queue_version
    except AttributeError:
        return None
    if version() != 1:
        return None
    return _QueueOperations(namespace.replica_route_payload, namespace.replica_route_summary, str(_LIBRARY_PATH))


def _queue_operations() -> Optional[_QueueOperations]:
    """Standalone counting must not initialize multicore or cache an unloaded adapter."""
    return _loaded_queue_operations() if native_ready("torch") else None


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


@lru_cache(maxsize=1)
def _route_payload_launch():
    """Keep old optional count libraries on their original compatible count path."""
    launch = getattr(_count_library(), "launch_replica_route_payload", None)
    if launch is not None:
        launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] + [ctypes.c_int] * 5
        launch.restype = ctypes.c_int
    return launch


def try_route_payload(ids: torch.Tensor, num_experts: int, autograd_mask: int) -> Optional[torch.Tensor]:
    """Generate a current INT32 control header and validated histogram together.

    Args:
        ids: Current NPU integer logical expert IDs, one Top-K row per token.
        num_experts: Logical experts supported by the optional count library.
        autograd_mask: Current local gradient controls carried by count agreement.

    Returns:
        Owned INT32 header and counts, or None to preserve the existing fallback.
    """
    if (ids.device.type != "npu" or ids.dtype not in (torch.int32, torch.int64) or ids.ndim != 2
            or not 1 <= ids.shape[1] <= _MAXIMUM_TOP_K or not 1 <= num_experts <= _MAXIMUM_EXPERTS
            or ids.numel() > 2**31 - 1):
        return None
    if not _counts_available(ids.device) or num_experts > _count_expert_limit():
        return None
    return launch_route_payload(ids, num_experts, autograd_mask)


def launch_route_payload(ids: torch.Tensor, num_experts: int, autograd_mask: int) -> Optional[torch.Tensor]:
    """Publish counts and invalid-ID flags without extra casts or header kernels.

    Args:
        ids: Supported NPU INT32 or INT64 route matrix.
        num_experts: Logical expert count validated by ``try_route_payload``.
        autograd_mask: Integer local gradient mask for coordinated validation.

    Returns:
        Packed current header and counts, or None for an older optional binary.
    """
    if (ids.device.type != "npu" or ids.dtype not in (torch.int32, torch.int64) or ids.ndim != 2
            or not 1 <= ids.shape[1] <= _MAXIMUM_TOP_K or not 1 <= num_experts <= _MAXIMUM_EXPERTS
            or ids.numel() > 2**31 - 1):
        raise ValueError("Route payload requires NPU INT32/INT64 IDs, TopK 1 to 8 and 1 to 512 experts")
    if isinstance(autograd_mask, bool) or not isinstance(autograd_mask, int) or not -(2**31) <= autograd_mask < 2**31:
        raise ValueError("Route payload autograd controls must fit a signed INT32 integer")
    launch = _route_payload_launch()
    if launch is None:
        return None
    queued = _queue_operations()
    if queued is not None and _route_summary_launch() is not None:
        return queued.payload(ids, num_experts, autograd_mask, queued.library_path)
    with torch.npu.device(ids.device):
        source = ids.contiguous()
        stride = (num_experts + _STRIDE_ALIGNMENT) // _STRIDE_ALIGNMENT * _STRIDE_ALIGNMENT
        partials = torch.empty((_BLOCKS, stride), dtype=torch.int64, device=ids.device)
        result = torch.empty(num_experts + 3, dtype=torch.int32, device=ids.device)
        stream = torch.npu.current_stream(ids.device)
        for tensor in (source, partials, result):
            tensor.record_stream(stream)
        status = launch(stream.npu_stream, source.data_ptr(), partials.data_ptr(), result.data_ptr(),
                        source.shape[0], source.shape[1], num_experts, stride, autograd_mask, source.element_size())
        if status:
            raise RuntimeError(f"Replica route payload launch failed with ACL error {status}")
    return result


@lru_cache(maxsize=1)
def _route_summary_launch():
    """Probe the optional exact count-table comparison without changing ABI 2."""
    launch = getattr(_count_library(), "launch_replica_route_summary", None)
    if launch is not None:
        launch.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 2
        launch.restype = ctypes.c_int
    return launch


def try_route_summary(payload: torch.Tensor, previous: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
    """Summarize fresh count agreement and compare every source/count/control value.

    Args:
        payload: Current contiguous INT32 source headers and logical counts.
        previous: Immutable earlier count table with the identical shape and device.

    Returns:
        Per-rank INT64 controls, sums, bounds, owner loads and exact-change flags,
        or None when the optional native operation is unavailable.
    """
    if not _route_summary_input_supported(payload):
        return None
    if not _counts_available(payload.device):
        return None
    launch = _route_summary_launch()
    if launch is None:
        return None
    if previous is not None and (previous.shape != payload.shape or previous.device != payload.device
                                 or previous.dtype != payload.dtype or not previous.is_contiguous()):
        previous = None
    queued = _queue_operations()
    if queued is not None and _route_payload_launch() is not None:
        return queued.summary(payload, previous, queued.library_path)
    with torch.npu.device(payload.device):
        result = torch.empty((payload.shape[0], 8), dtype=torch.int64, device=payload.device)
        stream = torch.npu.current_stream(payload.device)
        for tensor in (payload, result, previous):
            if tensor is not None:
                tensor.record_stream(stream)
        status = launch(stream.npu_stream, payload.data_ptr(), 0 if previous is None else previous.data_ptr(),
                        result.data_ptr(), payload.shape[0], payload.shape[1] - 3)
        if status:
            raise RuntimeError(f"Replica route summary launch failed with ACL error {status}")
    return result


def _route_summary_input_supported(payload: torch.Tensor) -> bool:
    """Require the exact source-count layout before probing any device operation."""
    if (payload.device.type != "npu" or payload.dtype != torch.int32 or payload.ndim != 2
            or payload.shape[0] < 1 or not 1 <= payload.shape[1] - 3 <= _MAXIMUM_EXPERTS
            or (payload.shape[1] - 3) % payload.shape[0] or not payload.is_contiguous()):
        return None
    return True
