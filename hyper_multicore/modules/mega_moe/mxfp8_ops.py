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
"""Byte-alias launch adapter for the separately versioned native MX task graph."""

from __future__ import annotations

from functools import lru_cache
from collections.abc import Sequence
import hashlib
import json

import torch

from hyper_multicore.replica._metadata import _async_device_bytes
from hyper_multicore.scheduler.mxfp8 import MxLaunchDescriptor, serialize_mx_launch_metadata
from hyper_multicore.torch.ops import _load_native
from hyper_multicore._loader import get_multicore_paths

from .spec import MegaMoeSpec
from .mxfp8_ingress import (MX_SOURCE_ROW_LAUNCH_RESERVED, MX_SOURCE_ROW_LAUNCH_VERSION,
                            source_row_physical_buffers, validate_source_row_buffers)


@lru_cache(maxsize=1)
def _load_mxfp8_native() -> None:
    _load_native()
    if torch.ops.hyper_multicore.mxfp8_task_graph_version() != 7:
        raise RuntimeError("MXFP8 task-graph adapter ABI mismatch; rebuild the local multicore component.")


def mxfp8_window_feature_available(feature: str) -> bool:
    """Check an optional feature against the actual native payload.

    Args:
        feature: Optional task capability to query.

    Returns:
        Whether the verified payload implements the requested feature.
    """
    _load_mxfp8_native()
    vendor, _ = get_multicore_paths()
    receipt = json.loads((vendor.parent.parent / "capabilities/mxfp8_window.json").read_text())
    expected = "PanguMxfp8TaskGraph_ed65f2bbd39f9cc68999acec5148f3f8.o"
    if receipt.get("schema") != 7:
        raise RuntimeError("MX window capability receipt has an unsupported schema.")
    if receipt.get("object") != expected:
        raise RuntimeError("MX window capability receipt names an unexpected binary.")
    binary = vendor / "op_impl/ai_core/tbe/kernel/ascend950/pangu_mxfp8_task_graph" / expected
    if hashlib.sha256(binary.read_bytes()).hexdigest() != receipt.get("sha256"):
        raise RuntimeError("MX window capability receipt does not match the loaded vendor payload.")
    return feature in receipt.get("features", ())


@lru_cache(maxsize=None)
def require_mxfp8_window_feature(feature: str) -> None:
    """Require a matching native capability receipt before submitting optional tasks.

    Args:
        feature: Optional task capability required by the current graph.
    """
    if not mxfp8_window_feature_available(feature):
        raise RuntimeError(f"MX window feature is unavailable: {feature}")


def _byte_aliases(buffers: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Keep exact byte views while avoiding redundant dtype or dimension conversions."""
    aliases = []
    for tensor in buffers:
        if not tensor.is_contiguous():
            raise ValueError("MXFP8 task buffers must be contiguous; implicit physical weight copies are forbidden.")
        alias = tensor if tensor.dtype == torch.uint8 else tensor.view(torch.uint8)
        aliases.append(alias if alias.ndim == 1 else alias.reshape(-1))
    return aliases


def serialize_mxfp8_launch(
    buffers: Sequence[torch.Tensor], runtime: torch.Tensor, events: torch.Tensor, spec: MegaMoeSpec,
    *, kernel_gradients: bool = False, weight_prefetch: bool = False, source_row_fp8: bool = False,
) -> bytes:
    """Serialize exact current byte bounds without reading device tensor contents.

    Args:
        buffers: Current physical buffers in the native launch order.
        runtime: Device runtime image with current task and event capacities.
        events: Symmetric event and persistent ready-generation storage.
        spec: Validated physical expert topology and current capacity.
    """
    content = serialize_mx_launch_metadata(
        [tensor.numel() * tensor.element_size() for tensor in buffers],
        rank=spec.rank_id,
        ep=spec.ep_size,
        experts=spec.num_experts,
        hidden=spec.hidden_size,
        local_tokens=spec.local_num_tokens,
        cube_count=spec.num_cube_cores,
        runtime_bytes=runtime.numel(),
        event_bytes=events.numel(),
        vector_policy=spec.mx_vector_policy,
    )
    if source_row_fp8:
        if kernel_gradients or weight_prefetch:
            raise ValueError("Source-row FP8 ingress cannot use the guest weight or gradient launch protocol.")
        validate_source_row_buffers(buffers, spec)
        record = MxLaunchDescriptor.from_buffer_copy(content)
        record.version, record.reserved = MX_SOURCE_ROW_LAUNCH_VERSION, MX_SOURCE_ROW_LAUNCH_RESERVED
        return bytes(record)
    if not kernel_gradients and not weight_prefetch:
        return content
    if spec.mx_vector_policy != "window2" or spec.replica_transport != "shmem_signal_kernel_gradient":
        raise ValueError("MX gradient return requires its dedicated transport and window2 queue")
    record = MxLaunchDescriptor.from_buffer_copy(content)
    record.version, record.reserved = ((13, 12) if kernel_gradients else (12, 11)) if weight_prefetch else (11, 10)
    return bytes(record)


def prepare_mxfp8_launch(
    buffers: Sequence[torch.Tensor], runtime: torch.Tensor, events: torch.Tensor, spec: MegaMoeSpec,
    *, kernel_gradients: bool = False, weight_prefetch: bool = False, source_row_fp8: bool = False,
) -> torch.Tensor:
    """Transfer the byte-bounds launch descriptor to the active device.

    Args:
        buffers: Current physical buffers in the native launch order.
        runtime: Device runtime image with current task and event capacities.
        events: Symmetric event and persistent ready-generation storage.
        spec: Validated physical expert topology and current capacity.
    """
    metadata = serialize_mxfp8_launch(buffers, runtime, events, spec, kernel_gradients=kernel_gradients,
                                     weight_prefetch=weight_prefetch, source_row_fp8=source_row_fp8)
    return _async_device_bytes(metadata, runtime.device)


def launch_mxfp8_task_graph(
    buffers: Sequence[torch.Tensor],
    runtime: torch.Tensor,
    events: torch.Tensor,
    diagnostic: torch.Tensor,
    launch: torch.Tensor,
    *,
    vector_policy: str = "legacy_odd",
    swiglu_row_only: bool = False,
    source_row_fp8: bool = False,
) -> None:
    """Run the native fused task graph with alias-preserving physical arguments.

    Args:
        buffers: Current physical buffers in the native launch order.
        runtime: Device runtime image with current task and event capacities.
        events: Symmetric event and persistent ready-generation storage.
        diagnostic: Explicit diagnostic storage; ordinary launches keep capture disabled.
        launch: Native task-graph launch callable.
    """
    aliases = _byte_aliases(source_row_physical_buffers(buffers) if source_row_fp8 else buffers)
    _load_mxfp8_native()
    if vector_policy not in ("legacy_odd", "window2") or (swiglu_row_only and vector_policy != "window2"):
        raise ValueError("The integrated MX payload supports legacy_odd and window2 graphs only.")
    if swiglu_row_only:
        require_mxfp8_window_feature("nograd_row_only")
        require_mxfp8_window_feature("nograd_x_row_only")
    if source_row_fp8:
        require_mxfp8_window_feature("mx_source_row_fp8")
        require_mxfp8_window_feature("mx_source_row_ready_event")
        require_mxfp8_window_feature("mx_source_row_base_bindings")
    torch.ops.hyper_multicore.mxfp8_task_graph(aliases, runtime, events, diagnostic, launch)
