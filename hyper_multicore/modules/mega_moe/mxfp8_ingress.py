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
"""Physical row-plane contract for optional original-token FP8 ingress."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import torch
from hyper_multicore._loader import load_torch_npu

MX_SOURCE_ROW_LAUNCH_VERSION = 14
MX_SOURCE_ROW_LAUNCH_RESERVED = 13
MX_SOURCE_SCALE_POSITION = 27
MX_SOURCE_INDEX_POSITION = 28
MX_SOURCE_ROW_BUFFER_COUNT = 29
MX_SOURCE_ROW_READY_TASK = 166


def source_row_fp8_enabled(*, vector_policy: str, row_only: bool, replica_slots: int, hidden_size: int) -> bool:
    """Resolve the optional original-forward protocol before creating shared resources.

    Args:
        vector_policy: Existing MX queue policy.
        row_only: Whether original no-grad forwards retain only the consumed row axis.
        replica_slots: Configured guest budget; only home-only invocations use the protocol.
        hidden_size: Feature width; packed scale copies must be aligned and fit the SDK UB lease.

    Returns:
        Whether eligible home-only no-grad invocations may use the source-row protocol.
    """
    value = os.environ.get("PANGU_MX_SOURCE_ROW_FP8", "0")
    if value not in ("0", "1"):
        raise ValueError("PANGU_MX_SOURCE_ROW_FP8 must be 0 or 1.")
    if value == "0":
        return False
    if vector_policy != "window2" or not row_only or replica_slots < 0:
        raise ValueError("Source-row FP8 ingress requires row-only window2 and a nonnegative guest budget.")
    if hidden_size < 1024 or hidden_size > 8192 or hidden_size % 1024:
        raise ValueError("Source-row FP8 ingress requires a hidden width divisible by 1024 and at most 8192.")
    return True


def borrow_row_planes(expert_buffer: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Alias complete FP8 and packed row-scale planes in the registered BF16 receive allocation.

    The caller owns the workspace lease and must pass its existing symmetric expert buffer. This function creates
    views only; it does not register ordinary storage or change the symmetric heap. The scale offset uses the
    full symmetric capacity, not this rank's actual row count, so every peer agrees on the two remote addresses.

    Args:
        expert_buffer: Contiguous registered BF16 matrix with global receive capacity and feature width.

    Returns:
        FP8 [capacity, H] and UINT8 [capacity, H/64, 2] views with disjoint logical byte regions.
    """
    if (expert_buffer.dtype != torch.bfloat16 or expert_buffer.ndim != 2 or not expert_buffer.is_contiguous()
            or expert_buffer.shape[0] < 1 or not 1024 <= expert_buffer.shape[1] <= 8192
            or expert_buffer.shape[1] % 1024):
        raise ValueError("Source-row ingress requires a contiguous BF16 receive matrix with aligned width.")
    capacity, width = expert_buffer.shape
    row_bytes = capacity * width
    raw = expert_buffer.view(torch.uint8).reshape(-1)
    row = raw[:row_bytes].view(torch.float8_e4m3fn).reshape(capacity, width)
    scale = raw[row_bytes:row_bytes + row_bytes // 32].reshape(capacity, width // 64, 2)
    return row, scale


def validate_source_row_buffers(buffers: Sequence[torch.Tensor], spec: Any) -> None:
    """Check the launch-version-14 business bindings without reading device contents.

    Args:
        buffers: Current no-grad forward buffers, including source scale and scalar route slots.
        spec: Bound B0 window2 topology.
    """
    if (len(buffers) != MX_SOURCE_ROW_BUFFER_COUNT or spec.replica_slots_per_rank
            or spec.mx_vector_policy != "window2"):
        raise ValueError("Source-row FP8 bindings require exactly 29 buffers and a B0 window2 topology.")
    if buffers[0].ndim != 2 or buffers[0].dtype != torch.bfloat16 or not buffers[0].is_contiguous():
        raise ValueError("Source-row FP8 requires the registered BF16 expert workspace.")
    capacity, width = buffers[0].shape
    if width != spec.hidden_size:
        raise ValueError("The source-row workspace width differs from the bound topology.")
    contracts = ((2, torch.float8_e4m3fn, (spec.local_num_tokens, width)),
                 (16, torch.float8_e4m3fn, (capacity, width)),
                 (17, torch.uint8, (capacity, width // 64, 2)),
                 (18, torch.float8_e4m3fn, (capacity, width)),
                 (MX_SOURCE_SCALE_POSITION, torch.uint8, (spec.local_num_tokens, width // 64, 2)),
                 (MX_SOURCE_INDEX_POSITION, torch.int32, (spec.local_num_tokens * spec.top_k,)))
    for position, dtype, shape in contracts:
        value = buffers[position]
        if value.dtype != dtype or value.shape != shape or not value.is_contiguous():
            raise ValueError(f"Source-row FP8 binding {position} has an invalid physical layout.")
    if (buffers[16].data_ptr() != buffers[0].data_ptr() or buffers[18].data_ptr() != buffers[0].data_ptr()
            or buffers[17].data_ptr() != buffers[0].data_ptr() + capacity * width):
        raise ValueError("Source-row receive planes must alias the existing symmetric expert workspace.")


def source_row_physical_buffers(buffers: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Pass complete registered storage through CANN while retaining separate logical plane bounds.

    Args:
        buffers: Validated logical source-row business bindings, with receive views in positions 16 through 18.

    Returns:
        Current bindings whose three receive roles share the complete registered base allocation. Native code
        restores the scale offset after checking the physical aliases; no data is copied or allocated here.
    """
    if len(buffers) != MX_SOURCE_ROW_BUFFER_COUNT:
        raise ValueError("Source-row physical bindings require exactly 29 buffers.")
    base = buffers[0]
    if base.dtype != torch.bfloat16 or base.ndim != 2 or not base.is_contiguous():
        raise ValueError("Source-row physical bindings require the contiguous BF16 expert allocation.")
    capacity, width = base.shape
    contracts = ((16, torch.float8_e4m3fn, (capacity, width), 0),
                 (17, torch.uint8, (capacity, width // 64, 2), capacity * width),
                 (18, torch.float8_e4m3fn, (capacity, width), 0))
    for position, dtype, shape, offset in contracts:
        value = buffers[position]
        if (value.dtype != dtype or value.shape != shape or not value.is_contiguous()
                or value.data_ptr() != base.data_ptr() + offset):
            raise ValueError("Source-row physical aliases must retain the validated logical receive planes.")
    physical = list(buffers)
    physical[16:19] = [base, base, base]
    return physical


def quantize_source_rows(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize the original T-by-H source once without materializing Top-K vectors.

    Args:
        hidden: Original BF16 token rows, before scalar routing.

    Returns:
        Original FP8 rows and packed UINT8 row scales; unused column outputs are not retained.
    """
    axes = load_torch_npu().npu_dynamic_mx_quant_with_dual_axis(
        hidden.contiguous(), dst_type=torch.float8_e4m3fn, scale_alg=1)
    row, scale = axes[:2]
    if (row.dtype != torch.float8_e4m3fn or row.shape != hidden.shape or not row.is_contiguous()
            or scale.dtype != torch.uint8 or scale.numel() != hidden.numel() // 32 or not scale.is_contiguous()):
        raise ValueError("Native source quantization returned an unsupported FP8 or scale layout.")
    return row, scale.reshape(hidden.shape[0], hidden.shape[1] // 64, 2)
