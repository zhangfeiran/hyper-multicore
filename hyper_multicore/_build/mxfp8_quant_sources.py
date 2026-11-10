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
"""Compose the pinned MX quantization kernels without modifying cached Git sources."""

from __future__ import annotations

from pathlib import Path
import shutil

QUANT_KERNELS = {
    "dynamic": "dynamic_mx_quant_with_dual_axis",
    "swiglu": "swiglu_mx_quant_with_dual_axis",
    "swiglu_row": "swiglu_mx_quant",
    "swiglu_grad": "swiglu_backward_mx_quant_with_dual_axis",
}
QUANT_SOURCE_PATHS = tuple(f"quant/{name}/op_kernel/arch35" for name in QUANT_KERNELS.values()) + (
    "common/inc/op_kernel/platform.h",
    "common/inc/op_kernel/kernel_utils.h",
)


def compose_quant_sources(ops_nn_source: Path, destination: Path) -> None:
    """Place each kernel and its relative common-header includes under one closure.

    Args:
        ops_nn_source: Verified pinned ops-nn source export.
        destination: New isolated directory receiving the quantization source closure.
    """
    for name, upstream in QUANT_KERNELS.items():
        target = destination / name
        shutil.copytree(ops_nn_source / "quant" / upstream / "op_kernel/arch35", target / "arch35")
        (target / "inc").mkdir()
        for header in ("platform.h", "kernel_utils.h"):
            shutil.copy2(ops_nn_source / "common/inc/op_kernel" / header, target / "inc" / header)
