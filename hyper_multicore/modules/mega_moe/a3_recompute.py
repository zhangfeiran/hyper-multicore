# Copyright 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Capability and device ABI for the opt-in A3 recompute-backward launch."""

import ctypes
from functools import lru_cache
import os

from hyper_multicore._loader import require_multicore_environment

SEQUENCE_BIT = 1 << 40
CALL_MAGIC = 0x41335245434F4D54
PACKET_FLOATS = 8
MAX_SOURCE_ROWS = 16384
TRACE_RECORDS_PER_WORKER = 1024
READY_WORDS_PER_EXPERT = 16
MAX_EARLY_EXPERTS_PER_RANK = 32
AUX_READY_BYTES = MAX_EARLY_EXPERTS_PER_RANK * READY_WORDS_PER_EXPERT * 4


def gradient_rows_per_batch() -> int:
    """Select the two-pass reference or a one-pass receive-side batch."""
    rows = os.environ.get("HMC_A3_RECOMPUTE_GRADIENT_ROWS", "0")
    if rows not in {"0", "1", "3"}:
        raise ValueError("HMC_A3_RECOMPUTE_GRADIENT_ROWS must be 0, 1, or 3")
    return int(rows)


def requested(specification) -> bool:
    """Include symmetric sidecars in the workspace contract for the supported geometry."""
    return (
        os.environ.get("HMC_A3_FUSED_RECOMPUTE") == "1"
        and specification["hot_replica"]
        and specification["precision"] == "bf16"
        and specification["dispatch_mode"] == "push"
        and not specification["apply_probs_before_fc2"]
        and (specification["hidden_size"], specification["intermediate_size"], specification["top_k"])
        == (4096, 1408, 8)
    )


@lru_cache(maxsize=1)
def require_vendor() -> None:
    """Reject an old operator package before submitting compound descriptors."""
    vendor = require_multicore_environment() / "op_api" / "lib" / "libcust_opapi.so"
    api = ctypes.CDLL(str(vendor), mode=ctypes.RTLD_LOCAL)
    version = getattr(api, "HmcA3RecomputeBackwardVersion", None)
    if version is None or version() != 5:
        raise RuntimeError("The loaded MegaMoE operator package does not support A3 fused recompute")
