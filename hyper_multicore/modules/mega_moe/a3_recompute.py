# Copyright 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Capability and device ABI for the opt-in A3 recompute-backward launch."""

import ctypes
from functools import lru_cache
import os

from hyper_multicore._loader import require_multicore_environment

from .a3_indexed import MAX_INDEXED_TOKENS, MAX_INDEXED_TOP_K, supports_hidden_size

SEQUENCE_BIT = 1 << 40
CALL_MAGIC = 0x41335245434F4D35
PACKET_FLOATS = 8
MAX_SOURCE_ROWS = MAX_INDEXED_TOKENS
_A3_UB_BYTES = 192 * 1024
TRACE_RECORDS_PER_WORKER = 1024
READY_WORDS_PER_EXPERT = 16
MAX_EARLY_EXPERTS_PER_RANK = 32


def gradient_rows_per_batch(hidden_size: int) -> int:
    """Select the two-pass reference or a one-pass receive-side batch."""
    rows = os.environ.get("HMC_A3_RECOMPUTE_GRADIENT_ROWS", "0")
    if rows not in {"0", "1", "3"}:
        raise ValueError("HMC_A3_RECOMPUTE_GRADIENT_ROWS must be 0, 1, or 3")
    # Keep two BF16 planes, two FP32 planes and one FP32 probability packet per row.
    bytes_per_row = hidden_size * (2 * 2 + 2 * 4) + PACKET_FLOATS * 4
    return 1 if rows == "3" and 3 * bytes_per_row > _A3_UB_BYTES else int(rows)


def requested(specification) -> bool:
    """Include symmetric sidecars in the workspace contract for the supported geometry."""
    return (
        os.environ.get("HMC_A3_FUSED_RECOMPUTE") == "1"
        and specification["hot_replica"]
        and specification["precision"] == "bf16"
        and specification["dispatch_mode"] == "push"
        and not specification["apply_probs_before_fc2"]
        and supports_hidden_size(specification["hidden_size"])
        and 1 <= specification["top_k"] <= MAX_INDEXED_TOP_K
    )


def requested_on_a5(specification) -> bool:
    """Use the 950 BF16 provider with the same checkpoint and SHMEM contract."""
    return (
        os.environ.get("HMC_A5_FUSED_RECOMPUTE") == "1"
        and specification["hot_replica"] and specification["precision"] == "bf16"
        and specification["dispatch_mode"] == "push" and not specification["apply_probs_before_fc2"]
        and specification["top_k"] == 8
        and (specification["hidden_size"], specification["intermediate_size"]) in ((4096, 1408), (5120, 1792))
    )


@lru_cache(maxsize=1)
def require_vendor(architecture: int = 2201) -> None:
    """Reject an old operator package before submitting compound descriptors."""
    vendor = require_multicore_environment() / "op_api" / "lib" / "libcust_opapi.so"
    api = ctypes.CDLL(str(vendor), mode=ctypes.RTLD_LOCAL)
    symbol, abi = ("HmcA5RecomputeBackwardVersion", 1) if architecture == 3510 else ("HmcA3RecomputeBackwardVersion", 5)
    version = getattr(api, symbol, None)
    if version is None or version() != abi:
        raise RuntimeError(f"The loaded MegaMoE operator package is missing fused recompute ABI {symbol}={abi}")
