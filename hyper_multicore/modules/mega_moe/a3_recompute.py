# Copyright 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Capability and device ABI for the opt-in A3 recompute-backward launch."""

import ctypes
from functools import lru_cache
import os

from hyper_multicore._loader import require_multicore_environment

SEQUENCE_BIT = 1 << 40
CALL_MAGIC = 0x41335245434F4D50
PACKET_FLOATS = 8
MAX_SOURCE_ROWS = 16384


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
    if version is None or version() != 1:
        raise RuntimeError("The loaded MegaMoE operator package does not support A3 fused recompute")
