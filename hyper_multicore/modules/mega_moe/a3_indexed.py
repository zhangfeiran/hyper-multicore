# Copyright 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""A3 BF16 indexed dispatch and the count exchange's readiness contract."""

from __future__ import annotations

import ctypes
from functools import lru_cache
import os

import torch

from hyper_multicore import _collective
from hyper_multicore._loader import require_multicore_environment

from .count_fence import COUNT_FENCE_BIT as COUNT_FENCE_BIT, CountFence as CountFence



MAX_INDEXED_TOKENS = 16384
MAX_INDEXED_EXPERTS = 512
MAX_INDEXED_TOP_K = 8
MAX_INDEXED_HIDDEN = 8192
INDEXED_HIDDEN_ALIGNMENT = 16
REPLAY_FENCE_BIT = 1 << 7
_SOURCE_BITS = 21
_TOP_K_SHIFT = 17
_FENCED = 1 << 16


def requested() -> bool:
    """Capture the opt-in as part of the execution group's configuration."""
    return os.environ.get("HMC_A3_INDEXED_PUSH") == "1"


@lru_cache(maxsize=1)
def _ordered_operator():
    from hyper_multicore.modules.hot_replica.counts import _operators  # pylint: disable=import-outside-toplevel

    operators = _operators()
    if operators is None:
        return None
    namespace, library = operators
    available = getattr(namespace, "a3_ordered_replica_route_available", None)
    if available is None or not available(library):
        return None
    vendor = require_multicore_environment() / "op_api" / "lib" / "libcust_opapi.so"
    api = ctypes.CDLL(str(vendor), mode=ctypes.RTLD_LOCAL)
    version = getattr(api, "HmcA3IndexedPushVersion", None)
    if version is None or version() != 2:
        return None
    return namespace.a3_ordered_replica_route, library


def qualify(spec: MegaMoeSpec, enabled: bool, rank_nodes: tuple[int, ...]) -> bool:
    """Agree on the supported geometry and bundled code before preparing routes."""
    if (not enabled or len(rank_nodes) != spec.ep_size or len(set(rank_nodes)) != 1
            or spec.architecture != 2201 or spec.precision != "bf16"
            or not spec.auto_tokens or spec.dispatch_mode != "push" or spec.transport_engine != "mte"
            or spec.apply_probs_before_fc2 or spec.batch_experts
            or not 1 <= spec.top_k <= min(MAX_INDEXED_TOP_K, spec.num_experts)
            or spec.num_experts > MAX_INDEXED_EXPERTS
            or not supports_hidden_size(spec.hidden_size)):
        return False
    local_error = None
    try:
        supported = _ordered_operator() is not None
    except Exception as error:  # pylint: disable=broad-exception-caught
        local_error = error
        supported = False
    failure = None if local_error is None else _collective.failure_report(local_error)
    reports = _collective.exchange((supported, failure), group=spec.ep_group,
                                   failure_prefix="A3 indexed dispatch capability")
    support, failures = zip(*reports)
    if any(failures):
        raise RuntimeError(f"A3 indexed dispatch capability failed on "
                           f"{_collective.format_collective_failures(failures)}") from local_error
    return all(support)


def supports_hidden_size(hidden_size: int) -> bool:
    """Check the A3 BF16 row alignment and indexed-transfer UB budget."""
    return 0 < hidden_size <= MAX_INDEXED_HIDDEN and hidden_size % INDEXED_HIDDEN_ALIGNMENT == 0


def ordered_count(ids: torch.Tensor, num_experts: int, mask: int):
    """Return the validated local wire, stable inverse and stable source rows."""
    operator, library = _ordered_operator()
    return operator(ids, num_experts, mask, library)


def note_source_ready(ids: torch.Tensor) -> None:
    """Mark final Router IDs on their producer stream before expert weights materialize."""
    if (requested() and ids.device.type == "npu" and ids.ndim == 2
            and 1 <= ids.shape[1] <= MAX_INDEXED_TOP_K
            and 0 < ids.shape[0] <= MAX_INDEXED_TOKENS
            and not ids.is_inference() and not hasattr(ids, "_hmc_indexed_source_ready")):
        ready = torch.npu.Event()
        ready.record(torch.npu.current_stream(ids.device))
        ids._hmc_indexed_source_ready = (ids._version, ready)


def take_source_ready(ids: torch.Tensor):
    """Consume the event only for the original, unmodified tensor."""
    marker = getattr(ids, "_hmc_indexed_source_ready", None)
    if marker is not None:
        del ids._hmc_indexed_source_ready
        version, ready = marker
        if version == ids._version:
            return ready
    return None


def pack_dispatch_offsets(offsets: torch.Tensor, source_rows: torch.Tensor) -> torch.Tensor:
    """Append the int32 source rows after the ordinary int64 offsets."""
    if source_rows.numel() % 2:
        source_rows = torch.cat((source_rows, source_rows.new_zeros(1)))
    return torch.cat((offsets, source_rows.view(torch.int64)))


def launch_sequence(plan_tokens: int, source_rows: int, top_k: int, fenced: bool) -> int:
    """Encode the source extent, Top-K and readiness in the private launch attribute."""
    return -((plan_tokens << _SOURCE_BITS) | (top_k << _TOP_K_SHIFT) | source_rows | (_FENCED if fenced else 0))

