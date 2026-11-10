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
"""Bounded MX-only UDMA gradient banks inside the unused BF16 weight prefix."""

import os
from typing import Any

import torch

from hyper_multicore.scheduler.window_queue import _validate_qp_owners
from hyper_multicore.shmem._runtime import configured_udma_qp_count

from .kernel_gradients import _UdmaGradientScratch

_HELPERS = 4
_BANKS_PER_HELPER = 2
_MAX_CHUNK_BYTES = 2 * 1024**2
_BYTE_ALIGNMENT = 32


def mx_udma_gradient_enabled(slots: int, transport: str) -> bool:
    """Validate the private opt-in before any execution resources exist.

    Args:
        slots: Per-rank physical replica budget.
        transport: Selected MX replica return protocol.
    """
    option = os.environ.get("HYPER_PARALLEL_MX_GRADIENT_UDMA", "0")
    if option not in ("0", "1"):
        raise ValueError("HYPER_PARALLEL_MX_GRADIENT_UDMA must be 0 or 1")
    enabled = option == "1"
    if enabled and slots and transport != "shmem_signal_kernel_gradient":
        raise ValueError("MX UDMA gradients require the symmetric FP32 gradient inbox")
    return enabled


def mx_gradient_peer_qps(config: Any, spec: Any) -> tuple[tuple[int, ...], ...]:
    """Reserve four unowned queues per peer without changing any token producer.

    Args:
        config: Final backward CSR task queues, including dispatch and combine.
        spec: Physical expert topology owning those queues.
    """
    qps = configured_udma_qp_count()
    vectors = [list(config.vector_task_indices[config.mix_task_indices[worker]:config.mix_task_indices[worker + 1]])
               for worker in range(64)]
    owners = _validate_qp_owners(config, spec, vectors, qps)
    rows = []
    for peer in range(spec.ep_size):
        # Keep QP0 available to generic control/signal operations on UDMA-only links.
        available = tuple(qp for qp in range(1, qps) if (peer, qp) not in owners)
        if len(available) < _HELPERS:
            raise ValueError(f"Insufficient unowned gradient QPs: peer={peer}, free={len(available)}")
        rows.append(available[:_HELPERS])
    return tuple(rows)


def mx_udma_gradient_scratch(provider: Any) -> _UdmaGradientScratch:
    """Borrow scratch without changing weight generations or FP32 inbox storage.

    Args:
        provider: Bound MX signal provider with unused BF16 weight prefixes.

    Returns:
        Four independent helper/QP pairs and two aligned banks per helper.
    """
    cached = getattr(provider, "_mx_udma_gradient_scratch", None)
    if cached is not None:
        return cached
    state = provider.runtime.debug_state()
    config = state.get("config") or {}
    if (state.get("backend_soc_type") != "Ascend950" or config.get("udma_qp_num") not in (8, 16, 32)
            or config.get("data_engine") != "mte_udma"):
        raise ValueError("MX UDMA gradients require an Ascend950 MTE/UDMA runtime with 8, 16 or 32 QPs")
    if provider.pool is None or not provider.pool.weights:
        raise ValueError("MX UDMA gradient scratch requires a bound signal provider")
    prefix = provider.pool.weights[0].view(torch.uint8).reshape(-1)
    chunk_bytes = min(_MAX_CHUNK_BYTES, prefix.numel() // (_HELPERS * _BANKS_PER_HELPER))
    chunk_bytes = chunk_bytes // _BYTE_ALIGNMENT * _BYTE_ALIGNMENT
    if chunk_bytes < _BYTE_ALIGNMENT:
        raise ValueError("The MX weight prefix cannot hold aligned helper ping-pong banks")
    scratch = prefix[:_HELPERS * _BANKS_PER_HELPER * chunk_bytes].view(torch.float32)
    layout = _UdmaGradientScratch(scratch, chunk_bytes // 4, _HELPERS, 16)
    provider._mx_udma_gradient_scratch = layout
    return layout
