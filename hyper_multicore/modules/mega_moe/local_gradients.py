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
"""Select direct BF16 home outputs with a proven native output protocol."""

from typing import Any

import torch

from hyper_multicore._loader import NativeComponentUnavailableError
from hyper_multicore.torch import ops


def resolve_local_gradient_bf16(requested: bool | None) -> bool:
    """Resolve automatic output support once per plan, keeping explicit requests fail-closed.

    Args:
        requested: None negotiates the payload capability; False disables it;
            True requires the exact supported output protocol.

    Returns:
        Whether this plan can use the proven local output protocol.
    """
    if requested is not None and not isinstance(requested, bool):
        raise ValueError("Local gradient mode must be a boolean or None for automatic selection")
    if requested is False:
        return False
    ops._load_native()  # pylint: disable=protected-access
    version = getattr(torch.ops.hyper_multicore, "mega_moe_local_gradient_version", None)
    supported = version is not None and version() == 1
    if requested is True and not supported:
        raise NativeComponentUnavailableError("Local BF16 gradients require a matching rebuilt multicore payload")
    return supported


def require_local_gradient_native() -> None:
    """Reject an explicitly requested output protocol on an older native payload."""
    resolve_local_gradient_bf16(True)


def use_local_gradient_bf16(plan: Any, route: Any, provider: Any, weights: tuple[torch.Tensor, ...]) -> bool:
    """Prove the same two-projection band path before selecting either home output dtype.

    Args:
        plan: Bound execution plan with an explicitly enabled native capability.
        route: Current agreed placement; only its rank-local ownership is inspected.
        provider: Transport holding the active backward lease.
        weights: Both saved home weights, or both proposed gradient buffers.

    Returns:
        True for two BF16 home outputs without owned edges, or with final-output support.
        Missing events or band storage retain the ordinary FP32 fallback.
    """
    if not _local_gradient_path_ready(plan, route, provider):
        return False
    if (len(weights) != 2 or any(weight is None or weight.dtype != torch.bfloat16 or weight.ndim != 3
                                 for weight in weights)
            or (not getattr(plan.spec, "_final_gradient_bf16", False)
                and any(edge.owner_rank == route.rank for edge in route.plan.transfers))):
        return False
    return all(provider.kernel_gradient_band_signals(matrix, plan.spec.num_cube_cores) is not None
               for matrix in (1, 0))


def _local_gradient_path_ready(plan: Any, route: Any, provider: Any) -> bool:
    """Require both actual producer projections and the provider band contract."""
    if (not getattr(getattr(plan, "spec", None), "_local_gradient_bf16", False)
            or route is None or not route.plan.transfers
            or not plan.replica_w2_events or not plan.replica_w13_events
            or not getattr(provider, "kernel_gradients", False) or not getattr(provider, "gradient_bands", False)):
        return False
    return True
