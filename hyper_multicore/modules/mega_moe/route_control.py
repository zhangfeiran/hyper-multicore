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
"""Owned pinned readback buffers with a completed event per workspace."""

import torch


class _ReplicaControlReadback:
    """Read small current control data without pageable staging allocations."""

    def __init__(self) -> None:
        """Allocate the completion event lazily with the workspace's active device."""
        self.buffers: dict[tuple, torch.Tensor] = {}
        self.ready = torch.npu.Event()

    def read(self, source: torch.Tensor) -> torch.Tensor:
        """Expose current CPU bytes only after this stream's pinned copy completes.

        Args:
            source: Control data ordered on the caller's current NPU stream.

        Returns:
            Owned pinned storage valid until the next same-shaped read.
        """
        key = tuple(source.shape), source.dtype
        host = self.buffers.get(key)
        if host is None:
            host = torch.empty(source.shape, dtype=source.dtype, device="cpu", pin_memory=True)
            self.buffers[key] = host
        host.copy_(source, non_blocking=True)
        self.ready.record(torch.npu.current_stream(source.device))
        self.ready.synchronize()
        return host
