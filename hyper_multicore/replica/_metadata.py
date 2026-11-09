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
"""Owned asynchronous uploads for replica protocol descriptors."""

import torch


def _owned_host_bytes(data: bytes, device: torch.device) -> torch.Tensor:
    """Keep asynchronous sources alive through the pinned allocator's copy event."""
    source = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    if device.type != "cpu":
        source = source.pin_memory(device=device.type)
    return source


def _async_device_bytes(data: bytes, device: torch.device) -> torch.Tensor:
    """Upload owned pinned bytes without draining earlier work on the device stream."""
    return _owned_host_bytes(data, device).to(device, non_blocking=True)


def _copy_device_bytes(data: bytes, destination: torch.Tensor) -> None:
    """Update a leased device suffix directly without allocating a device intermediate."""
    destination.copy_(_owned_host_bytes(data, destination.device), non_blocking=True)
