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
"""Immutable EP-rank node identities independent of rank numbering and device count."""

import socket
from collections.abc import Sequence
from pathlib import Path


def _local_node_identity() -> str:
    """Prefer a kernel boot identity shared by containers on the same machine."""
    try:
        identity = Path('/proc/sys/kernel/random/boot_id').read_text(encoding='utf-8').strip()
    except OSError:
        identity = ''
    return identity or socket.gethostname()


def _node_ids(identities: Sequence[str]) -> tuple[int, ...]:
    """Assign node labels in EP-rank order without assuming contiguous rank groups."""
    labels = {}
    result = []
    for identity in identities:
        if not isinstance(identity, str) or not identity:
            raise ValueError('EP topology requires a nonempty node identity for every EP rank')
        if identity not in labels:
            labels[identity] = len(labels)
        result.append(labels[identity])
    return tuple(result)
