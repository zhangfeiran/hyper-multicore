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
"""Order independent workspaces sharing the process-wide UDMA QP bank."""
import os
import threading
from typing import Any

from hyper_multicore.scheduler.config import RuntimeConfigC, TaskType
from hyper_multicore.shmem._runtime import configured_data_engine


def validate_udma_qp_owners(
    config: RuntimeConfigC, *, ep_size: int, num_experts: int, rank_id: int, qp_count: int,
) -> dict[tuple[int, int], int]:
    """Reject distinct physical workers that submit to the same peer/QP connection.

    The legacy odd-worker queue and every descriptor remain unchanged. Self copies
    and non-Put tasks do not use UDMA. Potentially empty remote tasks still count,
    since their payload size is determined only after the dynamic route exchange.

    Args:
        config: Final validated runtime image with the legacy odd-worker queue.
        ep_size: Number of ranks in the SHMEM root group.
        num_experts: Total experts, evenly partitioned across that group.
        rank_id: Calling rank in group-local coordinates.
        qp_count: Qualified number of connections per remote peer, either 16 or 32.

    Returns:
        Physical owner of every potentially active remote peer/QP pair.

    Raises:
        ValueError: The topology is unsupported or two workers alias a connection.
    """
    if qp_count not in (16, 32):
        raise ValueError("UDMA QP count must be 16 or 32")
    if config.num_workers != 64:
        raise ValueError("UDMA QP ownership requires the 64-worker Ascend950 plan")
    if ep_size <= 0 or not 0 <= rank_id < ep_size or num_experts <= 0 or num_experts % ep_size:
        raise ValueError("Invalid expert-parallel topology for UDMA QP ownership")
    owners = {}
    for position in range(config.task_index_num[1]):
        task_id = int(config.vector_task_indices[position])
        task = config.all_tasks[task_id]
        if task.task_type != TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
            continue
        if task.task_split_num <= 0 or task.task_split_num % num_experts:
            raise ValueError("UDMA Put task does not have a complete expert split")
        tasks_per_peer = task.task_split_num // ep_size
        peer = task.task_index // tasks_per_peer
        if not 0 <= peer < ep_size:
            raise ValueError("UDMA Put task peer is outside the EP group")
        if peer == rank_id:
            continue
        lane = position % 32
        worker = lane * 2 + 1
        key = (peer, lane if qp_count == 32 else lane // 2)
        previous = owners.setdefault(key, worker)
        if previous != worker:
            raise ValueError(f"UDMA peer/QP {key} has multiple physical owners: {previous} and {worker}")
    return owners


class SubmissionOrder:
    """Keep one host submitter and order device submissions across streams."""

    def __init__(self) -> None:
        """Initialize an unclaimed bank with no preceding device submission."""
        self.lock = threading.Lock()
        self.active = False
        self.stream = None
        self.completion = None

    def claim(self, stream: Any) -> None:
        """Order the next submission after the preceding workspace's stream."""
        with self.lock:
            if self.active:
                raise RuntimeError('UDMA QPs do not support concurrent workspace submissions')
            if self.completion is not None and self.stream != stream.npu_stream:
                stream.wait_event(self.completion)
            self.active = True

    def release(self, stream: Any, completion: Any) -> None:
        """Retain the completion event before admitting another workspace."""
        with self.lock:
            if not self.active:
                raise RuntimeError('UDMA QPs released without an active submission')
            self.stream = stream.npu_stream
            self.completion = completion
            self.active = False


def validate_transport() -> bool:
    """Reserve shared QP ordering only for the opt-in hybrid data engine."""
    engine = configured_data_engine()
    if engine == 'mte':
        return False
    if engine != 'mte_udma':
        raise ValueError('MegaMoE transport must use mte or mte_udma')
    return True


SUBMISSION_ORDER = SubmissionOrder()
