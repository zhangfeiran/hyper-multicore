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
"""Generate the measured two-expert window from the current task graph."""

from __future__ import annotations
import os
from typing import Any

from hyper_multicore.shmem._runtime import configured_udma_qp_count

from .config import TaskType
from .planned_queue import install_worker_queues


def _reuse_post_dispatch_math(
    cfg: Any, vectors: list[list[int]], combine_start: int, first_stage: tuple[int, int], *, active_replicas: bool,
) -> None:
    """Borrow drained lanes for all (1), input (2), or active-replica-only (3) quantization."""
    enabled = os.environ.get("PANGU_MX_REUSE_DISPATCH_WORKERS", "0")
    if enabled not in ("0", "1", "2", "3"):
        raise ValueError("PANGU_MX_REUSE_DISPATCH_WORKERS must be exactly 0, 1, 2 or 3")
    if enabled == "3":
        enabled = str(int(active_replicas))
    if enabled == "0":
        return
    for lane in range(32):
        math, communication = vectors[2 * lane], vectors[2 * lane + 1]
        if len(math) % 2:
            raise ValueError("Post-dispatch reuse requires paired logical quantization workers")
        boundary = next((index for index, task in enumerate(communication)
                         if task >= combine_start or cfg.all_tasks[task].task_type == 0), len(communication))
        # Preserving the full communication subsequence keeps every peer/QP owner.
        # Borrowing a lane before dispatch drains could introduce remote wait cycles.
        borrowed = math[1::2]
        if enabled == "2":
            start, count = first_stage
            borrowed = [task for task in borrowed if start <= task < start + count]
        moved = set(borrowed)
        vectors[2 * lane] = [task for task in math if task not in moved]
        vectors[2 * lane + 1] = communication[:boundary] + borrowed + communication[boundary:]


def _validate_qp_owners(cfg: Any, spec: Any, vectors: list[list[int]], qps: int) -> dict[tuple[int, int], int]:
    """Reject concurrent producers of one peer/QP without changing task order."""
    owners = {}
    for worker in range(1, 64, 2):
        qp = worker // (64 // qps)
        for tid in vectors[worker]:
            task = cfg.all_tasks[tid]
            if task.task_type != TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
                continue
            peer = task.task_index // (task.task_split_num // spec.ep_size)
            if peer == spec.rank_id:
                continue
            key = peer, qp
            owner = owners.setdefault(key, worker)
            if owner != worker:
                raise ValueError(f"QP owner collision: peer={peer}, qp={qp}, workers={owner}/{worker}")
    return owners


def _window_stages(graph: Any, experts: int) -> dict[str, tuple[int, int]]:
    """Locate expert-partitioned stages without changing their topological task IDs."""
    stages = {}
    offset = 0
    for op in graph.topological_sort():
        if op.name != "terminate" and op.task_num % experts:
            raise ValueError(f"Window stage {op.name} cannot partition {op.task_num} tasks across experts")
        stages[op.name] = (offset, op.task_num)
        offset += op.task_num
    return stages


def _window_cube_stages(
    cfg: Any, stages: dict[str, tuple[int, int]], *, forward: bool, activation_first: bool,
) -> tuple[str, ...]:
    """Prioritize activations only after their independent input events are wired."""
    if not activation_first:
        return ("up", "down") if forward else ("dw2", "da", "dw1", "dx")
    if forward:
        raise ValueError("Activation-first window queues require a backward graph")
    for target, source in (("da", "dw2"), ("dx", "dw1")):
        target_start, count = stages[target]
        source_start, source_count = stages[source]
        if count != source_count or any(
            cfg.all_tasks[target_start + lane].dependent_event != cfg.all_tasks[source_start + lane].dependent_event
            for lane in range(count)
        ):
            raise ValueError("Activation-first window queues require independent gradient dependencies")
    return ("da", "dw2", "dx", "dw1")


def configure_window_queue(
    cfg: Any, graph: Any, spec: Any, *, forward: bool, activation_first: bool = False,
) -> None:
    """Preserve tasks, dependencies and QP ownership without shape-specific files.

    Args:
        cfg: Mutable host runtime image being assembled.
        graph: Completed task graph with operator-local task ranges.
        spec: Validated physical expert topology and current capacity.
        forward: Whether this is a forward graph.
        activation_first: Prioritize independently ready activation gradients over weight gradients.
    """
    stages = _window_stages(graph, spec.local_experts)
    cube_stages = _window_cube_stages(cfg, stages, forward=forward, activation_first=activation_first)

    def tasks(stage: str, expert: int, lane: int) -> list[int]:
        """Select the task IDs owned by one expert and physical compute lane.

        Args:
            stage: Stage name selected for this expert window.
            expert: Local physical expert slot.
            lane: Physical math worker lane.
        """
        start, count = stages[stage]
        width = count // spec.local_experts
        return list(range(start + expert * width + lane, start + (expert + 1) * width, 32))

    inputs = ("quant_x", "swiglu_quant") if forward else ("quant_dy", "swiglu_grad_quant")
    front = 1 if forward else 2
    vectors = [[] for _ in range(64)]
    cubes = [[] for _ in range(32)]
    communication = list(cfg.vector_task_indices[: cfg.task_index_num[3]])
    for lane in range(32):
        for stage in cube_stages[:front]:
            cubes[lane].extend(tasks(stage, 0, lane))
        for expert in range(1, spec.local_experts):
            for stage in cube_stages[:front]:
                cubes[lane].extend(tasks(stage, expert, lane))
            for stage in cube_stages[front:]:
                cubes[lane].extend(tasks(stage, expert - 1, lane))
        for stage in cube_stages[front:]:
            cubes[lane].extend(tasks(stage, spec.local_experts - 1, lane))
        vectors[2 * lane + 1] = communication[lane::32]
        vectors[2 * lane].extend(tasks(inputs[0], 0, lane))
        for expert in range(spec.local_experts):
            if expert + 1 < spec.local_experts:
                vectors[2 * lane].extend(tasks(inputs[0], expert + 1, lane))
            vectors[2 * lane].extend(tasks(inputs[1], expert, lane))
    _reuse_post_dispatch_math(cfg, vectors, stages["combine"][0], stages[inputs[0]],
                             active_replicas=bool(spec.replica_slots_per_rank))
    qps = configured_udma_qp_count()
    if qps < 32:
        _validate_qp_owners(cfg, spec, vectors, qps)
    install_worker_queues(cfg, cubes, vectors)
