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
"""Independent distributed event machine with delayed Cube/AIV completions."""
from collections import defaultdict

from hyper_multicore.scheduler.config import EVENT_INVALID_ID, TaskType


def _initialise(configs, cores):
    """Build worker queues independently of the implementation transformation."""
    counters = [[0] * config.event_capacity for config in configs]
    done = [set() for _ in configs]
    readers, saves = [], []
    workers = []
    for rank, config in enumerate(configs):
        first, copies = defaultdict(set), defaultdict(set)
        cube = list(config.cube_task_indices[:config.task_index_num[0]])
        vector = list(config.vector_task_indices[:config.task_index_num[1]])
        for index in cube:
            task = config.all_tasks[index]
            if task.inputs[0].input_position == 0:
                first[task.task_index // cores].add(index)
        for index in vector:
            task = config.all_tasks[index]
            if task.task_type == TaskType.TASK_SAVE_DISPATCH:
                copies[task.task_index // (2 * cores)].add(index)
        readers.append(first)
        saves.append(copies)
        workers.extend([rank, False, lane, cube[lane::cores], None] for lane in range(cores))
        count = 2 * cores if config.vector_ready_queue else cores
        workers.extend([rank, bool(config.vector_ready_queue), lane, vector[lane::count], None]
                       for lane in range(count))
    return counters, done, readers, saves, workers


def simulate(configs: list, ep: int, local: int, cores: int) -> None:
    """Check actual worker queues, bounded bypass and complete X-reader fences."""
    counters, done, readers, saves, workers = _initialise(configs, cores)
    remaining = sum(len(worker[3]) for worker in workers)
    while remaining:
        progress = False
        for worker in workers:
            finished, advanced = _advance(worker, configs, counters, done, readers, saves, (ep, local, cores))
            remaining -= finished
            progress |= advanced
        if not progress:
            raise AssertionError(f"Distributed saved-dispatch schedule stalled with {remaining} tasks")


def _advance(worker, configs, counters, done, readers, saves, topology):
    """Delay task completion so queue order alone cannot masquerade as a fence."""
    rank, bypass, lane, queue, active = worker
    ep, local, cores = topology
    config = configs[rank]
    finished, progress = 0, False
    if active is not None:
        index, delay = active
        if delay:
            worker[4] = (index, delay - 1)
            return 0, True
        task = config.all_tasks[index]
        target = rank
        if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL:
            target = task.task_index // (local * (task.task_split_num // (ep * local)))
        counters[target][task.trigger_event] += 1
        done[rank].add(index)
        finished, progress, worker[4] = 1, True, None
    for position, index in enumerate(queue[:256] if bypass else queue[:1]):
        task = config.all_tasks[index]
        event = task.dependent_event
        if event != EVENT_INVALID_ID and counters[rank][event] < config.all_event_num_triggers[event]:
            continue
        _check_readers(task, rank, cores, local, readers, saves, done)
        queue.pop(position)
        delay = 7 if task.task_type == TaskType.TASK_GROUPED_MATMUL and lane else 1
        worker[4] = (index, delay)
        return finished, True
    return finished, progress


def _check_readers(task, rank, cores, local, readers, saves, done):
    if task.task_type == TaskType.TASK_SWI_GLU:
        expert = task.task_index // (task.task_split_num // local)
    elif task.task_type == TaskType.TASK_GROUPED_MATMUL and task.inputs[0].input_position == 8:
        expert = task.task_index // cores
    else:
        return
    pending = (readers[rank][expert] | saves[rank][expert]) - done[rank]
    if pending:
        raise AssertionError(f"X readers/save still active: rank={rank}, expert={expert}, pending={pending}")
