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
"""Apply an optional device event refresh policy during cached graph preparation."""

import os

from hyper_multicore.scheduler.config import RuntimeConfigC

__all__ = []


def event_refresh_intervals() -> tuple[int, int]:
    """Parse private Cube/Vector refresh microseconds; zero retains native defaults."""
    text = os.environ.get("HYPER_MULTICORE_MOE_EVENT_REFRESH_US", "0:0")
    parts = text.split(":")
    if len(parts) != 2 or any(not part.isascii() or not part.isdecimal() for part in parts):
        raise ValueError("HYPER_MULTICORE_MOE_EVENT_REFRESH_US must be two nonnegative integers: Cube:Vector")
    cube, vector = map(int, parts)
    if max(cube, vector) > 10000:
        raise ValueError("HYPER_MULTICORE_MOE_EVENT_REFRESH_US intervals must not exceed 10000 microseconds")
    return cube, vector


def apply_event_refresh(config: RuntimeConfigC, cube_us: int, vector_us: int) -> None:
    """Set existing dependency poll fields without changing tasks, producers or order.

    Args:
        config: Prepared forward or backward runtime image.
        cube_us: Cube refresh microseconds, or zero to retain each task's setting.
        vector_us: Vector refresh microseconds, or zero to retain each task's setting.
    """
    for kind, (queue, interval) in enumerate(
        ((config.cube_task_indices, cube_us), (config.vector_task_indices, vector_us))
    ):
        if interval:
            for task_id in queue[: config.task_index_num[kind]]:
                config.all_tasks[task_id].extra_value_2 = interval
