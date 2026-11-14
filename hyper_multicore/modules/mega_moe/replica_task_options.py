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
"""Replica gradient producer events extending the ordinary GMM task contract."""
from hyper_multicore.scheduler.config import TaskDescC, TaskType
from .task_options import validate_task_options


def validate_replica_task_options(task: TaskDescC) -> None:
    """Validate the legacy auxiliary event only on transposed weight-gradient tasks."""
    if task.task_type == TaskType.TASK_GROUPED_MATMUL and task.extra_value_1 > 3:
        if (task.num_outputs != 1 or task.outputs[0].input_position not in (6, 18)
                or task.inputs[0].transpose_flag != 1):
            raise ValueError("replica producer event requires a weight-gradient task")
        task = TaskDescC.from_buffer_copy(bytes(task))
        task.extra_value_1 = 0
    validate_task_options(task)
