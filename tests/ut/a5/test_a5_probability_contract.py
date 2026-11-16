# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0.
"""A5 completion credits must not discard the probability transport ABI."""

from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.modules.mega_moe.task_options import validate_task_options
from hyper_multicore.scheduler.config import TaskType


def test_a5_backward_completion_flag_preserves_probability_payload():
    """The 950 completion bit coexists with the target probability-gradient sidecar."""
    spec = MegaMoeSpec(256, 128, 512, 12, 2, 1.25, 640, 4, None, 0, 32,
                       dispatch_mode="push", apply_probs_before_fc2=True)
    _, _, _, backward = _build_runtime_artifacts(spec)
    puts = [task for task in backward.all_tasks[:backward.task_num]
            if task.task_type == TaskType.TASK_SHMEM_PUT_MEM_SIGNAL]
    assert puts
    weighted = [task for task in puts if task.num_outputs == 2]
    assert weighted
    assert {task.extra_value_1 & 12 for task in weighted} == {4, 8}
    for task in weighted:
        assert task.extra_value_1 & 1
        validate_task_options(task)
