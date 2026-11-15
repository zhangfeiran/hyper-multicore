# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Describe MX home and guest task operands without changing device queues."""

from copy import copy, deepcopy
from itertools import groupby
from typing import Any

from hyper_multicore.diagnostics.common.graph import operator_diagnostic_name
from hyper_multicore.scheduler.graph import ComputeGraph, OperatorNode


def _ports(task: Any) -> tuple[tuple[int, int], ...]:
    return tuple((int(item.input_position), int(item.data_type))
                 for item in list(task.inputs)[:task.num_inputs] + list(task.outputs)[:task.num_outputs])


def bind_mx_profile_graph(graph: ComputeGraph, config: Any) -> ComputeGraph:
    """Bind contiguous ABI groups on a diagnostic copy with the original task IDs.

    MX guest GMM tasks use separate physical weight buffers inside one scheduled
    stage. The diagnostic graph describes each homogeneous range separately;
    actual task descriptors, events, queues and their execution order are untouched.
    Tensor dumps remain unsupported by the MX capture entrypoint.
    """
    result = ComputeGraph()
    previous = None
    offset = 0
    substitutions = {("up", 5): 27, ("down", 9): 29, ("da", 7): 35, ("dx", 11): 33}
    for operator in graph.topological_sort():
        groups = [(list(ids), ports) for ports, ids in groupby(
            range(offset, offset + operator.task_num), key=lambda index: _ports(config.all_tasks[index]))]
        if not groups:
            groups = [([], tuple((spec.param_position, spec.dtype_size)
                                 for spec in operator.inputs + operator.outputs))]
        for group_index, (task_ids, ports) in enumerate(groups):
            for task_id in task_ids:
                task = config.all_tasks[task_id]
                if (int(task.num_inputs), int(task.num_outputs)) != (len(operator.inputs), len(operator.outputs)):
                    raise ValueError(f"MX profile stage {operator.name!r} has inconsistent operand counts")
            node = copy(operator)
            node.name = operator.name + (f"__mx_ports_{group_index}" if len(groups) > 1 else "")
            node.diagnostic_name = operator_diagnostic_name(operator)
            node.predecessors, node.successors = [], []
            node.task_num = len(task_ids)
            # Owner labels must retain the original complete stage's worker count.
            node._mx_profile_source_operator = operator
            specifications = []
            for original, (position, element_bytes) in zip(operator.inputs + operator.outputs, ports):
                if element_bytes != original.dtype_size:
                    raise ValueError(f"MX profile stage {operator.name!r} changes an operand element size")
                if (position != original.param_position
                        and position != substitutions.get((operator.name, original.param_position))):
                    raise ValueError(f"MX profile stage {operator.name!r} has an unexpected operand substitution")
                specification = deepcopy(original)
                specification.param_position = position
                specification.name = f"buffer_{position}"
                specifications.append(specification)
            node.inputs = specifications[:len(operator.inputs)]
            node.outputs = specifications[len(operator.inputs):]
            node.param_positions = [spec.param_position for spec in specifications]
            result.add_op(node)
            # This chain describes immutable task-ID ranges, not device scheduling.
            if previous is not None:
                result.add_edge(previous, node)
            previous = node
        offset += operator.task_num
    return result
