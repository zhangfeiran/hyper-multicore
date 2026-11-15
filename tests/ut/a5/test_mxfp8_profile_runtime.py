# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""Protect invocation-owned MX gradient metadata at the profiler boundary."""

import struct
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from hyper_multicore.modules.mega_moe.mxfp8_diagnostics import _bind_mx_profile_runtime
from hyper_multicore.modules.mega_moe.mxfp8_graph import build_mxfp8_runtime
from hyper_multicore.modules.mega_moe.mxfp8_profile_graph import bind_mx_profile_graph
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.diagnostics.common.graph import validate_mega_kernel_diagnostic_contract


def test_profile_retains_exact_current_gradient_envelope():
    """Distinct backward invocations must preserve their own destinations and epoch metadata."""
    graph = torch.arange(128, dtype=torch.uint8)
    profile = graph.clone()
    profile[20] = 1
    for pointer in (0x123400, 0xabcdef00):
        envelope = struct.pack("<II9Q", 0x53505754, 5, 6, 0, 0, pointer, pointer + 64,
                               0x200000, 0x200080, 1, pointer + 128)
        suffix = torch.tensor(list(envelope), dtype=torch.uint8)
        invocation = torch.cat((graph, suffix))
        runtime = SimpleNamespace(runtime=invocation, host=SimpleNamespace(kernel_gradients=True))
        actual = _bind_mx_profile_runtime(runtime, profile)
        assert torch.equal(actual[:128], profile)
        assert bytes(actual[128:].tolist()) == envelope
        assert torch.equal(invocation[:128], graph)
    assert profile.numel() == 128


def test_forward_reuses_profile_image_and_bad_envelopes_fail_before_launch():
    """No envelope is allocated for forward; malformed backward images cannot reach the device guard."""
    profile = torch.zeros(128, dtype=torch.uint8)
    runtime = SimpleNamespace(runtime=profile.clone(), host=SimpleNamespace(kernel_gradients=False))
    assert _bind_mx_profile_runtime(runtime, profile) is profile
    runtime.host.kernel_gradients = True
    with pytest.raises(ValueError, match="gradient envelope"):
        _bind_mx_profile_runtime(runtime, profile)
    runtime.runtime = torch.zeros(192, dtype=torch.uint8)
    with pytest.raises(ValueError, match="gradient envelope"):
        _bind_mx_profile_runtime(runtime, profile)


@pytest.mark.parametrize("forward", [True, False])
def test_guest_profile_ports_preserve_native_graph_and_reject_unknown_buffers(monkeypatch, forward):
    """Real B2 graphs keep task IDs and reject unrecognized physical weight operands."""
    monkeypatch.setenv("HYPER_PARALLEL_SHMEM_DATA_ENGINE", "mte_udma")
    monkeypatch.setenv("HYPER_PARALLEL_SHMEM_UDMA_QP_NUM", "16")
    monkeypatch.setenv("PANGU_MX_GMM_TILE", "t6")
    spec = MegaMoeSpec(4096, 5120, 1792, 16 * 6, 8, 1.25, 40960, 16, None, 15, 32,
                       mx_vector_policy="window2", dispatch_split=4096, combine_split=4096,
                       compute_stages=6)
    spec = replace(spec, replica_slots_per_rank=2, replica_transport="shmem_signal_kernel_gradient")
    host = build_mxfp8_runtime(spec, forward=forward, weight_prefetch=True)
    before = bytes(host.config)
    original = [(op.name, tuple(op.param_positions), op.task_num) for op in host.graph.topological_sort()]
    graph = bind_mx_profile_graph(host.graph, host.config)
    validate_mega_kernel_diagnostic_contract(graph, host.config, 68)
    assert bytes(host.config) == before
    assert [(op.name, tuple(op.param_positions), op.task_num) for op in host.graph.topological_sort()] == original
    assert sum(op.task_num for op in graph.topological_sort()) == sum(row[2] for row in original)
    guest = next(task for task in host.config.all_tasks
                 if task.task_type == 160 and task.inputs[1].input_position in (27, 29, 33, 35))
    guest.inputs[1].input_position = 63
    with pytest.raises(ValueError, match="unexpected operand substitution"):
        bind_mx_profile_graph(host.graph, host.config)
