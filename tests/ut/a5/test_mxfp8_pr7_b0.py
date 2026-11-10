# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Protect isolated PR7 polling and gradient-initialization experiments on A5."""

import ctypes
from dataclasses import replace
import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from hyper_multicore.modules.mega_moe.mxfp8_execution import MXFP8Execution
from hyper_multicore.modules.mega_moe.mxfp8_graph import build_mxfp8_runtime
from hyper_multicore.modules.mega_moe.mxfp8_module import _mx_b0_options
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.scheduler.config import EVENT_INVALID_ID
from hyper_multicore.scheduler.mxfp8 import MxGmmDescriptor
from hyper_multicore.scheduler.runtime import serialize_runtime_config


def _spec(**options: object) -> MegaMoeSpec:
    return replace(MegaMoeSpec(4096, 5120, 1792, 96, 8, 1.25, 40960, 16, None, 0, 32,
                              mx_vector_policy="window2", dispatch_split=4096,
                              combine_split=4096, compute_stages=6), **options)


@pytest.fixture(autouse=True)
def _environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {"PANGU_MX_GMM_TILE": "t6", "HYPER_PARALLEL_SHMEM_DATA_ENGINE": "mte_udma",
                        "HYPER_PARALLEL_SHMEM_UDMA_QP_NUM": "8", "PANGU_MX_REUSE_DISPATCH_WORKERS": "0",
                        "HYPER_PARALLEL_MX_B0_BACKWARD_DAG": "1"}.items():
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize("forward", [True, False])
def test_pr7_polling_preserves_every_dependency_and_math_descriptor(forward: bool) -> None:
    control = build_mxfp8_runtime(_spec(), forward=forward)
    candidate = build_mxfp8_runtime(_spec(mx_poll_policy="pr7"), forward=forward)
    assert candidate.gmm_descriptors == control.gmm_descriptors
    assert candidate.quant_descriptors == control.quant_descriptors
    changed = 0
    for before, after in zip(control.config.all_tasks, candidate.config.all_tasks):
        if after.extra_value_2 != before.extra_value_2:
            assert after.dependent_event != EVENT_INVALID_ID
            assert after.extra_value_2 == 5
            after.extra_value_2 = before.extra_value_2
            changed += 1
    assert changed > 0
    assert serialize_runtime_config(candidate.config) == serialize_runtime_config(control.config)


def test_cold_gradient_mode_only_changes_tn_initialization_flags() -> None:
    control = build_mxfp8_runtime(_spec(), forward=False)
    candidate = build_mxfp8_runtime(_spec(mx_skip_gradient_zero=True), forward=False)
    assert serialize_runtime_config(candidate.config) == serialize_runtime_config(control.config)
    assert candidate.quant_descriptors == control.quant_descriptors
    size = ctypes.sizeof(MxGmmDescriptor)
    changed = 0
    for offset in range(0, len(control.gmm_descriptors), size):
        before = MxGmmDescriptor.from_buffer_copy(control.gmm_descriptors[offset:offset + size])
        after = MxGmmDescriptor.from_buffer_copy(candidate.gmm_descriptors[offset:offset + size])
        if before.flags != after.flags:
            assert before.flags ^ after.flags == 2
            assert before.flags & 2 and not after.flags & 2
            after.flags = before.flags
            changed += 1
        assert bytes(after) == bytes(before)
    assert changed == 2 * 6 * 32
    assert (build_mxfp8_runtime(_spec(), forward=True).gmm_descriptors
            == build_mxfp8_runtime(_spec(mx_skip_gradient_zero=True), forward=True).gmm_descriptors)


def test_b0_uninitialized_outputs_retain_poison_until_tn_overwrites() -> None:
    plan = SimpleNamespace(spec=_spec(mx_skip_gradient_zero=True),
                           backward=SimpleNamespace(host=SimpleNamespace(kernel_gradients=False)))
    weights = tuple(SimpleNamespace(logical_shape=shape) for shape in ((6, 64, 128), (6, 128, 64)))
    with mock.patch("hyper_multicore.modules.mega_moe.mxfp8_execution.torch.empty",
                    side_effect=lambda shape, **kwargs: torch.full(shape, float("nan"), **kwargs)):
        outputs = MXFP8Execution._weight_gradient_buffers(plan, weights, "cpu")
    assert all(value.dtype == torch.float32 and torch.isnan(value).all() for value in outputs)


def test_cold_gradient_mode_rejects_replica_output_ownership() -> None:
    with pytest.raises(ValueError, match="B0"):
        build_mxfp8_runtime(_spec(mx_skip_gradient_zero=True, replica_slots_per_rank=2), forward=False)


@pytest.mark.parametrize("poll,clear", [("0", "0"), ("pr7", "0"), ("0", "1"), ("pr7", "1")])
def test_b0_options_are_independent_and_frozen_at_construction(poll: str, clear: str) -> None:
    with mock.patch.dict(os.environ, {"PANGU_MX_POLL_POLICY": poll, "PANGU_MX_B0_SKIP_GRAD_ZERO": clear}):
        assert _mx_b0_options(0) == (poll, clear == "1")


@pytest.mark.parametrize("name,value", [("PANGU_MX_POLL_POLICY", "5us"),
                                        ("PANGU_MX_B0_SKIP_GRAD_ZERO", "true")])
def test_invalid_options_fail_before_resource_creation(name: str, value: str) -> None:
    with mock.patch.dict(os.environ, {"PANGU_MX_POLL_POLICY": "0", "PANGU_MX_B0_SKIP_GRAD_ZERO": "0",
                                      name: value}):
        with pytest.raises(ValueError, match=name):
            _mx_b0_options(0)
