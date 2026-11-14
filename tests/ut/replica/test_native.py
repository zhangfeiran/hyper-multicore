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
"""CPU numerical contracts for native replica FP32 clamped SwiGLU."""

import unittest
from contextlib import nullcontext
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional

from hyper_multicore.replica import build_expert_replica_plan, native
from hyper_multicore.replica.routing import ReplicaRoute


def _reference(up: torch.Tensor, limit: float) -> torch.Tensor:
    """Use ordinary autograd as the independent activation oracle."""
    gate, value = up.float().chunk(2, dim=-1)
    if limit > 0:
        gate, value = (gate.clamp(max=limit), value.clamp(-limit, limit))
    return (functional.silu(gate) * value).to(up.dtype)


def _cpu_gmm(inputs: torch.Tensor, weights: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """Replace only the hardware grouped matmul with CPU matrix products."""
    parts, start = ([], 0)
    for index, end in enumerate(groups.tolist()):
        parts.append(inputs[start:end] @ weights[index])
        start = end
    return torch.cat(parts)


class TestNativeReplicaSwiGLU(unittest.TestCase):
    """Preserve clamp boundaries, empty ranks and both home/guest gradients."""

    def test_activation_and_gradient_match_autograd_at_clamp_boundaries(self) -> None:
        """The restored explicit backward must include both clamp endpoints."""
        for dtype, limit in product((torch.float32, torch.bfloat16), (-1.0, 0.0, 1.5, 10.0)):
            with self.subTest(dtype=dtype, limit=limit):
                up = torch.tensor([[-20.0, -1.5, 1.5, 20.0, -20.0, -1.5, 1.5, 20.0]], dtype=dtype)
                expected_up = up.clone().requires_grad_()
                expected = _reference(expected_up, limit)
                gradient = torch.tensor([[0.5, -0.75, 1.25, -1.5]], dtype=dtype)
                (expected * gradient).sum().backward()
                torch.testing.assert_close(native._replica_swiglu(up, limit), expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    native._replica_swiglu_backward(gradient, up, limit), expected_up.grad, rtol=1e-05, atol=1e-06
                )

    def test_invalid_limits_fail_before_native_execution(self) -> None:
        """Invalid metadata must fail before launching autograd or device operators."""
        values = SimpleNamespace(device=SimpleNamespace(type="npu"), dtype=torch.bfloat16)
        for limit in (True, float("nan"), float("inf"), "1.5"):
            with self.subTest(limit=limit), patch.object(native._NativeReplicaExperts, "apply") as execute:
                with self.assertRaisesRegex(ValueError, "swiglu_limit"):
                    native.native_replica_experts(values, None, None, None, None, swiglu_limit=limit)
                execute.assert_not_called()

    def test_default_limit_preserves_ordinary_native_activation(self) -> None:
        """Callers without a clamp keep the existing native kernel branch."""
        values = SimpleNamespace(device=SimpleNamespace(type="npu"), dtype=torch.bfloat16)
        with patch.object(native._NativeReplicaExperts, "apply") as execute:
            native.native_replica_experts(values, None, None, None, object())
        self.assertIsNone(execute.call_args.args[-1])

    def test_home_and_guest_forward_backward_and_empty_rank(self) -> None:
        """Run the real custom autograd with mocked transport and hardware kernels."""
        for dtype, limit, empty in product((torch.float32, torch.bfloat16), (-1.0, 0.0, 1.5), (False, True)):
            with self.subTest(dtype=dtype, limit=limit, empty=empty):
                self._check_replica_autograd(dtype, limit, empty)

    @staticmethod
    def _check_replica_autograd(dtype: torch.dtype, limit: float, empty: bool) -> None:
        generator = torch.Generator().manual_seed(473)
        logical_counts = [[0, 0], [0, 0]] if empty else [[6, 2], [6, 2]]
        plan = build_expert_replica_plan(logical_counts, 1)
        route = ReplicaRoute(plan, torch.empty(0), torch.empty(0), 1, None)
        counts = torch.tensor(plan.destination_counts[1])
        values = torch.randn(int(counts.sum()), 4, generator=generator, dtype=dtype).requires_grad_()
        home = [
            torch.randn(shape, generator=generator, dtype=dtype).requires_grad_() for shape in ((1, 4, 6), (1, 3, 4))
        ]
        guest = [torch.randn_like(weight) for weight in home]
        guest_gradients = tuple((torch.zeros_like(weight, dtype=torch.float32) for weight in guest))
        pool = SimpleNamespace(weights=guest, gradients=guest_gradients)
        expected_values = values.detach().clone().requires_grad_()
        expected_home = [weight.detach().clone().requires_grad_() for weight in home]
        expected_guest = [weight.detach().clone().requires_grad_() for weight in guest]
        if empty:
            expected = expected_values.clone()
        else:
            parts, start = ([], 0)
            for slot, count in enumerate(counts.tolist()):
                weights = expected_home if slot == 0 else expected_guest
                parts.append(_reference(expected_values[start : start + count] @ weights[0][0], limit) @ weights[1][0])
                start += count
            expected = torch.cat(parts)
        upstream = torch.randn(expected.shape, generator=generator, dtype=dtype)
        (expected * upstream).sum().backward()
        with patch.object(
            native, "prefetch_weights", side_effect=lambda *_args, **_kwargs: nullcontext(pool)
        ), patch.object(native, "_gmm", side_effect=_cpu_gmm), patch.object(
            native, "return_gradients", side_effect=lambda gradients, *_args, **_kwargs: gradients
        ), patch.object(
            native, "_npu_ops", side_effect=AssertionError("Clamped path used ordinary NPU SwiGLU")
        ):
            actual = native._NativeReplicaExperts.apply(values, home[0], home[1], counts, route, limit)
            (actual * upstream).sum().backward()
        tolerance = {"rtol": 1e-05, "atol": 2e-06} if dtype == torch.float32 else {"rtol": 0.02, "atol": 0.04}
        torch.testing.assert_close(actual, expected, **tolerance)
        torch.testing.assert_close(values.grad, expected_values.grad, **tolerance)
        for actual_weight, expected_weight in zip(home, expected_home):
            wanted = expected_weight.grad if expected_weight.grad is not None else torch.zeros_like(expected_weight)
            torch.testing.assert_close(actual_weight.grad, wanted, **tolerance)
        for gradient, expected_weight in zip(guest_gradients, expected_guest):
            wanted = expected_weight.grad if expected_weight.grad is not None else torch.zeros_like(expected_weight)
            torch.testing.assert_close(gradient.to(dtype), wanted, **tolerance)
