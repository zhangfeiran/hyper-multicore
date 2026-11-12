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
"""CPU contracts for framework-independent LongCat expert routing."""

from typing import Any
import unittest
from unittest.mock import patch

import torch

from hyper_multicore.longcat._routing import route_experts
from hyper_multicore.longcat import LongcatFlashExperts


class _Executor:
    """Apply a differentiable stand-in to compact real-expert assignments."""

    num_experts = 2

    def __call__(self, states: torch.Tensor, ids: torch.Tensor, _weights: torch.Tensor, *,
                 expert_weights: tuple) -> torch.Tensor:
        """Return a distinct output per real expert without accelerator code."""
        scale = expert_weights[0][ids.reshape(-1)]
        return states * scale[:, None]


class _IndexedRead(torch.autograd.Function):
    """Model a receiver computation with adapter-owned assignment gradients."""

    @staticmethod
    def forward(_ctx: Any, carrier: torch.Tensor, source: torch.Tensor,
                token: torch.Tensor) -> torch.Tensor:
        """Read original token values without reading the gradient carrier."""
        if carrier.shape[0] != token.numel():
            raise ValueError("Carrier and assignment row map differ")
        return source.index_select(0, token)

    @staticmethod
    def backward(_ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        """Return compact gradients through the ordered input boundary exactly once."""
        return gradient, None, None


class _IndexedExecutor(_Executor):
    """Model the indexed native boundary while retaining the existing reference arithmetic."""

    shmem_data_engine = "roce"

    def __call__(self, states: torch.Tensor, ids: torch.Tensor, _weights: torch.Tensor, *,
                 expert_weights: tuple, _indexed_source: tuple) -> torch.Tensor:
        """Consume values through explicit metadata and gradients through the carrier."""
        source, token = _indexed_source
        if states.untyped_storage().data_ptr() != source.untyped_storage().data_ptr():
            raise RuntimeError("The adapter materialized expanded activation storage")
        values = _IndexedRead.apply(states, source, token)
        return values * expert_weights[0][ids.reshape(-1), None]


class TestLongcatRouting(unittest.TestCase):
    """Check real and identity output and gradients with no distributed backend."""

    def test_torch_real_and_identity_gradients(self):
        """Match a per-token sorted reference for output, dX, dP and real weights."""
        source = torch.tensor([[1., -2., 3.], [4., 1., -3.]])
        ids = torch.tensor([[3, 0, 2], [1, 3, 0]])
        probability = torch.tensor([[0.1, 0.3, 0.6], [0.2, 0.4, 0.4]])
        scale = torch.tensor([1.5, 0.75])
        hidden = source.clone().requires_grad_()
        weights = probability.clone().requires_grad_()
        parameters = scale.clone().requires_grad_()
        actual = route_experts(_Executor(), hidden, ids, weights, 2, 2, 3, None,
                               (parameters, parameters), "torch")
        actual.square().sum().backward()

        ref_hidden = source.clone().requires_grad_()
        ref_weights = probability.clone().requires_grad_()
        ref_parameters = scale.clone().requires_grad_()
        output = []
        for token in range(ids.shape[0]):
            value = torch.zeros_like(ref_hidden[token])
            for slot in ids[token].argsort().tolist():
                expert = int(ids[token, slot])
                contribution = ref_hidden[token] * (ref_parameters[expert] if expert < 2 else 1.0)
                value = value + contribution * ref_weights[token, slot]
            output.append(value)
        expected = torch.stack(output)
        expected.square().sum().backward()
        for observed, reference in ((actual, expected), (hidden.grad, ref_hidden.grad),
                                    (weights.grad, ref_weights.grad), (parameters.grad, ref_parameters.grad)):
            torch.testing.assert_close(observed, reference, rtol=1e-6, atol=1e-6)

    def test_indexed_source_preserves_bf16_slot_gradient_order(self):
        """The indexed path shares original storage and matches output, dX, dP and dW exactly."""
        generator = torch.Generator().manual_seed(404)
        source = torch.randn(17, 13, generator=generator, dtype=torch.bfloat16)
        ids = torch.tensor([[3, 0, 2], [1, 3, 0], [2, 3, 0], [0, 1, 3]])[torch.arange(17) % 4]
        probabilities = torch.rand(17, 3, generator=generator)
        scale = torch.tensor([1.75, -0.625], dtype=torch.bfloat16)
        upstream = torch.randn(17, 13, generator=generator, dtype=torch.bfloat16)
        results = []
        for executor in (_Executor(), _IndexedExecutor()):
            hidden = source.clone().requires_grad_()
            weights = probabilities.clone().requires_grad_()
            parameters = scale.clone().requires_grad_()
            output = route_experts(executor, hidden, ids, weights, 2, 2, 3, None,
                                   (parameters, parameters), "torch")
            output.backward(upstream)
            results.append((output, hidden.grad, weights.grad, parameters.grad))
        for actual, expected in zip(results[1], results[0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_all_identity_preserves_router_and_input_gradients(self):
        """All-identity routes do not execute or update real-expert parameters."""
        hidden = torch.randn(3, 4, requires_grad=True)
        weights = torch.rand(3, 2, requires_grad=True)
        real = torch.ones(2, requires_grad=True)
        ids = torch.tensor([[2, 3], [3, 2], [2, 3]])
        output = route_experts(_Executor(), hidden, ids, weights, 2, 2, 2, None, (real, real), "torch")
        output.sum().backward()
        self.assertIsNone(real.grad)
        torch.testing.assert_close(hidden.grad, weights.sum(1)[:, None].expand_as(hidden))
        torch.testing.assert_close(weights.grad, hidden.sum(1)[:, None].expand_as(weights))

    def test_hccl_transport_keeps_value_inputs_with_roce_environment(self):
        """A RoCE local runtime must not give the outer HCCL transport a metadata-only carrier."""
        executor = _Executor()
        executor.shmem_data_engine = "roce"
        executor.communication_backend = "hccl"
        hidden = torch.tensor([[2., -3.], [5., 7.]], requires_grad=True)
        probability = torch.tensor([[0.25, 0.75], [0.6, 0.4]], requires_grad=True)
        scale = torch.tensor([2., 3.], requires_grad=True)
        output = route_experts(executor, hidden, torch.tensor([[0, 2], [1, 3]]), probability,
                               2, 2, 2, None, (scale, scale), "torch")
        expected_scale = torch.tensor([1.25, 2.2])
        torch.testing.assert_close(output, hidden * expected_scale[:, None])
        output.sum().backward()
        torch.testing.assert_close(hidden.grad, expected_scale[:, None].expand_as(hidden))

    def test_invalid_expert_id_fails(self):
        """Reject an out-of-range identity or real expert ID."""
        hidden, weights = torch.ones(1, 2), torch.ones(1, 1)
        for expert in (-1, 4):
            with self.subTest(expert=expert), self.assertRaises(ValueError):
                route_experts(_Executor(), hidden, torch.tensor([[expert]]), weights,
                              2, 2, 1, None, (weights, weights), "torch")

    def test_padded_eoi_contract_with_and_without_replicas(self):
        """Replica slicing preserves the owner's gradient and B0 keeps padded storage."""
        for slots in (0, 2):
            with self.subTest(slots=slots), patch("hyper_multicore.longcat.layer.MegaMoeExperts") as factory:
                layer = LongcatFlashExperts(hidden_size=3, intermediate_size=2, num_experts=8,
                                            zero_experts=4, top_k=4, gate_up_padded=True,
                                            replica_slots_per_rank=slots)
                gate = torch.randn(12, 4, 3, requires_grad=True)
                down = torch.randn(8, 3, 2, requires_grad=True)
                selected, selected_down = layer._weights((gate, down))
                self.assertEqual(selected.shape[0], 8 if slots else 12)
                self.assertIs(selected_down, down)
                self.assertEqual(factory.call_args.kwargs["expert_weight_layout"], "EOI")
                self.assertEqual(factory.call_args.kwargs["replica_granularity"], "expert" if slots else "token")
                self.assertEqual(factory.call_args.kwargs["gate_up_expert_padding"], 0 if slots else 4)
                if slots:
                    selected.sum().backward()
                    torch.testing.assert_close(gate.grad[:8], torch.ones_like(gate.grad[:8]))
                    torch.testing.assert_close(gate.grad[8:], torch.zeros_like(gate.grad[8:]))
