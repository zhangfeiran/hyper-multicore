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
"""Exercise real shared autograd contexts while emulating only device math on CPU."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock, patch

import torch
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from hyper_multicore.modules.mega_moe import function as moe_function
from hyper_multicore.modules.mega_moe import shared as moe_shared

# Pylint cannot infer the callable signature of Torch's C linear binding.
# pylint: disable=not-callable

def shared_math(
    source: torch.Tensor, first: torch.Tensor, second: torch.Tensor, limit: float | None = None,
) -> torch.Tensor:
    """Compute an independent dense SwiGLU reference."""
    gate, up = functional.linear(source, first).chunk(2, dim=-1)
    if limit is not None:
        gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    return functional.linear(functional.silu(gate) * up, second)


class TestSharedFunction(unittest.TestCase):
    """Check input/parameter gradient bindings after multiple deferred forwards."""

    # The private launch boundary is the lifecycle contract under test.
    # pylint: disable=protected-access

    def test_deferred_shared_outputs_and_all_parameter_gradients(self) -> None:
        """Release poisoning must not invalidate shared saved tensors or owned outputs."""
        for input_grad, limit in ((False, None), (True, None), (True, 1.25)):
            with self.subTest(input_grad=input_grad, limit=limit):
                self._check_deferred(input_grad, limit)

    def test_strided_shared_input_keeps_values_and_gradients(self) -> None:
        """Honor row strides at the public boundary before the raw device pointer is consumed."""
        self._check_deferred(True, None, strided=True)

    def test_independent_shared_switches_keep_deferred_gradients(self) -> None:
        """Feature: Independent shared kernel directions.
        Description: Exercise every direction with deferred backward, clamps and strided inputs.
        Expectation: Match dense autograd without redundant shared forward calls.
        """
        for forward in (False, True):
            for backward in (False, True):
                for input_grad, limit in ((False, None), (True, None), (True, 1.25)):
                    with self.subTest(forward=forward, backward=backward, input_grad=input_grad, limit=limit):
                        self._check_deferred(input_grad, limit, strided=input_grad,
                                             forward_shared=forward, backward_shared=backward)

    def test_shared_backward_retains_checkpoint_and_frozen_weight_contracts(self) -> None:
        """Feature: Shared backward under checkpoint and partial freezing.
        Description: Run both checkpoint modes with either shared weight frozen.
        Expectation: Unpack state once and return only requested gradients matching dense autograd.
        """
        for forward in (False, True):
            for recompute in (None, False, True):
                for freeze in (1, 2):
                    with self.subTest(forward=forward, recompute=recompute, freeze=freeze):
                        self._check_deferred(True, 1.25, forward_shared=forward, backward_shared=True,
                                             recompute=recompute, freeze=freeze)

    def test_shared_backward_handles_unused_outputs_and_retained_graphs(self) -> None:
        """Feature: Multiple-output autograd lifetime.
        Description: Consume one branch and execute a retained backward twice.
        Expectation: Zero the unused branch and retain shared intermediates until the final backward.
        """
        for forward in (False, True):
            for outputs in ("shared", "routed"):
                with self.subTest(forward=forward, outputs=outputs):
                    self._check_deferred(True, None, forward_shared=forward, backward_shared=True,
                                         outputs=outputs)

    def test_a5_native_activation_producers_keep_recompute_and_clamp_gradients(self) -> None:
        """Consume native activation gradients without rerunning them in the fused CPU kernel."""
        for forward in (False, True):
            for limit in (None, 1.25):
                for recompute in (None, False, True):
                    with self.subTest(forward=forward, limit=limit, recompute=recompute):
                        self._check_deferred(True, limit, forward_shared=forward, backward_shared=True,
                                             recompute=recompute, cube_cores=32)

    def _check_deferred(
        self, input_grad: bool, limit: float | None, *, strided: bool = False,
        forward_shared: bool = True, backward_shared: bool = False,
        recompute: bool | None = None, freeze: int | None = None, outputs: str = "both",
        cube_cores: int = 24,
    ) -> None:
        """Compare deferred shared gradients with an independent dense reference.

        Args:
            input_grad: Whether the caller requires input gradients.
            limit: Optional activation clamp.
            strided: Use a token view with a gap between its physical rows.
            forward_shared: Execute shared forward in the MegaMoe launch.
            backward_shared: Execute shared backward in the MegaMoe launch.
            recompute: Checkpoint mode; None executes without checkpoint.
            freeze: Optional first or second shared weight to freeze.
            outputs: Select the consumed branch; single branches also repeat backward with graph retention.
            cube_cores: Select full A3 fusion or the A5 native activation producer contract.
        """
        torch.manual_seed(73)
        spec = SimpleNamespace(hidden_size=4, intermediate_size=2, rank_id=0, ep_size=2,
                               num_experts=4, top_k=2, dispatch_mode="push", shared_intermediate_size=2,
                               forward_ingress_mode="legacy", swiglu_limit=limit,
                               forward_shared_expert=forward_shared, backward_shared_expert=backward_shared,
                               num_cube_cores=cube_cores, expert_weight_layout="EIO", replica_slots_per_rank=0, auto_tokens=False)
        plan = SimpleNamespace(spec=spec, reuse_backward_dispatch=False, ingress_diagnostics=None,
                               bwd_diagnostics=None)
        workspace = Mock(expert_buffer=torch.zeros(8, 4), routed_buffer=torch.zeros(4, 4),
                         gmm_workspace=torch.zeros(512, dtype=torch.uint8), swiglu_grad_workspace=torch.zeros(1),
                         replica_provider=None, replica_runtime_images={}, replica_task_prefixes={})
        workspace.release.side_effect = lambda: (workspace.expert_buffer.fill_(float("nan")),
                                                 workspace.routed_buffer.fill_(float("nan")))
        w1 = torch.randn(4, 4, requires_grad=True)
        w2 = torch.randn(4, 2, requires_grad=True)
        if limit is not None:
            with torch.no_grad():
                w1[:, 0].copy_(torch.tensor([limit, -limit, limit, -limit]))
        ref1, ref2 = w1.detach().clone().requires_grad_(), w2.detach().clone().requires_grad_()
        if freeze is not None:
            (w1 if freeze == 1 else w2).requires_grad_(False)
            (ref1 if freeze == 1 else ref2).requires_grad_(False)
        routed1, routed2 = torch.ones(2, 4, 4, requires_grad=True), torch.ones(2, 2, 4, requires_grad=True)

        def forward_kernel(
            _plan: Any, _route: Any, _source: Any, _first: Any, _second: Any, execution: Any, _pool: Any = None,
        ) -> None:
            """Fill independent outputs through the actual indirect tensor table."""
            self.assertEqual(execution.shared_ptrs is not None, forward_shared)
            if execution.shared_ptrs is not None:
                source, first, second, up, activation, output, _scratch = execution.shared_ptrs
                # Device shared_x descriptors carry dense dimensions and no stride metadata.
                source = torch.as_strided(source, source.shape, (source.shape[1], 1), source.storage_offset())
                up.copy_(functional.linear(source, first))
                gate, branch = up.chunk(2, dim=-1)
                if limit is not None:
                    gate, branch = gate.clamp(max=limit), branch.clamp(-limit, limit)
                activation.copy_(functional.silu(gate) * branch)
                output.copy_(functional.linear(activation, second))
            execution.dispatch.zero_()
            execution.combine.zero_()

        def backward_kernel(_plan: Any, _saved: Any, _gradient: Any, execution: Any,
                            _pool: Any = None, _return: Any = None) -> None:
            """Emulate device math while retaining the real saved-context and launch plumbing."""
            self.assertEqual(execution.shared_ptrs is not None, backward_shared)
            if execution.shared_ptrs is not None:
                dy, activation, dw2, second, dactivation, up, dup, first, dx, source, dw1 = execution.shared_ptrs
                if cube_cores != 32:
                    dactivation.copy_(torch.mm(dy, second))
                    activation_input = up
                    if limit is not None:
                        gate, branch = up.chunk(2, dim=-1)
                        activation_input = torch.cat((gate.clamp(max=limit), branch.clamp(-limit, limit)), dim=-1)
                    derivative = activation_backward(dactivation, activation_input, -1)
                    if limit is not None:
                        gate_grad, branch_grad = derivative.chunk(2, dim=-1)
                        derivative = torch.cat(
                            (torch.where(gate <= limit, gate_grad, 0),
                             torch.where((branch >= -limit) & (branch <= limit), branch_grad, 0)), -1,
                        )
                    dup.copy_(derivative)
                dx.copy_(torch.mm(dup, first))
                dw1.copy_(torch.mm(source.t(), dup))
                dw2.copy_(torch.mm(activation.t(), dy))
            execution.grad_x.zero_()

        def backward_pointers(tensors: tuple[torch.Tensor, ...], *, idle_vectors: bool = False) -> tuple:
            """Reserve vector workers only when the kernel still produces shared SwiGLU gradients."""
            self.assertEqual(idle_vectors, cube_cores != 32)
            return tensors

        def activation_backward(gradient: torch.Tensor, up: torch.Tensor, _dim: int) -> torch.Tensor:
            """Emulate only the native activation derivative on CPU."""
            with torch.enable_grad():
                value = up.detach().clone().requires_grad_()
                gate, branch = value.chunk(2, dim=-1)
                output = functional.silu(gate) * branch
                return torch.autograd.grad(output, value, gradient)[0]

        calls = []
        def activation_forward(up: torch.Tensor, dim: int = -1) -> torch.Tensor:
            """Keep the native forward differentiable when both shared switches are disabled."""
            gate, branch = up.chunk(2, dim=dim)
            return functional.silu(gate) * branch

        native_forward = Mock(side_effect=activation_forward)
        with (
            patch.object(moe_function, "pointer_table", side_effect=lambda tensors, **_: tensors),
            patch.object(moe_shared, "pointer_table", side_effect=backward_pointers),
            patch.object(moe_function.torch.ops.npu, "npu_swiglu_backward", side_effect=activation_backward),
            patch.object(moe_function, "load_torch_npu", return_value=SimpleNamespace(npu_swiglu=native_forward)),
            patch.object(moe_function, "_launch_forward_kernel", side_effect=forward_kernel),
            patch.object(moe_function, "_launch_backward_kernel", side_effect=backward_kernel),
            patch.object(moe_function.multicore_ops, "moe_token_permute_grad",
                         side_effect=lambda grad, _mapping, rows, top_k: grad.reshape(rows, top_k, 4).sum(1)),
        ):
            for _ in range(2):
                source = (torch.randn(2, 4) if limit is None else torch.eye(4)[:2]).requires_grad_(input_grad)
                if strided:
                    storage = torch.randn(4, 4, requires_grad=input_grad)
                    source = storage[::2]
                    source.retain_grad()
                reference = source.detach().clone().requires_grad_(input_grad)
                route = SimpleNamespace(replica_route=None, expert_capacity=3, group_list=torch.tensor([0, 3]))
                for name in ("dispatch_src_off", "dispatch_target_off", "dispatch_size",
                             "combine_src_off", "combine_target_off", "combine_size"):
                    setattr(route, name, torch.zeros(4, dtype=torch.int64))
                prepared = SimpleNamespace(metadata=route, routed_tokens=source.detach().repeat_interleave(2, dim=0),
                                           unpermute_mapping=torch.arange(4, dtype=torch.int32))
                def execute(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                    """Recompute through the public route boundary with the current token tensor."""
                    return moe_function.execute_mega_moe_with_permutation(
                        tokens, torch.zeros(2, 2, dtype=torch.int32), routed1, routed2, prepared, plan, workspace,
                        shared_weight1=w1, shared_weight2=w2)

                if recompute is None:
                    routed_output, shared_output = execute(source)
                else:
                    routed_output, shared_output = checkpoint(execute, source, use_reentrant=recompute)
                output = shared_output + routed_output.reshape(2, 2, 4).sum(1)
                expected = shared_math(reference, ref1, ref2, limit)
                if outputs == "shared":
                    output = shared_output
                elif outputs == "routed":
                    output = routed_output.reshape(2, 2, 4).sum(1)
                    expected = reference * 0 + (ref1.sum() + ref2.sum()) * 0
                torch.testing.assert_close(output, expected)
                calls.append((source, reference, output, expected, torch.randn_like(output)))
            self._backward_deferred(calls, ((w1, ref1), (w2, ref2)), outputs)
            if forward_shared:
                native_forward.assert_not_called()
            else:
                self.assertEqual(native_forward.call_count, {None: 2, False: 4, True: 4}[recompute])
        self.assertEqual(workspace.claim.call_count, (4 if recompute is None else 6) + (2 if outputs != "both" else 0))
        self.assertEqual(workspace.release.call_count, workspace.claim.call_count)

    def _backward_deferred(
        self, calls: list[tuple[torch.Tensor, ...]], weights: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        outputs: str,
    ) -> None:
        """Compare input and accumulated parameter gradients after deferred backward.

        Args:
            calls: Actual/reference inputs, outputs and upstream gradients per invocation.
            weights: Actual/reference shared parameter pairs.
            outputs: Consumed branch; single branches also retain and repeat backward.
        """
        for source, reference, output, expected, dy in reversed(calls):
            if outputs != "both":
                output.backward(dy, retain_graph=True)
                expected.backward(dy, retain_graph=True)
            output.backward(dy)
            expected.backward(dy)
            if source.requires_grad:
                torch.testing.assert_close(source.grad, reference.grad)
            else:
                self.assertIsNone(source.grad)
        for weight, reference in weights:
            if weight.requires_grad:
                torch.testing.assert_close(weight.grad, reference.grad)
            else:
                self.assertIsNone(weight.grad)


if __name__ == "__main__":
    unittest.main()
