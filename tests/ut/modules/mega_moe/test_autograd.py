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
"""CPU storage-lifetime tests for the MegaMoe autograd bridge."""

import os
import unittest
import weakref
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, PropertyMock, patch

import torch
from torch.utils.checkpoint import DefaultDeviceType, checkpoint, set_checkpoint_early_stop

from hyper_multicore.diagnostics import MegaKernelDiagnostics
from hyper_multicore.modules.mega_moe.backends.bf16 import autograd as function_module
from hyper_multicore.modules.mega_moe.backends.bf16.autograd import _MegaMoeFunction
from tests.common.mark_utils import arg_mark


# The private autograd bridge and saved execution buffers are the contracts under test.
class TestMegaMoeFunction(unittest.TestCase):
    """Exercise real autograd contexts with mocked communication and kernels."""

    # pylint: disable=protected-access

    def test_plan_stream_identity_preserves_registration(self) -> None:
        """Register plan storage without requesting a queue-draining stream handle."""
        device = Mock(type="npu")
        forward, backward, tiling, diagnostic = [Mock(spec=torch.Tensor) for _ in range(4)]
        for tensor in (forward, backward, tiling, diagnostic):
            tensor.device = device
        plan = function_module.MegaMoePlan(
            spec=None, forward_graph=None, event_counter_bytes=4096,
            fwd_diagnostics=SimpleNamespace(runtime_config=forward),
            up_proj_tiling=tiling, swiglu_tiling=None, down_proj_tiling=None,
            backward_graph=None, bwd_diagnostics=SimpleNamespace(runtime_config=backward),
            act_grad_tiling=None, gate_grad_tiling=None, w1_grad_tiling=None,
            w2_grad_tiling=None, swiglu_grad_tiling=None,
        )
        first = SimpleNamespace(stream_id=10)
        same = SimpleNamespace(stream_id=10)
        other = SimpleNamespace(stream_id=20)
        plan.record_stream(forward, stream=first)
        plan.record_stream(backward, stream=same)
        plan.record_stream(diagnostic, stream=same)
        plan.record_stream(forward, stream=other)
        for tensor in (forward, backward, tiling):
            self.assertEqual([entry.args[0] for entry in tensor.record_stream.call_args_list], [first, other])
        diagnostic.record_stream.assert_called_once_with(same)


    def test_ragged_pull_uses_reserved_source_prefix(self) -> None:
        """Forward and backward retain SHMEM addresses while exposing only actual rows."""
        buffer = torch.full((256, 4), float("nan"))
        workspace = SimpleNamespace(source_buffer=buffer)
        spec = SimpleNamespace(apply_probs_before_fc2=False, dispatch_mode="pull", hidden_size=4)
        rows = torch.arange(12).reshape(3, 4).float()
        dispatch, source = function_module._dispatch_and_source(spec, workspace, rows, 7)
        self.assertEqual(tuple(dispatch.shape), (7, 4))
        self.assertEqual(source.data_ptr(), buffer.data_ptr())
        torch.testing.assert_close(source, rows)
        self.assertTrue(torch.isnan(buffer[3:]).all())
        context = SimpleNamespace(plan=SimpleNamespace(spec=spec), workspace=workspace,
                                  source_rows=3, has_unpermute=False)
        staged, probabilities = function_module._stage_backward_source(context, rows + 1, ())
        torch.testing.assert_close(staged, rows + 1)
        self.assertIsNone(probabilities)
        context.source_rows = 0
        context.has_unpermute = True
        context.needs_input_grad = (False,) * 7 + (True,)
        context.topk_dtype = torch.float32
        with patch.object(function_module.multicore_ops, "mega_moe_unpermute_grad_out") as unpermute:
            empty, gradient = function_module._stage_backward_source(
                context, rows[:0], (torch.empty(0, dtype=torch.int32), rows[:0], torch.empty(0, 2)))
        unpermute.assert_not_called()
        self.assertEqual(tuple(empty.shape), (0, 4))
        self.assertEqual(tuple(gradient.shape), (0, 2))
        self.assertEqual(function_module._native_source(empty).data_ptr(), buffer.data_ptr())

    def test_empty_native_source_preserves_symmetric_storage(self) -> None:
        """Empty pull ranks must retain the symmetric base used to address remote ranks."""
        storage = torch.arange(32).reshape(8, 4)
        empty_source = storage[:0]
        native_source = function_module._native_source(empty_source)
        self.assertEqual(native_source.data_ptr(), storage.data_ptr())
        torch.testing.assert_close(native_source, storage[:1])
        self.assertEqual(tuple(empty_source.shape), (0, 4))
        ordinary_empty = torch.empty(0, 4)
        placeholder = function_module._native_source(ordinary_empty)
        self.assertEqual(tuple(placeholder.shape), (1, 4))
        self.assertNotEqual(placeholder.data_ptr(), 0)

    def test_eoi_weight_operands_and_gradients_keep_contiguous_layout(self) -> None:
        """Native EOI operands and gradients already match the parameter storage layout."""
        spec = SimpleNamespace(
            expert_weight_layout="EOI",
            local_experts=2,
            hidden_size=4,
            intermediate_size=3,
        )
        plan = SimpleNamespace(spec=spec)
        gate_up = torch.empty(2, 6, 4)
        down = torch.empty(2, 4, 3)
        self.assertIs(function_module._native_weight_operand(gate_up, plan), gate_up)
        self.assertIs(function_module._native_weight_operand(down, plan), down)

        gradients = function_module._allocate_backward_intermediates(
            spec,
            8,
            torch.empty(8, 4),
            gate_up,
            down,
        )
        for weight, gradient in ((gate_up, gradients.grad_weight1), (down, gradients.grad_weight2)):
            self.assertEqual(gradient.shape, weight.shape,
                             msg=f"Gradient shape: expected={weight.shape}, actual={gradient.shape}")
            self.assertEqual(gradient.stride(), weight.stride(),
                             msg=f"Gradient stride: expected={weight.stride()}, actual={gradient.stride()}")
            self.assertEqual(torch.count_nonzero(gradient).item(), 0,
                             msg=f"Empty-expert gradients: expected=0, actual={gradient}")


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_fp32_weight_gradients_preserve_activation_dtype(self) -> None:
        """Keep partial dW unquantized without promoting activation scratch."""
        spec = SimpleNamespace(hidden_size=4, intermediate_size=3)
        for shapes in (((2, 4, 6), (2, 3, 4)), ((2, 6, 4), (2, 4, 3))):
            with self.subTest(shapes=shapes):
                weights = tuple(torch.empty(shape, dtype=torch.bfloat16) for shape in shapes)
                gradients = function_module._allocate_backward_intermediates(
                    spec, 8, torch.empty(8, 4, dtype=torch.bfloat16), *weights,
                    weight_grad_dtype=torch.float32)
                for weight, gradient in zip(weights, (gradients.grad_weight1, gradients.grad_weight2)):
                    self.assertEqual(gradient.shape, weight.shape)
                    self.assertEqual(gradient.dtype, torch.float32)
                    self.assertEqual(torch.count_nonzero(gradient).item(), 0)
                for value in (gradients.act_grad, gradients.swiglu_grad, gradients.gate_dx):
                    self.assertEqual(value.dtype, torch.bfloat16)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_backward_unpacks_saved_tensors_once(self) -> None:
        """Feature: MegaKernel diagnostics.
        Description: Checkpoint unpack hooks must only be consumed once per backward call.
        Expectation: Backward unpacks saved tensors once.
        """
        tensors = tuple(torch.ones(1) for _ in range(13))
        context = Mock()
        context.weight_binding = None
        context.has_unpermute = False
        context.defer_forward_copies = False
        context.deferred_recompute = None
        context.needs_input_grad = (False,) * 9
        gradient = torch.ones(1)
        context.plan.spec.dispatch_mode = "push"
        context.plan.spec.apply_probs_before_fc2 = False
        execution = function_module._BackwardExecution(
            dispatch=torch.ones(1), grad_x=torch.ones(1), event_counters=Mock(),
            gmm_workspace=None, swiglu_workspace=None,
            intermediates=function_module._BackwardIntermediates(*(torch.ones(1) for _ in range(5))),
        )
        with (
            patch.object(type(context), "saved_tensors", create=True,
                         new_callable=PropertyMock, side_effect=[tensors, RuntimeError("duplicate unpack")]) as unpack,
            patch.object(function_module, "_prepare_backward_execution", return_value=execution),
            patch.object(function_module, "_launch_backward_kernel"),
            patch.object(function_module, "_restore_input_gradient", return_value=gradient) as restore,
            patch.object(function_module, "_release_completed_graph"),
        ):
            result = _MegaMoeFunction.backward(context, gradient)
        unpack.assert_called_once_with()
        restore.assert_called_once_with(context, execution.grad_x, tensors[12:])
        self.assertIs(result[0], gradient)
        self.assertIs(result[1], execution.intermediates.grad_weight1)
        self.assertIs(result[2], execution.intermediates.grad_weight2)
        context.workspace.release.assert_called_once_with(stream=context.workspace.lease_stream)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_deferred_backward_keeps_local_owned_storage_after_workspace_reuse(self) -> None:
        """Feature: Deferred backward storage.
        Description: Reuse the workspace between deferred forward and backward.
        Expectation: Backward retains its own local tensors.
        """
        self._check_deferred_backward(permuted=False)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_permutation_gradient_consumes_workspace_before_release(self) -> None:
        """Feature: Deferred backward storage.
        Description: Run the permutation gradient while tracking workspace release.
        Expectation: Consume the receive buffer before releasing its lease.
        """
        self._check_deferred_backward(permuted=True)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_frozen_tokens_do_not_launch_a_permutation_gradient(self) -> None:
        """Feature: Deferred backward storage.
        Description: Backpropagate with frozen token activations.
        Expectation: Skip the token permutation gradient.
        """
        self._check_deferred_backward(permuted=True, input_grad=False)

    def test_dynamic_tokens_restore_each_deferred_forward_length(self) -> None:
        """Use per-call token counts despite later forwards reusing a larger source buffer."""
        self._check_deferred_backward(permuted=True, vary_tokens=True)

    def test_non_reentrant_checkpoint_unpacks_saved_tensors_once(self) -> None:
        """Checkpoint recomputation must not fail when backward reads its permutation state."""
        # CPU-only inputs must not select an installed accelerator for autocast replay.
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")
        for permuted in (False, True):
            with self.subTest(permuted=permuted):
                self._check_deferred_backward(permuted=permuted, replay=True)

    def test_checkpoint_can_defer_expert_recompute_until_its_original_backward(self) -> None:
        """Downstream replay gets token output while only the original node launches R and B."""
        self._check_deferred_expert_recompute()

    def test_deferred_recompute_preserves_partial_weight_grads_with_compound_enabled(self) -> None:
        """Frozen projections keep the split backward even with the compound workspace."""
        for frozen in ("weight1", "weight2"):
            with self.subTest(frozen=frozen):
                self._check_deferred_expert_recompute(frozen=frozen)

    def _check_deferred_expert_recompute(self, *, frozen: str | None = None) -> None:
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")
        spec = SimpleNamespace(
            apply_probs_before_fc2=False, dispatch_mode="push", expert_weight_layout="EOI",
            hidden_size=2, intermediate_size=1, top_k=1, auto_tokens=True, architecture=2201,
            dynamic_tokens=True, plan_tokens=2,
        )
        outputs = function_module.BackwardOutputs(weight1=frozen != "weight1", weight2=frozen != "weight2")
        plan = SimpleNamespace(
            spec=spec, reuse_backward_dispatch=False, weight_grad_dtype=None, event_counter_bytes=8,
            backward_outputs=outputs,
        )
        metadata = function_module.RouteMetadata(
            *(torch.zeros(1, dtype=torch.int64) for _ in range(6)),
            group_list=torch.tensor([2]), expert_capacity=2, all_experts_active=True,
        )
        tokens = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        w1 = torch.tensor([2.0], requires_grad=outputs.weight1)
        w2 = torch.tensor([3.0], requires_grad=outputs.weight2)
        probs = torch.tensor([[0.25], [0.5]], requires_grad=True)
        ids = torch.zeros((2, 1), dtype=torch.int32)
        route = function_module.PreparedTopKRoute(
            tokens.detach(), torch.arange(2, dtype=torch.int32), None, None, metadata,
            indexed_dispatch_offsets=torch.zeros(2, dtype=torch.int64),
        )
        workspace = Mock(
            expert_buffer=torch.empty_like(tokens), routed_buffer=torch.empty_like(tokens),
            gmm_workspace=torch.empty(1), swiglu_grad_workspace=torch.empty(1),
            a3_fused_recompute=frozen is not None,
        )
        from hyper_multicore.modules.hot_replica.checkpoint import ReplayRecord
        record = ReplayRecord(owner=None, route=route)
        launches = []
        phase = False

        def execute(x, first, second, p):
            nonlocal phase
            output = function_module.execute_mega_moe_with_permutation(
                x, ids, first, second, route, plan, workspace, topk_weights=p,
                checkpoint_replay=phase, deferred_recompute=record,
            )
            phase = True
            return output.square().sum()

        def forward_kernel(_plan, _metadata, x, first, second, execution, **_options):
            launches.append("F or R")
            execution.dispatch.copy_(x)
            execution.combine.copy_(x + first + second)

        def backward_kernel(_plan, saved, source, execution, **options):
            launches.append("B")
            self.assertIsNone(options["recompute_call"])
            self.assertTrue(torch.equal(saved.dispatch, tokens))
            execution.grad_x.copy_(source)
            if outputs.weight1:
                execution.intermediates.grad_weight1.fill_(source.sum())
            if outputs.weight2:
                execution.intermediates.grad_weight2.fill_(source.sum())

        def unpermute_grad(y, g, _mapping, p, dx, dp):
            dx.copy_(g * p)
            dp.copy_((y * g).sum(-1, keepdim=True))

        vendor = SimpleNamespace(npu_moe_token_unpermute=lambda value, _mapping, probs: value * probs)
        with (
            patch.object(function_module, "_launch_forward_kernel", side_effect=forward_kernel),
            patch.object(function_module, "_launch_backward_kernel", side_effect=backward_kernel),
            patch.object(function_module, "load_torch_npu", return_value=vendor),
            patch.object(function_module.multicore_ops, "mega_moe_unpermute_grad_out", side_effect=unpermute_grad),
            patch.object(function_module.multicore_ops, "moe_token_permute_grad",
                         side_effect=lambda gradient, *_args: gradient.clone()),
            set_checkpoint_early_stop(False),
        ):
            inputs = tokens, w1, w2, probs
            grad_inputs = tuple(value for value in inputs if value.requires_grad)
            loss = checkpoint(execute, *inputs, use_reentrant=False)
            expected = ((tokens + w1 + w2) * probs).square().sum()
            reference = torch.autograd.grad(expected, grad_inputs)
            for retain in (True, False):
                actual = torch.autograd.grad(loss, grad_inputs, retain_graph=retain)
                for result, wanted in zip(actual, reference):
                    torch.testing.assert_close(result, wanted, atol=0, rtol=0)
                self.assertEqual(record.deferred_output is not None, retain)
        self.assertEqual(launches, ["F or R", "F or R", "B", "F or R", "B"])

    def test_compound_recompute_binds_disjoint_owner_buffers_in_physical_route_order(self) -> None:
        """Allocate both expert operands and restore the source-slot probability order."""
        hidden = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        ids = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.int32)
        probs = torch.arange(6, dtype=torch.float32).reshape_as(ids) / 5
        cap, source_slots = 8, ids.numel()
        metadata = function_module.RouteMetadata(
            *(torch.zeros(2, dtype=torch.int64) for _ in range(6)),
            group_list=torch.tensor([3, 6]), expert_capacity=6,
        )
        workspace = Mock(
            expert_capacity=cap, expert_buffer=hidden.new_empty((cap, 4)),
            routed_buffer=hidden.new_empty((source_slots, 4)), gmm_workspace=torch.empty(16, dtype=torch.uint8),
            recompute_workspace=torch.empty(cap * (4 * 2 + 32) + source_slots * 32, dtype=torch.uint8),
        )
        slot_to_physical = torch.argsort(torch.argsort(ids.reshape(-1), stable=True)).to(torch.int32)
        route = SimpleNamespace(metadata=metadata, indexed_dispatch_offsets=None,
                                unpermute_mapping=slot_to_physical)
        context = SimpleNamespace(
            weight_binding=None, workspace=workspace, deferred_recompute=SimpleNamespace(route=route),
            plan=SimpleNamespace(spec=SimpleNamespace(
                hidden_size=4, intermediate_size=2, top_k=2, dispatch_mode="push", apply_probs_before_fc2=False,
            ), event_counter_bytes=64),
        )
        weight = hidden.new_empty(0)
        saved, (inverse,), launch = function_module._prepare_fused_recompute(
            context, (hidden, weight, weight, probs, ids),
        )
        self.assertIs(launch.forward.source, hidden)
        self.assertEqual(saved.dispatch.data_ptr(), workspace.expert_buffer.data_ptr())
        self.assertEqual(launch.received_g.shape, (cap, 4))
        self.assertEqual(launch.received_p.data_ptr(), launch.received_g.data_ptr() + cap * 4 * 2)
        self.assertEqual(launch.returned_dp.data_ptr(), launch.received_p.data_ptr() + cap * 32)
        torch.testing.assert_close(launch.source_p[inverse, 0], probs.flatten(), atol=0, rtol=0)
        source_rows = launch.dispatch_offsets[2:].view(torch.int32)
        torch.testing.assert_close(source_rows, torch.argsort(inverse).div(2, rounding_mode="floor").int())

    def test_compound_call_owns_tiling_before_backward_submission(self) -> None:
        """Retain the tensor behind the nested tiling address while preparing B."""
        runtime = torch.zeros(128, dtype=torch.uint8)
        spec = SimpleNamespace(rank_id=0, ep_size=2, num_experts=24, hidden_size=4096,
                               plan_tokens=384, num_cube_cores=24)
        plan = SimpleNamespace(spec=spec, up_proj_tiling=None, swiglu_tiling=None, down_proj_tiling=None)
        ctx = SimpleNamespace(plan=plan, workspace=SimpleNamespace(lease_stream=None), token_rows=257, source_rows=771)
        tensor = torch.ones((1, 4), dtype=torch.bfloat16)
        trace = torch.empty_like(runtime)
        forward = SimpleNamespace(source=tensor, gmm_workspace=None, event_counters=runtime)
        saved = SimpleNamespace(weight1=tensor, weight2=tensor)
        recompute = SimpleNamespace(forward=forward, metadata=None, dispatch_offsets=None, barrier=None,
                                    received_g=None, source_p=None, received_p=None, local_dp=None, returned_dp=None)
        original_tensor = torch.tensor
        tiling_references = []

        def watch_tiling(words, **options):
            result = original_tensor(words, **options)
            if len(words) == 11:
                tiling_references.append(weakref.ref(result))
            return result

        with patch.dict(os.environ, {"HMC_A3_RECOMPUTE_GRADIENT_ROWS": "3"}), \
                patch.object(function_module, "_forward_kernel_inputs", return_value=(None,) * 16), \
                patch.object(torch, "tensor", new=watch_tiling), \
                patch.object(torch.Tensor, "record_stream", new=lambda *_: None):
            call = function_module._recompute_call(ctx, recompute, saved, tensor, runtime, trace=trace)

        tiling, = tiling_references
        self.assertIsNotNone(tiling())
        words = call.descriptor.view(torch.int64)
        self.assertEqual(words.numel(), 38)
        self.assertEqual(words[24].item(), tiling().data_ptr())
        self.assertEqual(words[-4:].tolist(), [trace.data_ptr(), 0, 3, 0])
        self.assertTrue(any(t is trace for t in call.operands))
        self.assertEqual(tiling()[6].item(), runtime.numel())
        self.assertEqual(tiling()[8:].tolist(), [257, 0, 771])
        del call
        self.assertIsNone(tiling())

    def _check_deferred_forward(
        self, source: Any, weight1: Any, weight2: Any, route: Any, plan: Any, workspace: Any,
        down_pointers: list[int],
        *, permuted: bool, vary_tokens: bool, replay: bool,
    ) -> Any:
        """Run one forward and check that its autograd storage survives workspace reuse."""
        tokens = source.shape[0]
        if permuted:
            ids = torch.tensor([[0, 1]] * tokens, dtype=torch.int32)
            mapping = (torch.arange(tokens * 2, dtype=torch.int32) if vary_tokens
                       else torch.tensor([0, 2, 1, 3], dtype=torch.int32))

            def operation(values: Any, first: Any, second: Any) -> Any:
                prepared = SimpleNamespace(
                    routed_tokens=values.detach().repeat_interleave(2, dim=0),
                    metadata=route, unpermute_mapping=mapping, indexed_dispatch_offsets=None,
                )
                return function_module.execute_mega_moe_with_permutation(
                    values, ids, first, second, prepared, plan, workspace,
                )
        else:
            operation = partial(function_module.execute_mega_moe, route=route, plan=plan, workspace=workspace)
        output = (checkpoint(operation, source, weight1, weight2, use_reentrant=False)
                  if replay else operation(source, weight1, weight2))
        if not replay:
            saved = output.grad_fn.saved_tensors
            self.assertEqual(len(saved), 13 if permuted and source.requires_grad else 12)
            self.assertEqual(saved[0].data_ptr(), down_pointers[-1])
            self.assertNotEqual(saved[0].data_ptr(), workspace.expert_buffer.data_ptr())
            self.assertEqual(saved[0].untyped_storage().nbytes(), route.expert_capacity * 4 * source.element_size())
        self.assertEqual(tuple(output.shape), (tokens * 2 if permuted else 4, 4))
        return output

    def _check_deferred_backward(self, *, permuted: bool, input_grad: bool = True,
                                vary_tokens: bool = False, replay: bool = False) -> None:
        """Exercise delayed backward with an optional input permutation boundary."""
        spec = SimpleNamespace(apply_probs_before_fc2=False, hidden_size=4, intermediate_size=2, rank_id=0,
                               ep_size=2, num_experts=4, local_num_tokens=2, top_k=2,
                               dispatch_mode="push", expert_weight_layout="EIO", auto_tokens=False,
                               architecture=2201, dynamic_tokens=False)
        plan = SimpleNamespace(
            spec=spec, reuse_backward_dispatch=False, weight_grad_dtype=None, event_counter_bytes=None,
            backward_outputs=function_module.BackwardOutputs(),
        )
        for name in ("up_proj", "swiglu", "down_proj", "act_grad", "gate_grad",
                     "w1_grad", "w2_grad", "swiglu_grad"):
            setattr(plan, f"{name}_tiling", None)
        runtime = SimpleNamespace(normal_tensor=None, kernel_name="mega_moe", rank=0)
        plan.fwd_diagnostics = MegaKernelDiagnostics(runtime)
        plan.bwd_diagnostics = MegaKernelDiagnostics(runtime)
        plan.fwd_runtime_config = plan.bwd_runtime_config = None
        workspace = Mock(
            expert_capacity=128,
            expert_buffer=torch.empty(128, 4),
            routed_buffer=torch.empty(8 if vary_tokens else 4, 4),
            forward_event_counters=torch.empty(16),
            backward_event_counters=torch.empty(16),
            gmm_workspace=torch.empty(1),
            swiglu_grad_workspace=torch.empty(1),
        )
        weight1 = torch.ones(2, 4, 4, requires_grad=True)
        weight2 = torch.ones(2, 2, 4, requires_grad=True)
        down_pointers = []
        backward_capacities = []

        def release_workspace(*, stream: Any = None) -> None:
            """Make accidental reads after the lease immediately observable."""
            self.assertIs(stream, workspace.lease_stream)
            workspace.expert_buffer.fill_(float("nan"))
            workspace.routed_buffer.fill_(float("nan"))

        workspace.release.side_effect = release_workspace

        def permutation_gradient(grad: Any, mapping: Any, token_rows: int, top_k: int) -> Any:
            """Require direct consumption of shared rows and return owned token rows."""
            self.assertEqual(grad.data_ptr(), workspace.routed_buffer.data_ptr())
            expected_rows = (1, 3, 2, 4)[int(grad[0, 0]) - 1] if vary_tokens else 2
            self.assertEqual((token_rows, top_k), (expected_rows, 2))
            self.assertEqual(grad.shape[0], token_rows * top_k)
            self.assertEqual(mapping.tolist(), list(range(token_rows * 2)) if vary_tokens else [0, 2, 1, 3])
            self.assertTrue(torch.isfinite(grad).all())
            return grad.reshape(token_rows, top_k, 4).sum(dim=1)

        def forward_kernel(*args: Any) -> None:
            """Fill every output with a call-specific sentinel."""
            dispatch, source = args[0], args[2]
            up_proj, activation, down_proj, combine = args[7], args[8], args[11], args[12]
            tag = source[0, 0].item()
            capacity = up_proj.shape[0]
            self.assertEqual(tuple(down_proj.shape), (capacity, 4))
            self.assertEqual(tuple(activation.shape), (capacity, 2))
            self.assertEqual(tuple(dispatch.shape), (128, 4))
            dispatch.fill_(tag)
            up_proj.fill_(tag + 10)
            activation.fill_(tag + 20)
            down_proj.fill_(tag + 30)
            combine.fill_(tag + 40)
            down_pointers.append(down_proj.data_ptr())

        def backward_kernel(*args: Any) -> None:
            """Check saved values before emulating the in-place gradient outputs."""
            saved_dispatch = args[17]
            capacity = saved_dispatch.shape[0]
            backward_capacities.append(capacity)
            tag = saved_dispatch[0, 0].item()
            self.assertTrue(torch.equal(saved_dispatch, torch.full_like(saved_dispatch, tag)))
            self.assertTrue(torch.equal(args[9], torch.full_like(args[9], tag + 10)))
            self.assertTrue(torch.equal(args[5], torch.full_like(args[5], tag + 20)))
            for tensor, width in ((args[8], 2), (args[10], 4), (args[12], 4)):
                self.assertEqual(tuple(tensor.shape), (capacity, width))
            self.assertEqual(tuple(args[13].shape), (8 if vary_tokens else 4, 4))
            self.assertEqual(torch.count_nonzero(args[6][0]).item(), 0)
            self.assertEqual(torch.count_nonzero(args[18][0]).item(), 0)
            args[13].fill_(tag)
            args[6].fill_(tag)
            args[18].fill_(tag)

        def zero_cold(first: torch.Tensor, second: torch.Tensor, groups: torch.Tensor) -> None:
            """Clear expert gradient slices whose cumulative token counts are unchanged."""
            previous = 0
            for expert, end in enumerate(groups.tolist()):
                if end == previous:
                    first[expert].zero_()
                    second[expert].zero_()
                previous = end

        capacities = (7, 2, 1, 5)
        outputs = []
        sources = []
        with (
            patch.object(function_module.multicore_ops, "zero_empty_expert_gradients", side_effect=zero_cold),
            patch.object(MegaKernelDiagnostics, "launch",
                         side_effect=AssertionError("ordinary path must bypass launch")),
            patch.object(
                function_module.multicore_ops,
                "mega_moe",
                side_effect=forward_kernel,
            ),
            patch.object(
                function_module.multicore_ops,
                "mega_moe_grad",
                side_effect=backward_kernel,
            ),
            patch.object(function_module.multicore_ops, "moe_token_permute_grad",
                         side_effect=permutation_gradient) as mock_permutation,
        ):
            for tag, capacity in enumerate(capacities, start=1):
                tokens = (1, 3, 2, 4)[tag - 1] if vary_tokens else 2
                source = torch.full((tokens if permuted else 4, 4), float(tag), requires_grad=input_grad)
                route = function_module.RouteMetadata(
                    *(torch.zeros(4, dtype=torch.int64) for _ in range(6)),
                    group_list=torch.tensor([0, capacity]), expert_capacity=capacity)
                output = self._check_deferred_forward(
                    source, weight1, weight2, route, plan, workspace, down_pointers,
                    permuted=permuted, vary_tokens=vary_tokens, replay=replay,
                )
                outputs.append(output)
                sources.append(source)
            for tag in range(len(outputs), 0, -1):
                self.assertTrue(torch.equal(outputs[tag - 1], torch.full_like(outputs[tag - 1], float(tag + 40))))
                outputs[tag - 1].sum().backward()
                source = sources[tag - 1]
                expected = torch.full_like(source, float(tag * (2 if permuted else 1)))
                if input_grad:
                    self.assertTrue(torch.equal(source.grad, expected))
                else:
                    self.assertIsNone(source.grad)
        self.assertTrue(torch.equal(weight1.grad, torch.full_like(weight1, 10.0)))
        self.assertTrue(torch.equal(weight2.grad, torch.full_like(weight2, 10.0)))
        self.assertEqual(mock_permutation.call_count, len(outputs) if permuted and input_grad else 0)
        self.assertEqual(backward_capacities, list(reversed(capacities)))
        self.assertEqual(workspace.claim.call_count, 12 if replay else 8)
        self.assertEqual(workspace.release.call_count, 12 if replay else 8)


if __name__ == "__main__":
    unittest.main()
