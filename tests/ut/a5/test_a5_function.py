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

import struct
import unittest
import weakref
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import torch
from torch.utils.checkpoint import DefaultDeviceType, checkpoint

from hyper_multicore.replica.recompute import replica_checkpoint as hyper_checkpoint
from hyper_multicore.replica.recompute import create_recompute_contexts
from hyper_multicore.replica import build_expert_replica_plan
from hyper_multicore.replica.signal_transport import (
    SignalReplicaTransport,
    signal_storage_bytes,
)
from hyper_multicore.diagnostics import MegaKernelDiagnostics
from hyper_multicore.modules.mega_moe import function as function_module
from hyper_multicore.modules.mega_moe.function import _MegaMoeFunction
from tests.common.mark_utils import arg_mark


# The private autograd bridge and saved execution buffers are the contracts under test.
class TestMegaMoeFunction(unittest.TestCase):
    """Exercise real autograd contexts with mocked communication and kernels."""

    # pylint: disable=protected-access

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_no_grad_unpermutation_owns_outputs_without_saving_expanded_rows(self) -> None:
        """Feature: No-grad integrated output ownership.
        Description: Reuse and poison SHMEM after two no-grad forwards with trainable weights.
        Expectation: Dense outputs stay exact without saving or cloning the expanded receive buffer.
        """
        spec = SimpleNamespace(dispatch_mode="pull", forward_ingress_mode="legacy", hidden_size=4,
                               intermediate_size=2, top_k=2, forward_shared_expert=True,
                               backward_shared_expert=False, expert_weight_layout="EIO",
                               replica_slots_per_rank=0, auto_tokens=False)
        plan = SimpleNamespace(spec=spec, ingress_diagnostics=None)
        workspace = Mock(source_buffer=torch.zeros(4, 4), routed_buffer=torch.zeros(4, 4),
                         gmm_workspace=torch.zeros(1))
        workspace.release.side_effect = lambda: workspace.routed_buffer.fill_(float("nan"))
        hidden = torch.ones(2, 4)
        first, second = torch.ones(2, 4, 4, requires_grad=True), torch.ones(2, 2, 4, requires_grad=True)
        metadata = SimpleNamespace(expert_capacity=4, group_list=torch.tensor([2, 4]), replica_route=None)
        for name in ("dispatch_src_off", "dispatch_target_off", "dispatch_size",
                     "combine_src_off", "combine_target_off", "combine_size"):
            setattr(metadata, name, torch.zeros(2, dtype=torch.int64))
        route = SimpleNamespace(metadata=metadata, routed_tokens=hidden.repeat_interleave(2, dim=0),
                                unpermute_mapping=torch.arange(4, dtype=torch.int32))
        probabilities = torch.tensor([[0.25, 0.75], [0.25, 0.75]])
        expected = torch.arange(16).reshape(2, 2, 4).float().mul(probabilities.unsqueeze(-1)).sum(1)
        launch_count = 0

        def launch(
            _plan: Any, _route: Any, _source: Any, _first: Any, _second: Any,
            execution: Any, _pool: Any = None,
        ) -> None:
            """Write distinct complete receive buffers on each invocation.

            Args:
                _plan: Unused device plan.
                _route: Unused communication metadata.
                _source: Unused published source.
                _first: Unused expert weight.
                _second: Unused expert weight.
                execution: Receive buffer to poison after output consumption.
                _pool: Unused replica pool.
            """
            nonlocal launch_count
            execution.combine.copy_(torch.arange(16).reshape(4, 4) + 100 * launch_count)
            launch_count += 1

        def unpermute(rows: torch.Tensor, _mapping: Any, *, probs: torch.Tensor) -> torch.Tensor:
            """Reduce expanded rows into independent dense storage.

            Args:
                rows: Current receive buffer.
                _mapping: Unused identity route mapping.
                probs: Per-token route probabilities.
            """
            return (rows.reshape(2, 2, 4) * probs.unsqueeze(-1)).sum(1)

        old_clone = torch.Tensor.clone
        expanded_clones = []

        def clone(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            """Count owned copies of the expanded receive view.

            Args:
                tensor: Tensor whose storage is copied.
                args: Positional clone options.
                kwargs: Keyword clone options.
            """
            if tuple(tensor.shape) == (4, 4):
                expanded_clones.append(tensor.data_ptr())
            return old_clone(tensor, *args, **kwargs)

        with (patch.object(torch.Tensor, "clone", new=clone),
              patch.object(function_module, "_launch_forward_kernel", side_effect=launch),
              patch.object(function_module, "_save_forward_state", wraps=function_module._save_forward_state) as save,
              patch.object(function_module, "load_torch_npu", return_value=SimpleNamespace(npu_moe_token_unpermute=unpermute))):
            outputs = []
            for context in (torch.no_grad, torch.inference_mode):
                with context():
                    output = function_module.execute_mega_moe_with_permutation(
                        hidden, torch.tensor([[0, 1], [0, 1]]), first, second, route, plan, workspace,
                        topk_weights=probabilities)
                outputs.append(output)
            torch.testing.assert_close(outputs[0], expected, rtol=0, atol=0)
            torch.testing.assert_close(outputs[1], expected + 100, rtol=0, atol=0)
            self.assertNotEqual(outputs[0].data_ptr(), workspace.routed_buffer.data_ptr())
            self.assertFalse(outputs[0].requires_grad)
            self.assertEqual(expanded_clones, [], f"expected=[] actual={expanded_clones}")
            save.assert_not_called()
            workspace.track_graph.assert_not_called()

    def test_ragged_pull_uses_reserved_source_prefix(self) -> None:
        """Forward and backward retain SHMEM addresses while exposing only actual rows."""
        buffer = torch.full((256, 4), float("nan"))
        workspace = SimpleNamespace(source_buffer=buffer)
        spec = SimpleNamespace(dispatch_mode="pull", hidden_size=4)
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

    def test_replica_home_initialization_only_clears_skipped_experts(self) -> None:
        """Retain zeros for skipped GMMs and preserve the ordinary baseline initialization."""
        spec = SimpleNamespace(expert_weight_layout="EIO", replica_slots_per_rank=1, intermediate_size=3,
                               hidden_size=4)
        weight1, weight2 = torch.ones(3, 4, 6), torch.ones(3, 3, 4)
        for empty in (None, (), (1,), (0, 1, 2)):
            with self.subTest(empty=empty):
                with patch.object(function_module.torch, "empty_like",
                                  side_effect=lambda value, **kwargs: torch.full_like(value, float("nan"), **kwargs)):
                    result = function_module._allocate_backward_intermediates(
                        spec, 4, torch.ones(4, 4), weight1, weight2, empty_experts=empty)
                for value in (result.grad_weight1, result.grad_weight2):
                    for expert in range(3):
                        if empty is None or expert in empty:
                            torch.testing.assert_close(value[expert], torch.zeros_like(value[expert]))
                        else:
                            self.assertTrue(torch.isnan(value[expert]).all())

    def test_eoi_weight_operands_and_gradients_are_zero_copy_views(self) -> None:
        """Native EOI operands retain storage and gradients restore caller orientation."""
        spec = SimpleNamespace(
            expert_weight_layout="EOI",
            replica_slots_per_rank=0,
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
        restored_gate_up = function_module._restore_weight_gradient_layout(
            spec, gradients.grad_weight1
        )
        restored_down = function_module._restore_weight_gradient_layout(
            spec, gradients.grad_weight2
        )
        self.assertEqual(tuple(restored_gate_up.shape), tuple(gate_up.shape))
        self.assertEqual(tuple(restored_down.shape), tuple(down.shape))
        self.assertEqual(
            restored_gate_up.data_ptr(), gradients.grad_weight1.data_ptr()
        )
        self.assertEqual(restored_down.data_ptr(), gradients.grad_weight2.data_ptr())

    def test_checkpoint_backward_reuses_replayed_replica_weights(self) -> None:
        """Replay refreshes the original context without a third weight transfer."""
        self.addCleanup(DefaultDeviceType.set_device_type, DefaultDeviceType.get_device_type())
        DefaultDeviceType.set_device_type("cpu")
        for engine in ("off", "torch", "hyper"):
            for pull in (False, True):
                for calls in (1, 2):
                    with self.subTest(engine=engine, pull=pull, calls=calls):
                        self._check_replica_recompute(engine=engine, pull=pull, calls=calls)


    def test_replayed_generation_requires_matching_placement_and_invocation(self) -> None:
        """Different placements, workspaces and checkpoint invocations cannot share a generation."""
        route = SimpleNamespace(plan=build_expert_replica_plan([[8, 0], [0, 0]], 1))
        changed = SimpleNamespace(plan=build_expert_replica_plan([[0, 0], [0, 8]], 1))
        workspace, other_workspace = object(), object()
        first, second, third = object(), object(), object()
        forward_context, replay_context = create_recompute_contexts()
        with forward_context:
            remembered = function_module._remember_replica_generation(first, route, workspace)
            skipped = function_module._remember_replica_generation(second, route, workspace)
            following = function_module._remember_replica_generation(second, route, workspace)
            other = function_module._remember_replica_generation(second, route, other_workspace)
        with replay_context:
            function_module._remember_replica_generation(third, changed, workspace)
            function_module._remember_replica_generation(None, route, workspace)
            function_module._remember_replica_generation(third, route, workspace)
            function_module._remember_replica_generation(third, route, other_workspace)
        self.assertIsNone(remembered.value)
        self.assertIsNone(skipped.value)
        self.assertIs(following.value, third)
        self.assertIs(other.value, third)
        # Clearing replay resources must not clear the holder still owned by backward.
        with replay_context:
            function_module._remember_replica_generation(first, route, other_workspace)
        self.assertIs(other.value, third)
        with create_recompute_contexts()[0]:
            isolated = function_module._remember_replica_generation(first, route, other_workspace)
        self.assertIsNot(isolated, other)
        self.assertIs(isolated.value, first)
        self.assertIsNone(function_module._remember_replica_generation(None, None, workspace))
        self.assertIs(function_module._remember_replica_generation(first, route, workspace), first)


    def _check_replica_recompute(self, *, engine: str, pull: bool, calls: int) -> None:
        """Exercise the real autograd bridge and signal protocol with CPU kernels."""
        weights = (torch.ones(1, 2, 2, requires_grad=True), torch.ones(1, 1, 2, requires_grad=True))
        shapes = tuple(tuple(weight.shape[1:]) for weight in weights)
        storage = torch.empty(signal_storage_bytes(shapes, 1, 2, 4), dtype=torch.uint8)
        provider = SignalReplicaTransport(MagicMock(), storage, 1, 2)
        replica = SimpleNamespace(plan=build_expert_replica_plan([[8, 0], [0, 0]], 1), rank=0)
        self.assertTrue(replica.plan.transfers)
        workspace = Mock(replica_provider=provider, in_use=False, replica_runtime_images={},
                         source_buffer=torch.empty(2, 2), expert_buffer=torch.empty(2, 2),
                         routed_buffer=torch.empty(2, 2), gmm_workspace=torch.empty(1),
                         swiglu_grad_workspace=torch.empty(1))

        def claim() -> None:
            """Reject overlapping execution on the shared workspace."""
            self.assertFalse(workspace.in_use)
            workspace.in_use = True

        def release() -> None:
            """End the current workspace lease."""
            self.assertTrue(workspace.in_use)
            workspace.in_use = False

        workspace.claim.side_effect = claim
        workspace.release.side_effect = release
        spec = SimpleNamespace(replica_slots_per_rank=1, hidden_size=2, intermediate_size=1,
                               top_k=1, dispatch_mode="pull" if pull else "push", auto_tokens=False, expert_weight_layout="EIO")
        runtime = SimpleNamespace(normal_tensor=None)
        plan = SimpleNamespace(spec=spec, fwd_task_prefixes=(), bwd_task_prefixes=(),
                               fwd_diagnostics=MegaKernelDiagnostics(runtime),
                               bwd_diagnostics=MegaKernelDiagnostics(runtime), reuse_backward_dispatch=False)
        route = SimpleNamespace(replica_route=replica, expert_capacity=2, group_list=torch.tensor([2]))
        for name in ("dispatch_src_off", "dispatch_target_off", "dispatch_size",
                     "combine_src_off", "combine_target_off", "combine_size"):
            setattr(route, name, torch.zeros(2, dtype=torch.int64))

        def forward_kernel(_plan: Any, _route: Any, source: Any, weight1: Any, weight2: Any,
                           execution: Any, _pool: Any) -> None:
            """Emulate a differentiable scalar projection into the real saved buffers."""
            execution.dispatch.copy_(source)
            execution.combine.copy_(source * (weight1.sum() + weight2.sum()))
            execution.intermediates.up_proj.zero_()
            execution.intermediates.activation.zero_()

        def backward_kernel(_plan: Any, saved: Any, gradient: Any, execution: Any,
                            _pool: Any, _gradient_return: Any = None) -> None:
            """Differentiate the CPU projection using the checkpoint-restored inputs."""
            execution.grad_x.copy_(gradient * (saved.weight1.sum() + saved.weight2.sum()))
            value = (gradient * saved.dispatch).sum()
            execution.intermediates.grad_weight1.fill_(value)
            execution.intermediates.grad_weight2.fill_(value)

        source = torch.arange(4, dtype=torch.float32).reshape(2, 2).requires_grad_()

        def operation(value: torch.Tensor, weight1: torch.Tensor, weight2: torch.Tensor) -> torch.Tensor:
            """Run serial MoE calls through one checkpoint region and shared workspace."""
            for _ in range(calls):
                value = function_module.execute_mega_moe(value, weight1, weight2, route, plan, workspace)
            return value

        with (
            patch.object(torch, "cpu", MagicMock()),
            patch.object(torch.npu, "current_stream"),
            patch.object(torch.Tensor, "record_stream", autospec=True),
            patch.object(type(provider), "kernel_gradients", new_callable=PropertyMock, return_value=False),
            patch.object(provider, "_overlapped_prefetch", wraps=provider._overlapped_prefetch) as transfers,
            patch.object(function_module, "_launch_forward_kernel", side_effect=forward_kernel),
            patch.object(function_module, "_launch_backward_kernel", side_effect=backward_kernel),
            patch.object(function_module, "return_gradients", side_effect=lambda gradients, *_args, **_kw: gradients),
        ):
            if engine == "torch":
                output = checkpoint(operation, source, *weights, use_reentrant=False,
                                    context_fn=create_recompute_contexts)
            elif engine == "hyper":
                output = hyper_checkpoint(operation, source, *weights, use_reentrant=False)
            else:
                output = operation(source, *weights)
            output.sum().backward()
        torch.testing.assert_close(output, source.detach() * 6 ** calls)
        torch.testing.assert_close(source.grad, torch.full_like(source, 6 ** calls))
        for weight in weights:
            torch.testing.assert_close(weight.grad, torch.full_like(weight, calls * 6 ** calls))
        self.assertEqual(transfers.call_count, (2 if engine == "off" else 3) * calls - 1)
        self.assertFalse(workspace.in_use)


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_backward_unpacks_saved_tensors_once(self) -> None:
        """Feature: MegaKernel diagnostics.
        Description: Checkpoint unpack hooks must only be consumed once per backward call.
        Expectation: Backward unpacks saved tensors once.
        """
        tensors = tuple(torch.ones(1) for _ in range(13))
        context = Mock()
        gradient = torch.ones(1)
        context.plan.spec.dispatch_mode = "push"
        context.plan.spec.replica_slots_per_rank = 0
        context.replica_route = None
        context.workspace.replica_provider = None
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
        context.workspace.release.assert_called_once_with()

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_static_runtime_cache_reuses_image_and_bounds_replacement(self) -> None:
        """
        Feature: Static runtime caching
        Description: Reuse writable scratch, then change base identity and suffix pointers.
        Expectation: Reuse identical images and replace only changed entries.
        """
        base = torch.arange(16, dtype=torch.uint8)
        cache = {}
        with patch.object(torch.npu, "current_stream"), patch.object(torch.Tensor, "record_stream", autospec=True):
            first = function_module._split_runtime(base, None, 6, home_only=True, cache=cache)
            first[base.numel() - 1] = 93
            repeated = function_module._split_runtime(base, None, 6, home_only=True, cache=cache)
            self.assertIs(repeated, first)
            self.assertEqual(repeated[base.numel() - 1].item(), 93)
            self.assertEqual(base[-1].item(), 15)
            weights, gradients = (torch.empty(1), torch.empty(2)), (torch.empty(3), torch.empty(4))
            pool = SimpleNamespace(weights=weights, gradients=gradients, projection_ready=None)
            active = function_module._split_runtime(base, pool, 6, cache=cache)
            self.assertIsNot(active, first)
            self.assertIs(function_module._split_runtime(base, pool, 6, cache=cache), active)
            backward = function_module._split_runtime(base, pool, 6, backward=True, cache=cache)
            self.assertIsNot(backward, active)
            self.assertEqual(len(cache), 1)
            self.assertEqual(struct.unpack("<II5Q", bytes(active[base.numel():].tolist()))[-2:], (0, 0))
            self.assertEqual(struct.unpack("<II5Q", bytes(backward[base.numel():].tolist()))[-2:],
                             tuple(t.data_ptr() for t in gradients))
            other_base = base.clone()
            other = function_module._split_runtime(other_base, pool, 6, cache=cache)
            self.assertIsNot(other, active)
            self.assertEqual(len(cache), 2)
            self.assertIs(cache[id(base)][0], base)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0",
              card_mark="onecard", essential_mark="essential")
    def test_dynamic_runtime_descriptors_update_suffix_without_resetting_scratch(self) -> None:
        """
        Feature: Dynamic runtime ownership
        Description: Advance forward and backward projection epochs while reusing a serial workspace image.
        Expectation: Update the suffix while retaining the kernel's writable scratch.
        """
        base = torch.arange(16, dtype=torch.uint8)
        weights, gradients = (torch.empty(1), torch.empty(2)), (torch.empty(3), torch.empty(4))
        descriptor = torch.empty(16, dtype=torch.uint8)
        for version in (4, 5):
            cache = {}
            pool = SimpleNamespace(weights=weights, gradients=gradients,
                                   projection_ready=((4096, 8192), 1))
            options = {"backward": True, "gradient_return": descriptor} if version == 5 else {}
            with patch.object(torch.npu, "current_stream"), patch.object(torch.Tensor, "record_stream", autospec=True):
                first = function_module._split_runtime(base, pool, 6, cache=cache, **options)
                first[base.numel() - 1] = 93
                pool.projection_ready = ((4096, 8192), 2)
                if version == 5:
                    options["gradient_return"] = descriptor.clone()
                second = function_module._split_runtime(base, pool, 6, cache=cache, **options)
                independent = function_module._split_runtime(base, pool, 6, **options)
            self.assertIs(first, second)
            self.assertEqual(second[base.numel() - 1].item(), 93)
            self.assertEqual(len(cache), 1)
            torch.testing.assert_close(second[base.numel():], independent[base.numel():])
            self.assertEqual(independent[base.numel() - 1].item(), 15)
            if version == 5:
                encoded = struct.unpack("<II9Q", bytes(second[base.numel():].tolist()))
                self.assertEqual(encoded[-2:], (2, options["gradient_return"].data_ptr()))

    def test_dynamic_runtime_reallocates_when_descriptor_layout_changes(self) -> None:
        """An ABI suffix-size change cannot overwrite the next allocation or stale fields."""
        base, cache = torch.arange(16, dtype=torch.uint8), {}
        pool = SimpleNamespace(weights=(torch.empty(1), torch.empty(2)),
                               gradients=(torch.empty(3), torch.empty(4)),
                               projection_ready=None)
        with patch.object(torch.npu, "current_stream"), patch.object(torch.Tensor, "record_stream"):
            first = function_module._split_runtime(base, pool, 6, cache=cache)
            pool.projection_ready = ((4096, 8192), 2)
            second = function_module._split_runtime(base, pool, 6, cache=cache)
        self.assertIsNot(first, second)
        self.assertEqual(second.numel() - first.numel(), 24)
        self.assertEqual(len(cache), 1)

    def test_split_runtime_p2p_encodes_home_and_guest_pointers(self) -> None:
        """P2P weights are ready before launch; backward adds the FP32 guest addresses."""
        base = torch.arange(16, dtype=torch.uint8)
        weights, gradients = (torch.empty(1), torch.empty(2)), (torch.empty(3), torch.empty(4))
        pool = SimpleNamespace(weights=weights, gradients=gradients, projection_ready=None)
        for backward in (False, True):
            with self.subTest(backward=backward), patch.object(torch.npu, "current_stream"), \
                    patch.object(torch.Tensor, "record_stream"):
                result = function_module._split_runtime(base, pool, 6, backward=backward)
            pointers = tuple(g.data_ptr() for g in gradients) if backward else (0, 0)
            torch.testing.assert_close(result[:16], base)
            self.assertEqual(struct.unpack("<II5Q", bytes(result[16:].tolist())),
                             (0x53505754, 2, 6, *(w.data_ptr() for w in weights), *pointers))

    def test_split_runtime_encodes_independent_projection_ready_v4(self) -> None:
        """Pass distinct W13/W2 cache-line bases and one epoch to forward/backward."""
        base = torch.arange(16, dtype=torch.uint8)
        weights = (torch.empty(1), torch.empty(2))
        gradients = (torch.empty(3), torch.empty(4))
        pool = SimpleNamespace(weights=weights, gradients=gradients, projection_ready=((4096, 8192), 23))
        for backward in (False, True):
            with self.subTest(backward=backward), patch.object(torch.npu, "current_stream"), \
                    patch.object(torch.Tensor, "record_stream"):
                result = function_module._split_runtime(base, pool, 6, backward=backward)
            torch.testing.assert_close(result[:16], base)
            pointers = tuple(g.data_ptr() for g in gradients) if backward else (0, 0)
            self.assertEqual(struct.unpack("<II8Q", bytes(result[16:].tolist())),
                             (0x53505754, 4, 6, *(w.data_ptr() for w in weights), *pointers, 4096, 8192, 23))

    def test_split_runtime_v5_retains_projection_prefix_and_gradient_descriptor(self) -> None:
        """Backward adds one invocation descriptor without changing existing projection addresses."""
        base = torch.arange(16, dtype=torch.uint8)
        weights = (torch.empty(1), torch.empty(2))
        gradients = (torch.empty(3), torch.empty(4))
        descriptor = torch.empty(128, dtype=torch.uint8)
        pool = SimpleNamespace(weights=weights, gradients=gradients, projection_ready=((4096, 8192), 23))
        with patch.object(torch.npu, "current_stream"), patch.object(torch.Tensor, "record_stream"):
            result = function_module._split_runtime(base, pool, 6, backward=True, gradient_return=descriptor)
        torch.testing.assert_close(result[:16], base)
        self.assertEqual(struct.unpack("<II9Q", bytes(result[16:].tolist())),
                         (0x53505754, 5, 6, *(w.data_ptr() for w in weights),
                          *(g.data_ptr() for g in gradients), 4096, 8192, 23, descriptor.data_ptr()))

    def test_kernel_return_requires_backward_projection_readiness(self) -> None:
        """Reject a descriptor that the consumer cannot safely interpret before allocation."""
        base, descriptor = torch.empty(16, dtype=torch.uint8), torch.empty(128, dtype=torch.uint8)
        for backward, pool in ((False, SimpleNamespace(projection_ready=((4096, 8192), 1))),
                               (True, SimpleNamespace(projection_ready=None))):
            with self.assertRaisesRegex(ValueError, "projection-ready backward"):
                function_module._split_runtime(base, pool, 6, backward=backward, gradient_return=descriptor)

    def test_backward_selects_runtime_from_global_replica_presence(self) -> None:
        """Repeated empty/nonempty routes choose schedules without mutating the plan."""
        regular, overlap = object(), object()
        plan = SimpleNamespace(spec=SimpleNamespace(dispatch_mode="push"), bwd_diagnostics=overlap,
                               bwd_diagnostics_no_replica=regular, reuse_backward_dispatch=False,
                               bwd_runtime_no_replica_reuse_dispatch=True)
        saved = SimpleNamespace(dispatch=torch.empty(3, 4), weight1=None, weight2=None)
        workspace = Mock(expert_buffer=torch.empty(3, 4), routed_buffer=torch.empty(3, 4),
                         gmm_workspace=torch.empty(1), swiglu_grad_workspace=torch.empty(1))
        with patch.object(function_module, "_allocate_backward_intermediates") as allocate:
            for transfers, w13 in ((False, True), (True, True), (True, False), (False, False), (True, True)):
                execution = function_module._prepare_backward_execution(
                    workspace, plan, saved, torch.empty(3, 4), has_replica_transfers=transfers, overlap_w13=w13)
                self.assertIs(execution.diagnostics, overlap if transfers and w13 else regular)
                if transfers and w13:
                    self.assertIsNone(allocate.call_args.args[-1])
                else:
                    self.assertEqual(allocate.call_args.args[-1].data_ptr(), workspace.expert_buffer.data_ptr())
            plan.bwd_diagnostics_no_replica = None
            execution = function_module._prepare_backward_execution(
                workspace, plan, saved, torch.empty(3, 4), has_replica_transfers=False)
            self.assertIs(execution.diagnostics, overlap)

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
        self._check_deferred_backward(permuted=False, replay=True)

    def test_empty_replica_plan_skips_leases_through_reversed_backward(self) -> None:
        """B stays enabled while an immutable global plan has no transfers."""
        for pull in (False, True):
            with self.subTest(pull=pull):
                self._check_deferred_backward(permuted=False, pull=pull, empty_replica_route=True)

    def test_saved_parameter_version_is_checked_before_claiming_backward_storage(self) -> None:
        """A mutated saved parameter must abort before any resident replica can be borrowed."""
        self._check_deferred_backward(permuted=False, mutate_before_backward=True)




    def _check_deferred_forward(
        self, source: Any, weight1: Any, weight2: Any, route: Any, plan: Any, workspace: Any,
        down_pointers: list[int],
        *, permuted: bool, vary_tokens: bool, replay: bool,
    ) -> Any:
        """Run one forward and check that its autograd storage survives workspace reuse."""
        tokens = source.shape[0]
        if permuted:
            ids = torch.tensor([[0, 1]] * tokens, dtype=torch.int32)
            prepared = SimpleNamespace(
                routed_tokens=source.detach().repeat_interleave(2, dim=0),
                metadata=route,
                unpermute_mapping=(torch.arange(tokens * 2, dtype=torch.int32) if vary_tokens
                                   else torch.tensor([0, 2, 1, 3], dtype=torch.int32)),
            )
            output = function_module.execute_mega_moe_with_permutation(
                source, ids, weight1, weight2, prepared, plan, workspace
            )
        elif replay:
            operation = partial(function_module.execute_mega_moe, route=route, plan=plan, workspace=workspace)
            output = checkpoint(operation, source, weight1, weight2, use_reentrant=False)
        else:
            output = function_module.execute_mega_moe(source, weight1, weight2, route, plan, workspace)
        if not replay:
            saved = output.grad_fn.saved_tensors
            self.assertEqual(len(saved), 13 if permuted and source.requires_grad else 12)
            self.assertEqual(saved[0].data_ptr(), down_pointers[-1])
            self.assertNotEqual(saved[0].data_ptr(), workspace.expert_buffer.data_ptr())
            self.assertEqual(saved[0].untyped_storage().nbytes(), route.expert_capacity * 4 * source.element_size())
        self.assertEqual(tuple(output.shape), (tokens * 2 if permuted else 4, 4))
        return output

    def _check_deferred_backward(
        self, *, permuted: bool, input_grad: bool = True, reuse: bool = False, pull: bool = False,
        empty_replica_route: bool = False, mutate_before_backward: bool = False,
        vary_tokens: bool = False, replay: bool = False,
    ) -> None:
        """Exercise delayed backward with an optional input permutation boundary."""
        spec = SimpleNamespace(replica_slots_per_rank=int(empty_replica_route), hidden_size=4,
                               intermediate_size=2, rank_id=0,
                               ep_size=2, num_experts=4, local_num_tokens=2, top_k=2,
                               dispatch_mode="pull" if pull else "push", expert_weight_layout="EIO", auto_tokens=False)
        plan = SimpleNamespace(spec=spec, reuse_backward_dispatch=reuse, fwd_task_prefixes=(), bwd_task_prefixes=())
        for name in ("up_proj", "swiglu", "down_proj", "act_grad", "gate_grad",
                     "w1_grad", "w2_grad", "swiglu_grad"):
            setattr(plan, f"{name}_tiling", None)
        runtime = SimpleNamespace(normal_tensor=torch.empty(16, dtype=torch.uint8) if empty_replica_route else None, kernel_name="mega_moe", rank=0)
        plan.fwd_diagnostics = MegaKernelDiagnostics(runtime)
        plan.bwd_diagnostics = MegaKernelDiagnostics(runtime)
        plan.fwd_runtime_config = plan.bwd_runtime_config = runtime.normal_tensor
        workspace = Mock(
            replica_inbox=None,
            replica_provider=None,
            replica_runtime_images={},
            in_use=False,
            expert_capacity=128,
            source_buffer=torch.empty(4, 4),
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
        dispatch_pointers = []
        backward_capacities = []
        scratch_references = []
        restore_gradient = function_module._restore_input_gradient  # pylint: disable=protected-access

        def restore_without_scratch(*args: Any) -> Any:
            """Require ordinary scratch and the optional SHMEM view to be released."""
            self.assertTrue(scratch_references)
            self.assertTrue(all(reference() is None for reference in scratch_references))
            return restore_gradient(*args)

        def release_workspace() -> None:
            """Make accidental reads after the lease immediately observable."""
            self.assertTrue(workspace.in_use)
            workspace.in_use = False
            workspace.source_buffer.fill_(float("nan"))
            workspace.expert_buffer.fill_(float("nan"))
            workspace.routed_buffer.fill_(float("nan"))

        def claim_workspace() -> None:
            """Reject nested claims while exercising the real autograd bridge."""
            self.assertFalse(workspace.in_use)
            workspace.in_use = True

        workspace.claim.side_effect = claim_workspace
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
            self.assertEqual(tuple(dispatch.shape), (capacity if pull else 128, 4))
            dispatch.fill_(tag)
            up_proj.fill_(tag + 10)
            activation.fill_(tag + 20)
            down_proj.fill_(tag + 30)
            combine.fill_(tag + 40)
            down_pointers.append(down_proj.data_ptr())
            dispatch_pointers.append(dispatch.data_ptr())

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
            self.assertEqual(args[12].data_ptr() == args[0].data_ptr(), reuse)
            self.assertNotEqual(saved_dispatch.data_ptr(), args[0].data_ptr())
            scratch_references[:] = [weakref.ref(args[index]) for index in (8, 10, 12)]
            self.assertEqual(torch.count_nonzero(args[6]).item(), 0)
            self.assertEqual(torch.count_nonzero(args[18]).item(), 0)
            args[13].fill_(tag)
            args[6].fill_(tag)
            args[18].fill_(tag)

        def _forward(source: torch.Tensor, route: Any) -> torch.Tensor:
            """Borrow the caller's lease when routing directly into pull source."""
            if not permuted:
                return function_module.execute_mega_moe(source, weight1, weight2, route, plan, workspace)
            tokens = source.shape[0]
            ids = torch.tensor([[0, 1]] * tokens, dtype=torch.int32)
            prepared = SimpleNamespace(
                routed_tokens=source.detach().repeat_interleave(2, dim=0), metadata=route,
                unpermute_mapping=(torch.arange(tokens * 2, dtype=torch.int32) if vary_tokens
                                   else torch.tensor([0, 2, 1, 3], dtype=torch.int32)),
            )
            if pull:
                workspace.claim()
                workspace.source_buffer.copy_(prepared.routed_tokens)
                prepared.routed_tokens = workspace.source_buffer
            output = function_module.execute_mega_moe_with_permutation(
                source, ids, weight1, weight2, prepared, plan, workspace, workspace_claimed=pull,
            )
            if pull:
                self.assertTrue(workspace.in_use)
                workspace.release()
            return output

        capacities = (7, 2, 1, 5)
        outputs = []
        sources = []
        with (
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
                new=backward_kernel,
            ),
            patch.object(torch.npu, "current_stream"),
            patch.object(torch.Tensor, "record_stream", autospec=True),
            patch.object(function_module, "prefetch_weights", side_effect=AssertionError("Unexpected replica lease")),
            patch.object(function_module, "return_gradients", side_effect=AssertionError("Unexpected replica return")),
            patch.object(function_module, "_restore_input_gradient", new=restore_without_scratch),
            patch.object(function_module.multicore_ops, "moe_token_permute_grad",
                         side_effect=permutation_gradient) as mock_permutation,
        ):
            for tag, capacity in enumerate(capacities, start=1):
                tokens = (1, 3, 2, 4)[tag - 1] if vary_tokens else 2
                source = torch.full((tokens if permuted else 4, 4), float(tag), requires_grad=input_grad)
                replica = SimpleNamespace(plan=SimpleNamespace(transfers=())) if empty_replica_route else None
                route = SimpleNamespace(replica_route=replica, expert_capacity=capacity, group_list=torch.tensor([0, capacity]))
                for name in ("dispatch_src_off", "dispatch_target_off", "dispatch_size",
                             "combine_src_off", "combine_target_off", "combine_size"):
                    setattr(route, name, torch.zeros(4, dtype=torch.int64))
                output = self._check_deferred_forward(
                    source, weight1, weight2, route, plan, workspace, dispatch_pointers if pull else down_pointers,
                    permuted=permuted, vary_tokens=vary_tokens, replay=replay,
                )
                outputs.append(output)
                sources.append(source)
                self.assertEqual(tuple(output.shape), (tokens * 2 if permuted else 4, 4))
                if replay:
                    continue
                saved = output.grad_fn.saved_tensors
                self.assertEqual(len(saved), 13 if permuted and input_grad else 12)
                self.assertTrue(all(t.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
                                    for t in saved))
                self.assertEqual(saved[0].data_ptr(), (dispatch_pointers if pull else down_pointers)[-1])
                self.assertNotEqual(saved[0].data_ptr(), workspace.expert_buffer.data_ptr())
                self.assertEqual(saved[0].untyped_storage().nbytes(), capacity * 4 * source.element_size())
            if mutate_before_backward:
                claims = workspace.claim.call_count
                with torch.no_grad():
                    weight1.add_(1)
                with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
                    outputs[-1].sum().backward()
                self.assertEqual(workspace.claim.call_count, claims)
                return
            for tag in range(len(outputs), 0, -1):
                self.assertTrue(torch.equal(outputs[tag - 1], torch.full_like(outputs[tag - 1], float(tag + 40))))
                outputs[tag - 1].sum().backward()
                self._assert_input_gradient(sources[tag - 1], tag, permuted=permuted, input_grad=input_grad)
        self.assertTrue(torch.equal(weight1.grad, torch.full_like(weight1, 10.0)))
        self.assertTrue(torch.equal(weight2.grad, torch.full_like(weight2, 10.0)))
        self.assertEqual(mock_permutation.call_count, len(outputs) if permuted and input_grad else 0)
        self.assertEqual(backward_capacities, list(reversed(capacities)))
        self.assertEqual(workspace.claim.call_count, 12 if replay else 8)
        self.assertEqual(workspace.release.call_count, 12 if replay else 8)

    def _assert_input_gradient(
        self, source: torch.Tensor, tag: int, *, permuted: bool, input_grad: bool,
    ) -> None:
        """Check reduced token gradients or the frozen-input contract."""
        if input_grad:
            expected = torch.full_like(source, float(tag * (2 if permuted else 1)))
            self.assertTrue(torch.equal(source.grad, expected))
        else:
            self.assertIsNone(source.grad)


if __name__ == "__main__":
    unittest.main()
