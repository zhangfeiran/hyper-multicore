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
"""Check B0 peer forward with an independent compact backward publication."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_multicore.modules.mega_moe import function as moe_function
from hyper_multicore.modules.mega_moe.module import MegaMoeExperts
from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.modules.mega_moe.workspace import _workspace_allocation_specs
from tests.common.mark_utils import arg_mark


class TestPeerIndexedBackward(unittest.TestCase):
    """Preserve the existing forward graph and complete indexed backward contract."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_public_constructor_accepts_only_b0_peer_token_and_tile_pairs(self) -> None:
        """Feature: Compact backward after peer forward.
        Description: Construct the intended pair without parameters or device initialization.
        Expectation: Each pair retains both selected modes and allocates no expert parameters.
        """
        for mode in ("peer_token", "peer_tile"):
            with self.subTest(mode=mode):
                module = MegaMoeExperts(hidden_size=5120, intermediate_size=1792, num_experts=32, top_k=8,
                                        ep_size=4, dispatch_mode="pull", create_parameters=False,
                                        forward_ingress_mode=mode, backward_ingress_mode="indexed")
                specification = module._resource_group.specification
                self.assertEqual(specification["forward_ingress_mode"], mode)
                self.assertEqual(specification["backward_ingress_mode"], "indexed")
                self.assertEqual(list(module.parameters()), [])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_incomplete_prepass_push_and_replica_pairs_remain_rejected(self) -> None:
        """Feature: Conservative ingress pairing boundary.
        Description: Request an unsupported forward representation, transport or replica budget.
        Expectation: The constructor rejects each unsupported pair before parameter allocation.
        """
        for mode, dispatch, replicas in (("legacy", "pull", 0), ("peer_dedup", "pull", 0),
                                          ("peer_overlap", "pull", 0), ("peer_token", "push", 0),
                                          ("peer_token", "pull", 1), ("peer_tile", "pull", 2)):
            with self.subTest(mode=mode, dispatch=dispatch, replicas=replicas), self.assertRaises(ValueError):
                MegaMoeExperts(hidden_size=5120, intermediate_size=1792, num_experts=32, top_k=8, ep_size=4,
                               dispatch_mode=dispatch, create_parameters=False, replica_slots_per_rank=replicas,
                               forward_ingress_mode=mode, backward_ingress_mode="indexed")

    @staticmethod
    def allocations(mode: str, tokens: int = 4096, hidden: int = 5120) -> dict:
        """Return ordinary allocation specifications without materializing NPU storage."""
        return dict(_workspace_allocation_specs(
            dispatch_mode="pull", routed_slots=tokens * 8, receive_capacity=tokens * 8, hidden_size=hidden,
            ep_size=4, num_experts=32, dtype=torch.bfloat16, top_k=8,
            forward_ingress_mode=mode, backward_ingress_mode="indexed"))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_compact_source_does_not_resize_forward_routes_or_received_rows(self) -> None:
        """Feature: Independent forward and backward storage.
        Description: Compare paired peer storage with the original peer legacy-backward allocation.
        Expectation: Only source rows become compact and the original indexed backward buffer is appended.
        """
        canonical = self.allocations("indexed")
        for mode in ("peer_token", "peer_tile"):
            with self.subTest(mode=mode):
                compact = self.allocations(mode)
                legacy = dict(_workspace_allocation_specs(
                    dispatch_mode="pull", routed_slots=32768, receive_capacity=32768, hidden_size=5120,
                    ep_size=4, num_experts=32, dtype=torch.bfloat16, top_k=8, forward_ingress_mode=mode))
                self.assertEqual(compact["source_buffer"].shape, (4096, 5120))
                self.assertEqual(legacy["source_buffer"].shape, (32768, 5120))
                self.assertEqual(compact["backward_route_metadata"], canonical["backward_route_metadata"])
                for name in legacy.keys() - {"source_buffer"}:
                    self.assertEqual(compact[name], legacy[name], f"Unrelated allocation changed: {name}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_pairing_preserves_each_complete_forward_and_backward_runtime_image(self) -> None:
        """Feature: Original arithmetic and synchronization preservation.
        Description: Build actual four-rank plans with both shared backward choices and two row counts.
        Expectation: Forward equals the existing peer graph and backward equals canonical indexed backward.
        """
        for tokens in (512, 4096):
            for rank in range(4):
                for shared_backward in (False, True):
                    base = MegaMoeSpec(tokens, 5120, 1792, 32, 8, None, tokens * 8, 4, None, rank, 32,
                                       dispatch_mode="pull", forward_ingress_mode="indexed",
                                       backward_ingress_mode="indexed", shared_intermediate_size=2048,
                                       backward_shared_expert=shared_backward)
                    canonical = _build_runtime_artifacts(base)
                    for mode in ("peer_token", "peer_tile"):
                        with self.subTest(tokens=tokens, rank=rank, shared=shared_backward, mode=mode):
                            paired = _build_runtime_artifacts(replace(base, forward_ingress_mode=mode))
                            legacy = _build_runtime_artifacts(replace(base, forward_ingress_mode=mode,
                                                                      backward_ingress_mode="legacy"))
                            self.assertEqual(bytes(paired[1]), bytes(legacy[1]), "Forward task image changed")
                            self.assertEqual(bytes(paired[3]), bytes(canonical[3]), "Backward task image changed")

    def publication_case(self, mode: str, *, pattern: str, reversed_rows: bool) -> None:
        """Exercise actual output ownership and backward publication through reused CPU buffers."""
        tokens, topk, hidden = 128, 8, 32
        specs = self.allocations(mode, tokens, hidden)
        buffers = {name: torch.zeros(spec.shape, dtype=spec.dtype) for name, spec in specs.items()}
        workspace = SimpleNamespace(**buffers)
        if pattern == "random":
            ids = torch.rand(tokens, 32, generator=torch.Generator().manual_seed(127)).argsort(-1)[:, :topk]
        elif pattern == "hot":
            ids = torch.arange(topk).repeat(tokens, 1)
        else:
            ids = torch.arange(tokens * topk).reshape(tokens, topk) % 32
        groups = [(ids.flatten() == expert).nonzero().flatten() for expert in range(32)]
        slots = torch.cat([group.flip(0) if reversed_rows else group for group in groups]).int()
        inverse = torch.empty_like(slots)
        inverse[slots.long()] = torch.arange(slots.numel(), dtype=torch.int32)
        probabilities = torch.softmax(torch.arange(tokens * topk).float().reshape(tokens, topk) % 7, -1).bfloat16()
        workspace.source_buffer.copy_(torch.arange(tokens * hidden).reshape(tokens, hidden))
        workspace.routed_buffer.copy_(torch.arange(tokens * topk * hidden).reshape(-1, hidden) % 19)
        original_output = workspace.routed_buffer.clone()
        context = SimpleNamespace(has_unpermute=True, source_rows=tokens * topk, token_rows=tokens,
                                  needs_input_grad=(True,) * 12, workspace=workspace,
                                  plan=SimpleNamespace(spec=SimpleNamespace(
                                      dispatch_mode="pull", forward_ingress_mode=mode,
                                      backward_ingress_mode="indexed")))
        execution = SimpleNamespace(combine=workspace.routed_buffer, dispatch=torch.empty(1, hidden))

        def reduce_rows(rows: torch.Tensor, mapping: torch.Tensor, *, probs: torch.Tensor) -> torch.Tensor:
            """Return owned dense output independently of the expanded receive workspace."""
            return (rows[mapping.long()].reshape(tokens, topk, hidden).float() * probs[..., None]).sum(1).bfloat16()

        with patch.object(moe_function, "load_torch_npu", return_value=SimpleNamespace(npu_moe_token_unpermute=reduce_rows)):
            output, _, saved = moe_function._restore_forward_output(
                context, context.plan.spec, execution, probabilities, inverse, slots, (inverse,), True)
        workspace.route_metadata.fill_(-1)
        workspace.routed_buffer.fill_(float("nan"))
        workspace.source_buffer.fill_(float("nan"))
        gradient = (torch.arange(tokens * hidden).reshape(tokens, hidden) % 13).bfloat16()
        with patch.object(moe_function.multicore_ops, "mega_moe_unpermute_grad_out") as expanded:
            published, grad_probabilities = moe_function._stage_backward_source(context, gradient, saved)
        expanded.assert_not_called()
        self.assertEqual(published.data_ptr(), workspace.source_buffer.data_ptr())
        self.assertEqual(published.shape[0], tokens)
        torch.testing.assert_close(published, gradient, rtol=0, atol=0)
        torch.testing.assert_close(saved[1], original_output, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(output).all(), "Workspace reuse invalidated the owned forward output")
        records = workspace.backward_route_metadata[40:40 + tokens * topk]
        torch.testing.assert_close(records[:, 0], slots, rtol=0, atol=0)
        torch.testing.assert_close(records[:, 2], inverse, rtol=0, atol=0)
        weights = records[:, 1].contiguous().view(torch.float32)
        torch.testing.assert_close(weights, probabilities.float().flatten()[slots.long()], rtol=0, atol=0)
        received = (published[records[:, 0].long() // topk].float() * weights[:, None]).bfloat16()
        reference = torch.cat([(gradient[group // topk].float()
                                * probabilities.float().flatten()[group, None]).bfloat16()
                               for group in [value.flip(0) if reversed_rows else value for value in groups]])
        torch.testing.assert_close(received, reference, rtol=0, atol=0)
        header = workspace.backward_route_metadata.view(torch.uint8).reshape(-1)[512:536].view(torch.int64)
        self.assertEqual(header.tolist(), [1, saved[1].data_ptr(), grad_probabilities.data_ptr()])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_backward_publication_survives_forward_metadata_and_source_reuse(self) -> None:
        """Feature: Cross-direction route and output lifetime.
        Description: Poison forward metadata, source and receive buffers before actual backward publication.
        Expectation: Compact dY, scalar probability bits, receiver row reconstruction and owned output stay exact.
        """
        for mode in ("peer_token", "peer_tile"):
            for pattern in ("balanced", "hot", "random"):
                for reverse in (False, True):
                    with self.subTest(mode=mode, pattern=pattern, reverse=reverse):
                        self.publication_case(mode, pattern=pattern, reversed_rows=reverse)
