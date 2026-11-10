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
"""Unit tests for MegaMoe route preparation."""

from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import torch
from torch.utils.checkpoint import DefaultDeviceType

from hyper_multicore.replica.recompute import replica_checkpoint as checkpoint
from hyper_multicore.replica import routing as replica_routing

from hyper_multicore.replica import build_expert_replica_plan
from hyper_multicore.replica.routing import (
    ReplicaRoute, _GatheredReplicaCounts, prepare_replica_route,
)
from hyper_multicore.replica.capacity import ExpertReplicaConfig
from hyper_multicore.modules.mega_moe import route as route_module
from hyper_multicore.modules.mega_moe.route import (
    _expert_capacity,
    _resolve_counts,
    prepare_topk_route,
)
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from tests.common.mark_utils import arg_mark


class TestMegaMoeRoute(unittest.TestCase):
    """Validate trusted Router counts and bounded capacity without hardware."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_automatic_route_grows_before_writing_pull_source(self) -> None:
        """Feature: MegaMoe route preparation.

        Description: One count exchange establishes capacity before a pull view can reference SHMEM.
        Expectation: Count agreement precedes growth and every pull write uses the new source buffer.
        """
        for mode in ("push", "pull"):
            with self.subTest(mode=mode):
                self._automatic_route_grows_before_writing_source(mode)

    def _automatic_route_grows_before_writing_source(self, mode: str) -> None:
        """Check one transport with callbacks bound outside the test loop."""
        spec = replace(self._spec(ep_size=2), auto_tokens=True, dispatch_mode=mode)
        hidden = torch.randn(257, 4)
        ids = torch.tensor([[0, 1]] * 257, dtype=torch.int32)
        weights = torch.ones(257, 2)
        payload = torch.tensor([[0, 257, 15, 257, 257, 0, 0], [0, 0, 15, 0, 0, 0, 0]])
        workspace = Mock(in_use=False, source_buffer=None)
        order = []

        def gather(output: Any, unused_value: Any, **_kwargs: Any) -> Mock:
            """Record completion before permitting the capacity callback."""
            output.copy_(payload.flatten())
            return Mock(wait=lambda: order.append("wait"))

        def grow(tokens: int, received: int) -> None:
            """Install the larger source allocation only after count agreement."""
            self.assertEqual(order, ["wait"] if mode == "pull" else ["permute", "wait"])
            self.assertEqual((tokens, received), (384, 514))
            workspace.source_buffer = torch.empty(tokens * 2, 4)
            workspace.in_use = True
            order.append("grow")

        def permute(*args: Any) -> tuple:
            """Observe the new allocation rather than an invalidated old view."""
            self.assertEqual(order, ["wait", "grow"] if mode == "pull" else [])
            order.append("permute")
            output = args[2] if mode == "pull" else hidden.repeat_interleave(2, dim=0)
            return output, torch.arange(514, dtype=torch.int32)

        with (
            patch.object(route_module.dist, "all_gather_into_tensor", side_effect=gather) as exchange,
            patch.object(route_module, "_permute_topk_input", side_effect=permute),
            patch.object(route_module, "_permute_topk_input_out", side_effect=permute),
        ):
            route = prepare_topk_route(hidden, ids, weights, spec, None, workspace,
                                       autograd_mask=15, prepare_workspace=grow)
        self.assertEqual(exchange.call_count, 1)
        self.assertEqual(order, ["wait", "grow", "permute"] if mode == "pull" else ["permute", "wait", "grow"])
        self.assertEqual(tuple(route.routed_tokens.shape), (514, 4))
        self.assertEqual(route.metadata.expert_capacity, 514)


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_invalid_automatic_pull_route_never_allocates_or_writes_source(self) -> None:
        """Feature: MegaMoe route preparation.

        Description: All ranks reject inconsistent participation before the allocation callback can run.
        Expectation: Invalid peers cannot trigger capacity allocation or a pull source write.
        """
        spec = replace(self._spec(ep_size=2), auto_tokens=True, dispatch_mode="pull")
        hidden, ids, weights = torch.ones(1, 4), torch.tensor([[0, 1]], dtype=torch.int32), torch.ones(1, 2)
        payload = torch.tensor([[0, 1, 15, 1, 1, 0, 0], [0, 1, 0, 1, 1, 0, 0]])

        def gather(output: Any, unused_value: Any, **_kwargs: Any) -> Mock:
            """Inject a peer that chose incompatible autograd participation."""
            output.copy_(payload.flatten())
            return Mock()

        callback = Mock()
        with (patch.object(route_module.dist, "all_gather_into_tensor", side_effect=gather),
              patch.object(route_module, "_permute_topk_input_out") as permute,
              self.assertRaisesRegex(RuntimeError, "autograd participation")):
            prepare_topk_route(hidden, ids, weights, spec, None, prepare_workspace=callback)
        callback.assert_not_called()
        permute.assert_not_called()


    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dynamic_ragged_counts_preserve_exact_round_trip_intervals(self) -> None:
        """Feature: MegaMoe route preparation.

        Description: Include an empty source receiving work and a source with an empty destination.
        Expectation: Source and destination intervals describe only the actual routed token rows.
        """
        counts = torch.tensor([[0, 0, 0, 0], [3, 1, 2, 0]], dtype=torch.int64)
        header = torch.tensor([[0, 0, 15], [0, 3, 15]], dtype=torch.int64)
        payload = torch.cat((header, counts), dim=1)
        for rank in range(2):
            spec = replace(self._spec(ep_size=2, rank_id=rank), max_local_num_tokens=257)
            capacity, plan_tokens, maximum = route_module._dynamic_summary(payload, spec)
            self.assertEqual((capacity, plan_tokens), ((4, 128) if rank == 0 else (2, 128)))
            self.assertEqual(maximum, 4)
            received = counts[:, rank * 2:rank * 2 + 2]
            meta = route_module._compute_route_metadata(counts[rank], counts, received, spec, capacity)
            self.assertEqual(meta.dispatch_src_off.tolist(), [0, 0, 0, 0] if rank == 0 else [0, 3, 4, 6])
            self.assertEqual(meta.group_list.tolist(), [3, 4] if rank == 0 else [2, 2])
            self.assertEqual(int(meta.dispatch_size.sum()), 0 if rank == 0 else 6)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dynamic_invalid_calls_fail_on_every_rank(self) -> None:
        """Feature: MegaMoe route preparation.

        Description: Reject over-limit tokens, inconsistent gradients/counts, and receive overflow together.
        Expectation: Each rank rejects invalid source lengths, counts and participation.
        """
        valid = torch.tensor([[0, 1, 15, 1, 1, 0, 0], [0, 2, 15, 2, 2, 0, 0]], dtype=torch.int64)
        for column, value, message in ((0, 1, "token capacity"), (1, 258, "token capacity"),
                                       (2, 0, "autograd participation"), (3, 9, "counts")):
            invalid = valid.clone()
            invalid[1, column] = value
            for rank in range(2):
                spec = replace(self._spec(ep_size=2, rank_id=rank), max_local_num_tokens=257)
                with self.subTest(column=column, rank=rank), self.assertRaisesRegex(RuntimeError, message):
                    route_module._dynamic_summary(invalid, spec)
        spec = replace(self._spec(ep_size=2), max_local_num_tokens=257, receive_capacity=4)
        self.assertEqual(route_module._dynamic_summary(valid, spec)[2], 6)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_empty_dynamic_route_keeps_autograd_without_native_permute(self) -> None:
        """Feature: MegaMoe route preparation.

        Description: Zero logical rows remain empty, including router and expert gradient connections.
        Expectation: Empty outputs retain gradients while native permutation is skipped.
        """
        spec = replace(self._spec(), max_local_num_tokens=128)
        hidden = torch.empty(0, 4, requires_grad=True)
        ids = torch.empty(0, 2, dtype=torch.int32)
        weights = torch.empty(0, 2, requires_grad=True)
        with patch.object(route_module, "load_torch_npu") as permute:
            route = prepare_topk_route(hidden, ids, weights, spec, None, autograd_mask=15)
        permute.assert_not_called()
        self.assertEqual(route.routed_tokens.shape[0], 0)
        self.assertEqual(route.metadata.expert_capacity, 1)
        self.assertEqual(route.metadata.plan_tokens, 128)
        output = route_module.restore_topk_output(hidden, route.unpermute_mapping, weights)
        output.sum().backward()
        self.assertEqual(tuple(hidden.grad.shape), (0, 4))
        self.assertEqual(tuple(weights.grad.shape), (0, 2))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_dynamic_gather_keeps_variable_payload_without_token_padding(self) -> None:
        """Feature: MegaMoe route preparation.

        Description: A rank with 129 tokens shares a 256 plan with an empty peer using one gather.
        Expectation: One count exchange preserves actual rows without padding source tensors.
        """
        spec = replace(self._spec(ep_size=2), max_local_num_tokens=257)
        hidden = torch.randn(129, 4)
        ids = torch.tensor([[0, 1]] * 129, dtype=torch.int32)
        weights = torch.ones(129, 2)
        spec = replace(spec, receive_capacity=512, local_num_tokens=384)
        payload = torch.tensor([[0, 129, 15, 129, 129, 0, 0], [0, 0, 15, 0, 0, 0, 0]])

        def gather(output: Any, value: Any, **kwargs: Any) -> Mock:
            """Emulate one asynchronous count exchange."""
            self.assertEqual(value[:3].tolist(), [0, 129, 15])
            self.assertTrue(kwargs["async_op"])
            output.copy_(payload.flatten())
            return Mock()

        with (
            patch.object(route_module.dist, "all_gather_into_tensor", side_effect=gather),
            patch.object(route_module, "_permute_topk_input",
                         return_value=(hidden.repeat_interleave(2, dim=0), torch.arange(258))) as permute,
        ):
            route = prepare_topk_route(hidden, ids, weights, spec, None, autograd_mask=15)
        self.assertIs(permute.call_args.args[0], hidden)
        self.assertEqual(tuple(route.routed_tokens.shape), (258, 4))
        self.assertEqual(route.metadata.plan_tokens, 256)
        self.assertEqual(route.metadata.local_num_tokens, 129)

    @staticmethod
    def _spec(
        *,
        ep_size: int = 1,
        rank_id: int = 0,
        initial_capacity_factor: float | None = None,
        receive_capacity: int = 128,
    ) -> MegaMoeSpec:
        """Build a small CPU-only route specification."""
        return MegaMoeSpec(
            local_num_tokens=2,
            hidden_size=4,
            intermediate_size=2,
            num_experts=4,
            top_k=2,
            initial_capacity_factor=initial_capacity_factor,
            receive_capacity=receive_capacity,
            ep_size=ep_size,
            ep_group=None,
            rank_id=rank_id,
            num_cube_cores=24,
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_router_counts_are_reused_or_computed_only_when_omitted(self) -> None:
        """Feature: MegaMoe route preparation.

        Description: Reuse supplied counts and compute a histogram only when omitted.
        Expectation: Trusted contiguous INT32 counts are reused and omitted counts are computed.
        """
        spec = self._spec()
        flat_ids = torch.tensor([0, 1, 1, 3], dtype=torch.int32)
        supplied_counts = torch.tensor([4, 0, 0, 0], dtype=torch.int32)

        with patch.object(torch, "bincount") as mock_bincount:
            supplied = _resolve_counts(flat_ids, supplied_counts, spec)

        mock_bincount.assert_not_called()
        self.assertIs(supplied, supplied_counts)
        computed = _resolve_counts(flat_ids, None, spec)
        expected = torch.tensor([1, 2, 0, 1], dtype=torch.int32)
        self.assertTrue(
            torch.equal(computed, expected),
            f"computed counts mismatch: expected={expected}, got={computed}",
        )

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_async_count_gather_overlaps_permute_and_builds_metadata(self) -> None:
        """Feature: MegaMoe route preparation.

        Description: Wait after permutation and derive every native offset from counts.
        Expectation: Permutation overlaps the count exchange and pull mappings retain their own storage.
        """
        spec = self._spec(ep_size=2, rank_id=1)
        hidden_states = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
        topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        topk_weights = torch.full((2, 2), 0.5, dtype=torch.float32)
        supplied_counts = torch.tensor([5, 6, 7, 8], dtype=torch.int32)
        global_counts = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            dtype=torch.int32,
        )
        routed_tokens = hidden_states.repeat_interleave(2, dim=0)
        unpermute_mapping = torch.arange(4, dtype=torch.int32)
        events = []
        workspace = Mock()
        workspace.wait_for_reuse.side_effect = lambda: events.append("reuse")
        work = Mock()
        work.wait.side_effect = lambda: events.append("wait")

        def gather_counts(output: Any, input_tensor: Any, **kwargs: Any) -> Any:
            """Populate the mocked rank-major gather output."""
            self.assertIs(input_tensor, supplied_counts)
            self.assertTrue(kwargs["async_op"])
            events.append("gather")
            output.copy_(global_counts.reshape(-1))
            return work

        def permute_input(
            input_hidden_states: Any,
            input_topk_ids: Any,
        ) -> tuple[Any, Any]:
            """Record the mocked payload permutation launch."""
            self.assertIs(input_hidden_states, hidden_states)
            self.assertIs(input_topk_ids, topk_ids)
            events.append("permute")
            return routed_tokens, unpermute_mapping

        with (
            patch.object(
                route_module.dist,
                "all_gather_into_tensor",
                side_effect=gather_counts,
            ) as mock_gather,
            patch.object(
                route_module,
                "_permute_topk_input",
                side_effect=permute_input,
            ),
        ):
            route = prepare_topk_route(
                hidden_states,
                topk_ids,
                topk_weights,
                spec,
                supplied_counts,
                workspace=workspace,
            )

        self.assertEqual(events, ["reuse", "gather", "permute", "wait"])
        self.assertTrue(mock_gather.call_args.kwargs["async_op"])
        self.assertIs(route.routed_tokens, routed_tokens)
        self.assertIs(route.unpermute_mapping, unpermute_mapping)
        self.assertTrue(
            torch.equal(
                route.received_counts,
                torch.tensor([[3, 4], [7, 8]], dtype=torch.int32),
            )
        )
        metadata = route.metadata
        self.assertTrue(
            torch.equal(metadata.dispatch_src_off, torch.tensor([0, 5, 11, 18]))
        )
        self.assertTrue(
            torch.equal(metadata.dispatch_target_off, torch.tensor([1, 8, 3, 14]))
        )
        self.assertTrue(torch.equal(metadata.dispatch_size, supplied_counts))
        self.assertTrue(
            torch.equal(metadata.combine_src_off, torch.tensor([0, 10, 3, 14]))
        )
        self.assertTrue(
            torch.equal(metadata.combine_target_off, torch.tensor([3, 6, 11, 18]))
        )
        self.assertTrue(torch.equal(metadata.combine_size, torch.tensor([3, 4, 7, 8])))
        self.assertTrue(torch.equal(metadata.group_list, torch.tensor([10, 22])))
        self.assertEqual(metadata.expert_capacity, 22)
        self.assertEqual((spec.dispatch_split, spec.combine_split, spec.swiglu_split), (128, 128, 128))
        events.clear()
        workspace.in_use, workspace.source_buffer = True, torch.empty_like(routed_tokens)
        workspace.wait_for_reuse.reset_mock()
        pull_spec = replace(spec, dispatch_mode="pull")

        def _permute_out(tokens, indices, output, mapping):
            self.assertIs(tokens, hidden_states)
            self.assertIs(output, workspace.source_buffer)
            self.assertTrue(indices.is_contiguous())
            self.assertEqual(mapping.dtype, torch.int32)
            events.append("permute-out")
            output.copy_(routed_tokens)
            mapping.copy_(unpermute_mapping)

        with (patch.object(route_module.dist, "all_gather_into_tensor", side_effect=gather_counts) as gather,
              patch.object(route_module.multicore_ops, "moe_token_permute_out", side_effect=_permute_out),
              patch.object(route_module, "_permute_topk_input") as allocating):
            first, pull = [prepare_topk_route(hidden_states, topk_ids.T.contiguous().T, topk_weights,
                                              pull_spec, supplied_counts, workspace) for _ in range(2)]
            self.assertIs(pull.routed_tokens, workspace.source_buffer)
            self.assertNotEqual(first.unpermute_mapping.data_ptr(), pull.unpermute_mapping.data_ptr())
            torch.testing.assert_close(first.unpermute_mapping, unpermute_mapping)
            for active, source in ((False, workspace.source_buffer), (True, None)):
                workspace.in_use, workspace.source_buffer = active, source
                gather.reset_mock()
                with self.assertRaisesRegex(RuntimeError, "initialized, claimed workspace"):
                    prepare_topk_route(hidden_states, topk_ids, topk_weights, pull_spec, supplied_counts, workspace)
                gather.assert_not_called()
        allocating.assert_not_called()
        workspace.wait_for_reuse.assert_not_called()
        self.assertEqual(events, ["gather", "permute-out", "wait"] * 2)
        self.assertEqual(pull.metadata.dispatch_src_off.tolist(), [3, 6, 11, 18])
        self.assertEqual(pull.metadata.dispatch_target_off.tolist(), [0, 10, 3, 14])
        self.assertEqual(pull.metadata.dispatch_size.tolist(), [3, 4, 7, 8])
        self.assertEqual(pull.maximum_received_slots, 22)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_local_capacity_tracks_routes_below_source_size_and_empty_ranks(self) -> None:
        """Feature: MegaMoe route preparation.

        Description: Shrink destination intermediates independently of source or SHMEM size.
        Expectation: Each rank uses its actual destination load with non-null storage for empty destinations.
        """
        spec = self._spec(ep_size=2, receive_capacity=256)
        routes = (
            ([[1, 1, 1, 1], [1, 1, 1, 1]], (4, 4)),
            ([[2, 1, 1, 0], [2, 1, 1, 0]], (6, 2)),
            ([[2, 2, 0, 0], [2, 2, 0, 0]], (8, 1)),
            ([[0, 0, 2, 2], [0, 0, 2, 2]], (1, 8)),
        )
        for counts, capacities in routes:
            for rank, expected in enumerate(capacities):
                with self.subTest(counts=counts, rank=rank):
                    rank_spec = replace(spec, rank_id=rank)
                    actual = _expert_capacity(torch.tensor(counts, dtype=torch.int32), rank_spec)
                    self.assertEqual(actual, (expected, max(capacities)))
                    self.assertEqual(rank_spec.receive_capacity, 256)
                    self.assertEqual(rank_spec.routed_slots, 4)

    def test_replica_plan_avoids_receive_readback_for_push_and_pull(self) -> None:
        """Reuse exact plan loads, including an empty destination and a growth trigger."""
        saw_empty_rank = False
        saw_growth = False
        for logical_counts in ([[4, 0, 0, 0]] * 2, [[2, 2, 0, 0]] * 2):
            for minimum, target in ((0, None), (100, None), (100, 8)):
                plan = build_expert_replica_plan(logical_counts, 1, minimum_replica_rows=minimum,
                                                 target_load=target)
                counts = torch.tensor(plan.dispatch_counts, dtype=torch.int32)
                for mode in ("push", "pull"):
                    for rank in range(2):
                        with self.subTest(counts=logical_counts, minimum=minimum, mode=mode, rank=rank, target=target):
                            spec = replace(self._spec(ep_size=2, rank_id=rank, receive_capacity=4),
                                           num_experts=plan.config.physical_experts, dispatch_mode=mode)
                            expected = _expert_capacity(counts, spec)
                            saw_empty_rank |= plan.destination_loads[rank] == 0
                            saw_growth |= expected[1] > spec.receive_capacity
                            ids = torch.zeros((2, 2), dtype=torch.int32)
                            replica = ReplicaRoute(plan, ids, counts, rank, None)
                            tokens = torch.ones((2, 4), dtype=torch.bfloat16)
                            permuted = tokens.repeat_interleave(2, dim=0)
                            mapping = torch.arange(4, dtype=torch.int32)
                            workspace = Mock(in_use=True, source_buffer=permuted)
                            with (patch.object(route_module, "_start_count_gather") as gather,
                                  patch.object(route_module, "_permute_topk_input",
                                               return_value=(permuted, mapping)),
                                  patch.object(route_module, "_permute_topk_input_out",
                                               return_value=(permuted, mapping)),
                                  patch.object(torch.Tensor, "tolist", side_effect=AssertionError("D2H readback"))):
                                result = prepare_topk_route(tokens, ids, torch.ones((2, 2)), spec,
                                                            counts[rank], workspace, replica)
                            gather.assert_not_called()
                            self.assertEqual((result.metadata.expert_capacity, result.maximum_received_slots), expected)
                            self.assertIs(result.metadata.replica_route, replica)
                            begin = rank * plan.config.slots_per_rank
                            torch.testing.assert_close(result.received_counts,
                                                       counts[:, begin:begin + plan.config.slots_per_rank])
        self.assertTrue(saw_empty_rank)
        self.assertTrue(saw_growth)



class TestMegaMoeDynamicMetadata(unittest.TestCase):
    """Keep dynamic validation and the native count ABI independent of count-table policy."""

    def test_cpu_replica_counts_pack_the_same_metadata_without_device_prefixes(self) -> None:
        """Retained CPU counts cover home-only, guest, empty-source and empty-owner routes."""
        for mode in ("push", "pull"):
            for compact in (False, True):
                for logical in ([[0] * 4, [2, 2, 0, 0]], [[1, 1, 1, 1]] * 2):
                    plan = build_expert_replica_plan(logical, 1)
                    if compact and plan.transfers:
                        continue
                    host = tuple(map(tuple, logical)) if compact else plan.dispatch_counts
                    counts = torch.tensor(host, dtype=torch.int32)
                    summary = tuple((0, sum(row) // 2, 31, sum(row), min(row), max(row), load)
                                    for row, load in zip(logical, plan.destination_loads))
                    for rank in range(2):
                        with self.subTest(mode=mode, compact=compact, logical=logical, rank=rank):
                            spec = replace(self._spec(), num_experts=counts.shape[1], ep_size=2,
                                           dispatch_mode=mode, rank_id=rank)
                            ids = torch.zeros(sum(logical[rank]) // 2, 2, dtype=torch.int32)
                            hidden = torch.ones(len(ids), spec.hidden_size)
                            replica = ReplicaRoute(plan, ids, counts, rank, None, home_only=compact,
                                                   dynamic_summary=summary, _host_counts=host)
                            permuted = hidden.repeat_interleave(2, dim=0)
                            workspace = Mock(in_use=True, source_buffer=permuted)
                            start = rank * spec.local_experts
                            reference = route_module._compute_route_metadata(
                                counts[rank], counts, counts[:, start:start + spec.local_experts], spec,
                                max(1, plan.destination_loads[rank]))
                            with (patch.object(route_module, "_compute_route_metadata",
                                               side_effect=AssertionError("device prefixes")),
                                  patch.object(route_module, "_start_dynamic_gather",
                                               side_effect=AssertionError("duplicate gather")),
                                  patch.object(route_module, "_permute_topk_input",
                                               return_value=(permuted, ids.flatten())),
                                  patch.object(route_module, "_permute_topk_input_out",
                                               return_value=(permuted, ids.flatten()))):
                                actual = prepare_topk_route(
                                    hidden, ids, torch.ones_like(ids).float(), spec, counts[rank], workspace, replica,
                                    prepare_workspace=lambda *_: None).metadata
                            for name in ("dispatch_src_off", "dispatch_target_off", "dispatch_size",
                                         "combine_src_off", "combine_target_off", "combine_size", "group_list"):
                                torch.testing.assert_close(getattr(actual, name), getattr(reference, name))

    def test_checkpoint_replays_hot_plan_after_a_skipped_balanced_call(self) -> None:
        """Skipping zero-copy placement must not shift the following checkpoint cache lookup."""
        previous = DefaultDeviceType.get_device_type()
        DefaultDeviceType.set_device_type("cpu")
        self.addCleanup(DefaultDeviceType.set_device_type, previous)
        home = replace(self._spec(4), ep_size=2)
        config = ExpertReplicaConfig(4, 2, 1)
        physical = replace(home, num_experts=6, logical_num_experts=4, replica_slots_per_rank=1)
        workspace = Mock(in_use=True, source_buffer=torch.empty(4, 16))
        decisions = []

        def gather(output: torch.Tensor, payload: torch.Tensor, **_options: Any) -> Mock:
            output.copy_(payload.repeat(2))
            return Mock()

        def choose(counts: torch.Tensor, summary: list[list[int]], host: list[list[int]] | None) -> Any:
            loads = [row[6] for row in summary]
            if min(loads) == max(loads):
                decisions.append(None)
                return None
            route = prepare_replica_route(
                torch.tensor([[0, 1], [0, 1]], dtype=torch.int32), config, compact_home=True,
                target_load=4, receive_capacity=4, dynamic_header=(0, 2, 31),
                _gathered_counts=_GatheredReplicaCounts(counts, summary, host))
            decisions.append(route.plan_reused)
            return (home if route.home_only else physical), route

        def operation(value: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                for values in ([[0, 1], [2, 3]], [[0, 1], [0, 1]]):
                    ids = torch.tensor(values, dtype=torch.int32)
                    counts = torch.bincount(ids.flatten().long(), minlength=4).int()
                    prepare_topk_route(
                        value, ids, torch.ones(2, 2), home, counts, workspace, autograd_mask=31,
                        prepare_workspace=lambda *_: None, _select_replica_route=choose)
            return value.sin().square()

        value = torch.randn(2, 16, requires_grad=True)
        with (patch.object(replica_routing, "dist") as distributed,
              patch.object(route_module.dist, "all_gather_into_tensor", side_effect=gather) as exchange,
              patch.object(replica_routing, "build_expert_replica_plan",
                           wraps=replica_routing.build_expert_replica_plan) as build,
              patch.object(route_module, "_permute_topk_input",
                           return_value=(torch.ones(4, 16), torch.arange(4)))):
            distributed.is_initialized.return_value = True
            distributed.get_rank.return_value = 0
            distributed.get_world_size.return_value = 2
            checkpoint(operation, value, use_reentrant=False).sum().backward()
        self.assertEqual(decisions, [None, False, None, True])
        self.assertEqual(build.call_count, 1)
        self.assertEqual(exchange.call_count, 4)
        torch.testing.assert_close(value.grad, 2 * value.detach().sin() * value.detach().cos())

    def test_shared_home_agreement_rejects_invalid_ids_before_selection(self) -> None:
        """Untrusted invalid routes fail collectively before allocation or placement."""
        spec = replace(self._spec(4), ep_size=1)
        for values in ([[0, 0], [2, 3]], [[-1, 1], [2, 3]], [[0, 4], [2, 3]]):
            with self.subTest(ids=values):
                select, grow = Mock(), Mock()
                ids = torch.tensor(values, dtype=torch.int32)
                with (patch.object(route_module, "_resolve_counts", side_effect=AssertionError("unsafe recount")),
                      patch.object(route_module, "_permute_topk_input") as permute):
                    with self.assertRaisesRegex(RuntimeError, "distinct Top-K"):
                        prepare_topk_route(torch.ones(2, 16), ids, torch.ones(2, 2), spec, None,
                                           prepare_workspace=grow, _select_replica_route=select)
                select.assert_not_called()
                grow.assert_not_called()
                permute.assert_not_called()

    def test_replica_selection_defers_only_the_unexecuted_owner_receive_limit(self) -> None:
        """A raw owner peak may exceed INT32 while a bounded physical placement remains legal."""
        spec = replace(self._spec(4), hidden_size=1)
        tokens = 600_000_000
        payload = torch.tensor([[0, tokens, 31, tokens, tokens, 0, 0]] * 4, dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "INT32"):
            route_module._read_dynamic_route(payload, spec)
        summary, _, rows = route_module._read_dynamic_route_with_summary(payload, spec, defer_receive_limit=True)
        self.assertEqual(summary[2], 4 * tokens)
        self.assertEqual(sum(row[3] for row in rows) // 4, tokens * 2)
        payload[0, 2] = 0
        with self.assertRaisesRegex(RuntimeError, "autograd participation"):
            route_module._read_dynamic_route_with_summary(payload, spec, defer_receive_limit=True)

    def test_shared_agreement_keeps_one_gather_for_home_guests_and_empty_sources(self) -> None:
        """Both summary policies select replicas before any receive growth or Pull write."""
        for experts in (4, 520):
            for mode in ("push", "pull"):
                for pattern in ("balanced", "hot", "empty"):
                    with self.subTest(experts=experts, mode=mode, pattern=pattern):
                        self._shared_agreement(experts, mode, pattern)

    def _shared_agreement(self, experts: int, mode: str, pattern: str) -> None:
        """Compare native offsets with the exact selected lossless physical layout."""
        tokens = 512
        logical = [[0] * experts for _ in range(2)]
        for row in logical:
            row[0] = row[1] = tokens
        if pattern == "balanced":
            for row in logical:
                row[0] = row[1] = tokens // 2
                row[experts // 2] = row[experts // 2 + 1] = tokens // 2
        if pattern == "empty":
            logical[0] = [0] * experts
        for rank in (0, 1):
            config = ExpertReplicaConfig(experts, 2, 1)
            home = replace(self._spec(experts, rank), ep_size=2, dispatch_mode=mode)
            physical = replace(home, num_experts=config.physical_experts, logical_num_experts=experts,
                               replica_slots_per_rank=1)
            payload = torch.tensor([[0, sum(row) // 2, 31, *row] for row in logical], dtype=torch.int32)
            count = sum(logical[rank]) // 2
            ids = torch.tensor([[0, 1]] * count, dtype=torch.int32).reshape(count, 2)
            hidden, probs = torch.ones(count, 16), torch.ones(count, 2)
            workspace = Mock(in_use=False, source_buffer=None)
            staged, mapping = torch.empty(count * 2, 16), torch.arange(count * 2, dtype=torch.int32)
            selected = []
            grown = []

            def gather(output: torch.Tensor, _source: torch.Tensor, **_options: Any) -> Mock:
                output.copy_(payload.flatten())
                return Mock()

            def choose(counts: torch.Tensor, summary: list[list[int]], host: list[list[int]] | None) -> Any:
                self.assertFalse(grown)
                self.assertIsNone(workspace.source_buffer)
                self.assertEqual(host is None, experts == 520)
                loads = [row[6] for row in summary]
                if min(loads) == max(loads):
                    return None
                with patch("hyper_multicore.replica.routing.dist") as distributed:
                    distributed.is_initialized.return_value = True
                    distributed.get_rank.return_value = rank
                    distributed.get_world_size.return_value = 2
                    route = prepare_replica_route(
                        ids, config, compact_home=True, dynamic_header=(0, count, 31),
                        target_load=sum(loads) // 2, receive_capacity=sum(loads) // 2,
                        _gathered_counts=_GatheredReplicaCounts(counts, summary, host))
                selected.append(route)
                return (home if route.home_only else physical), route

            def grow(plan_tokens: int, maximum: int) -> None:
                grown.append((plan_tokens, maximum))
                workspace.source_buffer = torch.empty(plan_tokens * 2, 16)
                workspace.in_use = True

            with (patch.object(route_module.dist, "all_gather_into_tensor", side_effect=gather) as exchange,
                  patch("hyper_multicore.replica.routing._gather_replica_counts",
                        side_effect=AssertionError("second gather")),
                  patch.object(route_module, "_permute_topk_input", return_value=(staged, mapping)) as permute,
                  patch.object(route_module, "_permute_topk_input_out", return_value=(staged, mapping)) as publish):
                actual = prepare_topk_route(
                    hidden, ids, probs, home, payload[rank, 3:], workspace, autograd_mask=31,
                    prepare_workspace=grow, _select_replica_route=choose)
            self.assertEqual(exchange.call_count, 1)
            self.assertEqual(len(grown), 1)
            self.assertEqual(permute.call_count, int(mode == "push"))
            self.assertEqual(publish.call_count, int(mode == "pull" and count > 0))
            replica = selected[0] if selected else None
            counts = torch.tensor(replica.plan.dispatch_counts if replica and not replica.home_only else logical,
                                  dtype=torch.int32)
            spec = physical if replica and not replica.home_only else home
            start = rank * spec.local_experts
            reference = route_module._compute_route_metadata(
                counts[rank], counts, counts[:, start:start + spec.local_experts], spec,
                actual.metadata.expert_capacity)
            for name in ("dispatch_src_off", "dispatch_target_off", "dispatch_size", "combine_src_off",
                         "combine_target_off", "combine_size", "group_list"):
                torch.testing.assert_close(getattr(actual.metadata, name), getattr(reference, name))
            self.assertIs(actual.metadata.replica_route, replica)

    @staticmethod
    def _spec(experts: int = 24, rank: int = 0, automatic: bool = True) -> MegaMoeSpec:
        """Construct a four-rank CPU specification without a process group."""
        return MegaMoeSpec(
            local_num_tokens=512,
            hidden_size=16,
            intermediate_size=8,
            num_experts=experts,
            top_k=2,
            initial_capacity_factor=None,
            receive_capacity=512,
            ep_size=4,
            ep_group=None,
            rank_id=rank,
            num_cube_cores=24,
            auto_tokens=automatic,
            max_local_num_tokens=None if automatic else 512,
        )

    @staticmethod
    def _payload(experts: int, dtype: torch.dtype) -> torch.Tensor:
        """Generate distinct Top-K histograms including an empty source."""
        rows = []
        for rank, tokens in enumerate((0, 1, 129, 257)):
            ids = (torch.arange(tokens * 2) + rank * 3) % experts
            counts = torch.bincount(ids, minlength=experts)
            rows.append([0, tokens, 31, *counts.tolist()])
        return torch.tensor(rows, dtype=dtype)

    @staticmethod
    def _reference(payload: torch.Tensor, spec: MegaMoeSpec) -> tuple[int, int, int]:
        """Use tensor reductions to independently check the Host list implementation."""
        counts = payload[:, 3:]
        loads = counts.reshape(spec.ep_size, spec.ep_size, spec.local_experts).sum((0, 2))
        maximum_tokens = int(payload[:, 1].max())
        schedule = max(128, (maximum_tokens + 127) // 128 * 128)
        return max(1, int(loads[spec.rank_id])), schedule, int(loads.max())

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_host_and_device_summary_policies_match_exact_route_counts(self) -> None:
        """Feature: dynamic count summary.

        Description: Cross table sizes, integer widths, transports and empty-source ranks.
        Expectation: Every policy preserves exact destination loads and aligned schedules.
        """
        for experts in (24, 256, 260, 1024):
            for dtype in (torch.int32, torch.int64):
                payload = self._payload(experts, dtype)
                for rank in range(4):
                    spec = self._spec(experts, rank)
                    with self.subTest(experts=experts, dtype=dtype, rank=rank):
                        self.assertEqual(
                            route_module._read_dynamic_route(payload, spec)[0], self._reference(payload, spec)
                        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_large_summary_never_copies_the_complete_count_table_to_host(self) -> None:
        """Feature: bounded summary transfer.

        Description: Prohibit raw readback for a count matrix above the Host policy limit.
        Expectation: Device reduction reads back a compact summary with the same results.
        """
        payload = self._payload(1024, torch.int64)
        spec = self._spec(1024)
        expected = self._reference(payload, spec)
        original = torch.Tensor.tolist

        def tolist(tensor: torch.Tensor) -> list:
            """Distinguish a full-table transfer from the compact device summary."""
            if tensor is payload:
                raise AssertionError(f"large raw count table copied to Host: shape={tuple(tensor.shape)}")
            return original(tensor)

        with patch.object(torch.Tensor, "tolist", tolist):
            self.assertEqual(route_module._read_dynamic_route(payload, spec)[0], expected)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_summary_validation_remains_collective_for_both_table_policies(self) -> None:
        """Feature: distributed dynamic validation.

        Description: Inject native overflow, mixed gradients and invalid histogram values.
        Expectation: Small and large tables reject invalid metadata on every rank.
        """
        for experts in (24, 1024):
            valid = self._payload(experts, torch.int64)
            for column, value, message in ((0, 1, "INT32"), (2, 0, "autograd participation"), (3, -1, "counts")):
                invalid = valid.clone()
                invalid[1, column] = value
                for rank in range(4):
                    with self.subTest(experts=experts, column=column, rank=rank):
                        with self.assertRaisesRegex(RuntimeError, message):
                            route_module._read_dynamic_route(invalid, self._spec(experts, rank))
            invalid = valid.clone()
            invalid[1, 1] = 2**31
            invalid[1, 3:] = 0
            invalid[1, 3:5] = 2**31
            with self.assertRaisesRegex(RuntimeError, "INT32"):
                route_module._read_dynamic_route(invalid, self._spec(experts))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_host_descriptors_match_independent_tensor_prefixes(self) -> None:
        """Feature: packed native route descriptors.

        Description: Compare every field across empty sources, uneven counts and both transports.
        Expectation: INT64 offsets, INT32 sizes and cumulative expert loads match tensor scans exactly.
        """
        for ep in (1, 2, 4, 8):
            for dtype in (torch.int32, torch.int64):
                for local in (2, 6, 64):
                    experts = ep * local
                    generator = torch.Generator().manual_seed(experts)
                    counts = torch.randint(0, 1024, (ep, experts), dtype=dtype, generator=generator)
                    counts[0] = 0
                    raw = torch.cat((torch.zeros(ep, 3, dtype=torch.int32), counts), dim=1).tolist()
                    for rank in range(ep):
                        for mode in ("push", "pull"):
                            spec = replace(self._spec(experts), ep_size=ep, rank_id=rank, dispatch_mode=mode)
                            received = counts[:, rank * local : (rank + 1) * local].contiguous()
                            expected = route_module._compute_route_metadata(counts[rank], counts, received, spec, 32768)
                            actual = route_module._host_route_metadata(raw, spec, 32768, torch.device("cpu"))
                            with self.subTest(ep=ep, local=local, rank=rank, mode=mode):
                                self.assertIsNone(expected.empty_experts,
                                                  "Device-only descriptors must not read counts back")
                                empty = tuple((received.sum(0) == 0).nonzero().flatten().tolist())
                                expected = replace(expected, empty_experts=empty)
                                for field, value in vars(expected).items():
                                    observed = getattr(actual, field)
                                    if isinstance(value, torch.Tensor):
                                        self.assertEqual(value.dtype, torch.int32 if field.endswith("size") else torch.int64)
                                        torch.testing.assert_close(observed, value, rtol=0, atol=0)
                                        self.assertTrue(observed.is_contiguous())
                                        offset = observed.data_ptr() - observed.untyped_storage().data_ptr()
                                        self.assertEqual(offset % 512, 0)
                                    else:
                                        self.assertEqual(observed, value)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_automatic_header_is_int32_and_bounded_header_remains_int64(self) -> None:
        """Feature: dynamic count wire format.

        Description: Exchange ragged source rows through both automatic and bounded routes.
        Expectation: Automatic metadata uses INT32 while the bounded format stays INT64.
        """
        for automatic in (False, True):
            spec = self._spec(automatic=automatic)
            hidden = torch.ones(129, 16)
            ids = torch.tensor([[0, 1]] * 129, dtype=torch.int32)
            counts = torch.tensor([129, 129] + [0] * 22, dtype=torch.int32)
            peer = torch.zeros(24, dtype=torch.int32)
            expected_dtype = torch.int32 if automatic else torch.int64
            header = torch.tensor(
                [[0, 129, 31, *counts.tolist()]] + [[0, 0, 31, *peer.tolist()]] * 3,
                dtype=expected_dtype).reshape(-1)

            def gather(output: torch.Tensor, value: torch.Tensor, payload: torch.Tensor = header,
                       **_kwargs: object) -> Mock:
                """Inspect the real locally packed header before providing peer counts."""
                self.assertEqual(value.dtype, payload.dtype)
                self.assertEqual(output.dtype, value.dtype)
                self.assertEqual(value[:3].tolist(), [0, 129, 31])
                output.copy_(payload)
                return Mock()

            with (patch.object(route_module.dist, "all_gather_into_tensor", side_effect=gather),
                  patch.object(route_module, "_permute_topk_input",
                               return_value=(hidden.repeat_interleave(2, dim=0), torch.arange(258)))):
                prepared = route_module.prepare_topk_route(hidden, ids, torch.ones(129, 2), spec, counts,
                                                          autograd_mask=31, prepare_workspace=Mock())
            self.assertEqual(prepared.metadata.local_num_tokens, 129)
            self.assertEqual(prepared.metadata.dispatch_size.dtype, torch.int32)
            self.assertEqual(prepared.metadata.combine_size.dtype, torch.int32)
            self.assertEqual(prepared.metadata.dispatch_src_off.dtype, torch.int64)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_unrepresentable_automatic_source_still_joins_rejection_collective(self) -> None:
        """Feature: coordinated automatic overflow.

        Description: Submit a mocked source shape too large for an INT32 wire header.
        Expectation: Send an invalid flag with a representable sentinel before every rank rejects.
        """
        hidden = SimpleNamespace(shape=(2**31 + 128, 16), device=torch.device("cpu"))
        ids = torch.empty((0, 2), dtype=torch.int32)
        callback = Mock()

        def gather(output: torch.Tensor, value: torch.Tensor, **_kwargs: object) -> Mock:
            """Require overflow rejection to occur after the same collective on all ranks."""
            self.assertEqual(value.dtype, torch.int32)
            self.assertEqual(output.dtype, value.dtype)
            self.assertEqual(value[:3].tolist(), [1, 2**31 - 1, 31])
            output.copy_(value.repeat(4))
            return Mock()

        with (patch.object(route_module.dist, "all_gather_into_tensor", side_effect=gather) as exchange,
              patch.object(route_module, "_permute_topk_input") as permute,
              self.assertRaisesRegex(RuntimeError, "INT32")):
            route_module._prepare_dynamic_route(hidden, ids, ids.reshape(-1), self._spec(), None, None, 31,
                                                prepare_workspace=callback)
        exchange.assert_called_once()
        callback.assert_not_called()
        permute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
