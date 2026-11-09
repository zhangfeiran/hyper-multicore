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
"""Current token remapping, coordinated invalid routes, and actual receive budgets."""

import unittest
from unittest.mock import patch

import torch

from hyper_multicore.replica import ExpertReplicaConfig, build_expert_replica_plan, routing
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
class TestReplicaRouting(unittest.TestCase):
    """Compare physical counts with independent stable per-expert occurrence quotas."""

    def test_current_ids_are_remapped_even_with_identical_histograms(self) -> None:
        """A plan contains quotas; every input needs its own current token permutation."""
        logical = ((6, 2, 0, 0), (4, 4, 0, 0))
        plan = build_expert_replica_plan(logical, 1)
        self.assertTrue(plan.transfers)
        ids = torch.tensor([0, 1, 0, 0, 1, 0, 0, 0]).reshape(-1, 1)
        reversed_ids = ids.flip(0)
        slots, lengths, counts = routing._upload_route_metadata(plan, 0, ids.device)
        outputs = [routing._remap_replica_ids(current, plan, slots, lengths) for current in (ids, reversed_ids)]
        for current, output in zip((ids, reversed_ids), outputs):
            actual = torch.bincount(output.flatten(), minlength=plan.config.physical_experts)
            torch.testing.assert_close(actual, counts[0].long())
            restored = torch.tensor(plan.physical_to_logical)[output]
            torch.testing.assert_close(restored, current)
            for expert in (0, 1):
                expected_slots = [
                    slot
                    for slot, rows in plan.source_runs_for(0)
                    for _ in range(rows)
                    if plan.physical_to_logical[slot] == expert
                ]
                self.assertEqual(output[current == expert].tolist(), expected_slots)
        self.assertFalse(torch.equal(outputs[0], outputs[1]))

    def test_exact_integer_count_flags_cover_invalid_and_empty_sources(self) -> None:
        """Invalid IDs still produce a fixed wire shape so peers can reject together."""
        for ids, invalid in (([[0, 1], [2, 3]], False), ([[0, 0]], True), ([[-1, 2]], True), ([[0, 4]], True)):
            with self.subTest(ids=ids):
                payload = routing._replica_count_payload(torch.tensor(ids), 4)
                self.assertEqual(tuple(payload.shape), (5,))
                self.assertEqual(payload.dtype, torch.int64)
                self.assertEqual(bool(payload[-1]), invalid)
                self.assertEqual(int(payload[:-1].sum()), len(ids) * 2)
        payload = routing._replica_count_payload(torch.empty(0, 2, dtype=torch.int64), 4)
        self.assertEqual(payload.tolist(), [0, 0, 0, 0, 0])

    def test_group_error_is_rejected_before_planning_and_sparse_transport(self) -> None:
        """A remote duplicate flag cannot be hidden by a valid local histogram."""
        config = ExpertReplicaConfig(4, 2, 1)
        ids = torch.tensor([[0, 1]], dtype=torch.int32)
        matrix = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 0, 0, 1]])
        with (
            patch.object(routing, "dist") as dist,
            patch.object(routing, "_gather_replica_counts", return_value=matrix),
            patch.object(routing, "build_expert_replica_plan") as solve,
        ):
            dist.is_initialized.return_value = True
            dist.get_world_size.return_value = 2
            dist.get_rank.return_value = 0
            with self.assertRaisesRegex(ValueError, "distinct expert IDs"):
                routing.prepare_replica_route(ids, config)
            solve.assert_not_called()

    def test_header_checks_precede_the_planner(self) -> None:
        """Reject autograd disagreement and false trusted totals from the same wire table."""
        invalid = ([0, 2, 31, 3, 0, 2], [0, 2, 15, 4, 0, 2], [1, 2, 31, 4, 0, 2])
        first = [0, 2, 31, 4, 0, 2]
        for peer in invalid:
            with self.subTest(peer=peer), self.assertRaises((ValueError, RuntimeError)):
                routing._validate_source_controls([first, peer], 2)

    def test_soft_receive_reservation_does_not_override_allocator_allowance(self) -> None:
        """Hard capacity comes from the actual Heap even when an initial factor is larger."""
        calls = []

        def budget(tokens: int) -> int:
            """Expose the live physical allowance for the requested source reservation."""
            calls.append(tokens)
            return 512

        target, allowance = routing._automatic_capacity_options(256, 2, None, None, 2.0, budget)
        self.assertEqual((target, allowance, calls), (1024, 512, [256]))
