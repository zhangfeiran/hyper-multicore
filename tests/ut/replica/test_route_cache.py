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
"""Exact current counts, current token order, and descriptor ownership boundaries."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_multicore.replica import ExpertReplicaConfig, routing
from hyper_multicore.replica._route_cache import _ReplicaRouteCache, _RouteData
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
class TestReplicaRouteCache(unittest.TestCase):
    """Check reuse decisions against independent unchanged planning and remapping."""

    def test_changed_counts_capacity_and_credit_replan(self) -> None:
        """Every planning input participates even when a resulting layout is equal."""
        counts = ((6, 2, 0, 0), (4, 4, 0, 0))
        config = ExpertReplicaConfig(4, 2, 1)
        options = {"target_load": 8, "receive_capacity": 16, "capacity_limit": 16}
        cache = _ReplicaRouteCache()
        with patch.object(routing, "build_expert_replica_plan", wraps=routing.build_expert_replica_plan) as build:
            first = routing._plan_replica_route(counts, config, None, 0, _route_cache=cache, **options)[0]
            same = routing._plan_replica_route([list(row) for row in counts], config, None, 0,
                                               _route_cache=cache, **options)[0]
            self.assertIs(first, same)
            self.assertEqual(build.call_count, 1)
            for changed, overrides in (
                (((5, 3, 0, 0), counts[1]), {}),
                (counts, {"receive_capacity": 15}),
                (counts, {"capacity_limit": 15}),
                (counts, {"transfer_row_credit": 1}),
                (counts, {"rank_nodes": (0, 1)}),
            ):
                with self.subTest(overrides=overrides):
                    before = build.call_count
                    routing._plan_replica_route(changed, config, None, 0, _route_cache=cache,
                                                **(options | overrides))
                    self.assertEqual(build.call_count, before + 1)
            with self.assertRaises(ValueError):
                routing._plan_replica_route(counts, config, None, 0, _route_cache=cache,
                                            **(options | {"receive_capacity": True}))

    def test_current_token_orders_preserve_stable_occurrence_quotas(self) -> None:
        """Only sorted destinations may be cached; current token permutations must change."""
        counts = ((6, 2, 0, 0), (4, 4, 0, 0))
        plan = routing.build_expert_replica_plan(counts, 1)
        ids = torch.tensor([[0], [1], [0], [0], [1], [0], [0], [0]])
        slots, lengths, table = routing._upload_route_metadata(plan, 0, ids.device)
        cache = _ReplicaRouteCache()
        outputs = []
        for shift in range(ids.numel()):
            current = ids.roll(shift, 0)
            expected = routing._remap_replica_ids(current, plan, slots, lengths)
            output = routing._remap_replica_ids(current, plan, slots, lengths, cache)
            torch.testing.assert_close(output, expected)
            torch.testing.assert_close(torch.tensor(plan.physical_to_logical)[output], current)
            torch.testing.assert_close(torch.bincount(output.flatten(), minlength=6), table[0].long())
            outputs.append(output)
        self.assertFalse(torch.equal(outputs[0], outputs[1]))
        self.assertEqual(cache.misses["destinations"], 1)
        self.assertEqual(cache.hits["destinations"], 7)

    def test_clear_and_uncertain_keys_release_cache_ownership(self) -> None:
        """Unkeyed descriptors and checkpoint calls must always execute the fresh path."""
        cache = _ReplicaRouteCache()
        build = Mock(side_effect=lambda: torch.tensor([1], dtype=torch.int32))
        first = cache.get_data("descriptors", (1,), build)
        self.assertIs(cache.get_data("descriptors", (1,), build), first)
        for _ in range(2):
            self.assertIsNot(cache.get_data("descriptors", None, build), first)
        with patch("hyper_multicore.replica._route_cache.get_recompute_state", return_value=object()):
            self.assertIsNot(cache.get_data("descriptors", (1,), build), first)
        self.assertEqual(build.call_count, 4)
        cache.clear()
        self.assertFalse(cache.entries)
        self.assertIsNone(cache.plan)
        with self.assertRaisesRegex(ValueError, "integer"):
            cache.get_data("descriptors", (2,), lambda: torch.ones(1))

    def test_cross_stream_reads_wait_and_record_every_retained_view(self) -> None:
        """Same-stream FIFO avoids waits while another stream retains the allocation."""
        source, current, ready = Mock(), Mock(), Mock()
        values = (Mock(device=SimpleNamespace(type="npu")), Mock())
        entry = _RouteData((1,), values, values, source, ready)
        with patch.object(torch.npu, "current_stream", return_value=current):
            self.assertIs(entry.consume(), values)
        current.wait_event.assert_called_once_with(ready)
        for value in values:
            value.record_stream.assert_called_once_with(current)
        with patch.object(torch.npu, "current_stream", return_value=source):
            entry.consume()
        source.wait_event.assert_not_called()

    def test_quota_cache_retains_detached_seeds_and_rejected_decisions(self) -> None:
        """Materialization may mutate a seed; the next invocation must receive a fresh copy."""
        counts = [[6, 2, 0, 0], [4, 4, 0, 0]]
        gathered = SimpleNamespace(host_counts=counts)
        logical = torch.tensor(counts)
        config = ExpertReplicaConfig(4, 2, 1)
        cache = _ReplicaRouteCache()
        options = (logical, gathered, config, (16, 0), 8, 16, 16, 1, 1000, 0, None)
        with patch.object(routing, "_admitted_uncalibrated_copies",
                          wraps=routing._admitted_uncalibrated_copies) as solve:
            first, _ = routing._compact_candidate(*options, route_cache=cache)
            expected = [dict(row) for row in first.copies]
            first.copies[0][0] = 999
            second, _ = routing._compact_candidate(*options, route_cache=cache)
            self.assertEqual(second.copies, expected)
            self.assertEqual(solve.call_count, 1)
            rejected = options[:8] + (6000,) + options[9:]
            for _ in range(2):
                candidate, _ = routing._compact_candidate(*rejected, route_cache=cache)
                self.assertIsNone(candidate)
            self.assertEqual(solve.call_count, 2)
