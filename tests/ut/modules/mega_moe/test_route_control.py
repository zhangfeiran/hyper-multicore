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
"""Verify pinned buffer reuse, local synchronization and exact fresh count validation."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hyper_multicore.modules.mega_moe import route, route_control
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.replica._route_cache import _ReplicaRouteCache


class TestRouteControl(unittest.TestCase):
    """Keep pinned reads ordered and cache reuse dependent on the current complete wire."""

    def test_owned_buffers_and_completed_event_reuse_follow_the_current_caller_stream(self) -> None:
        """Repeated same-shaped reads reuse storage only after the previous local copy completes."""
        order = []
        ready = Mock(record=lambda stream: order.append(("record", stream)),
                     synchronize=lambda: order.append("ready"))
        streams = [object() for _ in range(3)]
        backend = SimpleNamespace(Event=Mock(return_value=ready), current_stream=Mock(side_effect=streams))
        first, other = Mock(), Mock()
        first.copy_.side_effect = lambda value, **kwargs: order.append(("copy", value, kwargs))
        other.copy_.side_effect = first.copy_.side_effect
        sources = [SimpleNamespace(shape=(32, 8), dtype=torch.int64, device="npu:0"),
                   SimpleNamespace(shape=(32, 8), dtype=torch.int64, device="npu:0"),
                   SimpleNamespace(shape=(32, 195), dtype=torch.int32, device="npu:0")]
        with patch.object(route_control.torch, "npu", backend, create=True), \
                patch.object(route_control.torch, "empty", side_effect=[first, other]) as allocate:
            owner = route_control._ReplicaControlReadback()
            self.assertIs(owner.read(sources[0]), first)
            self.assertIs(owner.read(sources[1]), first)
            self.assertIs(owner.read(sources[2]), other)
        expected = []
        for source, stream in zip(sources, streams):
            expected.extend([("copy", source, {"non_blocking": True}), ("record", stream), "ready"])
        self.assertEqual(order, expected)
        backend.Event.assert_called_once_with()
        self.assertEqual(allocate.call_count, 2)
        allocate.assert_any_call((32, 8), dtype=torch.int64, device="cpu", pin_memory=True)
        allocate.assert_any_call((32, 195), dtype=torch.int32, device="cpu", pin_memory=True)
        self.assertEqual(len(owner.buffers), 2)

    def test_failed_copy_propagates_without_publishing_cpu_bytes(self) -> None:
        """A failed current D2H copy cannot be mistaken for a completed previous generation."""
        ready = Mock()
        backend = SimpleNamespace(Event=Mock(return_value=ready), current_stream=Mock())
        host = Mock(copy_=Mock(side_effect=RuntimeError("D2H failed")))
        source = SimpleNamespace(shape=(32, 8), dtype=torch.int64)
        with patch.object(route_control.torch, "npu", backend, create=True), \
                patch.object(route_control.torch, "empty", return_value=host):
            owner = route_control._ReplicaControlReadback()
            with self.assertRaisesRegex(RuntimeError, "D2H failed"):
                owner.read(source)
        ready.record.assert_not_called()
        ready.synchronize.assert_not_called()

    def test_readback_retains_fresh_wire_checks_and_refreshes_changed_current_counts(self) -> None:
        """Pinned storage changes neither validation nor ownership of the cached CPU snapshot."""
        spec = MegaMoeSpec(local_num_tokens=2, hidden_size=16, intermediate_size=8, num_experts=4,
                           top_k=2, initial_capacity_factor=1.25, receive_capacity=128, ep_size=1,
                           ep_group=None, rank_id=0, num_cube_cores=24, auto_tokens=True)
        payload = torch.tensor([[0, 2, 31, 1, 1, 1, 1]], dtype=torch.int32)
        changed = payload.clone()
        changed[0, 3:5] = torch.tensor([2, 0], dtype=torch.int32)
        summaries = [torch.tensor([[0, 2, 31, 4, 1, 1, 4, 1]]),
                     torch.tensor([[0, 2, 31, 4, 1, 1, 4, 0]]),
                     torch.tensor([[0, 2, 31, 4, 0, 2, 4, 1]])]
        readback = SimpleNamespace(read=Mock(side_effect=lambda value: value.clone()))
        cache = _ReplicaRouteCache()
        with patch.object(route, "try_route_summary", side_effect=summaries), \
                patch.object(torch.Tensor, "cpu", side_effect=AssertionError("pageable readback")):
            first = route._read_replica_dynamic_route(payload, spec, route_cache=cache, readback=readback)
            same = route._read_replica_dynamic_route(payload.clone(), spec, route_cache=cache, readback=readback)
            current = route._read_replica_dynamic_route(changed, spec, route_cache=cache, readback=readback)
        self.assertEqual(first[0], same[0])
        self.assertEqual(first[2], same[2])
        self.assertEqual(first[1], [list(row) for row in same[1]])
        self.assertEqual(list(current[1][0]), changed[0].tolist())
        self.assertEqual(cache.hits["validated_counts"], 1)
        self.assertEqual(readback.read.call_count, 5)
