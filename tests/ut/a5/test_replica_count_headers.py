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
"""Verify immutable count headers retain content, stream ordering and resource ownership."""

from __future__ import annotations

import struct
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from tests.common.mark_utils import arg_mark

from hyper_multicore.replica import _metadata, routing
from hyper_multicore.replica.capacity import ExpertReplicaConfig
from hyper_multicore.modules.mega_moe.module import _MegaMoeExecutionResources
from hyper_multicore.modules.mega_moe.mxfp8_module import _MXFP8Resources


class TestReplicaCountHeaders(unittest.TestCase):
    """Keep cached control bytes independent of logical count and planner decisions."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_exact_content_keys_and_clear(self) -> None:
        """Feature: Count header ownership.

        Description: Repeat one header, change each control, and clear the cache.
        Expectation: Only identical content aliases; retained tensors remain valid.
        """
        cache = _metadata._ReplicaCountHeaderCache()
        original = struct.pack('<4q', 0, 0, 4096, 31)
        first = cache.upload(original, torch.device('cpu'))
        self.assertIs(first, cache.upload(original, torch.device('cpu')))
        for values in ((1, 0, 4096, 31), (0, 1, 4096, 31), (0, 0, 4097, 31), (0, 0, 4096, 30)):
            with self.subTest(values=values):
                changed = cache.upload(struct.pack('<4q', *values), torch.device('cpu'))
                self.assertIsNot(changed, first)
                self.assertEqual(changed.view(torch.int64).tolist(), list(values))
        cache.clear()
        self.assertIsNot(cache.upload(original, torch.device('cpu')), first)
        self.assertEqual(first.view(torch.int64).tolist(), [0, 0, 4096, 31])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_cross_stream_hit_orders_and_records_read(self) -> None:
        """Feature: Asynchronous header lifetime.

        Description: Upload on one accelerator stream and consume on a second.
        Expectation: The consumer waits for upload readiness and records its read.
        """
        cache = _metadata._ReplicaCountHeaderCache()
        producer, consumer = Mock(), Mock()
        ready = Mock()
        value = Mock()
        backend = SimpleNamespace(current_stream=Mock(side_effect=[producer, consumer]), Event=Mock(return_value=ready))
        device = torch.device('cuda', 0)
        with (patch.object(torch, 'cuda', backend),
              patch.object(_metadata, '_async_device_bytes', return_value=value) as copy):
            first = cache.upload(struct.pack('<4q', 0, 0, 4096, 31), device)
            second = cache.upload(struct.pack('<4q', 0, 0, 4096, 31), device)
        self.assertIs(first, second)
        copy.assert_called_once()
        ready.record.assert_called_once_with(producer)
        consumer.wait_event.assert_called_once_with(ready)
        value.record_stream.assert_called_once_with(consumer)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_eviction_is_bounded_without_mutating_old_storage(self) -> None:
        """Feature: Bounded metadata retention.

        Description: Cycle through more headers than the cache can retain.
        Expectation: Evicted data uploads again and earlier consumer storage is unchanged.
        """
        cache = _metadata._ReplicaCountHeaderCache()
        first_data = struct.pack('<4q', 0, 0, 1, 31)
        with patch.object(_metadata, '_async_device_bytes', wraps=_metadata._async_device_bytes) as copy:
            first = cache.upload(first_data, torch.device('cpu'))
            for tokens in range(2, 66):
                cache.upload(struct.pack('<4q', 0, 0, tokens, 31), torch.device('cpu'))
            second = cache.upload(first_data, torch.device('cpu'))
        self.assertLessEqual(len(cache.entries), 8)
        self.assertEqual(copy.call_count, 66)
        self.assertIsNot(first, second)
        self.assertEqual(first.view(torch.int64).tolist(), [0, 0, 1, 31])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_failed_upload_does_not_publish_an_entry(self) -> None:
        """Feature: Failed upload ownership.

        Description: An asynchronous upload raises before publication.
        Expectation: Failure propagates and retry obtains a new complete value.
        """
        cache = _metadata._ReplicaCountHeaderCache()
        data = struct.pack('<4q', 0, 0, 4096, 31)
        with patch.object(_metadata, '_async_device_bytes', side_effect=RuntimeError('copy failed')):
            with self.assertRaisesRegex(RuntimeError, 'copy failed'):
                cache.upload(data, torch.device('cpu'))
        self.assertEqual(len(cache.entries), 0)
        self.assertEqual(cache.upload(data, torch.device('cpu')).view(torch.int64).tolist(), [0, 0, 4096, 31])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_routing_still_gathers_changed_counts_and_rejects_invalid_control(self) -> None:
        """Feature: Count agreement with cached headers.

        Description: Reuse headers across balanced and hot routes, then send an invalid control.
        Expectation: Every call gathers current counts and invalid control still fails.
        """
        cache = _metadata._ReplicaCountHeaderCache()
        config = ExpertReplicaConfig(2, 1, 1)
        with patch.object(routing, '_gather_replica_counts', wraps=routing._gather_replica_counts) as gather:
            for values in ([0, 1], [0, 0], [0, 1]):
                ids = torch.tensor(values, dtype=torch.int64).reshape(2, 1)
                counts = torch.bincount(ids.flatten(), minlength=2).int()
                options = {'compact_home': True, 'dynamic_header': (0, 2, 31), '_trusted_counts': counts}
                reference = routing.prepare_replica_route(ids, config, **options)
                actual = routing.prepare_replica_route(ids, config, _count_header_cache=cache, **options)
                self.assertEqual(actual.plan.logical_counts, reference.plan.logical_counts)
                self.assertTrue(torch.equal(actual.physical_ids, reference.physical_ids))
            ids = torch.tensor([[0], [1]])
            gather.return_value = torch.tensor([[1, 1, 1, 0, 2, 31]], dtype=torch.int64)
            with self.assertRaisesRegex(ValueError, 'in-range'):
                routing.prepare_replica_route(ids, config, compact_home=True, dynamic_header=(0, 2, 31),
                                              _trusted_counts=torch.tensor([1, 1], dtype=torch.int32),
                                              _count_header_cache=cache)
        self.assertEqual(gather.call_count, 7)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_failed_event_is_not_published(self) -> None:
        """Feature: Upload readiness publication.

        Description: Upload succeeds but recording its accelerator event fails.
        Expectation: The incomplete readiness entry is never reusable.
        """
        cache = _metadata._ReplicaCountHeaderCache()
        ready = Mock()
        ready.record.side_effect = RuntimeError('event failed')
        backend = SimpleNamespace(current_stream=Mock(), Event=Mock(return_value=ready))
        with patch.object(torch, 'cuda', backend), patch.object(_metadata, '_async_device_bytes'):
            with self.assertRaisesRegex(RuntimeError, 'event failed'):
                cache.upload(struct.pack('<4q', 0, 0, 4096, 31), torch.device('cuda', 0))
        self.assertEqual(len(cache.entries), 0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_resource_close_releases_cache_after_workspace(self) -> None:
        """Feature: Executor metadata lifecycle.

        Description: Close fails once before the workspace can release, then succeeds.
        Expectation: Cached headers survive failure and release only after successful close.
        """
        resources = _MXFP8Resources.__new__(_MXFP8Resources)
        resources.replica_headers = _metadata._ReplicaCountHeaderCache()
        resources.replica_headers.upload(struct.pack('<4q', 0, 0, 4096, 31), torch.device('cpu'))
        with patch.object(_MegaMoeExecutionResources, 'close', side_effect=RuntimeError('workspace busy')):
            with self.assertRaisesRegex(RuntimeError, 'workspace busy'):
                resources.close()
        self.assertEqual(len(resources.replica_headers.entries), 1)
        with patch.object(_MegaMoeExecutionResources, 'close') as close:
            resources.close()
        close.assert_called_once_with()
        self.assertEqual(len(resources.replica_headers.entries), 0)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_clear_cannot_remove_an_entry_during_consumer_stream_registration(self) -> None:
        """Feature: Concurrent metadata lifecycle.

        Description: Pause a hit in its stream wait while a second host thread clears the cache.
        Expectation: Clear waits until stream registration finishes and the hit returns its owned value.
        """
        cache = _metadata._ReplicaCountHeaderCache()
        waiting, release, clearing = Event(), Event(), Event()
        stream, ready, value = Mock(), Mock(), Mock()
        backend = SimpleNamespace(current_stream=Mock(return_value=stream), Event=Mock(return_value=ready))
        data = struct.pack('<4q', 0, 0, 4096, 31)
        device = torch.device('cuda', 0)

        def wait_for_release(_event: object) -> None:
            """Keep the consumer operation open while clear attempts to enter."""
            waiting.set()
            if not release.wait(3):
                raise RuntimeError('The test consumer was not released')

        def clear_cache() -> None:
            """Record that the clearing thread started before calling the real clear."""
            clearing.set()
            cache.clear()

        with patch.object(torch, 'cuda', backend), patch.object(_metadata, '_async_device_bytes', return_value=value):
            cache.upload(data, device)
            stream.wait_event.side_effect = wait_for_release
            with ThreadPoolExecutor(max_workers=2) as pool:
                hit = pool.submit(cache.upload, data, device)
                self.assertTrue(waiting.wait(3))
                clear = pool.submit(clear_cache)
                try:
                    self.assertTrue(clearing.wait(3))
                    with self.assertRaises(FutureTimeout):
                        clear.result(timeout=0.1)
                finally:
                    release.set()
                self.assertIs(hit.result(timeout=3), value)
                clear.result(timeout=3)
        self.assertEqual(len(cache.entries), 0)


if __name__ == '__main__':
    unittest.main()
