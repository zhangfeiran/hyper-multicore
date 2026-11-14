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
"""Preserve default accelerators and require evidence before skipping owner initialization."""

import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from hyper_multicore.modules.mega_moe.configuration import execution_options, qualify_topology_schedule
from hyper_multicore.modules.mega_moe import route
from hyper_multicore.modules.mega_moe.route import _owner_empty_experts, _read_replica_dynamic_route
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.replica.recompute import create_recompute_contexts
from hyper_multicore.replica._route_cache import _ReplicaRouteCache
from hyper_multicore.replica.routing import _hot_replay_readback_enabled
from tests.common.multicore.owned_mega_moe import OwnedMegaMoeExperts as MegaMoeExperts


class TestExecutionConfiguration(unittest.TestCase):
    """Separate no-table execution controls from any calibration dependency."""

    @staticmethod
    def _spec():
        """Describe CPU-test topology without binding a runtime or native adapter."""
        return MegaMoeSpec(
            local_num_tokens=2,
            hidden_size=16,
            intermediate_size=8,
            num_experts=4,
            top_k=2,
            initial_capacity_factor=1.25,
            receive_capacity=128,
            ep_size=1,
            ep_group=None,
            rank_id=0,
            num_cube_cores=24,
            auto_tokens=True,
            _selective_owner_gradient_init=True,
        )

    def test_defaults_and_research_controls_enter_resource_identity(self) -> None:
        """Native auto scheduling is a default while explicit unsupported values fail closed."""
        with patch.dict(os.environ, {}, clear=True):
            module = MegaMoeExperts(
                hidden_size=16, intermediate_size=8, num_experts=4, top_k=2, replica_slots_per_rank=1
            )
            values = module._resource_group.specification
            self.assertEqual(values["_gradient_queue_order"], "auto")
            self.assertTrue(values["_selective_owner_gradient_init"])
            self.assertFalse(values["_batch_empty_combine"])
            self.assertFalse(values["_compact_communication"])
        with patch.dict(os.environ, {"HYPER_MULTICORE_MOE_GRADIENT_QUEUE_ORDER": "unsupported"}, clear=True):
            with self.assertRaisesRegex(ValueError, "GRADIENT_QUEUE_ORDER"):
                execution_options("push", 1)

    def test_compact_communication_only_admits_push(self) -> None:
        """The internal switch can cover both owner and guest Push routes."""
        with patch.dict(os.environ, {"HYPER_MULTICORE_MOE_COMPACT_COMMUNICATION": "1"}, clear=True):
            for budget in (0, 6):
                self.assertTrue(execution_options("push", budget)["_compact_communication"])
                self.assertFalse(execution_options("pull", budget)["_compact_communication"])

    def test_owner_skip_requires_positive_k_and_no_guests(self) -> None:
        """Only a positive source covering every expert proves complete owner overwrites."""
        spec = self._spec()
        positive = [[0, 2, 15, 4, 1, 1, 4]]
        self.assertEqual(_owner_empty_experts(spec, positive, None), ())
        self.assertIsNone(_owner_empty_experts(spec, [[0, 2, 15, 4, 0, 2, 4]], None))
        self.assertIsNone(_owner_empty_experts(replace(spec, replica_slots_per_rank=1), positive, None))
        self.assertIsNone(_owner_empty_experts(replace(spec, _selective_owner_gradient_init=False), positive, None))

    def test_multi_node_auto_preserves_w2_communication_order(self) -> None:
        """Qualify owner and guest plans from physical EP topology before native selection."""
        nodes = (0,) * 16 + (1,) * 16
        for budget in (0, 1):
            spec = replace(self._spec(), replica_slots_per_rank=budget, _gradient_queue_order="auto")
            selected = qualify_topology_schedule(spec, nodes)
            self.assertEqual(selected._gradient_queue_order, "auto_w1")
            self.assertEqual(spec._gradient_queue_order, "auto")
            self.assertEqual(selected.replica_slots_per_rank, budget)
            self.assertIs(qualify_topology_schedule(spec, (0,) * 32), spec)
            explicit = replace(spec, _gradient_queue_order="device_active_lanes")
            self.assertIs(qualify_topology_schedule(explicit, nodes), explicit)

    def test_dynamic_summary_retains_owner_overwrite_evidence(self) -> None:
        """The normal wide-table summary can prove owner initialization without another readback."""
        spec = self._spec()
        payload = torch.tensor([[0, 2, 15, 1, 1, 1, 1]], dtype=torch.int32)
        _, _, controls = _read_replica_dynamic_route(payload, spec)
        self.assertEqual(_owner_empty_experts(spec, controls, None), ())

    def test_hot_readback_replays_only_an_active_checkpoint_ordinal(self) -> None:
        """Home calls keep their place and a hot replay reads the current source counts."""
        spec = replace(self._spec(), num_experts=192, ep_size=32)
        payload = torch.zeros(32, 195, dtype=torch.int32)
        payload[:, :3] = torch.tensor([0, 2, 15])
        payload[:, 3:7] = 1
        changed = payload.clone()
        changed[:, 3:] = changed[:, 3:].roll(1, dims=1)
        observed = []

        def select(
            _counts: torch.Tensor, _summary: list[list[int]], host_counts: np.ndarray | list[list[int]] | None
        ) -> tuple[MegaMoeSpec, SimpleNamespace] | None:
            """First invocation stays home, while the second executes a replica."""
            observed.append(host_counts)
            return None if len(observed) % 2 else (spec, SimpleNamespace(home_only=False))

        self.addCleanup(_hot_replay_readback_enabled.cache_clear)
        with patch.dict(os.environ, {"HYPER_MULTICORE_MOE_HOT_REPLAY_READBACK": "1"}):
            _hot_replay_readback_enabled.cache_clear()
            forward, recompute = create_recompute_contexts()
            with patch.object(route, "_read_hot_replay_route", wraps=route._read_hot_replay_route) as read_hot:
                with forward:
                    route._agree_replica_dynamic_route(payload, spec, select)
                    route._agree_replica_dynamic_route(payload, spec, select)
                with recompute:
                    route._agree_replica_dynamic_route(changed, spec, select)
                    route._agree_replica_dynamic_route(changed, spec, select)
                read_hot.assert_called_once()
        self.assertEqual(observed[:3], [None, None, None])
        self.assertIsInstance(observed[3], np.ndarray)
        np.testing.assert_array_equal(observed[3], changed[:, 3:].numpy())

    def test_hot_readback_requires_matching_forward_spec(self) -> None:
        """A different runtime spec falls back to current validated device summary."""
        spec = replace(self._spec(), num_experts=192, ep_size=32)
        payload = torch.zeros(32, 195, dtype=torch.int32)
        payload[:, :3] = torch.tensor([0, 2, 15])
        payload[:, 3:7] = 1
        self.addCleanup(_hot_replay_readback_enabled.cache_clear)
        with patch.dict(os.environ, {"HYPER_MULTICORE_MOE_HOT_REPLAY_READBACK": "1"}):
            _hot_replay_readback_enabled.cache_clear()
            forward, recompute = create_recompute_contexts()

            def select(
                _counts: torch.Tensor, _summary: list[list[int]], _host: np.ndarray | list[list[int]] | None
            ) -> tuple[MegaMoeSpec, SimpleNamespace]:
                """Represent an active route for this invocation's forward spec."""
                return spec, SimpleNamespace(home_only=False)

            with patch.object(route, "_read_hot_replay_route") as read_hot:
                with forward:
                    route._agree_replica_dynamic_route(payload, spec, select)
                with recompute:
                    route._agree_replica_dynamic_route(payload, replace(spec), select)
                read_hot.assert_not_called()

    def test_hot_readback_validates_the_same_current_controls(self) -> None:
        """Fresh host counts agree with device summaries, including remote errors."""
        spec = replace(self._spec(), num_experts=192, ep_size=32)
        payload = torch.zeros(32, 195, dtype=torch.int32)
        payload[:, :3] = torch.tensor([0, 2, 15])
        payload[:, 3:7] = 1
        summary, raw, rows = _read_replica_dynamic_route(payload, spec, defer_receive_limit=True)
        replay_summary, host_counts, replay_rows = route._read_hot_replay_route(payload, spec)
        self.assertEqual(raw, payload.tolist())
        with patch.object(route, "_HOST_REPLICA_SUMMARY_MAX_COUNTS", 2048):
            device_summary, device_raw, device_rows = _read_replica_dynamic_route(
                payload, spec, defer_receive_limit=True
            )
        self.assertIsNone(device_raw)
        self.assertEqual((summary, rows), (device_summary, device_rows))
        self.assertEqual((replay_summary, replay_rows), (summary, rows))
        np.testing.assert_array_equal(host_counts, payload[:, 3:].numpy())
        for column, value in ((0, 1), (2, 0), (3, -1)):
            invalid = payload.clone()
            invalid[1, column] = value
            with self.subTest(column=column):
                with self.assertRaises((ValueError, RuntimeError)) as original:
                    _read_replica_dynamic_route(invalid, spec, defer_receive_limit=True)
                with self.assertRaises(type(original.exception)) as replay:
                    route._read_hot_replay_route(invalid, spec)
                self.assertEqual(str(replay.exception), str(original.exception))

    def test_checkpoint_and_large_tables_retain_device_summary(self) -> None:
        """Bound host descriptor storage and leave checkpoint selection unchanged."""
        for experts, checkpoint in ((192, True), (384, False)):
            spec = replace(self._spec(), num_experts=experts, ep_size=32)
            payload = torch.zeros(32, experts + 3, dtype=torch.int32)
            payload[:, :3] = torch.tensor([0, 2, 15])
            payload[:, 3:7] = 1
            state = object() if checkpoint else None
            with self.subTest(experts=experts, checkpoint=checkpoint), patch.object(
                route, "get_recompute_state", return_value=state
            ):
                _, raw, _ = _read_replica_dynamic_route(payload, spec, defer_receive_limit=True)
                self.assertIsNone(raw)

    def test_exact_device_counts_reuse_owned_raw_and_refresh_on_same_summary_change(self) -> None:
        """Fresh equality flags guard full tables, beyond source sums and owner loads."""
        spec = replace(self._spec(), num_experts=4, ep_size=2)
        payload = torch.tensor([[0, 2, 15, 2, 1, 1, 0], [0, 2, 15, 1, 1, 1, 1]], dtype=torch.int32)
        _, _, rows = route._read_hot_replay_route(payload, spec)
        cache = _ReplicaRouteCache()
        cold = torch.tensor([row + [1] for row in rows], dtype=torch.int64)
        equal = cold.clone()
        equal[:, 7] = 0
        with patch.object(route, "try_route_summary", return_value=cold):
            first, raw, first_rows = _read_replica_dynamic_route(
                payload, spec, defer_receive_limit=True, route_cache=cache
            )
        raw[0][3] = 999
        with patch.object(route, "try_route_summary", return_value=equal) as compare, patch.object(
            route, "_summarize_dynamic_rows", side_effect=AssertionError("identical wire was already validated")
        ):
            same, raw, same_rows = _read_replica_dynamic_route(
                payload.clone(), spec, defer_receive_limit=True, route_cache=cache
            )
            self.assertIs(compare.call_args.args[1], payload)
        self.assertEqual([list(row) for row in raw], payload.tolist())
        self.assertIs(raw, cache.previous_counts(route._route_cache_signature(spec, payload.device))[1])
        with self.assertRaises(TypeError):
            raw[0][3] = 999
        self.assertEqual((same, same_rows), (first, first_rows))
        changed = payload.clone()
        changed[0, 3:5] = torch.tensor([1, 2], dtype=torch.int32)
        _, _, changed_rows = route._read_hot_replay_route(changed, spec)
        self.assertEqual(changed_rows, first_rows)
        different = equal.clone()
        different[0, 7] = 1
        with patch.object(route, "try_route_summary", return_value=different):
            _, current_raw, _ = _read_replica_dynamic_route(
                changed, spec, defer_receive_limit=True, route_cache=cache
            )
        self.assertEqual(current_raw, changed.tolist())
        self.assertEqual(cache.hits["counts"], 1)
        self.assertEqual(cache.misses["counts"], 2)
        cache.clear()
        self.assertIsNone(cache.count_table)

    def test_exact_counts_bind_validation_controls_and_own_summary_rows(self) -> None:
        """Identical bytes cannot bypass stricter bounds or retain externally mutable controls."""
        spec = replace(self._spec(), num_experts=4, ep_size=2)
        payload = torch.tensor([[0, 2, 15, 2, 1, 1, 0], [0, 2, 15, 1, 1, 1, 1]], dtype=torch.int32)
        _, _, rows = route._read_hot_replay_route(payload, spec)
        cold = torch.tensor([row + [1] for row in rows], dtype=torch.int64)
        equal = cold.clone()
        equal[:, 7] = 0
        cache = _ReplicaRouteCache()
        with patch.object(route, "try_route_summary", return_value=cold):
            summary, _, exposed_rows = _read_replica_dynamic_route(
                payload, spec, defer_receive_limit=True, route_cache=cache
            )
        exposed_rows[0][2] = -999
        with patch.object(route, "try_route_summary", return_value=equal):
            same, _, current_rows = _read_replica_dynamic_route(
                payload.clone(), spec, defer_receive_limit=True, route_cache=cache
            )
        self.assertEqual((same, current_rows), (summary, rows))
        with patch.object(route, "try_route_summary", return_value=equal), patch.object(
            route, "_summarize_dynamic_rows", side_effect=RuntimeError("stricter validation reached")
        ) as validate:
            with self.assertRaisesRegex(RuntimeError, "stricter validation reached"):
                _read_replica_dynamic_route(payload.clone(), spec, defer_receive_limit=False, route_cache=cache)
            validate.assert_called_once()

    def test_training_admission_preserves_inference_and_checkpoint_participation(self) -> None:
        """Only complete training projections use the traffic-reviewed lower relative floor."""
        module = SimpleNamespace(intermediate_size=1792)
        with patch("hyper_multicore.modules.mega_moe.module.get_recompute_state", return_value=None):
            for mask, expected in ((31, (0.25, 896)), (13, (0.25, 896)), (12, (0.4, 0)),
                                   (1, (0.4, 0)), (5, (0.4, 597)), (9, (0.4, 298))):
                with self.subTest(mask=mask):
                    self.assertEqual(MegaMoeExperts._replica_route_policy(module, mask), expected)
        with patch("hyper_multicore.modules.mega_moe.module.get_recompute_state", return_value=object()):
            self.assertEqual(MegaMoeExperts._replica_route_policy(module, 12), (0.25, 896))

    def test_summary_prefilter_uses_the_same_current_training_admission(self) -> None:
        """The early home decision cannot hide candidates admitted by the complete planner."""
        module = SimpleNamespace(_summary_home_admission=4000, _summary_home_bypasses=0, dispatch_mode="pull")
        resources = SimpleNamespace(spec=SimpleNamespace(dispatch_mode="pull"))
        summary, loads = [[0, 1, 31], [0, 1, 31]], [8, 4]
        self.assertTrue(MegaMoeExperts._summary_keeps_home(module, resources, summary, loads))
        self.assertFalse(MegaMoeExperts._summary_keeps_home(module, resources, summary, loads, 2500))
        self.assertEqual(module._summary_home_bypasses, 1)
        module._summary_home_admission = 0
        self.assertFalse(MegaMoeExperts._summary_keeps_home(module, resources, summary, loads, 2500))

    def test_home_agreement_does_not_materialize_unused_device_count_table(self) -> None:
        """Integer table views remain strided until a device metadata consumer needs packing."""
        spec = replace(self._spec(), ep_size=2)
        payload = torch.tensor([[0, 2, 31, 2, 1, 1, 0], [0, 2, 31, 1, 1, 1, 1]], dtype=torch.int32)
        ids, hidden = torch.tensor([[0, 1], [2, 3]]), torch.zeros(2, 16)
        with patch.object(route, "_dynamic_route_payload", return_value=(payload[0, 3:], payload[0], False)), \
                patch.object(route, "_start_dynamic_gather", return_value=(payload, None)), \
                patch.object(route, "_read_replica_dynamic_route", return_value=((4, 128, 4), payload.tolist(), [])):
            agreed = route._replica_agreement(hidden, ids, spec, None, None, 31, None)
        self.assertFalse(agreed.table.is_contiguous())
        self.assertEqual(agreed.table.data_ptr(), payload[:, 3:].data_ptr())
        torch.testing.assert_close(agreed.table, payload[:, 3:])
