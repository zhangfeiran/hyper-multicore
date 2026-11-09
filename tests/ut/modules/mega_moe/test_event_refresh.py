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
"""Validate optional event refresh overrides without changing execution order."""

import unittest
from dataclasses import replace
from unittest.mock import patch

from hyper_multicore.modules.mega_moe.module import MegaMoeExperts
from hyper_multicore.modules.mega_moe.plan import _build_runtime_artifacts
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec


class TestEventRefresh(unittest.TestCase):
    """Refresh policy must retain all producers, queues and tensor descriptors."""

    def test_runtime_changes_only_poll_intervals(self) -> None:
        """Every existing Cube and Vector task receives its requested local polling rate."""
        for mode in ("push", "pull"):
            for slots in (0, 2):
                with self.subTest(mode=mode, slots=slots):
                    spec = MegaMoeSpec(
                        local_num_tokens=512,
                        hidden_size=5120,
                        intermediate_size=1792,
                        num_experts=24 + 4 * slots,
                        top_k=8,
                        initial_capacity_factor=1.5,
                        receive_capacity=6144,
                        ep_size=4,
                        ep_group=None,
                        rank_id=1,
                        num_cube_cores=24,
                        replica_slots_per_rank=slots,
                        logical_num_experts=24,
                        dispatch_mode=mode,
                    )
                    _, forward, _, backward = _build_runtime_artifacts(spec)
                    _, new_forward, _, new_backward = _build_runtime_artifacts(
                        replace(spec, _cube_event_refresh_us=5, _vector_event_refresh_us=10)
                    )
                    for original, candidate in ((forward, new_forward), (backward, new_backward)):
                        self.assertEqual(bytes(original.cube_task_indices), bytes(candidate.cube_task_indices))
                        self.assertEqual(bytes(original.vector_task_indices), bytes(candidate.vector_task_indices))
                        for kind, interval in ((0, 5), (1, 10)):
                            queue = candidate.cube_task_indices if kind == 0 else candidate.vector_task_indices
                            for task_id in queue[: candidate.task_index_num[kind]]:
                                self.assertEqual(candidate.all_tasks[task_id].extra_value_2, interval)
                                candidate.all_tasks[task_id].extra_value_2 = original.all_tasks[task_id].extra_value_2
                        self.assertEqual(bytes(original), bytes(candidate))

    def test_validate_environment(self) -> None:
        """Malformed policies must fail before device initialization."""
        options = {"hidden_size": 16, "intermediate_size": 8, "num_experts": 4, "top_k": 2, "ep_size": 2}
        for value in ("bad", "10", "10:", "-1:30", "10:30:50", "10001:1"):
            with self.subTest(value=value):
                with patch.dict("os.environ", {"HYPER_MULTICORE_MOE_EVENT_REFRESH_US": value}):
                    with self.assertRaisesRegex(ValueError, "EVENT_REFRESH_US"):
                        MegaMoeExperts(**options)

    def test_policy_changes_resource_identity(self) -> None:
        """Different polling policies must not share execution resources."""
        options = {"hidden_size": 16, "intermediate_size": 8, "num_experts": 4, "top_k": 2, "ep_size": 2}
        with patch.dict("os.environ", {"HYPER_MULTICORE_MOE_EVENT_REFRESH_US": "0:0"}):
            original = MegaMoeExperts(**options)
        with patch.dict("os.environ", {"HYPER_MULTICORE_MOE_EVENT_REFRESH_US": "10:30"}):
            candidate = MegaMoeExperts(**options)
        self.addCleanup(original.close)
        self.addCleanup(candidate.close)
        self.assertNotEqual(original._resource_group.compatibility_key, candidate._resource_group.compatibility_key)
