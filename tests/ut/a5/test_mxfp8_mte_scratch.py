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
"""Protect token staging, immutable uploads and indexed source-row dispatch."""

from collections import OrderedDict
from dataclasses import replace
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hyper_multicore.modules.mega_moe import mxfp8_module
from hyper_multicore.modules.mega_moe.mxfp8_graph import (
    build_mxfp8_runtime,
    with_source_row_ingress,
    with_token_mte_scratch,
)
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.scheduler.runtime import serialize_runtime_config


def _shape(ep: int = 16, budget: int = 0) -> MegaMoeSpec:
    return MegaMoeSpec(
        4096, 5120, 1792, ep * (6 + budget), 8, 1.25, 40960, ep, None, 0, 32,
        mx_vector_policy="window2", dispatch_split=4096, combine_split=4096, compute_stages=6,
        mx_ready_depth=8, mx_swiglu_row_only=True, replica_slots_per_rank=budget,
        logical_num_experts=ep * 6 if budget else 0,
    )


class TestMXFP8MteScratch(unittest.TestCase):
    """Inspect real serialized runtime images without loading an NPU payload."""

    def setUp(self) -> None:
        """Use deterministic owner queues and isolate process-wide experiment switches."""
        environment = patch.dict(os.environ, {
            "PANGU_MX_GMM_TILE": "t6", "PANGU_MX_REUSE_DISPATCH_WORKERS": "3",
            "HYPER_PARALLEL_SHMEM_DATA_ENGINE": "mte_udma", "HYPER_PARALLEL_SHMEM_UDMA_QP_NUM": "8",
            "HYPER_PARALLEL_MX_B0_BACKWARD_DAG": "0",
        })
        environment.start()
        self.addCleanup(environment.stop)

    def test_sdk_default_keeps_the_original_runtime(self) -> None:
        """Keep the complete pre-existing control image when no private staging is selected."""
        spec = _shape()
        runtime = build_mxfp8_runtime(spec, forward=False)
        before = serialize_runtime_config(runtime.config)
        self.assertIs(with_token_mte_scratch(runtime, spec, 16384), runtime)
        self.assertEqual(serialize_runtime_config(runtime.config), before)

    def test_private_staging_preserves_queues_events_math_and_source_dispatch(self) -> None:
        """Change only ordinary-transfer markers across phases, physical budgets and topology."""
        for ep in (16, 32):
            for budget in (0, 2):
                spec = _shape(ep, budget)
                for phase in ("original", "recompute", "backward"):
                    with self.subTest(ep=ep, budget=budget, phase=phase):
                        runtime = build_mxfp8_runtime(
                            spec, forward=phase != "backward", swiglu_row_only=phase == "original")
                        if phase == "original" and not budget:
                            runtime = with_source_row_ingress(spec, runtime)
                        original = serialize_runtime_config(runtime.config)
                        for scratch, marker in ((32768, 0x4D585501), (65536, 0x4D585502)):
                            marked = with_token_mte_scratch(runtime, spec, scratch)
                            selected = [task for task in marked.config.all_tasks[:marked.config.task_num]
                                        if task.extra_value_0 == marker]
                            expected = spec.num_experts * (1 if runtime.source_row_fp8 else 2)
                            self.assertEqual(len(selected), expected)
                            if runtime.source_row_fp8:
                                self.assertTrue(all(task.extra_value_0 == 0 for task in
                                                    marked.config.all_tasks[:marked.config.task_num]
                                                    if task.task_type == 105 and task.inputs[1].input_position == 2))
                            for task in selected:
                                task.extra_value_0 = 0
                            self.assertEqual(serialize_runtime_config(marked.config), original)
                            self.assertEqual(serialize_runtime_config(runtime.config), original)
                            self.assertIs(marked.graph, runtime.graph)
                            self.assertEqual(marked.gmm_descriptors, runtime.gmm_descriptors)
                            self.assertEqual(marked.quant_descriptors, runtime.quant_descriptors)

    def test_conflicting_markers_and_unsupported_queues_fail_before_upload(self) -> None:
        """Reject ambiguous ownership and transport options without mutating the supplied graph."""
        spec = _shape()
        runtime = build_mxfp8_runtime(spec, forward=False)
        for scratch in (0, 8192, 131072, True):
            with self.subTest(scratch=scratch), self.assertRaisesRegex(ValueError, "scratch"):
                with_token_mte_scratch(runtime, spec, scratch)
        with self.assertRaisesRegex(ValueError, "window2"):
            with_token_mte_scratch(runtime, replace(spec, mx_vector_policy="legacy_odd"), 65536)
        marked = with_token_mte_scratch(runtime, spec, 65536)
        frozen = serialize_runtime_config(marked.config)
        with self.assertRaisesRegex(ValueError, "unmarked"):
            with_token_mte_scratch(marked, spec, 32768)
        self.assertEqual(serialize_runtime_config(marked.config), frozen)

    def test_actual_plan_cache_uploads_all_phases_once(self) -> None:
        """Use the real consumer to seal original F, retained R and both B plans before upload."""
        spec = _shape()
        owner = SimpleNamespace(spec=spec, home_spec=None, weight_prefetch=None,
                                token_mte_scratch_bytes=65536, _plans=OrderedDict())
        uploaded = []

        def upload(host: object, device: object, actual_spec: object) -> object:
            uploaded.append(host)
            return SimpleNamespace(host=host)

        with patch.object(mxfp8_module, "_device_runtime", upload):
            original = mxfp8_module._MXFP8Resources.plan_for_capacity(
                owner, 40960, "cpu", swiglu_row_only=True, source_row_fp8=True)
            mxfp8_module._MXFP8Resources.plan_for_capacity(owner, 40960, "cpu")
            repeated = mxfp8_module._MXFP8Resources.plan_for_capacity(
                owner, 40960, "cpu", swiglu_row_only=True, source_row_fp8=True)
        self.assertIs(repeated, original)
        self.assertEqual(len(uploaded), 4)
        self.assertTrue(original.forward.host.source_row_fp8)
        for host in uploaded:
            ordinary = [task for task in host.config.all_tasks[:host.config.task_num]
                        if task.task_type == 105 and not (
                            host.source_row_fp8 and task.inputs[1].input_position == 2)]
            self.assertTrue(ordinary)
            self.assertTrue(all(task.extra_value_0 == 0x4D585502 for task in ordinary))

    def test_private_staging_rejects_missing_duplicate_and_even_worker_ownership(self) -> None:
        """Reject damaged worker queues before a private staging marker can reach the device."""
        spec = _shape()
        for damage in ("missing", "duplicate", "even"):
            with self.subTest(damage=damage):
                runtime = build_mxfp8_runtime(spec, forward=False)
                config = runtime.config
                offsets = config.mix_task_indices[:65]
                positions = [position for worker in range(1, 64, 2)
                             for position in range(offsets[worker], offsets[worker + 1])
                             if config.all_tasks[config.vector_task_indices[position]].task_type == 105]
                first, second = positions[:2]
                if damage == "missing":
                    config.vector_task_indices[first] = next(
                        index for index in range(config.task_num) if config.all_tasks[index].task_type != 105)
                elif damage == "duplicate":
                    config.vector_task_indices[first] = config.vector_task_indices[second]
                else:
                    even = offsets[0]
                    config.vector_task_indices[first], config.vector_task_indices[even] = (
                        config.vector_task_indices[even], config.vector_task_indices[first])
                frozen = serialize_runtime_config(config)
                with self.assertRaisesRegex(ValueError, "odd-worker owner"):
                    with_token_mte_scratch(runtime, spec, 65536)
                self.assertEqual(serialize_runtime_config(config), frozen)
