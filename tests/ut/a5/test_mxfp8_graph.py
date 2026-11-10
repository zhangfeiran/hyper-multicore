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
"""Protect the ready-selection and no-grad activation contracts of the MX graph."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from tests.common.mark_utils import arg_mark
from hyper_multicore.modules.mega_moe import mxfp8_ops
from hyper_multicore.modules.mega_moe.mxfp8_execution import _use_row_only
from hyper_multicore.modules.mega_moe.mxfp8_graph import build_mxfp8_runtime
from hyper_multicore.modules.mega_moe.mxfp8_graph import (
    MX_PROJECTION_READY_MASK, MX_READY_DEPTH1, MX_READY_DEPTH8,
)
from hyper_multicore.modules.mega_moe.mxfp8_module import MXFP8MegaMoeExperts
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.scheduler.runtime import serialize_runtime_config
from hyper_multicore.scheduler.config import TaskType
from hyper_multicore.scheduler.mxfp8 import MX_GMM_TASK_TYPE
from hyper_multicore.scheduler.mxfp8_quant import (
    MX_QUANT_TASK, MX_SWIGLU_TASK, MX_SWIGLU_GRAD_TASK, MX_SWIGLU_ROW_TASK, MX_QUANT_ROW_TASK,
)


def _shape(ep: int, split: int) -> MegaMoeSpec:
    return MegaMoeSpec(4096, 5120, 1792, ep * 6, 8, 1.25, 40960, ep, None, ep - 1, 32,
                       mx_vector_policy="window2", dispatch_split=split, combine_split=split, compute_stages=6)


class TestMXFP8Graph(unittest.TestCase):
    """Exercise CPU runtime images without native initialization or real collectives."""

    def setUp(self) -> None:
        """Isolate runtime configuration and reset native feature receipts after each test."""
        self.environment = mock.patch.dict(os.environ, {"PANGU_MX_GMM_TILE": "t6",
                                                       "HYPER_PARALLEL_SHMEM_DATA_ENGINE": "mte_udma",
                                                       "HYPER_PARALLEL_SHMEM_UDMA_QP_NUM": "16",
                                                       "PANGU_MX_READY_DEPTH": "0",
                                                       "PANGU_MX_SWIGLU_ROW_ONLY": "0"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.addCleanup(mxfp8_ops.require_mxfp8_window_feature.cache_clear)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_disabled_switches_preserve_frozen_runtime_bytes(self) -> None:
        """Feature: Disabled MX readiness switches.
        Description: Compare twelve runtime images against the independent C+D baseline.
        Expectation: Every configuration, GMM and quantization byte has the same hash.
        """
        baseline = json.loads((Path(__file__).parent / "fixtures/mxfp8_window_baseline.json").read_text())
        self.assertEqual(baseline["source_commit"], "24fb62785cd2833a707c6ec82aac75cc53099a9e")
        for ep in (4, 8, 64):
            os.environ["HYPER_PARALLEL_SHMEM_UDMA_QP_NUM"] = "1" if ep == 64 else "16"
            for split in (512, 4096):
                for forward in (False, True):
                    with self.subTest(ep=ep, split=split, forward=forward):
                        runtime = build_mxfp8_runtime(_shape(ep, split), forward=forward)
                        actual = {name: hashlib.sha256(data).hexdigest() for name, data in (
                            ("config", serialize_runtime_config(runtime.config)),
                            ("gmm", runtime.gmm_descriptors), ("quant", runtime.quant_descriptors))}
                        self.assertEqual(actual, baseline[f"{ep}-{split}-{int(forward)}"])

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_switches_preserve_transport_and_complete_gmm_work(self) -> None:
        """Feature: Optional MX math selection.
        Description: Compare EP64/QP1 forward graphs for both communication splits.
        Expectation: Communication descriptors and queues remain unchanged.
        """
        os.environ["HYPER_PARALLEL_SHMEM_UDMA_QP_NUM"] = "1"
        for split in (512, 4096):
            spec = _shape(64, split)
            control = build_mxfp8_runtime(spec, forward=True)
            for depth in (0, 1, 8):
                for row_only in (False, True):
                    with self.subTest(split=split, depth=depth, row_only=row_only):
                        actual = build_mxfp8_runtime(spec, forward=True, ready_depth=depth,
                                                     swiglu_row_only=row_only)
                        self.assertEqual(actual.gmm_descriptors, control.gmm_descriptors)
                        self.assertEqual(bytes(actual.config.cube_task_indices),
                                         bytes(control.config.cube_task_indices))
                        self.assertEqual(bytes(actual.config.vector_task_indices),
                                         bytes(control.config.vector_task_indices))
                        for before, after in zip(control.config.all_tasks, actual.config.all_tasks):
                            if before.task_type in (TaskType.TASK_TERMINATE, TaskType.TASK_SHMEM_PUT_MEM_SIGNAL):
                                self.assertEqual(bytes(after), bytes(before))
                        self.assertEqual(sum(task.task_type == MX_SWIGLU_ROW_TASK for task in actual.config.all_tasks),
                                         384 if row_only else 0)
                        self.assertEqual(sum(task.task_type == MX_QUANT_TASK for task in actual.config.all_tasks),
                                         0 if row_only else 384)
                        self.assertEqual(sum(task.task_type == MX_QUANT_ROW_TASK for task in actual.config.all_tasks),
                                         384 if row_only else 0)

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_backward_changes_only_math_selection_flags(self) -> None:
        """Feature: Backward ready selection.
        Description: Inspect both readiness depths against the same backward control.
        Expectation: Only math-selection flags change, preserving all events and payloads.
        """
        os.environ["HYPER_PARALLEL_SHMEM_UDMA_QP_NUM"] = "1"
        for split in (512, 4096):
            spec = _shape(64, split)
            control = build_mxfp8_runtime(spec, forward=False)
            for depth in (1, 8):
                with self.subTest(split=split, depth=depth):
                    actual = build_mxfp8_runtime(spec, forward=False, ready_depth=depth)
                    self.assertEqual(actual.gmm_descriptors, control.gmm_descriptors)
                    self.assertEqual(actual.quant_descriptors, control.quant_descriptors)
                    for before, after in zip(control.config.all_tasks, actual.config.all_tasks):
                        if before.task_type in (MX_GMM_TASK_TYPE, MX_QUANT_TASK, MX_SWIGLU_TASK,
                                                MX_SWIGLU_GRAD_TASK, MX_SWIGLU_ROW_TASK):
                            self.assertEqual(after.extra_value_1, 0x00010000 if depth == 8 else 0x00020000)
                            after.extra_value_1 = 0
                        self.assertEqual(bytes(after), bytes(before))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_ready_selection_preserves_guest_projection_and_gradient_contracts(self) -> None:
        """Feature: B2 readiness and projection prefetch coexistence.
        Description: Compare EP4/16/64 forward and backward graphs at both readiness depths.
        Expectation: Every prefetch tag and gradient task matches after removing selection flags.
        """
        math_types = (MX_GMM_TASK_TYPE, MX_QUANT_TASK, MX_SWIGLU_TASK,
                      MX_SWIGLU_GRAD_TASK, MX_SWIGLU_ROW_TASK)
        for ep in (4, 16, 64):
            os.environ["HYPER_PARALLEL_SHMEM_UDMA_QP_NUM"] = "1" if ep == 64 else "16"
            spec = replace(_shape(ep, 4096), num_experts=ep * 8, logical_num_experts=ep * 6,
                           replica_slots_per_rank=2, replica_transport="shmem_signal_kernel_gradient")
            for forward in (True, False):
                control = build_mxfp8_runtime(spec, forward=forward, weight_prefetch=True)
                tagged = [task for task in control.config.all_tasks if task.extra_value_1]
                self.assertEqual(len(tagged), 2 * 2 * 32)
                self.assertTrue(control.weight_prefetch)
                for depth, flag in ((1, MX_READY_DEPTH1), (8, MX_READY_DEPTH8)):
                    with self.subTest(ep=ep, forward=forward, depth=depth):
                        actual = build_mxfp8_runtime(spec, forward=forward, weight_prefetch=True, ready_depth=depth)
                        self.assertEqual(actual.buffer_count, control.buffer_count)
                        self.assertEqual(actual.gmm_descriptors, control.gmm_descriptors)
                        self.assertEqual(actual.quant_descriptors, control.quant_descriptors)
                        for before, after in zip(control.config.all_tasks, actual.config.all_tasks):
                            self.assertEqual(after.extra_value_1 & MX_PROJECTION_READY_MASK, before.extra_value_1)
                            if before.task_type in math_types:
                                self.assertEqual(after.extra_value_1, before.extra_value_1 | flag)
                                after.extra_value_1 &= MX_PROJECTION_READY_MASK
                        self.assertEqual(serialize_runtime_config(actual.config),
                                         serialize_runtime_config(control.config))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_configuration_reaches_spec_and_preserves_saved_backward_axes(self) -> None:
        """Feature: MX optional graph configuration.
        Description: Propagate environment switches through the module and shared manifest.
        Expectation: Row-only activation is restricted to the no-grad forward pass.
        """
        os.environ.update(PANGU_MX_READY_DEPTH="8", PANGU_MX_SWIGLU_ROW_ONLY="1")
        module = MXFP8MegaMoeExperts(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                                   num_experts=384, top_k=8, ep_size=64)
        self.addCleanup(module.close)
        self.assertEqual(module._resource_group.specification["mx_ready_depth"], 8)
        self.assertTrue(module._resource_group.specification["mx_swiglu_row_only"])
        enabled = replace(_shape(64, 4096), mx_swiglu_row_only=True)
        self.assertTrue(_use_row_only(enabled, False))
        self.assertFalse(_use_row_only(enabled, True))

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_row_only_graph_does_not_reference_column_activation_bindings(self) -> None:
        """Feature: Row-only activation storage sharing.
        Description: Inspect every input and output position in EP4/16/64 prefetch graphs.
        Expectation: No task reads or writes either unused column activation binding.
        """
        for ep in (4, 16, 64):
            os.environ["HYPER_PARALLEL_SHMEM_UDMA_QP_NUM"] = "1" if ep == 64 else "16"
            spec = replace(_shape(ep, 512), num_experts=ep * 8, logical_num_experts=ep * 6,
                           replica_slots_per_rank=2, replica_transport="shmem_signal_kernel_gradient")
            runtime = build_mxfp8_runtime(spec, forward=True, swiglu_row_only=True, weight_prefetch=True)
            for task in runtime.config.all_tasks:
                inputs = [task.inputs[index].input_position for index in range(task.num_inputs)]
                outputs = [task.outputs[index].input_position for index in range(task.num_outputs)]
                self.assertTrue(set(inputs + outputs).isdisjoint({18, 19, 21, 22}),
                                msg=f"Unused column bindings appear in inputs={inputs}, outputs={outputs}")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_unsupported_environment_is_rejected_before_execution(self) -> None:
        """Feature: MX readiness configuration validation.
        Description: Supply invalid depths and an incompatible queue policy.
        Expectation: Each invalid configuration fails before execution.
        """
        for value in ("-1", "2", "64"):
            with self.subTest(value=value):
                os.environ["PANGU_MX_READY_DEPTH"] = value
                with self.assertRaisesRegex(ValueError, "READY_DEPTH"):
                    MXFP8MegaMoeExperts(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                                       num_experts=384, top_k=8, ep_size=64)
        os.environ["PANGU_MX_READY_DEPTH"] = "8"
        with self.assertRaisesRegex(ValueError, "require window2"):
            MXFP8MegaMoeExperts(local_num_tokens=4096, hidden_size=5120, intermediate_size=1792,
                               num_experts=384, top_k=8, ep_size=64, vector_policy="legacy_odd")

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_capability_rejects_stale_binary_and_unavailable_features(self) -> None:
        """Feature: MX native capability receipts.
        Description: Request optional features with a valid receipt and then alter the payload.
        Expectation: Unsupported features and mismatched binary bytes are rejected.
        """
        with TemporaryDirectory() as directory:
            root = Path(directory)
            vendor = root / "vendors/hyper_parallel_multicore_nn"
            kernels = vendor / "op_impl/ai_core/tbe/kernel/ascend950/pangu_mxfp8_task_graph"
            kernels.mkdir(parents=True)
            binary = kernels / "PanguMxfp8TaskGraph_ed65f2bbd39f9cc68999acec5148f3f8.o"
            binary.write_bytes(b"frozen-window-payload")
            receipt = root / "capabilities/mxfp8_window.json"
            receipt.parent.mkdir()
            receipt.write_text(json.dumps({"schema": 7, "object": binary.name,
                                           "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                                           "features": ["ready_math", "nograd_row_only"]}))
            with mock.patch.object(mxfp8_ops, "_load_mxfp8_native"), \
                    mock.patch.object(mxfp8_ops, "get_multicore_paths", return_value=(vendor, None)):
                mxfp8_ops.require_mxfp8_window_feature("ready_math")
                self.assertTrue(mxfp8_ops.mxfp8_window_feature_available("ready_math"))
                self.assertFalse(mxfp8_ops.mxfp8_window_feature_available("mx_token_mte_scratch_v1"))
                with self.assertRaisesRegex(RuntimeError, "unavailable"):
                    mxfp8_ops.require_mxfp8_window_feature("unknown")
                binary.write_bytes(b"old-payload")
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    mxfp8_ops.mxfp8_window_feature_available("mx_token_mte_scratch_v1")
                mxfp8_ops.require_mxfp8_window_feature.cache_clear()
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    mxfp8_ops.require_mxfp8_window_feature("ready_math")
