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
"""Keep inactive replicas on the optimized home graph and active guest protocols intact."""

from collections import OrderedDict
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_multicore.modules.mega_moe import mxfp8_execution, mxfp8_module
from hyper_multicore.modules.mega_moe.mxfp8_graph import build_mxfp8_runtime
from hyper_multicore.modules.mega_moe.mxfp8_ingress import source_row_fp8_enabled
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.scheduler.runtime import serialize_runtime_config


class _RouteObserved(RuntimeError):
    """Stop at the actual route-preparation boundary without initializing a device."""


def _shape(ep: int, budget: int) -> MegaMoeSpec:
    return MegaMoeSpec(
        4096, 5120, 1792, ep * (6 + budget), 8, 1.25, 40960, ep, None, 0, 32,
        mx_vector_policy="window2", dispatch_split=4096, combine_split=4096, compute_stages=6,
        mx_ready_depth=8, mx_swiglu_row_only=True, replica_slots_per_rank=budget,
        logical_num_experts=ep * 6 if budget else 0, mx_gradient_udma=True,
        replica_transport="shmem_signal_kernel_gradient",
    )


class TestMXFP8HomeFallback(unittest.TestCase):
    """Exercise actual invocation selection and cached uploads with CPU control images."""

    def setUp(self) -> None:
        """Isolate the best B0 options that must coexist with replica admission."""
        environment = patch.dict(os.environ, {
            "PANGU_MX_GMM_TILE": "t6", "PANGU_MX_REUSE_DISPATCH_WORKERS": "3",
            "HYPER_PARALLEL_SHMEM_DATA_ENGINE": "mte_udma", "HYPER_PARALLEL_SHMEM_UDMA_QP_NUM": "8",
            "HYPER_PARALLEL_MX_B0_BACKWARD_DAG": "1", "PANGU_MX_SOURCE_ROW_FP8": "1",
        })
        environment.start()
        self.addCleanup(environment.stop)

    def test_replica_resources_can_request_the_home_source_row_path(self) -> None:
        """Accept a bounded replica budget while preserving the row-only and window requirements."""
        self.assertTrue(source_row_fp8_enabled(
            vector_policy="window2", row_only=True, replica_slots=2, hidden_size=5120))

    def test_actual_forward_selects_source_rows_only_for_unretained_home_invocations(self) -> None:
        """Observe the real route consumer for home/active guests and original/retained forwards."""
        spec = _shape(16, 2)
        resources = SimpleNamespace(spec=spec, home_spec=mxfp8_module._mx_home_spec(spec),
                                    workspace=object(), source_row_fp8=True)
        for home_only in (False, True):
            for retain_backward in (False, True):
                with self.subTest(home_only=home_only, retain_backward=retain_backward):
                    execution = mxfp8_execution.MXFP8Execution(
                        resources, replica_route=SimpleNamespace(home_only=home_only))
                    with patch.object(mxfp8_execution, "prepare_topk_route", side_effect=_RouteObserved) as route:
                        with self.assertRaises(_RouteObserved):
                            execution.forward(torch.empty(1, 5120), torch.zeros(1, 8, dtype=torch.int64),
                                              torch.ones(1, 8), None, None, retain_backward)
                    selected = route.call_args.args[3]
                    expected = "indexed" if home_only and not retain_backward else "legacy"
                    self.assertEqual(selected.forward_ingress_mode, expected)
                    self.assertEqual(selected.replica_slots_per_rank, 0 if home_only else 2)

    def test_home_cached_uploads_match_b0_with_all_best_options(self) -> None:
        """Require exact home F/R/B images while the active physical B graph keeps gradient helpers."""
        for ep in (16, 32):
            with self.subTest(ep=ep):
                physical = _shape(ep, 2)
                home = mxfp8_module._mx_home_spec(physical)
                owners = [SimpleNamespace(spec=_shape(ep, 0), home_spec=None, weight_prefetch=None,
                                           token_mte_scratch_bytes=65536, _plans=OrderedDict()),
                          SimpleNamespace(spec=physical, home_spec=home, weight_prefetch=None,
                                           token_mte_scratch_bytes=65536, _plans=OrderedDict())]
                plans = []

                def upload(host: object, device: object, spec: object) -> object:
                    return SimpleNamespace(host=host)

                with patch.object(mxfp8_module, "_device_runtime", upload):
                    for index, owner in enumerate(owners):
                        phases = []
                        for original in (True, False):
                            phases.append(mxfp8_module._MXFP8Resources.plan_for_capacity(
                                owner, 40960, "cpu", home_only=bool(index), swiglu_row_only=original,
                                source_row_fp8=original))
                        plans.append(phases)
                    active = mxfp8_module._MXFP8Resources.plan_for_capacity(owners[1], 40960, "cpu")
                for reference, fallback in zip(*plans):
                    for direction in ("forward", "backward"):
                        first, second = getattr(reference, direction).host, getattr(fallback, direction).host
                        self.assertEqual(serialize_runtime_config(first.config), serialize_runtime_config(second.config))
                        self.assertEqual(first.gmm_descriptors, second.gmm_descriptors)
                        self.assertEqual(first.quant_descriptors, second.quant_descriptors)
                self.assertFalse(active.forward.host.source_row_fp8)
                self.assertTrue(active.backward.host.kernel_gradients)
                with patch.dict(os.environ, {"HYPER_PARALLEL_MX_B0_BACKWARD_DAG": "0"}):
                    original = build_mxfp8_runtime(physical, forward=False)
                selected = build_mxfp8_runtime(physical, forward=False)
                self.assertEqual(serialize_runtime_config(selected.config), serialize_runtime_config(original.config))
                self.assertEqual(selected.gmm_descriptors, original.gmm_descriptors)
                self.assertEqual(selected.quant_descriptors, original.quant_descriptors)
