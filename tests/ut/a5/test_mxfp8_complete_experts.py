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
"""Select complete expert reductions throughout MXFP8 checkpoint training."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from hyper_multicore.replica.recompute import create_recompute_contexts
from hyper_multicore.modules.mega_moe import mxfp8_module
from tests.common.mark_utils import arg_mark


class _RouteObserved(RuntimeError):
    """Stop at route selection before constructing or launching an MX executor."""


class TestMXFP8CompleteExpertSelection(unittest.TestCase):
    """Observe the real forward boundary without touching a device or process group."""

    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="dryrun", essential_mark="essential")
    def test_training_and_checkpoint_select_complete_reductions_while_inference_keeps_policy(self) -> None:
        """Feature: MXFP8 training execution selection.
        Description: Observe ordinary training, original no-grad checkpoint forward, replay and pure inference.
        Expectation: All training phases select complete reductions while pure inference retains its original policy.
        """
        for grad_enabled, phase, expected in ((True, None, True), (False, None, False),
                                              (False, 'forward', True), (True, 'replay', True)):
            with self.subTest(grad_enabled=grad_enabled, phase=phase):
                hidden = torch.ones(1, 64, requires_grad=True)
                weights = (torch.ones(1, 64, 128, requires_grad=True),
                           torch.ones(1, 64, 64, requires_grad=True))
                resources = SimpleNamespace(replica_headers=object())
                module = SimpleNamespace(
                    _validate_forward_inputs=Mock(return_value=hidden),
                    _resource_group=SimpleNamespace(specification={'local_num_tokens': 1}),
                    _get_execution_resources=Mock(return_value=resources), replica_slots_per_rank=2,
                    _prepare_replica_route=Mock(side_effect=_RouteObserved),
                )
                forward, replay = create_recompute_contexts()
                context = {'forward': forward, 'replay': replay}.get(phase, nullcontext())
                backend = SimpleNamespace(is_current_stream_capturing=lambda: False)
                with torch.set_grad_enabled(grad_enabled), context, patch.object(
                        mxfp8_module.torch, 'npu', backend, create=True), patch.object(
                        mxfp8_module, '_checkpoint_contract'), self.assertRaises(_RouteObserved):
                    mxfp8_module.MXFP8MegaMoeExperts.forward(
                        module, hidden, torch.zeros(1, 1, dtype=torch.int32), torch.ones(1, 1),
                        expert_weights=weights,
                    )
                selected = module._prepare_replica_route.call_args.kwargs['complete_experts']
                self.assertIs(selected, expected, f"phase={phase}, selected={selected}, expected={expected}")
