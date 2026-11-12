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
"""MTE topology rejection and explicit HCCL transport contracts."""

import unittest
from unittest.mock import Mock, patch

import torch

from hyper_multicore.modules.mega_moe.module import MegaMoeExperts
from hyper_multicore.modules.mega_moe import module as module_impl
from hyper_multicore.modules.mega_moe import transport
from hyper_multicore.modules.mega_moe.transport import HcclExpertTransport, require_mte_peers
from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="onecard", essential_mark="essential")
class TestExpertTransport(unittest.TestCase):
    """Prevent unsupported remote MTE submission without disabling fused compute."""

    def test_partially_reachable_group_is_rejected(self) -> None:
        """An asymmetric group must fail before a remote atomic signal can be submitted."""
        with self.assertRaisesRegex(RuntimeError, r"unreachable Root PEs: \[8, 9\].*communication_backend"):
            require_mte_peers([True] * 8 + [False, False] + [True] * 6, 16)

    def test_reachable_group_and_singleton(self) -> None:
        """Valid full-MTE groups and self-only local computation remain accepted."""
        require_mte_peers([True] * 16, 16)
        require_mte_peers([True], 1)
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            require_mte_peers([True] * 8, 16)

    def test_backend_validation_precedes_registration(self) -> None:
        """Unsupported options and replica transports never create runtime owners."""
        options = dict(hidden_size=6144, intermediate_size=2048, num_experts=8, top_k=1,
                       create_parameters=False)
        with self.assertRaisesRegex(ValueError, "communication_backend"):
            MegaMoeExperts(**options, communication_backend="rdma")
        with self.assertRaisesRegex(ValueError, "replica_slots_per_rank=0"):
            MegaMoeExperts(**options, communication_backend="hccl", replica_slots_per_rank=2)

    def test_hccl_constructor_is_lazy_and_keeps_layout(self) -> None:
        """HCCL construction neither initializes SHMEM nor changes caller-owned EOI storage."""
        with patch("hyper_multicore.shmem.acquire") as acquire:
            layer = MegaMoeExperts(hidden_size=6144, intermediate_size=2048, num_experts=8, top_k=1,
                                   create_parameters=False, expert_weight_layout="EOI",
                                   communication_backend="hccl", gate_up_expert_padding=4)
            self.assertEqual(layer.communication_backend, "hccl")
            self.assertEqual(layer.expert_weight_layout, "EOI")
            self.assertIsNone(layer.gate_up_weight)
            acquire.assert_not_called()
            layer.close()

    def test_shared_hccl_executor_closes_after_last_owner(self) -> None:
        """Serial owners retain one local fused executor until their last close."""
        options = dict(hidden_size=6144, intermediate_size=2048, num_experts=8, top_k=1,
                       create_parameters=False, communication_backend="hccl")
        first, second = MegaMoeExperts(**options), MegaMoeExperts(**options)
        MegaMoeExperts.share_execution_resources((first, second))
        executor = Mock()
        first._resource_group.hccl_executor = executor
        first.close()
        executor.close.assert_not_called()
        second.close()
        second.close()
        executor.close.assert_called_once_with()

    def test_hccl_rejects_mismatched_autograd_before_payload(self) -> None:
        """A peer that omits an activation gradient cannot strand an inverse All-to-All."""
        compute = Mock()
        hidden = torch.ones((2, 16), dtype=torch.bfloat16, requires_grad=True)
        ids = torch.zeros((2, 1), dtype=torch.int64)
        probabilities = torch.ones((2, 1))
        weights = (torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True))

        def exchange(output: torch.Tensor, source: torch.Tensor, **_kwargs: object) -> None:
            """Change one source's gradient contract without changing its payload counts."""
            output.copy_(source)
            output[1, -1] ^= 2

        with (patch.object(transport.dist, "is_initialized", return_value=True),
              patch.object(transport.dist, "get_world_size", return_value=2),
              patch.object(transport, "_local_compute_group", return_value=object()),
              patch.object(transport.dist, "all_to_all_single", side_effect=exchange),
              patch.object(transport, "load_torch_npu") as load):
            executor = HcclExpertTransport(object(), 16, lambda _group: compute)
            with self.assertRaisesRegex(RuntimeError, "autograd configuration must match"):
                executor(hidden, ids, probabilities, weights)
            compute.assert_not_called()
            load.assert_not_called()

    def test_hccl_registers_later_layers_before_first_heap_binding(self) -> None:
        """A later layer must reserve native storage before an earlier graph stays live."""
        options = dict(hidden_size=6144, intermediate_size=2048, num_experts=8, top_k=1,
                       create_parameters=False, communication_backend="hccl", initial_token_capacity=24576)
        first, second = MegaMoeExperts(**options), MegaMoeExperts(**options)
        transports = [Mock(), Mock()]
        with patch.object(module_impl, "HcclExpertTransport", side_effect=transports) as build:
            self.assertIs(first._hccl_transport(), transports[0])
            self.assertEqual(build.call_count, 2)
            self.assertIs(second._hccl_transport(), transports[1])
            self.assertEqual(build.call_count, 2)
            first.close()
            second.close()
