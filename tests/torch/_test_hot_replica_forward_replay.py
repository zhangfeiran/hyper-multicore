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
"""A3 two-rank forward plan replay and EP mode agreement."""
import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch_npu
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from hyper_multicore import MegaMoeExperts
from hyper_multicore.experimental import HotReplicaPlanner, MegaMoeTuningOptions
from hyper_multicore.modules.hot_replica import checkpoint as replay, prefetch


def test_forward_route_replay():
    """Compare voted and direct replay for serial and staged EOI expert calls."""
    rank = int(os.environ["LOCAL_RANK"])
    torch_npu.npu.set_device(rank)
    dist.init_process_group("hccl")
    try:
        device = torch.device("npu", rank)
        torch.manual_seed(901 + rank)
        experts = [MegaMoeExperts(hidden_size=512, intermediate_size=128, num_experts=8, top_k=2,
                                  ep_group=dist.group.WORLD,
                                  tuning_options=MegaMoeTuningOptions(expert_placement=HotReplicaPlanner()))
                   for _ in range(2)]
        MegaMoeExperts.share_execution_resources(experts)
        biases, values = [], []
        for branch in range(2):
            tokens = 128 * (1 + branch + rank)
            bias = torch.full((8,), -4.0, device=device)
            bias[4 * branch:4 * branch + 2] = 0
            biases.append(bias)
            values.extend((torch.randn(tokens, 512, device=device, dtype=torch.bfloat16) * .1,
                           torch.rand(tokens, 8, device=device),
                           torch.randn(4, 256, 512, device=device, dtype=torch.bfloat16) * .02,
                           torch.randn(4, 512, 128, device=device, dtype=torch.bfloat16) * .02))

        def _run(staged, direct=False, mixed_mode=False):
            topk_calls = []

            def _pair(*inputs):
                def _call(branch, **options):
                    hidden, scores, gate_up, down = inputs[4 * branch:4 * branch + 4]
                    expert = experts[branch]

                    topk_calls.append(branch)
                    ids = (scores.detach() + biases[branch]).topk(2, sorted=False).indices
                    probs = scores.gather(-1, ids)
                    return expert(hidden, ids, probs, expert_weights=(gate_up, down), **options)

                if not staged:
                    return _call(0), _call(1)
                generation = None

                def _first(understanding):
                    nonlocal generation
                    result = None

                    def _second(execute):
                        nonlocal result
                        result = understanding()
                        return execute()

                    generation = _call(1, route_schedule=_second)
                    return result

                u = _call(0, route_schedule=_first)
                return u, generation

            leaves = [v.detach().requires_grad_() for v in values]
            tape = replay.ReplayCheckpoint(lambda: 1, reuse_forward_plan=direct or (mixed_mode and rank == 0))
            with set_checkpoint_early_stop(False):
                if mixed_mode:
                    with pytest.raises(RuntimeError, match="forward plan replay mode differs across EP ranks"):
                        checkpoint(_pair, *leaves, use_reentrant=False, context_fn=tape.contexts)
                    return None
                out = checkpoint(_pair, *leaves, use_reentrant=False, context_fn=tape.contexts)
            resources = experts[0]._resource_group.resources  # pylint: disable=protected-access
            workspace, placement = resources.workspace, resources.placement
            with (patch.object(replay, "agree_replay", wraps=replay.agree_replay) as vote,
                  patch.object(prefetch, "start_replay_vote", wraps=prefetch.start_replay_vote) as staged_vote,
                  patch.object(placement, "select", wraps=placement.select) as plan,
                  patch.object(workspace, "wait_for_reuse", wraps=workspace.wait_for_reuse) as wait):
                sum(x.float().square().mean() for x in out).backward()
            assert vote.call_count + staged_vote.call_count == (0 if direct else 2)
            plan.assert_not_called()
            assert wait.call_count == (0 if direct else 2)
            assert topk_calls == [0, 1, 0, 1]
            assert len(tape.records) == 2
            assert all((record.ids is None) == direct for record in tape.records)
            return [v.detach().cpu() for v in (*out, *(leaf.grad for leaf in leaves))]

        serial = _run(False)
        for staged, direct in ((True, False), (False, True), (True, True)):
            for actual, expected in zip(_run(staged, direct), serial, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        _run(True, mixed_mode=True)
        for expert in experts:
            expert.close()
        print(f"rank {rank}: four exact replay variants and mixed-mode rejection passed", flush=True)
    finally:
        dist.destroy_process_group()
