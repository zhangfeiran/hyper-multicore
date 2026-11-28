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
"""A3 two-rank forward Router and plan replay."""
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
from hyper_multicore.modules.mega_moe import invocation


def test_forward_route_replay():
    """Verify that both checkpoint entry points reuse forward plans on A3."""
    rank = int(os.environ["LOCAL_RANK"])
    torch_npu.npu.set_device(rank)
    dist.init_process_group("hccl")
    try:
        for dispatch_mode in ("push", "pull"):
            _check_forward_plan_replay(rank, dispatch_mode)
    finally:
        dist.destroy_process_group()


def _check_forward_plan_replay(rank, dispatch_mode):
    """Compare the uncheckpointed reference with serial and paired replay."""
    device = torch.device("npu", rank)
    torch.manual_seed(901 + rank)
    experts = [MegaMoeExperts(hidden_size=512, intermediate_size=128, num_experts=8, top_k=2,
                              ep_group=dist.group.WORLD,
                              tuning_options=MegaMoeTuningOptions(
                                  dispatch_mode=dispatch_mode,
                                  expert_placement=HotReplicaPlanner(replica_slots_per_rank=2,
                                                                     receive_capacity_percent=100),
                              ))
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

    def _run(staged, mode):
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
        tape = replay.ReplayCheckpoint(lambda: 1)
        replay_mode = mode in ("reentrant", "nonreentrant")
        with set_checkpoint_early_stop(False):
            if mode == "mixed":
                with pytest.raises(RuntimeError, match="checkpoint participation differs across EP ranks"):
                    if rank == 0:
                        checkpoint(_pair, *leaves, use_reentrant=False, context_fn=tape.contexts)
                    else:
                        _pair(*leaves)
                return None
            if mode == "reentrant":
                out = checkpoint(replay.wrap_checkpoint(_pair, lambda: 1), *leaves, use_reentrant=True)
            elif mode == "nonreentrant":
                out = checkpoint(_pair, *leaves, use_reentrant=False, context_fn=tape.contexts)
                assert len(tape.records) == 2
                assert all(record.route.weight_binding is not None for record in tape.records)
            else:
                out = _pair(*leaves)
        resources = experts[0]._resource_group.resources  # pylint: disable=protected-access
        workspace, placement = resources.workspace, resources.placement
        assert placement.pool is not None, "The forced receive limit must exercise live replicas"
        replays = placement.statistics["route_replays"]
        with (patch.object(dist, "all_gather_into_tensor", wraps=dist.all_gather_into_tensor) as gather,
              patch.object(prefetch, "start_wire", wraps=prefetch.start_wire) as wire,
              patch.object(invocation, "prepare_topk_route", wraps=invocation.prepare_topk_route) as prepare,
              patch.object(placement, "select", wraps=placement.select) as plan,
              patch.object(workspace, "wait_for_reuse", wraps=workspace.wait_for_reuse) as wait):
            sum(x.float().square().mean() for x in out).backward()
        gather.assert_not_called()
        wire.assert_not_called()
        prepare.assert_not_called()
        plan.assert_not_called()
        wait.assert_not_called()
        assert placement.statistics["route_replays"] - replays == (2 if replay_mode else 0)
        assert topk_calls == [0, 1] * (2 if replay_mode else 1)
        print(f"rank {rank} {dispatch_mode} {mode} staged={staged}: "
              f"R topk={len(topk_calls)-2}, gathers={gather.call_count}, "
              f"placement={plan.call_count}, early_wait={wait.call_count}", flush=True)
        return [v.detach().cpu() for v in (*out, *(leaf.grad for leaf in leaves))]

    reference = _run(False, "ordinary")
    for mode in ("nonreentrant", "reentrant"):
        for staged in (False, True):
            for actual, expected in zip(_run(staged, mode), reference, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _run(True, "mixed")
    for expert in experts:
        expert.close()
    print(f"rank {rank} {dispatch_mode}: four exact replay comparisons and mixed registration rejection passed",
          flush=True)
