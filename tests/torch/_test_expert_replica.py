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
"""Standalone replica numerical and lifecycle acceptance on two NPU ranks."""

from __future__ import annotations

import argparse
import faulthandler
import gc
import json
import os
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch_npu

from hyper_multicore import MegaMoeExperts
from hyper_multicore.replica import ExpertReplicaConfig
from hyper_multicore.replica.native import combine_native_replicas, dispatch_native_replicas, native_replica_experts
from hyper_multicore.replica.recompute import replica_checkpoint
from tests.common.multicore.moe_reference import TorchExperts


class _NativeExperts(torch.nn.Module):
    """Exercise the standalone P2P adapter with ordinary owner parameters."""

    def __init__(self, reference: TorchExperts) -> None:
        """Copy only the original owner parameters into the shared EIO contract."""
        super().__init__()
        self.config = ExpertReplicaConfig(8, dist.get_world_size(), 1)
        self.gate_up_weight = torch.nn.Parameter(
            torch.cat((reference.w1, reference.w3), 1).transpose(1, 2).contiguous()
        )
        self.down_weight = torch.nn.Parameter(reference.w2.transpose(1, 2).contiguous())
        self.last_route = None

    def forward(self, hidden: torch.Tensor, ids: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        """Preserve the stock token permutation while dispatching exact integer quotas."""
        routed, mapping = torch_npu.npu_moe_token_permute(hidden, ids) if hidden.shape[0] else (hidden, ids.reshape(-1))
        counts = torch.bincount(ids.flatten().long(), minlength=8)
        inputs, state = dispatch_native_replicas((routed, counts), self.config, dist.group.WORLD)
        self.last_route = state.route
        output = native_replica_experts(inputs[0], self.gate_up_weight, self.down_weight, inputs[1], state.route)
        restored = combine_native_replicas(output, state)
        return (
            torch_npu.npu_moe_token_unpermute(restored, mapping, probabilities)
            if hidden.shape[0]
            else restored + probabilities.sum() * 0
        )

    def close(self) -> None:
        """P2P pools are group-owned and keep no layer parameters alive."""


def _find_route(output: torch.Tensor) -> Any:
    """Find the actual fused autograd invocation without replacing runtime functions."""
    pending, visited = [output.grad_fn], set()
    while pending:
        node = pending.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        if hasattr(node, "replica_route"):
            return node.replica_route
        pending.extend(child for child, _ in node.next_functions)
    return None


def _packed(reference: TorchExperts, *, gradients: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Compare the reference's owner shards in the exact candidate parameter layout."""
    first, third, second = reference.w1, reference.w3, reference.w2
    if gradients:
        first, third, second = first.grad, third.grad, second.grad
    if any(value is None for value in (first, third, second)):
        raise AssertionError("Reference owner gradient is missing")
    return torch.cat((first, third), 1).transpose(1, 2), second.transpose(1, 2)


def _assert_tensor(name: str, actual: torch.Tensor, expected: torch.Tensor, errors: dict) -> None:
    """Require finite values and retain each observed maximum absolute deviation."""
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.003, msg=name)
    error = float((actual - expected).abs().max()) if actual.numel() else 0.0
    errors[name] = max(errors.get(name, 0.0), error)


def _inputs(pattern: str, rank: int, step: int, microbatch: int, device: torch.device) -> tuple:
    """Use fixed rank-local seeds, distinct IDs, ragged and empty source lengths."""
    tokens = {"ragged": (1024, 769), "empty": (0, 2048)}.get(pattern, (1024, 1024))[rank]
    torch.manual_seed(7300 + rank * 100 + step * 10 + microbatch)
    hidden = torch.randn(tokens, 512, device=device, dtype=torch.bfloat16) * 0.1
    ids = torch.arange(tokens * 2, device=device).reshape(tokens, 2).remainder(8 if pattern == "balanced" else 2)
    probabilities = torch.full((tokens, 2), 0.5, device=device)
    upstream = torch.randn_like(hidden) * 0.01
    return hidden, ids.int(), probabilities, upstream


def _call(layer, hidden, ids, probabilities, checkpoint_mode):
    """Bind invocation state for either actual Torch checkpoint implementation."""
    if checkpoint_mode == "none":
        return layer(hidden, ids, probabilities)
    return replica_checkpoint(layer, hidden, ids, probabilities, use_reentrant=checkpoint_mode == "reentrant")


def _step(reference, candidate, pattern, rank, step, device, checkpoint_mode, errors):
    """Accumulate two deferred microbatches in reverse backward order, then update owners."""
    reference.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    saved, routes = [], []
    for microbatch in range(2):
        hidden, ids, probabilities, upstream = _inputs(pattern, rank, step, microbatch, device)
        ref_x, ref_p = hidden.clone().requires_grad_(), probabilities.clone().requires_grad_()
        new_x, new_p = hidden.clone().requires_grad_(), probabilities.clone().requires_grad_()
        expected = reference(ref_x, ids, ref_p)
        actual = _call(candidate, new_x, ids, new_p, checkpoint_mode)
        _assert_tensor("output", actual, expected, errors)
        route = candidate.last_route if isinstance(candidate, _NativeExperts) else _find_route(actual)
        routes.append(route)
        saved.append((expected, actual, ref_x, new_x, ref_p, new_p, upstream))
    for expected, actual, ref_x, new_x, ref_p, new_p, upstream in reversed(saved):
        expected.backward(upstream)
        actual.backward(upstream)
        _assert_tensor("input_gradient", new_x.grad, ref_x.grad, errors)
        _assert_tensor("router_gradient", new_p.grad, ref_p.grad, errors)
    expected_gradients = _packed(reference, gradients=True)
    for name, value, expected in zip(
        ("gate_up_gradient", "down_gradient"),
        (candidate.gate_up_weight.grad, candidate.down_weight.grad),
        expected_gradients,
    ):
        _assert_tensor(name, value, expected, errors)
    torch.optim.SGD(reference.parameters(), lr=0.01).step()
    torch.optim.SGD(candidate.parameters(), lr=0.01).step()
    for name, value, expected in zip(
        ("gate_up_weight", "down_weight"), (candidate.gate_up_weight, candidate.down_weight), _packed(reference)
    ):
        _assert_tensor(name, value, expected, errors)
    return [0 if route is None else len(route.plan.transfers) for route in routes]


def _candidate(reference, backend, mode, budget, device):
    """Keep all model-owned parameters at their original expert owners."""
    if backend == "native":
        return _NativeExperts(reference)
    candidate = MegaMoeExperts(
        hidden_size=512,
        intermediate_size=512,
        num_experts=8,
        top_k=2,
        ep_group=dist.group.WORLD,
        dispatch_mode=mode,
        initial_capacity_factor=1.0 if mode == "push" else None,
        replica_slots_per_rank=budget,
    ).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        first, second = _packed(reference)
        candidate.gate_up_weight.copy_(first)
        candidate.down_weight.copy_(second)
    return candidate


def _expects_copies(budget: int, mode: str, pattern: str) -> bool:
    """Distinguish admitted test inputs from an allowed owner-only empty-source Push route."""
    return bool(budget and pattern != "balanced" and not (mode == "push" and pattern == "empty"))


def _case(backend, mode, budget, pattern, checkpoint_mode, rank, device):
    """Compare a complete trainable replica case with an independent owner-sharded reference."""
    torch.manual_seed(20261109)
    reference = TorchExperts(dim=512, hidden_dim=512, num_experts=8, group=dist.group.WORLD)
    reference = reference.to(device=device, dtype=torch.bfloat16)
    candidate = _candidate(reference, backend, mode, budget, device)
    errors, transfers, matrices = {}, [], []
    expect_copies = _expects_copies(budget, mode, pattern)
    try:
        for step in range(2):
            transfers.extend(_step(reference, candidate, pattern, rank, step, device, checkpoint_mode, errors))
            if isinstance(candidate, MegaMoeExperts) and expect_copies:
                workspace = candidate._get_execution_resources(candidate.gate_up_weight).workspace
                provider = workspace.replica_provider
                matrices.append(tuple(provider.last_kernel_gradient_matrices))
                if matrices[-1] != (1, 0):
                    raise AssertionError(f"Both gradient projections must return in the kernel: {matrices[-1]}")
        if expect_copies and checkpoint_mode != "reentrant" and not all(transfers):
            raise AssertionError(f"Hot case did not execute actual guest transfers: {transfers}")
        if not budget and any(transfers):
            raise AssertionError("B0 produced guest transfers")
        torch.npu.synchronize(device)
        return {
            "backend": backend,
            "mode": mode,
            "budget": budget,
            "pattern": pattern,
            "checkpoint": checkpoint_mode,
            "max_absolute_errors": errors,
            "actual_transfers": transfers,
            "kernel_gradient_matrices": matrices,
            "status": "passed",
        }
    except Exception:
        traceback.print_exc()
        raise
    finally:
        candidate.close()
        del candidate, reference
        gc.collect()


def test_replica_training() -> None:
    """Run fixed numerical cases; every rank writes a terminal report after cleanup."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    rank, device = dist.get_rank(), torch.device("npu", local_rank)
    if dist.get_world_size() != 2:
        raise ValueError("Replica acceptance requires two explicit EP ranks")
    faulthandler.enable()
    faulthandler.dump_traceback_later(90, repeat=True)
    backend = os.environ.get("HP_REPLICA_BACKEND", "mega")
    mode = os.environ.get("HP_REPLICA_MODE", "push")
    result = {
        "rank": rank,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "source": str(Path(__file__).resolve()),
        "cases": [],
        "performance_acceptance": False,
    }
    try:
        patterns = (
            (0, "hot", "none"),
            (1, "balanced", "none"),
            (1, "hot", "none"),
            (1, "ragged", "none"),
            (1, "empty", "none"),
            (1, "hot", "nonreentrant"),
            (1, "hot", "reentrant"),
        )
        if backend == "native":
            patterns = ((1, "hot", "none"), (1, "empty", "nonreentrant"))
        for budget, pattern, checkpoint_mode in patterns:
            record = _case(backend, mode, budget, pattern, checkpoint_mode, rank, device)
            result["cases"].append(record)
            print(json.dumps({"rank": rank, **record}), flush=True)
        dist.barrier()
    finally:
        torch.npu.synchronize(device)
        result["cleanup"] = "released"
        result["status"] = "passed" if len(result["cases"]) == len(patterns) else "failed"
        directory = Path(os.environ.get("HP_REPLICA_RESULT_DIR", "/tmp/expert-replica-results"))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{backend}-{mode}-rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
        dist.destroy_process_group()
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    test_replica_training()
