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
"""Validate two full-width LongCat expert forwards before a shared reverse pass."""

from contextlib import ExitStack
import json
import os
from typing import Any

import torch
import torch.distributed as dist

from native_reference import NativeExperts
from validate import (
    arguments, candidate, compare, environment, fixture, hash_tensor, plan_observer,
    set_operator_timeout, start_shmem_lifetime,
)

from hyper_multicore import shmem
from hyper_multicore.longcat._routing import route_experts


def chain(executors: list, args: object, records: list, native: bool) -> torch.Tensor:
    """Retain both forward contexts before traversing the complete backward chain."""
    hidden = records[0]["hidden"]
    for executor, data in zip(executors, records):
        weights = data["gate"], data["down"]
        if native:
            hidden = route_experts(executor, hidden, data["ids"], data["probability"],
                                   args.real, args.zero, args.top_k, dist.group.WORLD, weights, "torch")
        else:
            hidden = executor(hidden, data["ids"], data["probability"], expert_weights=weights)
    hidden.backward(records[-1]["upstream"])
    return hidden


def results(data: list, output: torch.Tensor) -> dict:
    """Collect the final output, first input, both routers, and both sets of experts."""
    values = {"output": output, "input_gradient": data[0]["hidden"].grad}
    for index, item in enumerate(data):
        for key in ("gate", "down", "probability"):
            values[f"layer{index}_{key}_gradient"] = item[key].grad
    return values


def _copy_inputs(data: list[dict]) -> list[dict]:
    """Give each path independent leaves while retaining the same fixture values."""
    return [{key: value.detach().clone().requires_grad_(True) if key in
              ("gate", "down", "hidden", "probability") else value for key, value in item.items()}
             for item in data]


def _layer_inputs(args: Any, rank: int) -> tuple[list[dict], list[dict]]:
    """Use different layer weights and hot owners so stale guest state cannot pass parity."""
    data = [fixture(args, rank + index * args.ep) for index in range(2)]
    if args.route_fixture == "hot":
        for index, item in enumerate(data):
            real_ids = torch.arange(8, device="npu") + (index % args.ep) * 8
            identities = torch.arange(4, device="npu") + args.real
            item["ids"] = torch.cat((real_ids, identities)).expand(args.tokens, -1).clone()
    identities = [{key: hash_tensor(item[key]) for key in ("gate", "down", "ids")} for item in data]
    return data, identities


def _replica_chain(layers: list, args: Any, data: list[dict]) -> tuple[torch.Tensor, list, bool]:
    """Observe public replica decisions while retaining both layer contexts."""
    plans = []
    with ExitStack() as observers:
        if args.replica_slots:
            for layer in layers:
                observers.enter_context(plan_observer(layer.executor, plans))
        output = chain(layers, args, data, False)
    active = any(not plan["home_only"] and plan["transfers"] for plan in plans)
    transfers = torch.tensor([active], dtype=torch.int32, device="npu")
    dist.all_reduce(transfers, op=dist.ReduceOp.MAX)
    return output, plans, bool(transfers.item())


def main() -> None:
    """Write all-rank terminal parity, runtime scope, and cleanup evidence."""
    args = arguments()
    os.environ["HYPER_MULTICORE_SHMEM_DATA_ENGINE"] = args.shmem_data_engine
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    set_operator_timeout(args.operator_timeout)
    dist.init_process_group("hccl")
    if dist.get_world_size() != args.ep:
        raise ValueError("WORLD_SIZE must match --ep")
    start_shmem_lifetime()
    data, identities = _layer_inputs(args, rank)
    other = _copy_inputs(data)
    reference = [NativeExperts(args.real, args.ep, dist.group.WORLD, args.swiglu_precision) for _ in data]
    native = chain(reference, args, data, True)
    expected = {key: None if value is None else value.detach().cpu()
                for key, value in results(data, native).items()}
    layers = [candidate(args) for _ in data]
    output, plans, replicas_exercised = _replica_chain(layers, args, other)
    replicas_pass = not args.replica_slots or replicas_exercised
    comparison = compare(results(other, output), expected)
    local_pass = all(value["strict_pass"] for value in comparison.values())
    flags = torch.tensor([local_pass], dtype=torch.int32, device="npu")
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    active_runtime = dict(shmem.debug_state())
    for layer in layers:
        layer.close()
    torch.npu.synchronize()
    runtime = dict(shmem.debug_state())
    expected_root = 1 if args.communication_backend == "hccl" else args.ep
    passed = (bool(flags.item()) and replicas_pass and active_runtime["root_size"] == expected_root
              and runtime["state"] == "Uninitialized")
    report = {"rank": rank, "ep_size": args.ep, "expert_layers": len(layers),
              "communication_backend": args.communication_backend, "shmem_data_engine": args.shmem_data_engine,
              "replica_slots": args.replica_slots, "replicas_exercised": replicas_exercised,
              "layer_identities": identities, "replica_plans": plans,
              "shape": {key: getattr(args, key) for key in
                        ("tokens", "hidden", "intermediate", "real", "zero", "top_k")},
              "comparison": comparison, "strict_pass": passed, "environment": environment(),
              "active_shmem_root_size": active_runtime["root_size"], "shmem_after_close": runtime}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rank": rank, "strict_pass": passed}), flush=True)
    dist.destroy_process_group()
    if not passed:
        raise RuntimeError("Two-layer MegaMoE lifecycle or parity check failed")


if __name__ == "__main__":
    main()
