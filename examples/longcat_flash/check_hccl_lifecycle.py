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

import json
import os

import torch
import torch.distributed as dist

from native_reference import NativeExperts
from validate import arguments, candidate, compare, environment, fixture, start_shmem_lifetime

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


def main() -> None:
    """Write all-rank terminal parity, runtime scope, and cleanup evidence."""
    args = arguments()
    if args.communication_backend != "hccl" or args.replica_slots:
        raise ValueError("Lifecycle validation requires --communication-backend hccl and no replicas")
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    if dist.get_world_size() != args.ep:
        raise ValueError("WORLD_SIZE must match --ep")
    start_shmem_lifetime()
    data = [fixture(args, rank) for _ in range(2)]
    other = [{key: value.detach().clone().requires_grad_(True) if key in
              ("gate", "down", "hidden", "probability") else value for key, value in item.items()}
             for item in data]
    reference = [NativeExperts(args.real, args.ep, dist.group.WORLD, args.swiglu_precision) for _ in data]
    native = chain(reference, args, data, True)
    expected = {key: None if value is None else value.detach().cpu()
                for key, value in results(data, native).items()}
    layers = [candidate(args) for _ in data]
    output = chain(layers, args, other, False)
    comparison = compare(results(other, output), expected)
    local_pass = all(value["strict_pass"] for value in comparison.values())
    flags = torch.tensor([local_pass], dtype=torch.int32, device="npu")
    dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    active_runtime = dict(shmem.debug_state())
    for layer in layers:
        layer.close()
    torch.npu.synchronize()
    runtime = dict(shmem.debug_state())
    passed = bool(flags.item()) and active_runtime["root_size"] == 1 and runtime["state"] == "Uninitialized"
    report = {"rank": rank, "ep_size": args.ep, "expert_layers": len(layers),
              "shape": {key: getattr(args, key) for key in
                        ("tokens", "hidden", "intermediate", "real", "zero", "top_k")},
              "comparison": comparison, "strict_pass": passed, "environment": environment(),
              "active_shmem_root_size": active_runtime["root_size"], "shmem_after_close": runtime}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rank": rank, "strict_pass": passed}), flush=True)
    dist.destroy_process_group()
    if not passed:
        raise RuntimeError("Two-layer HCCL MegaMoE lifecycle or parity check failed")


if __name__ == "__main__":
    main()
