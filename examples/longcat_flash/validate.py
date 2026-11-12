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
"""Standalone LongCat expert-component precision and performance worker."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from typing import Any
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu

from _precision import STRICT_GATE, compare_tensors
from native_reference import NativeExperts

import hyper_multicore
from hyper_multicore import shmem
from hyper_multicore.longcat import LongcatFlashExperts
from hyper_multicore.longcat._routing import route_experts
from hyper_multicore.modules.mega_moe.plan import _build_task_values

def _validate_backend(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject replica fixture and backend combinations not supported by the worker."""
    if args.route_fixture == "hot" and (args.real != 8 * args.ep or args.zero != 4 * args.ep or args.top_k != 12):
        parser.error("the hot fixture requires eight real and four identity experts per rank and K=12")
    if args.replica_slots and (args.ep < 4 or args.real != 8 * args.ep or args.zero != 4 * args.ep
                               or args.top_k != 12):
        parser.error("the replica route fixtures require EP>=4, eight real and four identity experts per rank, K=12")
    if args.check == "accuracy" and args.backend == "native":
        parser.error("accuracy compares native against MegaMoE push or pull")
    if args.backend == "native" and args.replica_slots:
        parser.error("the native baseline cannot be configured with replica slots")


def _validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject inconsistent dimensions and replica fixtures before distributed startup."""
    args.real = 8 * args.ep if args.real is None else args.real
    args.zero = 4 * args.ep if args.zero is None else args.zero
    if min(args.ep, args.tokens, args.hidden, args.intermediate, args.real, args.top_k,
           args.updates, args.samples) <= 0 or min(args.zero, args.replica_slots, args.warmup) < 0:
        parser.error("invalid non-positive dimension, count or negative slot count")
    if args.real % args.ep or args.zero % args.ep or args.top_k > args.real + args.zero:
        parser.error("expert counts must divide EP and top-k must fit the total experts")
    _validate_backend(parser, args)


def arguments() -> argparse.Namespace:
    """Parse a single-node EP configuration with original LongCat widths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", choices=("accuracy", "benchmark"), default="accuracy")
    parser.add_argument("--backend", choices=("native", "push", "pull"), default="push")
    parser.add_argument("--communication-backend", choices=("shmem", "hccl"), default="shmem")
    parser.add_argument("--exercise-edge-routes", action="store_true",
                        help="Include hot owners, empty sources, and all-identity steps without requiring replicas")
    parser.add_argument("--routing-backend", choices=("torch", "triton"), default="torch")
    parser.add_argument("--swiglu-precision", choices=("fused", "hf"), default="fused")
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--real", type=int)
    parser.add_argument("--zero", type=int)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--replica-slots", type=int, default=0)
    parser.add_argument("--route-fixture", choices=("router", "hot"), default="router")
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _validate_arguments(parser, args)
    return args


def start_shmem_lifetime() -> None:
    """Choose a local SHMEM endpoint after the HCCL group is ready."""
    endpoint = [None]
    if dist.get_rank() == 0:
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            endpoint[0] = f"tcp://127.0.0.1:{server.getsockname()[1]}"
    dist.broadcast_object_list(endpoint, src=0)
    os.environ["HYPER_MULTICORE_SHMEM_BOOTSTRAP_ENDPOINT"] = endpoint[0]


def hash_tensor(value: torch.Tensor) -> str:
    """Hash each tensor byte outside timed intervals using bounded CPU chunks."""
    digest = hashlib.sha256()
    flattened = value.detach().reshape(-1)
    for start in range(0, flattened.numel(), 16 * 1024 * 1024):
        part = flattened[start:start + 16 * 1024 * 1024].cpu().contiguous().view(torch.uint8)
        digest.update(part.numpy().tobytes())
    return digest.hexdigest()


def environment() -> dict:
    """Capture selected source, Python/PTA, device, and loaded native payload."""
    source = Path(hyper_multicore.__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                           text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"],
                                         text=True, stderr=subprocess.DEVNULL).splitlines()
    except (OSError, subprocess.CalledProcessError):
        revision, status = None, []
    digest = hashlib.sha256()
    for path in sorted((source / "hyper_multicore").rglob("*.py")):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    worker_files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in Path(__file__).parent.glob("*.py")}
    mapped = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
                     if "/hyper_multicore/" in line and line.split()[-1].startswith("/")})
    libraries = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                 for name in mapped if Path(name).is_file()}
    hp_loaded = any(name == "hyper_parallel" or name.startswith("hyper_parallel.") for name in sys.modules)
    return {"source": str(source), "git_revision": revision, "git_status": status,
            "source_tree_sha256": digest.hexdigest(), "worker_sha256": worker_files,
            "python": platform.python_version(), "host_arch": platform.machine(), "torch": torch.__version__,
            "torch_npu": torch_npu.__version__, "chip": torch.npu.get_device_name(),
            "cann_root": os.environ.get("ASCEND_HOME_PATH"), "native_sha256": libraries,
            "hyper_parallel_loaded": hp_loaded}


def fixture(args: argparse.Namespace, rank: int) -> dict:
    """Generate deterministic owner-local EOI weights and HF-compatible router operands."""
    torch.npu.manual_seed(22000 + rank)
    shape_gate = ((args.real + args.zero) // args.ep, 2 * args.intermediate, args.hidden)
    shape_down = (args.real // args.ep, args.hidden, args.intermediate)
    gate = torch.randn(shape_gate, device="npu", dtype=torch.bfloat16).mul_(0.02)
    down = torch.randn(shape_down, device="npu", dtype=torch.bfloat16).mul_(0.02)
    torch.manual_seed(12000)
    router = torch.nn.Linear(args.hidden, args.real + args.zero, bias=False).to("npu", torch.bfloat16)
    torch.npu.manual_seed(32000 + rank)
    hidden = torch.randn(args.tokens, args.hidden, device="npu", dtype=torch.bfloat16)
    with torch.no_grad():
        scores = F.linear(hidden.float(), router.weight.float()).softmax(-1)  # pylint: disable=not-callable
        ids = torch.topk(scores, k=args.top_k, dim=-1, sorted=False)[1]
        probability = scores.gather(1, ids) * 6.0
        if args.route_fixture == "hot":
            real = torch.arange(8, device="npu")
            identity = torch.arange(4, device="npu") + args.real
            ids = torch.cat((real, identity)).expand(args.tokens, -1).clone()
    upstream = torch.randn_like(hidden)
    return {"gate": gate.requires_grad_(), "down": down.requires_grad_(), "hidden": hidden.requires_grad_(),
            "ids": ids, "probability": probability.detach().requires_grad_(), "upstream": upstream}


def candidate(args: argparse.Namespace) -> LongcatFlashExperts:
    """Build the production HPM API, retaining caller-owned padded weights."""
    return LongcatFlashExperts(hidden_size=args.hidden, intermediate_size=args.intermediate,
                               num_experts=args.real, zero_experts=args.zero, top_k=args.top_k,
                               ep_size=args.ep, ep_group=dist.group.WORLD, dispatch_mode=args.backend,
                               swiglu_precision=args.swiglu_precision, routing_backend=args.routing_backend,
                               communication_backend=args.communication_backend,
                               replica_slots_per_rank=args.replica_slots, gate_up_padded=True,
                               initial_token_capacity=args.tokens * args.top_k)


def routes(args: argparse.Namespace, data: dict, step: int, rank: int) -> tuple[str, torch.Tensor]:
    """Vary owner load and include empty-source and all-identity replica steps."""
    if not args.replica_slots and not args.exercise_edge_routes:
        return "router", data["ids"]
    kinds = ("hot", "hot", "normal", "hot", "empty_source", "identity", "hot", "normal", "hot", "hot")
    kind = kinds[step % len(kinds)]
    if kind == "normal":
        return kind, data["ids"]
    owner = (step // 3) % args.ep
    real = torch.arange(8, device="npu") + owner * 8
    identity = torch.arange(4, device="npu") + args.real
    ids = torch.cat((real, identity)).expand(args.tokens, -1).clone()
    if kind == "identity" or (kind == "empty_source" and rank == 0):
        ids = (torch.arange(args.top_k, device="npu") + args.real).expand(args.tokens, -1).clone()
    return kind, ids.long()


def execute(executor: object, args: argparse.Namespace, data: dict, ids: torch.Tensor,
            native: bool, backward: bool) -> torch.Tensor:
    """Execute one complete component step with fresh gradients."""
    for key in ("gate", "down", "hidden", "probability"):
        data[key].grad = None
    with torch.set_grad_enabled(backward):
        weights = (data["gate"], data["down"])
        if native:
            output = route_experts(executor, data["hidden"], ids, data["probability"],
                                   args.real, args.zero, args.top_k, dist.group.WORLD, weights,
                                   args.routing_backend if args.check == "benchmark" else "torch")
        else:
            output = executor(data["hidden"], ids, data["probability"], expert_weights=weights)
        if backward:
            output.backward(data["upstream"])
    return output


def values(data: dict, output: torch.Tensor) -> dict:
    """Collect output and all required input, router and parameter gradients."""
    return {"output": output, "input_gradient": data["hidden"].grad,
            "probability_gradient": data["probability"].grad,
            "gate_up_gradient": data["gate"].grad, "down_gradient": data["down"].grad}


def compare(actual: dict, expected: dict) -> dict:
    """Compare all tensor elements and preserve globally unused parameter semantics."""
    result = {}
    for name, value in actual.items():
        reference = expected[name]
        if value is None or reference is None:
            result[name] = {"strict_pass": value is None and reference is None,
                            "both_unused": value is None and reference is None}
        else:
            result[name] = compare_tensors(value, reference)
    return result


def plan_observer(engine: object, plans: list) -> Any:
    """Observe production replica decisions without changing the planner."""
    original = engine._prepare_replica_route

    def observed(*args: Any, **kwargs: Any) -> Any:
        """Record the plan returned by the unmodified replica planner."""
        route = original(*args, **kwargs)
        if route is not None:
            plans.append({"home_only": route.home_only,
                          "transfers": [asdict(item) for item in route.plan.transfers]})
        return route

    return patch.object(engine, "_prepare_replica_route", side_effect=observed)


def execution_plan(layer: LongcatFlashExperts) -> dict:
    """Capture the actual last MegaMoE plan, separate from the requested controls."""
    executor = layer.executor
    if executor.communication_backend == "hccl":
        executor = executor._resource_group.hccl_executor.compute
    resources = executor._resource_group.resources
    plan = resources.last_execution_plan
    spec = plan.spec
    task_values = _build_task_values(spec)
    operators = ("dispatch", "combine")
    tasks = {name: plan.forward_graph.get_op(name).split_spec.task_num_fn(task_values) for name in operators}
    return {"communication_backend": layer.executor.communication_backend, "compute_ep_size": spec.ep_size,
            "plan_tokens": spec.plan_tokens, "route_source_rows": spec.schedule_source_expert_rows,
            "route_expert_rows": spec.schedule_expert_rows, "communication_rows": spec._communication_rows,
            "compact_communication": spec._compact_communication, "forward_tasks": tasks}


def _update_parameters(data: dict, other: dict, masters: list | None) -> bool:
    """Apply independent FP32 master updates and compare the resulting BF16 weights."""
    if masters is None:
        return True
    with torch.no_grad():
        for index, item in enumerate((data, other)):
            for key in ("gate", "down"):
                if item[key].grad is not None:
                    masters[index][key].add_(item[key].grad.float(), alpha=-1e-5)
                    item[key].copy_(masters[index][key])
    return all(hash_tensor(data[key]) == hash_tensor(other[key]) for key in ("gate", "down"))


def _accuracy_step(args: argparse.Namespace, data: dict, other: dict, layer: LongcatFlashExperts,
                   reference: NativeExperts, masters: list | None, step: int, rank: int) -> dict:
    """Run, compare and update one native/MegaMoE route fixture."""
    kind, ids = routes(args, data, step, rank)
    ref_output = execute(reference, args, data, ids, native=True, backward=True)
    expected = {name: None if value is None else value.detach().cpu()
                for name, value in values(data, ref_output).items()}
    plans = []
    if args.replica_slots:
        with plan_observer(layer.executor, plans):
            out = execute(layer, args, other, ids, native=False, backward=True)
    else:
        out = execute(layer, args, other, ids, native=False, backward=True)
    comparison = compare(values(other, out), expected)
    actual_plan = None if kind == "identity" else execution_plan(layer)
    match = _update_parameters(data, other, masters)
    local_pass = all(item["strict_pass"] for item in comparison.values()) and match
    state = torch.tensor([local_pass], dtype=torch.int32, device="npu")
    dist.all_reduce(state, op=dist.ReduceOp.MIN)
    return {"step": step, "routes": kind, "comparison": comparison, "plans": plans, "execution_plan": actual_plan,
            "parameters_match": match, "all_ranks_strict_pass": bool(state.item())}


def accuracy(args: argparse.Namespace, data: dict, layer: LongcatFlashExperts, rank: int) -> dict:
    """Compare native and MegaMoE over independently updated FP32 masters."""
    reference = NativeExperts(args.real, args.ep, dist.group.WORLD, args.swiglu_precision)
    other = {key: value.detach().clone().requires_grad_(True) if key in ("gate", "down") else value
             for key, value in data.items()}
    masters = [{key: item[key].detach().float().clone() for key in ("gate", "down")}
               for item in (data, other)] if args.updates > 1 else None
    records = []
    for step in range(args.updates):
        records.append(_accuracy_step(args, data, other, layer, reference, masters, step, rank))
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"progress-rank{rank}.json").write_text(json.dumps(records, indent=2) + "\n")
        if not records[-1]["all_ranks_strict_pass"]:
            break
    active = any(not plan["home_only"] and plan["transfers"] for record in records for plan in record["plans"])
    transfer = torch.tensor([active], dtype=torch.int32, device="npu")
    dist.all_reduce(transfer, op=dist.ReduceOp.MAX)
    return {"records": records, "replicas_exercised": bool(transfer.item()),
            "strict_pass": len(records) == args.updates and all(r["all_ranks_strict_pass"] for r in records)
            and (not args.replica_slots or bool(transfer.item()))}


def benchmark(args: argparse.Namespace, data: dict, executor: object) -> dict:
    """Retain every synchronized local and rank-maximum forward/backward sample."""
    is_native = args.backend == "native"
    plans = []
    if args.replica_slots:
        with plan_observer(executor.executor, plans):
            execute(executor, args, data, data["ids"], native=False, backward=True)
    for _ in range(args.warmup):
        execute(executor, args, data, data["ids"], native=is_native, backward=True)
    torch.npu.synchronize()
    dist.barrier()
    local, maximum = [], []
    for _ in range(args.samples):
        start = time.perf_counter()
        out = execute(executor, args, data, data["ids"], native=is_native, backward=True)
        torch.npu.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        rank_max = torch.tensor([elapsed], device="npu")
        dist.all_reduce(rank_max, op=dist.ReduceOp.MAX)
        local.append(elapsed)
        maximum.append(float(rank_max.item()))
    finite = bool(torch.isfinite(out).all()) and all(
        data[key].grad is not None and bool(torch.isfinite(data[key].grad).all())
        for key in ("gate", "down", "hidden", "probability"))
    return {"local_ms": local, "rank_max_ms": maximum, "finite": finite, "replica_plans": plans}


def main() -> None:
    """Run one local EP worker and write all-rank terminal evidence."""
    args = arguments()
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    if dist.get_world_size() != args.ep or int(os.environ["LOCAL_WORLD_SIZE"]) != args.ep:
        raise ValueError("this worker requires a single node and WORLD_SIZE == LOCAL_WORLD_SIZE == EP")
    start_shmem_lifetime()
    data = fixture(args, rank)
    identities = {key: hash_tensor(value) for key, value in data.items()}
    layer = None if args.backend == "native" else candidate(args)
    executor = layer if layer is not None else NativeExperts(args.real, args.ep, dist.group.WORLD,
                                                              args.swiglu_precision)
    result = accuracy(args, data, layer, rank) if args.check == "accuracy" else benchmark(args, data, executor)
    final_plan = execution_plan(layer) if layer is not None else None
    if layer is not None:
        layer.close()
    torch.npu.synchronize()
    runtime_environment, final_shmem = environment(), shmem.debug_state()
    runtime_ok = (not runtime_environment["hyper_parallel_loaded"] and final_shmem["state"] == "Uninitialized"
                  and (layer is None or bool(runtime_environment["native_sha256"])))
    report = {"rank": rank, "check": args.check, "backend": args.backend,
              "shape": {key: getattr(args, key) for key in
                        ("ep", "tokens", "hidden", "intermediate", "real", "zero", "top_k")},
              "local_experts": {"real": args.real // args.ep, "identity": args.zero // args.ep},
              "weight_layout": "EOI", "gate_shape": list(data["gate"].shape),
              "down_shape": list(data["down"].shape), "replica_slots": args.replica_slots,
              "route_fixture": args.route_fixture,
              "compact_communication_requested": os.environ.get("HYPER_MULTICORE_MOE_COMPACT_COMMUNICATION", "0"),
              "last_execution_plan": final_plan,
              "routing_backend": args.routing_backend, "swiglu_precision": args.swiglu_precision,
              "scope": "expert component; precomputed router, no full model",
              "strict_gate": STRICT_GATE, "identities": identities, "result": result,
              "shmem_after_close": final_shmem, "runtime_ok": runtime_ok, "environment": runtime_environment}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()
    if not runtime_ok:
        raise RuntimeError("independent runtime, native library, or SHMEM cleanup check failed")
    if not result.get("strict_pass", result.get("finite", False)):
        raise RuntimeError("LongCat standalone validation failed; inspect all rank reports")


if __name__ == "__main__":
    main()
