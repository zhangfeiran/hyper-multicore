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
"""Compare literal HF experts, native NPU GMM, and standalone MegaMoE."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import transformers
from transformers import LongcatFlashConfig
from transformers.models.longcat_flash.modeling_longcat_flash import LongcatFlashExperts as HFExperts

from _precision import STRICT_GATE
from native_reference import NativeExperts
from validate import arguments, candidate, compare, environment, execute, fixture, hash_tensor, start_shmem_lifetime
from validate import values as component_values

from hyper_multicore import shmem


def snapshot(values: dict) -> dict:
    """Retain full reference tensors on CPU, including absent parameter gradients."""
    return {name: None if value is None else value.detach().cpu() for name, value in values.items()}


def hf_step(module: HFExperts, data: dict) -> dict:
    """Run the installed HF class on the same fixed IDs, probabilities and parameters."""
    hidden = data["hidden"].detach().clone().requires_grad_()
    probability = data["probability"].detach().clone().requires_grad_()
    module.zero_grad(set_to_none=True)
    output = module(hidden, data["ids"], probability)
    output.backward(data["upstream"])
    return snapshot({"output": output, "input_gradient": hidden.grad, "probability_gradient": probability.grad,
                     "gate_up_gradient": module.gate_up_proj.grad, "down_gradient": module.down_proj.grad})


def record(name: str, actual: dict, reference: dict, *, primary: bool) -> dict:
    """Retain elementwise details; HF versus GMM is diagnostic, not a primary gate."""
    comparison = compare(actual, reference)
    passed = all(item["strict_pass"] for item in comparison.values())
    return {"case": name, "primary": primary, "strict_pass": passed, "comparison": comparison}


def hf_module(args: object, data: dict) -> HFExperts:
    """Bind independent EOI parameters to the actual installed HF implementation."""
    config = LongcatFlashConfig.from_dict({"hidden_size": args.hidden,
                                           "expert_ffn_hidden_size": args.intermediate,
                                           "n_routed_experts": args.real,
                                           "zero_expert_num": args.zero, "moe_topk": args.top_k})
    with torch.device("meta"):
        module = HFExperts(config)
    module.gate_up_proj = torch.nn.Parameter(data["gate"].detach().clone())
    module.down_proj = torch.nn.Parameter(data["down"].detach().clone())
    return module


def native_and_mega(args: object, data: dict, expected_hf: dict, records: list) -> None:
    """Compare each activation mode against HF and independently against native GMM."""
    for mode in ("hf", "fused"):
        args.swiglu_precision = mode
        reference = NativeExperts(args.real, args.ep, dist.group.WORLD, mode)
        output = execute(reference, args, data, data["ids"], native=True, backward=True)
        native = snapshot(component_values(data, output))
        records.append(record(f"native_GMM_{mode}_vs_HF", native, expected_hf, primary=False))
        layer = candidate(args)
        try:
            output = execute(layer, args, data, data["ids"], native=False, backward=True)
            records.append(record(f"Mega_{mode}_vs_native_GMM_{mode}",
                                  component_values(data, output), native, primary=True))
        finally:
            layer.close()
        torch.npu.synchronize()
        start_shmem_lifetime()


def _validate_configuration(args: object) -> None:
    """Restrict the literal HF reference to the original-width single-rank fixture."""
    if (args.check, args.ep, args.real, args.zero, args.hidden, args.intermediate, args.top_k) != (
            "accuracy", 1, 8, 4, 6144, 2048, 12):
        raise ValueError("HF diagnosis requires EP1/E8/Z4/H6144/I2048/TopK12; only tokens can be reduced")
    if args.replica_slots or args.route_fixture != "router":
        raise ValueError("HF diagnosis uses the fixed router fixture and does not enable replicas")


def main() -> None:
    """Write the full HF diagnostic while gating only same-semantics comparisons."""
    args = arguments()
    _validate_configuration(args)
    output = args.output / "rank0.json"
    if output.exists():
        raise FileExistsError(f"Use a fresh diagnostic output directory: {output}")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    torch.use_deterministic_algorithms(True)
    dist.init_process_group("hccl")
    if dist.get_world_size() != 1:
        raise ValueError("HF diagnosis requires exactly one process")
    start_shmem_lifetime()
    data = fixture(args, 0)
    identities = {name: hash_tensor(value) for name, value in data.items()}
    module = hf_module(args, data)
    expected = hf_step(module, data)
    records = [record("HF_repeat", hf_step(module, data), expected, primary=True)]
    del module
    native_and_mega(args, data, expected, records)
    torch.npu.synchronize()
    runtime_environment = environment()
    state = shmem.debug_state()
    passed = all(row["strict_pass"] for row in records if row["primary"])
    passed = passed and not runtime_environment["hyper_parallel_loaded"] and state["state"] == "Uninitialized"
    hf_strict = all(row["strict_pass"] for row in records if not row["primary"])
    report = {"primary_status": "PASS" if passed else "FAIL",
              "hf_strict_status": "PASS" if hf_strict else "FAIL", "rank": 0,
              "transformers": transformers.__version__,
              "hf_source": str(Path(transformers.__file__).resolve()), "gate": STRICT_GATE, "records": records,
              "shape": {key: getattr(args, key) for key in ("ep", "tokens", "hidden", "intermediate",
                                                           "real", "zero", "top_k")},
              "identities": identities, "environment": runtime_environment, "shmem_after_close": state,
              "scope": "EP1 expert component; native versus HF comparisons are diagnostic, not primary acceptance"}
    args.output.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    dist.destroy_process_group()
    print(f"Primary: {report['primary_status']}; HF strict diagnostics: {report['hf_strict_status']}; "
          f"report: {output}", flush=True)
    if not passed:
        raise RuntimeError("HF repeat or same-semantics native/MegaMoE comparison failed")


if __name__ == "__main__":
    main()
