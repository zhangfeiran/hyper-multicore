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
"""Compare the installed replica count kernel against a CPU integer oracle."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_npu  # pylint: disable=unused-import

from hyper_multicore.replica import count_kernel


def expected_counts(ids: torch.Tensor, experts: int) -> torch.Tensor:
    """Count clamped IDs and independently mark invalid or duplicate selections."""
    values = ids.cpu()
    result = torch.bincount(values.clamp(0, experts - 1).flatten(), minlength=experts)
    invalid = bool(((values < 0) | (values >= experts)).any())
    if values.shape[1] > 1:
        ordered = values.sort(dim=1).values
        invalid = invalid or bool((ordered[:, 1:] == ordered[:, :-1]).any())
    return torch.cat((result, torch.tensor([int(invalid)], dtype=torch.int64)))


def run_case(experts: int, rows: int, kind: str) -> dict:
    """Compare every expert count and the invalid flag for one boundary case."""
    top_k = min(8, experts)
    base = torch.arange(rows, dtype=torch.int64)[:, None]
    offsets = torch.arange(top_k, dtype=torch.int64)[None, :]
    ids = (base * 17 + offsets * 7) % experts
    if kind == "invalid" and rows:
        ids[0, 0] = experts
    if kind == "duplicate" and rows and top_k > 1:
        ids[0, 1] = ids[0, 0]
    actual = count_kernel.try_count_payload(ids.to("npu"), experts)
    if actual is None:
        raise RuntimeError("The installed native count kernel is unavailable on this NPU")
    torch.npu.synchronize()
    reference = expected_counts(ids, experts)
    actual = actual.cpu()
    if not torch.equal(actual, reference):
        differing = (actual != reference).nonzero().reshape(-1)
        raise AssertionError(f"count mismatch: E={experts}, rows={rows}, fixture={kind}, "
                             f"first_indices={differing[:8].tolist()}")
    return {"experts": experts, "rows": rows, "top_k": top_k, "fixture": kind, "pass": True}


def main() -> None:
    """Run 64-column and 128-row boundaries and write an optional JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    library = count_kernel._count_library()
    abi = library.replica_counts_abi_version()
    if abi != 2:
        raise RuntimeError("This check requires the E512-capable replica count ABI 2")
    records = []
    for experts in (1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512):
        for rows in (0, 1, 31, 32, 33, 127, 128, 129, 2048):
            for kind in ("valid", "invalid", "duplicate"):
                records.append(run_case(experts, rows, kind))
    path = count_kernel._LIBRARY_PATH.resolve()
    report = {"device": torch.npu.get_device_name(), "abi": abi, "native_library": str(path),
              "native_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "cases": records,
              "status": "PASS"}
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(f"PASS: {len(records)} exact count cases; ABI {abi}; {path}")


if __name__ == "__main__":
    main()
