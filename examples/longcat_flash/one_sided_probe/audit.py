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
"""Audit every rank and recompute transport timings from terminal reports."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def audit(reports_dir: Path, kind: str, payload: Path) -> dict:
    """Reject partial runs and derive statistics from per-rank event measurements.

    Args:
        reports_dir: Directory containing all sixteen terminal rank reports.
        kind: Paired primitive or indexed per-expert pipeline protocol.
        payload: Isolated payload directory containing the build provenance.

    Returns:
        Compact, path-independent correctness and timing evidence.
    """
    manifest = json.loads((payload / "provenance.json").read_text())
    library_digest = manifest["native_sha256"]["liblongcat_one_sided_probe.so"]
    for name, digest in manifest["native_sha256"].items():
        if hashlib.sha256((payload / "lib" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Payload checksum mismatch: {name}")
    reports = [json.loads((reports_dir / f"rank{rank}.json").read_text()) for rank in range(16)]
    key_fields = ("mode", "bytes", "sample") if kind == "pair" else ("rows_per_source_expert", "sample")
    indexed = []
    for rank, report in enumerate(reports):
        if (report["rank"] != rank or report["world_size"] != 16 or report["status"] != "pass"
                or report["shmem_initialized_after_close"] or report["library_sha256"] != library_digest):
            raise ValueError(f"Invalid terminal state or artifact identity on rank {rank}")
        records = {tuple(record[field] for field in key_fields): record for record in report["records"]}
        if not records or len(records) != len(report["records"]):
            raise ValueError(f"Missing or duplicated cases on rank {rank}")
        if any(not record["all_ranks_pass"] for record in records.values()):
            raise ValueError(f"A correctness check failed on rank {rank}")
        status_field = "status" if kind == "pair" else "observer_status"
        if any(record[status_field] != 0 for record in records.values()):
            raise ValueError(f"The device protocol reported a failure on rank {rank}")
        indexed.append(records)
    keys = set(indexed[0])
    if any(set(records) != keys for records in indexed):
        raise ValueError("Ranks completed different case sets")
    if kind == "pair":
        engines = {report["engine"] for report in reports}
        if len(engines) != 1 or not engines.issubset({"mte", "roce"}):
            raise ValueError("Ranks selected different or invalid transport engines")
        modes = (0,) if engines == {"mte"} else (1, 2)
        expected_groups = {(mode, size) for mode in modes for size in (0, 16384, 1572864, 25165824)}
    else:
        expected_groups = {(0,), (128,)}
    if {key[:-1] for key in keys} != expected_groups:
        raise ValueError("The terminal reports omit required payload or protocol cases")
    sample_sets = [{key[-1] for key in keys if key[:-1] == group} for group in expected_groups]
    if any(samples != sample_sets[0] for samples in sample_sets):
        raise ValueError("Payload cases contain different sample sets")
    if sample_sets[0] != set(range(len(sample_sets[0]))):
        raise ValueError("The sample sequence is incomplete")
    groups = {}
    for key in sorted(keys):
        records = [items[key] for items in indexed]
        if len({record["warmup"] for record in records}) != 1:
            raise ValueError(f"Rank warmup masks disagree: {key}")
        elapsed = max(record["event_ms"] for record in records)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"Invalid device event duration: {key}")
        if any(not math.isclose(record["rank_max_event_ms"], elapsed, rel_tol=1e-6, abs_tol=1e-7)
               for record in records):
            raise ValueError(f"Stored rank maximum disagrees with events: {key}")
        if records[0]["warmup"]:
            continue
        group = groups.setdefault(key[:-1], {"rank_max_event_ms": [], "records": []})
        group["rank_max_event_ms"].append(elapsed)
        group["records"].extend(records)
    statistics_by_case = []
    if set(groups) != expected_groups:
        raise ValueError("A required payload case has no measured samples after warmup")
    for key, group in sorted(groups.items()):
        records = group["records"]
        summary = dict(zip(key_fields[:-1], key))
        summary.update({"measured_iterations": len(group["rank_max_event_ms"]),
                        "rank_samples": len(records),
                        "median_rank_max_event_ms": statistics.median(group["rank_max_event_ms"])})
        if kind == "pipeline":
            windows = [record["ready_window_ms"] for record in records]
            summary.update({"early_ready_rank_samples": sum(record["first_ready_before_last_credit"]
                                                            for record in records),
                            "ready_window_ms_min": min(windows),
                            "ready_window_ms_median": statistics.median(windows),
                            "ready_window_ms_max": max(windows)})
        statistics_by_case.append(summary)
    return {"status": "pass", "kind": kind, "ranks": 16, "cases_per_rank": len(keys),
            "total_checked_rank_cases": len(keys) * 16, "all_runtimes_closed": True,
            "library_sha256": library_digest, "shmem_commit": manifest["sdk"]["commit"],
            "source_sha256": manifest["sources"], "rdma_support": manifest["rdma_support"],
            "device": reports[0]["device"], "torch": reports[0]["torch"],
            "torch_npu": reports[0]["torch_npu"], "scope": reports[0]["scope"],
            "statistics": statistics_by_case}


def main() -> None:
    """Write a compact audit without importing Torch or requiring an NPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("pair", "pipeline"))
    parser.add_argument("--reports", required=True, type=Path)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.reports, args.kind, args.payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "ranks": result["ranks"],
                      "total_checked_rank_cases": result["total_checked_rank_cases"]}))


if __name__ == "__main__":
    main()
