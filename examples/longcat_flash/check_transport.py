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
"""Probe real SHMEM MTE connectivity and optionally verify Put completion on safe peers.

Run with torch.distributed.run. No remote operation is submitted on a missing MTE edge.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

from hyper_multicore import shmem
from hyper_multicore.modules.mega_moe.transport import require_mte_peers


def exercise(rank: int, size: int, sdma: bool) -> dict:
    """Check every source-to-destination Put and the following MTE Add signal.

    Args:
        rank: Root PE coordinate.
        size: Root size.
        sdma: Use host SDMA copies instead of device MTE copies.

    Returns:
        Terminal correctness metrics for both payload and completion counters.
    """
    rows = 4096
    destination = shmem.empty((size, rows), dtype=torch.int32)
    signal = shmem.empty((1,), dtype=torch.int32)
    source = torch.arange(rows, dtype=torch.int32, device="npu") + rank * rows
    destination.zero_()
    signal.zero_()
    shmem.host_barrier()
    for peer in range(size):
        shmem.put(destination[rank], source, peer, use_sdma=sdma)
        shmem.signal(signal, 1, peer, operation="add")
    shmem.host_barrier()
    torch.npu.synchronize()
    expected = torch.arange(size * rows, dtype=torch.int32, device="npu").view(size, rows)
    result = {"payload_equal": bool(torch.equal(destination, expected)),
              "signal_equal": int(signal.cpu()[0]) == size, "bytes_per_peer": source.numel() * source.element_size()}
    shmem.free(destination)
    shmem.free(signal)
    if not result["payload_equal"] or not result["signal_equal"]:
        raise RuntimeError(f"SHMEM {'SDMA' if sdma else 'MTE'} validation failed: {result}")
    return result


def main() -> None:
    """Write one terminal report per rank, including actual SDK reachability and cleanup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exercise", choices=("none", "mte", "sdma", "both"), default="none")
    args = parser.parse_args()
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("hccl")
    rank, size = dist.get_rank(), dist.get_world_size()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"rank": rank, "world_size": size, "device_name": torch.npu.get_device_name(),
              "torch": torch.__version__, "torch_npu": torch_npu.__version__}
    acquired = False
    try:
        shmem.acquire(dist.group.WORLD, heap_size_bytes=64 * 1024 * 1024)
        acquired = True
        reachable = shmem.peer_mte_accessibility()
        report["mte_reachable"] = list(reachable)
        peers = [None] * size
        dist.all_gather_object(peers, list(reachable))
        report["all_peers_mte_reachable"] = all(len(row) == size and all(row) for row in peers)
        report["guard_injection_pass"] = False
        injected = list(reachable)
        injected[(rank + 1) % size] = False
        try:
            require_mte_peers(injected, size)
        except RuntimeError as error:
            report["guard_injection_pass"] = "communication_backend" in str(error)
        if not report["guard_injection_pass"]:
            raise RuntimeError("The missing-MTE guard did not reject an unreachable peer")
        if args.exercise != "none":
            require_mte_peers(reachable, size)
            if not report["all_peers_mte_reachable"]:
                raise RuntimeError("Another rank has unreachable peers; no remote kernel was launched")
            for name in ("mte", "sdma"):
                if args.exercise in (name, "both"):
                    report[name] = exercise(rank, size, name == "sdma")
        report["status"] = "pass"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if acquired:
            shmem.release()
        report["shmem_after_close"] = dict(shmem.debug_state())
        (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"rank": rank, "status": report["status"],
                          "all_peers_mte_reachable": report.get("all_peers_mte_reachable")}), flush=True)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
