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
"""Validate device-initiated data/notify ordering without HCCL in the transfer primitive."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import traceback

import numpy as np
import torch
import torch.distributed as dist
import torch_npu


class Probe:
    """Own the isolated C ABI; the probe does not load the MegaMoE native payload."""

    def __init__(self, path: Path) -> None:
        """Load the reviewed C ABI from a concrete build artifact."""
        self.library = ctypes.CDLL(str(path.resolve()))
        signatures = {
            "probe_last_error": (ctypes.c_char_p, []),
            "probe_unique_id_size": (ctypes.c_uint64, []),
            "probe_unique_id": (ctypes.c_int, [ctypes.c_void_p]),
            "probe_open": (ctypes.c_int, [ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                         ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int]),
            "probe_transport_bit": (ctypes.c_uint32, [ctypes.c_int]),
            "probe_peer_transport": (ctypes.c_uint32, [ctypes.c_int]),
            "probe_prepare": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int32]),
            "probe_step": (ctypes.c_int, [ctypes.c_uint32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                                         ctypes.c_uint64, ctypes.POINTER(ctypes.c_float),
                                         ctypes.POINTER(ctypes.c_uint64)]),
            "probe_read": (ctypes.c_int, [ctypes.c_void_p, ctypes.c_uint64]),
            "probe_close": (ctypes.c_int, []),
            "probe_initialized": (ctypes.c_int, []),
        }
        for name, (result, arguments) in signatures.items():
            function = getattr(self.library, name)
            function.restype, function.argtypes = result, arguments

    def check(self, status: int) -> None:
        """Surface the actual CANN or SDK failure without advancing the protocol."""
        if status:
            raise RuntimeError(self.library.probe_last_error().decode(errors="replace"))

    def step(self, size: int, peer: int, generation: int, mode: int,
             wait_cycles: int) -> tuple[float, list[int]]:
        """Run one kernel containing data Put, notification, and receiver observation."""
        elapsed = ctypes.c_float()
        result = (ctypes.c_uint64 * 20)()
        self.check(self.library.probe_step(size, peer, generation, mode, wait_cycles,
                                          ctypes.byref(elapsed), result))
        return elapsed.value, list(result)


def tag(rank: int, generation: int) -> np.uint32:
    """Vary every payload word by source and generation to detect stale or partial delivery."""
    return np.uint32(((rank + 1) * 2654435761 + generation * 65537) & 0xFFFFFFFF)


def main() -> None:
    """Write all-rank data integrity, actual transport flags, timing, and cleanup evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=("mte", "roce"), required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    if args.samples <= 0 or args.warmup < 0:
        raise ValueError("--samples must be positive and --warmup must be nonnegative")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    rank, size = dist.get_rank(), dist.get_world_size()
    if size < 2 or size % 2:
        raise ValueError("The paired probe requires a positive even world size")
    peer = (rank + size // 2) % size
    probe = Probe(args.library)
    lib = probe.library
    args.output.mkdir(parents=True, exist_ok=True)
    uid_size = lib.probe_unique_id_size()
    uid = ctypes.create_string_buffer(uid_size)
    if rank == 0:
        probe.check(lib.probe_unique_id(uid))
    identifiers = [uid.raw if rank == 0 else None]
    dist.broadcast_object_list(identifiers, src=0)
    uid = ctypes.create_string_buffer(identifiers[0], uid_size)
    maximum = 2048 * 6144 * 2
    report = {"rank": rank, "world_size": size, "peer": peer, "engine": args.engine,
              "scope": "one AIV transfer protocol; no MoE or compute overlap acceptance",
              "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
              "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "device": torch.npu.get_device_name(), "records": []}
    opened = False
    try:
        status = lib.probe_open(rank, size, uid, 128 * 1024 * 1024, maximum, args.engine == "roce")
        opened = bool(lib.probe_initialized())
        statuses = [None] * size
        dist.all_gather_object(statuses, (status, lib.probe_last_error().decode(errors="replace")))
        if any(item[0] for item in statuses):
            raise RuntimeError(f"SHMEM initialization failed: {statuses}")
        masks = [lib.probe_peer_transport(other) for other in range(size)]
        bit = lib.probe_transport_bit(args.engine == "roce")
        report["peer_transport_masks"] = masks
        reachable = torch.tensor([bool(masks[peer] & bit)], device="npu", dtype=torch.int32)
        dist.all_reduce(reachable, op=dist.ReduceOp.MIN)
        if not bool(reachable.item()):
            raise RuntimeError("The requested transport is unavailable; no transfer kernel was launched")
        calibration = [probe.step(0, peer, 0, -1, 1_000_000) for _ in range(3)]
        ticks_per_ms = statistics.median(ticks[1] / ms for ms, ticks in calibration)
        wait_cycles = int(ticks_per_ms * 2000)
        report["ticks_per_ms"] = ticks_per_ms
        generation = 0
        modes = (0,) if args.engine == "mte" else (1, 2)
        for mode in modes:
            for length in (0, 16384, 128 * 6144 * 2, maximum):
                indices = np.arange(length // 4, dtype=np.uint32)
                for sample in range(args.warmup + args.samples):
                    generation += 1
                    source = indices ^ tag(rank, generation)
                    probe.check(lib.probe_prepare(source.ctypes.data, length, generation))
                    dist.barrier()
                    torch.npu.synchronize()
                    elapsed, result = probe.step(length, peer, generation, mode, wait_cycles)
                    expected = indices ^ tag(peer, generation)
                    readback = np.empty(length // 4 if length else 8, dtype=np.uint32)
                    probe.check(lib.probe_read(readback.ctypes.data, readback.nbytes))
                    expected = expected if length else np.full(8, 0xA5A5A5A5, dtype=np.uint32)
                    points = np.array([(len(expected) - 1) * index // 15 for index in range(16)])
                    immediate = np.asarray(result[4:20], dtype=np.uint32)
                    valid = result[0] == 0 and result[3] == generation
                    valid = valid and np.array_equal(readback, expected) and np.array_equal(immediate, expected[points])
                    state = torch.tensor([valid], dtype=torch.int32, device="npu")
                    dist.all_reduce(state, op=dist.ReduceOp.MIN)
                    maximum_time = torch.tensor([elapsed], dtype=torch.float32, device="npu")
                    dist.all_reduce(maximum_time, op=dist.ReduceOp.MAX)
                    record = {"mode": mode, "bytes": length, "sample": sample,
                              "warmup": sample < args.warmup, "all_ranks_pass": bool(state.item()),
                              "event_ms": elapsed, "rank_max_event_ms": float(maximum_time.item()),
                              "issue_us": result[1] * 1000 / ticks_per_ms,
                              "ready_us": result[2] * 1000 / ticks_per_ms, "status": result[0]}
                    report["records"].append(record)
                    if not record["all_ranks_pass"]:
                        raise RuntimeError(f"Data/notification ordering failed: {record}")
        report["status"] = "pass"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        if opened:
            probe.check(lib.probe_close())
        report["shmem_initialized_after_close"] = bool(lib.probe_initialized())
        (args.output / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"rank": rank, "status": report["status"],
                          "cases": len(report["records"])}), flush=True)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
