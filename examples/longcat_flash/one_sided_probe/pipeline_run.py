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
"""Exercise eight QP owners and per-expert credits from an unexpanded LongCat activation source."""

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

from run import Probe, tag


def expected_payload(rank: int, generation: int, rows: int) -> np.ndarray:
    """Reconstruct expert-major payloads from original token row indices."""
    if not rows:
        return np.full(8, 0xA5A5A5A5, dtype=np.uint32)
    opposite = 8 if rank < 8 else 0
    result = np.empty((8, 8, rows, 3072), dtype=np.uint32)
    columns = np.arange(3072, dtype=np.uint32)
    positions = np.arange(rows, dtype=np.uint32)
    for expert in range(8):
        for source in range(8):
            source_rank = opposite + source
            selected = (((rank % 8 * 8 + expert) * rows + positions) * 17 + source_rank * 13) % 2048
            words = selected[:, None] * np.uint32(3072) + columns[None, :]
            result[expert, source] = words ^ tag(source_rank, generation)
    return result


def main() -> None:
    """Record correctness and the expert-ready window while communication is still in flight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.samples <= 0 or args.warmup < 0:
        raise ValueError("--samples must be positive and --warmup must be nonnegative")
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 16:
        raise ValueError("The original-width pipeline probe requires exactly 16 ranks")
    probe = Probe(args.library)
    lib = probe.library
    lib.probe_pipeline_configure.restype, lib.probe_pipeline_configure.argtypes = ctypes.c_int, []
    lib.probe_pipeline_prepare.restype, lib.probe_pipeline_prepare.argtypes = (
        ctypes.c_int, [ctypes.c_void_p, ctypes.c_int32])
    lib.probe_pipeline_step.restype, lib.probe_pipeline_step.argtypes = (
        ctypes.c_int, [ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint64,
                      ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint64)])
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"rank": rank, "world_size": world, "source_shape_bf16": [2048, 6144],
              "local_real_experts": 8, "remote_peers": 8, "source_logical_bytes": 24 * 1024 * 1024,
              "receive_logical_bytes": 96 * 1024 * 1024, "symmetric_heap_bytes": 256 * 1024 * 1024,
              "scope": "indexed one-sided transport and expert-ready observation; no GMM or end-to-end acceptance",
              "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(), "records": [],
              "device": torch.npu.get_device_name(), "torch": torch.__version__, "torch_npu": torch_npu.__version__}
    uid_size = lib.probe_unique_id_size()
    uid = ctypes.create_string_buffer(uid_size)
    if rank == 0:
        probe.check(lib.probe_unique_id(uid))
    identifiers = [uid.raw if rank == 0 else None]
    dist.broadcast_object_list(identifiers, src=0)
    uid = ctypes.create_string_buffer(identifiers[0], uid_size)
    opened = False
    try:
        status = lib.probe_open(rank, world, uid, 256 * 1024 * 1024, 24 * 1024 * 1024, 1)
        opened = bool(lib.probe_initialized())
        failures = [None] * world
        dist.all_gather_object(failures, (status, lib.probe_last_error().decode(errors="replace")))
        if any(value[0] for value in failures):
            raise RuntimeError(f"RDMA setup failed: {failures}")
        status = lib.probe_pipeline_configure()
        dist.all_gather_object(failures, (status, lib.probe_last_error().decode(errors="replace")))
        if any(value[0] for value in failures):
            raise RuntimeError(f"Symmetric receive allocation failed: {failures}")
        masks = [lib.probe_peer_transport(peer) for peer in range(world)]
        report["peer_transport_masks"] = masks
        opposite = 8 if rank < 8 else 0
        reachable = all(masks[peer] & lib.probe_transport_bit(1) for peer in range(opposite, opposite + 8))
        available = torch.tensor([reachable], dtype=torch.int32, device="npu")
        dist.all_reduce(available, op=dist.ReduceOp.MIN)
        if not bool(available.item()):
            raise RuntimeError("The remote peer set is not RoCE reachable")
        calibration = [probe.step(0, (rank + 8) % 16, 0, -1, 1_000_000) for _ in range(3)]
        ticks_per_ms = statistics.median(ticks[1] / ms for ms, ticks in calibration)
        report["ticks_per_ms"] = ticks_per_ms
        generation = 0
        indices = np.arange(2048 * 3072, dtype=np.uint32)
        for rows in (0, 128):
            for sample in range(args.warmup + args.samples):
                generation += 1
                source = indices ^ tag(rank, generation)
                probe.check(lib.probe_pipeline_prepare(source.ctypes.data, generation))
                dist.barrier()
                torch.npu.synchronize()
                elapsed = ctypes.c_float()
                observations = (ctypes.c_uint64 * 512)()
                probe.check(lib.probe_pipeline_step(rows, generation, int(ticks_per_ms * 2000),
                                                    ctypes.byref(elapsed), observations))
                observed = np.asarray(list(observations), dtype=np.uint64)
                expected = expected_payload(rank, generation, rows)
                received = np.empty_like(expected)
                probe.check(lib.probe_read(received.ctypes.data, received.nbytes))
                immediate = observed[336:464].astype(np.uint32).reshape(8, 8, 2)
                points = (np.stack((expected[:, :, 0, 0], expected[:, :, -1, -1]), axis=-1)
                          if rows else np.full((8, 8, 2), 0xA5A5A5A5, dtype=np.uint32))
                credits = [int(observed[peer * 16] & np.uint64(0xFFFFFFFF)) for peer in range(8)]
                valid = int(observed[320]) == 0 and all(value == ((generation << 8) | 255) for value in credits)
                valid = valid and np.array_equal(received, expected) and np.array_equal(immediate, points)
                state = torch.tensor([valid], dtype=torch.int32, device="npu")
                dist.all_reduce(state, op=dist.ReduceOp.MIN)
                rank_max = torch.tensor([elapsed.value], dtype=torch.float32, device="npu")
                dist.all_reduce(rank_max, op=dist.ReduceOp.MAX)
                first = int(observed[322])
                last = int(observed[329])
                early_masks = [int(value) & 255 for value in observed[480:488]]
                record = {"rows_per_source_expert": rows, "sample": sample, "warmup": sample < args.warmup,
                          "all_ranks_pass": bool(state.item()), "rank_max_event_ms": float(rank_max.item()),
                          "event_ms": elapsed.value, "first_expert_ms": (first - int(observed[321])) / ticks_per_ms,
                          "last_expert_ms": (last - int(observed[321])) / ticks_per_ms,
                          "ready_window_ms": (last - first) / ticks_per_ms, "early_masks": early_masks,
                          "first_ready_before_last_credit": any(mask != 255 for mask in early_masks),
                          "observer_status": int(observed[320]), "final_credits": credits}
                report["records"].append(record)
                if not record["all_ranks_pass"]:
                    raise RuntimeError(f"Indexed payload or completion credit mismatch: {record}")
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
        print(json.dumps({"rank": rank, "status": report["status"], "cases": len(report["records"])}), flush=True)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
