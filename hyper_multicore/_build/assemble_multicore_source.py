# Copyright 2026 Huawei Technologies Co., Ltd.
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
"""Assemble the isolated Multicore operator closure from pinned sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from prepare_dependencies import dependency_revisions, load_dependency_locks, verify_source_archive


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MULTICORE_OPS = _REPO_ROOT / "hyper_multicore" / "ops"
_SHMEM_CCSRC = _REPO_ROOT / "hyper_multicore" / "shmem" / "ccsrc"
_HYPER_OPERATORS = (
    "hyper_mega_moe",
    "hyper_mega_moe_grad",
    "hyper_mega_gate",
    "hyper_mega_gate_grad",
    "hyper_mega_moe_mxfp8",
    "hyper_mega_moe_mxfp8_grad",
)


def _parse_args() -> argparse.Namespace:
    """Parse the isolated source assembly contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("arch22", "arch35"), required=True)
    parser.add_argument("--work-dir", required=True)
    return parser.parse_args()


def main() -> int:
    """Verify, export, adapt, and compose the selected kernel sources."""
    args = _parse_args()
    lock = load_dependency_locks()
    transformer_revisions = dependency_revisions(lock["ops_transformer"])
    work_dir = Path(args.work_dir).resolve()
    _validate_new_work_dir(work_dir)
    ops_nn_copy = work_dir / "adapter-inputs" / "ops-nn"
    transformer_copy = work_dir / "adapter-inputs" / "ops-transformer"
    transformer_mhc_copy = work_dir / "adapter-inputs" / "ops-transformer-mhc"
    verify_source_archive("ops_nn", lock["ops_nn"], export_dir=ops_nn_copy)
    verify_source_archive("ops_transformer.gmm", transformer_revisions["gmm"], export_dir=transformer_copy)
    verify_source_archive("ops_transformer.mhc", transformer_revisions["mhc"], export_dir=transformer_mhc_copy)
    _apply_locked_adapters(ops_nn_copy, lock["ops_nn"], architecture=args.architecture)
    _apply_locked_adapters(transformer_copy, transformer_revisions["gmm"])
    _apply_locked_adapters(transformer_mhc_copy, transformer_revisions["mhc"])

    source_root = work_dir / "source"
    _compose_hyper_multicore_ops(
        source_root,
        ops_nn_copy,
        transformer_copy,
        transformer_mhc_copy,
    )
    if args.architecture == "arch35":
        tensor_copy = work_dir / "adapter-inputs" / "ops-tensor"
        verify_source_archive("ops_tensor", lock["ops_tensor"], export_dir=tensor_copy)
        _apply_locked_adapters(tensor_copy, lock["ops_tensor"], architecture="arch35")
        _compose_arch35_sources(source_root, ops_nn_copy, tensor_copy, lock)
    _require_assembled_files(source_root)
    print(json.dumps({"source_root": str(source_root)}, sort_keys=True))
    return 0


def _compose_arch35_sources(source_root: Path, ops_nn: Path, ops_tensor: Path, lock: dict[str, Any]) -> None:
    """Share upstream arch35 mathematics without vendoring a second implementation.

    The task adapters may change logical worker identity and entry boundaries;
    mathematical kernels and relative includes are exported from locked Git trees.
    The manifest describes actual build input, including any locked adapters.
    """
    target = source_root / "arch35"
    shutil.copytree(ops_tensor / "include" / "blaze", target / "include" / "blaze")
    shutil.copytree(ops_nn / "activation/swi_glu_grad/op_kernel/arch35", target / "swi_glu_grad/arch35")
    for operator in ("dynamic_mx_quant_with_dual_axis", "swiglu_mx_quant_with_dual_axis",
                     "swiglu_backward_mx_quant_with_dual_axis"):
        destination = target / operator
        shutil.copytree(ops_nn / "quant" / operator / "op_kernel/arch35", destination / "arch35")
    # These are standard relative dependencies of the exported arch35 headers.
    for operator in ("swi_glu_grad", "dynamic_mx_quant_with_dual_axis", "swiglu_mx_quant_with_dual_axis",
                     "swiglu_backward_mx_quant_with_dual_axis"):
        shutil.copytree(ops_nn / "common/inc/op_kernel", target / operator / "inc")
    files = {str(path.relative_to(target)): _sha256(path)
             for path in sorted(target.rglob("*")) if path.is_file()}
    manifest = {"components": {name: {"commit": lock[name]["commit"],
                                     "patches": lock[name].get("patches", [])}
                               for name in ("ops_nn", "ops_tensor")},
                "files": files}
    (target / "sources.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_new_work_dir(work_dir: Path) -> None:
    """Reject broad or existing output paths."""
    if work_dir in {Path("/"), _REPO_ROOT, _REPO_ROOT.parent}:
        raise ValueError(f"Refusing unsafe multicore work directory: {work_dir}")
    if work_dir.exists():
        raise ValueError(f"Multicore work directory must not already exist: {work_dir}")
    work_dir.mkdir(parents=True)


def _apply_locked_adapters(
    source_root: Path, dependency_lock: dict[str, Any], *, architecture: str | None = None,
) -> None:
    """Verify and apply fusion adapters only to the isolated source copy."""
    for adapter in dependency_lock.get("patches", []):
        if adapter.get("architectures") and architecture not in adapter["architectures"]:
            continue
        adapter_path = (_REPO_ROOT / adapter["path"]).resolve()
        actual_hash = _sha256(adapter_path)
        if actual_hash != adapter["sha256"]:
            raise ValueError(
                f"Adapter hash mismatch for {adapter_path}: expected={adapter['sha256']}, actual={actual_hash}"
            )
        allow_zero_context = adapter.get("allow_zero_context", False)
        _run_git_apply(source_root, adapter_path, check_only=True, allow_zero_context=allow_zero_context)
        _run_git_apply(source_root, adapter_path, check_only=False, allow_zero_context=allow_zero_context)
        _run_git_apply(
            source_root,
            adapter_path,
            check_only=True,
            reverse=True,
            allow_zero_context=allow_zero_context,
        )


def _run_git_apply(
    source_root: Path,
    adapter_path: Path,
    check_only: bool,
    reverse: bool = False,
    allow_zero_context: bool = False,
) -> None:
    """Run Git's patch parser against one isolated source copy."""
    command = ["git", "apply", "--ignore-space-change"]
    if allow_zero_context:
        command.append("--unidiff-zero")
    if reverse:
        command.append("--reverse")
    if check_only:
        command.append("--check")
    command.append(str(adapter_path))
    environment = os.environ.copy()
    environment["GIT_CEILING_DIRECTORIES"] = str(source_root.parent)
    result = subprocess.run(
        command,
        cwd=source_root,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        phase = "reverse check" if reverse else "check" if check_only else "apply"
        raise ValueError(f"Adapter {phase} failed for {adapter_path}: {result.stdout.strip()}")


def _compose_hyper_multicore_ops(
    source_root: Path,
    ops_nn_copy: Path,
    transformer_copy: Path,
    transformer_mhc_copy: Path,
) -> None:
    """Compose HP operator code with selected adapted upstream kernel sources."""
    shutil.copytree(_MULTICORE_OPS / "runtime", source_root / "runtime")
    shutil.copytree(_MULTICORE_OPS / "common", source_root / "common")
    shmem_root = source_root / "shmem"
    shmem_root.mkdir(parents=True)
    (shmem_root / "cann").mkdir()
    shutil.copy2(_SHMEM_CCSRC / "cann" / "device.h", shmem_root / "cann" / "device.h")
    shutil.copy2(_SHMEM_CCSRC / "cann" / "signal.h", shmem_root / "cann" / "signal.h")
    (shmem_root / "data_plane").mkdir()
    shutil.copy2(_SHMEM_CCSRC / "data_plane" / "rma.h", shmem_root / "data_plane" / "rma.h")
    shutil.copy2(_SHMEM_CCSRC / "data_plane" / "sync.h", shmem_root / "data_plane" / "sync.h")
    shutil.copytree(
        transformer_mhc_copy / "common" / "include",
        source_root / "ops_transformer_common",
    )
    for operator_name in _HYPER_OPERATORS:
        operator_root = source_root / operator_name
        shutil.copytree(_MULTICORE_OPS / operator_name, operator_root)
        if operator_name == "hyper_mega_gate":
            shutil.copytree(_MULTICORE_OPS / "runtime", operator_root / "op_kernel" / "runtime")
            continue
        if operator_name in ("hyper_mega_moe_mxfp8", "hyper_mega_moe_mxfp8_grad"):
            # Share host ABI; MX device code is registered only for arch35.
            continue
        if operator_name == "hyper_mega_gate_grad":
            shutil.copytree(_MULTICORE_OPS / "runtime", operator_root / "op_kernel" / "runtime")
            shutil.copytree(
                ops_nn_copy / "index" / "linear_index" / "op_host" / "op_api",
                operator_root / "op_host" / "op_api" / "linear_index",
            )
            continue
        shutil.copytree(_MULTICORE_OPS / "runtime", operator_root / "op_kernel" / "runtime")
        shutil.copytree(_MULTICORE_OPS / "common", operator_root / "op_kernel" / "common")
        shutil.copytree(
            ops_nn_copy / "activation" / "swi_glu" / "op_kernel",
            operator_root / "op_kernel" / "swi_glu",
        )
        shutil.copytree(
            transformer_copy / "gmm" / "grouped_matmul" / "op_kernel",
            operator_root / "op_kernel" / "grouped_matmul",
        )
    shutil.copytree(
        ops_nn_copy / "activation" / "swi_glu_grad" / "op_kernel",
        source_root / "hyper_mega_moe_grad" / "op_kernel" / "swi_glu_grad",
    )
    mhc_root = source_root / "hyper_mega_mhc"
    shutil.copytree(_MULTICORE_OPS / "hyper_mega_mhc", mhc_root)
    shutil.copytree(_MULTICORE_OPS / "runtime", mhc_root / "op_kernel" / "runtime")
    shutil.copytree(
        ops_nn_copy / "norm" / "rms_norm" / "op_kernel",
        mhc_root / "op_kernel" / "rms_norm",
    )
    shutil.copytree(
        transformer_mhc_copy / "mhc" / "mhc_post" / "op_kernel" / "arch22",
        mhc_root / "op_kernel" / "mhc_post",
    )
    shutil.copytree(
        transformer_mhc_copy / "mhc" / "mhc_pre_sinkhorn" / "op_kernel",
        mhc_root / "op_kernel" / "mhc_pre_sinkhorn",
    )
    shutil.copytree(
        transformer_mhc_copy / "mhc" / "mhc_pre_sinkhorn" / "op_host" / "op_tiling",
        mhc_root / "op_host" / "mhc_pre_sinkhorn_tiling",
    )

    mhc_grad_root = source_root / "hyper_mega_mhc_grad"
    shutil.copytree(_MULTICORE_OPS / "hyper_mega_mhc_grad", mhc_grad_root)
    shutil.copytree(_MULTICORE_OPS / "runtime", mhc_grad_root / "op_kernel" / "runtime")
    shutil.copytree(
        ops_nn_copy / "norm" / "rms_norm_grad" / "op_kernel",
        mhc_grad_root / "op_kernel" / "rms_norm_grad",
    )
    shutil.copytree(
        transformer_mhc_copy / "mhc" / "mhc_post_backward" / "op_kernel" / "arch22",
        mhc_grad_root / "op_kernel" / "mhc_post_backward",
    )
    shutil.copytree(
        transformer_mhc_copy / "mhc" / "mhc_pre_sinkhorn_backward" / "op_kernel",
        mhc_grad_root / "op_kernel" / "mhc_pre_sinkhorn_backward",
    )
    shutil.copytree(
        transformer_mhc_copy / "mhc" / "mhc_pre_sinkhorn_backward" / "op_host" / "op_tiling",
        mhc_grad_root / "op_host" / "mhc_pre_sinkhorn_backward_tiling",
    )


def _require_assembled_files(source_root: Path) -> None:
    """Reject incomplete source closures before invoking the CANN toolchain."""
    required_paths = (
        source_root / "hyper_mega_moe" / "op_host" / "hyper_mega_moe_def.cpp",
        source_root / "hyper_mega_moe" / "op_kernel" / "hyper_mega_moe.cpp",
        source_root / "hyper_mega_moe" / "op_kernel" / "swi_glu" / "swi_glu.cpp",
        source_root / "hyper_mega_moe" / "op_kernel" / "grouped_matmul" / "grouped_matmul.cpp",
        source_root / "hyper_mega_moe_grad" / "op_host" / "hyper_mega_moe_grad_def.cpp",
        source_root / "hyper_mega_moe_grad" / "op_kernel" / "hyper_mega_moe_grad.cpp",
        source_root / "hyper_mega_moe_grad" / "op_kernel" / "swi_glu_grad" / "swi_glu_grad.cpp",
        source_root / "hyper_mega_gate" / "op_kernel" / "route_pipeline.h",
        source_root / "hyper_mega_gate" / "op_kernel" / "runtime" / "aiv_pipeline_worker.h",
        source_root / "hyper_mega_gate" / "op_host" / "hyper_mega_gate_route_def.cpp",
        source_root / "hyper_mega_gate" / "op_kernel" / "hyper_mega_gate_route.cpp",
        source_root / "hyper_mega_gate_grad" / "op_kernel" / "route_grad_pipeline.h",
        source_root / "hyper_mega_gate_grad" / "op_kernel" / "runtime" / "aiv_pipeline_worker.h",
        source_root / "hyper_mega_gate_grad" / "op_host" / "hyper_mega_gate_route_grad_def.cpp",
        source_root / "hyper_mega_gate_grad" / "op_host" / "op_api" / "linear_index" / "linear_index.cpp",
        source_root / "hyper_mega_gate_grad" / "op_kernel" / "hyper_mega_gate_route_grad.cpp",
        source_root / "common" / "mega_moe" / "transport_task.h",
        source_root / "common" / "mega_moe" / "mxfp8_worker.h",
        source_root / "common" / "mega_moe" / "op_host" / "mxfp8_op_api.inc",
        source_root / "common" / "mega_moe" / "op_host" / "mxfp8_schema.h",
        source_root / "hyper_mega_moe_mxfp8" / "op_host" / "hyper_mega_moe_mxfp8_def.cpp",
        source_root / "hyper_mega_moe_mxfp8_grad" / "op_host" / "hyper_mega_moe_mxfp8_grad_def.cpp",
        source_root / "shmem" / "data_plane" / "rma.h",
        source_root / "shmem" / "data_plane" / "sync.h",
        source_root / "hyper_mega_mhc" / "op_host" / "hyper_mega_mhc_def.cpp",
        source_root / "hyper_mega_mhc" / "op_kernel" / "hyper_mega_mhc.cpp",
        source_root / "hyper_mega_mhc" / "op_kernel" / "worker_kernel.cpp",
        source_root / "hyper_mega_mhc" / "op_kernel" / "mhc_post" / "mhc_post_arch22.h",
        source_root / "hyper_mega_mhc" / "op_kernel" / "rms_norm" / "rms_norm.h",
        source_root / "hyper_mega_mhc_grad" / "op_host" / "hyper_mega_mhc_grad_def.cpp",
        source_root / "hyper_mega_mhc_grad" / "op_kernel" / "hyper_mega_mhc_grad.cpp",
        source_root / "hyper_mega_mhc_grad" / "op_kernel" / "worker_kernel.cpp",
        source_root / "hyper_mega_mhc_grad" / "op_kernel" / "rms_norm_grad" / "rms_norm_grad_split_n_high_precision.h",
        source_root / "hyper_mega_mhc_grad" / "op_kernel" / "mhc_post_backward" / "mhc_post_backward_arch22.h",
        source_root
        / "hyper_mega_mhc_grad"
        / "op_kernel"
        / "mhc_pre_sinkhorn_backward"
        / "arch22"
        / "mhc_pre_grad_kernel.h",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise ValueError(f"Incomplete Multicore operator source closure: {missing}")


def _sha256(path: Path) -> str:
    """Return the SHA256 digest for one fusion adapter."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        sys.stderr.write(f"[HP-MULTICORE-SOURCE-ERROR] {error}\n")
        raise SystemExit(1) from error
