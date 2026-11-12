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
"""Build an isolated RDMA-enabled pinned SDK and a C-ABI one-sided protocol probe."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def run(command: list[str], **kwargs: object) -> None:
    """Execute one reviewed build command and preserve its failing status."""
    subprocess.run(command, check=True, **kwargs)


def main() -> None:
    """Build outside the normal payload so the delivered runtime stays unchanged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shmem-source", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    source_files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path(__file__).parent.iterdir()
                    if p.name == "CMakeLists.txt" or p.suffix in (".cpp", ".py")}
    build = args.build_root.resolve()
    build.mkdir(parents=True, exist_ok=True)
    source = build / "sdk-source"
    if source.exists():
        raise FileExistsError("Use a fresh --build-root to preserve source and build provenance")
    environment = os.environ.copy()
    environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    verification = subprocess.check_output([
        sys.executable, str(root / "hyper_multicore/_build/prepare_dependencies.py"),
        "--dependency", "shmem", "--verify-only", "--source-dir", str(args.shmem_source.resolve()),
    ], env=environment, text=True)
    locked = json.loads(verification)["shmem"]
    lock = json.loads((root / "hyper_multicore/_build/dependencies.lock.json").read_text())
    patches = lock["components"]["multicore"]["shmem"].get("patches", [])
    archive = build / "sdk-source.tar"
    run(["git", "-C", str(args.shmem_source.resolve()), "archive", "--format=tar", locked["commit"],
         "--output", str(archive)], env=environment)
    source.mkdir()
    run(["tar", "-xf", str(archive), "-C", str(source)])
    environment["GIT_CEILING_DIRECTORIES"] = str(source.parent)
    for patch in patches:
        path = root / patch["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != patch["sha256"]:
            raise ValueError(f"Locked adapter digest changed: {patch['path']}")
        run(["git", "apply", "--check", str(path)], cwd=source, env=environment)
        run(["git", "apply", str(path)], cwd=source, env=environment)
    sdk_build, sdk_install = build / "sdk-build", build / "sdk-install"
    run(["cmake", "-S", str(root / "hyper_multicore/shmem/cmake/shmem_wrapper"), "-B", str(sdk_build),
         f"-DHP_SHMEM_SOURCE_DIR={source}", "-DCMAKE_BUILD_TYPE=Release", "-DUSE_CXX11_ABI=ON",
         "-DACLSHMEM_RDMA_SUPPORT=ON", "-DUSE_EXAMPLES=OFF", "-DUSE_UNIT_TEST=OFF", "-DBUILD_PYTHON=OFF",
         "-DCMAKE_DISABLE_FIND_PACKAGE_MPI=TRUE", "-DSOC_TYPE=Ascend910B"], env=environment)
    run(["cmake", "--build", str(sdk_build), "--parallel", str(args.jobs)], env=environment)
    run(["cmake", "--install", str(sdk_build), "--prefix", str(sdk_install / "shmem")], env=environment)
    probe_build, payload = build / "probe-build", build / "payload"
    run(["cmake", "-S", str(Path(__file__).parent), "-B", str(probe_build),
         f"-DSHMEM_SDK_ROOT={sdk_install / 'shmem'}", "-DCMAKE_BUILD_TYPE=Release"], env=environment)
    run(["cmake", "--build", str(probe_build), "--parallel", str(args.jobs)], env=environment)
    run(["cmake", "--install", str(probe_build), "--prefix", str(payload)], env=environment)
    for path in (sdk_install / "shmem/lib").glob("*.so"):
        shutil.copy2(path, payload / "lib" / path.name)
    current_files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path(__file__).parent.iterdir()
                     if p.name == "CMakeLists.txt" or p.suffix in (".cpp", ".py")}
    if current_files != source_files:
        raise RuntimeError("Probe source changed during compilation; refusing mixed-source publication")
    provenance = {"scope": "one-sided transport protocol, no MegaMoE or performance acceptance",
                  "sdk": locked, "patches": patches, "rdma_support": True,
                  "sources": source_files,
                  "native_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (payload / "lib").glob("*.so")}}
    (payload / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"ONE_SIDED_PROBE_PAYLOAD={payload}", flush=True)


if __name__ == "__main__":
    main()
