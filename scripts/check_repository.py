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
"""Check standalone import, path and locked-patch boundaries without importing Torch."""

import ast
import hashlib
import json
from pathlib import Path
import re
import runpy

ROOT = Path(__file__).absolute().parents[1]
_LEGACY_ENVIRONMENT = {
    "HYPER_PARALLEL_MX_B0_BACKWARD_DAG", "HYPER_PARALLEL_MX_GRADIENT_UDMA",
    "HYPER_PARALLEL_REPLICA_CALIBRATION_DIR", "HYPER_PARALLEL_REPLICA_COMPLETE_GUEST_SEARCH",
    "HYPER_PARALLEL_REPLICA_COST_MODEL", "HYPER_PARALLEL_REPLICA_HOST_NATIVE",
    "HYPER_PARALLEL_REPLICA_METADATA_REUSE", "HYPER_PARALLEL_REPLICA_PROJECTED_GUEST_SEARCH",
    "HYPER_PARALLEL_REPLICA_WEIGHT_RESIDENCY", "HYPER_PARALLEL_SHMEM_",
    "HYPER_PARALLEL_SHMEM_BOOTSTRAP_ENDPOINT", "HYPER_PARALLEL_SHMEM_DATA_ENGINE",
    "HYPER_PARALLEL_SHMEM_FIXED_HEAP_REPACK", "HYPER_PARALLEL_SHMEM_HEAP_SIZE",
    "HYPER_PARALLEL_SHMEM_TIMEOUT_SEC", "HYPER_PARALLEL_SHMEM_UDMA_QP_NUM",
}


def main() -> None:
    """Reject old executable imports/paths and changed dependency patches; print counts."""
    failures = []
    checked = 0
    files = [path for directory in ("hyper_multicore", "tests", "scripts", "examples")
             for path in (ROOT / directory).rglob("*") if path.is_file()]
    files.extend(ROOT / name for name in ("setup.py", "build.sh", "MANIFEST.in", "requirements.txt"))
    for path in files:
        if path.suffix == ".py":
            checked += 1
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                imports = ([node.module or ""] if isinstance(node, ast.ImportFrom)
                           else [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                for module in imports:
                    if module == "hyper_parallel" or module.startswith("hyper_parallel."):
                        failures.append(f"Forbidden downstream import: {path}:{node.lineno}: {module}")
                    if module.startswith(("tests.torch.multicore", "tests.ut.core.multicore")):
                        failures.append(f"Stale test path: {path}:{node.lineno}: {module}")
                    if module.startswith(("hyper_multicore.", "tests.")):
                        candidate = ROOT.joinpath(*module.split("."))
                        if not candidate.is_dir() and not candidate.with_suffix(".py").is_file():
                            failures.append(f"Missing local import: {path}:{node.lineno}: {module}")
            if path.relative_to(ROOT).parts[:2] == ("tests", "torch") and path.name.startswith("test_"):
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [x.name for x in node.names]
                        if any(name.split(".")[0] in {"torch", "torch_npu", "hyper_multicore"} for name in names):
                            failures.append(f"Heavy import in launcher: {path}:{node.lineno}")
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                    name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                    value = node.args[0].value
                    if name in {"import_module", "__import__", "find_spec"} and isinstance(value, str):
                        if value == "hyper_parallel" or value.startswith("hyper_parallel."):
                            failures.append(f"Dynamic downstream lookup: {path}:{node.lineno}")
        if path.suffix in {".py", ".sh", ".bash", ".cmake", ".json", ".cpp", ".h", ".hpp"} or path.name in {
            "CMakeLists.txt", "MANIFEST.in", "requirements.txt"
        }:
            source = path.read_text(encoding="utf-8")
            old_name = "hyper" + "_parallel"
            pattern = rf"{old_name}[/\\]|{old_name}::|torch\.ops\.{old_name}"
            if re.search(pattern, source):
                failures.append(f"Old executable identity/path: {path}")
            legacy_names = set(re.findall(rf"{old_name.upper()}_[A-Z0-9_]+", source))
            unsupported = legacy_names - _LEGACY_ENVIRONMENT
            if unsupported:
                failures.append(f"Unsupported legacy identifiers in {path}: {sorted(unsupported)}")
    dependency_helpers = runpy.run_path(str(ROOT / "hyper_multicore/_build/prepare_dependencies.py"))
    dependencies = dependency_helpers["load_dependency_locks"]()
    patches = 0
    for dependency in dependencies.values():
        for revision in dependency_helpers["dependency_revisions"](dependency).values():
            for key in ("patches", "mxfp8_patches"):
                for patch in revision.get(key, []):
                    patches += 1
                    path = ROOT / patch["path"]
                    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != patch["sha256"]:
                        failures.append(f"Locked patch missing or changed: {path}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(json.dumps({"python_files_parsed": checked, "locked_patches_verified": patches,
                      "downstream_imports": 0, "stale_paths": 0}, indent=2))


if __name__ == "__main__":
    main()
