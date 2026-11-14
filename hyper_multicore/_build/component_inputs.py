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
"""Bounded native dependency closures with portable source names and explicit producer facts."""

import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import time

from . import component_cache as CACHE
from .artifacts import _source_identity
from .build_utils import atomic_json
from .command_runner import message


TOOLS = ("bash", "cmake", "make", "gcc", "g++", "ld", "as", "ar", "asc_opc", "bisheng", "strip", "python3")
FLAGS = ("CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH",
         "LIBRARY_PATH", "CMAKE_PREFIX_PATH", "CMAKE_TOOLCHAIN_FILE")
OPERATORS = {"HyperMegaMoe": "hyper_mega_moe", "HyperMegaMoeGrad": "hyper_mega_moe_grad",
             "HyperMegaMhc": "hyper_mega_mhc", "HyperMegaMhcGrad": "hyper_mega_mhc_grad",
             "HyperMegaGateRoute": "hyper_mega_gate", "HyperMegaGateRouteGrad": "hyper_mega_gate_grad"}
_HASHES = {}


def _hash(path: Path) -> str:
    resolved = path.resolve(strict=True)
    status = resolved.stat()
    key = str(resolved), status.st_ino, status.st_size, status.st_mtime_ns
    if key not in _HASHES:
        _HASHES[key] = CACHE.sha256(resolved)
    return _HASHES[key]


def add_file(inputs: dict, observed: dict, label: str, path: Path) -> None:
    """Keep logical content identity separate from the real input path."""
    digest = _hash(path)
    inputs[label] = digest
    observed[str(path.resolve(strict=True))] = digest


def add_tree(inputs: dict, observed: dict, label: str, directory: Path) -> None:
    """Hash selected installed trees, following SDK aliases while rejecting cycles."""
    if not directory.is_dir():
        raise ValueError(f"Component input directory is unavailable: {directory}")
    seen = set()
    for current, directories, filenames in os.walk(directory, followlinks=True):
        path = Path(current)
        resolved = path.resolve(strict=True)
        if resolved in seen:
            directories.clear()
            continue
        seen.add(resolved)
        directories[:] = sorted(name for name in directories if name not in {"__pycache__", ".git"})
        for name in sorted(filenames):
            source = path / name
            if source.suffix not in {".pyc", ".log"} and source.is_file():
                relative = source.relative_to(directory).as_posix()
                add_file(inputs, observed, label + "/" + relative, source)


def origin(root: Path) -> dict:
    """Record the actual producer checkout without putting its absolute path in the key."""
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=False,
                            capture_output=True, text=True)
    return {"root": str(root.resolve()), "commit": result.stdout.strip() if result.returncode == 0 else None,
            "source_sha256": _source_identity(root)["source_sha256"],
            "python": str(Path(sys.executable).absolute()),
            "cann_root": str(Path(os.environ["ASCEND_HOME_PATH"]).resolve())}


def normalize(value: str, replacements: dict[str, str]) -> str:
    """Replace only declared build/source locations; external search order remains explicit."""
    for path, token in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        value = value.replace(path, token)
    return value


def host_toolchain(inputs: dict, observed: dict, environment: dict) -> None:
    """Bind compiler-discovered system headers and host C++/C runtime bytes."""
    compilers = {item["path"] for name, item in inputs["tools"].items()
                 if name in {"gcc", "g++", "CC", "CXX", "bisheng"}}
    searches = {}
    for compiler in sorted(compilers):
        result = subprocess.run([compiler, "-E", "-v", "-x", "c++", "-"], input="", capture_output=True,
                                text=True, check=True, env=environment)
        text = result.stderr
        marker = '#include <...> search starts here:'
        if marker not in text or 'End of search list.' not in text:
            raise ValueError(f"Cannot identify compiler header search: {compiler}")
        section = text.split(marker, 1)[1].split('End of search list.', 1)[0]
        paths = [line.strip() for line in section.splitlines() if line.strip()]
        searches[compiler] = paths
    inputs["host_include_searches"] = searches
    for index, directory in enumerate(sorted({value for paths in searches.values() for value in paths})):
        add_tree(inputs["files"], observed, f"host-headers/{index}", Path(directory))
    compiler = inputs["tools"]["g++"]["path"]
    for name in ("libstdc++.so", "libgcc_s.so.1", "libc.so.6", "libm.so.6", "libpthread.so.0", "libdl.so.2"):
        result = subprocess.run([compiler, "-print-file-name=" + name], capture_output=True,
                                text=True, check=True, env=environment)
        path = Path(result.stdout.strip())
        if not path.is_file():
            raise ValueError(f"Cannot identify compiler runtime: {name}")
        add_file(inputs["files"], observed, "host-runtime/" + name, path)


def _compiler_tools(environment: dict, observed: dict) -> tuple[dict, bool]:
    tools = {}
    enabled = not any(environment.get(name) for name in FLAGS if name != "CMAKE_PREFIX_PATH")
    for name in TOOLS:
        found = shutil.which(name, path=environment.get("PATH"))
        if found is None:
            raise ValueError(f"Missing component compiler tool: {name}")
        path = Path(found).resolve(strict=True)
        tools[name] = {"path": str(path), "sha256": _hash(path)}
        observed[str(path)] = tools[name]["sha256"]
    for name in ("CC", "CXX"):
        command = shlex.split(environment.get(name, ""))
        if len(command) > 1:
            enabled = False
        if command:
            found = shutil.which(command[0], path=environment.get("PATH"))
            if found is None:
                raise ValueError(f"Missing selected compiler: {command[0]}")
            path = Path(found).resolve(strict=True)
            tools[name] = {"path": str(path), "sha256": _hash(path)}
            observed[str(path)] = tools[name]["sha256"]
    return tools, enabled


def base_inputs(root: Path, sdk: Path | None = None, environment: dict | None = None) -> tuple[dict, dict, bool]:
    """Identify the selected CANN compiler, headers, libraries and supported caller environment."""
    observed, files = {}, {}
    environment = os.environ if environment is None else environment
    cann = Path(environment["ASCEND_HOME_PATH"]).resolve(strict=True)
    replacements = {str(root.resolve()): "${SOURCE}", str(Path(sys.executable).parent.parent): "${PYTHON_ENV}"}
    if sdk is not None:
        replacements[str(sdk.resolve())] = "${SDK}"
    tools, enabled = _compiler_tools(environment, observed)
    version = cann / "opp/version.info"
    if version.is_file():
        add_file(files, observed, "cann/version", version)
    else:
        files["cann/version"] = None
        enabled = False
        message("CANN version record unavailable; component reuse disabled, continuing compilation", "WARN")
    for name in ("include", "pkg_inc", "asc", "compiler", "lib64", "opp/built-in/op_api/lib"):
        if (cann / name).is_dir():
            add_tree(files, observed, "cann/" + name, cann / name)
    for index, name in enumerate(environment.get("CMAKE_PREFIX_PATH", "").split(os.pathsep)):
        if name and Path(name).is_dir():
            add_tree(files, observed, f"cmake-search/{index}", Path(name))
    for path in sorted((root / "hyper_multicore/_build").glob("*.py")):
        add_file(files, observed, "recipe/" + path.name, path)
    for name in ("build.sh", "shmem/build.sh"):
        add_file(files, observed, "recipe/" + name, root / "hyper_multicore" / name)
    add_file(files, observed, "python/executable", Path(sys.executable).resolve())
    normalized_environment = {name: normalize(environment.get(name, ""), replacements) for name in (
        "LD_LIBRARY_PATH", "PYTHONPATH", "CMAKE_BUILD_TYPE", "CMAKE_GENERATOR", *FLAGS)}
    inputs = {"schema": 2, "backend_soc_type": environment.get("HP_SHMEM_SOC_TYPE"),
              "host_arch": platform.machine(), "python_version": sys.version,
              "python_abi": sys.implementation.cache_tag, "environment": normalized_environment, "tools": tools,
              "files": files}
    if enabled:
        try:
            host_toolchain(inputs, observed, environment)
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            enabled = False
            message(f"Compiler input identity unavailable; component reuse disabled ({error})", "WARN")
    return inputs, observed, enabled


def component_inputs(root: Path, kind: str, sdk: Path | None = None,
                     identity: dict | None = None, common: Path | None = None,
                     environment: dict | None = None) -> tuple[dict, dict, bool]:
    """Select actual component inputs so kernel-only edits cannot invalidate unrelated outputs."""
    inputs, observed, enabled = base_inputs(root, sdk, environment)
    files = inputs["files"]
    component = root / "hyper_multicore"
    paths = {"sdk": ("shmem/cmake/shmem_wrapper",),
             "shmem-ops": ("shmem/ccsrc/ops", "shmem/ccsrc/cann", "shmem/ccsrc/data_plane"),
             "torch": ("torch",), "shmem": ("shmem/ccsrc",)}
    if kind not in paths:
        raise ValueError(f"Unknown native component: {kind}")
    for name in paths[kind]:
        add_tree(files, observed, "source/" + name, component / name)
    add_file(files, observed, "recipe/hardening", component / "cmake/hardening.cmake")
    if kind == "sdk":
        lock = json.loads((component / "_build/dependencies.lock.json").read_text())["components"]["multicore"]["shmem"]
        inputs["dependency"] = lock
        if inputs["backend_soc_type"] == "Ascend950":
            dependencies = json.loads((component / "_build/dependencies.lock.json").read_text())
            inputs["json_dependency"] = dependencies["components"]["multicore"]["nlohmann_json"]
        for patch in lock.get("patches", []):
            add_file(files, observed, "patch/" + patch["path"], root / patch["path"])
        add_file(files, observed, "recipe/sdk", component / "shmem/_build/shmem_sdk.sh")
    elif kind not in {"counts", "torch"}:
        if sdk is None:
            raise ValueError(f"SDK is required for {kind}")
        add_tree(files, observed, "sdk", sdk / "shmem")
    if kind in {"torch", "shmem"}:
        _framework_inputs(inputs, observed, root, identity, common, kind)
        enabled = enabled and identity["python_abi"] == f"cp{sys.version_info.major}{sys.version_info.minor}"
    return inputs, observed, enabled


def _framework_inputs(inputs, observed, root, identity, common, kind):
    if identity is None or common is None:
        raise ValueError("Adapter cache requires probed framework identity and current common payload")
    inputs["abi"] = {name: identity[name] for name in (
        "torch_version", "pta_version", "torch_commit", "torch_cxx11_abi", "python_abi", "op_plugin_version")}
    files = inputs["files"]
    for name, location in identity["framework_locations"].items():
        package = Path(location)
        add_tree(files, observed, "framework/" + name + "/include", package / "include")
        libraries = ("libc10.so", "libtorch.so", "libtorch_cpu.so") if name == "torch" else ("libtorch_npu.so",)
        for library in libraries:
            add_file(files, observed, "framework/" + name + "/" + library, package / "lib" / library)
    add_tree(files, observed, "python/include", Path(sysconfig.get_path("include")))
    add_file(files, observed, "recipe/torch-compat", root / "hyper_multicore/cmake/torch_compat.cmake")
    add_file(files, observed, "recipe/controller", root / "hyper_multicore/_build/build_controller.py")
    if kind == "torch":
        add_tree(files, observed, "vendor/api", common / "lib/vendors/hyper_multicore_nn/op_api/include")
    else:
        backend = "arch35" if inputs["backend_soc_type"] == "Ascend950" else "arch22"
        add_file(files, observed, "shmem/ops", common / "shmem/lib" / backend / "libhyper_multicore_shmem_ops.so")
        add_tree(files, observed, "source/native-tests", root / "tests/ut/shmem/cpp")


def kernel_context(root: Path, sdk: Path, source: Path) -> dict:
    """Compute shared kernel inputs once before concurrent operator compilation."""
    started = time.monotonic()
    inputs, observed, enabled = base_inputs(root, sdk)
    add_tree(inputs["files"], observed, "sdk", sdk / "shmem")
    add_tree(inputs["files"], observed, "recipe/vendor", root / "hyper_multicore/cmake/vendor")
    for directory in source.iterdir():
        if directory.is_dir() and (directory / "op_host").is_dir():
            add_tree(inputs["files"], observed, "host/" + directory.name, directory / "op_host")
    for name in ("shmem", "ops_transformer_common"):
        add_tree(inputs["files"], observed, "shared/" + name, source / name)
    return {"inputs": inputs, "observations": observed, "enabled": enabled, "origin": origin(root),
            "root": str(root), "sdk": str(sdk), "source": str(source),
            "input_scan_seconds": time.monotonic() - started}


def main() -> None:
    """Serve the shell handoff without importing Torch or initializing devices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "stage", "kernel-context"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--sdk", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--kind")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    if args.operation == "stage":
        CACHE.stage(args.request)
    elif args.operation == "kernel-context":
        atomic_json(args.output, kernel_context(args.root, args.sdk, args.source))
    else:
        started = time.monotonic()
        inputs, observed, enabled = component_inputs(args.root, args.kind, args.sdk)
        scanned = time.monotonic() - started
        CACHE.prepare(args.request, args.kind, inputs, observed, origin(args.root), args.output,
                      clean=os.environ.get("HP_VENDOR_CACHE_CLEAN") == "1", enabled=enabled)
        receipt = json.loads(args.request.read_text())
        receipt["input_scan_seconds"] = scanned
        atomic_json(args.request, receipt)


if __name__ == "__main__":
    main()
