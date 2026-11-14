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
"""CMake launcher for complete CANN operator binaries; regenerate dynamic entrypoints on hits."""

import argparse
import configparser
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from _hyper_multicore_bootstrap import load_module

CACHE = load_module("component_cache")
INPUTS = load_module("component_inputs")


def run_command(command: list[str]) -> int:
    """Preserve GNU Make's advertised jobserver pipes through the extra launcher process."""
    match = re.search(r"--jobserver-(?:auth|fds)=(\d+),(\d+)", os.environ.get("MAKEFLAGS", ""))
    descriptors = []
    if match:
        for value in match.groups():
            try:
                os.fstat(int(value))
                descriptors.append(int(value))
            except OSError:
                descriptors.clear()
                break
    return subprocess.run(command, check=False, pass_fds=tuple(descriptors)).returncode


def kernel_request(context: dict, command: list[str]) -> tuple[dict, dict, Path, str]:
    """Bind the exact official command, operator closure and generated metadata."""
    options = dict(arg.split("=", 1) for arg in command[2:] if "=" in arg)
    if (Path(command[1]).name != "ascendc_compile_kernel.py"
            or options.get("--enable-binary") != "True" or options.get("--op-type") not in INPUTS.OPERATORS):
        raise ValueError("Unsupported CANN operator command")
    operation = options["--op-type"]
    source, sdk, root = (Path(context[name]) for name in ("source", "sdk", "root"))
    inputs, observations = deepcopy(context["inputs"]), dict(context["observations"])
    replacements = {str(source): "${ASSEMBLED}", str(sdk): "${SDK}", str(root): "${SOURCE}",
                    command[0]: "${PYTHON}"}
    inputs["command"] = [INPUTS.normalize(arg, replacements) for arg in command]
    INPUTS.add_tree(inputs["files"], observations, "operator", source / INPUTS.OPERATORS[operation])
    configuration = configparser.ConfigParser(interpolation=None)
    configuration.read(options["--config-ini"])
    binary_name = configuration[operation]["opFile.value"]
    if not binary_name.replace("_", "").isalnum():
        raise ValueError("Invalid CANN binary directory")
    metadata = [Path(options[name]) for name in ("--config-ini", "--json-file")]
    metadata.extend(sorted(Path(options["--auto-gen-path"]).glob("*.ini")))
    for index, path in enumerate(metadata):
        contents = INPUTS.normalize(path.read_text(), replacements)
        inputs["files"][f"generated/{index}-{path.name}"] = hashlib.sha256(contents.encode()).hexdigest()
        observations[str(path.resolve())] = CACHE.sha256(path)
    output = Path(options["--output-path"]) / options["--compute-unit"] / binary_name
    return inputs, observations, output, operation + "-" + options["--compute-unit"]


def run(context_path: Path, command: list[str]) -> int:
    """Restore only verified binaries; keep current generated Python and all packaging checks."""
    context = json.loads(context_path.read_text())
    try:
        inputs, observations, output, name = kernel_request(context, command)
    except (KeyError, ValueError, IndexError):
        return run_command(command)
    request = context_path.parent / ("component-request-" + name + ".json")
    hit = CACHE.prepare(request, name, inputs, observations, context["origin"], output,
                        clean=os.environ.get("HP_VENDOR_CACHE_CLEAN") == "1", enabled=context["enabled"])
    selected = (["--enable-binary=False" if arg == "--enable-binary=True" else arg for arg in command]
                if hit else command)
    result = run_command(selected)
    if result == 0:
        CACHE.stage(request)
    return result


def main() -> None:
    """Launch the exact command supplied by CMake, without shell interpolation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    arguments, command = parser.parse_known_args()
    if command and command[0] == "--":
        command = command[1:]
    raise SystemExit(run(arguments.context, command))


if __name__ == "__main__":
    main()
