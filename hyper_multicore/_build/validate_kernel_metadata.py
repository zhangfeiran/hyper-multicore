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
"""Enforce production device-debug limits in multicore kernel metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable


_ALLOWED_DEBUG_OPTIONS = frozenset(("", "assert"))
_MAX_DEBUG_BUFFER_BYTES = 76800


def _parse_args() -> argparse.Namespace:
    """Parse the metadata validation contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    return parser.parse_args()


def _debug_policy_violations(value: Any, location: str = "$") -> Iterable[str]:
    """Yield paths to values outside the production debug policy."""
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key == "debugOptions" and child not in (None, False):
                if not isinstance(child, str) or child not in _ALLOWED_DEBUG_OPTIONS:
                    yield f"{child_location}={child!r}"
            elif key == "debugBufSize" and child not in (None, 0, False):
                if (
                    not isinstance(child, int)
                    or isinstance(child, bool)
                    or child < 0
                    or child > _MAX_DEBUG_BUFFER_BYTES
                ):
                    yield f"{child_location}={child!r}"
            yield from _debug_policy_violations(child, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _debug_policy_violations(child, f"{location}[{index}]")


def validate_production_kernel_metadata(metadata_paths: Iterable[Path]) -> tuple[Path, ...]:
    """Validate that kernel metadata stays within production debug limits.

    Args:
        metadata_paths: JSON metadata files generated for production kernels.

    Returns:
        Resolved metadata paths that passed validation.

    Raises:
        ValueError: If no metadata is supplied or any file exceeds the policy.
    """
    resolved_paths = tuple(sorted(Path(path).resolve() for path in metadata_paths))
    if not resolved_paths:
        raise ValueError("No kernel metadata files were supplied.")
    violations = []
    for path in resolved_paths:
        with path.open(encoding="utf-8") as source:
            metadata = json.load(source)
        violations.extend(f"{path}:{field}" for field in _debug_policy_violations(metadata))
    if violations:
        raise ValueError(
            "Production kernel metadata violates the device debug policy: "
            + ", ".join(violations)
        )
    return resolved_paths


def main() -> int:
    """Validate every kernel metadata JSON below one root."""
    args = _parse_args()
    root = Path(args.root)
    metadata_paths = [root] if root.is_file() else sorted(root.glob("*.json"))
    validate_production_kernel_metadata(metadata_paths)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        sys.stderr.write(f"[HP-MULTICORE-METADATA-ERROR] {error}\n")
        raise SystemExit(1) from error
