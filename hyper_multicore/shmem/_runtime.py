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
"""Lazy access to the private Torch SHMEM Runtime binding."""

from __future__ import annotations

__all__ = [
    "_load_native",
    "_torch_modules",
]

import ctypes
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Final

from hyper_multicore._loader import (
    NativeComponentUnavailableError, load_torch_npu, native_loading, native_context, native_ready,
    require_multicore_environment, validate_native_compatibility,
    select_native_shmem,
)
from hyper_multicore._native_manifest import SHMEM_DEPENDENCIES, resource_path


_NATIVE_MODULE: Final = "hyper_multicore_shmem_torch"

_native: Any | None = None
_native_libraries: list[Any] = []


def _configured_shmem_setting(name: str, default: str) -> str:
    """Read both supported names and reject conflicting Runtime settings."""
    current_name = f"HYPER_MULTICORE_SHMEM_{name}"
    legacy_name = f"HYPER_PARALLEL_SHMEM_{name}"
    current, legacy = os.environ.get(current_name), os.environ.get(legacy_name)
    if current is not None and legacy is not None and current != legacy:
        raise ValueError(f"Conflicting {current_name} and {legacy_name}")
    return current if current is not None else legacy if legacy is not None else default


def configured_data_engine() -> str:
    """Return the selected process-wide data engine."""
    return _configured_shmem_setting("DATA_ENGINE", "mte")


def configured_udma_qp_count() -> int:
    """Read the configured process-wide UDMA bank size before resource binding."""
    value = _configured_shmem_setting("UDMA_QP_NUM", "32")
    if value not in ("1", "8", "16", "32"):
        raise ValueError("HYPER_PARALLEL_SHMEM_UDMA_QP_NUM must be 1, 8, 16 or 32")
    if value != "32" and configured_data_engine() != "mte_udma":
        raise ValueError("A reduced UDMA QP bank requires mte_udma")
    return int(value)


def _load_native() -> Any:
    """Import and cache the private binding after the first successful load."""
    global _native  # pylint: disable=global-statement

    if native_ready("shmem"):
        return _native
    require_multicore_environment()
    load_torch_npu()
    selected = validate_native_compatibility()
    backend, resources = select_native_shmem(selected)
    path = resource_path(selected["root"], selected["variant"]["shmem_so"][backend])
    with native_loading("shmem") as needed:
        if needed:
            for dependency in SHMEM_DEPENDENCIES:
                library = resource_path(selected["root"], f"{resources['shmem_root']}/{dependency}")
                _native_libraries.append(ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL))
            cached = sys.modules.get(_NATIVE_MODULE)
            if cached is not None:
                if Path(getattr(cached, "__file__", "")).resolve() != path:
                    raise NativeComponentUnavailableError(f"Unexpected SHMEM module already loaded: {cached!r}")
                _native = cached
            else:
                try:
                    spec = importlib.util.spec_from_file_location(_NATIVE_MODULE, path)
                    if spec is None or spec.loader is None:
                        raise ImportError(f"No extension loader for {path}")
                    native = importlib.util.module_from_spec(spec)
                    sys.modules[_NATIVE_MODULE] = native
                    spec.loader.exec_module(native)
                    _native = native
                except BaseException as error:
                    sys.modules.pop(_NATIVE_MODULE, None)
                    raise NativeComponentUnavailableError(
                        f"[HYPER-MULTICORE-SHMEM-LOAD-FAILED] path={path} {native_context()}: {error}"
                    ) from error
    return _native


def _torch_modules() -> tuple[Any, Any]:
    """Load Torch only when the internal SHMEM boundary is used."""
    import torch  # pylint: disable=C0415
    import torch.distributed as dist  # pylint: disable=C0415

    return torch, dist
