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
"""Cached device capabilities shared by execution and placement kernels."""
import configparser
import ctypes
from functools import lru_cache
import os
from pathlib import Path
import platform
import re
from typing import Any

import torch

# Public enum from CANN's acl/acl_platform.h; the capacity itself is queried.
_ACL_PLATFORM_AICORE_UB_SIZE = 1


@lru_cache(maxsize=1)
def _platform_query() -> Any:
    """Resolve the active CANN runtime without starting the Python compiler stack."""
    library = ctypes.CDLL("libascendcl.so")
    query = getattr(library, "aclplatformGetDeviceInfo", None)
    if query is None:
        if _is_cann_85():
            return None
        raise RuntimeError("The selected CANN runtime must provide the platform-query API matching its native build")
    query.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    query.restype = ctypes.c_int
    return query


def _is_cann_85() -> bool:
    """Limit legacy platform-config reading to CANN versions without the ACL API."""
    try:
        version_file = Path(os.environ["ASCEND_HOME_PATH"]) / "opp/version.info"
        return any(line.strip().startswith("Version=8.5.") for line in version_file.read_text().splitlines())
    except (OSError, KeyError):
        return False


def _legacy_device_ub_bytes(device_index: int) -> int:
    """Read the exact device's CANN platform config; zero disables the fast path."""
    name = torch.npu.get_device_name(device_index)
    if not isinstance(name, str) or not re.fullmatch(r"Ascend910(?:B[0-9A-Za-z-]*|_93[0-9A-Za-z-]*)", name):
        return 0
    root = Path(os.environ["ASCEND_HOME_PATH"])
    for directory in (root / "data/platform_config", root / f"{platform.machine()}-linux/data/platform_config"):
        filename = directory / f"{name}.ini"
        if not filename.is_file():
            continue
        try:
            config = configparser.ConfigParser(interpolation=None)
            config.read(filename, encoding="utf-8")
            size = config.getint("AICoreSpec", "ub_size")
        except (OSError, ValueError, configparser.Error):
            return 0
        return max(size, 0)
    return 0


@lru_cache(maxsize=None)
def device_ub_bytes(device_index: int) -> int:
    """Read the UB size, or zero when legacy metadata cannot safely enable the fast path."""
    query = _platform_query()
    if query is None:
        return _legacy_device_ub_bytes(device_index)
    value = ctypes.create_string_buffer(32)
    with torch.npu.device(device_index):
        status = query(_ACL_PLATFORM_AICORE_UB_SIZE, value, ctypes.sizeof(value))
    if status:
        raise RuntimeError(f"CANN UB query failed for device {device_index}: status={status}")
    try:
        size = int(value.value)
    except ValueError as error:
        raise RuntimeError(f"CANN returned an invalid UB size: {value.value!r}") from error
    if size <= 0:
        raise RuntimeError(f"CANN returned a nonpositive UB size for device {device_index}: {size}")
    return size


def device_core_limit(device: torch.device, name: str) -> int:
    """Read a positive core limit from the active NPU runtime configuration."""
    index = device.index if device.index is not None else torch.npu.current_device()
    cores = torch.npu.get_device_limit(index).get(name)
    if not isinstance(cores, int) or isinstance(cores, bool) or cores <= 0:
        raise RuntimeError(
            f"torch.npu.get_device_limit() returned an invalid {name} "
            f"for NPU {index}: {cores!r}."
        )
    return cores
