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
"""Optional exact integer primitives for the no-table replica planner."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from importlib import import_module
import os
from pathlib import Path
import sysconfig
from types import ModuleType


@lru_cache(maxsize=1)
def _native_backend() -> ModuleType | None:
    """Load a prebuilt CPython extension without initializing Torch or CANN."""
    if os.environ.get("HYPER_MULTICORE_REPLICA_HOST_NATIVE", "1") == "0":
        return None
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    path = Path(__file__).parent / "_host_kernel" / f"_replica_host{suffix}"
    if not path.is_file():
        return None
    module = import_module(f"{__package__}._host_kernel._replica_host")
    if module.abi_version() != 3:
        raise RuntimeError("Rebuild the optional replica host primitives: incompatible ABI")
    return module


def _validate_counts(rows: tuple[tuple[int, ...], ...]) -> bool:
    """Accept ordinary bounded integers natively; retain Python for other inputs."""
    module = _native_backend()
    return module is not None and module.validate_counts(rows)


def _column_totals(rows: Sequence[Sequence[int]]) -> list[int] | None:
    """Reduce ordinary columns while preserving older ABI3 and arbitrary-precision fallbacks."""
    module = _native_backend()
    if module is None or not hasattr(module, "column_totals"):
        return None
    return module.column_totals(rows)


def _locality_moves(copies: list[dict[int, int]], home: int, budget: int, nodes: tuple[int, ...]) -> tuple | None:
    """Use an additive ABI3 primitive while preserving the Python fallback and dict order."""
    module = _native_backend()
    if module is None or not hasattr(module, "locality_moves"):
        return None
    return module.locality_moves(copies, home, budget, nodes)


def _materialize_dispatch(
    counts: tuple[tuple[int, ...], ...], copies: list[dict[int, int]], home: int, budget: int, nodes: tuple[int, ...]
) -> tuple | None:
    """Use the additive integer primitive while retaining optional backend fallback."""
    module = _native_backend()
    if module is None or not hasattr(module, "materialize_dispatch"):
        return None
    return module.materialize_dispatch(counts, copies, home, budget, nodes)
