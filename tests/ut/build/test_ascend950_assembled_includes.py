# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.
"""Verify runtime header lookup in the layout assembled for CANN operators."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_replica_gradient_udma_resolves_adjacent_runtime_headers(tmp_path: Path) -> None:
    """An operator may include the helper without adding its parent to the search path."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ preprocessor is required")
    source = Path(__file__).resolve().parents[3] / "hyper_multicore/ops/runtime"
    runtime = tmp_path / "op_kernel/runtime"
    runtime.mkdir(parents=True)
    for name in ("replica_gradient_udma.h", "replica_gradient.h", "runtime_config.hpp"):
        shutil.copy2(source / name, runtime / name)
    sdk = tmp_path / "sdk"
    (sdk / "data_plane").mkdir(parents=True)
    for name in ("shmem.h", "kernel_operator.h", "data_plane/rma.h"):
        (sdk / name).touch()
    result = subprocess.run(
        [compiler, "-E", "-x", "c++", "-I", str(sdk), "-"],
        input=f'#include "{runtime / "replica_gradient_udma.h"}"\n',
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
