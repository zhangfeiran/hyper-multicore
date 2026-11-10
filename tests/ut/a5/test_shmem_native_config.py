# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Check actual C++ SHMEM environment parsing without an accelerator."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def reader(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the production config parser as a small host-only executable."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required")
    directory = tmp_path_factory.mktemp("shmem-config")
    source = directory / "reader.cpp"
    source.write_text('''#include "runtime/config.h"
#include <iostream>
int main() {
  using namespace hyper_multicore::shmem::runtime;
  auto result = LoadConfigFromEnvironment();
  if (!result.ok()) { std::cerr << result.error().message; return 2; }
  const auto &config = result.value();
  std::cout << config.heap_size_bytes << ' ' << config.timeout_seconds << ' '
            << static_cast<int>(config.data_engine) << ' ' << config.udma_qp_num << ' '
            << config.bootstrap_endpoint_base;
}
''', encoding="utf-8")
    output = directory / "reader"
    include = ROOT / "hyper_multicore/shmem/ccsrc"
    subprocess.run([compiler, "-std=c++17", f"-I{include}", str(source), str(include / "runtime/config.cpp"),
                    "-o", str(output)], check=True, capture_output=True, text=True)
    return output


def run_reader(reader: Path, settings: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run with a controlled SHMEM environment and preserve diagnostics."""
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(("HYPER_MULTICORE_SHMEM_", "HYPER_PARALLEL_SHMEM_"))}
    environment.update(settings)
    return subprocess.run([str(reader)], env=environment, check=False, capture_output=True, text=True)


@pytest.mark.parametrize("prefix", ["HYPER_MULTICORE", "HYPER_PARALLEL"])
def test_native_parser_accepts_supported_environment_names(reader: Path, prefix: str) -> None:
    """Both names produce the same explicit heap, transport and bootstrap config."""
    result = run_reader(reader, {f"{prefix}_SHMEM_HEAP_SIZE": "4194304",
                                 f"{prefix}_SHMEM_TIMEOUT_SEC": "90",
                                 f"{prefix}_SHMEM_DATA_ENGINE": "mte_udma",
                                 f"{prefix}_SHMEM_UDMA_QP_NUM": "1",
                                 f"{prefix}_SHMEM_BOOTSTRAP_ENDPOINT": "tcp://127.0.0.1:9000"})
    assert result.returncode == 0, result.stderr
    assert result.stdout == "4194304 90 1 1 tcp://127.0.0.1:9000"


def test_native_parser_rejects_conflicting_names(reader: Path) -> None:
    """The C++ runtime must reject the same ambiguous settings as Python."""
    result = run_reader(reader, {"HYPER_MULTICORE_SHMEM_DATA_ENGINE": "mte",
                                 "HYPER_PARALLEL_SHMEM_DATA_ENGINE": "mte_udma"})
    assert result.returncode == 2
    assert "Conflicting" in result.stderr


def test_native_parser_rejects_reduced_qps_without_udma(reader: Path) -> None:
    """A reduced queue count is legal only when the UDMA engine is enabled."""
    result = run_reader(reader, {"HYPER_MULTICORE_SHMEM_UDMA_QP_NUM": "1"})
    assert result.returncode == 2
    assert "requires mte_udma" in result.stderr
