"""Protect the standalone A5 public API and CPU-only import boundary."""

import os
import subprocess
import sys


def test_mxfp8_executor_import_does_not_initialize_the_npu_backend() -> None:
    """Both experts APIs remain importable without loading the accelerator runtime."""
    code = """
import importlib.abc
import sys

class RejectNpuBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch_npu':
            raise ImportError('NPU backend is intentionally unavailable')
        return None

sys.meta_path.insert(0, RejectNpuBackend())
import hyper_multicore
assert 'MXFP8MegaMoeExperts' in hyper_multicore.__all__
from hyper_multicore import MegaMoeExperts, MXFP8MegaMoeExperts
assert MegaMoeExperts.__name__ == 'MegaMoeExperts'
assert MXFP8MegaMoeExperts.__name__ == 'MXFP8MegaMoeExperts'
assert 'torch_npu' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
