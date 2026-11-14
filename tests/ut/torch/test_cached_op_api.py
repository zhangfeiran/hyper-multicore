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
"""Exercise cached context callbacks with native queue ordering and CPU stubs."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
          card_mark="allcards", essential_mark="essential")
class TestCachedOpApi(unittest.TestCase):
    """Require symbol reuse without changing per-invocation context ownership."""

    def test_lifecycle_callbacks_preserve_native_queue_semantics(self) -> None:
        """Feature: Cached PTA context symbols.
        Description: Compile the actual bridge and exercise both native queue paths and optional callbacks.
        Expectation: Callback order and threads remain native; only audited symbols stop being looked up repeatedly.
        """
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("g++ is required for the CPU native bridge fixture")
        source = Path(__file__).resolve().parent
        root = source.parents[2]
        with tempfile.TemporaryDirectory(prefix="hmc-cached-context-") as directory:
            work = Path(directory)
            header = work / "op_plugin/include/npu_cpp_extension.h"
            header.parent.mkdir(parents=True)
            shutil.copy2(source / "cached_op_api_stub.h", header)
            for version in (209, 210):
                with self.subTest(version=version):
                    binary = work / f"context-{version}"
                    compiled = subprocess.run(
                        [compiler, "-std=c++17", "-pthread", f"-DTEST_VERSION={version}",
                         "-I", str(work), "-I", str(root / "hyper_multicore/torch/csrc"),
                         str(source / "cached_op_api_fixture.cpp"), "-o", str(binary)],
                        capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
                    for callbacks in ("present", "absent"):
                        with self.subTest(callbacks=callbacks):
                            result = subprocess.run([str(binary), callbacks], capture_output=True,
                                                    text=True, check=False)
                            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                            self.assertIn("Verified V1/V2 lifecycle order", result.stdout)
