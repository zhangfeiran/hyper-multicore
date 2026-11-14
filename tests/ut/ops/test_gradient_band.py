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
"""Compile the production gradient publisher with observable device primitive stubs."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
class TestGradientBand(unittest.TestCase):
    """Observe actual header control flow; accelerator visibility remains a separate ST."""

    def test_final_store_fence_and_empty_producers(self) -> None:
        """Compile the actual header and require publication after the last retired tile."""
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("g++ is required for the host publisher harness")
        root = Path(__file__).resolve().parents[3]
        fixture = Path(__file__).with_name("gradient_band_fixture.cpp")
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "kernel_operator.h").write_text("// Device primitives are supplied by the host fixture.\n")
            binary = work / "producer"
            result = subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-I",
                    str(work),
                    "-I",
                    str(root / "hyper_multicore/ops/runtime"),
                    str(fixture),
                    "-o",
                    str(binary),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("Verified final-tile fence", run.stdout)
