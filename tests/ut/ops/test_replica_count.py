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
"""Compare the production fixed-width count loop against its generic reference."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
class TestReplicaCount(unittest.TestCase):
    """Compile both real header paths for valid, duplicate and out-of-range rows."""

    def test_fixed_topk_preserves_counts_and_invalid_flags(self) -> None:
        """Integer width, expert bounds and duplicate semantics remain exactly equal."""
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("g++ is required for the host count harness")
        root = Path(__file__).resolve().parents[3]
        fixture = Path(__file__).with_name("replica_count_fixture.cpp")
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "counter"
            compiled = subprocess.run(
                [compiler, "-std=c++17", "-O2", "-I", str(root / "hyper_multicore/replica/_device_kernel"),
                 str(fixture), "-o", str(binary)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            run = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("Verified 14000 exact fixed-TopK histograms", run.stdout)
