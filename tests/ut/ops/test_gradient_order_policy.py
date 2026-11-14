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
"""Compile the device-independent gradient order policy against a CPU oracle."""

from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest


def _shape_join(rows):
    """Independent threshold oracle for the optional device shape policy."""
    descending = sorted(rows, reverse=True)
    return (
        len(rows) > 1
        and sum(rows) >= 49152
        and (descending[1] >= 4096)
        and (descending[0] <= 8 * descending[1])
        and (max(rows) - min(rows) >= 4096)
        and (sum(rows) >= 98304 or (descending[0] >= 28672 and 2 * descending[0] >= 5 * descending[1]))
    )


class TestGradientOrderPolicy(unittest.TestCase):
    """Exercise real C++ policy with held-out histograms and stable ties."""

    def test_native_policy_matches_independent_sort_and_prefix_rules(self) -> None:
        """Every candidate retains a permutation and falls back deterministically."""
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("The gradient order policy test requires g++")
        root = Path(__file__).resolve().parents[3]
        fixture = Path(__file__).with_name("gradient_order_policy.cpp")
        rng = random.Random(20261105)
        cases = [
            [0] * 6,
            [4096] * 6,
            [29952, 3648, 8128, 5632, 6912, 7168],
            [2368, 11392, 4480, 8768, 20544, 4416],
            [65536, 63680, 80, 2736, 0, 0],
            [65536, 0, 0, 0, 0, 0],
            [24576, 24576, 0, 0, 0, 0],
            [24575, 24576, 0, 0, 0, 0],
            [34000, 4200, 4100, 4000, 3900, 3800],
            [8192] * 6,
            [4464, 8064, 32512, 16368, 7360, 2032],
            [32768, 32768, 32768, 0, 0, 0],
        ]
        cases += [[rng.randrange(0, 65537) for _ in range(n)] for n in (1, 2, 6, 8, 16, 32, 64) for _ in range(20)]
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder) / "policy"
            build = subprocess.run(
                [compiler, "-std=c++17", "-O2", "-I", str(root), str(fixture), "-o", str(binary)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            for rows in cases:
                for policy in (1, 2, 3, 4, 8):
                    process = subprocess.run(
                        [str(binary), str(policy), *[str(x) for x in rows]], capture_output=True, text=True, check=False
                    )
                    self.assertEqual(process.returncode, 0, process.stderr)
                    values = [int(x) for x in process.stdout.split()]
                    if policy in (1, 2, 3):
                        expected = sorted(range(len(rows)), key=lambda i: (rows[i] if policy != 2 else -rows[i], i))
                        join = policy == 3
                    elif policy == 4:
                        expected = list(range(len(rows)))
                        join = rows[0] * len(rows) > 2 * sum(rows) and rows[0] == max(rows)
                    else:
                        descending = sorted(rows, reverse=True)
                        join = _shape_join(rows)
                        expected = sorted(range(len(rows)), key=rows.__getitem__) if join else list(range(len(rows)))
                    self.assertEqual(values, [int(join), *expected])
