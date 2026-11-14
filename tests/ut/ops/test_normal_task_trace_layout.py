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
"""Compile the actual native wire layout and decode its bytes with Python structs."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from hyper_multicore.modules.mega_moe.normal_trace import CORE_HEADER, TASK_RECORD, TRACE_MAGIC


class TestNormalTaskTraceLayout(unittest.TestCase):
    """Protect bounded allocation arithmetic and the complete cross-language ABI."""

    def test_native_bytes_and_storage_bounds(self):
        """Actual native aggregates must match the independently specified wire values."""
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("Native wire-layout verification requires g++")
        root = Path(__file__).resolve().parents[3]
        fixture = Path(__file__).with_name("normal_task_trace_layout.cpp")
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "trace-layout"
            build = subprocess.run([compiler, "-std=c++17", "-Wall", "-Werror", "-I", str(root),
                                    str(fixture), "-o", str(binary)], capture_output=True, check=False)
            self.assertEqual(build.returncode, 0, build.stderr)
            result = subprocess.run([str(binary)], capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout), 128)
        self.assertEqual(CORE_HEADER.unpack_from(result.stdout), (TRACE_MAGIC, 10, 20, 1, 0, 1024, 2, 47, 1, 0, 0))
        self.assertEqual(TASK_RECORD.unpack_from(result.stdout, 64), (11, 12, 13, 14, 123, 142, 47, 99, 100, 0, 576, 8))

    def test_architecture_worker_roles(self):
        """C220 and C950 compilation must assign disjoint Cube and Vector trace slots."""
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("Native core-role verification requires g++")
        root = Path(__file__).resolve().parents[3]
        fixture = Path(__file__).with_name("normal_task_trace_layout.cpp")
        for architecture, role, expected in ((2201, "__DAV_C220_CUBE__", 1),
                                              (2201, "__DAV_C220_VEC__", 0),
                                              (3510, "__DAV_CUBE__", 1),
                                              (3510, "__DAV_VEC__", 0)):
            with self.subTest(architecture=architecture, role=role), tempfile.TemporaryDirectory() as directory:
                binary = Path(directory) / "trace-layout"
                result = subprocess.run([compiler, "-std=c++17", "-Wall", "-Werror", "-I", str(root),
                                         f"-D__NPU_ARCH__={architecture}", f"-D{role}=1",
                                         f"-DEXPECT_CUBE_ROLE={expected}", str(fixture), "-o", str(binary)],
                                        capture_output=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
