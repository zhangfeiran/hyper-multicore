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
"""Keep vector-only gradient primitives out of every architecture's Cube compilation."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.common.mark_utils import arg_mark


@arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential")
class TestReplicaGradient(unittest.TestCase):
    """Compile the actual helper without providing any Vector device primitives."""

    def test_cube_compilation_requires_no_vector_primitives(self) -> None:
        """Both legacy and Ascend950 Cube roles exclude remote vector loads, adds and casts."""
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("g++ is required for the host compilation fixture")
        runtime = Path(__file__).resolve().parents[3] / "hyper_multicore/ops/runtime"
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "kernel_operator.h").write_text(
                "#pragma once\n#include <cstdint>\n#define __aicore__\n#define __gm__\n"
                "using GM_ADDR = unsigned char *;\n"
                "constexpr int PIPE_ALL = 0, DATA_CACHE_LINE_SIZE = 64;\n"
                "namespace AscendC {\n"
                "template<class T> struct GlobalTensor {\n"
                "  void SetGlobalBuffer(T *, int) {}\n"
                "  void SetValue(int, T) {}\n"
                "};\n"
                "enum class CacheLine { SINGLE_CACHE_LINE };\n"
                "enum class DcciDst { CACHELINE_OUT };\n"
                "template<class T, CacheLine, DcciDst> void DataCacheCleanAndInvalid(GlobalTensor<T>) {}\n"
                "template<int> void PipeBarrier() {}\n"
                "inline uint64_t GetSystemCycle() { return 0; }\n"
                "}\n"
            )
            (work / "data_plane").mkdir()
            (work / "data_plane/rma.h").write_text("#pragma once\n")
            source = work / "cube.cpp"
            source.write_text(
                '#include "replica_gradient.h"\n'
                "int main() {\n"
                "  MulticoreRuntime::AddReplicaGradient(nullptr, nullptr, 0, 0, 0, 1);\n"
                "  MulticoreRuntime::AddReplicaGradient<true>(nullptr, nullptr, 0, 0, 0, 1, nullptr);\n"
                "}\n"
            )
            for architecture in (220, 3510):
                with self.subTest(architecture=architecture):
                    defines = ["-D__DAV_CUBE__", f"-D__NPU_ARCH__={architecture}"]
                    if architecture == 220:
                        defines.append("-D__DAV_C220_CUBE__")
                    result = subprocess.run(
                        [compiler, "-std=c++17", *defines, "-I", str(work), "-I", str(runtime),
                         str(source), "-o", str(work / "cube")],
                        check=False, capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
