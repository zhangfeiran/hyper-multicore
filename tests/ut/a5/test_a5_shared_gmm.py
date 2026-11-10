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
"""CPU contracts for static A5 shared BF16 GMM geometry and dispatch."""
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from hyper_multicore.modules.mega_moe.bf16_gmm_tiling import (
    SHARED_BLAZE_TAG, retile_a5_shared_bf16,
)
from hyper_multicore.modules.mega_moe.shared import shared_tiling_bytes
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from tests.common.mark_utils import arg_mark


class TestA5SharedGmm(unittest.TestCase):
    """Preserve every static matrix extent and pointer before device qualification."""

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_shared_runtime_preserves_group_vector_and_matrix_extents(self) -> None:
        """Feature: Shared A5 tiling geometry.
        Description: Transform real forward and backward records for several static token counts.
        Expectation: Matrix extents, group list and vector records remain exact while resources fit A5.
        """
        for tokens in (512, 2048, 4096):
            spec = MegaMoeSpec(tokens, 5120, 1792, 32, 8, None, tokens * 8, 4, None, 0, 32,
                               dispatch_mode="pull", shared_intermediate_size=2048)
            for backward in (False, True):
                with self.subTest(tokens=tokens, backward=backward):
                    old = shared_tiling_bytes(spec, backward=backward)
                    new = retile_a5_shared_bf16(old, 32, backward=backward)

                    self.assertEqual(len(new), len(old))
                    self.assertEqual(new[:8], old[:8])
                    self.assertEqual(new[-80:], old[-80:])
                    for offset in range(8, len(old) - 80, 2016):
                        before = struct.unpack_from("<504I", old, offset)
                        after = struct.unpack_from("<504I", new, offset)
                        start = before.index(393216, 350) - 19
                        self.assertEqual(after[:start + 5], before[:start + 5])
                        self.assertEqual((after[start + 8], after[start + 9], after[start + 10]), (256, 256, 64))
                        self.assertEqual(after[start + 32], 1)
                        self.assertLessEqual(after[start + 8] * after[start + 9] * 4, 262144)
                    with self.assertRaises(ValueError):
                        retile_a5_shared_bf16(old[:-1], 32, backward=backward)
                    with self.assertRaises(ValueError):
                        retile_a5_shared_bf16(old, 24, backward=backward)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_actual_shared_dispatch_preserves_static_shapes_layouts_and_addresses(self) -> None:
        """Feature: Direct shared BF16 GMM dispatch.
        Description: Compile the actual helper with a mock math backend and real problem definitions.
        Expectation: All six matrix roles keep their static extents and resolved addresses; AIV returns early.
        """
        source = r'''
#include <cassert>
#include <cstdint>
#include <type_traits>
#include "bf16_gmm/problem.h"
#define HYPER_MULTICORE_BF16_GMM_BLAZE_WORKER_H_
#define __aicore__
bool on_aiv = false;
int overflow_calls = 0;
namespace AscendC::AscendCUtils {
void SetOverflow(int value) { assert(value == 1); ++overflow_calls; }
}
#define ASCEND_IS_AIV (on_aiv)
using GM_ADDR = unsigned char*;
struct TCubeTiling { int32_t M, N, Ka; };
#define GET_TILING_DATA_MEMBER(type, member, value, address) \
  const auto value = *reinterpret_cast<const TCubeTiling*>(address)
namespace asc::te { struct nd_ext_layout_ptn {}; struct dn_ext_layout_ptn {}; }
namespace MulticoreBf16 {
struct Observed { uint64_t m, n, k; GM_ADDR a, b, c; uint32_t lane, cubes; bool ta, tb; };
Observed observed{};
int calls = 0;
template<class A, class B>
void RunBlaze(const Problem& p, GM_ADDR a, GM_ADDR b, GM_ADDR c, const TCubeTiling&, uint32_t lane,
              uint32_t cubes) {
  observed = {p.m,p.n,p.k,a,b,c,lane,cubes,std::is_same_v<A,asc::te::dn_ext_layout_ptn>,
              std::is_same_v<B,asc::te::dn_ext_layout_ptn>};
  ++calls;
}
}
#include "bf16_gmm/shared_worker.h"
struct Input { uint32_t transpose_flag; };
struct Task { Input inputs[2]; };
int main() {
  assert(MulticoreBf16::SHARED_BLAZE_TAG == EXPECTED_TAG);
  struct Shape { int32_t m,n,k; bool ta,tb; };
  const Shape shapes[] = {{4096,4096,5120,false,true}, {4096,5120,2048,false,true},
      {4096,2048,5120,false,false}, {4096,5120,4096,false,false},
      {5120,4096,4096,true,false}, {2048,5120,4096,true,false}};
  unsigned char a[1], b[1], c[1];
  for (const auto& s : shapes) {
    TCubeTiling tiling{s.m,s.n,s.k};
    Task task{{{uint32_t(s.ta)},{uint32_t(s.tb)}}};
    MulticoreBf16::ExecuteSharedBlazeTask(task,a,b,c,reinterpret_cast<GM_ADDR>(&tiling),7,32);
    const auto& got = MulticoreBf16::observed;
    assert(got.m == uint64_t(s.m) && got.n == uint64_t(s.n) && got.k == uint64_t(s.k));
    assert(got.a == a && got.b == b && got.c == c && got.lane == 7 && got.cubes == 32);
    assert(got.ta == s.ta && got.tb == s.tb);
    assert(overflow_calls == MulticoreBf16::calls);
    on_aiv = true;
    const int before = MulticoreBf16::calls;
    MulticoreBf16::ExecuteSharedBlazeTask(task,a,b,c,reinterpret_cast<GM_ADDR>(&tiling),7,32);
    assert(MulticoreBf16::calls == before);
    assert(overflow_calls == before);
    on_aiv = false;
  }
}
'''
        source = source.replace("EXPECTED_TAG", str(SHARED_BLAZE_TAG))
        include = Path(__file__).resolve().parents[3] / "hyper_multicore/ops"
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "shared-gmm-dispatch"
            built = subprocess.run(["c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", f"-I{include}",
                                    "-x", "c++", "-", "-o", str(executable)],
                                   input=source, capture_output=True, text=True, check=False)
            self.assertEqual(built.returncode, 0, f"expected=0 actual={built.returncode}: {built.stderr}")
            result = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, f"expected=0 actual={result.returncode}: {result.stderr}")
