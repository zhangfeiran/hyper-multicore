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
"""Differential checks for optional CPU planner primitives without NPU work."""

from collections import OrderedDict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from hyper_multicore.replica import host, planner
from hyper_multicore.replica import _dispatch as dispatch_ops
from hyper_multicore.replica.capacity import ExpertReplicaConfig
from tests.common.mark_utils import arg_mark


def _selected_copies(selected):
    """Derive materialization input from an independently selected Python plan."""
    return [
        {expert: rows for expert, rows in zip(slots, destination) if expert >= 0}
        for slots, destination in zip(selected.slot_to_logical, selected.destination_counts)
    ]


@unittest.skipUnless(shutil.which("cmake") and shutil.which("c++"), "CPU native tests require CMake and C++")
class TestReplicaHost(unittest.TestCase):
    """Compile the actual host source and compare it with the Python policy."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the actual CPU extension independently of installed optional binaries."""
        directory = tempfile.mkdtemp()
        cls.addClassCleanup(shutil.rmtree, directory)
        build = Path(directory)
        source = Path(host.__file__).parent / "_host_kernel"
        commands = (
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                f"-DPython3_EXECUTABLE={sys.executable}",
                "-DCMAKE_BUILD_TYPE=Release",
            ],
            ["cmake", "--build", str(build), "--parallel", "2"],
        )
        for command in commands:
            try:
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f"Host test build failed: {error.stdout}") from error
        library = next(build.glob("_replica_host*.so"))
        spec = spec_from_file_location("_replica_host", library)
        cls.native = module_from_spec(spec)
        spec.loader.exec_module(cls.native)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_column_totals_match_logical_and_physical_count_matrices(self) -> None:
        """Feature: Exact native column reductions.
        Description: Sum real logical and padded physical matrix geometries with empty source ranks.
        Expectation: Every integer total exactly matches the independent Python reduction.
        """
        generator = random.Random(20261105)
        for peers in (1, 4, 16, 64):
            for budget in (0, 1, 2, 6):
                width = peers * (6 + budget)
                matrix = tuple(
                    (
                        tuple((0 if rank % 3 == 2 else generator.randrange(2000) for _ in range(width)))
                        for rank in range(peers)
                    )
                )
                for values in (matrix, [list(row) for row in matrix]):
                    with self.subTest(peers=peers, budget=budget, mutable=isinstance(values, list)):
                        self.assertEqual(self.native.column_totals(values), list(map(sum, zip(*matrix))))

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_column_totals_keep_python_fallback_for_unsupported_values_and_overflow(self) -> None:
        """Feature: Native integer reduction boundaries.
        Description: Cross scalar, tuple, shape and signed-64-bit accumulation limits.
        Expectation: Unsupported inputs return None without replacing arbitrary-precision Python behavior.
        """

        class CountSubclass(int):
            """Keep custom scalar arithmetic on the Python reference path."""

        class RowSubclass(tuple):
            """Keep custom sequence iteration on the Python reference path."""

        class ListSubclass(list):
            """Keep custom mutable sequence behavior on the reference path."""

        cases = (
            (),
            ((),),
            ((1,), ()),
            ((True,),),
            ((-1,),),
            ((1.5,),),
            ((2**80,),),
            ((2**63 - 1,), (1,)),
            ((CountSubclass(1),),),
            (RowSubclass((1,)),),
            ListSubclass([(1,)]),
        )
        for matrix in cases:
            with self.subTest(matrix=matrix):
                self.assertIsNone(self.native.column_totals(matrix))
        for matrix in (((2**63 - 1, 0), (0, 1)), ((0, 0), (0, 0)), [(1,)], ([1, 2], (3, 4))):
            self.assertEqual(self.native.column_totals(matrix), list(map(sum, zip(*matrix))))

    @unittest.skipUnless(sys.platform == "linux", "The allocation boundary probe uses Linux address-space accounting")
    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_column_totals_reject_jagged_rows_before_native_allocation(self) -> None:
        """Feature: Shape-first native reduction.
        Description: Limit a child process after constructing a wide first row and an empty second row.
        Expectation: Invalid shape returns None before a native vector can exceed the small allocation allowance.
        """
        program = f"""
import importlib.util
import os
from pathlib import Path
import resource
spec = importlib.util.spec_from_file_location("_replica_host", {self.native.__file__!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
matrix = ((0,) * (2**20), ())
current = int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf("SC_PAGE_SIZE")
limit = current + 4 * 1024**2
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
if module.column_totals(matrix) is not None:
    raise AssertionError("A jagged matrix entered native column reduction")
print("shape rejected before allocation")
"""
        completed = subprocess.run(
            [sys.executable, "-S", "-c", program], check=True, capture_output=True, text=True, timeout=15
        )
        self.assertIn("shape rejected before allocation", completed.stdout)

    def test_count_validation_retains_python_boundary_behavior(self) -> None:
        """Feature: Host count validation.
        Description: Compare valid, invalid, generator and large-integer inputs.
        Expectation: Both backends return identical matrices or validation errors.
        """
        for values in ([[0, 3], [4, 5]], [[True]], [[-1]], [[1.5]], [[1], [2, 3]], [[2**80]]):
            outcomes = []
            for backend in (None, self.native):
                with patch.object(host, "_native_backend", return_value=backend):
                    try:
                        outcomes.append(("value", planner._counts_matrix(iter(map(iter, values)))))
                    except ValueError as error:
                        outcomes.append(("error", str(error)))
            self.assertEqual(outcomes[0], outcomes[1])

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_native_materialization_matches_all_source_quotas_and_slots(self) -> None:
        """Feature: Native integer dispatch materialization.
        Description: Replay empty sources, guest budgets, and arbitrary node labels through the real C API.
        Expectation: Every physical slot and source quota exactly matches the Python reference.
        """
        generator = random.Random(2641105)
        for size in (2, 4, 8, 16, 64):
            for budget in (0, 1, 2, 6):
                for iteration in range(8):
                    with self.subTest(size=size, budget=budget, iteration=iteration):
                        config = ExpertReplicaConfig(size * 6, size, budget)
                        counts = tuple(
                            (
                                tuple(
                                    (
                                        (
                                            0
                                            if iteration % 2 and rank % 3 == 0
                                            else generator.randrange(80) * (8 if expert < 6 else 1)
                                        )
                                        for expert in range(size * 6)
                                    )
                                )
                                for rank in range(size)
                            )
                        )
                        nodes = tuple((101 + 7 * (rank // 2) for rank in range(size))) if iteration % 3 else ()
                        with patch.object(host, "_native_backend", return_value=None):
                            selected = planner.build_expert_replica_plan(counts, budget, rank_nodes=nodes)
                            copies = _selected_copies(selected)
                            expected = dispatch_ops._materialize(counts, copies, config, rank_nodes=nodes)
                        before = [tuple(row.items()) for row in copies]
                        slots, dispatch = self.native.materialize_dispatch(counts, copies, 6, budget, nodes)
                        self.assertEqual(slots, expected.slot_to_logical)
                        self.assertEqual(dispatch, expected.dispatch_counts)
                        self.assertEqual(before, [tuple(row.items()) for row in copies])

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_native_materialization_preserves_fallback_and_capacity_errors(self) -> None:
        """Feature: Native materialization boundary.
        Description: Cross Python integer limits and mapping subclasses with exact capacity checks.
        Expectation: Unsupported inputs retain the Python result and capacity errors remain descriptive.
        """
        counts = ((12, 0, 0, 0), (0, 0, 0, 0))
        copies = [{0: 4, 1: 0}, {2: 0, 3: 0, 0: 8}]
        config = ExpertReplicaConfig(4, 2, 1)
        unsupported = (
            (((2**80, 0, 0, 0), (0, 0, 0, 0)), [{0: 2**80 - 8, 1: 0}, copies[1]]),
            (counts, [OrderedDict(copies[0]), copies[1]]),
        )
        for values, quotas in unsupported:
            with self.subTest(values=values, quotas=quotas):
                self.assertIsNone(self.native.materialize_dispatch(values, quotas, 2, 1, (0, 0)))
                outcomes = []
                for backend in (None, self.native):
                    with patch.object(host, "_native_backend", return_value=backend):
                        outcomes.append(dispatch_ops._materialize(values, quotas, config, rank_nodes=(0, 0)))
                self.assertEqual(outcomes[0], outcomes[1])
        with patch.object(host, "_native_backend", return_value=self.native):
            with self.assertRaisesRegex(ValueError, "exceeds capacity_limit"):
                dispatch_ops._materialize(counts, copies, config, 7)

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_native_materializer_is_reached_by_the_real_planner(self) -> None:
        """Feature: Native materialization reachability.
        Description: Spy on the concrete primitive used by an active guest plan.
        Expectation: The planner consumes the native dispatch result and validates its complete plan.
        """
        counts = ((600, 300, 50, 50, 20, 10, 0, 0),) * 4
        with patch.object(host, "_native_backend", return_value=self.native), patch.object(
            dispatch_ops, "_materialize_dispatch", wraps=host._materialize_dispatch
        ) as called:
            plan = planner.build_expert_replica_plan(counts, 2, rank_nodes=(0, 0, 1, 1))
        plan.validate()
        self.assertTrue(plan.transfers)
        called.assert_called_once()

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_prior_domain_abi_retains_python_materialization_without_the_additive_method(self) -> None:
        """Feature: Optional native materialization capability.
        Description: Present a domain-compatible module without the additional integer primitive.
        Expectation: Dispatch uses the Python reference while the existing domain ABI remains unchanged.
        """
        counts = ((12, 0, 0, 0), (0, 0, 0, 0))
        copies = [{0: 4, 1: 0}, {2: 0, 3: 0, 0: 8}]
        config = ExpertReplicaConfig(4, 2, 1)
        with patch.object(host, "_native_backend", return_value=None):
            expected = dispatch_ops._materialize(counts, copies, config)
        with patch.object(host, "_native_backend", return_value=object()):
            actual = dispatch_ops._materialize(counts, copies, config)
        self.assertEqual(actual, expected)
        self.assertEqual(self.native.abi_version(), 3)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Address-space validation requires Linux procfs")
    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_materializer_validates_declared_width_before_allocating(self) -> None:
        """Feature: Native input-shape admission.
        Description: Present an empty row with a declared width beyond a bounded child address space.
        Expectation: The native function returns the reference fallback without attempting the large allocation.
        """
        script = """
import importlib.util
import os
from pathlib import Path
import resource
import sys

spec = importlib.util.spec_from_file_location('_replica_host', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
virtual = int(Path('/proc/self/statm').read_text().split()[0]) * os.sysconf('SC_PAGE_SIZE')
limit = virtual + 4 * 1024**2
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
assert module.materialize_dispatch(((),), [{}], 2**23, 0, ()) is None
"""
        result = subprocess.run(
            [sys.executable, "-S", "-c", script, self.native.__file__],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, f"returncode={result.returncode} stderr={result.stderr}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0", card_mark="allcards", essential_mark="essential"
    )
    def test_native_materialization_retains_interleaved_nodes_and_fully_migrated_owner(self) -> None:
        """Feature: Interleaved physical-node placement.
        Description: Replay full owner migration and node-local residual filling across noncontiguous ranks.
        Expectation: Native slots and all source dispatch counts exactly match the Python reference.
        """
        counts = tuple(((rows,) + (0,) * 7 for rows in (1, 4, 5, 2)))
        nodes = (101, 108, 101, 108)
        config = ExpertReplicaConfig(8, 4, 1)
        for owner, first, last in ((0, 0, 12), (6, 3, 3)):
            copies = [{0: owner, 1: 0}, {2: 0, 3: 0}, {4: 0, 5: 0}, {6: 0, 7: 0, 0: last}]
            if first:
                copies[1][0] = first
            with patch.object(host, "_native_backend", return_value=None):
                expected = dispatch_ops._materialize(counts, copies, config, rank_nodes=nodes)
            slots, dispatch = self.native.materialize_dispatch(counts, copies, 2, 1, nodes)
            self.assertEqual((slots, dispatch), (expected.slot_to_logical, expected.dispatch_counts))

    def test_stale_optional_binary_cannot_bypass_the_new_domain_contract(self) -> None:
        """Feature: Native domain ABI.
        Description: Present a binary advertising the previous optional CPU ABI.
        Expectation: Loading fails explicitly instead of accepting old domain behavior.
        """

        class StaleBinary:
            """Represent the previously accepted optional module contract."""

            @staticmethod
            def abi_version() -> int:
                """Report the previous ABI without loading any hardware code."""
                return 2

        host._native_backend.cache_clear()
        self.addCleanup(host._native_backend.cache_clear)
        with patch.dict("os.environ", {"HYPER_MULTICORE_REPLICA_HOST_NATIVE": "1"}), patch.object(
            host.Path, "is_file", return_value=True
        ), patch.object(host, "import_module", return_value=StaleBinary()):
            with self.assertRaisesRegex(RuntimeError, "incompatible ABI"):
                host._native_backend()
