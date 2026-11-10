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

from dataclasses import asdict, fields
from collections import OrderedDict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hyper_multicore.replica import cost, host, planner
from hyper_multicore.replica import routing
from hyper_multicore.replica.capacity import ExpertReplicaConfig
from hyper_multicore.replica.critical_path import DeviceCriticalPathCalibration
from hyper_multicore.replica.plan import ExpertExecutionPlan
from tests.common.mark_utils import arg_mark


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
        commands = (["cmake", "-S", str(source), "-B", str(build),
                     f"-DPython3_EXECUTABLE={sys.executable}", "-DCMAKE_BUILD_TYPE=Release"],
                    ["cmake", "--build", str(build), "--parallel", "2"])
        for command in commands:
            try:
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f"Host test build failed: {error.stdout}") from error
        library = next(build.glob("_replica_host*.so"))
        spec = spec_from_file_location("_replica_host", library)
        cls.native = module_from_spec(spec)
        spec.loader.exec_module(cls.native)

    @staticmethod
    def _model(size: int = 4, **options: object) -> cost.ExpertReplicaCostModel:
        values = {"calibration_version": 2, "backend": "push", "hidden_size": 16, "intermediate_size": 8,
                  "ep_size": size, "transport": "p2p",
                  "forward_ms": ((0, 0), (13, .031), (127, .317), (4096, 1.77)),
                  "backward_ms": ((0, 0), (7, .037), (137, .779), (4096, 4.31)),
                  "weight_ms": .007, "gradient_ms": .011, "token_ms_per_row": .0001,
                  "forward_repeats": 2, "backward_weight_ms": 0}
        return cost.ExpertReplicaCostModel(**(values | options))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_column_totals_match_logical_and_physical_count_matrices(self) -> None:
        """Feature: Exact native column reductions.
        Description: Sum real logical and padded physical matrix geometries with empty source ranks.
        Expectation: Every integer total exactly matches the independent Python reduction.
        """
        generator = random.Random(20261105)
        for peers in (1, 4, 16, 64):
            for budget in (0, 1, 2, 6):
                width = peers * (6 + budget)
                matrix = tuple(tuple(0 if rank % 3 == 2 else generator.randrange(2000)
                                     for _ in range(width)) for rank in range(peers))
                for values in (matrix, [list(row) for row in matrix]):
                    with self.subTest(peers=peers, budget=budget, mutable=isinstance(values, list)):
                        self.assertEqual(self.native.column_totals(values), list(map(sum, zip(*matrix))))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
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

        cases = ((), ((),), ((1,), ()), ((True,),), ((-1,),), ((1.5,),), ((2**80,),),
                 ((2**63-1,), (1,)), ((CountSubclass(1),),), (RowSubclass((1,)),), ListSubclass([(1,)]))
        for matrix in cases:
            with self.subTest(matrix=matrix):
                self.assertIsNone(self.native.column_totals(matrix))
        for matrix in (((2**63-1, 0), (0, 1)), ((0, 0), (0, 0)), [(1,)], ([1, 2], (3, 4))):
            self.assertEqual(self.native.column_totals(matrix), list(map(sum, zip(*matrix))))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_column_totals_reach_native_agreement_prepared_candidate_admission(self) -> None:
        """Feature: Native agreement CPU reduction.
        Description: Build a real prepared candidate from the native agreement's ordinary host lists.
        Expectation: The C API consumes that list without another gather or adaptive search and preserves the plan.
        """
        config = ExpertReplicaConfig(4, 2, 1)
        ids = torch.tensor([[0, 1]] * 4, dtype=torch.int32)
        counts = torch.tensor([[4, 4, 0, 0], [2, 2, 0, 0]], dtype=torch.int64)
        summary = [[0, 4, 31, 8, 0, 4, 12], [0, 2, 31, 4, 0, 2, 0]]
        prepared = routing._GatheredReplicaCounts(counts, summary, counts.tolist())
        observations = []
        reduce_columns = self.native.column_totals

        def observe(values: object) -> object:
            """Record actual list consumption while retaining the compiled reduction."""
            result = reduce_columns(values)
            observations.append((isinstance(values, list), result is not None))
            return result

        plans = []
        with patch.object(routing, "dist") as distributed:
            distributed.is_initialized.return_value = True
            distributed.get_rank.return_value = 0
            distributed.get_world_size.return_value = 2
            for backend in (None, self.native):
                with patch.object(host, "_native_backend", return_value=backend), \
                        patch.object(self.native, "column_totals", side_effect=observe), \
                        patch.object(routing, "_gather_replica_counts", side_effect=AssertionError("Extra gather")), \
                        patch.object(planner, "_adaptive_copies", side_effect=AssertionError("Duplicate search")):
                    route = routing.prepare_replica_route(
                        ids, config, dynamic_header=(0, 4, 31), target_load=6, receive_capacity=12,
                        compact_home=True, minimum_load_reduction=.4, _gathered_counts=prepared)
                    route.plan.validate()
                    plans.append(route.plan)
        self.assertEqual(plans[0], plans[1])
        self.assertIn((True, True), observations)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_column_totals_reach_owner_and_destination_summaries_without_changing_types(self) -> None:
        """Feature: Real planner summary integration.
        Description: Run owner and destination summaries through the actual compiled primitive and fallback.
        Expectation: Both call sites consume native results and preserve list, tuple and overflow behavior.
        """
        matrix = ((3, 1, 0, 0, 5, 0, 0, 0),) * 4
        config = ExpertReplicaConfig(8, 4, 0)
        slots = tuple(tuple(range(rank*2, (rank+1)*2)) for rank in range(4))
        with patch.object(host, "_native_backend", return_value=None):
            expected_home = planner._home_layout(matrix, config)
            expected_destination = ExpertExecutionPlan(config, matrix, slots, matrix).destination_counts
        with patch.object(host, "_native_backend", return_value=self.native), \
                patch.object(self.native, "column_totals", wraps=self.native.column_totals) as actual:
            home = planner._home_layout(matrix, config)
            destination = ExpertExecutionPlan(config, matrix, slots, matrix).destination_counts
        self.assertEqual(actual.call_count, 2)
        self.assertEqual(home, expected_home)
        self.assertIs(type(home[0]), list)
        self.assertEqual(destination, expected_destination)
        self.assertEqual(tuple(map(type, destination)), (tuple,) * len(destination))

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_column_totals_callers_keep_big_integer_and_older_abi3_results(self) -> None:
        """Feature: End-to-end column reduction fallback.
        Description: Run both consumers with overflowing column sums and an older ABI3 capability set.
        Expectation: Arbitrary-precision values and original container types match the Python reference.
        """
        older = SimpleNamespace(abi_version=lambda: 3, validate_counts=self.native.validate_counts,
                                materialize_dispatch=self.native.materialize_dispatch)
        for matrix in (((3, 1), (4, 0)), ((2**63-1, 0), (1, 0))):
            config = ExpertReplicaConfig(2, 2, 0)
            slots = ((0,), (1,))
            with patch.object(host, "_native_backend", return_value=None):
                expected_home = planner._home_layout(matrix, config)
                expected_destination = ExpertExecutionPlan(config, matrix, slots, matrix).destination_counts
            for backend in (self.native, older):
                with self.subTest(matrix=matrix, older=backend is older), \
                        patch.object(host, "_native_backend", return_value=backend):
                    actual_home = planner._home_layout(matrix, config)
                    actual_destination = ExpertExecutionPlan(config, matrix, slots, matrix).destination_counts
                    self.assertEqual(actual_home, expected_home)
                    self.assertIs(type(actual_home[0]), list)
                    self.assertEqual(actual_destination, expected_destination)
                    self.assertEqual(tuple(map(type, actual_destination)), (tuple,) * len(actual_destination))

    @unittest.skipUnless(sys.platform == "linux", "The allocation boundary probe uses Linux address-space accounting")
    @arg_mark(plat_marks=["cpu_linux"], level_mark="level0", card_mark="allcards", essential_mark="essential")
    def test_column_totals_reject_jagged_rows_before_native_allocation(self) -> None:
        """Feature: Shape-first native reduction.
        Description: Limit a child process after constructing a wide first row and an empty second row.
        Expectation: Invalid shape returns None before a native vector can exceed the small allocation allowance.
        """
        program = f'''
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
'''
        completed = subprocess.run([sys.executable, "-S", "-c", program], check=True,
                                   capture_output=True, text=True, timeout=15)
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

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
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
                        counts = tuple(tuple(0 if iteration % 2 and rank % 3 == 0 else
                                             generator.randrange(80) * (8 if expert < 6 else 1)
                                             for expert in range(size * 6)) for rank in range(size))
                        nodes = tuple(101 + 7 * (rank // 2) for rank in range(size)) if iteration % 3 else ()
                        with patch.object(host, "_native_backend", return_value=None):
                            selected = planner._build_expert_replica_plan(
                                counts, budget, adaptive=bool(budget), _rank_nodes=nodes)
                            copies = [{expert: rows for expert, rows in zip(slots, destination) if expert >= 0}
                                      for slots, destination in zip(selected.slot_to_logical,
                                                                    selected.destination_counts)]
                            expected = planner._materialize(counts, copies, config, rank_nodes=nodes)
                        before = [tuple(row.items()) for row in copies]
                        slots, dispatch = self.native.materialize_dispatch(counts, copies, 6, budget, nodes)
                        self.assertEqual(slots, expected.slot_to_logical)
                        self.assertEqual(dispatch, expected.dispatch_counts)
                        self.assertEqual(before, [tuple(row.items()) for row in copies])

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
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
                        outcomes.append(planner._materialize(values, quotas, config, rank_nodes=(0, 0)))
                self.assertEqual(outcomes[0], outcomes[1])
        with patch.object(host, "_native_backend", return_value=self.native):
            with self.assertRaisesRegex(ValueError, "exceeds capacity_limit"):
                planner._materialize(counts, copies, config, 7)

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_native_materializer_is_reached_by_the_real_planner(self) -> None:
        """Feature: Native materialization reachability.
        Description: Spy on the concrete primitive used by an active guest plan.
        Expectation: The planner consumes the native dispatch result and validates its complete plan.
        """
        counts = ((600, 300, 50, 50, 20, 10, 0, 0),) * 4
        with patch.object(host, "_native_backend", return_value=self.native), patch.object(
                planner, "_materialize_dispatch", wraps=host._materialize_dispatch) as called:
            plan = planner._build_expert_replica_plan(counts, 2, adaptive=True, _rank_nodes=(0, 0, 1, 1))
        plan.validate()
        self.assertTrue(plan.transfers)
        called.assert_called_once()

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_prior_domain_abi_retains_python_materialization_without_the_additive_method(self) -> None:
        """Feature: Optional native materialization capability.
        Description: Present a domain-compatible module without the additional integer primitive.
        Expectation: Dispatch uses the Python reference while the existing domain ABI remains unchanged.
        """
        counts = ((12, 0, 0, 0), (0, 0, 0, 0))
        copies = [{0: 4, 1: 0}, {2: 0, 3: 0, 0: 8}]
        config = ExpertReplicaConfig(4, 2, 1)
        with patch.object(host, "_native_backend", return_value=None):
            expected = planner._materialize(counts, copies, config)
        with patch.object(host, "_native_backend", return_value=object()):
            actual = planner._materialize(counts, copies, config)
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
        result = subprocess.run([sys.executable, "-S", "-c", script, self.native.__file__],
                                check=False, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, f"returncode={result.returncode} stderr={result.stderr}")

    @arg_mark(plat_marks=["cpu_linux", "cpu_macos"], level_mark="level0",
              card_mark="allcards", essential_mark="essential")
    def test_native_materialization_retains_interleaved_nodes_and_fully_migrated_owner(self) -> None:
        """Feature: Interleaved physical-node placement.
        Description: Replay full owner migration and node-local residual filling across noncontiguous ranks.
        Expectation: Native slots and all source dispatch counts exactly match the Python reference.
        """
        counts = tuple((rows,) + (0,) * 7 for rows in (1, 4, 5, 2))
        nodes = (101, 108, 101, 108)
        config = ExpertReplicaConfig(8, 4, 1)
        for owner, first, last in ((0, 0, 12), (6, 3, 3)):
            copies = [{0: owner, 1: 0}, {2: 0, 3: 0}, {4: 0, 5: 0}, {6: 0, 7: 0, 0: last}]
            if first:
                copies[1][0] = first
            with patch.object(host, "_native_backend", return_value=None):
                expected = planner._materialize(counts, copies, config, rank_nodes=nodes)
            slots, dispatch = self.native.materialize_dispatch(counts, copies, 2, 1, nodes)
            self.assertEqual((slots, dispatch), (expected.slot_to_logical, expected.dispatch_counts))

    def test_ordered_costs_match_double_bits_and_cache(self) -> None:
        """Feature: Ordered calibrated costs.
        Description: Compare 120 layouts with interpolation and shuffled expert order.
        Expectation: Every score bit and invocation-local duration cache matches.
        """
        generator = random.Random(9021)
        config = ExpertReplicaConfig(24, 4, 6)
        for _ in range(120):
            counts = tuple(tuple(generator.randrange(600) for _ in range(24)) for _ in range(4))
            copies = []
            for rank in range(4):
                experts = list(range(rank * 6, rank * 6 + 6))
                experts += generator.sample([e for e in range(24) if e // 6 != rank], 4)
                generator.shuffle(experts)
                copies.append({expert: generator.choice((0, 7, 13, 127, 137, generator.randrange(1500)))
                               for expert in experts})
            model = self._model()
            states = []
            for backend in (None, self.native):
                with patch.object(host, "_native_backend", return_value=backend):
                    states.append(cost._QuotaCostState(model, copies, counts, config))
            self.assertEqual([[v.hex() for v in pair] for pair in states[0].costs],
                             [[v.hex() for v in pair] for pair in states[1].costs])
            self.assertEqual(states[0].durations, states[1].durations)
            self.assertEqual(states[0].score().hex(), states[1].score().hex())

    def test_complete_plans_retain_calibration_capacity_and_topology(self) -> None:
        """Feature: Complete native plans.
        Description: Replay random layouts with admission, capacity and topology constraints.
        Expectation: Slots, dispatch counts, scores and capacity rejections match.
        """
        generator = random.Random(2811)
        for _ in range(100):
            size, home_count = generator.choice((2, 3, 4)), generator.choice((1, 3, 6))
            budget = generator.choice((0, 1, 2, 6))
            counts = [[generator.randrange(50) for _ in range(size * home_count)] for _ in range(size)]
            model = self._model(size, _rank_nodes=tuple(rank // 2 for rank in range(size)))
            total = sum(map(sum, counts))
            limit = generator.choice(((total + size - 1) // size, total))
            outcomes = []
            for backend in (None, self.native):
                with patch.object(host, "_native_backend", return_value=backend):
                    try:
                        plan = planner._build_expert_replica_plan(
                            counts, budget, adaptive=bool(budget), capacity_limit=limit,
                            cost_model=model, _rank_nodes=model._rank_nodes, minimum_replica_rows=4,
                        )
                        plan.validate()
                        outcomes.append(("plan", plan.slot_to_logical, plan.dispatch_counts,
                                         model.estimate_ms(plan).hex()))
                    except ValueError as error:
                        outcomes.append(("error", str(error)))
            self.assertEqual(outcomes[0], outcomes[1])

    def test_out_of_range_cost_is_rejected(self) -> None:
        """Feature: Calibration bounds.
        Description: Request a row count beyond the final calibrated knot.
        Expectation: A descriptive ValueError rejects extrapolation.
        """
        with self.assertRaisesRegex(ValueError, "exceed"):
            self.native.rank_compute({0: 4097}, None, 1, 0, {}, ((0, 0.), (4096, 1.)), ((0, 0.), (4096, 2.)))

    def test_large_integer_curve_uses_reference_fallback(self) -> None:
        """Feature: Python integer fallback.
        Description: Compare models with counts and calibration knots exceeding int64.
        Expectation: The native-enabled path retains the Python reference costs.
        """
        model = self._model(2, forward_ms=((0, 0), (2**80, 2)), backward_ms=((0, 0), (2**80, 3)))
        config = ExpertReplicaConfig(2, 2, 1)
        copies, counts = [{0: 2**70}, {1: 0}], ((2**70, 0), (0, 0))
        states = []
        for backend in (None, self.native):
            with patch.object(host, "_native_backend", return_value=backend):
                states.append(cost._QuotaCostState(model, copies, counts, config))
        self.assertEqual(states[0].costs, states[1].costs)
        calibration = DeviceCriticalPathCalibration("a" * 64, model.forward_ms, model.backward_ms, .01, .02, (0, 1))
        critical = self._model(2, transport="shmem_signal_kernel_gradient_bands", _rank_nodes=(0, 1),
                               forward_ms=model.forward_ms, backward_ms=model.backward_ms, critical_path=calibration)
        estimates = []
        for backend in (None, self.native):
            with patch.object(host, "_native_backend", return_value=backend):
                estimates.append(self._bits(asdict(calibration.estimate(critical, copies, counts, config, trace=True))))
        self.assertEqual(estimates[0], estimates[1])

    @staticmethod
    def _bits(value: object) -> object:
        """Retain IEEE bits while comparing complete immutable DAG explanations."""
        if isinstance(value, float):
            return value.hex()
        if isinstance(value, dict):
            return {key: TestReplicaHost._bits(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return tuple(TestReplicaHost._bits(item) for item in value)
        return value

    def test_critical_path_trace_preserves_tasks_order_and_score_bits(self) -> None:
        """Feature: Ordered native DAG inputs.
        Description: Compare full plans and traces under varied domains and gradient bands.
        Expectation: Every task, dependency time, owner completion and score bit matches.
        """
        generator = random.Random(7092)
        for index in range(120):
            size, home, budget = generator.choice((2, 3, 4)), 3, generator.choice((0, 1, 2, 6))
            nodes = tuple(rank // 2 for rank in range(size))
            overlap = bool(index % 2)
            calibration = DeviceCriticalPathCalibration(
                execution_signature="a" * 64, home_forward_ms=((0, 0), (17, .12), (4096, 3.1)),
                home_backward_ms=((0, 0), (31, .22), (4096, 8.7)), intra_node_gradient_ms=.013,
                inter_node_gradient_ms=.071, communication_domains=tuple(rank % 2 for rank in range(size)),
                overlap_verified=overlap, gradient_workers=1 + index % 4,
                gradient_bands=((0, .5, .003, .009), (1, 1., .006, .012)) if overlap else (),
                credit_ms=.002, weight_ack_ms=.003, gradient_ack_ms=.007,
            )
            model = self._model(size, transport="shmem_signal_kernel_gradient_bands", _rank_nodes=nodes,
                                critical_path=calibration)
            counts = [[generator.randrange(80) for _ in range(size * home)] for _ in range(size)]
            outcomes = []
            for backend in (None, self.native):
                with patch.object(host, "_native_backend", return_value=backend):
                    plan = planner._build_expert_replica_plan(counts, budget, adaptive=bool(budget),
                                                            cost_model=model, _rank_nodes=nodes)
                    plan.validate()
                    copies = [{expert: rows for expert, rows in zip(slots, values) if expert >= 0}
                              for slots, values in zip(plan.slot_to_logical, plan.destination_counts)]
                    for row in copies:
                        values = list(row.items())
                        generator.shuffle(values)
                        row.clear()
                        row.update(values)
                    # Both paths consume the same insertion order for this plan.
                    with patch.object(host, "_native_backend", return_value=None):
                        reference = calibration.estimate(model, copies, plan.logical_counts, plan.config, trace=True)
                    actual = calibration.estimate(model, copies, plan.logical_counts, plan.config, trace=True)
                    self.assertEqual(self._bits(asdict(reference)), self._bits(asdict(actual)))
                    outcomes.append((plan.slot_to_logical, plan.dispatch_counts, model.estimate_ms(plan).hex()))
            self.assertEqual(outcomes[0], outcomes[1])

    def test_critical_path_subclass_retains_reference_interpolation(self) -> None:
        """Feature: Custom model fallback.
        Description: Score an overridden compute method with native primitives available.
        Expectation: The custom method remains authoritative and the native phase is skipped.
        """
        class CustomModel(cost.ExpertReplicaCostModel):
            """Exercise a user-defined interpolation contract."""

            def compute_ms(self, rows: int, *, backward: bool = False) -> float:
                """Add a deliberately observable custom service duration."""
                return super().compute_ms(rows, backward=backward) + .125

        calibration = DeviceCriticalPathCalibration("a" * 64, ((0, 0), (4096, 4)),
                                                   ((0, 0), (4096, 8)), .01, .02, (0, 1))
        original = self._model(2, transport="shmem_signal_kernel_gradient_bands", _rank_nodes=(0, 1),
                               critical_path=calibration)
        model = CustomModel(**{field.name: getattr(original, field.name) for field in fields(original)})
        with patch.object(host, "_native_backend", return_value=self.native), patch.object(
                cost, "_ordered_compute", side_effect=AssertionError("Custom models need the Python contract")):
            result = calibration.estimate(model, [{0: 4}, {1: 0, 0: 4}], ((8, 0), (0, 0)),
                                          ExpertReplicaConfig(2, 2, 1), trace=True)
        with patch.object(host, "_native_backend", return_value=None):
            reference = calibration.estimate(model, [{0: 4}, {1: 0, 0: 4}], ((8, 0), (0, 0)),
                                             ExpertReplicaConfig(2, 2, 1), trace=True)
        self.assertEqual(self._bits(asdict(result)), self._bits(asdict(reference)))
        self.assertGreater(result.total_ms, 0)

    def test_critical_path_range_errors_preserve_reference_messages(self) -> None:
        """Feature: Calibrated domain failures.
        Description: Exceed owner-only coverage and active replica compute coverage.
        Expectation: Both paths reject with the same descriptive exception.
        """
        calibration = DeviceCriticalPathCalibration("a" * 64, ((0, 0), (8, 4)),
                                                   ((0, 0), (8, 8)), .01, .02, (0, 1))
        model = self._model(2, transport="shmem_signal_kernel_gradient_bands", _rank_nodes=(0, 1),
                            critical_path=calibration)
        config = ExpertReplicaConfig(2, 2, 1)
        for copies in ([{0: 9}, {1: 0}], [{0: 1}, {1: 0, 0: 4097}]):
            outcomes = []
            for backend in (None, self.native):
                with patch.object(host, "_native_backend", return_value=backend):
                    with self.assertRaises(ValueError) as caught:
                        calibration.estimate(model, copies, ((4098, 0), (0, 0)), config)
                    outcomes.append(str(caught.exception))
            self.assertEqual(outcomes[0], outcomes[1])

    def test_ordered_dict_fallback_preserves_public_trace_and_legacy_sums(self) -> None:
        """Feature: Dictionary subclass fallback.
        Description: Reorder owned experts and guests through OrderedDict.move_to_end.
        Expectation: Public calibration traces and legacy sums retain Python items order.
        """
        curve = ((0, 0.), (1, .1), (2, .2), (3, .3), (4096, 409.6))
        owner = OrderedDict(((0, 1), (1, 2), (2, 3)))
        owner.move_to_end(0)
        guests = OrderedDict(((0, 0), (1, 0), (2, 1), (3, 100)))
        guests.move_to_end(2)
        cases = (([owner, {3: 0, 4: 0, 5: 0}], ((1, 2, 3, 0, 0, 0), (0,) * 6), 6, 1),
                 ([guests, {2: 0, 3: 0}], ((0, 0, 1, 100), (0,) * 4), 4, 2))
        for copies, counts, experts, budget in cases:
            calibration = DeviceCriticalPathCalibration("a" * 64, curve, curve, .01, .02, (0, 1),
                                                       overlap_verified=True)
            model = self._model(2, forward_ms=curve, backward_ms=curve, critical_path=calibration,
                                transport="shmem_signal_kernel_gradient_bands", _rank_nodes=(0, 1))
            config = ExpertReplicaConfig(experts, 2, budget)
            traces, costs = [], []
            for backend in (None, self.native):
                with patch.object(host, "_native_backend", return_value=backend):
                    traces.append(self._bits(asdict(calibration.estimate(model, copies, counts, config, trace=True))))
                    legacy = self._model(2, forward_ms=curve, backward_ms=curve)
                    costs.append(cost._QuotaCostState(legacy, copies, counts, config).score().hex())
            self.assertEqual(traces[0], traces[1])
            self.assertEqual(costs[0], costs[1])
            self.assertIsNone(self.native.ordered_compute(copies[0], config.home_experts, 0, curve, curve, True))

    def test_integer_time_interpolation_retains_exact_python_division(self) -> None:
        """Feature: Integer time-knot fallback.
        Description: Interpolate a legal int64 knot immediately above the FP64 exact range.
        Expectation: Public and legacy scores retain the reference division bits.
        """
        curve = ((0, 0), (2**53 + 1, 78))
        calibration = DeviceCriticalPathCalibration("a" * 64, curve, curve, .01, .02, (0, 1))
        model = self._model(2, forward_ms=curve, backward_ms=curve, critical_path=calibration,
                            transport="shmem_signal_kernel_gradient_bands", _rank_nodes=(0, 1))
        config, copies, counts = ExpertReplicaConfig(2, 2, 1), [{0: 4096}, {1: 0}], ((4096, 0), (0, 0))
        traces, costs = [], []
        for backend in (None, self.native):
            with patch.object(host, "_native_backend", return_value=backend):
                result = calibration.estimate(model, copies, counts, config, trace=True)
                self.assertEqual(result.forward_ms.hex(), "0x1.37fffffffffffp-35")
                traces.append(self._bits(asdict(result)))
                legacy = self._model(2, forward_ms=curve, backward_ms=curve)
                costs.append(cost._QuotaCostState(legacy, copies, counts, config).score().hex())
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(costs[0], costs[1])
        self.assertIsNone(self.native.ordered_compute(copies[0], 1, 0, curve, curve, False))

    def test_duration_cache_subclass_keeps_python_get_semantics(self) -> None:
        """Feature: Legacy duration-cache fallback.
        Description: Supply a dictionary subclass with an overridden get method.
        Expectation: Native availability cannot bypass the cache's Python behavior.
        """
        class CustomDurations(dict):
            """Make the mapping's Python lookup observably different from its storage."""

            def get(self, key: int, default: object = None) -> tuple[float, float]:
                """Return a deliberate custom duration for every row count."""
                return .5, .75

        values = []
        for backend in (None, self.native):
            with patch.object(host, "_native_backend", return_value=backend):
                state = cost._QuotaCostState(self._model(2), [{0: 4}, {1: 0}], ((4, 0), (0, 0)),
                                             ExpertReplicaConfig(2, 2, 1), CustomDurations())
                values.append(state.score().hex())
        self.assertEqual(values[0], values[1])

    def test_supported_plain_dict_hook_is_observed_at_actual_call_site(self) -> None:
        """Feature: Reachable concrete-model hook.
        Description: Spy on the alias called by the cost model using supported float knots.
        Expectation: Every rank and phase reaches the existing ordered native primitive.
        """
        calibration = DeviceCriticalPathCalibration("a" * 64, ((0, 0.), (4096, 4.)),
                                                   ((0, 0.), (4096, 8.)), .01, .02, (0, 1))
        model = self._model(2, critical_path=calibration, transport="shmem_signal_kernel_gradient_bands",
                            _rank_nodes=(0, 1))
        with patch.object(host, "_native_backend", return_value=self.native), patch.object(
                cost, "_ordered_compute", wraps=cost._ordered_compute) as called:
            calibration.estimate(model, [{0: 4}, {1: 0, 0: 4}], ((8, 0), (0, 0)), ExpertReplicaConfig(2, 2, 1))
        self.assertEqual(called.call_count, 4)

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
        with patch.dict("os.environ", {"HYPER_PARALLEL_REPLICA_HOST_NATIVE": "1"}), patch.object(
                host.Path, "is_file", return_value=True), patch.object(host, "import_module", return_value=StaleBinary()):
            with self.assertRaisesRegex(RuntimeError, "incompatible ABI"):
                host._native_backend()
