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
"""Regression coverage for single-backend placement and original-route fallback."""

import struct
from dataclasses import replace
from typing import get_type_hints
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from hyper_multicore import MegaMoeExperts
from hyper_multicore.experimental import BasePlanner, ExpertPlacement, HotReplicaPlanner, MegaMoeTuningOptions, PlacementRequest
from hyper_multicore.modules.mega_moe import route, resources as resource_module
from hyper_multicore.modules.hot_replica import capability, checkpoint as replay, mx_adapter
from hyper_multicore.modules.hot_replica._planner.plan import ExpertExecutionPlan, _replica_transfers
from hyper_multicore.modules.hot_replica._planner.capacity import ExpertReplicaConfig
from hyper_multicore.modules.hot_replica._planner.receive import ReplicaCost
from hyper_multicore.modules.hot_replica.bank_ops import reduce_banks
from hyper_multicore.modules.hot_replica import bounded_placement
from hyper_multicore.modules.hot_replica.bounded_placement import _placement_metadata, read_placement_wire
from hyper_multicore.modules.hot_replica.shmem_transfer import ReplicaExecutionMode, SharedExpertPool
from hyper_multicore.modules.hot_replica.mega_moe_adapter import (
    ExpertWeightBinding, PlacementResources, _get_mega_bank, _release_mega_bank,
)
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.modules.mega_moe.numerics import MegaMoeNumericsOptions
from hyper_multicore.modules.mega_moe.workspace import _workspace_event_experts
from hyper_multicore.modules.hot_replica.placement import validate_quotas
from hyper_multicore.modules.hot_replica.mx_gradient import (
    GRADIENT_HELPERS, MXGradientPool, MXGradientStorage, pack_mx_gradient_return,
)


@pytest.mark.parametrize(
    "precision,mode,value,error",
    [
        ("bf16", "push", True, TypeError),
        ("mxfp8", "push", HotReplicaPlanner(), ValueError),
        ("mxfp8", "pull", HotReplicaPlanner(), ValueError),
    ],
)
def test_unsupported_configuration_fails_before_binding(precision, mode, value, error):
    """Reject unsupported placement options before device resource binding."""
    with pytest.raises(error):
        MegaMoeExperts(
            hidden_size=128,
            intermediate_size=128,
            num_experts=4,
            top_k=2,
            precision=precision,
            tuning_options=MegaMoeTuningOptions(dispatch_mode=mode, expert_placement=value),
        )


def test_placement_requires_multiple_ranks():
    with pytest.raises(ValueError, match="EP group"):
        MegaMoeExperts(
            hidden_size=128,
            intermediate_size=128,
            num_experts=4,
            top_k=2,
            tuning_options=MegaMoeTuningOptions(expert_placement=HotReplicaPlanner()),
        )


@pytest.mark.parametrize("mode", ["push", "pull"])
def test_bf16_transports_accept_the_same_placement_limits(mode):
    """Both transports bind explicit topology, receive and guest-slot settings."""
    from hyper_multicore.modules.mega_moe import module

    with patch.object(module, "_resolve_ep_size", return_value=2):
        planner = HotReplicaPlanner(replica_slots_per_rank=7, receive_capacity_percent=102, cross_node_cost=1.5)
        expert = MegaMoeExperts(
            hidden_size=128, intermediate_size=128, num_experts=16, top_k=2,
            tuning_options=MegaMoeTuningOptions(dispatch_mode=mode, expert_placement=planner),
        )
    try:
        spec = expert._resource_group.specification  # pylint: disable=protected-access
        assert spec["dispatch_mode"] == mode and spec["hot_replica"]
        assert (spec["replica_slots_per_rank"], spec["receive_capacity_percent"], spec["cross_node_cost"]) == (7, 102, 1.5)
    finally:
        expert.close()


@pytest.mark.parametrize("slots", [None, 1, 2, 6])
def test_mx_budget_sizes_the_route_and_workspace(slots):
    from hyper_multicore.modules.mega_moe import module

    with patch.object(module, "_resolve_ep_size", return_value=2):
        expert = MegaMoeExperts(
            hidden_size=128, intermediate_size=128, num_experts=12, top_k=2, precision="mxfp8",
            numerics_options=MegaMoeNumericsOptions(partial_weight_grad_dtype="float32"),
            tuning_options=MegaMoeTuningOptions(expert_placement=HotReplicaPlanner(replica_slots_per_rank=slots)),
        )
    try:
        options = expert._resource_group.specification  # pylint: disable=protected-access
        parent = SimpleNamespace(spec=SimpleNamespace(num_experts=12, ep_size=2, local_experts=6))
        resources = mx_adapter.MXPlacementResources(parent, replica_slots_per_rank=options["replica_slots_per_rank"])
        cap = 6 if slots is None else slots
        assert resources.config.replica_slots_per_rank == cap
        assert _workspace_event_experts(options) == resources.config.physical_experts == 12 + 2 * cap
    finally:
        expert.close()


@pytest.mark.parametrize("runtime_rows", [False, True])
def test_a3_bf16_pull_requires_runtime_row_get(tmp_path, runtime_rows):
    """Check capability agreement against both legacy and current A3 wheels."""
    tensor = SimpleNamespace(device=SimpleNamespace(type="npu", index=0), dtype=torch.bfloat16)
    artifacts = {"soc": "ascend910_93", "bridge": "bridge.so", "kernels": "kernels.so"}
    (tmp_path / artifacts["bridge"]).touch()
    (tmp_path / artifacts["kernels"]).touch()
    build_info = {"a3_replica_bf16_partials": True}
    if runtime_rows:
        build_info["a3_bf16_pull_runtime_rows"] = True
    selected = {"root": tmp_path, "variant": {"hot_replica": artifacts}, "build_info": build_info}
    with (
        patch.object(capability, "load_torch_npu"),
        patch.object(capability, "query_npu_arch", return_value=2201),
        patch.object(torch.npu, "get_device_name", return_value="Ascend910_9391"),
        patch.object(capability, "validate_native_compatibility", return_value=selected),
        patch.object(capability._collective, "exchange", side_effect=lambda value, **_: [value, value]),
    ):
        assert capability.require_capability(tensor, None) == (True, (0, 0))
        expected = nullcontext() if runtime_rows else pytest.raises(RuntimeError, match="runtime-row GET support")
        with expected:
            assert capability.require_capability(tensor, None, dispatch_mode="pull") == (True, (0, 0))


@pytest.mark.parametrize("dispatch_mode", ["push", "pull"])
@pytest.mark.parametrize("rank", range(3))
def test_a3_bounded_metadata_matches_the_ordinary_route(dispatch_mode, rank):
    """The quota route packs each source and destination in the ordinary descriptor order."""
    counts = np.array([[2, 0, 3, 1, 4, 2], [0, 1, 0, 2, 6, 3], [2, 8, 0, 1, 0, 1]], dtype=np.int32)
    spec = MegaMoeSpec(12, 64, 128, 6, 1, None, 128, 3, None, rank, 24, dispatch_mode=dispatch_mode)
    wire = [[12, 0, 0, *row] for row in counts.tolist()]
    expected = route._host_route_metadata(wire, spec, 128, torch.device("cpu"))
    observed = _placement_metadata(counts, spec, 128, 128, 12, torch.device("cpu"))
    assert (observed.local_num_tokens, observed.plan_tokens) == (12, 128)
    for name in ("dispatch_src_off", "dispatch_target_off", "dispatch_size", "combine_src_off",
                 "combine_target_off", "combine_size", "group_list"):
        assert torch.equal(getattr(observed, name), getattr(expected, name))
        assert torch.equal(getattr(observed.for_kernel(64), name), getattr(expected.for_kernel(64), name))


def test_placement_wire_is_reused_only_for_equal_agreed_counts():
    ranks, experts = 32, 384
    spec = MegaMoeSpec(128, 64, 128, experts, 1, None, 128, ranks, None, 0, 24, auto_tokens=True)
    wire = torch.zeros((ranks, experts + 3), dtype=torch.int32)
    wire[:, 1:4] = 1
    workspace = SimpleNamespace(placement_wire=None)
    first = read_placement_wire(wire, spec, workspace, "stream")
    assert read_placement_wire(wire, spec, workspace, "stream") is first
    wire[0, 3], wire[0, 3 + experts // ranks] = 0, 1
    second = read_placement_wire(wire, spec, workspace, "stream")
    assert second is not first and first.wire[0, 3] == 1
    assert second.summary.maximum == ranks - 1
    assert second.source_rows == [1] * ranks
    wire[0, 3] = -1
    with pytest.raises(RuntimeError, match="actual distinct Top-K"):
        read_placement_wire(wire, spec, workspace, "stream")
    assert workspace.placement_wire is second


@pytest.mark.parametrize("options", [None, MegaMoeTuningOptions()])
def test_default_keeps_placement_disabled(options):
    expert = MegaMoeExperts(hidden_size=128, intermediate_size=128, num_experts=4, top_k=2, tuning_options=options)
    try:
        assert expert._resource_group.specification["hot_replica"] is False  # pylint: disable=protected-access
    finally:
        expert.close()


@pytest.mark.parametrize("name,arch", [("Ascend910B2", 2201), ("Ascend950", 3510)])
def test_architecture_alone_does_not_enable_hot_replica(name, arch):
    """Require the actual A3 device identity in addition to the architecture."""
    tensor = SimpleNamespace(device=SimpleNamespace(type="npu", index=5), dtype=torch.bfloat16)
    with (
        patch.object(capability, "load_torch_npu"),
        patch.object(capability, "query_npu_arch", return_value=arch),
        patch.object(torch.npu, "get_device_name", return_value=name) as device_name,
        patch.object(capability, "validate_native_compatibility") as manifest,
        pytest.raises(ValueError, match="Atlas A3"),
    ):
        capability._require_a3(tensor)  # pylint: disable=protected-access
    device_name.assert_called_once_with(5)
    manifest.assert_not_called()


def test_supported_device_requires_selected_adapter_artifacts():
    tensor = SimpleNamespace(device=SimpleNamespace(type="npu", index=3), dtype=torch.bfloat16)
    with (
        patch.object(capability, "load_torch_npu"),
        patch.object(capability, "query_npu_arch", return_value=2201),
        patch.object(torch.npu, "get_device_name", return_value="Ascend910_9391"),
        patch.object(capability, "validate_native_compatibility", return_value={"variant": {}}),
        pytest.raises(RuntimeError, match="no Atlas A3"),
    ):
        capability._require_a3(tensor)  # pylint: disable=protected-access


def test_no_move_preserves_original_prepared_route_and_capacity():
    """Preserve validated routing and capacity when placement is declined."""
    hidden = torch.empty(2, 8)
    ids = torch.tensor([[0], [1]])
    spec, workspace, metadata = SimpleNamespace(dispatch_mode="push"), object(), object()
    counts = torch.tensor([1, 1])
    gathered = torch.tensor([[0, 2, 1, 1, 1]])
    permutation = (hidden, torch.tensor([0, 1]))
    original = object()
    prepare, select, work = Mock(), Mock(return_value=None), Mock()
    summary = route._RouteSummary(2, 128, 2)
    with (
        patch.object(route, "_read_cached_route", return_value=(summary, [], metadata)) as read,
        patch.object(route, "_finalize_dynamic_route", return_value=original) as finish,
    ):
        result = route._prepare_placed_route(  # pylint: disable=protected-access
            hidden, ids, spec, workspace, permutation, counts, gathered, work, None, prepare, select
        )
    assert result is original
    read.assert_called_once()
    work.wait.assert_called_once()
    prepare.assert_called_once_with(128, 2)
    assert finish.call_args.args[0] is permutation
    assert finish.call_args.args[4] is metadata
    assert finish.call_args.args[7] is spec
    assert finish.call_args.args[8] is workspace


def test_failed_weight_publication_releases_bank():
    pool = Mock(requires_mte_edges=False)
    pool.prefetch.side_effect = RuntimeError("launch failed")
    binding = ExpertWeightBinding(pool, SimpleNamespace(copies=object(), borrowed_capacity=1), "EOI")
    with pytest.raises(RuntimeError, match="launch failed"):
        binding.acquire((object(), object()))
    pool.release.assert_called_once()


def test_failed_split_publication_drains_streams_before_bank_release():
    pool = Mock(requires_mte_edges=False)
    pool.pipeline.begin.side_effect = RuntimeError("peer copy failed")
    binding = ExpertWeightBinding(pool, Mock(), "EIO", execution_mode=ReplicaExecutionMode.FP32_BAND)
    with patch("hyper_multicore.modules.hot_replica.shmem_transfer._ordered_transfers",
               return_value=nullcontext()), pytest.raises(RuntimeError, match="peer copy failed"):
        binding.acquire((object(), object()))
    assert [name for name, _, _ in pool.method_calls][-2:] == ["pipeline.complete", "release"]
    assert pool.pipeline.resident is None


def test_retained_weight_bank_prevents_parent_cleanup():
    parent = PlacementResources(Mock(), {}, False)
    parent.pool = Mock()
    parent.pool.can_close.return_value = False
    assert not parent.can_close()
    parent.pool.can_close.return_value = True
    assert parent.can_close()


def test_fp32_owner_gradient_reaches_native_reduction_without_rounding():
    gradient = torch.tensor([[[1.003]], [[2.003]]], dtype=torch.float32)
    banks = (torch.zeros(4, 1, 1, dtype=torch.bfloat16),)
    copies = torch.empty(1, 0, dtype=torch.int32)
    bridge = Mock()
    with patch("hyper_multicore.modules.hot_replica.bank_ops.load_bridge", return_value=bridge):
        actual = reduce_banks((gradient,), banks, copies, 0, 1)[0]
    torch.testing.assert_close(actual, gradient, rtol=0, atol=0)
    torch.testing.assert_close(bridge.reduce.call_args.args[0], gradient, rtol=0, atol=0)


@pytest.mark.parametrize(("partial_dtype", "new_bridge", "native"), [
    (torch.bfloat16, True, True), (torch.bfloat16, False, False), (torch.float32, True, False),
])
def test_native_home_gradient_keeps_fp32_guest_payload_and_legacy_fallback(partial_dtype, new_bridge, native):
    """Keep the local partial dtype and publish the replica in FP32."""
    gradient = torch.arange(12, dtype=partial_dtype).view(3, 2, 2)
    bank = torch.empty(4, 2, 2, dtype=torch.bfloat16)
    copies = torch.tensor([[2], [0]], dtype=torch.int32)
    bridge = SimpleNamespace(reduce=Mock())
    if new_bridge:
        bridge.native_home_gradient = True
    with patch("hyper_multicore.modules.hot_replica.bank_ops.load_bridge", return_value=bridge):
        result, = reduce_banks((gradient,), (bank,), copies, 0, 1, (torch.bfloat16,))
    bridge.reduce.assert_called_once()
    source, output, slots, *_ = bridge.reduce.call_args.args
    assert source.dtype == (partial_dtype if native else torch.float32)
    assert output is result and result.dtype == torch.bfloat16
    assert result.untyped_storage().data_ptr() != gradient.untyped_storage().data_ptr()
    torch.testing.assert_close(slots, gradient[2:].float(), rtol=0, atol=0)


def test_placement_reuses_validated_source_totals_and_autograd_mask():
    """Reuse existing source counts and gradient participation without rereading."""
    hidden = torch.empty(2, 8)
    ids = torch.tensor([[0], [1]])
    counts = torch.tensor([1, 1])
    gathered = torch.tensor([[0, 2, 1, 1, 1]])
    selection = Mock(return_value=None)

    def read(
        payload: object,
        spec: object,
        workspace: object,
        device: object,
        source_rows: list[int],
        **kwargs: object
    ) -> tuple:
        """Expose source totals from the existing validation readback."""
        del payload, spec, workspace, device, kwargs
        source_rows.extend((2, 6))
        return route._RouteSummary(2, 128, 6), None, None

    with patch.object(route, "_read_cached_route", side_effect=read), patch.object(route, "_finalize_dynamic_route"):
        route._prepare_placed_route(  # pylint: disable=protected-access
            hidden, ids, SimpleNamespace(dispatch_mode="push"), object(), (hidden, ids), counts, gathered,
            None, None, Mock(), selection, 1
        )
    assert selection.call_args.args[4] == [2, 6]
    assert selection.call_args.args[6] == 1


def test_configuration_mismatch_fails_before_optional_capability_collective():
    """Reject inconsistent rank settings before optional capability collectives."""
    cls = resource_module._MegaMoeExecutionResources  # pylint: disable=protected-access
    specification = {"hot_replica": True, "ep_group": object()}
    spec = SimpleNamespace(local_num_tokens=128)
    with (
        patch.object(resource_module, "bind_mega_moe_spec", return_value=spec),
        patch.object(resource_module.dist, "is_initialized", return_value=True),
        patch.object(cls, "_validate_distributed_configuration", side_effect=ValueError("configurations differ")),
        patch.object(capability, "require_capability") as gate,
        patch.object(resource_module, "_get_heap_manager") as heap,
        pytest.raises(ValueError, match="configurations differ"),
    ):
        cls(
            specification,
            torch.empty(1, 1),
            expert_weight_layout="EIO",
            shared=False,
            active_specifications=(specification,),
        )
    gate.assert_not_called()
    heap.assert_not_called()


@pytest.mark.parametrize(("mode", "requested", "expected"), [
    ("push", None, 12), ("push", 8, 8), ("push", 2, 2), ("pull", None, 12), ("pull", 8, 8),
])
def test_a3_bounded_planning_uses_the_configured_slot_budget(mode, requested, expected):
    """Check the budget passed into the actual A3 bounded planner."""
    spec = MegaMoeSpec(512, 5120, 1792, 192, 8, 1.25, 8192, 16, None, 0, 24,
                       dispatch_mode=mode)
    placement = bounded_placement.BoundedPlacement(spec, replica_slots_per_rank=requested)
    placement.nodes = (0,) * 8 + (1,) * 8
    counts = np.zeros((16, 192), dtype=np.int32)
    hidden = torch.empty(512, 1)
    ids = torch.empty(512, spec.top_k, dtype=torch.int32)
    with patch.object(bounded_placement, "select_replica_plan", return_value=None) as build:
        for _ in range(2):
            assert placement.select(hidden, ids, torch.from_numpy(counts), [512*spec.top_k]*spec.ep_size,
                                    None, ReplicaCost()) is None
    build.assert_called_once()
    assert build.call_args.args[1].replica_slots_per_rank == expected



def test_checkpoint_agrees_on_version_and_activation(monkeypatch):
    """Reject stale parameters across ranks, even if the token route matches."""
    monkeypatch.setattr(torch.npu, "current_stream", lambda _device: "s")
    monkeypatch.setattr(replay.dist, "all_gather_into_tensor", lambda out, x, **_: out.copy_(x.expand_as(out)))
    spec = SimpleNamespace(ep_size=2, ep_group=object())
    hidden, probabilities = torch.ones((1, 4)), torch.ones((1, 2))
    ids = torch.tensor([[0, 1]])
    owner = object()
    version = [0]
    binding = Mock()

    def _invocation(counts=None):
        record, recomputing = replay.route_record(owner)
        if recomputing:
            route_matches = replay.agree_replay(
                record, hidden, ids, probabilities, counts, spec, 15, weight_error=False,
            )
            return route_matches, record.allow_activation
        record.ids, record.stream, record.route = ids.clone(), "s", SimpleNamespace(dump_weights=False)
        record.weight_binding = binding
        record.inputs = hidden.detach(), probabilities.detach()
        record.input_versions = (0, 0)
        record.activation = object()
        return None

    wrapped = replay.wrap_checkpoint(_invocation, lambda: version[0], prepare_replay=True)
    with torch.no_grad():
        wrapped()
    binding.prepare_replay.assert_not_called()
    assert wrapped() == (True, True)
    binding.prepare_replay.assert_called_once_with()
    with torch.no_grad(), pytest.raises(RuntimeError, match="separate wrap_checkpoint"):
        wrapped()
    assert wrapped() == (True, True)
    binding.prepare_replay.reset_mock()
    assert wrapped(torch.tensor([1, 1])) == (False, False)
    version[0] += 1
    assert wrapped() == (False, False)
    binding.prepare_replay.assert_called_once_with()


@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("bad_weight_rank", [None, 0, 1])
def test_checkpoint_converges_weight_errors_before_building_routes(monkeypatch, supplied, bad_weight_rank):
    """An invalid rank votes against reuse, then both ranks reach route validation."""
    from hyper_multicore.modules.mega_moe import invocation

    monkeypatch.setattr(torch.npu, "current_stream", lambda _device: "stream")
    ids = torch.tensor([[0, 1]], dtype=torch.int32)
    counts = torch.ones(2, dtype=torch.int32)
    hidden, probs = torch.ones(1, 2), torch.ones(1, 2)
    weights = (torch.ones(1, 2), torch.ones(1, 2))
    route_counts = counts if supplied else None
    for rank in (0, 1):
        spec = SimpleNamespace(ep_size=2, ep_group=object(), dispatch_mode="push", auto_tokens=True, top_k=2,
                               precision="bf16")
        placement = SimpleNamespace(statistics={"route_replays": 0, "route_replay_misses": 0})
        resources = SimpleNamespace(spec=spec, workspace=Mock(), placement=placement, ensure_capacity=Mock())
        votes = []
        def gather(out, local, **_kwargs):
            votes.append(int(local))
            out[:] = 0 if supplied else 1
            if bad_weight_rank is not None:
                out[bad_weight_rank] = 0
            assert int(out[rank]) == int(local)
        def prepare(hidden, *_args, **options):
            if votes and bad_weight_rank is not None:
                rows = [[0, 1, 1, 2, 1, 1, 2] for _ in range(2)]
                rows[bad_weight_rank][2] = route._INVALID_WEIGHT_MASK
                assert options["autograd_mask"] == rows[rank][2]
                route._validate_dynamic_rows(rows, spec)
            return route.PreparedTopKRoute(hidden, ids, counts, counts, object())

        monkeypatch.setattr(replay.dist, "all_gather_into_tensor", gather)
        monkeypatch.setattr(invocation, "prepare_topk_route", prepare)
        monkeypatch.setattr(invocation, "_materialize_route_permutation", lambda *args: (hidden, ids))
        tape = replay.ReplayCheckpoint(lambda: 1)
        with tape.scope(False), invocation.prepare_expert_invocation(hidden, ids, probs, route_counts, weights, resources):
            pass
        local_error = rank == bad_weight_rank
        current = (None, weights[1]) if local_error else weights
        expected = pytest.raises((TypeError, ValueError), match="expert weights") if bad_weight_rank is not None else nullcontext()
        with tape.scope(True), expected, invocation.prepare_expert_invocation(
                hidden, ids, probs, route_counts, current, resources, weight_error=local_error):
            pass
        assert len(votes) == 1
        assert placement.statistics["route_replays"] == (not supplied and bad_weight_rank is None)


def test_early_weight_restore_uses_bank_generation(monkeypatch):
    """A copy can be consumed after its stream event or recovered if the bank changed."""
    monkeypatch.setattr(torch.distributed, "get_rank", lambda _: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _: 2)
    copy_stream = SimpleNamespace(device="npu:0", wait_event=Mock())
    compute_stream = SimpleNamespace(device="npu:0", wait_event=Mock())
    active = [copy_stream]
    monkeypatch.setattr(torch.npu, "current_stream", lambda _: active[0])
    monkeypatch.setattr(torch.npu, "Stream", lambda **_: copy_stream)
    monkeypatch.setattr(torch.npu, "Event", Mock)
    monkeypatch.setattr(torch.npu, "stream", lambda _: nullcontext())
    pool = SharedExpertPool(torch.empty((0, 4)), object(), 1, 2, 1, "EIO", root=Mock(), access=nullcontext)
    pool.weights = (torch.zeros(2, 4, 4), torch.zeros(2, 2, 4))
    pool.completion_event = Mock()
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *_: None)
    placement = SimpleNamespace(copies=torch.empty((0,), dtype=torch.int32), borrowed_capacity=1)
    binding = ExpertWeightBinding(pool, placement, "EIO")
    tails = tuple(torch.full_like(bank[1:], 2) for bank in pool.weights)
    ready = object()
    binding.snapshot = tails, ready
    binding.prepare_replay()
    copy_stream.wait_event.assert_called_once_with(ready)
    pool.completion_event.record.assert_called_once_with(copy_stream)

    active[0] = compute_stream
    homes = tuple(torch.full_like(bank[:1], 3) for bank in pool.weights)
    weights = binding.acquire(homes)
    compute_stream.wait_event.assert_called_once_with(pool.completion_event)
    assert pool.statistics["early_restore_hits"] == 1
    for weight, home, tail in zip(weights, homes, tails):
        torch.testing.assert_close(weight, torch.cat((home, tail)))
    binding.release()

    active[0] = copy_stream
    binding.prepare_replay()
    # A later lease allowed another graph to overwrite the bank.
    pool.generation += 1
    for bank in pool.weights:
        bank.zero_()
    active[0] = compute_stream
    weights = binding.acquire(homes)
    for weight, home, tail in zip(weights, homes, tails):
        torch.testing.assert_close(weight, torch.cat((home, tail)))
    assert pool.statistics["early_restores"] == 2
    assert pool.statistics["early_restore_hits"] == 1
    binding.release()


def test_forward_uses_injected_planner_after_count_exchange():
    """Use the injected instance on the normal forward path."""
    from hyper_multicore.modules.mega_moe import invocation, module

    class RecordingPlanner(BasePlanner):
        def plan(self, request):
            self.seen = request

    planner = RecordingPlanner()
    with patch.object(module, "_resolve_ep_size", return_value=2):
        options = MegaMoeTuningOptions(expert_placement=planner)
        expert = MegaMoeExperts(hidden_size=8, intermediate_size=4, num_experts=4, top_k=1, tuning_options=options)
    hidden, ids = torch.empty(2, 8), torch.zeros(2, 1, dtype=torch.int32)
    probs, roots = torch.ones(2, 1), (torch.empty(2, 8, 8), torch.empty(2, 4, 8))
    spec = SimpleNamespace(architecture=0, auto_tokens=True, dispatch_mode="push", precision="bf16",
                           intermediate_size=4, rank_id=0, ep_size=2, ep_group=None, transport_engine="mte",
                           num_cube_cores=24, apply_probs_before_fc2=True)
    owner = SimpleNamespace(spec=spec, workspace=Mock(), prepare_auto_workspace=Mock())
    # pylint: disable-next=protected-access
    owner.placement = PlacementResources(owner, expert._resource_group.specification, False)
    counts = torch.tensor([[0, 2, 1, 2, 0, 0, 0], [0, 2, 1, 0, 0, 0, 2]])
    ordinary = object()

    def exchanged(*args, **kwargs):
        assert args[0] is hidden
        # pylint: disable-next=protected-access
        return route._prepare_placed_route(
            hidden, ids, spec, owner.workspace, (hidden, ids), counts[0, 3:], counts, None, None,
            kwargs["prepare_workspace"], kwargs["select_placement"], kwargs["autograd_mask"],
        )

    summary = route._RouteSummary(2, 128, 2)
    try:
        with (
            patch.object(expert, "_prepare_forward_inputs", return_value=(hidden, False)),
            patch.object(expert, "_get_execution_resources", return_value=owner),
            patch.object(torch.npu, "is_current_stream_capturing", return_value=False),
            patch.object(replay, "route_record", return_value=(None, False)),
            patch.object(invocation, "prepare_topk_route", side_effect=exchanged) as exchange,
            patch.object(route, "_read_cached_route", return_value=(summary, None, None)) as read,
            patch.object(route, "_finalize_dynamic_route", return_value=ordinary),
            # pylint: disable-next=protected-access
            patch.object(expert._backend, "forward", return_value=hidden) as forward,
        ):
            assert expert(hidden, ids, probs, expert_weights=roots) is hidden
        exchange.assert_called_once()
        read.assert_called_once()
        assert planner.seen.counts_by_source == ((2, 0, 0, 0), (0, 0, 0, 2))
        assert forward.call_args.args[-1] is ordinary
    finally:
        expert.close()


def test_experimental_placement_uses_host_quotas():
    """Resolve the public type and derive a single canonical copy table."""
    assert get_type_hints(MegaMoeTuningOptions)["expert_placement"] == BasePlanner | None
    request = PlacementRequest(((3, 2, 7, 5), (1, 0, 2, 1)), 15, 21, 0, 128, 64, 24, True)
    quotas = ExpertPlacement(((2, 3), (-1, -1)),
                             ((3, 2, 3, 0, 4, 5, 0, 0), (1, 0, 0, 0, 2, 1, 0, 0)))
    plan = validate_quotas(quotas, request, slots=2)
    assert plan.destination_loads == (9, 12)
    assert plan.slot_to_logical == ((0, 1, 2, -1), (2, 3, -1, -1))


def test_checkpoint_request_uses_the_agreed_training_phase():
    """A no-grad checkpoint forward uses the cost of subsequent training."""
    spec = SimpleNamespace(architecture=0, rank_id=0, intermediate_size=4, num_cube_cores=24,
                           apply_probs_before_fc2=True, dispatch_mode="push", transport_engine="mte",
                           ep_size=2)
    resources = PlacementResources(SimpleNamespace(spec=spec), {}, False)
    ids = torch.tensor([[0]])
    planner = Mock()
    with (
        patch.object(resources, "_select_custom", return_value=None) as select,
        patch("hyper_multicore.modules.hot_replica.mega_moe_adapter.is_checkpointing", return_value=True),
    ):
        resources.select(torch.zeros(1, 4), ids, torch.zeros(2, 5, dtype=torch.int32),
                         route._RouteSummary(1, 128, 1), [1, 1], None, 0, planner=planner)
    request = select.call_args.args[1]
    assert request.training and request.checkpointing
    assert request.ordinary_receive_capacity is None


@pytest.mark.parametrize("batch_experts", [False, True])
def test_a3_wide_receive_limit_uses_the_same_plan_as_auto(batch_experts):
    """Keep the same geometry in training, checkpoint and nonbinding-cap calls."""
    spec = MegaMoeSpec(128, 4096, 1408, 8, 2, 1.25, 384, 2, None, 0, 24,
                       expert_weight_layout="EOI", batch_experts=batch_experts, auto_tokens=True)
    parent = SimpleNamespace(spec=spec, workspace=SimpleNamespace(capacity_floor=384))
    tokens = 12288
    hidden = torch.empty(1, spec.hidden_size, dtype=torch.bfloat16).expand(tokens, -1)
    ids = torch.arange(2, dtype=torch.int32).expand(tokens, -1).contiguous()
    counts = [tokens, tokens, 0, 0, 0, 0, 0, 0]
    wire = torch.tensor([[0, tokens, 13, *counts]]*2, dtype=torch.int32)
    snapshot = read_placement_wire(wire, spec, parent.workspace, None)

    def select(percent, checkpointing, mask):
        planner = HotReplicaPlanner(receive_capacity_percent=percent)
        options = {"planner_key": (type(planner).__module__, type(planner).__qualname__),
                   "receive_capacity_percent": planner.receive_capacity_percent}
        resources = PlacementResources(parent, options, False, bf16_partials=True, rank_nodes=(0, 0))
        with (
            patch.object(resources, "bind", side_effect=lambda _, placement, **_kwargs: placement),
            patch("hyper_multicore.modules.hot_replica.mega_moe_adapter.is_checkpointing", return_value=checkpointing),
        ):
            return resources.select(hidden, ids, wire, snapshot.summary, snapshot.source_rows, None,
                                    mask, planner=planner, snapshot=snapshot)
    auto = select("auto", False, 13)
    assert auto is not None
    for percent, checkpointing, mask in (("auto", True, 0), (200, False, 13), (200, True, 0)):
        placed = select(percent, checkpointing, mask)
        assert placed is not None
        assert (placed.receive_capacity, placed.borrowed_capacity) == (auto.receive_capacity, auto.borrowed_capacity)
        for field in ("virtual_ids", "counts_by_source", "copies"):
            torch.testing.assert_close(getattr(placed, field), getattr(auto, field))


def test_manifest_checks_custom_policy_settings():
    """Different settings on two ranks fail in the resource configuration exchange."""
    # pylint: disable-next=protected-access
    validate = resource_module._MegaMoeExecutionResources._validate_distributed_configuration
    owner = SimpleNamespace(spec=SimpleNamespace(ep_size=2, ep_group=object(), expert_weight_layout="EIO",
                                                 architecture="ascend910_93"))
    def manifest(key):
        with patch.object(resource_module.dist, "all_gather_object") as gather:
            validate(owner, ({"planner_key": ("example", "Skew", key)},))
        return gather.call_args.args[1]

    remote = manifest("1.5")
    def mismatch(results, local, group):
        assert group is owner.spec.ep_group
        results[:] = [local, remote]

    with patch.object(resource_module.dist, "all_gather_object", side_effect=mismatch):
        with pytest.raises(ValueError, match="configurations differ"):
            validate(owner, ({"planner_key": ("example", "Skew", "2.0")},))


def test_mx_transfer_slots_match_the_execution_plan_and_gradient_record():
    """The wire record indexes guest banks while TN events index physical slots."""
    config = ExpertReplicaConfig(num_experts=12, ep_size=2, replica_slots_per_rank=2)
    home = config.home_experts
    copies = ((-1, 8), (1, -1))
    physical = (tuple(range(home)) + copies[0], tuple(range(home, 2 * home)) + copies[1])
    counts = ((0,) * config.num_experts,) * config.ep_size
    dispatch = ((0,) * config.physical_experts,) * config.ep_size
    plan = ExpertExecutionPlan(config, counts, physical, dispatch)
    plan.validate()
    edges = _replica_transfers(copies, home)
    assert edges == plan.transfers
    assert [(edge.target_rank, edge.target_slot, edge.guest_slot(home)) for edge in edges] == [
        (0, home + 1, 1), (1, home, 0),
    ]

    placement = SimpleNamespace(borrowed_capacity=2, copies_by_rank=copies)
    storage = MXGradientStorage(256, 512, 768, 1024, 1280, 1536, 256)
    events = tuple(range(100, 100 + config.slots_per_rank))
    payload = pack_mx_gradient_return(placement, 0, home, events, 128, storage, 1, 16)
    words = struct.unpack(f"<{len(payload) // 8}Q", payload)
    assert words[16:19] == (1, 1, events[home + 1])  # Incoming owner, guest bank, TN event.
    assert words[19:23] == (1, 0, 1, events[1])  # Target, guest bank, owner slot and event.


@pytest.mark.parametrize("local_experts", (4, 6, 8))
def test_mx_snapshot_reuses_validated_totals_in_the_active_plan(local_experts):
    """The validated snapshot takes the same exact placement as the raw count path."""
    ep, tokens, top_k = 32, 4096, 8
    spec = MegaMoeSpec(local_num_tokens=tokens, hidden_size=5120, intermediate_size=1792,
                       num_experts=ep * local_experts, top_k=top_k, initial_capacity_factor=4,
                       receive_capacity=ep * tokens * top_k, ep_size=ep, ep_group=None,
                       rank_id=0, num_cube_cores=32, auto_tokens=True, architecture=3510, precision="mxfp8")
    resource = mx_adapter.MXPlacementResources(SimpleNamespace(spec=spec, workspace=object()))
    assert resource.config.replica_slots_per_rank == local_experts
    resource.nodes = tuple(rank // 8 for rank in range(ep))
    resource._previous_active = True
    wire = torch.zeros((ep, spec.num_experts + 3), dtype=torch.int32)
    wire[:, 1] = tokens
    wire[:, 2] = mx_adapter._TRAINING_MASK
    wire[:, 3:3 + top_k] = tokens
    hidden = SimpleNamespace(shape=(tokens, spec.hidden_size), device=wire.device)
    source_rows = []
    summary, _, _, snapshot = resource._read_route(wire, hidden, None, source_rows, use_snapshot=True)

    def _geometry(plan, *_):
        return SimpleNamespace(plan=plan, storage_bytes=lambda: 0)

    with patch.object(mx_adapter, "_render_geometry", side_effect=_geometry):
        baseline = resource._select(hidden, wire, summary, wire.tolist(), source_rows,
                                    mx_adapter._TRAINING_MASK, None)
        other = mx_adapter.MXPlacementResources(resource.parent, rank_nodes=resource.nodes)
        with patch.object(mx_adapter, "select_replica_plan", wraps=mx_adapter.select_replica_plan) as planner:
            actual = other._select(hidden, wire, summary, None, source_rows, mx_adapter._TRAINING_MASK, None,
                                   snapshot=snapshot, use_snapshot=True)
            repeated = other._select(hidden, wire, summary, None, source_rows, mx_adapter._TRAINING_MASK, None,
                                     snapshot=snapshot, use_snapshot=True)
            planner.assert_called_once()
            assert planner.call_args.kwargs["totals"] == snapshot.expert_totals.tolist()
    assert baseline.plan.transfers
    baseline.plan.validate()
    assert actual.plan == baseline.plan
    assert repeated is actual

    # A negative expert count is invalid even when the source sum is correct.
    wire[0, 3 + top_k] = -1
    wire[0, 4 + top_k] = 1
    with pytest.raises(RuntimeError, match="actual distinct Top-K"):
        resource._read_route(wire, hidden, None, [], use_snapshot=True)


@pytest.mark.parametrize("guest_slots", [2, 6])
def test_mx_gradient_bank_registers_with_workspace_growth(guest_slots):
    """Reserve the FP32 guest inbox and both projections in the route transaction."""
    hidden = torch.empty((0, 128), dtype=torch.bfloat16)
    spec = SimpleNamespace(local_experts=6, hidden_size=128, intermediate_size=64, num_cube_cores=24,
                           ep_group=object(), expert_weight_layout="EIO", architecture=3510, transport_engine="udma")
    manager = Mock()
    manager.access.side_effect = nullcontext
    owner = Mock(bounded=None, dedicated_bank=False)
    workspace = object()
    with (patch("torch.distributed.get_rank", return_value=0),
          patch("torch.distributed.get_world_size", return_value=16),
          patch.object(MXGradientPool, "allocate") as allocate):
        pool = _get_mega_bank(hidden, SimpleNamespace(spec=spec, heap_manager=manager, workspace=workspace), owner,
                              mx_guest_slots=guest_slots, tokens=4096, received=32768)
    assert manager.reserve.call_args.kwargs['workspace_growth'] == (workspace, 4096, 32768)
    ((key, layout),), = manager.reserve.call_args.args
    assert key["kind"] == "hot_replica_mx_gradients"
    assert pool.guest_slots == guest_slots
    assert layout.collective_signature[-1] == guest_slots
    fp32_guests = guest_slots * 3 * spec.hidden_size * spec.intermediate_size * 4
    flags = (3 * pool.flag_words + GRADIENT_HELPERS * 16) * 4
    scratch = GRADIENT_HELPERS * 2 * pool.chunk_elements * 4
    assert layout.required_bytes == fp32_guests + flags + scratch + 1024
    assert owner in pool.owners and pool.pipeline is None
    manager.bind.assert_called_once_with(pool, spec, key)
    allocate.assert_called_once_with()


def test_mx_bank_shares_compatible_owners_and_closes_only_after_last_owner():
    hidden = torch.empty((0, 128), dtype=torch.bfloat16)
    spec = SimpleNamespace(local_experts=6, hidden_size=128, intermediate_size=64, num_cube_cores=32,
                           ep_group=object(), expert_weight_layout="EIO", architecture=3510, transport_engine="udma")
    manager = Mock()
    manager.access.side_effect = nullcontext
    parent = SimpleNamespace(spec=spec, heap_manager=manager, workspace=object())
    owners = [Mock(dedicated_bank=dedicated) for dedicated in (False, False, False, True)]
    with (patch("torch.distributed.get_rank", return_value=0),
          patch("torch.distributed.get_world_size", return_value=2),
          patch.object(MXGradientPool, "allocate") as allocate):
        for owner, slots in zip(owners, (2, 2, 1, 2)):
            owner.pool = _get_mega_bank(hidden, parent, owner, tokens=1024, received=2048, mx_guest_slots=slots)
    pool = owners[0].pool
    assert owners[1].pool is pool
    assert len({pool, owners[2].pool, owners[3].pool}) == allocate.call_count == 3
    _release_mega_bank(parent, owners[0])
    manager.close_workspace.assert_not_called()
    _release_mega_bank(parent, owners[1])
    manager.close_workspace.assert_called_once_with(pool)


def test_mx_capability_checks_cube_geometry_before_loading_replica_artifacts() -> None:
    """Converge a device limitation during binding, before a hotspot can allocate guest resources."""
    tensor = SimpleNamespace(device=SimpleNamespace(type="npu", index=3), dtype=torch.bfloat16)
    with (
        patch.object(capability, "load_torch_npu"),
        patch.object(capability, "query_npu_arch", return_value=3510),
        patch.object(capability, "validate_native_compatibility") as manifest,
        patch.object(capability._collective, "exchange", side_effect=lambda value, **_: [value]),
        pytest.raises(RuntimeError, match="HotReplica capability failed.*requires 32 Cube cores, got 24"),
    ):
        capability.require_capability(tensor, None, mx_cube_cores=24)
    manifest.assert_not_called()


def test_a5_bf16_requires_its_own_replica_artifacts(tmp_path):
    """The arch35 BF16 bank is independent of the arch22 and MX artifacts."""
    tensor = SimpleNamespace(device=SimpleNamespace(type="npu", index=0), dtype=torch.bfloat16)
    artifacts = {"soc": "ascend950", "bridge": "bridge.so", "kernels": "kernels.so",
                 "kernel_pipeline": capability.A5_BF16_KERNEL_PIPELINE_VERSION}
    selected = {"root": tmp_path, "variant": {"hot_replica_bf16_arch35": artifacts}}
    (tmp_path / artifacts["bridge"]).touch()
    with (
        patch.object(capability, "validate_native_compatibility", return_value=selected),
        patch.object(capability._collective, "exchange", side_effect=lambda value, **_: [value]),
    ):
        with pytest.raises(RuntimeError, match="Ascend950 BF16 HotReplica artifact is missing: kernels"):
            capability.require_capability(tensor, None, architecture=3510)
        (tmp_path / artifacts["kernels"]).touch()
        assert capability.require_capability(tensor, None, architecture=3510) == (True, (0,))


def test_a5_mte_peers_require_both_directions_and_refresh_after_restart():
    """A half-reachable peer cannot enter either replica protocol."""
    manager = Mock(root_group=object(), members=(4, 9, 12), epoch=0)
    rows = ((True, True, True), (True, True, True), (False, True, True))
    with (
        patch.object(torch.distributed, "get_rank", return_value=0),
        patch.object(capability, "_local_mte_peers", return_value=rows[0]) as probe,
        patch.object(capability._collective, "exchange", return_value=[(row, None) for row in rows]) as agree,
    ):
        edges = capability.a5_mte_edges(None, manager)
        assert edges == ((True, True, False), (True, True, True), (False, True, True))
        assert capability.a5_mte_edges(None, manager) is edges
        probe.assert_called_once_with(None, 0, 3)
        manager.epoch += 1
        agree.return_value = [((True, True, True), None)] * 3
        assert capability.a5_mte_edges(None, manager) == ()
        assert probe.call_count == 2


def test_a5_rejects_stale_edges_before_acquiring_bank():
    """A checkpoint or pending backward rechecks its edges after a heap restart."""
    pool = Mock(requires_mte_edges=True, local_experts=2)
    placement = SimpleNamespace(copies_by_rank=((-1,), (0,)))
    binding = ExpertWeightBinding(pool, placement, "EIO")
    with (
        patch("hyper_multicore.modules.hot_replica.shmem_transfer.a5_mte_edges",
              return_value=((True, False), (False, True))),
        pytest.raises(RuntimeError, match=r"edge \(0, 1\).*MTE"),
    ):
        binding.acquire((object(), object()), backward=True)
    pool.claim.assert_not_called()


def test_device_prefetch_does_not_save_unused_checkpoint_weights():
    pool = Mock()
    pool.pipeline.device_prefetch = True
    binding = ExpertWeightBinding(pool, Mock(), "EIO", execution_mode=ReplicaExecutionMode.FP32_BAND)
    binding.save_snapshot((object(), object()))
    assert binding.snapshot is None


def test_rank_local_tensor_dump_agrees_via_the_route_wire_and_cache():
    spec = MegaMoeSpec(16, 128, 64, 4, 1, 4, 256, 2, None, 0, 32, auto_tokens=True)
    raw = [[0, 1, 13, 1, 0, 0, 0], [0, 1, 13, 0, 0, 1, 0]]
    summary = route._summarize_host_route(raw, spec)
    cache = route._HostRouteCache((), raw, summary, None)
    capture = [raw[0].copy(), raw[1].copy()]
    capture[1][2] |= route._TENSOR_DUMP_MASK
    assert not summary.dump_weights
    assert route._summarize_host_route(capture, spec).dump_weights
    cached, _, _ = route._reuse_cached_route(cache, capture, spec, None, torch.device("cpu"), None)
    assert cached.dump_weights
    capture[1][2] ^= 1
    with pytest.raises(RuntimeError, match="autograd participation"):
        route._summarize_host_route(capture, spec)


@pytest.mark.parametrize(("mode", "transport", "manifest_pipeline", "fused"), [
    ("push", "mte", True, True), ("push", "udma", True, True), ("pull", "udma", True, True),
    ("pull", "udma", False, False), ("pull", "mte", True, False),
])
def test_a5_bf16_hard_limit_picks_active_quotas_with_either_execution(mode, transport, manifest_pipeline, fused):
    """A hard limit spills a small overload with both fused and separate replica execution."""
    planner = HotReplicaPlanner(replica_slots_per_rank=1, receive_capacity_percent=100, cross_node_cost=1.5)
    spec = MegaMoeSpec(1024, 256, 256, 12, 1, 4, 2048, 2, None, 0, 32, auto_tokens=True,
                       architecture=3510, dispatch_mode=mode, transport_engine=transport)
    settings = {"receive_capacity_percent": 100, "replica_slots_per_rank": 1, "cross_node_cost": 1.5,
                "planner_key": (type(planner).__module__, type(planner).__qualname__, planner.configuration_key())}
    parent = SimpleNamespace(spec=spec, workspace=None)
    resource = PlacementResources(parent, settings, False, a5_pipeline=manifest_pipeline, rank_nodes=(0, 1))
    assert resource.pipeline_enabled is fused
    ids = torch.tensor([0] * 544 + [6] * 480, dtype=torch.int32).view(-1, 1)
    hidden = torch.empty((1024, 256), dtype=torch.bfloat16)
    wire = torch.zeros((2, 15), dtype=torch.int32)
    wire[:, 3], wire[:, 9] = 544, 480
    summary = route._RouteSummary(1088, 1024, 1088)
    with (patch.object(resource, "bind", side_effect=lambda _, placement, **_kwargs: placement),
          patch.object(resource, "_peer_edges", return_value=()) as peers):
        placed = resource.select(hidden, ids, wire, summary, [1024, 1024], torch.arange(1024), 13,
                                 planner=planner, host_rows=wire.tolist())
        peers.return_value = ((True, False), (False, True))
        with pytest.raises(ValueError, match="bidirectional MTE"):
            resource.select(hidden, ids, wire, summary, [1024, 1024], torch.arange(1024), 13,
                            planner=planner, host_rows=wire.tolist())
    assert placed.receive_capacity == 1024
    assert placed.borrowed_capacity == 1
    assert placed.counts_by_source.sum(1).tolist() == [1024, 1024]


def test_a5_mx_hard_limit_spills_a_k64_group_before_optional_admission():
    spec = MegaMoeSpec(1024, 256, 256, 12, 1, 4, 2048, 2, None, 0, 32, auto_tokens=True,
                       architecture=3510, precision="mxfp8")
    parent = SimpleNamespace(spec=spec, workspace=None, prepare_auto_workspace=Mock(),
                             _route_spec=lambda experts: replace(spec, num_experts=experts))
    resources = mx_adapter.MXPlacementResources(parent, replica_slots_per_rank=2,
                                               receive_capacity_percent=100, rank_nodes=(0, 1))
    wire = torch.zeros((2, 15), dtype=torch.int32)
    wire[:, 3], wire[:, 9] = 544, 480
    hidden = torch.empty((1024, 256), dtype=torch.bfloat16)
    with patch.object(mx_adapter, "_render_geometry", wraps=mx_adapter._render_geometry) as render:
        geometry = resources._select(hidden, wire, route._RouteSummary(1088, 1024, 1088),
                                     wire.tolist(), [1024, 1024], mx_adapter._TRAINING_MASK, None)
    plan = render.call_args.args[0]
    plan.validate()
    assert plan.destination_loads == (1024, 1024)
    edge, = plan.transfers
    assert plan.intervals[edge.target_rank * plan.config.slots_per_rank + edge.target_slot] == (1024, 1088)
    ids = torch.tensor([0] * 544 + [6] * 480, dtype=torch.int32).view(-1, 1)
    inverse = torch.arange(1024, dtype=torch.int32)
    resources.pool = Mock(prefetch=object())
    with patch.object(route, "_permute_topk_input", return_value=(hidden, inverse)) as permute:
        prepared = resources._prepare_virtual(hidden, ids, wire, geometry, 1024, inverse)
    binding = prepared.weight_binding
    assert binding.gradients is resources.pool
    assert binding.placement.route_metadata is prepared.metadata is geometry.metadata
    assert resources.config.replica_slots_per_rank == 2
    assert binding.placement.published_slots == binding.placement.borrowed_capacity == 1
    assert binding.placement.counts_by_source.shape == (2, 14)
    assert binding.placement.virtual_ids.tolist() == [[0]] * 544 + [[7]] * 480
    assert prepared.maximum_received_slots == binding.placement.local_receive_capacity == 1024
    assert permute.call_args.args[1] is binding.placement.virtual_ids
    parent.prepare_auto_workspace.assert_called_once_with(1024, 1024, hidden)
