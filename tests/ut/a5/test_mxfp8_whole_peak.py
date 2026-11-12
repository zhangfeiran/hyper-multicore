# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Preserve the selected complete-expert receive peak during transfer pruning."""

from __future__ import annotations

import pytest

from hyper_multicore.replica import routing
from hyper_multicore.replica.capacity import ExpertReplicaConfig
from hyper_multicore.replica.recompute import create_recompute_contexts
from hyper_multicore.replica.whole import build_whole_expert_plan


def hot75_counts() -> tuple[tuple[int, ...], ...]:
    """Reproduce the device-qualified EP16/E96/4K distinct-TopK8 fixture."""
    row = [0] * 96
    for token in range(4096):
        for lane in range(8):
            expert = lane if token < 3072 else (token * 8 + lane) % 96
            row[expert] += 1
    return (tuple(row),) * 16


@pytest.mark.parametrize("budget", (1, 2, 6))
def test_hotspot_keeps_achievable_peak_and_complete_source_reductions(budget: int) -> None:
    counts = hot75_counts()
    config = ExpertReplicaConfig(96, 16, budget)
    plan = build_whole_expert_plan(counts, config, capacity_limit=393216,
                                   minimum_load_reduction=0.4, minimum_replica_rows=1024,
                                   rank_nodes=(0,) * 8 + (1,) * 8)
    assert max(plan.destination_loads) <= 58688
    assert len(plan.transfers) == 6
    assert all(edge.logical_expert < 8 for edge in plan.transfers)
    assert max(plan.replica_slots_used) <= budget
    assert plan.logical_counts == counts
    for expert in range(config.num_experts):
        destinations = {slot for row in plan.dispatch_counts for slot, value in enumerate(row)
                        if value and plan.slot_to_logical[slot // config.slots_per_rank][slot % config.slots_per_rank]
                        == expert}
        assert len(destinations) == 1
    plan.validate()


@pytest.mark.parametrize("credit", (0, 896, 393216))
def test_optional_admission_threshold_still_rejects_small_improvement(credit: int) -> None:
    counts = ((10, 10, 9, 9, 9, 9, 9, 9), (0,) * 8, (0,) * 8, (0,) * 8)
    config = ExpertReplicaConfig(8, 4, 2)
    plan = build_whole_expert_plan(counts, config, capacity_limit=100,
                                   minimum_load_reduction=0.4, minimum_replica_rows=1, transfer_row_credit=credit)
    assert plan.transfers == ()
    assert max(plan.destination_loads) == 20


@pytest.mark.parametrize("credit", (0, 896, 393216))
def test_necessary_capacity_moves_keep_the_existing_small_replica_exception(credit: int) -> None:
    counts = ((10, 10, 1, 1, 1, 1, 1, 1), (0,) * 8, (0,) * 8, (0,) * 8)
    config = ExpertReplicaConfig(8, 4, 1)
    plan = build_whole_expert_plan(counts, config, capacity_limit=12,
                                   minimum_load_reduction=0.4, minimum_replica_rows=1024, transfer_row_credit=credit)
    assert plan.transfers
    assert max(plan.destination_loads) <= 12
    assert max(plan.replica_slots_used) <= 1


@pytest.mark.parametrize("credit", (0, 896))
def test_balanced_and_budget_zero_preserve_home_only_layout(credit: int) -> None:
    balanced = ((1,) * 96,) * 16
    for budget in (0, 1, 2, 6):
        config = ExpertReplicaConfig(96, 16, budget)
        plan = build_whole_expert_plan(balanced, config, capacity_limit=393216,
                                       minimum_load_reduction=0.4, minimum_replica_rows=1024,
                                       transfer_row_credit=credit)
        assert plan.transfers == ()
        assert plan.destination_loads == (96,) * 16


def test_empty_experts_do_not_create_guest_transfers() -> None:
    config = ExpertReplicaConfig(96, 16, 2)
    plan = build_whole_expert_plan(((0,) * 96,) * 16, config, capacity_limit=393216,
                                   minimum_load_reduction=0.4, minimum_replica_rows=1024)
    assert plan.transfers == ()
    assert plan.destination_loads == (0,) * 16


def marginal_hot_counts() -> tuple[tuple[int, ...], ...]:
    """Keep the device-reproduced one-token imbalance between complete experts."""
    row = [0] * 96
    for expert in (0, 8, 23, 36, 62, 65, 78, 81):
        row[expert] = 4096
    row[5], row[23] = 1, 4095
    return (tuple(row),) * 16


@pytest.mark.parametrize("budget", (1, 2, 6))
def test_gradient_credit_removes_a_matrix_for_only_sixteen_extra_rows(budget: int) -> None:
    config = ExpertReplicaConfig(96, 16, budget)
    plan = build_whole_expert_plan(
        marginal_hot_counts(), config, capacity_limit=393216, minimum_load_reduction=0.4,
        minimum_replica_rows=1024, transfer_row_credit=896, rank_nodes=(0,) * 8 + (1,) * 8)
    assert max(plan.destination_loads) == 65552
    assert len(plan.transfers) == 2
    plan.validate()


@pytest.mark.parametrize("budget", (1, 2, 6))
def test_gradient_credit_keeps_the_large_hotspot_improvement(budget: int) -> None:
    config = ExpertReplicaConfig(96, 16, budget)
    plan = build_whole_expert_plan(
        hot75_counts(), config, capacity_limit=393216, minimum_load_reduction=0.4,
        minimum_replica_rows=1024, transfer_row_credit=896, rank_nodes=(0,) * 8 + (1,) * 8)
    assert max(plan.destination_loads) == 58688
    assert len(plan.transfers) == 6
    assert all(edge.logical_expert < 8 for edge in plan.transfers)


@pytest.mark.parametrize("credit", (-1, True))
def test_complete_expert_credit_rejects_invalid_values(credit: int) -> None:
    with pytest.raises(ValueError, match="Replica transfer row credit"):
        build_whole_expert_plan(
            marginal_hot_counts(), ExpertReplicaConfig(96, 16, 2), capacity_limit=393216,
            minimum_load_reduction=0.4, transfer_row_credit=credit)


def test_complete_route_uses_the_existing_gradient_transfer_credit() -> None:
    plan = routing._build_route_plan(
        marginal_hot_counts(), ExpertReplicaConfig(96, 16, 2), True,
        {"capacity_limit": 393216, "minimum_load_reduction": 0.4, "minimum_replica_rows": 1024,
         "_replica_transfer_row_credit": 896, "_rank_nodes": (0,) * 8 + (1,) * 8})
    assert max(plan.destination_loads) == 65552
    assert len(plan.transfers) == 2


@pytest.mark.parametrize("forward_credit,replay_credit", ((0, 896), (896, 0)))
def test_replay_keeps_the_original_plan_when_gradient_credit_changes(
    forward_credit: int, replay_credit: int,
) -> None:
    config = ExpertReplicaConfig(96, 16, 2)
    group = object()
    options = {"capacity_limit": 393216, "minimum_load_reduction": 0.4, "minimum_replica_rows": 1024,
               "_rank_nodes": (0,) * 8 + (1,) * 8}
    forward, replay = create_recompute_contexts()
    with forward:
        initial, reused = routing._plan_replica_route(
            marginal_hot_counts(), config, group, 0, _complete_experts=True,
            _replica_transfer_row_credit=forward_credit, **options)
    assert not reused
    with replay:
        repeated, reused = routing._plan_replica_route(
            marginal_hot_counts(), config, group, 0, _complete_experts=True,
            _replica_transfer_row_credit=replay_credit, **options)
    assert reused
    assert repeated is initial
    assert len(initial.transfers) == (2 if forward_credit else 3)
    assert max(initial.destination_loads) == (65552 if forward_credit else 65536)
