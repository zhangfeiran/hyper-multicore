# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Preserve the selected complete-expert receive peak during transfer pruning."""

from __future__ import annotations

import pytest

from hyper_multicore.replica.capacity import ExpertReplicaConfig
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
    assert max(plan.replica_slots_used) <= budget
    assert plan.logical_counts == counts
    for expert in range(config.num_experts):
        destinations = {slot for row in plan.dispatch_counts for slot, value in enumerate(row)
                        if value and plan.slot_to_logical[slot // config.slots_per_rank][slot % config.slots_per_rank]
                        == expert}
        assert len(destinations) == 1
    plan.validate()


def test_optional_admission_threshold_still_rejects_small_improvement() -> None:
    counts = ((10, 10, 9, 9, 9, 9, 9, 9), (0,) * 8, (0,) * 8, (0,) * 8)
    config = ExpertReplicaConfig(8, 4, 2)
    plan = build_whole_expert_plan(counts, config, capacity_limit=100,
                                   minimum_load_reduction=0.4, minimum_replica_rows=1)
    assert plan.transfers == ()
    assert max(plan.destination_loads) == 20


def test_necessary_capacity_moves_keep_the_existing_small_replica_exception() -> None:
    counts = ((10, 10, 1, 1, 1, 1, 1, 1), (0,) * 8, (0,) * 8, (0,) * 8)
    config = ExpertReplicaConfig(8, 4, 1)
    plan = build_whole_expert_plan(counts, config, capacity_limit=12,
                                   minimum_load_reduction=0.4, minimum_replica_rows=1024)
    assert plan.transfers
    assert max(plan.destination_loads) <= 12
    assert max(plan.replica_slots_used) <= 1


def test_balanced_and_budget_zero_preserve_home_only_layout() -> None:
    balanced = ((1,) * 96,) * 16
    for budget in (0, 1, 2, 6):
        config = ExpertReplicaConfig(96, 16, budget)
        plan = build_whole_expert_plan(balanced, config, capacity_limit=393216,
                                       minimum_load_reduction=0.4, minimum_replica_rows=1024)
        assert plan.transfers == ()
        assert plan.destination_loads == (96,) * 16
