# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
"""Independent global row-order and quantization-group invariants for MX fragments."""
import random

import pytest
import torch

from hyper_multicore.replica.aligned import build_aligned_expert_plan
from hyper_multicore.replica.capacity import ExpertReplicaConfig
from hyper_multicore.replica._a5_routing import _remap_replica_ids, _plan_replica_route
from hyper_multicore.replica.recompute import create_recompute_contexts


def assert_actual_permutation(plan):
    """Apply the production remapper then reconstruct destination source concatenation."""
    received = {slot: [] for slot in range(plan.config.physical_experts)}
    originals = {e: [] for e in range(plan.config.num_experts)}
    for source, counts in enumerate(plan.logical_counts):
        # Interleave logical experts instead of handing the remapper already sorted data.
        pairs = [(e, i) for e, n in enumerate(counts) for i in range(n)]
        random.Random(17 + source).shuffle(pairs)
        ids = torch.tensor([e for e, _ in pairs], dtype=torch.int64).reshape(-1, 1)
        runs = plan.source_runs_for(source)
        physical = _remap_replica_ids(ids, plan, torch.tensor([s for s, _ in runs]),
                                      torch.tensor([n for _, n in runs])).flatten().tolist()
        for index, ((expert, _), slot) in enumerate(zip(pairs, physical)):
            token = (source, index)
            originals[expert].append(token)
            received[slot].append(token)
    for slot, expert in enumerate(plan.physical_to_logical):
        if expert < 0:
            assert received[slot] == []
            continue
        start, stop = plan.intervals[slot]
        assert received[slot] == originals[expert][start:stop]
        # All complete 32-row groups and 64-row scale blocks retain their original rows.
        for group in (32, 64):
            for pos in range(0, stop - start, group):
                assert received[slot][pos:pos + group] == originals[expert][start + pos:min(stop, start + pos + group)]


def test_cross_source_cut_matches_documented_example():
    plan = build_aligned_expert_plan(((50, 0, 0), (90, 0, 0), (116, 0, 0)),
                                     ExpertReplicaConfig(3, 3, 1), capacity_limit=1000)
    edge, = plan.transfers
    slot = edge.target_rank * plan.config.slots_per_rank + edge.target_slot
    assert plan.intervals[slot] == (128, 256)
    assert [row[slot] for row in plan.dispatch_counts] == [0, 12, 116]
    assert_actual_permutation(plan)


@pytest.mark.parametrize('tail', (0, 1, 31, 32, 63, 64, 65, 174))
@pytest.mark.parametrize('owner', (0, 3))
def test_tails_and_guest_before_owner_preserve_global_order(tail, owner):
    rows = [[0] * 8 for _ in range(4)]
    for source in range(4):
        rows[source][owner * 2] = 128 + (tail if source == 3 else 0)
        rows[source][(owner * 2 + 3) % 8] = source + 1
    plan = build_aligned_expert_plan(rows, ExpertReplicaConfig(8, 4, 1), capacity_limit=2000)
    plan.validate()
    assert max(plan.replica_slots_used) <= 1
    assert_actual_permutation(plan)


def test_zero_budget_balanced_and_threshold_fall_back():
    for budget in (0, 1, 2):
        plan = build_aligned_expert_plan(((7,) * 8,) * 4, ExpertReplicaConfig(8, 4, budget),
                                         capacity_limit=1000, minimum_load_reduction=.25)
        assert not plan.transfers
        assert_actual_permutation(plan)


def test_checkpoint_replay_keeps_partition_and_rejects_changed_policy():
    counts = ((100, 0, 0, 0),) * 4
    config = ExpertReplicaConfig(4, 4, 1)
    options = dict(_complete_experts=True, _aligned_experts=True, capacity_limit=1000,
                   minimum_replica_rows=64)
    fwd, replay = create_recompute_contexts()
    with fwd:
        plan, reused = _plan_replica_route(counts, config, None, 0, **options)
    with replay:
        restored, reused = _plan_replica_route(counts, config, None, 0, **options)
    assert restored is plan and reused
    assert_actual_permutation(plan)


def test_random_exact_conservation_and_order():
    rng = random.Random(950)
    for _ in range(40):
        counts = tuple(tuple(rng.randrange(120) if e < 3 else rng.randrange(5)
                             for e in range(12)) for _ in range(4))
        plan = build_aligned_expert_plan(counts, ExpertReplicaConfig(12, 4, 2), capacity_limit=10000)
        assert_actual_permutation(plan)


def test_equivalent_whole_experts_preferred_with_same_transfer_count():
    rows=[]
    for rank in range(8):
        counts=[0]*48
        for token in range(4096):
            for lane in range(8):
                expert=lane if token<3072 else (token*8+lane+rank*4096*8)%48
                counts[expert]+=1
        rows.append(counts)
    plan=build_aligned_expert_plan(rows,ExpertReplicaConfig(48,8,2),capacity_limit=262144,
        minimum_replica_rows=1024,minimum_load_reduction=.25,transfer_row_credit=896)
    for expert in range(48):
        assert sum(stop>start for slot,(start,stop) in enumerate(plan.intervals)
                   if plan.physical_to_logical[slot]==expert)==1
    assert len(plan.transfers)==6


def test_equal_load_fragments_prefer_the_owners_node():
    plan=build_aligned_expert_plan(((0,0,0,256),)*4,ExpertReplicaConfig(4,4,1),capacity_limit=4096,
                                   rank_nodes=(1,0,1,0))
    assert len(plan.transfers)==1
    assert plan.transfers[0].owner_rank==3 and plan.transfers[0].target_rank==1
    assert_actual_permutation(plan)


def test_required_capacity_moves_override_optional_guest_threshold():
    plan=build_aligned_expert_plan(((6,6,0,0),(0,0,0,0)),ExpertReplicaConfig(4,2,1),
        capacity_limit=6,minimum_replica_rows=10000,minimum_load_reduction=.9)
    assert max(plan.destination_loads)==6 and len(plan.transfers)==1
    assert_actual_permutation(plan)
