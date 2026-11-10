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
"""CPU-only plans and reductions for isolated replica contention measurements."""

from dataclasses import dataclass
import math
import statistics
from typing import Any

from hyper_multicore.replica.capacity import ExpertReplicaConfig
from hyper_multicore.replica.plan import ExpertExecutionPlan
from hyper_multicore.replica.planner import _materialize


@dataclass(frozen=True)
class ContentionCase:
    """Disjoint owner and guest roles with one reserved guest slot per rank."""

    fanout: int
    placement: str
    owners: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]

    @property
    def name(self) -> str:
        """Return a stable output key."""
        return f'owners{len(self.owners)}-fanout{self.fanout}-{self.placement}'

    def role(self, rank: int) -> str:
        """Return a mutually exclusive rank role."""
        if rank in self.owners:
            return 'owner'
        return 'guest' if any(target == rank for _, target in self.edges) else 'idle'


def make_case(nodes: tuple[int, ...], fanout: int, owner_count: int, placement: str) -> ContentionCase:
    """Allocate guests by actual node identity, balancing inter-node targets in rounds.

    Owners span distinct nodes first. Mixed uses alternating local/remote edges,
    starting locally. Owner ranks never act as guests and guests are unique.
    """
    if fanout < 1 or owner_count < 1 or not nodes or placement not in ('intra', 'inter', 'mixed'):
        raise ValueError('Positive fanout/owner count, nonempty topology and a known placement are required')
    if placement == 'mixed' and fanout < 2:
        raise ValueError('Mixed placement needs fanout >= 2 to include both node kinds')
    labels = tuple(dict.fromkeys(nodes))
    members = {node: [rank for rank, label in enumerate(nodes) if label == node] for node in labels}
    ordered = [members[node][depth] for depth in range(max(map(len, members.values())))
               for node in labels if depth < len(members[node])]
    if owner_count * (fanout + 1) > len(nodes):
        raise ValueError('Not enough ranks for disjoint owners and unique B1 guests')
    owners = tuple(ordered[:owner_count])
    used = set(owners)
    edges = []
    for step in range(fanout):
        local = placement == 'intra' or (placement == 'mixed' and step % 2 == 0)
        for owner in owners:
            node = nodes[owner]
            offset = labels.index(node)
            peers = (node,) if local else labels[offset + 1:] + labels[:offset]
            candidates = [members[peer][depth] for depth in range(max(map(len, members.values())))
                          for peer in peers if depth < len(members[peer])]
            target = next((rank for rank in candidates if rank not in used), None)
            if target is None:
                raise ValueError(f'Insufficient {"local" if local else "remote"} guests for owner {owner}')
            used.add(target)
            edges.append((owner, target))
    return ContentionCase(fanout, placement, owners, tuple(edges))


def case_plan(case: ContentionCase, size: int, home: int) -> ExpertExecutionPlan:
    """Force every edge to copy one complete expert; token quotas only encode ownership."""
    if home < 1 or size < 2:
        raise ValueError('Positive home experts and EP size >= 2 are required')
    if (len(set(case.owners)) != len(case.owners) or len(case.edges) != len(case.owners) * case.fanout
            or any(sum(source == owner for source, _ in case.edges) != case.fanout for owner in case.owners)
            or any(source not in case.owners or target in case.owners for source, target in case.edges)
            or len({target for _, target in case.edges}) != len(case.edges)
            or any(not 0 <= rank < size for rank in case.owners + tuple(target for _, target in case.edges))):
        raise ValueError('Invalid disjoint owner/guest edges or fanout')
    config = ExpertReplicaConfig(size * home, size, 1)
    counts = [[0] * config.num_experts for _ in range(size)]
    copies = [{expert: 0 for expert in range(rank * home, (rank + 1) * home)} for rank in range(size)]
    for owner, target in case.edges:
        expert = owner * home
        counts[owner][expert] += 1
        copies[target][expert] = 1
    plan = _materialize(tuple(map(tuple, counts)), copies, config)
    plan.validate()
    if len(plan.transfers) != len(case.edges):
        raise ValueError('Materialized transfers do not match requested fanout')
    return plan


def summarize_ranks(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Take rank maxima per iteration first, then medians without trimming samples."""
    if not records or sorted(row['ep_rank'] for row in records) != list(range(len(records))):
        raise ValueError('Every EP rank must occur exactly once')
    count = len(records[0]['samples'])
    metrics = ('weight_ms', 'physical_gradient_ms', 'weight_wall_ms', 'gradient_wall_ms')
    if (not count or any(len(row['samples']) != count or row['role'] not in ('owner', 'guest', 'idle')
                         or not row['checks_passed'] for row in records)):
        raise ValueError('Incomplete samples, invalid roles or failed matrix checks')
    if any(not math.isfinite(sample[key]) or sample[key] < 0
           for row in records for sample in row['samples'] for key in metrics):
        raise ValueError('Timing samples must be finite and nonnegative')
    groups = {'all': records, 'active': [row for row in records if row['role'] != 'idle']}
    groups.update({role: [row for row in records if row['role'] == role] for role in ('owner', 'guest', 'idle')})
    samples = []
    for index in range(count):
        maxima = {key: {metric: max(row['samples'][index][metric] for row in rows) for metric in metrics}
                  for key, rows in groups.items() if rows}
        critical = {metric: [row['global_rank'] for row in records
                            if row['samples'][index][metric] == maxima['all'][metric]] for metric in metrics}
        samples.append(dict(iteration=index, maxima=maxima, critical_global_ranks=critical))
    medians = {key: {metric: statistics.median(sample['maxima'][key][metric] for sample in samples)
                     for metric in metrics} for key, rows in groups.items() if rows}
    return dict(samples=samples, median_of_iteration_rank_max_ms=medians,
                role_counts={role: len(groups[role]) for role in ('owner', 'guest', 'idle')}, trimming=False)
