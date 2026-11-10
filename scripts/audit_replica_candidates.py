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
"""Reproduce capacity-pruned feasible replica candidates in a bounded CPU audit.

This diagnostic changes only an intermediate search bound in its labeled probe.
It never patches production planner code, changes coefficients or launches NPUs.
"""

from dataclasses import asdict, replace
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from typing import Any

from hyper_multicore.replica import cost, planner
from hyper_multicore.replica.capacity import ExpertReplicaConfig
from hyper_multicore.replica.cost import ExpertReplicaCostModel
from hyper_multicore.replica.plan import ExpertExecutionPlan


def plan_digest(plan: ExpertExecutionPlan) -> str:
    """Hash all immutable physical plan fields."""
    return hashlib.sha256(json.dumps(asdict(plan), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_fixture(path: Path) -> tuple[dict[str, Any], ExpertExecutionPlan]:
    """Reject corrupted fixtures or an original plan that the reference cannot reproduce."""
    data = json.loads(path.read_text())
    if data['schema_version'] != 1:
        raise ValueError('Unsupported actual-plan fixture schema')
    saved, decision = data['plan'], data['original_decision']
    plan = ExpertExecutionPlan(ExpertReplicaConfig(**saved['config']), saved['logical_counts'],
                               saved['slot_to_logical'], saved['dispatch_counts'])
    plan.validate()
    if plan_digest(plan) != data['plan_sha256']:
        raise ValueError('Actual-plan checksum mismatch')
    if decision.get('model') is None:
        raise ValueError('Actual capture has no cost model; load-only decisions are outside this scoring audit')
    options = dict(decision['options'])
    options['_rank_nodes'] = tuple(options['_rank_nodes'])
    original = planner._build_expert_replica_plan(plan.logical_counts, plan.config.replica_slots_per_rank,
        cost_model=ExpertReplicaCostModel(**decision['model']), **options)
    if original.slot_to_logical != plan.slot_to_logical or original.dispatch_counts != plan.dispatch_counts:
        raise ValueError('Original saved plan did not reproduce')
    return data, plan


def quotas(plan: ExpertExecutionPlan) -> list[dict[int, int]]:
    """Recover exact destination quotas without changing physical placement."""
    return [{expert: count for expert, count in zip(slots, counts) if expert >= 0}
            for slots, counts in zip(plan.slot_to_logical, plan.destination_counts)]


def quota_digest(copies: list[dict[int, int]], home: int) -> str:
    """Match layouts only; this order-insensitive digest never memoizes scores."""
    normalized = [sorted((expert, rows) for expert, rows in row.items() if rows or expert // home == rank)
                  for rank, row in enumerate(copies)]
    return hashlib.sha256(json.dumps(normalized, separators=(',', ':')).encode()).hexdigest()


def promoted_plan(source: ExpertExecutionPlan, target: ExpertReplicaConfig) -> ExpertExecutionPlan:
    """Pad inactive slots while preserving every source-to-destination dispatch."""
    if (source.config.num_experts != target.num_experts or source.config.ep_size != target.ep_size
            or source.config.slots_per_rank > target.slots_per_rank):
        raise ValueError("Promotion requires the same logical topology and at least as many target slots")
    before, after = source.config.slots_per_rank, target.slots_per_rank
    slots = [tuple(row) + (-1,) * (after - before) for row in source.slot_to_logical]
    dispatch = []
    for row in source.dispatch_counts:
        expanded = []
        for rank in range(target.ep_size):
            expanded.extend(row[rank * before:(rank + 1) * before])
            expanded.extend([0] * (after - before))
        dispatch.append(expanded)
    result = ExpertExecutionPlan(target, source.logical_counts, slots, dispatch)
    result.validate()
    for old, new in zip(source.dispatch_counts, result.dispatch_counts):
        if any(tuple(old[rank * before:(rank + 1) * before]) !=
               tuple(new[rank * after:rank * after + before]) for rank in range(target.ep_size)):
            raise ValueError('Promotion redistributed actual source dispatch')
    return result


def traced_replay(counts: tuple[tuple[int, ...], ...], config: ExpertReplicaConfig,
                  model: ExpertReplicaCostModel, options: dict[str, Any]) -> tuple[ExpertExecutionPlan, dict[str, Any]]:
    """Trace bounded candidates in an isolated process without changing planner semantics."""
    context = dict(phase='outside', budget=None)
    thin, scores = [], []
    original_bounded = planner._bounded_copies
    original_adaptive = planner._adaptive_copies
    original_score = cost._QuotaCostState.score

    def bounded(*args: Any, **kwargs: Any) -> list[dict[int, int]]:
        result = original_bounded(*args, **kwargs)
        candidate = args[2]
        context['budget'] = candidate.replica_slots_per_rank
        peak = max(sum(row.values()) for row in result)
        thin.append(dict(budget=context['budget'], peak_rows=peak,
                         exceeds_hard_limit=peak > options['capacity_limit'],
                         layout_hash=quota_digest(result, config.home_experts)))
        return result

    def adaptive(*args: Any, **kwargs: Any) -> tuple:
        context['phase'] = 'adaptive'
        result = original_adaptive(*args, **kwargs)
        context['phase'] = 'outside'
        return result

    def score(state: Any, *args: Any, **kwargs: Any) -> float:
        result = original_score(state, *args, **kwargs)
        if context['phase'] == 'adaptive':
            scores.append(dict(budget=context['budget'], value_ms=result,
                peak_rows=max(sum(row.values()) for row in state.copies),
                edges=sum(expert // config.home_experts != rank and rows > 0
                          for rank, row in enumerate(state.copies) for expert, rows in row.items()),
                layout_hash=quota_digest(state.copies, config.home_experts)))
        return result

    with (patch.object(planner, '_bounded_copies', bounded),
          patch.object(planner, '_adaptive_copies', adaptive),
          patch.object(cost._QuotaCostState, 'score', score)):
        result = planner._build_expert_replica_plan(counts, config.replica_slots_per_rank,
                                                   cost_model=model, **options)
    result.validate()
    return result, dict(thin_candidates=thin, scored_candidates=scores,
                       selected_hash=quota_digest(quotas(result), config.home_experts),
                       selected_score_ms=model.estimate_ms(result), selected_edges=len(result.transfers),
                       selected_peak_rows=max(result.destination_loads))

def main() -> None:
    """Audit original middle/hot fixtures; do not change coefficients or execute a candidate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = []
    for pattern in ('middle', 'hot'):
        f1, p1 = load_fixture(args.fixtures / f'push-b1-{pattern}.json')
        f6, p6 = load_fixture(args.fixtures / f'push-b6-{pattern}.json')
        if p1.logical_counts != p6.logical_counts:
            raise ValueError('Actual B1/B6 source histograms differ')
        options1 = f1['original_decision']['options']
        options6 = dict(f6['original_decision']['options'])
        options6['_rank_nodes'] = tuple(options6['_rank_nodes'])
        model1 = replace(ExpertReplicaCostModel(**f1['original_decision']['model']),
                         _rank_nodes=tuple(options1['_rank_nodes']))
        model6 = replace(ExpertReplicaCostModel(**f6['original_decision']['model']),
                         _rank_nodes=options6['_rank_nodes'])
        if model1 != model6:
            raise ValueError('Actual B1/B6 effective models differ; the scoring comparison is not controlled')
        policy1 = dict(options1, _rank_nodes=tuple(options1['_rank_nodes']))
        if any(policy1[key] != options6[key] for key in policy1 if key != 'capacity_limit'):
            raise ValueError('Actual B1/B6 policies differ beyond the recorded theoretical hard bound')
        if model6.critical_path is not None or model1.critical_path is not None:
            raise ValueError('This bounded audit targets the captured legacy cost path only')
        promoted = promoted_plan(p1, p6.config)
        if max(promoted.destination_loads) > options6['capacity_limit']:
            raise ValueError('Promoted original B1 layout violates B6 hard capacity')
        if max(promoted.destination_loads) > options6['receive_capacity']:
            raise ValueError('Promoted original B1 layout requires extra receive memory')
        original, observed = traced_replay(p6.logical_counts, p6.config, model6, options6)
        if plan_digest(original) != plan_digest(p6):
            raise ValueError('Original B6 runtime plan did not reproduce')
        search_options = dict(options6, capacity_limit=options1['capacity_limit'])
        expanded, probe = traced_replay(p6.logical_counts, p6.config, model6, search_options)
        target_hash = quota_digest(quotas(promoted), p6.config.home_experts)
        result = dict(pattern=pattern, counts_equal=True, effective_models_equal=model1 == model6,
            runtime_model='legacy v2 calibration with critical_path=None',
            cost_disabled=False, source_reference=str(planner.__file__),
            option_differences={key:dict(b1=options1.get(key), b6=f6['original_decision']['options'].get(key))
                                for key in set(options1) | set(options6)
                                if options1.get(key) != f6['original_decision']['options'].get(key)},
            original_model=asdict(model6), original_config=asdict(p6.config),
            original_model_score_ms=model6.estimate_ms(p6),
            promoted_b1=dict(config=asdict(promoted.config), plan_sha256=plan_digest(promoted),
                edges=len(promoted.transfers), peak_rows=max(promoted.destination_loads),
                model_score_ms=model6.estimate_ms(promoted), layout_hash=target_hash,
                source_dispatch_preserved=True, legal_under_original_hard_and_receive_limits=True,
                reached_original_scoring=any(row['layout_hash'] == target_hash
                                            for row in observed['scored_candidates'])),
            original_search=observed,
            intermediate_bound_probe=dict(search_hard_limit=search_options['capacity_limit'],
                original_receive_limit_unchanged=search_options['receive_capacity'],
                result_fits_original_hard_limit=max(expanded.destination_loads) <= options6['capacity_limit'],
                promoted_layout_scored=any(row['layout_hash'] == target_hash for row in probe['scored_candidates']),
                **probe),
            scope='Bounded CPU candidate audit. Only intermediate search bound changed in probe; '
                  'no coefficients or NPU run')
        results.append(result)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    for row in results:
        print(json.dumps(dict(pattern=row['pattern'], models_equal=row['effective_models_equal'],
            original_score_ms=row['original_model_score_ms'], promoted=row['promoted_b1'],
            original_thin=row['original_search']['thin_candidates'],
            probe_edges=row['intermediate_bound_probe']['selected_edges'],
            probe_score_ms=row['intermediate_bound_probe']['selected_score_ms'],
            probe_promoted_scored=row['intermediate_bound_probe']['promoted_layout_scored'])))


if __name__ == '__main__':
    main()
