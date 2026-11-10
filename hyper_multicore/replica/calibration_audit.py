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
"""Read-only calibration audits, exact plan replay and disjoint-group service checks.

Provider service observations are deliberately separate from a complete device
critical-path fit. This module never writes installed calibration tables.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any

from .capacity import ExpertReplicaConfig
from .cost import ExpertReplicaCostModel
from .critical_path import execution_fingerprint
from .plan import ExpertExecutionPlan
from .planner import _build_expert_replica_plan


STAGES = ('links-push', 'links-pull', 'contention-push', 'contention-pull', 'components')


class SampleRejected(ValueError):
    """Keep a machine-readable reason for an unusable or mismatched observation."""


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise SampleRejected(f'invalid_service:{name}')
    return float(value)


def _signature(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
        raise SampleRejected(f'invalid_signature:{name}')
    return value


def _require(record: dict[str, Any], names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in record or record[name] is None]
    if missing:
        raise SampleRejected('missing_fields:' + ','.join(missing))


def _p95(values: list[float]) -> float | None:
    return sorted(values)[math.ceil(.95 * len(values)) - 1] if values else None


def _error_summary(prediction: float, values: list[float]) -> dict[str, Any]:
    errors = [prediction - value for value in values]
    return dict(samples=len(values), median_observed_ms=statistics.median(values),
                median_signed_error_ms=statistics.median(errors),
                median_absolute_error_ms=statistics.median(abs(value) for value in errors),
                p95_absolute_error_ms=_p95([abs(value) for value in errors]),
                median_relative_error=statistics.median(abs(prediction - value) / value for value in values),
                observations_ms=values)


def fit_provider_services(payload: dict[str, Any]) -> dict[str, Any]:
    """Fit physical services on one EP group and validate on disjoint groups.

    The observed-domain signature binds the available adapter/shape/protocol
    evidence. It does not claim complete vendor, host or execution identity.
    Raw slow samples and signed errors remain in the returned holdout records.
    """
    _require(payload, ('configuration', 'reports', 'measurement'))
    config, reports = payload['configuration'], payload['reports']
    keys = ('backend', 'replica_transport', 'ep_size', 'hidden', 'intermediate', 'home_experts',
            'reference_top_k', 'iterations', 'warmup')
    _require(config, keys)
    if len(reports) < 2:
        raise SampleRejected('holdout_requires_disjoint_ep_groups')
    identity = []
    expected_bytes = 3 * config['hidden'] * config['intermediate'] * 2
    records = []
    membership: set[int] = set()
    for report in sorted(reports, key=lambda row: row['global_leader']):
        _require(report, ('global_leader', 'global_ep_members', 'rank_nodes', 'soc_name', 'torch_npu',
                          'native_adapter_sha256', 'pairs'))
        _signature(report['native_adapter_sha256'], 'native_adapter_sha256')
        members, nodes = report['global_ep_members'], report['rank_nodes']
        if len(set(members)) != config['ep_size'] or len(nodes) != len(members) or membership.intersection(members):
            raise SampleRejected('invalid_or_overlapping_ep_groups')
        membership.update(members)
        identity.append((report['soc_name'], report['torch_npu'], report['native_adapter_sha256']))
        for pair in report['pairs']:
            owner, target = pair['owner'], pair['target']
            if not 0 <= owner < len(nodes) or not 0 <= target < len(nodes) or owner == target:
                raise SampleRejected('invalid_link_peer')
            kind = 'intra_node' if nodes[owner] == nodes[target] else 'inter_node'
            if pair['kind'] != kind:
                raise SampleRejected('link_topology_mismatch')
            if pair['bf16_weight_bytes'] != expected_bytes or pair['fp32_gradient_bytes'] != expected_bytes * 2:
                raise SampleRejected('physical_matrix_bytes_mismatch')
            if len(pair['samples']) != config['iterations']:
                raise SampleRejected('incomplete_service_samples')
            for index, sample in enumerate(pair['samples']):
                _require(sample, ('weight_ms', 'physical_gradient_ms', 'weight_wall_ms', 'gradient_wall_ms'))
                for field in ('weight_ms', 'physical_gradient_ms', 'weight_wall_ms', 'gradient_wall_ms'):
                    _positive(sample[field], field)
                records.append(dict(group=report['global_leader'], kind=kind, owner=owner, target=target,
                                    sample_index=index, **sample))
    if len(set(identity)) != 1:
        raise SampleRejected('service_identity_mismatch')
    contract = {name: config[name] for name in keys if name not in ('iterations', 'warmup')}
    contract.update(soc_name=identity[0][0], torch_npu=identity[0][1], native_adapter_sha256=identity[0][2],
                    bf16_weight_bytes=expected_bytes, fp32_gradient_bytes=expected_bytes * 2,
                    measurement=payload['measurement'])
    train = min(record['group'] for record in records)
    domains = {}
    for kind in ('intra_node', 'inter_node'):
        training = [row for row in records if row['group'] == train and row['kind'] == kind]
        heldout = [row for row in records if row['group'] != train and row['kind'] == kind]
        if not training or not heldout:
            raise SampleRejected(f'missing_domain_holdout:{kind}')
        services = {}
        for name in ('weight_ms', 'physical_gradient_ms'):
            estimate = statistics.median(row[name] for row in training)
            services[name] = dict(candidate_ms=estimate,
                                  heldout=_error_summary(estimate, [row[name] for row in heldout]))
        domains[kind] = dict(train_samples=len(training), heldout_samples=len(heldout), services=services)
    return dict(observed_domain_signature=execution_fingerprint(contract), observed_contract=contract,
                train_group=train, heldout_groups=sorted({row['group'] for row in records if row['group'] != train}),
                domains=domains, raw_records=records, installable=False, complete_execution_binding=False,
                missing_execution_binding=['runtime_vendor_content_hash', 'host_path_identity', 'residency_identity',
                                           'matched_component_execution_signature'],
                scope='Physical provider services only; not a fitted fused-kernel residual or critical-path profile')


def replay_decision(record: dict[str, Any], signature: str, expected_signature: str) -> dict[str, Any]:
    """Validate conservation/capacity and compare a saved plan with the exact CPU ref."""
    _signature(signature, 'execution_signature')
    if signature != expected_signature:
        raise SampleRejected('execution_signature_mismatch')
    _require(record, ('model', 'options', 'config', 'source_counts', 'slots', 'dispatch', 'calls', 'plan_reuses'))
    config = ExpertReplicaConfig(**record['config'])
    model = ExpertReplicaCostModel(**record['model'])
    if config.ep_size != model.ep_size:
        raise SampleRejected('plan_model_shape_mismatch')
    if model.critical_path is not None and model.critical_path.execution_signature != signature:
        raise SampleRejected('model_execution_signature_mismatch')
    options = dict(record['options'])
    _require(options, ('capacity_limit', '_rank_nodes', 'adaptive'))
    options['_rank_nodes'] = tuple(options['_rank_nodes'])
    if len(options['_rank_nodes']) != config.ep_size:
        raise SampleRejected('missing_topology')
    if model._rank_nodes and model._rank_nodes != options['_rank_nodes']:
        raise SampleRejected('model_topology_mismatch')
    plan = ExpertExecutionPlan(config, record['source_counts'], record['slots'], record['dispatch'])
    try:
        plan.validate()
    except ValueError as error:
        raise SampleRejected(f'invalid_plan:{error}') from error
    if max(plan.destination_loads) > options['capacity_limit']:
        raise SampleRejected('hard_capacity_violation')
    reproduced = _build_expert_replica_plan(plan.logical_counts, config.replica_slots_per_rank,
                                           cost_model=model, **options)
    matched = reproduced.slot_to_logical == plan.slot_to_logical and reproduced.dispatch_counts == plan.dispatch_counts
    replay_only = record['calls'] == record['plan_reuses']
    return dict(valid=True, exact_fresh_replay=matched, recompute_only=replay_only,
                reason=None if matched else ('recompute_only_fresh_mismatch' if replay_only else 'unreproduced_plan'),
                transfers=len(plan.transfers), peak_rows=max(plan.destination_loads),
                counts_signature=execution_fingerprint({'counts': plan.logical_counts}),
                eligible_for_fit=matched and not replay_only)


@dataclass(frozen=True)
class FitContract:
    """Require observed native, host, residency and schedule identity before fitting."""

    execution_signature: str
    native_payload_sha256: str
    host_signature: str
    residency_signature: str
    schedule_contract: str
    observation_mode: str = 'off'


def validate_fit_sample(sample: dict[str, Any], expected: FitContract) -> None:
    """Reject incomplete observations instead of silently filling synthetic values."""
    _require(sample, ('execution', 'decision', 'measurement'))
    identity = sample['execution']
    for name in ('execution_signature', 'native_payload_sha256', 'host_signature', 'residency_signature'):
        _require(identity, (name,))
        if _signature(identity[name], name) != getattr(expected, name):
            raise SampleRejected(f'{name}_mismatch')
    if identity.get('schedule_contract') != expected.schedule_contract:
        raise SampleRejected('schedule_contract_mismatch')
    if identity.get('observation_mode') != expected.observation_mode:
        raise SampleRejected('observation_mode_mismatch')
    if not identity.get('schedule_equivalence_verified', False):
        raise SampleRejected('schedule_equivalence_unverified')
    measurement = sample['measurement']
    _require(measurement, ('forward_ms', 'recompute_ms', 'backward_ms', 'host_exposed_ms', 'host_wall_ms',
                           'profile_on', 'clock_domain'))
    if measurement['profile_on'] is not False:
        raise SampleRejected('profiled_sample_not_clean_holdout')
    if measurement['clock_domain'] != 'rank_local_stream_elapsed':
        raise SampleRejected('unaligned_clock_domain')
    for name in ('forward_ms', 'backward_ms', 'host_wall_ms'):
        _positive(measurement[name], name)
    for name in ('recompute_ms', 'host_exposed_ms'):
        value = measurement[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise SampleRejected(f'invalid_duration:{name}')
    if measurement['host_exposed_ms'] > measurement['host_wall_ms']:
        raise SampleRejected('exposed_host_exceeds_wall')
    replay = replay_decision(sample['decision'], expected.execution_signature, expected.execution_signature)
    if not replay['eligible_for_fit']:
        raise SampleRejected(replay['reason'] or 'recompute_only_sample')


def decision_metrics(samples: list[dict[str, Any]], expected_signature: str, margin_ms: float = 0.) -> dict[str, Any]:
    """Keep wrong-direction, false-admission and unknown samples in a held-out ledger."""
    if not math.isfinite(margin_ms) or margin_ms < 0:
        raise ValueError('margin_ms must be finite and nonnegative')
    rows, unknown, rejected = [], [], []
    for sample in samples:
        if sample.get('execution_signature') != expected_signature:
            rejected.append(dict(id=sample.get('id'), reason='execution_signature_mismatch'))
            continue
        predicted, observed = sample.get('predicted_delta_ms'), sample.get('observed_delta_ms')
        if any(value is None or isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in (predicted, observed)):
            unknown.append(sample.get('id'))
            continue
        rows.append(dict(sample, admitted=predicted < -margin_ms, profitable=observed < 0,
                         direction_correct=(predicted < 0) == (observed < 0)))
    admitted = sum(row['admitted'] for row in rows)
    profitable = sum(row['profitable'] for row in rows)
    false = sum(row['admitted'] and not row['profitable'] for row in rows)
    missed = sum(not row['admitted'] and row['profitable'] for row in rows)
    return dict(known=len(rows), unknown=unknown, rejected=rejected, admitted=admitted, profitable=profitable,
                false_admissions=false, false_admission_rate=false / admitted if admitted else None,
                missed_benefits=missed, missed_benefit_rate=missed / profitable if profitable else None,
                direction_accuracy=sum(row['direction_correct'] for row in rows) / len(rows) if rows else None,
                absolute_error_p95_ms=_p95([abs(row['predicted_delta_ms'] - row['observed_delta_ms']) for row in rows]),
                rows=rows)


def _tail(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 4096))
        return stream.read().decode(errors='replace').splitlines()[-4:]


def audit_cluster(run_root: Path, nodes: int = 8, dies_per_node: int = 16) -> dict[str, Any]:
    """Audit all node exits and all worker completion records without modifying the run."""
    node_records = []
    for node in range(nodes):
        path = run_root / 'nodes' / f'node_{node}'
        exits = {stage: int((path / f'{stage}-exit_status.txt').read_text())
                 if (path / f'{stage}-exit_status.txt').is_file() else None for stage in STAGES}
        terminal = int((path / 'exit_status.txt').read_text()) if (path / 'exit_status.txt').is_file() else None
        node_records.append(dict(node=node, started=(path / 'started').exists(), exits=exits, terminal=terminal,
                                 terminal_npu_snapshot=(path / 'npu-terminal.txt').is_file(),
                                 launcher_tail=_tail(path / 'launcher.log')))
    stages = {}
    expected = nodes * dies_per_node
    for stage in STAGES:
        placements = list((run_root / stage / 'placement').glob('placement-rank*.json'))
        completed = []
        malformed = []
        for file in placements:
            try:
                record = json.loads(file.read_text())
                if record.get('completed'):
                    completed.append(record['global_rank'])
            except (ValueError, KeyError) as error:
                malformed.append(dict(file=str(file), reason=str(error)))
        exits = [row['exits'][stage] for row in node_records]
        stages[stage] = dict(node_exits=exits, started_ranks=len(placements), completed_ranks=sorted(completed),
                             malformed=malformed, complete=all(value == 0 for value in exits)
                             and sorted(completed) == list(range(expected)))
    return dict(observed_utc=datetime.now(timezone.utc).isoformat(), run_root=str(run_root),
                nodes=node_records, stages=stages, terminal=all(row['terminal'] is not None for row in node_records),
                all_nodes_success=all(row['terminal'] == 0 for row in node_records),
                cleanup_verified=False,
                scope='Raw node exits and per-rank placement completion; no PID cleanup inference')


def main() -> None:
    """Publish observations and service-only candidates to a separate audit directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = dict(cluster=audit_cluster(args.run_root), services={}, rejected=[])
    for backend in ('push', 'pull'):
        path = args.run_root / f'links-{backend}' / 'link-measurements.json'
        if not result['cluster']['stages'][f'links-{backend}']['complete']:
            result['rejected'].append(dict(path=str(path), reason='link_stage_incomplete'))
            continue
        payload = json.loads(path.read_text())
        try:
            result['services'][backend] = fit_provider_services(payload)
        except SampleRejected as error:
            result['rejected'].append(dict(path=str(path), reason=str(error)))
    result['critical_profile'] = dict(installable=False, reasons=[
        'Device scheduling crosswalk requires template revision before a full critical-path fit',
        'v1 baseline bypass does not retain actual source histograms per step',
        'v1 host samples measure wall time, not exposed host critical path',
        'Provider-only records do not bind full runtime vendor and residency identity'])
    output = args.output / 'calibration-audit.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), complete_stages=[stage for stage, row in
          result['cluster']['stages'].items() if row['complete']], service_domains=list(result['services']),
          critical_profile_installable=False)))


if __name__ == '__main__':
    main()
