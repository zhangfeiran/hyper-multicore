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
"""Script-only fitting of execution-bound effective replica service models."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .calibration_catalog import runtime_contract
from .capacity import ExpertReplicaConfig
from .cost import ExpertReplicaCostModel
from .plan import ExpertExecutionPlan

MEASUREMENT_CONTRACT = 'rank_cumulative_f_r_b_v2'
CAPTURE_POLICY = 'runtime_default_admission_v1'
HOT_OWNER_EXCESS = 2.5


class CalibrationDesignRejected(ValueError):
    """Report valid captures that cannot identify a stable TRAIN-only service model."""


def load_manifest(path: Path) -> dict[str, Any]:
    """Validate a model-owned manifest without importing the model or probing devices."""
    value = json.loads(path.read_text())
    for name in ('hidden_size', 'intermediate_size', 'num_experts', 'top_k', 'ep_size', 'local_num_tokens'):
        if type(value.get(name)) is not int or value[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if value['ep_size'] < 2 or value['num_experts'] % value['ep_size'] or value['top_k'] > value['num_experts']:
        raise ValueError('Expert count, EP size and TopK are inconsistent')
    if value.get('checkpointed') is not True:
        raise ValueError('The first automatic calibration contract requires real checkpoint replay')
    defaults = {'dispatch_modes': ['push', 'pull'], 'replica_caps': [1, 2, 6], 'dtype': 'bfloat16',
                'gradient_dtype': 'float32', 'expert_weight_layout': 'EIO', 'swiglu_limit': None,
                'residency_mode': 'force_normal_miss', 'checkpoint_contract': 'nonreentrant',
                'measurement_contract': MEASUREMENT_CONTRACT, 'capture_policy': CAPTURE_POLICY,
                'frames': 12, 'samples': 3, 'warmup': 1}
    result = defaults | value
    if result['checkpoint_contract'] not in ('reentrant', 'nonreentrant'):
        raise ValueError('checkpoint_contract must be reentrant or nonreentrant')
    if result['measurement_contract'] != MEASUREMENT_CONTRACT:
        raise ValueError('Automatic calibration requires fresh rank-cumulative F/R/B component timing')
    if result['capture_policy'] != CAPTURE_POLICY:
        raise ValueError('Automatic calibration requires the runtime default admission policy')
    if (result['dtype'], result['gradient_dtype'], result['expert_weight_layout'], result['residency_mode']) != (
            'bfloat16', 'float32', 'EIO', 'force_normal_miss'):
        raise ValueError('Automatic calibration requires BF16/EIO, FP32 partials and force_normal_miss')
    if not result['dispatch_modes'] or any(mode not in ('push', 'pull') for mode in result['dispatch_modes']):
        raise ValueError('dispatch_modes must contain push and/or pull')
    if not result['replica_caps'] or any(type(cap) is not int or cap <= 0 for cap in result['replica_caps']):
        raise ValueError('replica_caps must contain positive integer caps')
    if len(set(result['replica_caps'])) != len(result['replica_caps']) or len(set(result['dispatch_modes'])) != len(
            result['dispatch_modes']):
        raise ValueError('Calibration modes and caps must be unique')
    limit = result['swiglu_limit']
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, (float, int))
                              or not math.isfinite(limit) or limit <= 0):
        raise ValueError('swiglu_limit must be None or a positive finite number')
    for name, minimum in (('frames', 12), ('samples', 3), ('warmup', 1)):
        if type(result[name]) is not int or result[name] < minimum:
            raise ValueError(f'{name} must be at least {minimum}')
    if result.get('replica_transport', 'shmem_signal_kernel_gradient_bands') != 'shmem_signal_kernel_gradient_bands':
        raise ValueError('Automatic calibration requires gradient-band transport')
    templates = result.get('route_templates', [])
    if templates:
        if not isinstance(templates, list) or result['frames'] != 12 + len(templates):
            raise ValueError('Route templates must follow the twelve bounded calibration frames')
        names = set()
        for template in templates:
            counts = template.get('source_expert_counts', [])
            name, family = template.get('name'), template.get('family')
            if (not isinstance(name, str) or not name or name in names
                    or not isinstance(family, str) or not family
                    or len(counts) != result['num_experts']
                    or any(type(count) is not int or not 0 <= count <= result['local_num_tokens'] for count in counts)
                    or sum(counts) != result['local_num_tokens'] * result['top_k']):
                raise ValueError('Route templates require unique names, families and conserved distinct-TopK counts')
            names.add(name)
    return result


def calibration_ids(tokens: int, manifest: dict[str, Any], frame: int, nodes: tuple[int, ...]) -> torch.Tensor:
    """Build bounded expert histograms with distinct logical experts in each token row."""
    experts, top_k, size = manifest['num_experts'], manifest['top_k'], manifest['ep_size']
    templates = manifest.get('route_templates', [])
    if templates and frame >= 12:
        if tokens != manifest['local_num_tokens'] or frame - 12 >= len(templates):
            raise ValueError('A frozen route template requires its exact declared source-token shape')
        counts = templates[frame - 12]['source_expert_counts']
        flat = torch.repeat_interleave(torch.arange(experts), torch.tensor(counts))
        return flat.reshape(top_k, tokens).T.contiguous().int()
    counts = [tokens * top_k // experts] * experts
    for expert in range(tokens * top_k % experts):
        counts[expert] += 1
    kind, home = frame % 4, experts // size
    if kind in (1, 2):
        owner = (frame // 4) % size
        owners = [owner] if kind == 1 else [rank for rank, node in enumerate(nodes) if node == nodes[owner]]
        recipients = [rank * home for rank in owners]
        # Strong probes must reach runtime admission without forcing a different policy.
        desired = int(tokens * top_k / size * HOT_OWNER_EXCESS)
        donors = [expert for expert in range(experts) if expert // home not in owners]
        pending = min(desired * len(owners), sum(counts[expert] for expert in donors))
        for expert in recipients:
            moved = min(desired, pending, tokens - counts[expert])
            counts[expert] += moved
            pending -= moved
            left = moved
            for donor in donors:
                amount = min(left, counts[donor])
                counts[donor] -= amount
                left -= amount
                if not left:
                    break
    flat = torch.repeat_interleave(torch.arange(experts), torch.tensor(counts))
    return flat.reshape(top_k, tokens).T.contiguous().int()


def plan_features(record: dict[str, Any]) -> tuple[np.ndarray, ExpertExecutionPlan]:
    """Represent the existing legacy max-rank predictor without labeling effective terms physical."""
    plan = ExpertExecutionPlan(ExpertReplicaConfig(**record['config']), record['source_counts'],
                               record['slots'], record['dispatch'])
    plan.validate()
    nodes = record['rank_nodes']
    size = plan.config.ep_size
    if len(nodes) != size:
        raise ValueError('Observation topology does not match its plan')
    local, remote = np.zeros(size), np.zeros(size)
    for edge in plan.transfers:
        target = local if nodes[edge.owner_rank] == nodes[edge.target_rank] else remote
        target[edge.owner_rank] += 1
        target[edge.target_rank] += 1
    # Columns: forward fixed/row, backward fixed/row, local/remote ready,
    # local/remote gradient tail, shared token service, active-path phase term.
    result = np.zeros((2, size, 10), dtype=np.float64)
    for rank, (slots, counts) in enumerate(zip(plan.slot_to_logical, plan.destination_counts)):
        rows = sum(counts)
        tasks = sum(count > 0 for count in counts)
        hits = sum(min(plan.logical_counts[rank][expert], count)
                   for expert, count in zip(slots, counts) if expert >= 0)
        token_rows = (sum(plan.logical_counts[rank]) + rows - 2 * hits) / 2
        result[0, rank, [0, 1, 4, 5, 8, 9]] = tasks, rows, local[rank], remote[rank], token_rows, bool(plan.transfers)
        result[1, rank, [2, 3, 6, 7, 8]] = tasks, rows, local[rank], remote[rank], token_rows
    return result, plan


def fit_nonnegative_services(features: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit identifiable rank-local effective services by convex nonnegative least squares."""
    if np.any(observed < 0):
        raise ValueError('Calibration samples require nonnegative device durations')
    return _fit_nonnegative_services(features, observed)


def _fit_nonnegative_services(features: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Allow signed paired residuals while retaining nonnegative service coefficients."""
    if features.ndim != 2 or observed.shape != (features.shape[0],):
        raise ValueError('Expected observation x coefficient features and one rank-local target per observation')
    if not np.isfinite(features).all() or not np.isfinite(observed).all():
        raise ValueError('Calibration samples must be finite')
    scales = np.maximum(np.linalg.norm(features, axis=0), 1.)
    x = features / scales
    rank = int(np.linalg.matrix_rank(x, tol=1e-7))
    if rank != features.shape[1]:
        raise ValueError(f'Unidentifiable calibration design: rank={rank}, coefficients={features.shape[1]}')
    condition = float(np.linalg.cond(x))
    if condition > 1e8:
        raise ValueError(f'Ill-conditioned calibration design: {condition}')
    gram, target = x.T @ x, x.T @ observed
    lipschitz = max(float(np.linalg.eigvalsh(gram)[-1]), 1e-12)
    beta = np.zeros(features.shape[1])
    accelerated, momentum = beta.copy(), 1.
    for iteration in range(20000):
        proposal = np.maximum(0., accelerated - (gram @ accelerated - target) / lipschitz)
        if np.linalg.norm(proposal - beta) < 1e-11 * (1 + np.linalg.norm(beta)):
            beta = proposal
            break
        next_momentum = (1 + math.sqrt(1 + 4 * momentum * momentum)) / 2
        accelerated = proposal + (momentum - 1) / next_momentum * (proposal - beta)
        beta, momentum = proposal, next_momentum
    error = x @ beta - observed
    return beta / scales, {'iterations': iteration + 1, 'design_rank': rank, 'condition_number': condition,
                           'train_absolute_error_p95_ms': float(np.quantile(np.abs(error), .95)),
                           'train_worst_underprediction_ms': float(np.max(-error))}


def _samples(reports: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    result = []
    for report in reports:
        if report['budget'] != budget:
            continue
        for row in report['records']:
            if row.get('measurement_contract') != MEASUREMENT_CONTRACT:
                raise ValueError('Capture requires fresh rank-cumulative F/R/B component timing')
            if row.get('capture_policy') != CAPTURE_POLICY:
                raise ValueError('Capture requires the runtime default admission policy')
            if row.get('execution_signature', report['execution_signature']) != report['execution_signature']:
                raise ValueError('Observation changed its live execution or token domain')
            if row.get('profile_on') is not False or not row.get('finite'):
                raise ValueError('Fit requires finite profiler-off observations')
            features, plan = plan_features(row)
            durations = row['rank_phase_ms']
            if len(durations) != plan.config.ep_size:
                raise ValueError('Missing rank-local phase events')
            target = np.array([(max(rank['forward'] for rank in durations)
                                + max(rank['recompute'] for rank in durations)) / 2,
                               max(rank['backward'] for rank in durations)])
            local_target = np.array([rank['forward'] + rank['recompute'] + rank['backward'] for rank in durations])
            component = np.array(row['rank_component_ms'])
            if (component.shape != local_target.shape or not np.isfinite(component).all()
                    or np.any(component <= 0) or np.any(local_target > component + .1 + .01 * component)):
                raise ValueError('Cumulative F/R/B must fit inside each measured whole component interval')
            result.append(row | {'features': features, 'target': target, 'local_target': local_target, 'plan': plan,
                                 'observed_ms': float(max(local_target)),
                                 'component_rank_max_ms': float(max(component)),
                                 'phase_rank_max_sum_ms': float(2 * target[0] + target[1])})
    return result


def _candidate(manifest: dict[str, Any], report: dict[str, Any], budget: int,
               beta: np.ndarray, margin: float, coverage: int, source_floor: float = 0.) -> dict[str, Any]:
    def curve(fixed: float, slope: float) -> tuple:
        return ((0, 0.), (1, float(fixed + slope)), (max(2, coverage), float(fixed + slope * max(2, coverage))))
    model = ExpertReplicaCostModel(
        calibration_version=2, backend=report['backend'], hidden_size=manifest['hidden_size'],
        intermediate_size=manifest['intermediate_size'], ep_size=manifest['ep_size'],
        transport='shmem_signal_kernel_gradient_bands', forward_ms=curve(beta[0] / 3, beta[1] / 3),
        backward_ms=curve(beta[0] / 3, beta[1] / 3), weight_ms=0., gradient_ms=0.,
        token_ms_per_row=float(beta[4] / 3),
        forward_repeats=2, backward_weight_ms=0., replica_overhead_ms=float(beta[5] / 2),
        intra_node_weight_ms=float(beta[2] / 3), inter_node_weight_ms=float(beta[3] / 3),
        intra_node_gradient_ms=float(beta[2] / 3), inter_node_gradient_ms=float(beta[3] / 3),
        intra_node_backward_weight_ms=0., inter_node_backward_weight_ms=0.,
        effective_group_ms=tuple(float(value) for value in beta), _rank_nodes=tuple(report['rank_nodes']),
        effective_source_row_floor=source_floor,
        schedule=report['schedule'], minimum_gain_ms=margin, minimum_replica_gain_ms=margin)
    return {'selector': {'soc_name': report['soc_name'], 'num_experts': manifest['num_experts'],
                         'top_k': manifest['top_k'], 'execution_signature': report['execution_signature']},
            'model': asdict(model), 'budget': budget,
            'provenance': {'measurement_semantics': 'effective_fused_execution_coefficients',
                           'measurement_contract': MEASUREMENT_CONTRACT,
                           'physical_service_claim': False, 'overlap_verified': False,
                           'scope': 'Group component maxima predict rank-cumulative F/R/B, '
                                    'not identified physical compute/band/ACK services'}}


def _paired_training(train: list[dict[str, Any]]) -> tuple[np.ndarray, ...]:
    """Keep each B0/Bn pair and all repeated route-family samples in the same TRAIN fold."""
    if any(row['frame'] % 3 == 2 for row in train):
        raise ValueError('Service fitting accepts TRAIN frames only')
    if not train or len({row['group_leader'] for row in train}) != 1:
        raise ValueError('Service fitting requires one TRAIN EP group')
    indexed = {}
    for row in train:
        key = row['group_leader'], row['frame'], row['sample']
        budget = row['plan'].config.replica_slots_per_rank
        role = int(budget > 0)
        if (key, role) in indexed:
            raise ValueError('Duplicate B0 or replica training observation')
        indexed[key, role] = row
    features, target, gains, folds, source_rows = [], [], [], [], []
    for key in sorted({key for key, _ in indexed}):
        if (key, 0) not in indexed or (key, 1) not in indexed:
            raise ValueError('Replica fitting requires a matched B0 for every training observation')
        owner, replica = indexed[key, 0], indexed[key, 1]
        if (owner['plan'].logical_counts != replica['plan'].logical_counts
                or tuple(owner['rank_nodes']) != tuple(replica['rank_nodes'])):
            raise ValueError('Matched training pairs require the same source histogram and topology')
        family = lambda row: row.get('cv_group', str(row['frame'] % 4 + 4 * int(4 <= row['frame'] % 12 < 8)))
        if family(owner) != family(replica):
            raise ValueError('Matched observations cannot cross route-family folds')
        features.append([row['features'][0][:, [0, 1, 4, 5, 8, 9]].max(axis=0) for row in (owner, replica)])
        target.append([owner['observed_ms'], replica['observed_ms']])
        gains.append(owner['component_rank_max_ms'] - replica['component_rank_max_ms'])
        folds.append(family(owner))
        source_rows.append(max(map(sum, owner['plan'].logical_counts)))
    if len(set(folds)) < 2:
        raise CalibrationDesignRejected('Paired calibration requires multiple TRAIN route families')
    return np.array(features), np.array(target), np.array(gains), np.array(folds), np.array(source_rows)


def _owner_services(features: np.ndarray, target: np.ndarray,
                    folds: np.ndarray, source_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Anchor fixed, receive-row and token services exclusively to owner-only observations."""
    candidates, errors = [], []
    bases = ((0, 1, 4), (0, 1), (1, 4), (0, 4), (0,), (1,), (4,))
    for floor, basis in ((floor, basis) for floor in (0., .5, .75, 1.) for basis in bases):
        design = features.copy()
        design[:, 1] = np.maximum(design[:, 1], source_rows * floor)
        columns = list(basis)
        try:
            predicted = np.zeros_like(target)
            coefficients = np.zeros((len(target), 6))
            for fold in sorted(set(folds)):
                fit_rows, valid_rows = folds != fold, folds == fold
                values, _ = fit_nonnegative_services(design[fit_rows][:, columns], target[fit_rows])
                predicted[valid_rows] = design[valid_rows][:, columns] @ values
                coefficients[np.ix_(valid_rows, columns)] = values
            relative = np.abs(predicted - target) / np.maximum(target, 1e-6)
            cv = dict(relative_error_p95=float(np.quantile(relative, .95)),
                      worst_underprediction_ms=float(max(target - predicted)))
            candidates.append((cv['relative_error_p95'], len(columns), floor, columns, cv, coefficients))
        except ValueError as error:
            errors.append(str(error))
    if not candidates:
        raise CalibrationDesignRejected(f'No identifiable B0 TRAIN basis: {errors}')
    _, _, floor, columns, cv, coefficients = min(candidates, key=lambda row: (row[0], row[1], row[2], row[3]))
    design = features.copy()
    design[:, 1] = np.maximum(design[:, 1], source_rows * floor)
    values, fit = fit_nonnegative_services(design[:, columns], target)
    beta = np.zeros(6)
    beta[columns] = values
    fit.update(basis=columns, source_row_floor=floor, cross_validation=cv, rejected_training_bases=errors)
    return beta, coefficients, fit


def _replica_services(features: np.ndarray, target: np.ndarray, gains: np.ndarray, folds: np.ndarray,
                      owner: np.ndarray, owner_cv: np.ndarray,
                      nodes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit only paired incremental costs; tied links are a fallback when topology is not identifiable."""
    possible = {2: len(set(nodes)) < len(nodes), 3: len(set(nodes)) > 1}
    bases = (((2,), (3,), (5,)), ((2,), (3,)), ((2, 3), (5,)), ((2, 3),))
    delta = features[:, 1] - features[:, 0]
    candidates, errors = [], []
    for basis in bases:
        groups = tuple(tuple(column for column in group if possible.get(column, True)) for group in basis)
        groups = tuple(group for group in groups if group)
        design = np.stack([features[:, 1, list(group)].sum(axis=1) for group in groups], axis=1)
        try:
            predicted = np.zeros_like(gains)
            candidate_cv = np.zeros_like(gains)
            for fold in sorted(set(folds)):
                fit_rows, valid_rows = folds != fold, folds == fold
                base = owner_cv[valid_rows][0]
                residual = target[fit_rows, 1] - target[fit_rows, 0] - delta[fit_rows] @ base
                values, _ = _fit_nonnegative_services(design[fit_rows], residual)
                predicted[valid_rows] = -(delta[valid_rows] @ base + design[valid_rows] @ values)
                candidate_cv[valid_rows] = features[valid_rows, 1] @ base + design[valid_rows] @ values
            cv = dict(gain_absolute_error_p95_ms=float(np.quantile(np.abs(predicted - gains), .95)),
                      worst_gain_overprediction_ms=float(max(predicted - gains)))
            candidates.append((cv['gain_absolute_error_p95_ms'], len(groups), groups, cv, candidate_cv))
        except ValueError as error:
            errors.append(str(error))
    if not candidates:
        raise CalibrationDesignRejected(f'No identifiable paired replica TRAIN basis: {errors}')
    _, _, groups, cv, predicted = min(candidates, key=lambda row: (row[0], row[1], row[2]))
    design = np.stack([features[:, 1, list(group)].sum(axis=1) for group in groups], axis=1)
    residual = target[:, 1] - target[:, 0] - delta @ owner
    values, fit = _fit_nonnegative_services(design, residual)
    beta = np.zeros(6)
    for group, value in zip(groups, values):
        beta[list(group)] = value
    fit.update(link_groups=groups, cross_validation=cv, rejected_training_bases=errors)
    return beta, predicted, fit


def _fit_services(train: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit owner-only absolute services, then matched replica increments using TRAIN-only folds."""
    features, target, gains, folds, source_rows = _paired_training(train)
    owner, owner_cv, owner_fit = _owner_services(features[:, 0], target[:, 0], folds, source_rows)
    features[:, :, 1] = np.maximum(features[:, :, 1], source_rows[:, None] * owner_fit['source_row_floor'])
    replica, replica_cv, replica_fit = _replica_services(
        features, target, gains, folds, owner, owner_cv, tuple(train[0]['rank_nodes']))
    beta = owner + replica
    predicted = features @ beta
    cv_predicted = np.stack(((features[:, 0] * owner_cv).sum(axis=1), replica_cv), axis=1)
    cv_error = cv_predicted - target
    gain_error = predicted[:, 0] - predicted[:, 1] - gains
    cv_gain_error = cv_predicted[:, 0] - cv_predicted[:, 1] - gains
    absolute_p95 = float(np.quantile(np.abs(predicted - target), .95))
    fit = dict(parameterization='b0_anchored_paired_group_nnls_v2', measurement_contract=MEASUREMENT_CONTRACT,
        basis=sorted(set(owner_fit['basis']) | {column for group in replica_fit['link_groups'] for column in group}),
        owner_fit=owner_fit, replica_fit=replica_fit, paired_train_samples=len(gains),
        source_row_floor=owner_fit['source_row_floor'],
        train_absolute_error_p95_ms=absolute_p95, train_group_absolute_error_p95_ms=absolute_p95,
        train_worst_underprediction_ms=float(np.max(target - predicted)),
        train_gain_absolute_error_p95_ms=float(np.quantile(np.abs(gain_error), .95)),
        cross_validation=dict(relative_error_p95=float(np.quantile(np.abs(cv_error) / np.maximum(target, 1e-6), .95)),
            worst_underprediction_ms=float(np.max(-cv_error)),
            gain_absolute_error_p95_ms=float(np.quantile(np.abs(cv_gain_error), .95)),
            worst_gain_overprediction_ms=float(np.max(cv_gain_error))),
        absolute_error_bound_ms=max(.05, 2 * absolute_p95, float(np.max(-cv_error))),
        decision_margin_ms=max(.05, 2 * float(np.quantile(np.abs(gain_error), .95)), float(np.max(cv_gain_error))),
        planner_cpu_max_ms=max(row.get('planner_cpu_ms', 0.) for row in train),
        uncertainty_contract='Paired whole-component gains already include host enqueue gaps; no CPU-wall subtraction',
        separate_phase_services_identified=False)
    return beta, fit


def validate_profile(payload: dict[str, Any], samples: list[dict[str, Any]], baseline: list[dict[str, Any]],
                     *, maximum_relative_error: float = .2) -> dict[str, Any]:
    """Gate untouched holdouts and harmful false admissions using the actual planner score."""
    model = ExpertReplicaCostModel(**payload['model'])
    absolute_bound = payload.get('fit', {}).get('absolute_error_bound_ms', model.minimum_gain_ms)
    if not isinstance(absolute_bound, (int, float)) or not math.isfinite(absolute_bound) or absolute_bound < 0:
        raise ValueError('Absolute prediction uncertainty must be finite and nonnegative')
    reference = {(row['group_leader'], row['frame'], row['sample']): row for row in baseline}
    errors, under, harmful, gain_errors, matched, required = [], [], [], [], 0, 0
    for row in samples:
        effective = ExpertReplicaCostModel(**(payload['model'] | {'_rank_nodes': tuple(row['rank_nodes'])}))
        predicted = effective.estimate_ms(row['plan'])
        observed = row['observed_ms']
        errors.append(abs(predicted - observed) / max(observed, 1e-6))
        under.append(observed - predicted)
        key = row['group_leader'], row['frame'], row['sample']
        if key not in reference:
            raise ValueError('Missing matched B0 holdout')
        owner = reference[key]
        predicted_gain = effective.estimate_ms(owner['plan']) - predicted
        observed_gain = owner['component_rank_max_ms'] - row['component_rank_max_ms']
        gain_errors.append(predicted_gain - observed_gain)
        if predicted_gain > model.minimum_gain_ms and observed_gain < -model.minimum_gain_ms:
            harmful.append({'frame': row['frame'], 'predicted_gain_ms': predicted_gain,
                            'observed_component_gain_ms': observed_gain})
        home = row['plan'].config.home_experts
        loads = [sum(sum(source[start:start + home]) for source in row['plan'].logical_counts)
                 for start in range(0, row['plan'].config.num_experts, home)]
        lookup = min(loads) != max(loads)
        required += int(lookup)
        matched += int(lookup and row.get('cost_model_matched', False))
    if len(errors) < 4:
        raise ValueError('Need at least four disjoint holdout observations')
    p95 = float(np.quantile(errors, .95))
    worst = max(under)
    gain_gate = payload.get('fit', {}).get('parameterization') == 'b0_anchored_paired_group_nnls_v2'
    return {'status': 'passed' if p95 <= maximum_relative_error and worst <= absolute_bound
            and (not gain_gate or max(gain_errors) <= model.minimum_gain_ms) and not harmful else 'failed',
            'holdout_samples': len(errors), 'relative_error_p95': p95,
            'worst_underprediction_ms': worst, 'maximum_relative_error': maximum_relative_error,
            'harmful_false_admissions': harmful, 'runtime_matched_samples': matched,
            'runtime_lookup_required_samples': required,
            'absolute_error_gate_ms': absolute_bound, 'admission_error_gate_ms': model.minimum_gain_ms,
            'gain_absolute_error_p95_ms': float(np.quantile(np.abs(gain_errors), .95)),
            'worst_gain_overprediction_ms': max(gain_errors),
            'note': 'No holdout coefficient refitting or slow-sample exclusion'}


def fit_directory(manifest: dict[str, Any], source: Path, output: Path, *, verify: bool = False) -> dict[str, Any]:
    """Fit per-mode/per-cap candidates, or validate model-in-loop observations and publish atomically."""
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'catalog.json').exists():
        raise ValueError('Published calibration catalogs are immutable; use a fresh output directory')
    reports = [json.loads(path.read_text()) for path in sorted(source.glob('*-group*.json'))]
    if not reports:
        raise ValueError('No completed group observations found')
    contract = runtime_contract()
    worlds = {report.get('world_size') for report in reports}
    if len(worlds) != 1 or None in worlds:
        raise ValueError('Missing or inconsistent cluster world size')
    world = worlds.pop()
    for rank in range(world):
        status = json.loads((source / f'status-rank{rank}.json').read_text())
        if status.get('rank') != rank or status.get('completed') is not True:
            raise ValueError(f'Incomplete rank evidence: {rank}')
    for report in reports:
        if (report.get('completed') is not True or report['runtime_contract'] != contract
                or report['manifest'] != manifest):
            raise ValueError('Observation runtime or model manifest mismatch')
    results, catalog = [], []
    for backend in manifest['dispatch_modes']:
        group = [row for row in reports if row['backend'] == backend]
        baseline = _samples(group, 0)
        for budget in manifest['replica_caps']:
            selected = [row for row in group if row['budget'] == budget]
            if not selected:
                raise ValueError(f'Missing observations: {backend}/B{budget}')
            signatures = {row['execution_signature'] for row in selected}
            members = [rank for row in selected for rank in row['ep_members']]
            if sorted(members) != list(range(world)):
                raise ValueError('Incomplete or overlapping EP groups')
            if len(signatures) != 1 or any(len(row['records']) != manifest['frames'] * manifest['samples']
                                           for row in selected):
                raise ValueError('Incomplete observations or incompatible execution signatures')
            rows = _samples(selected, budget)
            leaders = sorted({row['group_leader'] for row in rows})
            train_leader = leaders[0]
            holdout = [row for row in rows if row['frame'] % 3 == 2
                       and (len(leaders) == 1 or row['group_leader'] != train_leader)]
            filename = f'{backend}-b{budget}.json'
            if verify:
                payload = json.loads((output / filename).read_text())
                if payload['selector']['execution_signature'] not in signatures:
                    raise ValueError('Candidate execution signature changed during verification')
                validation = validate_profile(payload, holdout, baseline)
                if validation['runtime_matched_samples'] != validation['runtime_lookup_required_samples']:
                    validation['status'] = 'failed'
                    validation['reason'] = 'Candidate did not reach every runtime holdout'
            else:
                train = [row for row in rows + baseline if row['frame'] % 3 != 2
                         and row['group_leader'] == train_leader]
                beta, fit = _fit_services(train)
                margin = fit['decision_margin_ms']
                coverage = max(max(max(rank) for rank in row['plan'].destination_counts) for row in train)
                payload = _candidate(manifest, selected[0], budget, beta, margin, coverage,
                                     fit.get('source_row_floor', 0.))
                payload['fit'] = fit
                payload['fit']['train_group_leader'] = train_leader
                payload['fit']['holdout_group_leaders'] = sorted({row['group_leader'] for row in holdout})
                payload['fit']['disjoint_ep_group_holdout'] = len(leaders) > 1
                validation = validate_profile(payload, holdout, baseline)
                validation['status'] = 'candidate' if validation['status'] == 'passed' else 'failed'
            payload['validation'] = validation
            raw = json.dumps(payload, indent=2, allow_nan=False).encode() + b'\n'
            (output / filename).write_bytes(raw)
            results.append({'backend': backend, 'budget': budget, **validation})
            catalog.append({'file': filename, 'sha256': hashlib.sha256(raw).hexdigest()})
    accepted = all(row['status'] == ('passed' if verify else 'candidate') for row in results)
    report = {'status': ('validated' if verify else 'candidate') if accepted else 'failed', 'profiles': results,
              'runtime_contract': contract, 'script_only': True, 'collection_complete': True,
              'calibration_status': ('validated' if verify else 'candidate') if accepted else 'rejected',
              'catalog_published': bool(accepted and verify)}
    (output / 'fit-report.json').write_text(json.dumps(report, indent=2) + '\n')
    if accepted and verify:
        artifact = {'schema_version': 1, 'status': 'validated', 'runtime_contract': contract, 'profiles': catalog}
        temporary = output / 'catalog.json.tmp'
        temporary.write_text(json.dumps(artifact, indent=2) + '\n')
        temporary.replace(output / 'catalog.json')
    return report


def main() -> None:
    """Fit or verify a completed cluster capture; no manual editing of model parameters is required."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    try:
        report = fit_directory(load_manifest(args.manifest), args.input, args.output, verify=args.verify)
    except CalibrationDesignRejected as error:
        args.output.mkdir(parents=True, exist_ok=True)
        report = {'status': 'failed', 'calibration_status': 'rejected', 'collection_complete': True,
                  'reason': str(error), 'catalog_published': False}
        (args.output / 'fit-report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        raise SystemExit(20) from error
    except (ValueError, KeyError, FileNotFoundError) as error:
        args.output.mkdir(parents=True, exist_ok=True)
        report = {'status': 'failed', 'reason': str(error), 'catalog_published': False}
        (args.output / 'failure-report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
        raise SystemExit(1) from error
    print(json.dumps(report, indent=2))
    if report['status'] == 'failed':
        raise SystemExit(20)


if __name__ == '__main__':
    main()
