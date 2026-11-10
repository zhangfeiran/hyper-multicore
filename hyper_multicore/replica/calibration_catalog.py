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
"""Validated external calibration artifacts bound to the deployed execution sources."""

from __future__ import annotations

import hashlib
from importlib import metadata
from importlib.machinery import EXTENSION_SUFFIXES
import json
import os
from pathlib import Path
from typing import Any

from hyper_multicore.shmem._runtime import _configured_shmem_setting

EXECUTION_FILES = tuple('hyper_multicore/' + name for name in (
    'modules/mega_moe/module.py', 'modules/mega_moe/function.py', 'modules/mega_moe/heap_manager.py',
    'modules/mega_moe/spec.py', 'modules/mega_moe/plan.py', 'modules/mega_moe/inactive_tasks.py',
    'modules/mega_moe/workspace.py', 'modules/mega_moe/mxfp8_module.py', 'modules/mega_moe/mxfp8_execution.py',
    'modules/mega_moe/mxfp8_graph.py', 'modules/mega_moe/mxfp8_gradients.py',
    'modules/mega_moe/mxfp8_gradient_udma.py', 'scheduler/mxfp8.py', 'replica/residency.py',
    'replica/host.py', 'replica/policy.py', 'replica/signal_transport.py', 'replica/routing.py',
    'replica/planner.py', 'replica/cost.py', 'replica/recompute.py'))
ENVIRONMENT_DEFAULTS = {'HYPER_PARALLEL_REPLICA_WEIGHT_RESIDENCY': '1',
                        'HYPER_PARALLEL_REPLICA_METADATA_REUSE': '0',
                        'HYPER_PARALLEL_REPLICA_HOST_NATIVE': '1'}


def _host_extension(directory: Path) -> Path | None:
    """Follow the interpreter's extension suffix order, including packaged ABI3 builds."""
    return next((directory / ('_replica_host' + suffix) for suffix in EXTENSION_SUFFIXES
                 if (directory / ('_replica_host' + suffix)).is_file()), None)


def runtime_contract() -> dict[str, Any]:
    """Fingerprint execution sources and policy switches without querying a device."""
    root = Path(__file__).resolve().parents[2]
    flags = {name: os.environ.get(name, default) for name, default in ENVIRONMENT_DEFAULTS.items()}
    if any(value not in ('0', '1') for value in flags.values()):
        raise ValueError('Replica execution policy switches must be 0 or 1')
    versions = {}
    for package in ('torch', 'torch-npu'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    cann = Path(os.environ.get('ASCEND_HOME_PATH', '/nonexistent')) / 'share/info/runtime/version.info'
    host = _host_extension(Path(__file__).with_name('_host_kernel'))
    return {'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                             for name in EXECUTION_FILES}, 'environment': flags,
            'framework_versions': versions, 'cann_version_info': cann.read_text() if cann.is_file() else None,
            'host_native_sha256': hashlib.sha256(host.read_bytes()).hexdigest() if host is not None else None,
            'heap_budget': _configured_shmem_setting('HEAP_SIZE', '') or None,
            'allocation_scope': {name: os.environ.get(name) for name in
                                 ('MASTER_ADDR', 'VC_WORKER_HOSTS', 'NNODES', 'NUM_CARDS')}}


def external_profiles(directory: Path) -> list[dict[str, Any]]:
    """Load only atomically published, validated and source-matched external profiles.

    Args:
        directory: Output directory containing catalog.json and checksummed model JSON files.

    Returns:
        Validated model payloads; workload selection remains the runtime policy's responsibility.
    """
    catalog = json.loads((directory / 'catalog.json').read_text())
    if catalog.get('schema_version') != 1 or catalog.get('status') != 'validated':
        raise ValueError('External replica calibration catalog is not validated')
    if catalog.get('runtime_contract') != runtime_contract():
        raise ValueError('External replica calibration runtime contract mismatch')
    profiles = catalog.get('profiles')
    if not isinstance(profiles, list) or not profiles:
        raise ValueError('External replica calibration catalog has no profiles')
    result = []
    for record in profiles:
        filename = record['file']
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith('.json'):
            raise ValueError('External calibration profile must be a JSON filename within its catalog')
        path = directory / filename
        if path.resolve().parent != directory.resolve():
            raise ValueError('External calibration profile escapes its catalog')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record['sha256']:
            raise ValueError(f'External replica calibration checksum mismatch: {filename}')
        payload = json.loads(raw)
        signature = payload['selector'].get('execution_signature')
        if (not isinstance(signature, str) or len(signature) != 64
                or any(value not in '0123456789abcdef' for value in signature)):
            raise ValueError('External calibration requires an observed execution signature')
        if payload.get('validation', {}).get('status') != 'passed':
            raise ValueError('External calibration profile has not passed held-out validation')
        result.append(payload)
    return result
