# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Verify standalone and migrated SHMEM configuration select one runtime."""

import pytest

from hyper_multicore.shmem._runtime import configured_data_engine, configured_udma_qp_count


@pytest.mark.parametrize('prefix', ['HYPER_MULTICORE', 'HYPER_PARALLEL'])
def test_each_namespace_selects_the_same_reduced_qp_bank(monkeypatch: pytest.MonkeyPatch, prefix: str) -> None:
    """Both supported namespaces configure the same data engine and QP bank."""
    monkeypatch.setenv(f'{prefix}_SHMEM_DATA_ENGINE', 'mte_udma')
    monkeypatch.setenv(f'{prefix}_SHMEM_UDMA_QP_NUM', '1')
    assert configured_data_engine() == 'mte_udma'
    assert configured_udma_qp_count() == 1


def test_conflicting_names_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Conflicting settings fail before a misleading execution plan is created."""
    monkeypatch.setenv('HYPER_MULTICORE_SHMEM_DATA_ENGINE', 'mte')
    monkeypatch.setenv('HYPER_PARALLEL_SHMEM_DATA_ENGINE', 'mte_udma')
    with pytest.raises(ValueError, match='Conflicting'):
        configured_data_engine()


def test_reduced_qp_bank_requires_udma(monkeypatch: pytest.MonkeyPatch) -> None:
    """An MTE-only process cannot request a reduced UDMA bank."""
    monkeypatch.setenv('HYPER_MULTICORE_SHMEM_DATA_ENGINE', 'mte')
    monkeypatch.setenv('HYPER_MULTICORE_SHMEM_UDMA_QP_NUM', '1')
    with pytest.raises(ValueError, match='requires mte_udma'):
        configured_udma_qp_count()
