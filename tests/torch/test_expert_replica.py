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
"""Lightweight launchers for standalone expert-replica acceptance."""

from pathlib import Path

import pytest

from tests.common.mark_utils import arg_mark
from tests.common.multicore._case_launcher import run_multicore_case
from tests.common.multicore._test_env import prepare_multicore_test_environment, without_inherited_rank_environment


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@pytest.mark.parametrize("backend,mode", [("mega", "push"), ("mega", "pull"), ("native", "push")])
def test_replica_training(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend: str, mode: str) -> None:
    """Compare deferred two-step training, actual copies and complete owner gradients."""
    prepare_multicore_test_environment()
    monkeypatch.setenv("HP_REPLICA_BACKEND", backend)
    monkeypatch.setenv("HP_REPLICA_MODE", mode)
    monkeypatch.setenv("HP_REPLICA_RESULT_DIR", str(tmp_path))
    worker = str(Path(__file__).with_name("_test_expert_replica.py"))
    with without_inherited_rank_environment():
        run_multicore_case(worker, "test_replica_training", num_proc=2, timeout=900)
