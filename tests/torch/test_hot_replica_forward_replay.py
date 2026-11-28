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
"""Launch the A3 HotReplica forward plan replay regression."""
from pathlib import Path

from tests.common.mark_utils import arg_mark
from tests.common.multicore._case_launcher import run_multicore_case
from tests.common.multicore._test_env import prepare_multicore_test_environment, without_inherited_rank_environment
from tests.common.port_utils import allocate_port


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_forward_route_replay(monkeypatch):
    """Exercise exact gradients, zero replay votes and consistent EP setup rejection."""
    prepare_multicore_test_environment()
    monkeypatch.setenv("HYPER_MULTICORE_SHMEM_BOOTSTRAP_ENDPOINT", f"tcp://127.0.0.1:{allocate_port()}")
    with without_inherited_rank_environment():
        run_multicore_case(str(Path(__file__).with_name("_test_hot_replica_forward_replay.py")),
                           "test_forward_route_replay", num_proc=2, timeout=420)
