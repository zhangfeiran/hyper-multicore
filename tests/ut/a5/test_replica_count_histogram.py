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
"""Verify bounded FP32 histograms preserve exact replica count and validation payloads."""

from collections import Counter
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from hyper_multicore.replica import routing


def oracle(ids: torch.Tensor, experts: int) -> torch.Tensor:
    """Count clamped integers and independently detect invalid or duplicate token IDs."""
    rows = ids.tolist()
    invalid = any(value < 0 or value >= experts for row in rows for value in row)
    invalid = invalid or any(len(set(row)) != len(row) for row in rows)
    counts = Counter(min(experts - 1, max(0, value)) for row in rows for value in row)
    return torch.tensor([counts[index] for index in range(experts)] + [int(invalid)], dtype=torch.int64)


def test_histogram_uses_unit_width_bins_and_keeps_integer_payload() -> None:
    """Exercise the candidate Torch operations on CPU without pretending to validate an NPU."""
    ids = torch.tensor([[0, 1, 95], [94, 95, 0], [-1, 96, 2**63 - 1]], dtype=torch.int64)
    with (mock.patch.object(routing, "_use_exact_count_histogram", return_value=True, create=True),
          mock.patch.object(torch, "histc", wraps=torch.histc) as histogram):
        actual = routing._torch_replica_count_payload(ids, 96)
    histogram.assert_called_once()
    assert histogram.call_args.kwargs == {"bins": 96, "min": 0, "max": 96}
    assert actual.dtype == torch.int64 and actual.is_contiguous()
    assert torch.equal(actual, oracle(ids, 96))


@pytest.mark.parametrize("experts", [1, 96, 384, 512])
def test_histogram_matches_independent_oracle_for_hot_empty_and_strided_routes(experts: int) -> None:
    """Cover legal tails, cold bins, duplicates, bounds and noncontiguous logical ID views."""
    topk = min(8, experts)
    rows = (torch.arange(4096, dtype=torch.int64)[:, None] * 13 + torch.arange(topk) * 11) % experts
    strided = torch.empty((4096, topk * 2), dtype=torch.int64)
    strided[:, ::2] = rows
    cases = [rows, strided[:, ::2], torch.zeros_like(rows), torch.empty((0, topk), dtype=torch.int64),
             torch.tensor([[-1, 0, experts, -(2**63), 2**63 - 1]], dtype=torch.int64)]
    with mock.patch.object(routing, "_use_exact_count_histogram", return_value=True):
        for ids in cases:
            actual = routing._torch_replica_count_payload(ids, experts)
            assert torch.equal(actual, oracle(ids, experts))


@pytest.mark.parametrize("slots,experts,device,dtype,expected", [
    (32768, 96, "npu", torch.int64, True),
    (2**24, 512, "npu", torch.int64, True),
    (2**24 + 1, 96, "npu", torch.int64, False),
    (0, 96, "npu", torch.int64, False),
    (32768, 513, "npu", torch.int64, False),
    (32768, 0, "npu", torch.int64, False),
    (32768, 96, "cpu", torch.int64, False),
    (32768, 96, "cuda", torch.int64, False),
    (32768, 96, "npu", torch.int32, False),
])
def test_histogram_eligibility_preserves_integer_fallbacks(
    slots: int, experts: int, device: str, dtype: torch.dtype, expected: bool,
) -> None:
    """Validate metadata boundaries without allocating a sixteen-million-element tensor."""
    ids = SimpleNamespace(device=SimpleNamespace(type=device), dtype=dtype, numel=lambda: slots)
    assert routing._use_exact_count_histogram(ids, experts) is expected


def test_cpu_fallback_and_optional_native_precedence_are_unchanged() -> None:
    """Keep unsupported devices and the existing native validation/count ABI on their old paths."""
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    with mock.patch.object(torch, "histc", side_effect=AssertionError("Unexpected CPU histogram")):
        assert torch.equal(routing._torch_replica_count_payload(ids, 4), oracle(ids, 4))
    expected = torch.tensor([1, 1, 1, 1, 0], dtype=torch.int64)
    with (mock.patch.object(routing, "try_count_payload", return_value=expected),
          mock.patch.object(routing, "_torch_replica_count_payload", side_effect=AssertionError("Native bypassed"))):
        assert routing._replica_count_payload(ids, 4) is expected
