# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the MXFP8 guest weight storage without a device."""
from contextlib import nullcontext
from unittest.mock import Mock

import torch

from hyper_multicore.modules.hot_replica import mx_prefetch


def test_persistent_guest_storage_is_writable_after_inference(monkeypatch):
    monkeypatch.setattr(torch.npu, "Stream", lambda device: Mock(), raising=False)
    monkeypatch.setattr(torch.npu, "Event", Mock(), raising=False)
    monkeypatch.setattr(torch.npu, "stream", lambda stream: nullcontext(), raising=False)
    monkeypatch.setattr(mx_prefetch.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(mx_prefetch.dist, "get_process_group_ranks", lambda group: [0, 1])
    with torch.inference_mode():
        prefetch = mx_prefetch.MXWeightPrefetch(Mock(), torch.device("cpu"), 6, 6)
        weight = torch.zeros((6, 8), dtype=torch.uint8)
        guest, = prefetch._guest_banks((weight,), backward=False)
    assert not torch.is_inference(guest)
    assert not torch.is_inference(prefetch.ready)
    guest.fill_(17)
    assert prefetch._guest_banks((weight,), backward=False)[0] is guest
    assert guest[0].tolist() == [17] * 8
