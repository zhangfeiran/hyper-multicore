# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Reuse a validated logical agreement without changing the fixed MX execution graph."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hyper_multicore.modules.mega_moe import mxfp8_module
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec


def test_control_agreement_uses_logical_topology_and_retains_exact_counts():
    physical = MegaMoeSpec(4096,5120,1792,64,8,1.25,40960,8,None,0,32,
        replica_slots_per_rank=2,logical_num_experts=48,compute_stages=6)
    control = replace(physical,num_experts=48,replica_slots_per_rank=0,auto_tokens=True)
    resources = SimpleNamespace(spec=physical,replica_control_spec=control,workspace=object())
    counts = torch.zeros(8,48,dtype=torch.int32)
    header = [0,4096,31]
    raw = [header+[int(i==source)*32768 for i in range(48)] for source in range(8)]
    agreement = SimpleNamespace(table=counts,raw=raw,source_summary=[[0,4096,31,32768,0,32768,32768]]*8)
    x,ids = torch.empty(4096,5120),torch.zeros(4096,8,dtype=torch.int32)
    with patch.object(mxfp8_module,'_replica_agreement',return_value=agreement) as gather:
        result=mxfp8_module._gather_mx_replica_counts(x,ids,resources,None,31)
    assert result.counts is counts
    assert result.host_counts==[row[3:] for row in raw]
    assert result.source_summary is agreement.source_summary
    args=gather.call_args.args
    assert args[:3]==(x,ids,control)
    assert args[4] is resources.workspace and args[5]==31 and args[6](None,None,None) is None
    assert resources.spec is physical and not physical.auto_tokens and physical.num_experts==64


def test_large_agreement_does_not_force_a_second_host_readback():
    table=torch.zeros(16,96,dtype=torch.int32)
    agreement=SimpleNamespace(table=table,raw=None,source_summary=[])
    resources=SimpleNamespace(replica_control_spec=object(),workspace=object())
    with patch.object(mxfp8_module,'_replica_agreement',return_value=agreement):
        result=mxfp8_module._gather_mx_replica_counts(None,None,resources,None,31)
    assert result.counts is table and result.host_counts is None
