# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""Check complete transport ownership and unchanged math for the B0 QP0 policy."""
from collections import Counter
from dataclasses import replace
import os
from unittest.mock import patch

import pytest

from hyper_multicore.modules.mega_moe.mxfp8_graph import build_mxfp8_runtime
from hyper_multicore.modules.mega_moe.mxfp8_module import _mx_peer_qp0_option
from hyper_multicore.modules.mega_moe.spec import MegaMoeSpec
from hyper_multicore.scheduler.window_queue import MX_PEER_QP0_TAG


@pytest.mark.parametrize('ep,qps', [(4,1),(8,8),(16,16),(64,32)])
@pytest.mark.parametrize('forward', [False,True])
def test_peer_owners_preserve_all_tasks_math_and_credits(ep, qps, forward):
    spec=MegaMoeSpec(4096,5120,1792,ep*6,8,1.25,40960,ep,None,ep-1,32,
        mx_vector_policy='window2',dispatch_split=512,combine_split=512,compute_stages=6)
    with patch.dict(os.environ, {'HYPER_MULTICORE_SHMEM_DATA_ENGINE':'mte_udma',
            'HYPER_MULTICORE_SHMEM_UDMA_QP_NUM':'32', 'PANGU_MX_REUSE_DISPATCH_WORKERS':'0'}):
        before=build_mxfp8_runtime(spec,forward=forward)
        os.environ['HYPER_MULTICORE_SHMEM_UDMA_QP_NUM']=str(qps)
        after=build_mxfp8_runtime(replace(spec,mx_peer_qp0=True),forward=forward)
    assert before.gmm_descriptors==after.gmm_descriptors
    assert before.quant_descriptors==after.quant_descriptors
    a,b=before.config,after.config
    assert bytes(a.cube_task_indices)==bytes(b.cube_task_indices)
    assert bytes(a.all_event_num_triggers)==bytes(b.all_event_num_triggers)
    for tid in range(a.task_num):
        original,current=a.all_tasks[tid],type(b.all_tasks[tid]).from_buffer_copy(bytes(b.all_tasks[tid]))
        if int(original.task_type) in (105,141):
            assert current.extra_value_0==MX_PEER_QP0_TAG and current.extra_value_2==1
            current.extra_value_0=original.extra_value_0
            current.extra_value_2=original.extra_value_2
        assert bytes(original)==bytes(current)
    owners={};coverage=[]
    for worker in range(64):
        queue=list(b.vector_task_indices[b.mix_task_indices[worker]:b.mix_task_indices[worker+1]])
        old=list(a.vector_task_indices[a.mix_task_indices[worker]:a.mix_task_indices[worker+1]])
        if worker%2==0:assert queue==old
        coverage+=queue
        for tid in queue:
            task=b.all_tasks[tid]
            if int(task.task_type) not in (105,141):continue
            peer=task.task_index//(task.task_split_num//ep)
            if peer!=spec.rank_id:
                expected=2*(((peer-spec.rank_id)%ep)%32)+1
                assert worker==expected
                assert owners.setdefault(peer,worker)==worker
    assert len(owners)==ep-1
    assert Counter(coverage)==Counter(a.vector_task_indices[:a.task_index_num[1]])
    assert len(coverage)==len(set(coverage))


@pytest.mark.parametrize('replicas,policy,source,engine', [
    (1,'window2',False,'mte_udma'), (0,'legacy_odd',False,'mte_udma'),
    (0,'window2',True,'mte_udma'), (0,'window2',False,'mte')])
def test_peer_qp0_rejects_incompatible_modes(replicas,policy,source,engine):
    with patch.dict(os.environ,{'PANGU_MX_PEER_QP0':'1','HYPER_MULTICORE_SHMEM_DATA_ENGINE':engine}):
        with pytest.raises(ValueError,match='B0 window2'):
            _mx_peer_qp0_option(replicas,policy,source)


def test_peer_qp0_is_explicit_and_frozen():
    with patch.dict(os.environ,{'PANGU_MX_PEER_QP0':'0'}):
        assert not _mx_peer_qp0_option(0,'window2',False)
    with patch.dict(os.environ,{'PANGU_MX_PEER_QP0':'1','HYPER_MULTICORE_SHMEM_DATA_ENGINE':'mte_udma'}):
        assert _mx_peer_qp0_option(0,'window2',False)
    with patch.dict(os.environ,{'PANGU_MX_PEER_QP0':'auto'}):
        with pytest.raises(ValueError,match='0 or 1'):_mx_peer_qp0_option(0,'window2',False)


def test_peer_qp0_uses_a_separate_launch_version():
    import torch
    from hyper_multicore.modules.mega_moe.mxfp8_ops import serialize_mxfp8_launch
    from hyper_multicore.scheduler.mxfp8 import MxLaunchDescriptor
    spec=MegaMoeSpec(4096,5120,1792,48,8,1.25,40960,8,None,0,32,
                    mx_vector_policy='window2',compute_stages=6)
    buffers=[torch.zeros(64,dtype=torch.uint8) for _ in range(27)]
    runtime=torch.zeros(1024,dtype=torch.uint8)
    events=torch.zeros(65536,dtype=torch.uint8)
    old=serialize_mxfp8_launch(buffers,runtime,events,spec)
    new=serialize_mxfp8_launch(buffers,runtime,events,replace(spec,mx_peer_qp0=True))
    record=MxLaunchDescriptor.from_buffer_copy(new)
    assert (record.version,record.reserved)==(15,14)
    record.version,record.reserved=6,5
    assert bytes(record)==old
    with pytest.raises(ValueError,match='cannot share'):
        serialize_mxfp8_launch(buffers,runtime,events,replace(spec,mx_peer_qp0=True),kernel_gradients=True)


def test_peer_qp0_capability_requires_native_probe_and_matching_object(tmp_path):
    from types import SimpleNamespace
    import hashlib,json
    from hyper_multicore.modules.mega_moe import mxfp8_ops
    vendor=tmp_path/'lib/vendors/hyper_multicore_nn'
    binary=vendor/'op_impl/ai_core/tbe/kernel/arch35/pangu_mxfp8_task_graph/PanguMxfp8TaskGraph_ed65f2bbd39f9cc68999acec5148f3f8.o'
    binary.parent.mkdir(parents=True);binary.write_bytes(b'qualified native payload')
    receipt=tmp_path/'lib/capabilities/mxfp8_window.json';receipt.parent.mkdir()
    receipt.write_text(json.dumps(dict(schema=7,object=binary.name,features=[],sha256=hashlib.sha256(binary.read_bytes()).hexdigest())))
    with patch.object(mxfp8_ops,'_load_mxfp8_native'),patch.object(mxfp8_ops,'get_multicore_paths',return_value=(vendor,None)),patch.object(mxfp8_ops.torch.ops,'hyper_multicore',SimpleNamespace()):
        assert not mxfp8_ops.mxfp8_window_feature_available('mx_peer_qp0')
        mxfp8_ops.require_mxfp8_window_feature.cache_clear()
        with pytest.raises(RuntimeError,match='unavailable'):
            mxfp8_ops.require_mxfp8_window_feature('mx_peer_qp0')
        mxfp8_ops.torch.ops.hyper_multicore.mxfp8_peer_qp0_version=lambda:1
        assert mxfp8_ops.mxfp8_window_feature_available('mx_peer_qp0')
        binary.write_bytes(b'stale binary')
        with pytest.raises(RuntimeError,match='does not match'):
            mxfp8_ops.mxfp8_window_feature_available('mx_peer_qp0')
    mxfp8_ops.require_mxfp8_window_feature.cache_clear()
