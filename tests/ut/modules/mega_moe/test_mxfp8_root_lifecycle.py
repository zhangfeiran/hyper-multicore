"""Backward must resolve freshly gathered physical storage on the original root."""
import pytest
import torch

from hyper_multicore import MegaMoeExperts
from hyper_multicore.experimental import HotReplicaPlanner, MegaMoeTuningOptions
from hyper_multicore.modules.mega_moe import module
from hyper_multicore.modules.mega_moe.numerics import MegaMoeNumericsOptions

from hyper_multicore.modules.mega_moe.backends.mxfp8.autograd import MXFP8ForwardState, execute_mxfp8_experts
from hyper_multicore.modules.mega_moe.backends.mxfp8.weights import MXFP8ExpertWeightBinding


def physical(root, shape):
    """Attach packed row and column weights to the logical root."""
    e, m, n = shape
    root.col_data = torch.empty(shape, dtype=torch.float8_e4m3fn)
    root.col_scale = torch.empty((e, m // 64, n, 2), dtype=torch.uint8)
    root.row_data = torch.empty(shape, dtype=torch.float8_e4m3fn)
    root.row_scale = torch.empty((e, m, n // 64, 2), dtype=torch.uint8)


def test_hot_replica_requires_fp32_roots_and_ordinary_mx_accepts_bf16(monkeypatch):
    monkeypatch.setattr(module, "_resolve_ep_size", lambda _group: 2)
    monkeypatch.setattr(torch.Tensor, "is_npu", property(lambda _tensor: True))
    args = dict(hidden_size=64, intermediate_size=64, num_experts=2, top_k=1, precision="mxfp8")
    ordinary = MegaMoeExperts(**args)
    placed = MegaMoeExperts(
        **args, numerics_options=MegaMoeNumericsOptions(partial_weight_grad_dtype="float32"),
        tuning_options=MegaMoeTuningOptions(expert_placement=HotReplicaPlanner(replica_slots_per_rank=1)),
    )
    hidden = torch.empty(0, 64, dtype=torch.bfloat16)
    shapes = ((1, 64, 128), (1, 64, 64))
    try:
        for dtype in (torch.bfloat16, torch.float32):
            roots = tuple(torch.empty(shape, dtype=dtype) for shape in shapes)
            for root, shape in zip(roots, shapes):
                physical(root, shape)
            ordinary._validate_tensors(hidden, roots)
            if dtype == torch.float32:
                placed._validate_tensors(hidden, roots)
            else:
                with pytest.raises(ValueError, match="FP32 expert roots"):
                    placed._validate_tensors(hidden, roots)
    finally:
        ordinary.close()
        placed.close()


def test_missing_backward_storage_reports_required_axis():
    """Report the missing row axis when resolving backward weights."""
    shape = (1, 64, 128)
    root = torch.empty(shape, dtype=torch.bfloat16)
    physical(root, shape)
    del root.row_data
    binding = MXFP8ExpertWeightBinding(root, shape)
    binding.resolve('forward')
    with pytest.raises(ValueError, match='missing required row data/scale'):
        binding.resolve('backward')


@pytest.mark.parametrize("row_only", [True, False])
def test_fresh_physical_root_in_backward_and_no_grad_retention(row_only):
    """Verify phase-local physical weights and inference lifetime."""
    shapes = ((1, 64, 128), (1, 64, 64))
    roots = [torch.nn.Parameter(torch.empty(shape, dtype=torch.bfloat16)) for shape in shapes]
    for root, shape in zip(roots, shapes):
        physical(root, shape)
    x = torch.ones((4, 64), dtype=torch.bfloat16, requires_grad=True)
    probs = torch.ones((4, 1), requires_grad=True)

    class Execution:
        """Record which physical parameters each phase consumes."""
        def forward(self, hidden, route, probabilities, w1, w2, retained):
            """Record the forward weight and retain the requested state."""
            del route, probabilities, w2
            self.retained = retained
            assert w1.col_data.data_ptr() == roots[0].col_data.data_ptr()
            assert w1.row_data is None
            return MXFP8ForwardState(hidden.clone(), (), None)

        def backward(self, grad, saved, metadata, w1, w2):
            """Check that backward consumes the newly materialized weights."""
            del saved, metadata
            for weight, root in zip((w1, w2), roots):
                assert weight.row_data.data_ptr() == root.row_data.data_ptr()
                assert weight.row_scale.data_ptr() == root.row_scale.data_ptr()
                assert weight.col_data is None and weight.col_scale is None
            return grad, torch.zeros_like(probs), *(torch.zeros_like(root) for root in roots)

    execution = Execution()
    y = execute_mxfp8_experts(x, *roots, None, probs, shapes, execution)
    assert execution.retained
    for root, shape in zip(roots, shapes):
        physical(root, shape)
        if row_only:
            root.col_data = root.col_scale = None
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    assert all(torch.equal(root.grad, torch.zeros_like(root)) for root in roots)
    for root, shape in zip(roots, shapes):
        physical(root, shape)
        root.row_data = root.row_scale = None
    with torch.no_grad():
        execute_mxfp8_experts(x, *roots, None, probs, shapes, execution)
    assert not execution.retained
