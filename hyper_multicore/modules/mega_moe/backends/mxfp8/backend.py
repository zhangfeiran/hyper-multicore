# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
"""MX experts reuse the managed MoE lifecycle and precision-independent transport."""
from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import cached_property

import torch

from hyper_multicore.scheduler.runtime import serialize_runtime_config
from ...resources import _MegaMoeExecutionResources
from .diagnostics import prepare_mx_diagnostics
from .execution import MXFP8Execution
from .autograd import execute_mxfp8_experts
from .graph import build_mxfp8_runtime
from .weights import MXFP8ExpertWeightBinding


def _device_bytes(data, device):
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).to(device=device)


@dataclass
class _DeviceRuntime:
    """Hold device copies of an immutable MX runtime and its descriptors."""
    runtime: torch.Tensor
    gmm: torch.Tensor
    quant: torch.Tensor
    event_capacity: int
    host: object
    spec: object

    @cached_property
    def diagnostics(self):
        """Build the diagnostic binding only on first use."""
        return prepare_mx_diagnostics(self.host, self.spec, self.runtime.device)


def _device_runtime(host, device, spec):
    return _DeviceRuntime(_device_bytes(serialize_runtime_config(host.config), device),
                          _device_bytes(host.gmm_descriptors, device),
                          _device_bytes(host.quant_descriptors, device), host.config.event_capacity, host, spec)


@dataclass(frozen=True)
class _MXFP8Plan:
    spec: object
    forward: _DeviceRuntime
    backward: _DeviceRuntime


class Resources(_MegaMoeExecutionResources):
    """Own MX execution plans for the bound workspace."""
    def __init__(self, specification, tensor, *, expert_weight_layout, shared, active_specifications):
        super().__init__(
            specification, tensor, expert_weight_layout=expert_weight_layout, shared=shared,
            active_specifications=active_specifications,
        )
        self._plans = OrderedDict()
        self._plan_cache_size = 8 if self.placement is not None else 4

    def _initial_plan(self, tensor):
        # Capability rejection participates in the common construction collective.
        if self.spec.architecture != 3510:
            raise ValueError("MXFP8 experts require arch35")

    def plan_for_capacity(self, capacity, device, *, tokens=None, route_experts=None):
        """Cache pointer-free plans by graph width and committed capacities."""
        coverage = self.spec.local_num_tokens if tokens is None else tokens
        if coverage > self._token_capacity:
            raise ValueError("MX plan exceeds committed source capacity")
        if self.spec.dispatch_mode == 'pull':
            limit = min(self.spec.ep_size * coverage * self.spec.top_k, (1 << 32) - 1)
            if not 1 <= capacity <= limit:
                raise ValueError(f"Pull receive capacity must be in [1, {limit}]")
            capacity = limit
        key = (coverage, capacity, route_experts)
        if key not in self._plans:
            spec = replace(self._route_spec(route_experts), local_num_tokens=coverage, receive_capacity=capacity)
            home_experts = self.spec.local_experts if route_experts is not None else None
            forward = build_mxfp8_runtime(spec, forward=True, home_experts=home_experts)
            backward = build_mxfp8_runtime(spec, forward=False, home_experts=home_experts,
                                          gradient_helpers=home_experts is not None)
            self._plans[key] = _MXFP8Plan(spec, _device_runtime(forward, device, spec),
                                        _device_runtime(backward, device, spec))
        self._plans.move_to_end(key)
        while len(self._plans) > self._plan_cache_size:
            self._plans.popitem(last=False)
        return self._plans[key]


def validate_weights(module, weights, hidden):
    """Check logical shapes and physical forward inputs."""
    shapes = ((module.local_experts, module.hidden_size, 2 * module.intermediate_size),
              (module.local_experts, module.intermediate_size, module.hidden_size))
    for root, shape in zip(weights, shapes):
        binding = MXFP8ExpertWeightBinding(root, shape)
        physical = binding.resolve("forward")
        if any(t.device != hidden.device for t in (root, physical.col_data, physical.col_scale)):
            raise ValueError("MX physical weights and hidden states must share a device")


def forward(module, hidden_states, hidden_flat, topk_ids, topk_weights, weights, resources, route):
    """Launch MX experts over the route already prepared by the frontend."""
    del hidden_flat, topk_ids
    shapes = ((module.local_experts, module.hidden_size, 2 * module.intermediate_size),
              (module.local_experts, module.intermediate_size, module.hidden_size))
    binding = route.weight_binding
    if binding is None:
        execution = MXFP8Execution(resources, nograd_no_clone=module.nograd_no_clone)
    else:
        execution = MXFP8Execution(resources, nograd_no_clone=module.nograd_no_clone,
                                   placement=binding.placement, weight_prefetch=binding.prefetch,
                                   gradient_pool=binding.gradients)
    return execute_mxfp8_experts(hidden_states, *weights, route, topk_weights, shapes, execution)
