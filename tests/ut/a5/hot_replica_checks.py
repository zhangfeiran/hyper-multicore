# Copyright 2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""CPU route evidence helpers shared by replica transport tests."""

import torch

from hyper_multicore.replica.routing import ReplicaRoute


def saved_route(output: torch.Tensor) -> ReplicaRoute:
    """Find the route actually retained by the autograd graph."""
    pending, visited = [output.grad_fn], set()
    while pending:
        node = pending.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        for name in ("replica_route", "route"):
            route = getattr(node, name, None)
            if isinstance(route, ReplicaRoute):
                return route
        pending.extend(parent for parent, _ in node.next_functions)
    raise AssertionError("Candidate output has no saved replica route")


def route_evidence(output: torch.Tensor, planner: str, *, home_route: ReplicaRoute | None = None) -> dict:
    """Check the selected planner and expose its actual immutable route."""
    if home_route is not None and home_route.plan.transfers:
        raise AssertionError("A home-only route cannot bypass saved replica-plan checks")
    route = saved_route(output) if home_route is None else home_route
    actual = "device" if route.device_plan is not None else "cpu"
    if actual != planner:
        raise AssertionError(f"Requested planner {planner}, executed {actual}")
    return {"planner": actual, "slots": route.plan.slot_to_logical,
            "counts": route.plan.destination_counts, "transfers": len(route.plan.transfers)}
