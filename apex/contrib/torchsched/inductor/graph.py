"""Scheduling abstractions on PyTorch Inductor GraphLowering level."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from torch._inductor.graph import GraphLowering
from torch._inductor.scheduler import Scheduler
from torch._inductor.virtualized import V

if TYPE_CHECKING:
    from torch.fx.node import Node

from apex.contrib.torchsched.inductor.scheduler import MultiCudaStreamScheduler


@functools.wraps(GraphLowering.codegen)
def _codegen(graph: GraphLowering) -> tuple[str, list[tuple[int, Node]]]:
    graph.init_wrapper_code()

    if graph.device_type == "cuda":
        graph.scheduler = MultiCudaStreamScheduler(graph.operations)
    else:
        graph.scheduler = Scheduler(graph.operations)
    V.debug.draw_orig_fx_graph(graph.orig_gm, graph.scheduler.nodes)

    graph.wrapper_code.push_codegened_graph(graph)
    graph.scheduler.codegen()
    result = graph.wrapper_code.generate(graph.is_inference)
    graph.wrapper_code.pop_codegened_graph()

    return result


_origin_codegen = GraphLowering.codegen


def patch_graph_lowering(patch: bool = True) -> None:
    """Patch PyTorch Inductor lowerings with multi-stream scheduling.

    This function patches the `torch.compile` stack on the GraphLowering level,
    i.e., the compute graph has been captured by Dynamo and it has undergone
    post-auto-gradient passes, including pattern-matching optimizations and
    preliminary operator fusions. At that point, most nodes in the graph are
    either fused Triton templates, or function calls to external libraries. The
    multi-stream scheduler then finds the longest critical path in this graph,
    and schedule other nodes to side streams to exploit the inherent parallelism
    of the given compute graph.

    Args:
        patch: Whether to patch Inductor `GraphLowering` with multi-stream
            scheduler. Set to `False` to restore the default `torch.compile`
            behavior. (default: `True`)
    """
    if patch:
        GraphLowering.codegen = _codegen
    else:
        GraphLowering.codegen = _origin_codegen
