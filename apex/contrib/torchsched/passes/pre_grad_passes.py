"""Customized Inductor passes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch._dynamo.utils import counters
from torch.fx import replace_pattern

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

from apex.contrib.torchsched import config

__all__ = ["pre_grad_custom_pass"]

# pass name to (pattern replacement) mapping
PRE_GRAD_PASS_PATTERNS: dict[str, tuple[Callable, Callable]] = {}


def register_pattern(name: str, pattern: Callable, replacement: Callable) -> None:
    assert name not in PRE_GRAD_PASS_PATTERNS
    PRE_GRAD_PASS_PATTERNS[name] = pattern, replacement


def replace_layer_norm(
    x: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    y, x_mean, x_invstd = torch.ops.cudnn.layer_norm(
        x,
        normalized_shape,
        weight,
        bias,
        eps,
    )
    return y


register_pattern(
    "cudnn_layer_norm",
    torch.nn.functional.layer_norm,
    replace_layer_norm,
)


def run_pre_grad_pass(
    name: str,
    graph: torch.fx.Graph,
    pattern: Callable,
    replacement: Callable,
) -> int:
    """Run a pre-gradient pass on the given graph.

    Args:
        name (str): A string identifier for the pass.
        graph (torch.fx.Graph): The graph to be transformed.
        pattern (Callable): A callable that defines the pattern to match in the graph.
        replacement (Callable): A callable that defines the replacement for matched patterns.

    Returns:
        An integer representing the number of transformations applied.

    Note:
        These two doesn't match because of kwargs (Inductor vs. torch.fx.symbolic_trace):

            %layer_norm : [num_users=1] = call_function[target=torch.nn.functional.layer_norm](
                args = (%l_args_0_, (320,), %l_fn_parameters_weight_, %l_fn_parameters_bias_,
                1e-05), kwargs = {})
            %layer_norm : [num_users=1] = call_function[target=torch.nn.functional.layer_norm](
                args = (%input_1, %normalized_shape), kwargs = {weight: %weight, bias: %bias,
                eps: %eps})
    """
    # Manually trace the graph and move kwargs to args
    pattern_graph = torch.fx.symbolic_trace(pattern).graph
    for node in pattern_graph.nodes:
        if node.op == "call_function" and node.target == pattern:
            node.args = node.args + tuple(node.kwargs.values())
            node.kwargs = {}
    pattern_graph.owning_module.recompile()

    matched = replace_pattern(graph.owning_module, pattern_graph, replacement)
    graph.owning_module.recompile()
    graph.lint()

    return len(matched)


def pre_grad_custom_pass(graph: torch.fx.Graph) -> None:
    """Run customized pre-grad passes.

    Args:
        graph (torch.fx.Graph): The FX graph to be optimized.
    """
    passes = config.pre_grad_pass_options
    for pass_name in passes:
        assert pass_name in PRE_GRAD_PASS_PATTERNS, f"Unknown pre_grad pass: {pass_name}"
        pattern, replacement = PRE_GRAD_PASS_PATTERNS[pass_name]
        replaced = run_pre_grad_pass(pass_name, graph, pattern, replacement)
        counters["torchsched"][f"pre_grad_{pass_name}"] += replaced
        logging.debug("Pre grad pass %s replaced %d sub-graphs", pass_name, replaced)
