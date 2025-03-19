"""Scheduling abstractions on PyTorch Inductor level."""

from apex.contrib.torchsched.inductor.graph import patch_graph_lowering

__all__ = ["patch_graph_lowering"]
