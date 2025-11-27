"""Scheduling abstractions on PyTorch Inductor GraphLowering level."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch._inductor.codegen.common import get_scheduling_for_device
from torch._inductor.codegen.common import get_wrapper_codegen_for_device
from torch._inductor.codegen.common import register_backend_for_device
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.graph import GraphLowering
from torch._inductor.scheduler import Scheduler
from torch._inductor.virtualized import V

if TYPE_CHECKING:
    from torch._inductor.utils import ValueWithLineMap

from apex.contrib.torchsched import config as torchsched_config
from apex.contrib.torchsched.inductor.scheduler import MultiCudaStreamScheduler
from apex.contrib.torchsched.inductor.wrapper import MultiStreamWrapperCodegen

_inductor_codegen = GraphLowering.codegen
patching_device_type = "cuda"
schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")


@functools.wraps(GraphLowering.codegen)
def _torchsched_codegen(
    graph: GraphLowering,
) -> tuple[ValueWithLineMap, ValueWithLineMap]:
    # Move patching logic here as post_grad_graph_id was not available until now.
    cpp_wrapper_cls = get_wrapper_codegen_for_device(patching_device_type, cpp_wrapper=True)
    only_cpu = len(graph.device_types - {"cpu", "meta"}) == 0
    scheduling_cls = get_scheduling_for_device(patching_device_type)
    wrapper_cls = get_wrapper_codegen_for_device(patching_device_type)
    write_get_raw_stream = PythonWrapperCodegen.write_get_raw_stream
    if not only_cpu and graph.post_grad_graph_id not in torchsched_config.skip_post_grad_graph_ids:
        patched_scheduler_cls = MultiCudaStreamScheduler
        patched_wrapper_cls = MultiStreamWrapperCodegen
        # torch.compile explicitly calls `write_get_raw_stream` via wrapper's class method in its
        # lowering process to walk around the wrapper-stream LRU cache mechanism. To be compatible
        # with this, we got to patch wrapper's class method as well.
        PythonWrapperCodegen.write_get_raw_stream = MultiStreamWrapperCodegen._write_get_raw_stream
    else:
        patched_scheduler_cls = Scheduler
        patched_wrapper_cls = PythonWrapperCodegen
    register_backend_for_device(
        device=patching_device_type,
        device_scheduling=scheduling_cls,
        device_wrapper_codegen=patched_wrapper_cls,
        device_cpp_wrapper_codegen=cpp_wrapper_cls,
    )

    graph.init_wrapper_code()
    graph.scheduler = patched_scheduler_cls(graph.operations)
    V.debug.draw_orig_fx_graph(graph.orig_gm, graph.scheduler.nodes)
    graph.wrapper_code.push_codegened_graph(graph)
    graph.scheduler.codegen()
    result = graph.wrapper_code.generate(graph.is_inference)
    graph.wrapper_code.pop_codegened_graph()

    PythonWrapperCodegen.write_get_raw_stream = write_get_raw_stream
    register_backend_for_device(
        device=patching_device_type,
        device_scheduling=scheduling_cls,
        device_wrapper_codegen=wrapper_cls,
        device_cpp_wrapper_codegen=cpp_wrapper_cls,
    )

    return result


@functools.wraps(GraphLowering.codegen)
def _mixed_codegen(graph: GraphLowering) -> tuple[ValueWithLineMap, ValueWithLineMap]:
    assert torchsched_config.dump_code_dir
    output_code_per_backend: dict[str, tuple[ValueWithLineMap, ValueWithLineMap]] = {}

    for backend in torchsched_config.dump_code_backends:
        if backend == "torchsched":
            codegen = _torchsched_codegen
        elif backend == "inductor":
            codegen = _inductor_codegen
        else:
            raise ValueError(f"Unknown {backend=} from {torchsched_config.dump_code_backends=}")
        wrapper_code, kernel_code = codegen(graph)
        output_code_per_backend[backend] = (wrapper_code, kernel_code)

    for backend, (wrapper_code, kernel_code) in output_code_per_backend.items():
        backend_dir = Path(torchsched_config.dump_code_dir) / backend
        backend_dir.mkdir(parents=True, exist_ok=True)
        graph_id = graph.post_grad_graph_id
        (backend_dir / f"graph_{graph_id}_wrapper_code.py").write_text(wrapper_code.value)
        if kernel_code.value.strip():
            # Kernel_code is only available in AOTInductor mode.
            (backend_dir / f"graph_{graph_id}_kernel_code.py").write_text(kernel_code.value)

    return output_code_per_backend["torchsched"]


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
    if patch and torchsched_config.dump_code_dir:
        GraphLowering.codegen = _mixed_codegen
    elif patch:
        GraphLowering.codegen = _torchsched_codegen
    else:
        GraphLowering.codegen = _inductor_codegen
