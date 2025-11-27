"""Configurations for graph scheduler."""

import functools
import os
import re
import sys

# Debug info and dump grpahs
debug = os.getenv("TORCH_SCHED_DEBUG", "0") == "1"

# Toggle pre_grad_pass for various pattern matches
enable_pre_grad_pass = False

# Pre grad pass patterns
pre_grad_pass_options: list[str] = ["cudnn_layer_norm"]

# Number of CUDA streams used for multi-stream scheduling.
# The first stream will be critical path stream, operators on non-critical path will be
# scheduled to other streams in a round-robin way.
num_streams = int(os.getenv("TORCH_SCHED_NUM_STREAMS", "8"))


def _get_skip_post_grad_graph_ids() -> set[int]:
    if ids := os.environ.get("TORCH_SCHED_SKIP_GRAPH_IDS"):
        result: set[int] = set()
        for part in ids.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                result.update(range(start, end + 1))
            else:
                result.add(int(part))
        return result
    else:
        return set()


# IDs of post AOT-autograd graphs that should be skipped for multi-stream scheduling. Can be
# specified via TORCH_SCHED_SKIP_GRAPH_IDS environment variable in a SLURM-like scheme, e.g.,
# TORCH_SCHED_SKIP_GRAPH_IDS=1,2,3-5,7-10
skip_post_grad_graph_ids: set[int] = _get_skip_post_grad_graph_ids()

# Reduce the number of allocated CUDA Events in the generated program by:
# 1. Track reference count of each CUDA Event in the scheduling phase. Skip generating CUDA Events
#    that have no reference counts, i.e., have not been waited by other streams;
# 2. Reuse allocated CUDA Events when feasible.
# This option is enable by default.
reuse_cuda_event: bool = os.getenv("TORCH_SCHED_REUSE_CUDA_EVENT", "1") == "1"


@functools.lru_cache
def __get_dump_code_backends_and_dir(
    dump_code: str | None,
) -> tuple[list[str], str | None]:
    pattern = r"(?:\+(?P<backend>\w+),)?(?P<dir>[\w\/\.\-\s@#~]+)"
    backends, dir = ["torchsched"], None
    if dump_code and (match := re.match(pattern, dump_code)):
        if backend := match.group("backend"):
            backends.append(backend)
        dir = os.path.abspath(match.group("dir"))
    return backends, dir


# Specify dump code backend types and output directory by::
#
#   TORCH_SCHED_DUMP_CODE='+inductor,/dir/to/save/code'
#
# Where `+inductor` enables dump both Inductor and torchsched code. If omitted, only dump
# torchsched code. `/dir/to/save/code` specifies a directory to dump code to.
(
    dump_code_backends,
    dump_code_dir,
) = __get_dump_code_backends_and_dir(os.getenv("TORCH_SCHED_DUMP_CODE"))

from torch.utils._config_module import install_config_module  # noqa: E402

# adds patch, save_config, etc
install_config_module(sys.modules[__name__])
