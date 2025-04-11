"""Configurations for graph scheduler."""

import os
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

from torch.utils._config_module import install_config_module  # noqa: E402

# adds patch, save_config, etc
install_config_module(sys.modules[__name__])
