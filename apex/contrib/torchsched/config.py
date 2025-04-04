"""Configurations for graph scheduler."""""

import sys

# Pre grad pass patterns
pre_grad_pass_options: list[str] = ["cudnn_layer_norm"]

from torch.utils._config_module import install_config_module  # noqa: E402

# adds patch, save_config, etc
install_config_module(sys.modules[__name__])
