"""Custom PyTorch operators."""

import torch

__all__: list[str] = []

# Register custom operators
torch.ops.import_module("apex.contrib.torchsched.ops.layer_norm")
