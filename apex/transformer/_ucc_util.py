from packaging.version import Version, parse

import torch
from torch import distributed as dist
from torch.utils import collect_env

_HAS_TORCH_UCC = None
try:
    import torch_ucc
    _HAS_TORCH_UCC = True
except ImportError:
    _HAS_TORCH_UCC = False
HAS_UCC = dist.is_ucc_available() or _HAS_TORCH_UCC
