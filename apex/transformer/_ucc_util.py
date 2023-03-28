from torch import distributed as dist

HAS_UCC = hasattr(dist, "is_ucc_available") and dist.is_ucc_available()
if not HAS_UCC:
    try:
        import torch_ucc
        HAS_UCC = True
    except ImportError:
        HAS_UCC = False
