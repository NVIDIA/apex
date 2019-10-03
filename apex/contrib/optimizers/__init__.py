from .fp16_optimizer import FP16_Optimizer
try:
    import torch
    import fused_adam_cuda
    from .fused_adam import FusedAdam
    del torch
    del fused_adam_cuda
    del fused_adam
except ImportError as err:
    print("apex was installed without --fusedadam flag, contrib.fused_adam is not available")
