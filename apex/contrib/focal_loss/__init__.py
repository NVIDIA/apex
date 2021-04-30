
try:
    import torch
    import focal_loss_cuda
    from .focal_loss import FocalLoss
    del torch
    del focal_loss_cuda
    del focal_loss
except ImportError as err:
    print("apex was installed without --focal-loss flag, contrib.focal_loss is not available")
