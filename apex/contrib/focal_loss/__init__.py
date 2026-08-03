try:
    import focal_loss_cuda
    import torch

    from .focal_loss import focal_loss

    del torch
    del focal_loss_cuda
    del focal_loss
except ImportError:
    print("apex was installed without --focal_loss flag, apex.contrib.focal_loss is not available")
