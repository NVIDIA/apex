try:
    import torch
    from apex._extensions import focal_loss_cuda
    from .focal_loss import focal_loss

    del torch
    del focal_loss_cuda
    del focal_loss
except ImportError:
    print("apex was installed without --focal_loss flag, apex.contrib.focal_loss is not available")
