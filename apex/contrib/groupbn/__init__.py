try:
    import torch
    import bnp
    from .batch_norm import BatchNorm2d_NHWC
    del torch
    del bnp
    del batch_norm
except ImportError as err:
    print("apex was installed without --bnp flag, contrib.groupbn is not available")
