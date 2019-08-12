try:
    import torch
    import xentropy_cuda
    from .softmax_xentropy import SoftmaxCrossEntropyLoss
    del torch
    del xentropy_cuda
    del softmax_xentropy
except ImportError as err:
    print("apex was installed without --xentropy flag, contrib.xentropy is not available")
