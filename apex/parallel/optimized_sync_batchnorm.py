import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import functional as F

from .optimized_sync_batchnorm_kernel import SyncBatchnormFunction


class SyncBatchNorm(_BatchNorm):
    r"""Applies Synced Batch Normalization over a > 2D input

    This layer has the same interface as with torch.nn.BatchNormNd.
    Synchronization meaning, for distributed training, nomalization is applied
    across all batches on different GPUs.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)

    def forward(self, input):
        if not self.training and self.track_running_stats:
            # fall back to pytorch implementation for inference
            return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
        else:
            self.num_batches_tracked += 1
            return SyncBatchnormFunction.apply(input, self.weight, self.bias, self.running_mean, self.running_var, self.eps, self.track_running_stats, self.momentum)
