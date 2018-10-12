import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import functional as F

from .optimized_sync_batchnorm_kernel import SyncBatchnormFunction


class SyncBatchNorm(_BatchNorm):
    """
    synchronized batch normalization module extented from `torch.nn.BatchNormNd`
    with the added stats reduction across multiple processes.
    :class:`apex.parallel.SyncBatchNorm` is designed to work with
    `DistributedDataParallel`.

    When running in training mode, the layer reduces stats across all processes
    to increase the effective batchsize for normalization layer. This is useful
    in applications where batch size is small on a given process that would
    diminish converged accuracy of the model. The model uses collective
    communication package from `torch.distributed`.

    When running in evaluation mode, the layer falls back to
    `torch.nn.functional.batch_norm`

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, L)` or :math:`L` from input of size :math:`(N, L)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``

    Examples::
        >>> sbn = apex.parallel.SyncBatchNorm(100).cuda()
        >>> inp = torch.randn(10, 100, 14, 14).cuda()
        >>> out = sbn(inp)
        >>> inp = torch.randn(3, 100, 20).cuda()
        >>> out = sbn(inp)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def forward(self, input):
        if not self.training and self.track_running_stats:
            # fall back to pytorch implementation for inference
            return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
        else:
            self.num_batches_tracked += 1
            return SyncBatchnormFunction.apply(input, self.weight, self.bias, self.running_mean, self.running_var, self.eps, self.track_running_stats, self.momentum)
