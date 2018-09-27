import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import functional as F

from .sync_batchnorm_kernel import SyncBatchnormFunction


class SyncBatchNorm(_BatchNorm):
    r"""Applies Synced Batch Normalization over a > 2D input

    This layer has the same interface as with torch.nn.BatchNormNd.
    Synchronization meaning, for distributed training, nomalization is applied
    across all batches on different GPUs.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)

    def forward(self, input):
        torch.cuda.nvtx.range_push("sync_bn_fw_with_mean_var")
        mean = None
        var = None
        if not self.training and self.track_running_stats:
            # fall back to pytorch implementation for inference
            torch.cuda.nvtx.range_pop()
            return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
        else:
            self.num_batches_tracked += 1
            with torch.no_grad():
                channel_first_input = input.transpose(0, 1).contiguous()
                squashed_input_tensor_view = channel_first_input.view(
                    channel_first_input.size(0), -1)
                # total number of data points for each variance entry. Used to calculate unbiased variance estimate
                m = None
                local_m = float(squashed_input_tensor_view.size()[1])
                local_mean = torch.mean(squashed_input_tensor_view, 1)
                local_sqr_mean = torch.pow(
                    squashed_input_tensor_view, 2).mean(1)
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(
                        local_mean, op=torch.distributed.reduce_op.SUM)
                    mean = local_mean / torch.distributed.get_world_size()
                    torch.distributed.all_reduce(
                        local_sqr_mean, op=torch.distributed.reduce_op.SUM)
                    sqr_mean = local_sqr_mean / torch.distributed.get_world_size()
                    m = local_m * torch.distributed.get_world_size()
                else:
                    m = local_m
                    mean = local_mean
                    sqr_mean = local_sqr_mean
                # var(x) = E (( x - mean_x ) ** 2)
                #        = 1 / N * sum ( x - mean_x ) ** 2
                #        = 1 / N * sum (x**2) - mean_x**2
                var = sqr_mean - mean.pow(2)

                if self.track_running_stats:
                    if self.running_mean is not None:
                        self.running_mean = self.momentum * mean + \
                            (1 - self.momentum) * self.running_mean
                    if self.running_var is not None:
                        # as noted by the paper, we used unbiased variance estimate of the mini-batch
                        # Var[x] = m / (m-1) * Eb (sample_variance)
                        self.running_var = m / \
                            (m-1) * self.momentum * var + \
                            (1 - self.momentum) * self.running_var
            torch.cuda.nvtx.range_pop()
            return SyncBatchnormFunction.apply(input, self.weight, self.bias, mean, var, self.eps)
