import torch
from torch.nn.parameter import Parameter
from torch.nn import Module

from .sync_batchnorm_kernel import SyncBatchnormFunction


class SyncBatchNorm(Module):
    r"""Applies Synced Batch Normalization over a > 2D input

    This layer has the same interface as with torch.nn.BatchNormNd.
    Synchronization meaning, for distributed training, nomalization is applied
    across all batches on different GPUs.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(SyncBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def forward(self, input):
        training = torch.is_grad_enabled()
        mean = None
        var = None

        if(training):
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
                    # TODO(jie): would a single broadcast save some communication time?
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
        else:
            # TODO(jie): what if we don't have track_running_stats & running in non-training? Does this apply that user would provide running_mean & running_var to the layer prior to execution?
            mean = self.running_mean
            var = self.running_var

        return SyncBatchnormFunction.apply(input, self.weight, self.bias, mean, var, self.eps)


# Quick drop-in replace hack
def replace_with_SYNCBN():
    torch.nn.BatchNorm2d = SyncBatchNorm
