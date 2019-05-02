import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import functional as F

from .sync_batchnorm_kernel import SyncBatchnormFunction
from apex.parallel import ReduceOp


class SyncBatchNorm(_BatchNorm):
    """
    synchronized batch normalization module extented from ``torch.nn.BatchNormNd``
    with the added stats reduction across multiple processes.
    :class:`apex.parallel.SyncBatchNorm` is designed to work with
    ``DistributedDataParallel``.

    When running in training mode, the layer reduces stats across all processes
    to increase the effective batchsize for normalization layer. This is useful
    in applications where batch size is small on a given process that would
    diminish converged accuracy of the model. The model uses collective
    communication package from ``torch.distributed``.

    When running in evaluation mode, the layer falls back to
    ``torch.nn.functional.batch_norm``.

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

    Example::

        >>> sbn = apex.parallel.SyncBatchNorm(100).cuda()
        >>> inp = torch.randn(10, 100, 14, 14).cuda()
        >>> out = sbn(inp)
        >>> inp = torch.randn(3, 100, 20).cuda()
        >>> out = sbn(inp)
    """

    warned = False

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, process_group=None, channel_last=False):
        if channel_last == True:
            raise AttributeError("channel_last is not supported by primitive SyncBatchNorm implementation. Try install apex with `--cuda_ext` if channel_last is desired.")

        if not SyncBatchNorm.warned:
            print("Warning:  using Python fallback for SyncBatchNorm, possibly because apex was installed without --cuda_ext.  The exception raised when attempting to import the cuda backend was: ", self.syncbn_import_error)
            SyncBatchNorm.warned = True

        super(SyncBatchNorm, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.process_group = process_group

    def _specify_process_group(self, process_group):
        self.process_group = process_group

    def forward(self, input):
        torch.cuda.nvtx.range_push("sync_bn_fw_with_mean_var")
        mean = None
        var = None
        cast = None
        out = None

        # casting to handle mismatch input type to layer type
        if self.running_mean is not None:
            if self.running_mean.dtype != input.dtype:
                input = input.to(self.running_mean.dtype)
                cast = input.dtype
        elif self.weight is not None:
            if self.weight.dtype != input.dtype:
                input = input.to(self.weight.dtype)
                cast = input.dtype

        if not self.training and self.track_running_stats:
            # fall back to pytorch implementation for inference
            torch.cuda.nvtx.range_pop()
            out = F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
        else:
            process_group = self.process_group
            world_size = 1
            if not self.process_group:
                process_group = torch.distributed.group.WORLD
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
                    world_size = torch.distributed.get_world_size(process_group)
                    torch.distributed.all_reduce(
                        local_mean, ReduceOp.SUM, process_group)
                    mean = local_mean / world_size
                    torch.distributed.all_reduce(
                        local_sqr_mean, ReduceOp.SUM, process_group)
                    sqr_mean = local_sqr_mean / world_size
                    m = local_m * world_size
                else:
                    m = local_m
                    mean = local_mean
                    sqr_mean = local_sqr_mean
                # var(x) = E (( x - mean_x ) ** 2)
                #        = 1 / N * sum ( x - mean_x ) ** 2
                #        = 1 / N * sum (x**2) - mean_x**2
                var = sqr_mean - mean.pow(2)

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
            out = SyncBatchnormFunction.apply(input, self.weight, self.bias, mean, var, self.eps, process_group, world_size)
        out = out.to(cast)
