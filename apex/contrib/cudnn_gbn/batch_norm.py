import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import functional as F
from torch import Tensor
import peer_memory_cuda as pm
import cudnn_gbn_lib
from torch.cuda.amp import custom_fwd, custom_bwd

class _GroupBatchNorm2d(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight, bias, running_mean, running_variance,
                minibatch_mean, minibatch_inv_var, momentum, eps, group_size, group_rank, fwd_buffers, bwd_buffers):
        ctx.save_for_backward(input, weight, minibatch_mean, minibatch_inv_var)
        ctx.eps = eps
        ctx.bn_group = group_size
        ctx.rank_id = group_rank
        ctx.peer_buffers = bwd_buffers
        return cudnn_gbn_lib.forward(input, weight, bias, running_mean, running_variance,
                                     minibatch_mean, minibatch_inv_var, momentum, eps, group_size, group_rank, fwd_buffers)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        x, scale, minibatch_mean, minibatch_inv_var = ctx.saved_variables
        eps = ctx.eps
        bn_group = ctx.bn_group
        rank_id = ctx.rank_id
        peer_buffers = ctx.peer_buffers
        dx, dscale, dbias = cudnn_gbn_lib.backward(x,
                       grad_output,
                       scale,
                       minibatch_mean,
                       minibatch_inv_var,
                       eps,
                       bn_group,
                       rank_id,
                       peer_buffers)
        return dx, dscale, dbias, None, None, None, None, None, None, None, None, None, None



class GroupBatchNorm2d(_BatchNorm):
    """
    synchronized batch normalization module extented from ``torch.nn.BatchNormNd``
    with the added stats reduction across multiple processes.

    When running in training mode, the layer reduces stats across process groups
    to increase the effective batchsize for normalization layer. This is useful
    in applications where batch size is small on a given process that would
    diminish converged accuracy of the model.

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

        >>> sbn = apex.contrib.GroupBatchNorm2d(100).cuda()
        >>> inp = torch.randn(10, 100, 14, 14).cuda()
        >>> out = sbn(inp)
        >>> inp = torch.randn(3, 100, 20).cuda()
        >>> out = sbn(inp)
    """

    def __init__(self, num_features, group_size, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):

        super(GroupBatchNorm2d, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.group_size = group_size
        rank = torch.distributed.get_rank()
        self.group_id = rank // group_size
        self.group_rank = rank % group_size
        self.fwd_peer_buffers = self.get_peer_buffers(num_features)
        self.bwd_peer_buffers = self.get_peer_buffers(num_features)
        self.minibatch_mean = torch.cuda.FloatTensor(num_features)
        self.minibatch_inv_var = torch.cuda.FloatTensor(num_features)

    def get_peer_buffers(self, num_features):
        # group_size * 2 (low-latency algo) * 2 (mean+var) * channels * 4 (float32)
        peer_size = self.group_size * 4 * num_features * 4
        raw = pm.allocate_raw(peer_size)
        # exchange peer pointers with nccl
        world_size = torch.distributed.get_world_size()
        raw_ipc = pm.get_raw_ipc_address(raw).cuda()
        raw_ipcs = [torch.empty_like(raw_ipc) for _ in range(world_size)]
        torch.distributed.all_gather(raw_ipcs, raw_ipc)
        group_ipcs = [raw_ipcs[x] for x in range(self.group_id * self.group_size, (self.group_id * self.group_size) + self.group_size)]
        peer_raw_ipcs = torch.stack(group_ipcs).cpu()
        return pm.get_raw_peers(peer_raw_ipcs, self.group_rank, raw)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError(
                "expected 4D input (got {}D input)".format(input.dim())
            )

    def _check_input_channels(self, input):
        if input.size(1) % 8 != 0:
            raise ValueError(
                "GroupBatchNorm2d number of input channels should be a multiple of 8"
            )

    def forward(self, input : Tensor) -> Tensor:
        # currently only GPU input is supported
        if not input.is_cuda:
            raise ValueError("GroupBatchNorm2d expected input tensor to be on GPU")
        if not input.is_contiguous(memory_format=torch.channels_last):
            raise ValueError("GroupBatchNorm2d expected input tensor to be in channels last memory format")
        if torch.is_autocast_enabled():
            input = input.to(torch.get_autocast_gpu_dtype())
        if input.dtype != torch.float16:
            raise ValueError("GroupBatchNorm2d expected input tensor in float16")
        self._check_input_dim(input)
        self._check_input_channels(input)

        if not self.training:
            # fall back to pytorch implementation for inference
            return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, False, self.momentum, self.eps)

        return _GroupBatchNorm2d.apply(input,
                                       self.weight, self.bias,
                                       self.running_mean, self.running_var,
                                       self.minibatch_mean, self.minibatch_inv_var,
                                       self.momentum,
                                       self.eps,
                                       self.group_size,
                                       self.group_rank,
                                       self.fwd_peer_buffers,
                                       self.bwd_peer_buffers)
