import torch
from torch.autograd.function import Function

import syncbn
from apex.parallel import ReduceOp

class SyncBatchnormFunction(Function):

    @staticmethod
    def forward(ctx, input, z, weight, bias, running_mean, running_variance, eps, track_running_stats = True, momentum = 1.0, process_group = None, channel_last = False, fuse_relu = False):
        input = input.contiguous()
        world_size = 0

        mean = None
        var_biased = None
        inv_std = None
        var = None
        out = None
        count = None
        if track_running_stats:
            if channel_last:
                count = int(input.numel()/input.size(-1))
                mean, var_biased = syncbn.welford_mean_var_c_last(input)
                num_channels = input.size(-1)
            else:
                count = int(input.numel()/input.size(1))
                mean, var_biased = syncbn.welford_mean_var(input)
                num_channels = input.size(1)

            if torch.distributed.is_initialized():
                if not process_group:
                    process_group = torch.distributed.group.WORLD
                device = mean.device
                world_size = torch.distributed.get_world_size(process_group)

                count_t = torch.empty(1, dtype=mean.dtype, device=mean.device).fill_(count)
                combined = torch.cat([mean.view(-1), var_biased.view(-1), count_t], dim=0)
                combined_list = [torch.empty_like(combined) for k in range(world_size)]
                torch.distributed.all_gather(combined_list, combined, process_group)
                combined = torch.stack(combined_list, dim=0)
                mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)
                count_all = count_all.view(-1)
                mean, var, inv_std = syncbn.welford_parallel(mean_all, invstd_all, count_all.to(torch.int32), eps)
            else:
                device = mean.device
                count_all = torch.cuda.IntTensor([count], device=device)
                inv_std = 1.0 / torch.sqrt(var_biased + eps)
                var = var_biased * (count) / (count-1)

            if count == 1 and world_size < 2:
                raise ValueError('Expected more than 1 value per channel when training, got input size{}'.format(input.size()))

            r_m_inc = mean if running_mean.dtype != torch.float16 else mean.half()
            r_v_inc = var if running_variance.dtype != torch.float16 else var.half()
            running_mean.data = running_mean.data * (1-momentum) + momentum*r_m_inc
            running_variance.data = running_variance.data * (1-momentum) + momentum*r_v_inc
        else:
            mean = running_mean.data
            inv_std = 1.0 / torch.sqrt(running_variance.data + eps)

        ctx.save_for_backward(input, weight, mean, inv_std, z, bias, count_all.to(torch.int32))
        ctx.process_group = process_group
        ctx.channel_last = channel_last
        ctx.world_size = world_size
        ctx.fuse_relu = fuse_relu

        if channel_last:
            out = syncbn.batchnorm_forward_c_last(input, z, mean, inv_std, weight, bias, fuse_relu)
        else:
            out = syncbn.batchnorm_forward(input, mean, inv_std, weight, bias)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        # mini batch mean & var are calculated by forward path.
        # mu = 1./N*np.sum(h, axis = 0)
        # var = 1./N*np.sum((h-mu)**2, axis = 0)
        saved_input, weight, mean, inv_std, z, bias, count = ctx.saved_tensors
        process_group = ctx.process_group
        channel_last = ctx.channel_last
        world_size = ctx.world_size
        fuse_relu = ctx.fuse_relu
        grad_input = grad_z = grad_weight = grad_bias = None

        if fuse_relu:
            grad_output = syncbn.relu_bw_c_last(grad_output, saved_input, z, mean, inv_std, weight, bias)
        if isinstance(z, torch.Tensor) and ctx.needs_input_grad[1]:
            grad_z = grad_output.clone()

        # TODO: update kernel to not pre_divide by item_num
        if channel_last:
            sum_dy, sum_dy_xmu, grad_weight, grad_bias = syncbn.reduce_bn_c_last(grad_output, saved_input, mean, inv_std, weight)
        else:
            sum_dy, sum_dy_xmu, grad_weight, grad_bias = syncbn.reduce_bn(grad_output, saved_input, mean, inv_std, weight)

        # calculate grad_input
        if ctx.needs_input_grad[0]:

            if torch.distributed.is_initialized():
                num_channels = sum_dy.shape[0]
                combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
                torch.distributed.all_reduce(
                    combined, torch.distributed.ReduceOp.SUM, process_group, async_op=False)
                sum_dy, sum_dy_xmu = torch.split(combined, num_channels)

            if channel_last:
                grad_input = syncbn.batchnorm_backward_c_last(grad_output, saved_input, mean, inv_std, weight, sum_dy, sum_dy_xmu, count)
            else:
                grad_input = syncbn.batchnorm_backward(grad_output, saved_input, mean, inv_std, weight, sum_dy, sum_dy_xmu, count)

        if weight is None or not ctx.needs_input_grad[2]:
            grad_weight = None

        if weight is None or not ctx.needs_input_grad[3]:
            grad_bias = None

        return grad_input, grad_z, grad_weight, grad_bias, None, None, None, None, None, None, None, None
