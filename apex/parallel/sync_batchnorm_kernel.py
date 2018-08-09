import torch
from torch.autograd.function import Function


class SyncBatchnormFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias, running_mean, running_variance, eps):
        # transpose it to channel last to support broadcasting for input with different rank
        c_last_input = input.transpose(1, -1).contiguous()

        ctx.save_for_backward(c_last_input, weight, bias,
                              running_mean, running_variance)
        ctx.eps = eps

        c_last_input = (c_last_input - running_mean) / \
            torch.sqrt(running_variance + eps)

        if weight is not None:
            c_last_input = c_last_input * weight
        if bias is not None:
            c_last_input = c_last_input + bias

        return c_last_input.transpose(1, -1).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        # mini batch mean & var are calculated by forward path.
        # mu = 1./N*np.sum(h, axis = 0)
        # var = 1./N*np.sum((h-mu)**2, axis = 0)
        c_last_input, weight, bias, running_mean, running_variance = ctx.saved_tensors
        eps = ctx.eps
        grad_input = grad_weight = grad_bias = None
        num_features = running_mean.size()[0]

        # transpose it to channel last to support broadcasting for input with different rank
        c_last_grad = grad_output.transpose(1, -1).contiguous()
        # squash non-channel dimension so we can easily calculate mean
        c_grad = c_last_grad.view(-1, num_features)

        # calculate grad_input
        if ctx.needs_input_grad[0]:
            # dh = gamma * (var + eps)**(-1. / 2.) * (dy - np.mean(dy, axis=0)
            #     - (h - mu) * (var + eps)**(-1.0) * np.mean(dy * (h - mu), axis=0))
            mean_dy = c_grad.mean(0)
            mean_dy_xmu = (c_last_grad * (c_last_input -
                                          running_mean)).view(-1, num_features).mean(0)
            # all reduce on th
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    mean_dy, op=torch.distributed.reduce_op.SUM)
                mean_dy = mean_dy / torch.distributed.get_world_size()
                torch.distributed.all_reduce(
                    mean_dy_xmu, op=torch.distributed.reduce_op.SUM)
                mean_dy_xmu = mean_dy_xmu / torch.distributed.get_world_size()
            # tricky precision!!
            c_last_grad_input = (c_last_grad - mean_dy - (c_last_input - running_mean) / (
                running_variance + eps) * mean_dy_xmu) / torch.sqrt(running_variance + eps)
            if weight is not None:
                c_last_grad_input.mul_(weight)
            grad_input = c_last_grad_input.transpose(1, -1).contiguous()

        # calculate grad_weight
        if weight is not None and ctx.needs_input_grad[1]:
            # dgamma = np.sum((h - mu) * (var + eps)**(-1. / 2.) * dy, axis=0)
            # DDP will handle all_reduce (mean), so we compensate that if running in DDP
            grad_weight = ((c_last_input - running_mean) / torch.sqrt(
                running_variance + eps) * c_last_grad).view(-1, num_features).sum(0)
            if torch.distributed.is_initialized():
                grad_weight *= torch.distributed.get_world_size()

        # calculate grad_bias
        if bias is not None and ctx.needs_input_grad[2]:
            # dbeta = np.sum(dy, axis=0)
            # DDP will handle all_reduce (mean), so we compensate that if running in DDP
            grad_bias = c_grad.sum(0)
            if torch.distributed.is_initialized():
                grad_bias *= torch.distributed.get_world_size()

        return grad_input, grad_weight, grad_bias, None, None, None
