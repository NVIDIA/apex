import torch
from torch.nn import init

from apex._autocast_utils import _cast_if_autocast_enabled
import fast_layer_norm


class FastLayerNormFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, epsilon, memory_efficient=False):
        ctx.x_shape = x.shape
        ctx.memory_efficient = memory_efficient

        x = x.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()
        hidden_size = gamma.numel()
        xmat = x.view((-1, hidden_size))
        ymat, mu, rsigma = fast_layer_norm.ln_fwd(xmat, gamma, beta, epsilon)
        if ctx.memory_efficient:
            ctx.save_for_backward(ymat, gamma, None, rsigma, beta)
        else:
            ctx.save_for_backward(xmat, gamma, mu, rsigma, None)
        return ymat.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        # assert dy.is_contiguous()
        dy = dy.contiguous()  # this happens!
        x_or_y_mat, gamma, mu, rsigma, beta = ctx.saved_tensors
        dymat = dy.view(x_or_y_mat.shape)
        dxmat, dgamma, dbeta, _, _ = fast_layer_norm.ln_bwd(dymat, x_or_y_mat, mu, rsigma, gamma, beta, ctx.memory_efficient)
        dx = dxmat.view(ctx.x_shape)
        return dx, dgamma, dbeta, None, None


def _fast_layer_norm(x, weight, bias, epsilon, memory_efficient):
    args = _cast_if_autocast_enabled(x, weight, bias, epsilon, memory_efficient)
    with torch.cuda.amp.autocast(enabled=False):
        return FastLayerNormFN.apply(*args)


class FastLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5, memory_efficient=False):
        super().__init__()
        self.epsilon = eps
        self.memory_efficient = memory_efficient
        self.weight = torch.nn.Parameter(torch.empty(hidden_size))
        self.bias = torch.nn.Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x):
        return _fast_layer_norm(x, self.weight, self.bias, self.epsilon, self.memory_efficient)
