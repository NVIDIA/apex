import torch
from torch.nn import init

from apex._autocast_utils import _cast_if_autocast_enabled
import fast_layer_norm


class FastLayerNormFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, epsilon):
        x = x.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()
        hidden_size = gamma.numel()
        xmat = x.view((-1, hidden_size))
        ymat, mu, rsigma = fast_layer_norm.ln_fwd(xmat, gamma, beta, epsilon, -0.0)
        ctx.save_for_backward(x, gamma, mu, rsigma)
        return ymat.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        # assert dy.is_contiguous()
        dy = dy.contiguous()  # this happens!
        x, gamma, mu, rsigma = ctx.saved_tensors
        hidden_size = gamma.numel()
        xmat = x.view((-1, hidden_size))
        dymat = dy.view(xmat.shape)
        dxmat, dgamma, dbeta, _, _ = fast_layer_norm.ln_bwd(dymat, xmat, mu, rsigma, gamma, -0.0)
        dx = dxmat.view(x.shape)
        return dx, dgamma, dbeta, None


def _fast_layer_norm(x, weight, bias, epsilon):
    args = _cast_if_autocast_enabled(x, weight, bias, epsilon)
    with torch.cuda.amp.autocast(enabled=False):
        return FastLayerNormFN.apply(*args)


class FastLayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.epsilon = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size))
        self.bias = torch.nn.Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x):
        return _fast_layer_norm(x, self.weight, self.bias, self.epsilon)


class FastLayerNorm1pFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, beta, epsilon, gamma_shift=1.0):
        x = x.contiguous()
        gamma = gamma.contiguous()
        beta = beta.contiguous()
        hidden_size = gamma.numel()
        xmat = x.view((-1, hidden_size))
        ymat, mu, rsigma = fast_layer_norm.ln_fwd(xmat, gamma, beta, epsilon, gamma_shift)
        ctx.gamma_shift = gamma_shift
        ctx.save_for_backward(x, gamma, mu, rsigma)
        return ymat.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        # assert dy.is_contiguous()
        dy = dy.contiguous()  # this happens!
        x, gamma, mu, rsigma = ctx.saved_tensors
        hidden_size = gamma.numel()
        xmat = x.view((-1, hidden_size))
        dymat = dy.view(xmat.shape)
        dxmat, dgamma, dbeta, _, _ = fast_layer_norm.ln_bwd(dymat, xmat, mu, rsigma, gamma, ctx.gamma_shift)
        dx = dxmat.view(x.shape)
        return dx, dgamma, dbeta, None, None


def _fast_layer_norm_1p(x, weight, bias, epsilon, gamma_shift=1.0):
    args = _cast_if_autocast_enabled(x, weight, bias, epsilon, gamma_shift)
    with torch.cuda.amp.autocast(enabled=False):
        return FastLayerNorm1pFN.apply(*args)


class FastLayerNorm1p(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5, gamma_shift=1.0):
        super().__init__()
        self.epsilon = eps
        self.weight = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.bias = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.gamma_shift = gamma_shift
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, 1 - self.gamma_shift)
        init.zeros_(self.bias)

    def forward(self, x):
        return _fast_layer_norm_1p(x, self.weight, self.bias, self.epsilon, self.gamma_shift)