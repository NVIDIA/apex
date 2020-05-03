import torch
from apex.multi_tensor_apply import multi_tensor_applier


class FusedAdagrad(torch.optim.Optimizer):
    """Implements Adagrad algorithm.

    Currently GPU-only.  Requires Apex to be installed via
    ``pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./``.

    This version of fused Adagrad implements 2 fusions.
      * Fusion of the Adagrad update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.

    :class:`apex.optimizers.FusedAdagrad`'s usage is identical to any ordinary Pytorch optimizer::
        opt = apex.optimizers.FusedAdagrad(model.parameters(), lr = ....)
        ...
        opt.step()

    :class:`apex.optimizers.FusedAdagrad` may be used with or without Amp.  If you wish to use :class:`FusedAdagrad` with Amp,
    you may choose any ``opt_level``::
        opt = apex.optimizers.FusedAdagrad(model.parameters(), lr = ....)
        model, opt = amp.initialize(model, opt, opt_level="O0" or "O1 or "O2")
        ...
        opt.step()
    In general, ``opt_level="O1"`` is recommended.

    It has been proposed in `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        adagrad_w_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay (also known as AdamW) (default: False)

    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """
    def __init__(self, params, lr=1e-2, eps=1e-10,
                 weight_decay=0., set_grad_none=True, adagrad_w_mode=False):

        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super(FusedAdagrad, self).__init__(params, defaults)
        self.adagrad_w_mode = 1 if adagrad_w_mode else 0
        self.set_grad_none = set_grad_none

        if multi_tensor_applier.available:
            import amp_C
            # Skip buffer
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_adagrad = amp_C.multi_tensor_adagrad
        else:
            raise RuntimeError('apex.optimizers.FusedAdagrad requires cuda extensions')

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super(FusedAdagrad, self).zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # create lists for multi-tensor apply
            g_16, p_16, h_16 = [], [], []
            g_32, p_32, h_32 = [], [], []

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError('FusedAdagrad does not support sparse gradients')

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['sum'] = torch.zeros_like(p.data)
                if p.dtype == torch.float16:
                    g_16.append(p.grad.data)
                    p_16.append(p.data)
                    h_16.append(state['sum'])
                elif p.dtype == torch.float32:
                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    h_32.append(state['sum'])
                else:
                    raise RuntimeError('FusedAdagrad only support fp16 and fp32.')

            if(len(g_16) > 0):
                multi_tensor_applier(self.multi_tensor_adagrad,
                                     self._dummy_overflow_buf,
                                     [g_16, p_16, h_16],
                                     group['lr'],
                                     group['eps'],
                                     self.adagrad_w_mode,
                                     group['weight_decay'])
            if(len(g_32) > 0):
                multi_tensor_applier(self.multi_tensor_adagrad,
                                     self._dummy_overflow_buf,
                                     [g_32, p_32, h_32],
                                     group['lr'],
                                     group['eps'],
                                     self.adagrad_w_mode,
                                     group['weight_decay'])

        return loss