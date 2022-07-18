import torch
from apex.multi_tensor_apply import multi_tensor_applier

class FusedLANS(torch.optim.Optimizer):

    """Implements LANS algorithm.

    Currently GPU-only.  Requires Apex to be installed via
    ``pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./``.

    This version of fused LANS implements 2 fusions.

      * Fusion of the LANS update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.

    :class:`apex.optimizers.FusedLANS`'s usage is identical to any ordinary Pytorch optimizer::

        opt = apex.optimizers.FusedLANS(model.parameters(), lr = ....)
        ...
        opt.step()

    :class:`apex.optimizers.FusedLANS` may be used with or without Amp.  If you wish to use :class:`FusedLANS` with Amp,
    you may choose any ``opt_level``::

        opt = apex.optimizers.FusedLANS(model.parameters(), lr = ....)
        model, opt = amp.initialize(model, opt, opt_level="O0" or "O1 or "O2")
        ...
        opt.step()

    In general, ``opt_level="O1"`` is recommended.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam_w_mode (boolean, optional): Apply L2 regularization or weight decay
            True for decoupled weight decay(also known as AdamW) (default: True)
        grad_averaging (bool, optional): whether apply (1-beta2) to grad when
            calculating running averages of gradient. (default: True)
        set_grad_none (bool, optional): whether set grad to None when zero_grad()
            method is called. (default: True)
        normalize_grad (bool, optional): whether to normalize per-tensor grad
            (default: False)
        stochastic_rounding (bool, optional): whether to perform stochastic rounding for bfloat16 update
            (default: False)

    .. _Accelerated Large Batch Optimization of BERT Pretraining in 54 minutes:
        https://arxiv.org/abs/2006.13484
    """

    def __init__(self, params, lr=1e-3, bias_correction=True,
                 betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01,
                 adam_w_mode=True, grad_averaging=True, set_grad_none=True,
                 normalize_grad=False, stochastic_rounding=False):
        defaults = dict(lr=lr, bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        grad_averaging=grad_averaging,
                        normalize_grad=normalize_grad,
                        stochastic_rounding=stochastic_rounding)
        super(FusedLANS, self).__init__(params, defaults)
        if multi_tensor_applier.available:
            import amp_C
            # Skip buffer
            self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int,
                                                    device=self.param_groups[0]["params"][0].device)
            self.multi_tensor_lans = amp_C.multi_tensor_lans
        else:
            raise RuntimeError('apex.optimizers.FusedLANS requires cuda extensions')

        self.adam_w_mode = 1 if adam_w_mode else 0
        self.set_grad_none = set_grad_none

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super(FusedLANS, self).zero_grad()

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
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']
            grad_averaging = 1 if group['grad_averaging'] else 0

            # create lists for multi-tensor apply
            g_16, q_16, p_16, m_16, v_16 = [], [], [], [], []
            g_32, q_32, p_32, m_32, v_32 = [], [], [], [], []

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError('FusedLANS does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Buffer for scaled grad
                scaled_grad = torch.zeros_like(p.data)
                if p.dtype == torch.float16 or p.dtype == torch.bfloat16:
                    g_16.append(p.grad.data)
                    q_16.append(scaled_grad)
                    p_16.append(p.data)
                    m_16.append(state['exp_avg'])
                    v_16.append(state['exp_avg_sq'])
                elif p.dtype == torch.float32:
                    assert not group['stochastic_rounding'], 'stochastic rounding has to be disabled when float32 optimizer is used'
                    g_32.append(p.grad.data)
                    q_32.append(scaled_grad)
                    p_32.append(p.data)
                    m_32.append(state['exp_avg'])
                    v_32.append(state['exp_avg_sq'])
                else:
                    raise RuntimeError('FusedLANS only support fp16, bfloat16, and fp32.')

            if(len(g_16) > 0):
                state['step'] += 1
                multi_tensor_applier(self.multi_tensor_lans,
                                     self._dummy_overflow_buf,
                                     [g_16, q_16, p_16, m_16, v_16],
                                     group['lr'],
                                     beta1,
                                     beta2,
                                     group['eps'],
                                     state['step'],
                                     bias_correction,
                                     group['weight_decay'],
                                     grad_averaging,
                                     self.adam_w_mode,
                                     group['normalize_grad'],
                                     group['stochastic_rounding'])
            if(len(g_32) > 0):
                state['step'] += 1
                multi_tensor_applier(self.multi_tensor_lans,
                                     self._dummy_overflow_buf,
                                     [g_32, q_32, p_32, m_32, v_32],
                                     group['lr'],
                                     beta1,
                                     beta2,
                                     group['eps'],
                                     state['step'],
                                     bias_correction,
                                     group['weight_decay'],
                                     grad_averaging,
                                     self.adam_w_mode,
                                     group['normalize_grad'],
                                     False)

        return loss
