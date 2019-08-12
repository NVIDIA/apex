import torch
from apex.multi_tensor_apply import multi_tensor_applier
from amp_C import multi_tensor_novograd

class NovoGrad(torch.optim.Optimizer):

    """Implements NovoGrad algorithm. Currently GPU-only.  Requires Apex to be installed via
    ``python setup.py install --cuda_ext --cpp_ext``.

    It has been proposed in `Jasper: An End-to-End Convolutional Neural Acoustic Model`_.
    More info: https://nvidia.github.io/OpenSeq2Seq/html/optimizers.html#novograd

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            NOT SUPPORTED now! (default: False)
        reg_inside_moment (bool, optional): whether do regularization (norm and L2)
            in momentum calculation. True for include, False for not include and
            only do it on update term. (default: False)
        grad_averaging (bool, optional): whether apply (1-beta2) to grad when
            calculating running averages of gradient. (default: True)
        norm_type (int, optional): which norm to calculate for each layer.
            2 for L2 norm, and 0 for infinite norm. These 2 are only supported
            type now. (default: 2)
        init_zero (bool, optional): whether init norm with 0 (start averaging on
            1st step) or first step norm (start averaging on 2nd step). True for
            init with 0. (default: False)
        set_grad_none (bool, optional): whether set grad to None when zero_grad()
            method is called. (default: True)

    .. _Jasper\: An End-to-End Convolutional Neural Acoustic Mode:
        https://arxiv.org/abs/1904.03288
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, bias_correction=True,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.,
                 amsgrad=False, reg_inside_moment=False,
                 grad_averaging=True, norm_type=2, init_zero=False,
                 set_grad_none=True):
        if amsgrad:
            raise RuntimeError('NovoGrad does not support the AMSGrad variant.')
        defaults = dict(lr=lr, bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay,
                        grad_averaging=grad_averaging, norm_type=norm_type,
                        init_zero=init_zero)
        super(NovoGrad, self).__init__(params, defaults)
        self.moment_mode = 0 if reg_inside_moment else 1
        self.dummy_overflow_buf = torch.cuda.IntTensor([0])
        self.set_grad_none = set_grad_none

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super(NovoGrad, self).zero_grad()

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

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            # create lists for multi-tensor apply
            p_list, g_list, m1_list = [], [], []

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError('NovoGrad does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)

                p_list.append(p.data)
                g_list.append(p.grad.data)
                m1_list.append(state['exp_avg'])

            # we will store per weight norm as one tensor for a group
            # different rom optim.Adam, we store norm here(not ^2) so we can unify 2 norm type
            if 'exp_avg_sq' not in group:
                if group['init_zero']:
                    group['exp_avg_sq'] = torch.cuda.FloatTensor(len(g_list)).contiguous().fill_(0)
                else: # init with first step norm, so first blend have no effect
                    if group['norm_type'] == 0:
                        m2 = [torch.max(torch.abs(g)).item() for g in g_list]
                    elif group['norm_type'] == 2:
                        m2 = [torch.sum(torch.pow(g, 2)).sqrt().item() for g in g_list]
                    else:
                        raise RuntimeError('NovoGrad only support l2/inf norm now.')
                    group['exp_avg_sq'] = torch.cuda.FloatTensor(m2)
            else:
                assert(len(g_list) == group['exp_avg_sq'].numel())

            multi_tensor_applier(multi_tensor_novograd,
                                 self.dummy_overflow_buf,
                                 [g_list, p_list, m1_list],
                                 group['exp_avg_sq'],
                                 group['lr'],
                                 beta1,
                                 beta2,
                                 group['eps'],
                                 group['step'],
                                 bias_correction,
                                 group['weight_decay'],
                                 grad_averaging,
                                 self.moment_mode,
                                 group['norm_type'])

        return loss
