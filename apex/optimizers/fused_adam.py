import torch
import fused_adam_cuda

class FusedAdam(torch.optim.Adam):

    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False) NOT SUPPORTED in FusedAdam!

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, eps_inside_sqrt = False):
        if amsgrad:
            raise RuntimeError('FusedAdam does not support the AMSGrad variant.')
        super(FusedAdam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.eps_mode = 0 if  eps_inside_sqrt else 1

    def step(self, closure=None, grads=None, output_params=None, scale=1.):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        if grads is not None:
           assert len(self.param_groups)==1, "mixed precision optimizer works for a single group only"
        for group in self.param_groups:
            if grads is None:
               grads = [None]*len(group['params'])
            if output_params is None:
               output_params = [None]*len(group['params'])
            for p, grad, output_param in zip(group['params'],grads, output_params):
                #note: p.grad should not ever be set for correct operation of mixed precision optimizer that sometimes sends None gradients
                if p.grad is None and grad is None:
                    continue
                if grad is None:
                    grad = p.grad.data 
                if grad.is_sparse:
                    raise RuntimeError('FusedAdam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                out_p = torch.tensor([], dtype = torch.float) if output_param is None else output_param
                fused_adam_cuda.adam(p.data,
                                     out_p,
                                     exp_avg,
                                     exp_avg_sq,
                                     grad,
                                     group['lr'],
                                     beta1,
                                     beta2,
                                     group['eps'],
                                     scale,
                                     state['step'],
                                     self.eps_mode)
        return loss

