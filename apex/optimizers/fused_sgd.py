import torch
from torch.optim.optimizer import Optimizer, required

from apex.multi_tensor_apply import multi_tensor_applier

class FusedSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,
                 wd_after_momentum=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

        self.wd_after_momentum = wd_after_momentum

        if multi_tensor_applier.available:
            import amp_C
            # Skip buffer
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_sgd = amp_C.multi_tensor_sgd
        else:
            raise RuntimeError('apex.optim.SGD requires cuda extensions')

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

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
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            params = [p for p in group['params'] if p is not None]
            grads = [p.grad for p in params]
            momentums = []
            for p in params:
                param_state = self.state[p]
                # torch.optim.SGD initializes momentum in the main loop, we have
                # to do it here, and track whether or not we've done so, so that
                # momentum application can be skipped in the main kernel.
                if 'momentum_buffer' not in param_state:
                    first_run = True
                    buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    momentums.append(buf)
                else:
                    first_run = False
                    momentums.append(param_state['momentum_buffer'])

            # We have all parameters now, split them into appropriate groups for
            # parallel execution, following the 4 possible combos that the underlying
            # kernels support:
            # grad_type, param_type, momentum_type, requires_fp16_copy
            # 1. fp16, fp16, fp16, No
            # 2. fp16, fp32, fp32, No
            # 3. fp16, fp32, fp32, Yes
            # 4. fp32, fp32, fp32, No
            # As in the kernel, easier to hardcode these options

            # Store only indices into the weight / grad / momentum lists
            # { gradient-type : { param-type : List } | List }
            param_sets = { 'fp16' : { 'fp16' : [], 'fp32' : [] }, 'fp32' : [] }

            for i, (g, p) in enumerate(zip(grads, params)):
                if g.dtype == torch.float16:
                    # fp16 grads, fp16 params
                    if p.dtype == torch.float16:
                        param_sets['fp16']['fp16'].append(i)
                    # fp16 grads, fp32 params
                    elif p.dtype == torch.float32:
                        param_sets['fp16']['fp32'].append(i)
                    else:
                        raise RuntimeError('fp16 gradients need either fp16 or fp32 weights')
                # fp32 grads, fp32 params
                elif g.dtype == torch.float32:
                    param_sets['fp32'].append(i)
                else:
                    raise RuntimeError('gradients must either be fp16 or fp32')

            def launch_sgd_set(param_set):
                local_params, local_grads, local_momentums = [], [], []
                if len(param_set) == 0:
                    return

                # launch update using multi tensor applier
                # modifies weight and momentum values inplace.
                multi_tensor_applier(
                    self.multi_tensor_sgd,
                    self._dummy_overflow_buf,
                    # Note: Need to do this as list comprehensions otherwise
                    # things don't seem to update properly.
                    [[grads[i] for i in param_set],
                     [params[i] for i in param_set],
                     [momentums[i] for i in param_set]],
                    weight_decay,
                    momentum,
                    dampening,
                    group['lr'],
                    nesterov,
                    first_run,
                    self.wd_after_momentum)

            # Explicitly go over the cases
            launch_sgd_set(param_sets['fp16']['fp16'])
            launch_sgd_set(param_sets['fp16']['fp32'])
            launch_sgd_set(param_sets['fp32'])

        return loss
