import types
import torch
from torch.optim.optimizer import Optimizer, required

from apex.multi_tensor_apply import multi_tensor_applier

class FusedSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    This version of fused SGD implements 2 fusions.
      * Fusion of the SGD update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.

    :class:`apex.contrib.optimizers.FusedSGD` should be used without AMP.
   
    :class:`apex.contrib.optimizers.FusedSGD` only works in the case where all parameters require grad. 

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
        model = ...
        model.half()
        optimizer = apex.contrib.optimizers.FusedSGD(model.parameters())
        # wrap with FP16_Optimizer
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        optimizer.zero_grad()
	...
        optimizer.backward(loss)
        optmizer.step()

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
                 wd_after_momentum=False,
                 materialize_master_grads=True):
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
        super(FusedSGD, self).__init__(params, defaults)

        self.wd_after_momentum = wd_after_momentum

        if multi_tensor_applier.available:
            import amp_C
            # Skip buffer
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_sgd = amp_C.multi_tensor_sgd
        else:
            raise RuntimeError('apex.contrib.optimizers.FusedSGD requires cuda extensions')

    def __setstate__(self, state):
        super(FusedSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def get_momentums(self, params):
        momentums = []
        first_run = True
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
        return momentums, first_run
    
    def step(self, closure=None, grads=None, output_params=None, scale=1., grad_norms=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            grads (list of tensors, optional): weight gradient to use for the
                optimizer update. If gradients have type torch.half, parameters
                are expected to be in type torch.float. (default: None)
            output_params (list of tensors, optional): A reduced precision copy
                of the updated weights written out in addition to the regular
                updated weights. Have to be of same type as gradients. (default: None)
            scale (float, optional): factor to divide gradient tensor values
                by before applying to weights. (default: 1)
        """
        if hasattr(self, "_amp_stash"):
            raise RuntimeError('apex.contrib.optimizers.FusedSGD should not be used with AMP.')

        loss = None
        if closure is not None:
            loss = closure()

        if grads is None:
            raise RuntimeError('apex.contrib.optimizers.FusedSGD must be wrapped \
	                       with apex.contrib.optimizers.FP16_Optimizer \
			       which provides grads.')
        # backward compatibility
        # assuming a list/generator of parameter means single group
        elif isinstance(grads, types.GeneratorType):
            grads_group = [grads]
        elif type(grads[0])!=list:
            grads_group = [grads]
        else:
            grads_group = grads

        if output_params is None:
            raise RuntimeError('apex.contrib.optimizers.FusedSGD must be wrapped \
                               with apex.contrib.optimizers.FP16_Optimizer \
                               which provides output_params.')
        elif isinstance(output_params, types.GeneratorType):
            output_params_group = [output_params]
        elif type(output_params[0])!=list:
            output_params_group = [output_params]
        else:
            output_params_group = output_params

        for group, grads_this_group, output_params_this_group in zip(self.param_groups, 
	                                                             grads_group, 
                                                                     output_params_group):
            if grads_this_group is None or output_params_this_group is None: 
                raise RuntimeError('apex.contrib.optimizers.FusedSGD only works \
                                    when all parameters require grad.')
            
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            first_runs = [True, True]
            
            # output_params_this_group: original weights (either fp16 or fp32)
            # group['params']: master weights (fp32)

            # grad_type, param_to_update_type, momentum_type, requires_fp16_model_copy
            # fp32, fp32, fp32, No
            fp32_grads = [g for (p, g) in zip(output_params_this_group, grads_this_group) if p.dtype == torch.float32]
            fp32_params = [p2 for (p1, p2) in zip(output_params_this_group, group['params']) if p1.dtype == torch.float32]
            fp32_momentums, first_runs[1] = self.get_momentums(fp32_params)
            fp32_set = [fp32_grads, fp32_params, fp32_momentums]

            # fp16, fp32, fp32, Yes
            fp16_grads = [g for (p, g) in zip(output_params_this_group, grads_this_group) if p.dtype == torch.float16]
            fp32_from_fp16_params = [p2 for (p1, p2) in zip(output_params_this_group, group['params']) if p1.dtype == torch.float16]
            fp32_from_fp16_momentums, first_runs[0] = self.get_momentums(fp32_from_fp16_params)
            fp16_params = [p1 for (p1, p2) in zip(output_params_this_group, group['params']) if p1.dtype == torch.float16]
            fp16_set = [fp16_grads, fp32_from_fp16_params, fp32_from_fp16_momentums, fp16_params]

            launch_sets = [fp16_set, fp32_set]

            for launch_set, first_run in zip(launch_sets, first_runs):
                assert len(launch_set[0]) == len(launch_set[1])
                assert len(launch_set[0]) == len(launch_set[2])
                if len(launch_set[0]) > 0:
                    multi_tensor_applier(
                        self.multi_tensor_sgd,
                        self._dummy_overflow_buf,
                        launch_set,
                        weight_decay,
                        momentum,
                        dampening,
                        lr,
                        nesterov,
                        first_run,
                        self.wd_after_momentum,
                        1.0/scale)

        return loss
