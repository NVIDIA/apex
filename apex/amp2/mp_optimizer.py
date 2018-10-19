import contextlib
import torch
from torch import optim

# High level TODO question: can we assume that all the desired cast-to-half of model
# params have already occurred when this object is constructed? If not, we need to
# provide a mechanism to "refresh" its view on what to manage.
class MPOptimizer(optim.Optimizer):
    def __init__(self, optimizer):
        # TODO: things to support
        # - Disabling (so can control w/ cmd line switch)
        # - Loss scale config / state / etc.
        # - Delayed updates (loss accumulation)
        # - Knowledge of the amp cache for step()

        if not isinstance(optimizer, optim.Optimizer):
            raise ArgumentError('Argument `optimizer` must be an instance of ' +
                                'torch.optim.Optimizer')
        self._optimizer = optimizer
        self._all_float_params = [] # Make sure we maintain iteration order
        self._managed_params = []

        for pg in self._optimizer.param_groups:
            for i, param in enumerate(pg['params']):
                if not param.requires_grad:
                    continue

                if param.dtype == torch.float16:
                    master_param = param.detach().clone().float()
                    master_param.requires_grad = True
                    pg['params'][i] = master_param
                    if param in self._optimizer.state:
                        self._optimizer.state[master_param] = self._optimizer.state.pop(param)
                    self._managed_params.append((master_param, param))
                    self._all_float_params.append(master_param)
                elif param.dtype == torch.float32:
                    self._all_float_params.append(param)
                else:
                    raise ArgumentError('Cannot use MPOptimizer with parameters of type {}'.
                                        format(param.type()))

        # Re-cast optimizer state to float everywhere
        if len(self._managed_params) > 0:
            self._optimizer.load_state_dict(self._optimizer.state_dict())

    @contextlib.contextmanager
    def scale_loss(self, loss):
        loss_scale = 1024.
        yield loss * loss_scale

        # TODO: move out to utils file
        for master_param, model_param in self._managed_params:
            if model_param.grad is not None:
                if master_param.grad is None:
                    master_param.grad = torch.empty_like(model_param.grad, dtype=master_param.dtype)
                master_param.grad.copy_(model_param.grad)
            else:
                master_param.grad = None

        for p in self._all_float_params:
            p.grad.mul_(1. / loss_scale)

    def master_parameters(self):
        # TODO: ensure that all updates / unscaling have been applied
        #       in case of delayed updates, etc.
        for p in self._all_float_params:
            yield p

    def step(self, closure=None):
        # TODO: handle closures
        assert closure is None
        self._optimizer.step(closure=closure)
        for master_param, model_param in self._managed_params:
            with torch.no_grad():
                model_param.copy_(master_param)

    def zero_grad(self):
        self._optimizer.zero_grad()
        for _, p in self._managed_params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
