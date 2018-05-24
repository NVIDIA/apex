import contextlib
import logging
import warnings

import torch

from ._C import scale_lib

class OptimWrapper(object):
    def __init__(self, optimizer):
        self._optimizer = optimizer
        self._skip_next = False

    def step(self, closure=None):
        if closure is not None:
            raise NotImplementedError(
                'The `closure` argument is unsupported by the amp ' +
                'optimizer wrapper.')
        if self._skip_next:
            print('SKIP!')
            self._skip_next = False
        else:
            return self._optimizer.step(closure=closure)

    def skip_next_(self):
        self._skip_next = True

    def __getattr__(self, attr):
        return getattr(self._optimizer, attr)

    # Forward all torch.optim.Optimizer methods
    def __getstate__(self):
        return self._optimizer.__getstate__()

    def __setstate__(self):
        return self._optimizer.__setstate__()

    def __repr__(self):
        return self._optimizer.__repr__()

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict):
        return self._optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        return self._optimizer.zero_grad()

    def add_param_group(self, param_group):
        return self._optimizer.add_param_group(param_group)

class AmpHandle(object):
    def __init__(self, enable_caching=True):
        self._enable_caching = enable_caching
        self._cache = dict()
        self._loss_scale = 2.**16
        self._max_loss_scale = 2.**24
        self._scale_seq_len = 2000
        self._unskipped = 0
        self._overflow_buf = torch.cuda.ByteTensor(1024,)
        self._optimizer = None

    def wrap_optimizer(self, optimizer):
        self._optimizer = OptimWrapper(optimizer)
        return self._optimizer

    @contextlib.contextmanager
    def scale_loss(self, loss):
        loss_backward = loss.backward
        def warning_wrapper():
            warnings.warn("You called .backward() on the unscaled loss "
                          "inside a scale_loss block. This is almost "
                          "certainly an error.", stacklevel=2)
            loss_backward()

        loss.backward = warning_wrapper
        yield loss * self._loss_scale
        loss.backward = loss_backward

        self._overflow_buf.zero_()
        for group in self._optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    scale_lib.scale_check_overflow(p.grad.data,
                                                   1. / self._loss_scale,
                                                   self._overflow_buf)
        if self._overflow_buf.any():
            self._loss_scale /= 2.
            self._optimizer.skip_next_()
            #optimizer_step = optimizer.step
            #def skip_step():
            #    logging.info('Gradient overflow, skipping update')
            #    optimizer.step = optimizer_step
            #optimizer.step = skip_step
            self._unskipped = 0
        else:
            self._unskipped += 1

        if self._unskipped == self._scale_seq_len:
            self._loss_scale = min(self._max_loss_scale, self._loss_scale * 2.)
            self._unskipped = 0

        self._clear_cache()

    def _clear_cache(self):
        self._cache.clear()

    @property
    def has_cache(self):
        return self._enable_caching

    @property
    def cache(self):
        return self._cache
