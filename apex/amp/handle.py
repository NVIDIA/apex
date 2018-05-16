import contextlib
import logging
import warnings

import torch

from ._C import scale_lib

class AmpHandle(object):
    def __init__(self, enable_caching=True):
        self._enable_caching = enable_caching
        self._cache = dict()
        self._loss_scale = 2.**16
        self._max_loss_scale = 2.**24
        self._scale_seq_len = 2000
        self._unskipped = 0
        self._overflow_buf = torch.cuda.ByteTensor(1024,)

    @contextlib.contextmanager
    def scale_loss(self, loss, optimizer):
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
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    scale_lib.scale_check_overflow(p.grad.data,
                                                   1. / self._loss_scale,
                                                   self._overflow_buf)
        if self._overflow_buf.any():
            self._loss_scale /= 2.
            optimizer_step = optimizer.step
            def skip_step():
                logging.info('Gradient overflow, skipping update')
                optimizer.step = optimizer_step
            optimizer.step = skip_step
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
