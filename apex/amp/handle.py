import contextlib
import logging
import warnings

from . import dbg
from .opt import OptimWrapper
from .scaler import LossScaler

class AmpHandle(object):
    def __init__(self, enable_caching=True, verbose=False):
        self._enable_caching = enable_caching
        self._verbose = verbose
        self._cache = dict()
        self._default_scaler = LossScaler()
        self._is_active = True

    def is_active(self):
        return self._is_active

    @contextlib.contextmanager
    def _disable_casts(self):
        self._is_active = False
        yield
        self._is_active = True

    def wrap_optimizer(self, optimizer, num_loss=1):
        self._default_scaler = None
        return OptimWrapper(optimizer, self, num_loss)

    @contextlib.contextmanager
    def scale_loss(self, loss, optimizer):
        if not self.is_active():
            yield loss
            return

        if self._default_scaler is None:
            raise RuntimeError(
                'After calling `handle.wrap_optimizer()`, you must explicitly ' +
                'use `optimizer.scale_loss(loss)`.')

        # TODO: this code block is duplicated here and `opt.py`. Unify.
        loss_backward = loss.backward
        def warning_wrapper():
            warnings.warn("You called .backward() on the unscaled loss "
                          "inside a scale_loss block. This is almost "
                          "certainly an error.", stacklevel=2)
            loss_backward()
        loss.backward = warning_wrapper
        loss_scale = self._default_scaler.loss_scale()
        yield loss * loss_scale
        loss.backward = loss_backward

        should_skip = self._default_scaler.unscale_and_update(
            optimizer.param_groups, loss_scale)
        if should_skip:
            optimizer_step = optimizer.step
            def skip_step():
                logging.info('Gradient overflow, skipping update')
                optimizer.step = optimizer_step
            optimizer.step = skip_step

        self._clear_cache()

    def _clear_cache(self):
        self._cache.clear()

    @property
    def has_cache(self):
        return self._enable_caching

    @property
    def cache(self):
        return self._cache

    def remove_cache(self, param):
        if self.has_cache and param in self.cache:
            del self.cache[param]

    @property
    def verbose(self):
        return self._verbose

    def run_debug(self, model, loss_fn):
        if not self.is_active():
            raise RuntimeError('can call debug() on only an active amp handle')

        # Disable caching during debug
        enable_caching = self._enable_caching
        self._enable_caching = False

        dbg.run(self, model, loss_fn)

        # Reset caching state
        self._clear_cache()
        self._enable_caching = enable_caching

class NoOpHandle(object):
    def is_active(self):
        return False

    @contextlib.contextmanager
    def _disable_casts(self):
        yield

    def wrap_optimizer(self, optimizer, num_loss=1):
        return OptimWrapper(optimizer, self, num_loss)

    @contextlib.contextmanager
    def scale_loss(self, loss, optimizer):
        yield loss

    @property
    def has_cache(self):
        return False

    @property
    def verbose(self):
        return False

    def run_debug(self, model, loss_fn):
        raise RuntimeError('can call debug() on only an active amp handle')
