import contextlib
import warnings
import torch

from . import utils
from .opt import OptimWrapper
from .scaler import LossScaler
from ._amp_state import _amp_state, master_params, maybe_print
from ..fp16_utils import FP16_Optimizer as FP16_Optimizer_general
from ..optimizers import FP16_Optimizer as FP16_Optimizer_for_fused


# There's no reason to expose the notion of a "handle". Everything can happen through amp.* calls.
@contextlib.contextmanager
def scale_loss(loss,
               optimizers,
               loss_id=0,
               model=None,
               delay_unscale=False):
    """
    On context manager entrance, creates ``scaled_loss = (loss.float())*current loss scale``.
    ``scaled_loss`` is yielded so that the user can call ``scaled_loss.backward()``::

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

    On context manager exit (if ``delay_unscale=False``), the gradients are checked for infs/NaNs
    and unscaled, so that ``optimizer.step()`` can be called.

    .. note::
        If Amp is using explicit FP32 master params (which is the default for ``opt_level=O2``, and
        can also be manually enabled by supplying ``master_weights=True`` to ``amp.initialize``)
        any FP16 gradients are copied to FP32 master gradients before being unscaled.
        ``optimizer.step()`` will then apply the unscaled master gradients to the master params.

    .. warning::
        If Amp is using explicit FP32 master params, only the FP32 master gradients will be
        unscaled.  The direct ``.grad`` attributes of any FP16
        model params will remain scaled after context manager exit.
        This subtlety affects gradient clipping.  See "Gradient clipping" under
        `Advanced Amp Usage`_ for best practices.

    Args:
        loss(Tensor):  Typically a scalar Tensor. The ``scaled_loss`` that the context
            manager yields is simply ``loss.float()*loss_scale``, so in principle
            ``loss`` could have more than one element, as long as you call
            ``backward()`` on ``scaled_loss`` appropriately within the context manager body.
        optimizer:  Must be an optimizer or list of optimizers returned from an earlier call
            to ``amp.initialize``.
        loss_id(int, optional, default=0):  When used in conjunction with the ``num_losses`` argument
            to ``amp.initialize``, enables Amp to use a different loss scale per loss.  ``loss_id``
            must be an integer between 0 and ``num_losses`` that tells Amp which loss is
            being used for the current backward pass.  See "Multiple backward
            passes per iteration" under `Advanced Amp Usage`_ for examples.
        model(torch.nn.Module, optional, default=None):  Currently unused, reserved to enable future
            optimizations.
        delay_unscale(bool, optional, default=False):  ``delay_unscale`` is a ninja option that only
            serves as a minor performance optimization, so only use it if you know what you're doing.
            If ``True``, Amp will not unscale the gradients or perform model->master
            gradient copies on context manager exit.
            `Advanced Amp Usage`_ illustrates situations where this CAN be used.

    .. warning::
        If ``delay_unscale`` is ``True`` for a given backward pass, ``optimizer.step()`` cannot be
        called yet after context manager exit, and must wait for another, later backward context
        manager invocation with ``delay_unscale`` left to False.
        See `Advanced Amp Usage`_ for examples.

    .. _`Advanced Amp Usage`:
        https://nvidia.github.io/apex/advanced.html
    """
    if not _amp_state.opt_properties.enabled:
        yield loss
        return

    if isinstance(optimizers, torch.optim.Optimizer):
        optimizers = [optimizers]

    # this is what happens when i have to support tools from different sources under the same API...
    # TODO:  Rewrite FusedAdam to use multi-tensor apply and the same loss scaler.
    if isinstance(optimizers, FP16_Optimizer_for_fused):
        loss_scale = optimizers.cur_scale
    else:
        loss_scaler = _amp_state.loss_scalers[loss_id]
        loss_scale = loss_scaler.loss_scale()

    if ((not _amp_state.opt_properties.master_weights)
        and (not loss_scaler.dynamic)
        and loss_scale == 1.0):
        yield loss.float()
        # Needing to drop the cache here as well is an ugly gotcha.
        # But for now I think it's necessary to short-circuit.
        # Probably ok to skip this if not delay_unscale
        if _amp_state.opt_properties.patch_torch_functions:
            _amp_state.handle._clear_cache()
        return

    if isinstance(optimizers, list):
        for optimizer in optimizers:
            optimizer._prepare_amp_backward()

    yield (loss.float())*loss_scale

    # this isn't pretty but it unifies things.  Once I deprecate the old API entirely,
    # I will have freedom to clean this up.  Maybe instead of wrapping optimizers,
    # I can simply construct a set of attributes (e.g. master params) and assign them
    # directly to optimizer instances.
    if not delay_unscale:
        # The FP16_Optimizer for FusedAdam will take care of unscaling as part of
        # its step() method.
        if not isinstance(optimizers, FP16_Optimizer_for_fused):
            loss_scaler.clear_overflow_state()
            for optimizer in optimizers:
                optimizer._post_amp_backward(loss_scaler)
            # For future fused optimizers that enable sync-free dynamic loss scaling,
            # should_skip will always be False.
            should_skip = loss_scaler.update_scale()
            if should_skip:
                optimizer_step = optimizer.step
                def skip_step():
                    maybe_print(("Gradient overflow.  Skipping step, loss scaler {} reducing " +
                                "loss scale to {}").format(loss_id, loss_scaler.loss_scale()))
                    optimizer.step = optimizer_step
                optimizer.step = skip_step

    # Probably ok to skip this if not delay_unscale
    if _amp_state.opt_properties.patch_torch_functions:
        _amp_state.handle._clear_cache()


# Free function version of AmpHandle.disable_casts, another step on the
# path to removing the concept of "AmpHandle"
@contextlib.contextmanager
def disable_casts():
    _amp_state.handle._is_active = False
    yield
    _amp_state.handle._is_active = True


class AmpHandle(object):
    def __init__(self, loss_scale="dynamic", enable_caching=True, verbose=False):
        self._enable_caching = enable_caching
        self._verbose = verbose
        self._cache = dict()
        self._default_scaler = LossScaler(loss_scale)
        self._is_active = True
        self._all_wrappers = []

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
        raise RuntimeError("The old Amp API is no longer supported.  Please move to the new API, "
            "documented here:  https://nvidia.github.io/apex/amp.html.  Transition guide:  "
            "https://nvidia.github.io/apex/amp.html#transition-guide-for-old-api-users")

        if not self.is_active():
            yield loss
            return

        if self._default_scaler is None:
            raise RuntimeError(
                'After calling `handle.wrap_optimizer()`, you must explicitly ' +
                'use `optimizer.scale_loss(loss)`.')

        # TODO: this code block is duplicated here and `opt.py`. Unify.
        loss_scale = self._default_scaler.loss_scale()
        yield loss * loss_scale

        self._default_scaler.clear_overflow_state()
        self._default_scaler.unscale(
            master_params(optimizer),
            master_params(optimizer),
            loss_scale)
        should_skip = self._default_scaler.update_scale()
        if should_skip:
            optimizer_step = optimizer.step
            def skip_step():
                maybe_print('Gradient overflow, skipping update')
                optimizer.step = optimizer_step
            optimizer.step = skip_step

        self._clear_cache()

    def _clear_cache(self):
        self._cache.clear()

    # Experimental support for saving / restoring uncasted versions of functions
    def _save_func(self, mod, fn, func):
        self._all_wrappers.append((mod, fn, func))

    def _deactivate(self):
        for mod, fn, func in self._all_wrappers:
            utils.set_func(mod, fn, func)
        self._all_wrappers = []

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

    def _clear_cache(self):
        pass

    def _deactivate(self):
        pass
