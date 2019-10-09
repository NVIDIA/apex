import contextlib
import warnings
import sys
import torch

from . import utils
from .opt import OptimWrapper
from .scaler import LossScaler
from ._amp_state import _amp_state, master_params, maybe_print

if torch.distributed.is_available():
    from ..parallel.LARC import LARC


# There's no reason to expose the notion of a "handle". Everything can happen through amp.* calls.
@contextlib.contextmanager
def scale_loss(loss,
               optimizers,
               loss_id=0,
               model=None,
               delay_unscale=False,
               delay_overflow_check=False):
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
        optimizers:  All optimizer(s) for which the current backward pass is creating gradients.
            Must be an optimizer or list of optimizers returned from an earlier call
            to ``amp.initialize``.  For example use with multiple optimizers, see
            "Multiple models/optimizers/losses" under `Advanced Amp Usage`_.
        loss_id(int, optional, default=0):  When used in conjunction with the ``num_losses`` argument
            to ``amp.initialize``, enables Amp to use a different loss scale per loss.  ``loss_id``
            must be an integer between 0 and ``num_losses`` that tells Amp which loss is
            being used for the current backward pass.  See "Multiple models/optimizers/losses"
            under `Advanced Amp Usage`_ for examples.  If ``loss_id`` is left unspecified, Amp
            will use the default global loss scaler for this backward pass.
        model(torch.nn.Module, optional, default=None):  Currently unused, reserved to enable future
            optimizations.
        delay_unscale(bool, optional, default=False):  ``delay_unscale`` is never necessary, and
            the default value of ``False`` is strongly recommended.
            If ``True``, Amp will not unscale the gradients or perform model->master
            gradient copies on context manager exit.
            ``delay_unscale=True`` is a minor ninja performance optimization and can result
            in weird gotchas (especially with multiple models/optimizers/losses),
            so only use it if you know what you're doing.
            "Gradient accumulation across iterations" under `Advanced Amp Usage`_
            illustrates a situation where this CAN (but does not need to) be used.

    .. warning::
        If ``delay_unscale`` is ``True`` for a given backward pass, ``optimizer.step()`` cannot be
        called yet after context manager exit, and must wait for another, later backward context
        manager invocation with ``delay_unscale`` left to False.

    .. _`Advanced Amp Usage`:
        https://nvidia.github.io/apex/advanced.html
    """
    if not hasattr(_amp_state, "opt_properties"):
        raise RuntimeError("Invoked 'with amp.scale_loss`, but internal Amp state has not been initialized.  "
                           "model, optimizer = amp.initialize(model, optimizer, opt_level=...) must be called "
                           "before `with amp.scale_loss`.")

    if not _amp_state.opt_properties.enabled:
        yield loss
        return

    if isinstance(optimizers, torch.optim.Optimizer) or ('LARC' in sys.modules and isinstance(optimizers, LARC)):
        optimizers = [optimizers]

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

    if not delay_unscale:
        if isinstance(optimizers, list):
            for optimizer in optimizers:
                if not optimizer._amp_stash.params_have_scaled_gradients:
                    optimizer._prepare_amp_backward()

    yield (loss.float())*loss_scale

    if delay_unscale:
        for optimizer in optimizers:
            optimizer._amp_stash.params_have_scaled_gradients = True
    else:
        # FusedSGD may take care of unscaling as part of their step() methods.
        # if not isinstance(optimizers, FP16_Optimizer_for_fused):
            loss_scaler.clear_overflow_state()
            for optimizer in optimizers:
                optimizer._post_amp_backward(loss_scaler)
                optimizer._amp_stash.params_have_scaled_gradients = False
            # For future fused optimizers that enable sync-free dynamic loss scaling,
            # should_skip will always be False.
            should_skip = False if delay_overflow_check else loss_scaler.update_scale()
            if should_skip:
                for optimizer in optimizers:
                    if not optimizer._amp_stash.already_patched:
                        # Close on loss_scaler and loss_id as well, to be safe.  Probably not
                        # necessary because amp.scale_loss is already creating a temporary scope.
                        def patch_step(opt, loss_scaler, loss_id):
                            opt_step = opt.step
                            def skip_step(closure=None):
                                if closure is not None:
                                    raise RuntimeError("Currently, Amp does not support closure use with optimizers.")
                                maybe_print(("Gradient overflow.  Skipping step, loss scaler " +
                                             "{} reducing loss scale to {}").format(loss_id,
                                             loss_scaler.loss_scale()))
                                # TODO:  I don't like the special casing for different optimizer implementations.
                                # Maybe skip should delegate to a method owned by the optimizers themselves.
                                if hasattr(opt._amp_stash, "all_fp32_from_fp16_params"):
                                    # Clear the master grads that wouldn't be zeroed by model.zero_grad()
                                    for param in opt._amp_stash.all_fp32_from_fp16_params:
                                        param.grad = None
                                if hasattr(opt, "most_recent_scale"):
                                    opt.most_recent_scale = 1.0
                                    opt.scale_set_by_backward = False
                                opt.step = opt_step
                                opt._amp_stash.already_patched = False
                            return skip_step
                        optimizer.step = patch_step(optimizer, loss_scaler, loss_id)
                        optimizer._amp_stash.already_patched = True

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
