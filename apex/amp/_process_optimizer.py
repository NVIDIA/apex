import types
from ..fp16_utils import master_params_to_model_params
from ..multi_tensor_apply import multi_tensor_applier
from ._amp_state import maybe_print
import torch
from ..optimizers import FusedAdam, FusedSGD


class AmpOptimizerState(object):
    def __init__(self):
        pass


def _master_params_to_model_params(self):
    stash = self._amp_stash
    if multi_tensor_applier.available:
        if len(stash.all_fp16_params) > 0:
            multi_tensor_applier(
                stash.multi_tensor_scale,
                stash.dummy_overflow_buf,
                [stash.all_fp32_from_fp16_params, stash.all_fp16_params],
                1.0)
    else:
        for fp16_group, fp32_from_fp16_group in zip(stash.fp16_groups, stash.fp32_from_fp16_groups):
            master_params_to_model_params(fp16_group, fp32_from_fp16_group)


def lazy_init_with_master_weights(self):
        stash = self._amp_stash
        stash.fp16_groups = []
        stash.fp32_from_fp16_groups = []
        stash.fp32_from_fp32_groups = []
        for i, param_group in enumerate(self.param_groups):
            # maybe_print("FP16_Optimizer processing param group {}:".format(i))
            fp16_params_this_group = []
            fp32_params_this_group = []
            fp32_from_fp16_params_this_group = []
            for i, param in enumerate(param_group['params']):
                if param.requires_grad:
                    if param.type() == 'torch.cuda.HalfTensor':
                        # maybe_print("FP16_Optimizer received torch.cuda.HalfTensor with {}"
                        #             .format(param.size()))
                        fp16_params_this_group.append(param)
                        master_param = param.detach().clone().float()
                        master_param.requires_grad = True
                        param_group['params'][i] = master_param
                        fp32_from_fp16_params_this_group.append(master_param)
                        # Reset existing state dict key to the new master param.
                        # We still need to recast per-param state tensors, if any, to FP32.
                        if param in self.state:
                           self.state[master_param] = self.state.pop(param)
                    elif param.type() == 'torch.cuda.FloatTensor':
                        # maybe_print("FP16_Optimizer received torch.cuda.FloatTensor with {}"
                        #             .format(param.size()))
                        fp32_params_this_group.append(param)
                        param_group['params'][i] = param
                    else:
                        raise TypeError("Optimizer's parameters must be either "
                                        "torch.cuda.FloatTensor or torch.cuda.HalfTensor. "
                                        "Received {}".format(param.type()))

            stash.fp16_groups.append(fp16_params_this_group)
            stash.fp32_from_fp16_groups.append(fp32_from_fp16_params_this_group)
            stash.fp32_from_fp32_groups.append(fp32_params_this_group)

        stash.all_fp16_params = []
        for group in stash.fp16_groups:
            stash.all_fp16_params += group

        stash.all_fp32_from_fp16_params = []
        for group in stash.fp32_from_fp16_groups:
            stash.all_fp32_from_fp16_params += group

        stash.all_fp32_from_fp32_params = []
        for group in stash.fp32_from_fp32_groups:
            stash.all_fp32_from_fp32_params += group

        # all_fp16_grad_stash is only needed for fused optimizers.
        stash.all_fp16_grad_stash = [None for _ in stash.all_fp16_params]
        # stash.all_fp32_from_fp16_grad_stash = [None for _ in stash.all_fp32_from_fp16_params]
        stash.all_fp32_from_fp32_grad_stash = [None for _ in stash.all_fp32_from_fp32_params]

        for param in stash.all_fp32_from_fp16_params:
            param.grad = None

        for param in stash.all_fp32_from_fp32_params:
            param.grad = None

        # Leverage state_dict() and load_state_dict() to recast preexisting per-param state tensors
        self.load_state_dict(self.state_dict())


def post_backward_models_are_masters(scaler, params, stashed_grads):
        # This is a lot of python overhead...
        grads_needing_unscale = []
        grads_needing_unscale_with_stash = []
        stashed = []
        for param, stashed_grad in zip(params, stashed_grads):
            if param.grad is None and stashed_grad is not None:
                param.grad = stashed_grad
            elif param.grad is not None and stashed_grad is None:
                grads_needing_unscale.append(param.grad)
            elif param.grad is not None and stashed_grad is not None:
                grads_needing_unscale_with_stash.append(param.grad)
                stashed.append(stashed_grad)
            else: # param.grad is None and stashed_grad is None
                continue

        if len(grads_needing_unscale) > 0:
            scaler.unscale(
                grads_needing_unscale,
                grads_needing_unscale,
                scaler.loss_scale(),
                models_are_masters=True)

        if len(grads_needing_unscale_with_stash) > 0:
            scaler.unscale_with_stashed(
                grads_needing_unscale_with_stash,
                stashed,
                grads_needing_unscale_with_stash)

        # Clear the stash.
        for i in range(len(stashed_grads)):
            stashed_grads[i] = None


def prepare_backward_with_master_weights(self):
    stash = self._amp_stash

    self._amp_lazy_init()

    for i, param in enumerate(stash.all_fp16_params):
        # Set up to leverage grad copy elision.
        # This may behave differently from an unpatched optimizer if zero_grad is used and the param is unused.
        param.grad = None

    # for i, param in enumerate(stash.all_fp32_from_fp16_params):
    #     stash.all_fp32_from_fp16_grad_stash[i] = param.grad

    for i, param in enumerate(stash.all_fp32_from_fp32_params):
        stash.all_fp32_from_fp32_grad_stash[i] = param.grad
        # Set up to leverage grad copy elision:
        param.grad = None


def post_backward_with_master_weights(self, scaler):
    stash = self._amp_stash

    self._amp_lazy_init()

    # This is a lot of python overhead...
    fp16_grads_needing_unscale = []
    new_fp32_grads = []
    fp16_grads_needing_unscale_with_stash = []
    preexisting_fp32_grads = []
    for fp16_param, fp32_param in zip(stash.all_fp16_params,
                                      stash.all_fp32_from_fp16_params):
        if fp16_param.grad is None and fp32_param.grad is not None:
            continue
        elif fp16_param.grad is not None and fp32_param.grad is None:
            fp32_param.grad = torch.empty_like(fp32_param)
            fp16_grads_needing_unscale.append(fp16_param.grad)
            new_fp32_grads.append(fp32_param.grad)
        elif fp16_param.grad is not None and fp32_param.grad is not None:
            fp16_grads_needing_unscale_with_stash.append(fp16_param.grad)
            preexisting_fp32_grads.append(fp32_param.grad)
        else: # fp16_param.grad is None and fp32_param.grad is None:
            continue

    if len(fp16_grads_needing_unscale) > 0:
        scaler.unscale(
            fp16_grads_needing_unscale,
            new_fp32_grads,
            scaler.loss_scale(),
            models_are_masters=False)

    if len(fp16_grads_needing_unscale_with_stash) > 0:
        scaler.unscale_with_stashed(
            fp16_grads_needing_unscale_with_stash,
            preexisting_fp32_grads,
            preexisting_fp32_grads)

    # fp32 params can be treated as they would be in the "no_master_weights" case.
    post_backward_models_are_masters(
        scaler,
        stash.all_fp32_from_fp32_params,
        stash.all_fp32_from_fp32_grad_stash)


def lazy_init_no_master_weights(self):
    stash = self._amp_stash
    stash.all_fp16_params = []
    stash.all_fp32_params = []
    for i, param_group in enumerate(self.param_groups):
        for i, param in enumerate(param_group['params']):
            if param.type() == 'torch.cuda.HalfTensor':
                stash.all_fp16_params.append(param)
            elif param.type() == 'torch.cuda.FloatTensor':
                stash.all_fp32_params.append(param)
            else:
                raise TypeError("Optimizer's parameters must be either "
                                "torch.cuda.FloatTensor or torch.cuda.HalfTensor. "
                                "Received {}".format(param.type()))

    stash.all_fp16_grad_stash = [None for _ in stash.all_fp16_params]
    stash.all_fp32_grad_stash = [None for _ in stash.all_fp32_params]


def prepare_backward_no_master_weights(self):
    stash = self._amp_stash

    self._amp_lazy_init()

    for i, param in enumerate(stash.all_fp16_params):
        stash.all_fp16_grad_stash[i] = param.grad
        # Set up to leverage grad copy elision:
        param.grad = None

    for i, param in enumerate(stash.all_fp32_params):
        stash.all_fp32_grad_stash[i] = param.grad
        # Set up to leverage grad copy elision:
        param.grad = None


def post_backward_no_master_weights(self, scaler):
    stash = self._amp_stash

    self._amp_lazy_init()

    split_types = ((stash.all_fp16_params, stash.all_fp16_grad_stash),
             (stash.all_fp32_params, stash.all_fp32_grad_stash))

    for params, stashed_grads in split_types:
        post_backward_models_are_masters(scaler, params, stashed_grads)


#####################################################################################
# FusedAdam versions
#####################################################################################

def prepare_backward_with_master_weights_FusedAdam(self):
    stash = self._amp_stash

    self._amp_lazy_init()


def post_backward_with_master_weights_FusedAdam(self, scaler):
    stash = self._amp_stash

    self._amp_lazy_init()

    stash.scale = scaler.loss_scale()
    stash.grads = [[param.grad.data for param in group] for group in stash.fp16_groups]
    stash.output_params = [[param for param in group] for group in stash.fp16_groups]

    norm_groups = []
    skip = False
    for grad_group in stash.grads:
        norm = multi_tensor_applier(
            stash.multi_tensor_l2norm,
            stash.dummy_overflow_buf,
            [grad_group])
        # Still syncing here for now.
        norm = float(norm)
        norm_groups.append(norm)
        if norm == float('inf') or norm == -float('inf') or norm != norm:
            skip = True
    if skip:
        scaler._overflow_buf.fill_(1.)
        scaler._has_overflow = True

    stash.grad_norms = norm_groups


def prepare_backward_no_master_weights_FusedAdam(self):
    stash = self._amp_stash

    self._amp_lazy_init()


def post_backward_no_master_weights_FusedAdam(self, scaler):
    stash = self._amp_stash

    self._amp_lazy_init()

    stash.scale = scaler.loss_scale()
    stash.grads = None
    stash.output_params = None
    stash.grad_norms = None


#####################################################################################
# FusedSGD versions
# Eat this ugly code duplication for now.  First make it work, then make it clean.
# It's difficult to anticipate what can be unified between the FusedAdam and FusedSGD
# implementations until I have them both working.
#####################################################################################

# FusedSGD never explicitly materializes the fp32 gradients for "fp32 from fp16" master params
# outside the kernel, so we must accumulate directly into the model grads.
def prepare_backward_with_master_weights_FusedSGD(self):
    if self.materialize_master_grads:
        prepare_backward_with_master_weights(self)
    else:
        stash = self._amp_stash

        self._amp_lazy_init()

        for i, param in enumerate(stash.all_fp16_params):
            stash.all_fp16_grad_stash[i] = param.grad
            # Set up to leverage grad copy elision:
            param.grad = None

        for i, param in enumerate(stash.all_fp32_from_fp32_params):
            stash.all_fp32_from_fp32_grad_stash[i] = param.grad
            # Set up to leverage grad copy elision:
            param.grad = None


def post_backward_with_master_weights_FusedSGD(self, scaler):
    if self.materialize_master_grads:
        post_backward_with_master_weights(self, scaler)
    else:
        # TODO:  handle gradient clipping and removal of any lingering scale here.
        stash = self._amp_stash

        self._amp_lazy_init()

        split_types = ((stash.all_fp16_params, stash.all_fp16_grad_stash),
                 (stash.all_fp32_from_fp32_params, stash.all_fp32_from_fp32_grad_stash))

        for params, stashed_grads in split_types:
            post_backward_models_are_masters(scaler, params, stashed_grads)


def prepare_backward_no_master_weights_FusedSGD(self):
    prepare_backward_no_master_weights(self)


def post_backward_no_master_weights_FusedSGD(self, scaler):
    post_backward_no_master_weights(self, scaler)


def _amp_lazy_init(self):
    stash = self._amp_stash

    if not stash.lazy_init_called:
        self._lazy_init_maybe_master_weights()
        stash.lazy_init_called = True


def _process_optimizer(optimizer, properties):
    if hasattr(optimizer, "_amp_stash"):
        raise RuntimeError("A given optimizer should only be passed through amp.initialize once.")
    else:
        optimizer._amp_stash = AmpOptimizerState()

    optimizer._amp_stash.lazy_init_called = False
    optimizer._amp_stash.already_patched = False
    optimizer._amp_stash.params_have_scaled_gradients = False

    for name in ("_lazy_init_maybe_master_weights",
                 "_master_params_to_model_params",
                 "_prepare_amp_backward",
                 "_post_amp_backward",
                 "_amp_lazy_init"):
        if hasattr(optimizer, name):
            raise RuntimeError("Incoming optimizer already has {} defined.".format(name))

    # TODO:  Centralize exposure and import error checking for the C backend.
    if multi_tensor_applier.available:
        import amp_C
        optimizer._amp_stash.multi_tensor_scale = amp_C.multi_tensor_scale
        optimizer._amp_stash.multi_tensor_l2norm = amp_C.multi_tensor_l2norm
        optimizer._amp_stash.dummy_overflow_buf = torch.cuda.IntTensor([0]);

    if properties.master_weights:
        optimizer._lazy_init_maybe_master_weights = types.MethodType(
            lazy_init_with_master_weights, optimizer)

        optimizer._master_params_to_model_params = types.MethodType(
            _master_params_to_model_params, optimizer)

        old_step = optimizer.step
        def new_step(self):
            retval = old_step()
            if not (isinstance(self, FusedAdam) or isinstance(self, FusedSGD)):
                self._master_params_to_model_params()
            # Clear the master grads that wouldn't be zeroed by model.zero_grad()
            for param in self._amp_stash.all_fp32_from_fp16_params:
                param.grad = None
            return retval
        optimizer.step = types.MethodType(new_step, optimizer)

        old_zero_grad = optimizer.zero_grad
        def new_zero_grad(self):
            stash = self._amp_stash
            self._amp_lazy_init()
            # Zero the model grads.
            for param in stash.all_fp16_params:
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()
            for param in stash.all_fp32_from_fp32_params:
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()
            # Clear the master grads that are independent of model grads
            for param in self._amp_stash.all_fp32_from_fp16_params:
                param.grad = None
        optimizer.zero_grad = types.MethodType(new_zero_grad, optimizer)

        if isinstance(optimizer, FusedAdam):
            optimizer._prepare_amp_backward = types.MethodType(
                prepare_backward_with_master_weights_FusedAdam, optimizer)
            optimizer._post_amp_backward = types.MethodType(
                post_backward_with_master_weights_FusedAdam, optimizer)
        elif isinstance(optimizer, FusedSGD):
            optimizer._prepare_amp_backward = types.MethodType(
                prepare_backward_with_master_weights_FusedSGD, optimizer)
            optimizer._post_amp_backward = types.MethodType(
                post_backward_with_master_weights_FusedSGD, optimizer)
        else:
            optimizer._prepare_amp_backward = types.MethodType(
                prepare_backward_with_master_weights, optimizer)
            optimizer._post_amp_backward = types.MethodType(
                post_backward_with_master_weights, optimizer)
    else:
        optimizer._lazy_init_maybe_master_weights = types.MethodType(
            lazy_init_no_master_weights, optimizer)

        if isinstance(optimizer, FusedAdam):
            optimizer._prepare_amp_backward = types.MethodType(
                prepare_backward_no_master_weights_FusedAdam, optimizer)
            optimizer._post_amp_backward = types.MethodType(
                post_backward_no_master_weights_FusedAdam, optimizer)
        elif isinstance(optimizer, FusedSGD):
            optimizer._prepare_amp_backward = types.MethodType(
                prepare_backward_no_master_weights_FusedSGD, optimizer)
            optimizer._post_amp_backward = types.MethodType(
                post_backward_no_master_weights_FusedSGD, optimizer)
        else:
            optimizer._prepare_amp_backward = types.MethodType(
                prepare_backward_no_master_weights, optimizer)
            optimizer._post_amp_backward = types.MethodType(
                post_backward_no_master_weights, optimizer)

    optimizer._amp_lazy_init = types.MethodType(_amp_lazy_init, optimizer)

    return optimizer
