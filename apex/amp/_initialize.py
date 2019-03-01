import torch
from torch._six import container_abcs, string_classes
import functools
from ._amp_state import _amp_state
from .handle import disable_casts
from .scaler import LossScaler
from apex.fp16_utils import convert_network
from ..fp16_utils import FP16_Optimizer as FP16_Optimizer_general
from ..optimizers import FP16_Optimizer as FP16_Optimizer_for_fused
from ..optimizers import FusedAdam
from ..parallel import DistributedDataParallel as apex_DDP


def to_type(dtype, t):
    if not t.is_cuda:
        print("Warning:  input tensor was not cuda.  Call .cuda() on your data before passing it.")
    if t.requires_grad:
        print("Warning:  input data requires grad.  Since input data is not a model parameter,\n"
              "its gradients will not be properly allreduced by DDP.")
    if t.is_floating_point():
        return t.to(dtype)
    return t


# Modified from torch.optim.optimizer.py.  This is a bit more general than casted_args in utils.py.
def applier(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    elif isinstance(value, string_classes):
        return value
    elif isinstance(value, container_abcs.Mapping):
        return {applier(k, fn) : applier(v, fn) for k, v in value.items()}
    elif isinstance(value, container_abcs.Iterable):
        return type(value)(applier(v, fn) for v in value)
    else:
        return value


def check_models(models):
    for model in models:
        parallel_type = None
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            parallel_type = "torch.nn.parallel.DistributedDataParallel"
        if isinstance(model, apex_DDP):
            parallel_type = "apex.parallel.DistributedDataParallel"
        if isinstance(model, torch.nn.parallel.DataParallel):
            parallel_type = "torch.nn.parallel.DataParallel"
        if parallel_type is not None:
            raise RuntimeError("Incoming model is an instance of {}. ".format(parallel_type) +
                "Parallel wrappers should only be applied to the model(s) AFTER \n"
                "the model(s) have been returned from amp.initialize.")


def check_params_fp32(models):
    for model in models:
        for name, param in model.named_parameters():
            if param.is_floating_point() and param.type() != "torch.cuda.FloatTensor":
                print("Warning:  Found param {} with type {}, expected torch.cuda.FloatTensor.\n"
                      "When using amp.initialize, you do not need to call .half() on your model\n"
                      "before passing it, no matter what optimization level you choose.".format(
                      name, param.type()))

        for name, buf in model.named_buffers():
            if buf.is_floating_point() and buf.type() != "torch.cuda.FloatTensor":
                print("Warning:  Found buffer {} with type {}, expected torch.cuda.FloatTensor.\n"
                      "When using amp.initialize, you do not need to call .half() on your model\n"
                      "before passing it, no matter what optimization level you choose.".format(
                      name, buf.type()))


def check_optimizers(optimizers):
    for optim in optimizers:
        bad_optim_type = None
        if isinstance(optim, FP16_Optimizer_general):
            bad_optim_type = "apex.fp16_utils.FP16_Optimizer"
        if isinstance(optim, FP16_Optimizer_for_fused):
            bad_optim_type = "apex.optimizers.FP16_Optimizer"
        if bad_optim_type is not None:
            raise RuntimeError("An incoming optimizer is an instance of {}. ".format(optim_type) +
                               "The optimizer(s) passed to amp.initialize() should be bare \n"
                               "instances of either ordinary Pytorch optimizers, or Apex fused \n"
                               "optimizers (currently just FusedAdam, but FusedSGD will be added \n"
                               "soon).  You should not manually wrap your optimizer in either \n"
                               "apex.fp16_utils.FP16_Optimizer or apex.optimizers.FP16_Optimizer. \n"
                               "amp.initialize will take care of that for you (if necessary) based \n"
                               "on the specified opt_level (and optional overridden properties).")


def wrap_fused_adam(optimizer, properties):
    msg = 'Currently, the usage of FusedAdam is restricted to '\
          'amp.initialize(..., opt_level="O2", keep_batchnorm_fp32=False, '\
          'loss_scale=float or "dynamic").  We are working on enabling more general usage.'

    assert properties.master_weights is True, msg
    assert properties.cast_model_type is torch.float16, msg
    assert (properties.keep_batchnorm_fp32 is False or
            properties.keep_batchnorm_fp32 is None), msg

    if properties.loss_scale == "dynamic":
        return FP16_Optimizer_for_fused(optimizer, dynamic_loss_scale=True)
    else:
        return FP16_Optimizer_for_fused(optimizer, static_loss_scale=properties.loss_scale)


def _initialize(models, optimizers, properties):
    from apex.parallel import DistributedDataParallel as apex_DDP
    from .amp import init as amp_init

    if isinstance(optimizers, torch.optim.Optimizer):
        optimizers_was_list = False
        optimizers = [optimizers]
    elif isinstance(optimizers, list):
        optimizers_was_list = True
    else:
        raise TypeError("optimizers must be either a single optimizer or a list of optimizers.")

    if isinstance(models, torch.nn.Module):
        models_was_list = False
        models = [models]
    elif isinstance(models, list):
        models_was_list = True
    else:
        raise TypeError("models must be either a single model or a list of models.")

    check_models(models)

    check_params_fp32(models)

    check_optimizers(optimizers)

    # In the future, when FP16_Optimizer can be deprecated and master weights can
    # become an attribute, remember to stash master weights before casting the model.

    if properties.cast_model_type:
        if properties.keep_batchnorm_fp32:
            for model in models:
                convert_network(model, properties.cast_model_type)
        else:
            for model in models:
                model.to(properties.cast_model_type)

        caster = functools.partial(to_type, properties.cast_model_type)

        # Patch the forward method to cast incoming data to the correct type.
        # I like writing things explicitly more than decorators.
        def patch_forward(old_fwd):
            def new_fwd(*args, **kwargs):
                return old_fwd(*applier(args, caster),
                               **applier(kwargs, caster))
            return new_fwd

        model.forward = patch_forward(model.forward)

        # State dict trick to recast any preexisting per-param state tensors 
        for optimizer in optimizers:
            optimizer.load_state_dict(optimizer.state_dict())

    if properties.master_weights:
        for i, optimizer in enumerate(optimizers):
            if isinstance(optimizer, FusedAdam):
                optimizers[i] = wrap_fused_adam(optimizer, properties)
            if properties.loss_scale == "dynamic":
                optimizers[i] = FP16_Optimizer_general(optimizer,
                                                       dynamic_loss_scale=True)
            else:
                optimizers[i] = FP16_Optimizer_general(optimizer,
                                                       static_loss_scale=properties.loss_scale)
    else:
        for optimizer in optimizers:
            optimizer.loss_scaler = LossScaler(properties.loss_scale)

    if properties.patch_torch_functions:
        # handle is unused here. It's accessible later through a global value anyway.
        handle = amp_init(loss_scale=properties.loss_scale)
        for optimizer in optimizers:
            # Disable Amp casting for the optimizer step, because it should only be
            # applied to FP32 master params anyway.
            def patch_step(old_step):
                def new_step(*args, **kwargs):
                    with disable_casts():
                        output = old_step(*args, **kwargs)
                    return output
                return new_step

            optimizer.step = patch_step(optimizer.step)

    if optimizers_was_list:
        if models_was_list:
            return models, optimizers
        else:
            return models[0], optimizers
    else:
        if models_was_list:
            return models, optimizers[0]
        else:
            return models[0], optimizers[0]
