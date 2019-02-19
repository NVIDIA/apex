import torch
from torch._six import container_abcs, string_classes
import functools
from apex.fp16_utils import convert_network
from ._amp_state import _amp_state
from .scaler import LossScaler
from ..fp16_utils import FP16_Optimizer


def check_params_fp32(model):
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
                "Parallel wrappers should only be applied AFTER the model(s) have been "
                "returned from amp.initialize.")

    for model in models:
        check_params_fp32(model)

    # Stash master weights before casting the model.
    # if properties.master_weights:

    if properties.cast_model_type:
        if properties.cast_batchnorm:
            for model in models:
                convert_network(model, properties.cast_model_type)
        else:
            for model in models:
                model.to(properties.cast_model_type)

        caster = functools.partial(to_type, properties.cast_model_type)

        # Patch the forward method to cast incoming data to the correct type.
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
            if properties.loss_scale == "dynamic":
                optimizers[i] = FP16_Optimizer(optimizers[i], dynamic_loss_scale=True)
            else:
                optimizers[i] = FP16_Optimizer(optimizers[i], static_loss_scale=properties.loss_scale)
    else:
        for optimizer in optimizers:
            optimizer.loss_scaler = LossScaler(properties.loss_scale)

    if properties.cast_torch_functions:
        handle = amp_init(loss_scale=properties.loss_scale)

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
