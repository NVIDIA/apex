import torch
from torch._six import container_abcs, string_classes
import functools


def to_type(dtype, t):
    if not t.is_cuda:
        print("Warning:  input tensor was not cuda.  Call .cuda() on your data before passing it.")
    if t.requires_grad:
        print("Warning:  input data requires grad.  Since input data is not a model parameter,\n"
              "its gradients will not be properly allreduced by DDP.")
    if t.is_floating_point():
        return t.half()
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


def initialize(optimizers, models, properties):

    # Stash master weights before casting the model.
    # if properties.master_weights:

    if properties.cast_model_type is not None:
        if properties.cast_batchnorm is not None:
            for model in models:
                model.to(properties.cast_model_type)
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
