# apex.transformer

`apex.transformer` is a module which enables efficient large Transformer models at scale.

`apex.transformer.tensor_parallel` and `apex.transformer.pipeline_parallel` are both based on [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)'s module.
The former is based on `megatron.mpu` and the latter is on `megatron.schedules` and `megatron.p2p_communication`.

## Tensor Model Parallel (TP)

APEX's tensor model parallel utilities provides some `torch.nn.Module`'s, custom fused kernels, and PRNG state handling.
See Appendix B.2 of [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) for the details of
PRNG state handling.

## Pipeline Model Parallel (PP)
APEX's pipeline model parallel functions require models to have `.set_input_tensor` because
the input tensor for `.forward` method can be `None`.

The following is a really casual sketch of training script with apex pp.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from apex.transformer import parallel_state
from apex.transformer.pipeline_parallel import get_forward_backward_func


class Model(nn.Module):

    ...

    def __init__(self, *args, **kwargs):
        super().__init__()
        pre_process = kwargs.pop("pre_process")
        post_process = kwargs.pop("post_process")

    def set_input_tensor(self, tensor):
        self.input_tensor = tensor

    def forward(self, x, ...):
        if parallel_state.is_pipeline_first_stage():
            input = x
        else:
            input = self.input_tensor
        ...


def model_provider_func(*args, **kwargs):
    return Model(*args, **kwargs)


def loss_func(pred, label):
    loss = ...
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {'nice_loss': averaged_loss}


def forward_step_func(batch, model):
    input, label = process_batch(batch)
    out = model(input)
    return out, partial(loss_func, label)


forward_backward_func = get_forward_backward_func(virtual_pipeline_model_parallel_size, pipeline_model_parallel_size)


parallel_state.initialize_model_parallel(
    tensor_model_parallel_size,
    pipeline_model_parallel_size,
    virtual_pipeline_model_parallel_size,
)
# The following line basically is equivalent to `build_model(Model, wrap_with_ddp, virtual_pipeline_model_parallel_size, *model_args, **model_kwargs)`
model = build_model(model_provider_func, wrap_with_ddp, virtual_pipeline_model_parallel_size, *model_args, **model_kwargs)
optimizer = ...
data_loader = ...
for epoch in range(num_epochs):
    for batch in data_loader:
        forward_backward_func(forward_step_func, batch, model, forward_only=False, tensor_shape)
        optimizer.step()
```
