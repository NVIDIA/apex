from typing import Optional, Union, List

import torch
import torch.nn as nn

import apex
from apex.transformer import parallel_state
from apex.transformer.pipeline_parallel import get_forward_backward_func
from apex.transformer.pipeline_parallel.schedules.common import _get_params_for_weight_decay_optimization
from apex.transformer.pipeline_parallel.schedules.common import build_model
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining import forward_backward_no_pipelining
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_with_interleaving import _forward_backward_pipelining_with_interleaving
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import forward_backward_pipelining_without_interleaving
from apex.transformer.pipeline_parallel.utils import average_losses_across_data_parallel_group
from apex.transformer.pipeline_parallel.utils import setup_microbatch_calculator
from apex.transformer.pipeline_parallel.utils import update_num_microbatches
from apex.transformer.testing import global_vars
from apex.transformer.testing.commons import TEST_SUCCESS_MESSAGE
from apex.transformer.testing.commons import initialize_distributed
from apex.transformer.testing.commons import print_separator
from apex.transformer.log_util import get_transformer_logger, set_logging_level


# set_logging_level("INFO")
_logger = get_transformer_logger("pipeline_parallel_test")
global_vars.set_global_variables()


batch_size, micro_batch_size = None, None
hidden_size = 16
fwd_bwd_functions = {
    "no_pipelining": forward_backward_no_pipelining,
    "no_interleaving": forward_backward_pipelining_without_interleaving,
    "interleaving": _forward_backward_pipelining_with_interleaving,
}


# note (mkozuki): `pre_process` and `post_process` are a placeholder until interleaving schedule test comes.
class MyLayer(nn.Module):

    def __init__(self, pre_process: bool, post_process: bool):
        super().__init__()
        self.pre_process = pre_process
        self.post_process = post_process
        self.layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.layer(x)

class MyModel(nn.Module):

    def __init__(self, pre_process: bool = False, post_process: bool = False) -> None:
        super().__init__()
        self.pre_process = pre_process
        self.post_process = post_process
        self.layer = MyLayer(pre_process=pre_process, post_process=post_process)
        self.input_tensor = None

    def set_input_tensor(self, input_tensor: Union[torch.Tensor, List[torch.Tensor]]) -> None:
        self.input_tensor = input_tensor

    def forward(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        if self.input_tensor is None:
            return self.layer(x)
        return self.layer(self.input_tensor)



def model_provider_func(pre_process, post_process) -> MyModel:
    return MyModel(pre_process, post_process)


def process_batch(batch):
    if isinstance(batch, list):
        x = batch[0]
    else:
        x = batch
    return x


def fwd_step_func(batch, model):
    x = process_batch(batch)
    y = model(x)

    # note (mkozuki): I don't think this function is nice but I do think this is enough for now
    # just to check the sanity of ported pipeline functions.
    def loss_func(x):
        loss = torch.sum(x)
        averaged_loss = average_losses_across_data_parallel_group([loss])
        return loss, {'avg': averaged_loss}
    return y, loss_func


# TODO (mkozuki): Add a case with `autocast` and `GradScaler`.
# Run forward & backward for one minibatch.
def forward_backward_func_template(
        name: str,
        forward_backward_func,
        pipeline_model_parallel_size: int,
        forward_only: bool,
) -> None:
    print_separator(f"name: {name}, pipeline model parallel size: {pipeline_model_parallel_size}")
    virtual_pipeline_model_parallel_size = 2 if name == "interleaving" else None
    if name == "no_pipelining":
        # note (mkozuki): `forward_backward_no_pipelining` is **NOTE** compatible with
        # pipeline_model_parallel_size>1. So use pipeline_model_parallel_size as
        # tensor_model_parallel_size and set pipeline_model_parallel_size to 1.
        parallel_state.initialize_model_parallel(1, 1, None)
    else:
        # NOTE (mkozuki): `virtual_pipeline_model_parallel_size` is necessary to enable interleaving scheduling
        # In megatron, `args.virtual_pipeline_model_parallel_size` is computed in megatron/arguments.py and
        # used ubiquitously but this test uses custom model so it's safe to abuse.
        parallel_state.initialize_model_parallel(
            1, pipeline_model_parallel_size, virtual_pipeline_model_parallel_size)
        if virtual_pipeline_model_parallel_size is not None:
            # Check the experimental warning message
            get_forward_backward_func(virtual_pipeline_model_parallel_size, pipeline_model_parallel_size)
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()

    model = build_model(
        model_provider_func,
        wrap_with_ddp=True,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
    )
    assert isinstance(model, list)
    assert len(model) == (1 if virtual_pipeline_model_parallel_size is None else virtual_pipeline_model_parallel_size)
    _param_groups = _get_params_for_weight_decay_optimization(model)
    torch.optim.Adam(_param_groups, lr=1e-4)

    tensor_shape = [batch_size // parallel_state.get_data_parallel_world_size(), hidden_size]
    batch = (torch.randn(tensor_shape).cuda(),)
    tensor_shape[0] = micro_batch_size

    update_num_microbatches(0)
    forward_backward_func(
        fwd_step_func, batch, model, forward_only=forward_only, tensor_shape=tensor_shape)

    if not forward_only:
        for m in model:
            for p in m.parameters():
                if p.grad is None:
                    raise RuntimeError("grad not found")
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(TEST_SUCCESS_MESSAGE)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    n_tests = 0
    failures = []

    initialize_distributed()
    world_size = torch.distributed.get_world_size()
    args = global_vars.get_args()
    batch_size = args.global_batch_size
    micro_batch_size = args.micro_batch_size
    setup_microbatch_calculator(
        args.rank,
        args.rampup_batch_size,
        args.global_batch_size,
        args.micro_batch_size,
        1,  # args.data_parallel_size,
    )
    for forward_only in (True, False):
        for name, forward_backward_func in fwd_bwd_functions.items():
            n_tests += 1
            # TODO (mkozuki): Test with data parallel size > 1.
            pipeline_model_parallel_size = world_size
            try:
                forward_backward_func_template(
                    name,
                    forward_backward_func,
                    pipeline_model_parallel_size,
                    forward_only,
                )
            except Exception as e:
                failures.append(
                    f"\t# {name} failed with pipeline size: {pipeline_model_parallel_size} "
                    f"and forward_only: {forward_only}\n"
                    f"pipeline rank: {parallel_state.get_pipeline_model_parallel_rank()}, "
                    f"virtual pipeline rank: {parallel_state.get_virtual_pipeline_model_parallel_rank()}\n"
                    f"{str(e)}"
                )
            finally:
                parallel_state.destroy_model_parallel()
        else:
            print_separator(f"{name} works")
    print_separator("TEST RESULT")
    if failures:
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print("\n".join(failures))
        msg = f"{len(failures)} / {n_tests} cases failed"
        raise RuntimeError(msg)
    else:
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print("### PASS!")
