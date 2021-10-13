from typing import Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from apex.transformer import parallel_state
from apex.transformer.pipeline_parallel.utils import setup_microbatch_calculator
from apex.transformer.pipeline_parallel.utils import update_num_microbatches
from apex.transformer.pipeline_parallel.utils import average_losses_across_data_parallel_group
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_no_pipelining import forward_backward_no_pipelining
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_with_interleaving import forward_backward_pipelining_with_interleaving
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import forward_backward_pipelining_without_interleaving

from apex.transformer.testing import global_vars
from apex.transformer.testing.commons import print_separator
from apex.transformer.testing.commons import initialize_distributed
from apex.transformer.testing.commons import TEST_SUCCESS_MESSAGE


global_vars.set_global_variables()


batch_size, micro_batch_size = None, None
hidden_size = 16
fwd_bwd_functions = {
    "no_pipelining": forward_backward_no_pipelining,
    "no_interleaving": forward_backward_pipelining_without_interleaving,
    # "interleaving": forward_backward_pipelining_with_interleaving,
}


# note (mkozuki): `pre_process` and `post_process` are a placeholder until interleaving schedule test comes.
class MyLayer(nn.Module):

    def __init__(self, pre_process: bool = False, post_process: bool = False):
        super().__init__()
        self.pre_process = pre_process
        self.post_process = post_process
        self.layer = nn.Sequential(nn.Linear(hidden_size, hidden_size))

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor

    def forward(self, x):
        # note (mkozuki): The latter condition is for no pipelining
        if parallel_state.is_pipeline_first_stage() or x is not None:
            input = x
        else:
            input = self.input_tensor
        return self.layer(input)


class MyModel(nn.Module):

    def __init__(self, pre_process: bool, post_process: bool) -> None:
        super().__init__()
        self.pre_process = pre_process
        self.post_process = post_process
        self.layer = MyLayer(pre_process=pre_process, post_process=post_process)

    def set_input_tensor(self, input_tensor: Optional[torch.Tensor]) -> None:
        self.layer.set_input_tensor(input_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


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


# Run forward & backward for one minibatch.
def forward_backward_func_template(
        name: str,
        forward_backward_func,
        pipeline_model_parallel_size: int,
        forward_only: bool,
) -> None:
    print_separator(f"name: {name}, forward_only: {forward_only}, pipeline model parallel size: {pipeline_model_parallel_size}")
    parallel_state.initialize_model_parallel(1, pipeline_model_parallel_size, None)
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()

    model = MyModel(False, False).cuda()
    model = DDP(model)
    tensor_shape = [batch_size, hidden_size]
    batch = (torch.randn(tensor_shape).cuda(),)
    tensor_shape[0] = micro_batch_size

    update_num_microbatches(0)
    forward_backward_func(
        fwd_step_func, batch, model, forward_only=forward_only, tensor_shape=tensor_shape)

    if not forward_only:
        for p in model.parameters():
            assert p.grad is not None
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(TEST_SUCCESS_MESSAGE)


if __name__ == "__main__":
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
        args.data_parallel_size,
    )
    print(
        f">>> global batch size: {args.global_batch_size}\n"
        f">>> micro batch size: {args.micro_batch_size}\n"
        f">>> data parallel size: {args.data_parallel_size}"
    )
    for name, forward_backward_func in fwd_bwd_functions.items():
        for forward_only in (True, False):
            # if name == "no_pipelining" and not forward_only:
            # TODO (mkozuki): Check with backward
            if not forward_only:
                continue
            n_tests += 1
            # print_separator(f"{name} - {forward_only}")
            pipeline_model_parallel_size = 2
            while pipeline_model_parallel_size <= min(world_size, 4):
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
                        f"{str(e)}"
                    )
                    break
                else:
                    pipeline_model_parallel_size *= 2
                finally:
                    parallel_state.destroy_model_parallel()
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
