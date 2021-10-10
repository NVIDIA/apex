import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from apex.transformer import parallel_state
from apex.transformer.pipeline_parallel.utils import setup_microbatch_calculator
from apex.transformer.pipeline_parallel.utils import update_num_microbatches
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import forward_backward_pipelining_without_interleaving
from apex.transformer.testing import global_vars
from apex.transformer.testing.commons import print_separator
from apex.transformer.testing.commons import initialize_distributed


global_vars.set_global_variables()


N_MAX_DEVICES = 4
N_LAYERS = 4
HIDDEN_SIZE = 128


class MyLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE))

    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor

    def forward(self, x):
        if parallel_state.is_pipeline_first_stage():
            input = x
        else:
            input = self.input_tensor
        return self.layer(input)


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = MyLayer()

    def set_input_tensor(self, input_tensor):
        self.layer.set_input_tensor(input_tensor)

    def forward(self, x):
        return self.layer(x)


def fwd_step_func(batch, model):
    y = model(batch)

    # note (mkozuki): I don't think this function is nice but I do think this is enough for now
    # just to check the sanity of ported pipeline functions.
    def loss_func(x):
        return torch.mean(x), {'avg': torch.mean(x)}
    return y, loss_func


# note (mkozuki): This currently only checks the functionality of `forward_backward_pipelining_without_interleaving`.
# i.e. checks whether we can run 1F1B for one minibatch
def test_pipeline_parallel_no_interleaving(pipeline_model_parallel_size):
    print_separator("pipeline without interleaving")
    parallel_state.initialize_model_parallel(1, pipeline_model_parallel_size, None)
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_rank()

    model = MyModel().cuda()
    model = DDP(model)
    forward_only = False
    batch = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE).cuda()

    update_num_microbatches(0)
    forward_backward_pipelining_without_interleaving(fwd_step_func, batch, model, forward_only)

    for p in model.parameters():
        assert p.grad is not None
    torch.distributed.barrier()


if __name__ == "__main__":
    failures = []

    initialize_distributed()
    world_size = torch.distributed.get_world_size()
    args = global_vars.get_args()
    setup_microbatch_calculator(
        args.rank,
        args.rampup_batch_size,
        args.global_batch_size,
        args.micro_batch_size,
        args.data_parallel_size,
    )
    pipeline_model_parallel_size = 1
    while pipeline_model_parallel_size <= min(world_size, N_MAX_DEVICES):
        try:
            test_pipeline_parallel_no_interleaving(pipeline_model_parallel_size)
        except Exception as e:
            failures.append(f"pipeline parallel size - {pipeline_model_parallel_size}, w/o interleaving: {str(e)}")
            break
        else:
            pipeline_model_parallel_size *= 2
        finally:
            parallel_state.destroy_model_parallel()
    if failures:

        raise RuntimeError("\n".join(failures))
