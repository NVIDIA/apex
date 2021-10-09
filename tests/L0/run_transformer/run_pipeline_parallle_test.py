import torch
import torch.nn as nn

from apex.transformer import parallel_state
from apex.transformer.pipeline_parallel.utils import update_num_microbatches
from apex.transformer.pipeline_parallel.schedules.fwd_bwd_pipelining_without_interleaving import forward_backward_pipelining_without_interleaving
from apex.transformer.tensor_parallel.tests import global_vars
from apex.transformer.tensor_parallel.tests.commons import set_random_seed
from apex.transformer.tensor_parallel.tests.commons import print_separator
from apex.transformer.tensor_parallel.tests.commons import initialize_distributed
from apex.transformer.tensor_parallel.tests.commons import TEST_SUCCESS_MESSAGE


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
        input_ = self.input_tensor
        return self.layer(input_)


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([MyLayer() for _ in range(N_LAYERS)])

    def set_input_tensor(self, input_tensor):
        self.layer.set_input_tensor(input_tensor)

    def forward(self, x):
        return self.layer(x)


def fwd_step_func(batch, model):
    y = model(batch)
    return y, torch.mean


def test_pipeline_parallel_no_interleaving(pipeline_model_parallel_size):
    print_separator("pipeline without interleaving")
    parallel_state.initialize_model_parallel(1, pipeline_model_parallel_size, None)
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_rank()

    model = MyModel().cuda()
    forward_only = False
    batch = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE).cuda()

    update_num_microbatches(0)
    forward_backward_pipelining_without_interleaving(fwd_step_func, batch, model, forward_only)


if __name__ == "__main__":
    initialize_distributed()
    world_size = torch.distributed.get_world_size()
    pipeline_model_parallel_size = 1
    while pipeline_model_parallel_size <= min(world_size, N_MAX_DEVICES):
        test_pipeline_parallel_no_interleaving(pipeline_model_parallel_size)
        pipeline_model_parallel_size *= 2
    # Reset groups
    parallel_state.destroy_model_parallel()
