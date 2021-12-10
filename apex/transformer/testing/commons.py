# coding=utf-8
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import random
from typing import Optional, Union, List

import numpy
import torch
import torch.nn as nn

from apex import transformer
from apex.transformer.pipeline_parallel.utils import average_losses_across_data_parallel_group
from apex.transformer.testing import global_vars


TEST_SUCCESS_MESSAGE = ">> passed the test :-)"


# note (mkozuki): `pre_process` and `post_process` are a placeholder until interleaving schedule test comes.
class MyLayer(nn.Module):

    def __init__(self, hidden_size: int, pre_process: bool, post_process: bool):
        super().__init__()
        self.pre_process = pre_process
        self.post_process = post_process
        self.layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.layer(x)

class MyModel(nn.Module):

    def __init__(self, hidden_size: int, pre_process: bool = False, post_process: bool = False) -> None:
        super().__init__()
        self.pre_process = pre_process
        self.post_process = post_process
        self.layer = MyLayer(hidden_size=hidden_size, pre_process=pre_process, post_process=post_process)
        self.input_tensor = None

    def set_input_tensor(self, input_tensor: Union[torch.Tensor, List[torch.Tensor]]) -> None:
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        self.input_tensor = input_tensor[0]

    def forward(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        if self.input_tensor is None:
            return self.layer(x)
        return self.layer(self.input_tensor)


def model_provider_func(hidden_size, pre_process, post_process) -> MyModel:
    return MyModel(hidden_size, pre_process, post_process)


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


class IdentityLayer(torch.nn.Module):
    def __init__(self, size, scale=1.0):
        super(IdentityLayer, self).__init__()
        self.weight = torch.nn.Parameter(scale * torch.randn(size))

    def forward(self):
        return self.weight


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    transformer.tensor_parallel.model_parallel_cuda_manual_seed(seed)


def initialize_distributed(backend='nccl'):
    """Initialize torch.distributed."""
    # Get local rank in case it is provided.
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--local_rank', type=int, default=None,
    #                    help='local rank passed from distributed launcher')
    # args = parser.parse_args()
    args = global_vars.get_args()
    local_rank = args.local_rank

    # Get rank and world size.
    rank = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv("WORLD_SIZE", '1'))

    print('> initializing torch.distributed with local rank: {}, '
          'rank: {}, world size: {}'.format(local_rank, rank, world_size))

    # Set the device id.
    device = rank % torch.cuda.device_count()
    if local_rank is not None:
        device = local_rank
    torch.cuda.set_device(device)

    # Call the init process.
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
        init_method=init_method)


def print_separator(message):
    torch.distributed.barrier()
    filler_len = (78 - len(message)) // 2
    filler = '-' * filler_len
    string = '\n' + filler + ' {} '.format(message) + filler
    if torch.distributed.get_rank() == 0:
        print(string, flush=True)
    torch.distributed.barrier()
