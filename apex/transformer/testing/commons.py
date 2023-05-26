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
from dataclasses import dataclass
import datetime
import os
import random
from typing import Optional, Union, List, Tuple, Callable, Dict

import numpy
import torch
import torch.nn as nn

from apex import transformer
from apex.transformer.tensor_parallel import(
    ColumnParallelLinear,
    RowParallelLinear,
    scatter_to_sequence_parallel_region,
)
from apex.transformer.pipeline_parallel.utils import (
    average_losses_across_data_parallel_group,
)
from apex.transformer.pipeline_parallel.schedules.common import (
    Batch,
)
from apex.transformer.testing import global_vars
from apex.transformer._ucc_util import HAS_UCC

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
    def __init__(
        self,
        hidden_size: int, pre_process: bool = False, post_process: bool = False,
        *,
        add_encoder: bool = False, add_decoder: bool = False,
    ) -> None:
        super().__init__()
        self.pre_process = pre_process
        self.post_process = post_process
        self.layer = MyLayer(
            hidden_size=hidden_size, pre_process=pre_process, post_process=post_process
        )
        self.input_tensor = None

    def set_input_tensor(
        self, input_tensor: Union[torch.Tensor, List[torch.Tensor]]
    ) -> None:
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        self.input_tensor = input_tensor[0]

    def forward(self, x: Optional[torch.Tensor]) -> torch.Tensor:
        if self.input_tensor is None:
            return self.layer(x)
        return self.layer(self.input_tensor)


class ToyParallelMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int, pre_process: bool = False, post_process: bool = False,
        *,
        sequence_parallel_enabled: bool = False,
        # TODO(mkozuki): Support these two?
        add_encoder: bool = False, add_decoder: bool = False,
    ) -> None:
        super().__init__()
        self.pre_process = pre_process
        self.post_process = post_process
        self.sequence_parallel_enabled = sequence_parallel_enabled

        ffn_hidden_size = 4 * hidden_size
        self.dense_h_to_4h = ColumnParallelLinear(
            hidden_size,
            ffn_hidden_size,
            gather_output=False,
            # init_method=init_method,
            skip_bias_add=True,
            # use_cpu_initialization=use_cpu_initialization,
            bias=True,
            sequence_parallel_enabled=sequence_parallel_enabled,
            no_async_tensor_model_parallel_allreduce=True,
        )
        self.dense_4h_to_h = RowParallelLinear(
            ffn_hidden_size,
            hidden_size,
            input_is_parallel=True,
            # init_method=output_layer_init_method,
            skip_bias_add=False,
            # use_cpu_initialization=use_cpu_initialization,
            bias=True,
            sequence_parallel_enabled=sequence_parallel_enabled,
        )
        self.activation_func = torch.nn.GELU()

    def set_input_tensor(
        self,
        input_tensor: Union[torch.Tensor, List[torch.Tensor]],
    ) -> None:
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        self.input_tensor = input_tensor[0]

    def forward(
        self,
        x: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward of Simplified ParallelMLP.

        Args:
            x: :obj:`None` if pipeline rank != pippeline first rank. When :obj:`None`,
                `self.input_tensor` is taken care of by `forward_step` defined in
                apex/transformer/pipeline_parallel/schedules/common.py
        """
        # [s, b, h]
        if self.input_tensor is None:
            input = x
        else:
            input = self.input_tensor
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(input)

        if bias_parallel is not None:
            intermediate_parallel += bias_parallel
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output


def model_provider_func(
    hidden_size: int,
    pre_process: bool,
    post_process: bool,
    *,
    add_encoder: bool = False,
    add_decoder: bool = False) -> MyModel:
    return MyModel(hidden_size, pre_process, post_process, add_encoder=add_encoder, add_decoder=add_decoder)


def mlp_provider_func(
    hidden_size: int,
    pre_process: bool,
    post_process: bool,
    *,
    add_encoder: bool = False,
    add_decoder: bool = False,
    sequence_parallel_enabled: bool = False,
) -> ToyParallelMLP:
    return ToyParallelMLP(
        hidden_size,
        pre_process,
        post_process,
        add_encoder=add_encoder,
        add_decoder=add_decoder,
        sequence_parallel_enabled=sequence_parallel_enabled,
    )


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
        return loss, {"avg": averaged_loss}

    return y, loss_func


@dataclass(frozen=True)
class ToyParallelMLPFwdBwdStepFunc:

    sequence_parallel_enabled: bool

    def __call__(
        self,
        batch: Batch,
        model: torch.nn.Module,
    ) -> Tuple[torch.Tensor, Callable[[torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]]:
        x = batch[0] if isinstance(batch, list) else batch
        if isinstance(x, torch.Tensor):
            x = x.transpose(0, 1).contiguous()
            if self.sequence_parallel_enabled:
                x = scatter_to_sequence_parallel_region(x)
        y = model(x)

        # note (mkozuki): I don't think this function is nice but I do think this is enough for now
        # just to check the sanity of ported pipeline functions.
        def loss_func(x):
            loss = torch.sum(x)
            averaged_loss = average_losses_across_data_parallel_group([loss])
            return loss, {"avg": averaged_loss}

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


def initialize_distributed(backend="nccl"):
    """Initialize torch.distributed."""
    # Get local rank in case it is provided.
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--local_rank', type=int, default=None,
    #                    help='local rank passed from distributed launcher')
    # args = parser.parse_args()
    if backend not in ("nccl", "ucc"):
        raise RuntimeError(f"Currently only nccl & ucc are supported but {backend}")
    if backend == "ucc":
        if not HAS_UCC:
            raise ImportError("UCC backend requires pytorch source build with UCC installed and enabled")
    args = global_vars.get_args()
    local_rank = args.local_rank

    # Get rank and world size.
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    print(
        "> initializing torch.distributed with local rank: {}, "
        "rank: {}, world size: {}".format(local_rank, rank, world_size)
    )

    # Set the device id.
    device = rank % torch.cuda.device_count()
    if local_rank is not None:
        device = local_rank
    torch.cuda.set_device(device)

    # Call the init process.
    init_method = "tcp://"
    master_ip = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "6000")
    init_method += master_ip + ":" + master_port
    torch.distributed.init_process_group(
        backend=backend, world_size=world_size, rank=rank, init_method=init_method,
        timeout=datetime.timedelta(seconds=60),
    )


def print_separator(message):
    filler_len = (78 - len(message)) // 2
    filler = "-" * filler_len
    string = "\n" + filler + " {} ".format(message) + filler
    if torch.distributed.get_rank() == 0:
        print(string, flush=True)
