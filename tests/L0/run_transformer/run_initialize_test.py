# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import torch

from apex.transformer import initialize
from apex.transformer.tensor_shard.tests import global_vars
from apex.transformer.tensor_shard.tests.commons import print_separator
from apex.transformer.tensor_shard.tests.commons import initialize_distributed
from apex.transformer.tensor_shard.tests.commons import TEST_SUCCESS_MESSAGE


global_vars.set_global_variables()


def test_initialize_model_parallel(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing initialize_model_parallel with size {} ...'.format(
            tensor_model_parallel_size))
    tensor_model_parallel_size_ = min(
        tensor_model_parallel_size,
        torch.distributed.get_world_size(),
    )
    assert not initialize.model_parallel_is_initialized()
    initialize.initialize_model_parallel(tensor_model_parallel_size_)
    assert initialize.model_parallel_is_initialized()

    # Checks.
    def check(group, world_size, rank):
        assert world_size == torch.distributed.get_world_size(group=group)
        assert rank == torch.distributed.get_rank(group=group)

    # Model parallel.
    world_size = tensor_model_parallel_size_
    rank = torch.distributed.get_rank() % tensor_model_parallel_size_
    assert world_size == initialize.get_tensor_model_parallel_world_size()
    assert rank == initialize.get_tensor_model_parallel_rank()
    check(initialize.get_tensor_model_parallel_group(), world_size, rank)

    # Data parallel.
    world_size = torch.distributed.get_world_size() // tensor_model_parallel_size_
    rank = torch.distributed.get_rank() // tensor_model_parallel_size
    assert world_size == initialize.get_data_parallel_world_size()
    assert rank == initialize.get_data_parallel_rank()
    check(initialize.get_data_parallel_group(), world_size, rank)

    # Reset groups
    initialize.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(TEST_SUCCESS_MESSAGE)


def test_get_tensor_model_parallel_src_rank(tensor_model_parallel_size_):

    if torch.distributed.get_rank() == 0:
        print('> testing get_tensor_model_parallel_src_rank with size {} ...'.format(
            tensor_model_parallel_size_))
    tensor_model_parallel_size = min(
        tensor_model_parallel_size_,
        torch.distributed.get_world_size(),
    )
    assert not initialize.model_parallel_is_initialized()
    initialize.initialize_model_parallel(tensor_model_parallel_size)
    assert initialize.model_parallel_is_initialized()

    # Checks
    src_rank = torch.distributed.get_rank() - initialize.get_tensor_model_parallel_rank()
    assert initialize.get_tensor_model_parallel_src_rank() == src_rank

    # Reset groups
    initialize.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>> passed the test :-)')


if __name__ == '__main__':

    initialize_distributed()
    world_size = torch.distributed.get_world_size()
    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator('test initialize model parallel')
        test_initialize_model_parallel(tensor_model_parallel_size)
        print_separator('test model parallel source rank')
        test_get_tensor_model_parallel_src_rank(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2
