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
import torch

from apex.transformer import parallel_state
from apex.transformer import tensor_parallel
from apex.transformer.testing import global_vars
from apex.transformer.testing.commons import print_separator
from apex.transformer.testing.commons import initialize_distributed
from apex.transformer.testing.commons import TEST_SUCCESS_MESSAGE


global_vars.set_global_variables()


def test_set_cuda_rng_state(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing set_rng_state with size {} ...'.
              format(tensor_model_parallel_size))

    parallel_state.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()

    size = 123
    seed = 1234
    torch.cuda.manual_seed(seed)
    tensor = torch.cuda.FloatTensor(size)

    # Get the state
    rng_state = torch.cuda.get_rng_state()
    rng_state_copy = rng_state.clone()

    # Do some stuff.
    for _ in range(5):
        torch.randn(size, out=tensor)
    result_1 = tensor.clone()

    assert rng_state.sub(rng_state_copy).max() == 0
    assert torch.cuda.get_rng_state().sub(rng_state_copy).max() > 0

    # State should be different.
    new_rng_state = torch.cuda.get_rng_state()
    max_diff = new_rng_state.sub(rng_state).max()
    print('   max diff in rng state (should be non-zero) on global rank {}: {}'.
          format(torch.distributed.get_rank(), max_diff))
    assert max_diff > 0

    # Reset the rng state and do the same stuff.
    tensor_parallel.random._set_cuda_rng_state(rng_state)
    for _ in range(5):
        torch.randn(size, out=tensor)
    tensor_parallel.random._set_cuda_rng_state(rng_state)
    for _ in range(5):
        torch.randn(size, out=tensor)
    result_2 = tensor.clone()

    # Results should be the same
    error = result_2.sub(result_1).abs().max()
    print('   max error in generated tensors (should be zero) on '
          'global rank {}: {}'.format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Input state should have remained intact.
    error = rng_state.sub(rng_state_copy).max()
    print('   max error in rng state (should be zero) on global rank {}: {}'.
          format(torch.distributed.get_rank(), error))
    assert error == 0

    # Reset groups
    parallel_state.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(TEST_SUCCESS_MESSAGE)


def test_cuda_rng_tracker(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing cuda rng tracker with size {} ...'.
              format(tensor_model_parallel_size))

    parallel_state.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()

    seed_1 = 1234
    seed_2 = 4321
    size = [12, 21]
    tensor = torch.cuda.FloatTensor(size)

    # Set to seed_1 and generate two tensors.
    torch.cuda.manual_seed(seed_1)
    torch.randn(size, out=tensor)
    target_11 = tensor.clone()
    torch.randn(size, out=tensor)
    target_12 = tensor.clone()

    # Set to seed_2 and generate two tensors.
    torch.cuda.manual_seed(seed_2)
    torch.randn(size, out=tensor)
    target_21 = tensor.clone()
    torch.randn(size, out=tensor)
    target_22 = tensor.clone()

    # Now if we interleave seed_1 and seed_2,
    # we should still get the same tensors
    torch.cuda.manual_seed(seed_1)
    tensor_parallel.random.get_cuda_rng_tracker().add('test', seed_2)

    torch.randn(size, out=tensor)
    result_11 = tensor.clone()

    with tensor_parallel.random.get_cuda_rng_tracker().fork('test'):
        torch.randn(size, out=tensor)
        result_21 = tensor.clone()

    torch.randn(size, out=tensor)
    result_12 = tensor.clone()

    with tensor_parallel.random.get_cuda_rng_tracker().fork('test'):
        torch.randn(size, out=tensor)
        result_22 = tensor.clone()

    diff = result_11.sub(result_21).abs().max()
    diff = min(diff, result_12.sub(result_22).abs().max())
    print('   max diff in generated tensors (should be non-zero) on '
          'global rank {}: {}'.format(torch.distributed.get_rank(), diff))
    assert diff > 1.0e-6
    error = max(result_11.sub(target_11).abs().max(),
                result_12.sub(target_12).abs().max())
    error = max(error, result_21.sub(target_21).abs().max())
    error = max(error, result_22.sub(target_22).abs().max())
    print('   max error in generated tensors (should be zero) on '
          'global rank {}: {}'.format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset the tracker
    tensor_parallel.random.get_cuda_rng_tracker().reset()

    # Reset groups
    parallel_state.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(TEST_SUCCESS_MESSAGE)


def test_model_parallel_cuda_manual_seed(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print(
            '> testing model parallel cuda manual seed with size {} ...'.format(
                tensor_model_parallel_size))

    parallel_state.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()

    tensor_parallel.random.model_parallel_cuda_manual_seed(12345)
    assert torch.cuda.initial_seed() == 12345
    with tensor_parallel.random.get_cuda_rng_tracker().fork():
        assert (
            torch.cuda.initial_seed() ==
            12345 + 2718 + parallel_state.get_tensor_model_parallel_rank()
        )

    # Reset the tracker
    tensor_parallel.random.get_cuda_rng_tracker().reset()

    # Reset groups
    parallel_state.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(TEST_SUCCESS_MESSAGE)


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    initialize_distributed()
    world_size = torch.distributed.get_world_size()

    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator('test set rng state')
        test_set_cuda_rng_state(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2

    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator('test cuda rng tracker')
        test_cuda_rng_tracker(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2

    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator('test model parallel cuda manual seed')
        test_model_parallel_cuda_manual_seed(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2
