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
import torch.nn.functional as F

from apex.transformer import parallel_state
from apex.transformer import tensor_parallel
from apex.transformer.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from apex.transformer.testing import global_vars
from apex.transformer.testing.commons import set_random_seed
from apex.transformer.testing.commons import IdentityLayer
from apex.transformer.testing.commons import print_separator
from apex.transformer.testing.commons import initialize_distributed
from apex.transformer.testing.commons import TEST_SUCCESS_MESSAGE


global_vars.set_global_variables()


def torch_cross_entropy(batch_size, seq_length, vocab_size,
                        logits_scale, seed):
    set_random_seed(seed)
    identity = IdentityLayer((batch_size, seq_length, vocab_size),
                             scale=logits_scale).cuda()
    logits = identity()
    target = torch.cuda.LongTensor(
        size=(batch_size, seq_length)).random_(0, vocab_size)
    loss = F.cross_entropy(logits.view(-1, logits.size()[-1]),
                           target.view(-1),
                           reduction='none').view_as(target).mean()
    loss.backward()
    return loss, identity.weight.grad


def tensor_sharded_cross_entropy(batch_size, seq_length, vocab_size, logits_scale, seed):
    set_random_seed(seed)
    identity = IdentityLayer((batch_size, seq_length, vocab_size), scale=logits_scale).cuda()
    logits = identity()
    logits_parallel = tensor_parallel.scatter_to_tensor_model_parallel_region(logits)
    target = torch.cuda.LongTensor(
        size=(batch_size, seq_length)).random_(0, vocab_size)
    logits_parallel_ = logits_parallel.clone().detach()
    loss = vocab_parallel_cross_entropy(logits_parallel, target).mean()
    loss.backward()
    # check for mutation
    assert torch.equal(logits_parallel_, logits_parallel)
    return loss, identity.weight.grad


def test_cross_entropy(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing cross entropy with model parallel size {} ...'.
              format(tensor_model_parallel_size))

    parallel_state.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()

    batch_size = 13
    seq_length = 17
    vocab_size_per_partition = 11
    logits_scale = 1000.0
    vocab_size = vocab_size_per_partition * tensor_model_parallel_size
    seed = 1234

    loss_torch, grad_torch = torch_cross_entropy(batch_size, seq_length, vocab_size, logits_scale, seed)
    loss_mpu, grad_mpu = tensor_sharded_cross_entropy(batch_size, seq_length, vocab_size, logits_scale, seed)

    error = loss_torch.sub_(loss_mpu).abs().max()
    print('   max error in loss on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    error = grad_torch.sub_(grad_mpu).abs().max()
    print('   max error in grad on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

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
        print_separator('test cross entropy')
        test_cross_entropy(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2
