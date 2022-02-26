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
import torch.nn.init as init
from torch.nn.parameter import Parameter

from apex.transformer import parallel_state
from apex.transformer.tensor_parallel import layers
from apex.transformer.testing import global_vars
from apex.transformer.testing.commons import set_random_seed
from apex.transformer.testing.commons import print_separator
from apex.transformer.testing.commons import initialize_distributed
from apex.transformer.testing.commons import TEST_SUCCESS_MESSAGE


global_vars.set_global_variables()


class IdentityLayer3D(torch.nn.Module):
    def __init__(self, m, n, k):
        super(IdentityLayer3D, self).__init__()
        self.weight = Parameter(torch.Tensor(m, n, k))
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self):
        return self.weight


def test_parallel_embedding(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing parallel embedding with model parallel size {} ...'.
              format(tensor_model_parallel_size))

    parallel_state.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()

    batch_size = 17
    seq_length = 23
    vocab_size = 48
    hidden_size = 16
    seed = 1236

    set_random_seed(123)
    input_data = torch.LongTensor(
        size=(batch_size, seq_length)).random_(0, vocab_size).cuda()
    loss_weight = torch.randn([batch_size, seq_length, hidden_size]).cuda()

    set_random_seed(seed)
    embedding_original = torch.nn.Embedding(vocab_size, hidden_size).cuda()

    output = embedding_original(input_data)
    loss_original = torch.mul(output, loss_weight).sum()
    loss_original.backward()

    set_random_seed(seed)
    embedding_parallel = layers.ParallelEmbedding(
        vocab_size, hidden_size, init_method=init.normal_).cuda()
    output = embedding_parallel(input_data)
    loss_parallel = torch.mul(output, loss_weight).sum()
    loss_parallel.backward()

    set_random_seed(seed)
    embedding_vocab_parallel = layers.VocabParallelEmbedding(
        vocab_size, hidden_size, init_method=init.normal_).cuda()
    output = embedding_vocab_parallel(input_data)
    loss_vocab_parallel = torch.mul(output, loss_weight).sum()
    loss_vocab_parallel.backward()

    torch.distributed.barrier()
    error = loss_parallel.sub(loss_original).abs()
    print('   error in loss (parallel) on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-12, 'error: {}'.format(error)

    torch.distributed.barrier()
    error = loss_vocab_parallel.sub(loss_original).abs()
    print('   error in loss (vocab parallel) on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-12, 'error: {}'.format(error)

    weight_grad_orig = torch.split(embedding_original.weight.grad,
                                   hidden_size // tensor_model_parallel_size,
                                   1)[parallel_state.get_tensor_model_parallel_rank()]
    error = embedding_parallel.weight.grad.sub(weight_grad_orig).abs().max()
    print('   error in grad (parallel) on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-12, 'error: {}'.format(error)

    weight_grad_orig = torch.split(embedding_original.weight.grad,
                                   vocab_size // tensor_model_parallel_size,
                                   0)[parallel_state.get_tensor_model_parallel_rank()]
    error = embedding_vocab_parallel.weight.grad.sub(
        weight_grad_orig).abs().max()
    print('   error in grad (vocab parallel) on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-12, 'error: {}'.format(error)

    # Reset groups
    parallel_state.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>> passed the test :-)')


def test_initialize_affine_weight(tensor_model_parallel_size, device):

    parallel_state.initialize_model_parallel(tensor_model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print('> testing initialize_affine_weight with model parallel '
              'size: {}'.format(tensor_model_parallel_size))
    tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()

    seed = 12345
    input_size_coeff = 13
    input_size = input_size_coeff * tensor_model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * tensor_model_parallel_size

    # ---------------
    # Column parallel
    # ---------------
    weight = torch.empty(output_size_coeff, input_size)
    set_random_seed(seed)
    if device == 'cpu':
        layers._initialize_affine_weight_cpu(weight, output_size, input_size,
                                             output_size_coeff, 0,
                                             torch.nn.init.normal_,
                                             params_dtype=global_vars.get_args().params_dtype,
                                             )
    else:
        layers._initialize_affine_weight_gpu(weight, torch.nn.init.normal_, 0)

    # Target.
    set_random_seed(seed)
    master_weight = torch.empty(output_size, input_size)
    torch.nn.init.normal_(master_weight)
    rank = parallel_state.get_tensor_model_parallel_rank()
    my_weight = torch.split(master_weight, output_size_coeff,
                            dim=0)[rank].contiguous().clone()

    # Compare.
    error = weight.sub(my_weight).abs().max()
    torch.distributed.barrier()
    print('   column parallel max error (should be zero) on global rank '
          '{}: {}'.format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # ------------
    # Row parallel
    # ------------
    weight = torch.empty(output_size, input_size_coeff)
    set_random_seed(seed)
    if device == 'cpu':
        layers._initialize_affine_weight_cpu(
            weight, output_size, input_size, input_size_coeff, 1, torch.nn.init.normal_,
            params_dtype=global_vars.get_args().params_dtype)

    else:
        layers._initialize_affine_weight_gpu(weight, torch.nn.init.normal_, 1)

    # Target.
    set_random_seed(seed)
    master_weight = torch.empty(output_size, input_size)
    torch.nn.init.normal_(master_weight)
    rank = parallel_state.get_tensor_model_parallel_rank()
    my_weight = torch.split(master_weight, input_size_coeff,
                            dim=1)[rank].contiguous().clone()

    # Compare.
    error = weight.sub(my_weight).abs().max()
    torch.distributed.barrier()
    print('   row parallel max error (should be zero) on global rank '
          '{}: {}'.format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset groups
    parallel_state.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


class IdentityLayer2D(torch.nn.Module):
    def __init__(self, m, n):
        super(IdentityLayer2D, self).__init__()
        self.weight = Parameter(torch.Tensor(m, n))
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self):
        return self.weight


def test_column_parallel_linear(tensor_model_parallel_size):

    parallel_state.initialize_model_parallel(tensor_model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print('> testing ColumnParallelLinear with model parallel '
              'size: {}'.format(tensor_model_parallel_size))
    tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)
    input_size_coeff = 13
    input_size = input_size_coeff * tensor_model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * tensor_model_parallel_size
    batch_size = 7
    hidden_size = 9

    # Network
    gradient_accumulation_fusion = True
    identity_layer = IdentityLayer3D(batch_size, hidden_size, input_size).cuda()
    linear_layer = layers.ColumnParallelLinear(
        input_size, output_size, keep_master_weight_for_test=True,
        params_dtype=global_vars.get_args().params_dtype,
        use_cpu_initialization=global_vars.get_args().use_cpu_initialization,
        gradient_accumulation_fusion=gradient_accumulation_fusion,
    ).cuda()
    with torch.no_grad():
        linear_layer.weight.main_grad = torch.randn_like(linear_layer.weight)

    loss_weight = torch.randn([batch_size, hidden_size, output_size]).cuda()
    # Forward
    input_ = identity_layer()
    output, _ = linear_layer(input_)
    assert list(output.shape) == [batch_size, hidden_size, output_size]
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    # TODO (mkozuki): Fix the following commented out lines 
    # as `gradient_accumulation_fusion` only takes 3D tensors.
    # Values.
    # dLdY = loss_weight  # (7, 9, 17)
    # X = identity_layer.weight  # (7, 9, 13)
    # A = linear_layer.master_weight.cuda()  # (17, 13)
    # print(f"dLdY.shape, X.shape, A.shape = {dLdY.shape, X.shape, A.shape}")
    # dLdA = torch.matmul(dLdY.view(-1, 17).t(), X.view(-1, 13))
    # print(f"dLdA.shape = {dLdA.shape}")
    # ones = torch.ones(batch_size, hidden_size, 1).cuda()
    # print(f"dLdY.shape, ones.shape = {dLdY.shape, ones.shape}")
    # dLdb = torch.matmul(ones, dLdY).view(-1)
    # dLdX = torch.matmul(dLdY, A)

    # rank = parallel_state.get_tensor_model_parallel_rank()
    # my_dLdA = torch.split(dLdA, output_size_coeff,
    #                       dim=0)[rank].contiguous().clone()
    # error = my_dLdA.sub(linear_layer.weight.grad).abs().max()
    # torch.distributed.barrier()
    # print('   error in dLdA on global rank {}: {}'.format(
    #     torch.distributed.get_rank(), error))
    # assert error < 1.0e-6

    # my_dLdb = torch.split(dLdb, output_size_coeff,
    #                       dim=0)[rank].contiguous().clone()
    # error = my_dLdb.sub(linear_layer.bias.grad).abs().max()
    # torch.distributed.barrier()
    # print('   error in dLdb on global rank {}: {}'.format(
    #     torch.distributed.get_rank(), error))
    # assert error < 1.0e-6

    # error = dLdX.sub(identity_layer.weight.grad).abs().max()
    # torch.distributed.barrier()
    # print('   error in dLdX on global rank {}: {}'.format(
    #     torch.distributed.get_rank(), error))
    # assert error < 1.0e-6

    # Reset groups
    parallel_state.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


def test_column_parallel_linear_with_async_allreduce_autocast(tensor_model_parallel_size):
    autocast_dtypes = (torch.half, torch.bfloat16) if torch.cuda.is_bf16_supported() else (torch.half,)

    parallel_state.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)
    input_size_coeff = 13
    input_size = input_size_coeff * tensor_model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * tensor_model_parallel_size
    batch_size = 7

    # Network
    identity_layer = IdentityLayer3D(batch_size, batch_size, input_size).cuda()
    linear_layer = layers.ColumnParallelLinear(
        input_size, output_size, keep_master_weight_for_test=True,
        params_dtype=global_vars.get_args().params_dtype,
        use_cpu_initialization=global_vars.get_args().use_cpu_initialization,
    ).cuda()
    assert linear_layer.async_tensor_model_parallel_allreduce or tensor_model_parallel_size == 1
    # Forward
    for dtype in autocast_dtypes:
        loss_weight = torch.randn([batch_size, output_size]).cuda()
        with torch.cuda.amp.autocast(dtype=dtype):
            output, _ = linear_layer(identity_layer())
            loss = torch.mul(output, loss_weight).sum()
        assert output.dtype == dtype
        # Backward
        loss.backward()
        torch.distributed.barrier()

    # Reset groups
    parallel_state.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


def test_column_parallel_linear_with_async_allreduce_custom_amp(tensor_model_parallel_size):
    dtypes = (torch.half, torch.bfloat16) if torch.cuda.is_bf16_supported() else (torch.half,)

    parallel_state.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)
    input_size_coeff = 13
    input_size = input_size_coeff * tensor_model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * tensor_model_parallel_size
    batch_size = 7

    for dtype in dtypes:
        # Network
        identity_layer = IdentityLayer3D(batch_size, batch_size, input_size).to(device="cuda", dtype=dtype)
        linear_layer = layers.ColumnParallelLinear(
            input_size, output_size, keep_master_weight_for_test=True,
            params_dtype=global_vars.get_args().params_dtype,
            use_cpu_initialization=global_vars.get_args().use_cpu_initialization,
        ).to(device="cuda", dtype=dtype)
        # Forward
        loss_weight = torch.randn([batch_size, output_size]).cuda()
        output, _ = linear_layer(identity_layer())
        loss = torch.mul(output, loss_weight).sum()
        loss.backward()
        torch.distributed.barrier()

        assert output.dtype == dtype

    # Reset groups
    parallel_state.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


def test_row_parallel_linear(tensor_model_parallel_size):

    parallel_state.initialize_model_parallel(tensor_model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print('> testing RowParallelLinear with model parallel '
              'size: {}'.format(tensor_model_parallel_size))
    tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)
    input_size_coeff = 13
    input_size = input_size_coeff * tensor_model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * tensor_model_parallel_size
    batch_size = 7

    # Network
    identity_layer = IdentityLayer2D(batch_size, input_size).cuda()
    linear_layer = layers.RowParallelLinear(
        input_size, output_size, keep_master_weight_for_test=True,
        params_dtype=global_vars.get_args().params_dtype,
        use_cpu_initialization=global_vars.get_args().use_cpu_initialization,
    ).cuda()
    loss_weight = torch.randn([batch_size, output_size]).cuda()
    # Forward
    input_ = identity_layer()
    output, _ = linear_layer(input_)
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    # Values.
    dLdY = loss_weight
    X = identity_layer.weight
    A = linear_layer.master_weight.cuda()
    dLdA = torch.matmul(dLdY.t(), X)
    dLdb = torch.matmul(torch.ones(batch_size, 1).cuda().t(), dLdY).view(-1)
    dLdX = torch.matmul(dLdY, A)

    rank = parallel_state.get_tensor_model_parallel_rank()
    my_dLdA = torch.split(dLdA, input_size_coeff,
                          dim=1)[rank].contiguous().clone()
    error = my_dLdA.sub(linear_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdA on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    error = dLdb.sub(linear_layer.bias.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdb on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    error = dLdX.sub(identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdX on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset groups
    parallel_state.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


def parallel_self_attention(tensor_model_parallel_size, num_att_heads_per_partition,
                            hidden_size_per_att_head, dropout_prob, batch_size,
                            sequence_length):
    parallel_state.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)

    num_att_heads = num_att_heads_per_partition * \
        torch.distributed.get_world_size()
    hidden_size = hidden_size_per_att_head * num_att_heads

    # Network
    identity_layer = IdentityLayer3D(batch_size, sequence_length,
                                     hidden_size).cuda()
    attention_layer = parallel_state.BertParallelSelfAttention(hidden_size, num_att_heads,
                                                    dropout_prob).cuda()
    loss_weight = torch.randn([batch_size, sequence_length, hidden_size]).cuda()
    attention_mask = torch.randn([batch_size, 1, 1, sequence_length]).cuda()
    # Forward
    input_ = identity_layer()
    output = attention_layer(input_, attention_mask)
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    rank = parallel_state.get_tensor_model_parallel_rank()
    parallel_state.destroy_model_parallel()
    return rank, hidden_size, tensor_model_parallel_size, loss, \
        attention_layer, identity_layer


def test_parallel_self_attention(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing ParallelSelfAttention with model parallel '
              'size: {}'.format(tensor_model_parallel_size))

    num_att_heads_per_partition = 3
    hidden_size_per_att_head = 7
    dropout_prob = 0.0  # has to be zero
    batch_size = 5
    sequence_length = 13

    rank_1, hideen_size_1, tensor_model_parallel_size_1, loss_1, \
        attention_layer_1, identity_layer_1 = parallel_self_attention(
            1, num_att_heads_per_partition,
            hidden_size_per_att_head, dropout_prob, batch_size, sequence_length)

    rank, hidden_size, tensor_model_parallel_size, loss, \
        attention_layer, identity_layer = parallel_self_attention(
            tensor_model_parallel_size, num_att_heads_per_partition,
            hidden_size_per_att_head, dropout_prob, batch_size, sequence_length)
    assert hideen_size_1 == hidden_size

    error = loss_1.sub(loss).abs().max()
    torch.distributed.barrier()
    print('   loss error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 5.0e-6

    my_lin_grad_list = torch.split(
        attention_layer_1.query_key_value.weight.grad,
        hidden_size // tensor_model_parallel_size, 0)[rank::tensor_model_parallel_size]
    my_lin_grad = torch.cat(my_lin_grad_list, dim=0)
    error = my_lin_grad.sub(
        attention_layer.query_key_value.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   weight gradient error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 5.0e-6

    error = identity_layer_1.weight.grad.sub(
        identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   input gradient error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 5.0e-6

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


def parallel_transformer(tensor_model_parallel_size, num_att_heads_per_partition,
                         hidden_size_per_att_head, batch_size, sequence_length):

    parallel_state.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)

    num_att_heads = num_att_heads_per_partition * \
        torch.distributed.get_world_size()
    hidden_size = hidden_size_per_att_head * num_att_heads
    intermediate_size = 4 * hidden_size

    # Network
    identity_layer = IdentityLayer3D(batch_size, sequence_length,
                                     hidden_size).cuda()
    transformer_layer = parallel_state.BertParallelTransformerLayer(
        hidden_size, intermediate_size, num_att_heads, 0.0, 0.0,
        torch.nn.functional.relu, 1.0e-5).cuda()

    loss_weight = torch.randn([batch_size, sequence_length, hidden_size]).cuda()
    attention_mask = torch.randn([batch_size, 1, 1, sequence_length]).cuda()
    # Forward
    input_ = identity_layer()
    output = transformer_layer(input_, attention_mask)
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    rank = parallel_state.get_tensor_model_parallel_rank()
    parallel_state.destroy_model_parallel()
    return rank, hidden_size, tensor_model_parallel_size, loss, \
        transformer_layer, identity_layer


def test_parallel_transformer_layer(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing ParallelTransformerLayer with model parallel '
              'size: {}'.format(tensor_model_parallel_size))

    num_att_heads_per_partition = 3
    hidden_size_per_att_head = 7
    batch_size = 5
    sequence_length = 13

    rank_1, hidden_size_1, tensor_model_parallel_size_1, loss_1, \
        transformer_layer_1, identity_layer_1 = parallel_transformer(
            1, num_att_heads_per_partition,
            hidden_size_per_att_head, batch_size, sequence_length)

    rank, hidden_size, tensor_model_parallel_size, loss, \
        transformer_layer, identity_layer = parallel_transformer(
            tensor_model_parallel_size, num_att_heads_per_partition,
            hidden_size_per_att_head, batch_size, sequence_length)

    error = loss_1.sub(loss).abs().max()
    torch.distributed.barrier()
    print('   loss error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 5.0e-5, 'error: {}'.format(error)

    error = identity_layer_1.weight.grad.sub(
        identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   input gradient error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 5.0e-5, 'error: {}'.format(error)

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(TEST_SUCCESS_MESSAGE)


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    initialize_distributed()
    world_size = torch.distributed.get_world_size()

    exceptions = []

    print_separator('test initialize affine weight cpu')
    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        try:
            test_initialize_affine_weight(tensor_model_parallel_size, 'cpu')
        except Exception as e:
            exceptions.append(f"test_initialize_affine_weight-cpu with tensor model parallel size of {tensor_model_parallel_size} failed: {str(e)}")
            # Reset groups
            parallel_state.destroy_model_parallel()
            break
        else:
            tensor_model_parallel_size *= 2
    # Reset groups
    parallel_state.destroy_model_parallel()

    print_separator('test initialize affine weight gpu')
    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        try:
            test_initialize_affine_weight(tensor_model_parallel_size, 'gpu')
        except Exception as e:
            exceptions.append(f"test_initialize_affine_weight-gpu with tensor model parallel size of {tensor_model_parallel_size} failed: {str(e)}")
            # Reset groups
            parallel_state.destroy_model_parallel()
            break
        else:
            tensor_model_parallel_size *= 2

    # Deleted, replaced with vocab parallel embedding?
    #tensor_model_parallel_size = 1
    #while tensor_model_parallel_size <= world_size:
    #    print_separator('test parallel embedding')
    #    test_parallel_embedding(tensor_model_parallel_size)
    #    tensor_model_parallel_size *= 2

    print_separator('test column-parallel linear')
    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        try:
            test_column_parallel_linear(tensor_model_parallel_size)
        except Exception as e:
            exceptions.append(f"test_column_parallel_linear with tensor model parallel size of {tensor_model_parallel_size} failed: {str(e)}")
            # Reset groups
            parallel_state.destroy_model_parallel()
            break
        else:
            tensor_model_parallel_size *= 2

    print_separator('test row-parallel linear')
    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        try:
            test_row_parallel_linear(tensor_model_parallel_size)
        except Exception as e:
            exceptions.append(f"test_row_parallel_linear with tensor model parallel size of {tensor_model_parallel_size} failed: {str(e)}")
            # Reset groups
            parallel_state.destroy_model_parallel()
            break
        else:
            tensor_model_parallel_size *= 2

    print_separator("test ColumnParallelLinearWithAsyncAllreduce - autocast")
    tensor_model_parallel_size = 2
    while tensor_model_parallel_size <= world_size:
        try:
            test_column_parallel_linear_with_async_allreduce_autocast(tensor_model_parallel_size)
        except Exception as e:
            exceptions.append(f"test_column_parallel_linear_with_async_allreduce_autocast with tensor model parallel size of {tensor_model_parallel_size} failed: {str(e)}")
            # Reset groups
            parallel_state.destroy_model_parallel()
            break
        else:
            tensor_model_parallel_size *= 2

    print_separator("test ColumnParallelLinearWithAsyncAllreduce - custom AMP")
    tensor_model_parallel_size = 2
    while tensor_model_parallel_size <= world_size:
        try:
            test_column_parallel_linear_with_async_allreduce_custom_amp(tensor_model_parallel_size)
        except Exception as e:
            exceptions.append(f"test_column_parallel_linear_with_async_allreduce_custom_amp with tensor model parallel size of {tensor_model_parallel_size} failed: {str(e)}")
            # Reset groups
            parallel_state.destroy_model_parallel()
            break
        else:
            tensor_model_parallel_size *= 2

    if exceptions:
        raise RuntimeError("\n".join(exceptions))
    # Deleted
    #print_separator('test parallel self-attention')
    #tensor_model_parallel_size = 1
    #while tensor_model_parallel_size <= world_size:
    #    test_parallel_self_attention(tensor_model_parallel_size)
    #    tensor_model_parallel_size *= 2

    #Deleted because PararallelTransformerLayer no longer exists
    # print_separator('test parallel transformer')
    # tensor_model_parallel_size = 1
    # while tensor_model_parallel_size <= world_size:
    #     test_parallel_transformer_layer(tensor_model_parallel_size)
    #     tensor_model_parallel_size *= 2
