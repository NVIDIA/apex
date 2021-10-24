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
import functools
import operator

import torch

from apex.transformer import parallel_state
from apex.transformer.tensor_parallel import data as data_utils
from apex.transformer.testing import global_vars
from apex.transformer.testing.commons import print_separator
from apex.transformer.testing.commons import initialize_distributed
from apex.transformer.testing.commons import TEST_SUCCESS_MESSAGE

global_vars.set_global_variables()


def test_broadcast_data(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing broadcast_data with model parallel size {} ...'.
              format(tensor_model_parallel_size))

    parallel_state.initialize_model_parallel(tensor_model_parallel_size)
    torch.manual_seed(1234 + parallel_state.get_data_parallel_rank())
    tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()

    key_size_t = {
        'key1': [7, 11],
        'key2': [8, 2, 1],
        'key3': [13],
        'key4': [5, 1, 2],
        'key5': [5, 12],
    }
    keys = list(key_size_t.keys())

    data = {}
    data_t = {}
    for key in key_size_t:
        data[key] = torch.LongTensor(size=key_size_t[key]).random_(0, 1000)
        data_t[key] = data[key].clone()
    data['keyX'] = torch.FloatTensor(size=(5, )).random_(0, 1000)
    data_t['keyX'] = data['keyX'].clone()
    if parallel_state.get_tensor_model_parallel_rank() != 0:
        data = None

    data_utils._check_data_types(keys, data_t, torch.int64)
    key_size, key_numel, \
        total_numel = data_utils._build_key_size_numel_dictionaries(keys, data)
    for key in keys:
        assert key_size[key] == key_size_t[key]
    total_numel_t = 0
    for key in keys:
        target_size = functools.reduce(operator.mul, key_size_t[key], 1)
        assert key_numel[key] == target_size
        total_numel_t += target_size
    assert total_numel == total_numel_t

    data_b = data_utils.broadcast_data(keys, data, torch.int64)
    for key in keys:
        tensor = data_t[key].cuda()
        assert data_b[key].sub(tensor).abs().max() == 0

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
        print_separator('test test broadcast data')
        test_broadcast_data(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2
