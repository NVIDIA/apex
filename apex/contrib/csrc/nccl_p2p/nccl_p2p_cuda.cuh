/**
 * Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <torch/extension.h>
#ifndef _nccl_p2p_h_
#define _nccl_p2p_h_

namespace apex { namespace contrib { namespace nccl_p2p {
at::Tensor get_unique_nccl_id(int n);
int init_nccl_comm(
        at::Tensor unique_nccl_id, 
        int my_rank, 
        int num_ranks
        );
void nccl_send(
        int handle, 
        at::Tensor input, 
        int destination
        );
void nccl_recv(
        int handle, 
        at::Tensor input, 
        int sender
        );
void left_right_halo_exchange_inplace(
        int handle,
	bool left_zero,
	bool right_zero,
	at::Tensor left_output_halo,
	at::Tensor right_output_halo,
	at::Tensor left_input_halo,
	at::Tensor right_input_halo,
	int group_size);
std::vector<at::Tensor> left_right_halo_exchange(
        int handle,
	bool left_zero,
	bool right_zero,
        at::Tensor left_output_halo, 
        at::Tensor right_output_halo,
        int group_size
        );
void add_delay(int delay);
}}}
#endif
