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
#ifndef _peer_memory_h_
#define _peer_memory_h_

namespace apex { namespace contrib { namespace peer_memory {
    int64_t allocate_raw(int64_t size);
    void free_raw(int64_t raw);
    void zero(int64_t raw, int64_t size);
    at::Tensor get_raw_ipc_address(int64_t raw);
    std::vector<int64_t> get_raw_peers(at::Tensor ipc_addresses, int peer_rank, int64_t raw);
    at::Tensor blob_view_half(int64_t raw, std::vector<int64_t> shape, bool channels_last);
    at::Tensor blob_view_float(int64_t raw, std::vector<int64_t> shape, bool channels_last);
    at::Tensor blob_view_int(int64_t raw, std::vector<int64_t> shape, bool channels_last);
    void push_pull_halos_1d(
        bool diagnostics,
        bool explicit_nhwc,
        int numSM,                      // number of SMs to use
        int peer_rank,                  // rank in spatial parallel group
	bool top_zero,			// if top halo should be zeroed
        at::Tensor top_out_halo,        // top output halo buffer (in local device memory, received from top neighbor)
	at::Tensor top_inp_transfer,    // top input transfer buffer (in local peer memory)
        at::Tensor top_out_transfer,    // top output transfer buffer (in top neighbor peer memory)
        at::Tensor top_inp_halo,        // top input halo buffer (in local device memory, sent to top neighbor)
	bool btm_zero,			// if btm halo should be zeroed
        at::Tensor btm_out_halo,        // btm output halo buffer (in local device memory, received from btm neighbor)
	at::Tensor btm_inp_transfer,	// btm input transfer buffer (in local peer memory)
        at::Tensor btm_out_transfer,    // btm output transfer buffer (in btm neighbor peer memory)
        at::Tensor btm_inp_halo         // btm input halo buffer (in local device memory, sent to btm neighbor)
        );
} } }
#endif
