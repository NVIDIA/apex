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

#include <torch/library.h>

#include "nccl_p2p_cuda.cuh"

namespace {
at::Tensor apex_nccl_p2p_get_unique_nccl_id(int64_t n) {
  return apex::contrib::nccl_p2p::get_unique_nccl_id(static_cast<int>(n));
}

int64_t apex_nccl_p2p_init_nccl_comm(at::Tensor unique_nccl_id, int64_t my_rank, int64_t num_ranks) {
  return apex::contrib::nccl_p2p::init_nccl_comm(unique_nccl_id, static_cast<int>(my_rank),
                                                 static_cast<int>(num_ranks));
}

void apex_nccl_p2p_left_right_halo_exchange_inplace(int64_t handle, int64_t left_rank, int64_t right_rank,
                                                    at::Tensor left_output_halo, at::Tensor right_output_halo,
                                                    at::Tensor left_input_halo, at::Tensor right_input_halo) {
  apex::contrib::nccl_p2p::left_right_halo_exchange_inplace(static_cast<int>(handle), static_cast<int>(left_rank),
                                                            static_cast<int>(right_rank), left_output_halo,
                                                            right_output_halo, left_input_halo, right_input_halo);
}

std::vector<at::Tensor> apex_nccl_p2p_left_right_halo_exchange(int64_t handle, int64_t left_rank, int64_t right_rank,
                                                               at::Tensor left_output_halo,
                                                               at::Tensor right_output_halo) {
  return apex::contrib::nccl_p2p::left_right_halo_exchange(static_cast<int>(handle), static_cast<int>(left_rank),
                                                           static_cast<int>(right_rank), left_output_halo,
                                                           right_output_halo);
}

void apex_nccl_p2p_add_delay(int64_t delay) { apex::contrib::nccl_p2p::add_delay(static_cast<int>(delay)); }
}  // namespace

TORCH_LIBRARY_FRAGMENT(apex, m) {
  m.def("nccl_p2p_get_unique_nccl_id(int n) -> Tensor");
  m.def("nccl_p2p_init_nccl_comm(Tensor unique_nccl_id, int my_rank, int num_ranks) -> int");
  m.def(
      "nccl_p2p_left_right_halo_exchange_inplace(int handle, int left_rank, int right_rank, "
      "Tensor left_output_halo, Tensor right_output_halo, Tensor(a!) left_input_halo, "
      "Tensor(b!) right_input_halo) -> ()");
  m.def(
      "nccl_p2p_left_right_halo_exchange(int handle, int left_rank, int right_rank, Tensor left_output_halo, "
      "Tensor right_output_halo) -> Tensor[]");
  m.def("nccl_p2p_add_delay(int delay) -> ()");
}

TORCH_LIBRARY_IMPL(apex, CompositeExplicitAutograd, m) {
  m.impl("nccl_p2p_get_unique_nccl_id", &apex_nccl_p2p_get_unique_nccl_id);
  m.impl("nccl_p2p_init_nccl_comm", &apex_nccl_p2p_init_nccl_comm);
  m.impl("nccl_p2p_add_delay", &apex_nccl_p2p_add_delay);
}

TORCH_LIBRARY_IMPL(apex, CUDA, m) {
  m.impl("nccl_p2p_left_right_halo_exchange_inplace", &apex_nccl_p2p_left_right_halo_exchange_inplace);
  m.impl("nccl_p2p_left_right_halo_exchange", &apex_nccl_p2p_left_right_halo_exchange);
}
