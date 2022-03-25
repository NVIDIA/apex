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

#include "nccl_p2p_cuda.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_unique_nccl_id", &apex::contrib::nccl_p2p::get_unique_nccl_id, "get_unique_nccl_id");
  m.def("init_nccl_comm", &apex::contrib::nccl_p2p::init_nccl_comm, "init_nccl_comm");
  m.def("nccl_send", &apex::contrib::nccl_p2p::nccl_send, "nccl_send");
  m.def("nccl_recv", &apex::contrib::nccl_p2p::nccl_recv, "nccl_recv");
  m.def("left_right_halo_exchange_inplace", &apex::contrib::nccl_p2p::left_right_halo_exchange_inplace, "left_right_halo_exchange_inplace");
  m.def("left_right_halo_exchange", &apex::contrib::nccl_p2p::left_right_halo_exchange, "left_right_halo_exchange");
  m.def("add_delay", &apex::contrib::nccl_p2p::add_delay, "add_delay");
}
