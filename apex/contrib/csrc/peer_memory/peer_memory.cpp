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

#include "peer_memory_cuda.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("allocate_raw", &apex::contrib::peer_memory::allocate_raw, "allocate_raw");
    m.def("free_raw", &apex::contrib::peer_memory::free_raw, "free_raw");
    m.def("zero", &apex::contrib::peer_memory::zero, "zero");
    m.def("get_raw_ipc_address", &apex::contrib::peer_memory::get_raw_ipc_address, "get_raw_ipc_address");
    m.def("get_raw_peers", &apex::contrib::peer_memory::get_raw_peers, "get_raw_peers");
    m.def("blob_view_half", &apex::contrib::peer_memory::blob_view_half, "blob_view_half");
    m.def("blob_view_float", &apex::contrib::peer_memory::blob_view_float, "blob_view_float");
    m.def("blob_view_int", &apex::contrib::peer_memory::blob_view_int, "blob_view_int");
    m.def("push_pull_halos_1d", &apex::contrib::peer_memory::push_pull_halos_1d, "push_pull_halos_1d");
}
