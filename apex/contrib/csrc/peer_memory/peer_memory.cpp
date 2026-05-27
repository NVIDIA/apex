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

#include "peer_memory_cuda.cuh"

namespace {
std::vector<int64_t> apex_peer_memory_get_raw_peers(at::Tensor ipc_addresses, int64_t peer_rank, int64_t raw) {
  return apex::contrib::peer_memory::get_raw_peers(ipc_addresses, static_cast<int>(peer_rank), raw);
}

at::Tensor apex_peer_memory_blob_view_half(int64_t raw, at::IntArrayRef shape, bool channels_last) {
  return apex::contrib::peer_memory::blob_view_half(raw, std::vector<int64_t>(shape.begin(), shape.end()),
                                                    channels_last);
}

at::Tensor apex_peer_memory_blob_view_float(int64_t raw, at::IntArrayRef shape, bool channels_last) {
  return apex::contrib::peer_memory::blob_view_float(raw, std::vector<int64_t>(shape.begin(), shape.end()),
                                                     channels_last);
}

at::Tensor apex_peer_memory_blob_view_int(int64_t raw, at::IntArrayRef shape, bool channels_last) {
  return apex::contrib::peer_memory::blob_view_int(raw, std::vector<int64_t>(shape.begin(), shape.end()),
                                                   channels_last);
}

void apex_peer_memory_push_pull_halos_1d(bool diagnostics, bool explicit_nhwc, int64_t numSM, int64_t rank,
                                         bool top_zero, at::Tensor top_in_halo, at::Tensor top_in_transfer,
                                         at::Tensor top_out_transfer, at::Tensor top_out_halo, bool btm_zero,
                                         at::Tensor btm_in_halo, at::Tensor btm_in_transfer,
                                         at::Tensor btm_out_transfer, at::Tensor btm_out_halo) {
  apex::contrib::peer_memory::push_pull_halos_1d(diagnostics, explicit_nhwc, static_cast<int>(numSM),
                                                 static_cast<int>(rank), top_zero, top_in_halo, top_in_transfer,
                                                 top_out_transfer, top_out_halo, btm_zero, btm_in_halo, btm_in_transfer,
                                                 btm_out_transfer, btm_out_halo);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(apex, m) {
  m.def("peer_memory_allocate_raw(int size) -> int");
  m.def("peer_memory_free_raw(int raw) -> ()");
  m.def("peer_memory_zero(int raw, int size) -> ()");
  m.def("peer_memory_get_raw_ipc_address(int raw) -> Tensor");
  m.def("peer_memory_get_raw_peers(Tensor ipc_addresses, int peer_rank, int raw) -> int[]");
  m.def("peer_memory_blob_view_half(int raw, int[] shape, bool channels_last) -> Tensor");
  m.def("peer_memory_blob_view_float(int raw, int[] shape, bool channels_last) -> Tensor");
  m.def("peer_memory_blob_view_int(int raw, int[] shape, bool channels_last) -> Tensor");
  m.def(
      "peer_memory_push_pull_halos_1d(bool diagnostics, bool explicit_nhwc, int numSM, int rank, bool top_zero, "
      "Tensor(a!) top_in_halo, Tensor(b!) top_in_transfer, Tensor(c!) top_out_transfer, Tensor(d!) top_out_halo, "
      "bool btm_zero, Tensor(e!) btm_in_halo, Tensor(f!) btm_in_transfer, Tensor(g!) btm_out_transfer, "
      "Tensor(h!) btm_out_halo) -> ()");
}

TORCH_LIBRARY_IMPL(apex, CompositeExplicitAutograd, m) {
  m.impl("peer_memory_allocate_raw", &apex::contrib::peer_memory::allocate_raw);
  m.impl("peer_memory_free_raw", &apex::contrib::peer_memory::free_raw);
  m.impl("peer_memory_zero", &apex::contrib::peer_memory::zero);
  m.impl("peer_memory_get_raw_ipc_address", &apex::contrib::peer_memory::get_raw_ipc_address);
  m.impl("peer_memory_get_raw_peers", &apex_peer_memory_get_raw_peers);
  m.impl("peer_memory_blob_view_half", &apex_peer_memory_blob_view_half);
  m.impl("peer_memory_blob_view_float", &apex_peer_memory_blob_view_float);
  m.impl("peer_memory_blob_view_int", &apex_peer_memory_blob_view_int);
}

TORCH_LIBRARY_IMPL(apex, CUDA, m) { m.impl("peer_memory_push_pull_halos_1d", &apex_peer_memory_push_pull_halos_1d); }
