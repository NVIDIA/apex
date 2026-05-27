/* coding=utf-8
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <torch/library.h>

#include <vector>

namespace multihead_attn {
namespace fused_softmax {
namespace scaled_upper_triang_masked_softmax {

at::Tensor fwd_cuda(at::Tensor const& input, float scale_factor);

at::Tensor bwd_cuda(at::Tensor const& output_grads, at::Tensor const& softmax_results, float scale_factor);

at::Tensor fwd(at::Tensor const& input, float scale_factor) {
  TORCH_CHECK(input.dim() == 3, "expected 3D tensor");
  TORCH_CHECK((input.scalar_type() == at::ScalarType::Half) || (input.scalar_type() == at::ScalarType::BFloat16),
              "Only fp16 and bf16 are supported");

  return fwd_cuda(input, scale_factor);
}

at::Tensor bwd(at::Tensor const& output_grads, at::Tensor const& softmax_results, float scale_factor) {
  TORCH_CHECK(output_grads.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(softmax_results.dim() == 3, "expected 3D tensor");

  TORCH_CHECK(
      (output_grads.scalar_type() == at::ScalarType::Half) || (output_grads.scalar_type() == at::ScalarType::BFloat16),
      "Only fp16 and bf16 are supported");
  TORCH_CHECK((softmax_results.scalar_type() == at::ScalarType::Half) ||
                  (softmax_results.scalar_type() == at::ScalarType::BFloat16),
              "Only fp16 and bf16 are supported");

  return bwd_cuda(output_grads, softmax_results, scale_factor);
}

at::Tensor fwd_dispatch(at::Tensor const& input, double scale_factor) {
  return fwd(input, static_cast<float>(scale_factor));
}

at::Tensor bwd_dispatch(at::Tensor const& output_grads, at::Tensor const& softmax_results, double scale_factor) {
  return bwd(output_grads, softmax_results, static_cast<float>(scale_factor));
}

}  // end namespace scaled_upper_triang_masked_softmax
}  // end namespace fused_softmax
}  // end namespace multihead_attn

TORCH_LIBRARY_FRAGMENT(apex, m) {
  m.def("scaled_upper_triang_masked_softmax_forward(Tensor input, float scale_factor) -> Tensor");
  m.def("scaled_upper_triang_masked_softmax_backward(Tensor output_grads, Tensor softmax_results, "
        "float scale_factor) -> Tensor");
}

TORCH_LIBRARY_IMPL(apex, CUDA, m) {
  m.impl("scaled_upper_triang_masked_softmax_forward",
         &multihead_attn::fused_softmax::scaled_upper_triang_masked_softmax::fwd_dispatch);
  m.impl("scaled_upper_triang_masked_softmax_backward",
         &multihead_attn::fused_softmax::scaled_upper_triang_masked_softmax::bwd_dispatch);
}
