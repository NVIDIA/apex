/* coding=utf-8
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <torch/extension.h>

namespace fused_rope {

torch::Tensor fwd_cuda(
    torch::Tensor const& input,
    torch::Tensor const& cos,
    torch::Tensor const& sin);

torch::Tensor bwd_cuda(
    torch::Tensor const& output_grads,
    torch::Tensor const& cos,
    torch::Tensor const& sin);

torch::Tensor fwd(
    torch::Tensor & input,
    torch::Tensor & cos,
    torch::Tensor & sin) {
  if (!input.is_contiguous())
    input = input.contiguous();
  if (!cos.is_contiguous())
    cos = cos.contiguous();
  if (!sin.is_contiguous())
    sin = sin.contiguous();

  return fwd_cuda(input, cos, sin);
}

torch::Tensor bwd(
    torch::Tensor & output_grads,
    torch::Tensor & cos,
    torch::Tensor & sin) {
  if (!output_grads.is_contiguous())
    output_grads = output_grads.contiguous();
  if (!cos.is_contiguous())
    cos = cos.contiguous();
  if (!sin.is_contiguous())
    sin = sin.contiguous();

  return bwd_cuda(output_grads, cos, sin);
}

}  // end namespace fused_rope

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fused_rope::fwd,
        "Fused Rotary Positional Embedding -- Forward.");
  m.def("backward", &fused_rope::bwd,
        "Fused Rotary Positional Embedding -- Backward.");
}
