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

torch::Tensor fwd_cuda(const torch::Tensor &input, const torch::Tensor &cos,
                       const torch::Tensor &sin);

torch::Tensor bwd_cuda(const torch::Tensor &output_grads,
                       const torch::Tensor &cos, const torch::Tensor &sin);

torch::Tensor fwd(const at::Tensor &input_, const at::Tensor &cos_,
                  const at::Tensor &sin_) {
  auto input = input_.contiguous();
  auto cos = cos_.contiguous();
  auto sin = sin_.contiguous();
  TORCH_CHECK(input.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(cos.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(sin.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(input.size(0) == cos.size(0),
              "expected input and cos tensor have the same sequence length");
  TORCH_CHECK(input.size(0) == sin.size(0),
              "expected input and sin tensor have the same sequence length");
  TORCH_CHECK(cos.size(1) == 1 && cos.size(2) == 1,
              "expected the second and third dims of the cos tensor equal 1");
  TORCH_CHECK(sin.size(1) == 1 && sin.size(2) == 1,
              "expected the second and third dims of the sin tensor equal 1");
  TORCH_CHECK(input.size(3) >= cos.size(3),
              "expected the last dim of the input tensor is greater than the "
              "cos tensor");
  TORCH_CHECK(input.size(3) >= sin.size(3),
              "expected the last dim of the input tensor is greater than the "
              "sin tensor");

  return fwd_cuda(input, cos, sin);
}

torch::Tensor bwd(const torch::Tensor &output_grads_, const at::Tensor &cos_,
                  const at::Tensor &sin_) {
  auto output_grads = output_grads_.contiguous();
  auto cos = cos_.contiguous();
  auto sin = sin_.contiguous();
  TORCH_CHECK(output_grads.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(cos.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(sin.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(
      output_grads.size(0) == cos.size(0),
      "expected output_grads and cos tensor have the same sequence length");
  TORCH_CHECK(
      output_grads.size(0) == sin.size(0),
      "expected output_grads and sin tensor have the same sequence length");
  TORCH_CHECK(cos.size(1) == 1 && cos.size(2) == 1,
              "expected the second and third dims of the cos tensor equal 1");
  TORCH_CHECK(sin.size(1) == 1 && sin.size(2) == 1,
              "expected the second and third dims of the sin tensor equal 1");
  TORCH_CHECK(
      output_grads.size(3) >= cos.size(3),
      "expected the last dim of the output_grads tensor is greater than the "
      "cos tensor");
  TORCH_CHECK(
      output_grads.size(3) >= sin.size(3),
      "expected the last dim of the output_grads tensor is greater than the "
      "sin tensor");

  return bwd_cuda(output_grads, cos, sin);
}

}  // end namespace fused_rope

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fused_rope::fwd,
        "Fused Rotary Positional Embedding -- Forward.");
  m.def("backward", &fused_rope::bwd,
        "Fused Rotary Positional Embedding -- Backward.");
}
