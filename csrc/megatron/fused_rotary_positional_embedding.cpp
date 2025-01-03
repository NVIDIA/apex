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

torch::Tensor fwd_cuda(const torch::Tensor &input, const torch::Tensor &freqs,
                       const bool transpose_output);

torch::Tensor bwd_cuda(const torch::Tensor &output_grads,
                       const torch::Tensor &freqs, const bool transpose_output);

torch::Tensor fwd_cached_cuda(const torch::Tensor &input,
                              const torch::Tensor &cos,
                              const torch::Tensor &sin,
                              const bool transpose_output);

torch::Tensor bwd_cached_cuda(const torch::Tensor &output_grads,
                              const torch::Tensor &cos,
                              const torch::Tensor &sin,
                              const bool transpose_output);

torch::Tensor fwd_thd_cuda(const torch::Tensor &input,
                           const torch::Tensor &cu_seqlens,
                           const torch::Tensor &freqs);

torch::Tensor bwd_thd_cuda(const torch::Tensor &output_grads,
                           const torch::Tensor &cu_seqlens,
                           const torch::Tensor &freqs);

torch::Tensor fwd_2d_cuda(const torch::Tensor &input,
                          const torch::Tensor &cos_h,
                          const torch::Tensor &sin_h,
                          const torch::Tensor &cos_w,
                          const torch::Tensor &sin_w);

torch::Tensor bwd_2d_cuda(const torch::Tensor &output_grads,
                          const torch::Tensor &cos_h,
                          const torch::Tensor &sin_h,
                          const torch::Tensor &cos_w,
                          const torch::Tensor &sin_w);

torch::Tensor fwd(const at::Tensor &input, const at::Tensor &freqs,
                  const bool transpose_output) {
  TORCH_CHECK(input.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(freqs.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(input.size(0) == freqs.size(0),
              "expected input and freqs tensor have the same sequence length");
  TORCH_CHECK(freqs.size(1) == 1 && freqs.size(2) == 1,
              "expected the second and third dims of the freqs tensor equal 1");
  TORCH_CHECK(input.size(3) >= freqs.size(3),
              "expected the last dim of the input tensor equals or is "
              "greater than the freqs tensor");
  TORCH_CHECK(freqs.scalar_type() == at::ScalarType::Float,
              "Dtype of the freqs tensor must be float");

  return fwd_cuda(input, freqs, transpose_output);
}

torch::Tensor bwd(const torch::Tensor &output_grads, const at::Tensor &freqs,
                  const bool transpose_output) {
  TORCH_CHECK(output_grads.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(freqs.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(
      output_grads.size(0) == freqs.size(0),
      "expected output_grads and freqs tensor have the same sequence length");
  TORCH_CHECK(freqs.size(1) == 1 && freqs.size(2) == 1,
              "expected the second and third dims of the freqs tensor equal 1");
  TORCH_CHECK(output_grads.size(3) >= freqs.size(3),
              "expected the last dim of the output_grads tensor equals or is "
              "greater than the freqs tensor");
  TORCH_CHECK(freqs.scalar_type() == at::ScalarType::Float,
              "Dtype of the freqs tensor must be float");

  return bwd_cuda(output_grads, freqs, transpose_output);
}

torch::Tensor fwd_cached(const at::Tensor &input, const at::Tensor &cos,
                         const at::Tensor &sin, const bool transpose_output) {
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
  TORCH_CHECK(cos.size(3) == sin.size(3),
              "expected cos and sin tensor have the same last dim");
  TORCH_CHECK(input.size(3) >= cos.size(3),
              "expected the last dim of the input tensor equals or is "
              "greater than the cos tensor");
  TORCH_CHECK(cos.scalar_type() == sin.scalar_type(),
              "expected cos and sin tensor have the same dtype");

  return fwd_cached_cuda(input, cos, sin, transpose_output);
}

torch::Tensor bwd_cached(const torch::Tensor &output_grads,
                         const at::Tensor &cos, const at::Tensor &sin,
                         const bool transpose_output) {
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
  TORCH_CHECK(cos.size(3) == sin.size(3),
              "expected cos and sin tensor have the same last dim");
  TORCH_CHECK(output_grads.size(3) >= cos.size(3),
              "expected the last dim of the output_grads tensor equals or is "
              "greater than the cos tensor");
  TORCH_CHECK(cos.scalar_type() == sin.scalar_type(),
              "expected cos and sin tensor have the same dtype");

  return bwd_cached_cuda(output_grads, cos, sin, transpose_output);
}

torch::Tensor fwd_thd(const torch::Tensor &input,
                      const torch::Tensor &cu_seqlens,
                      const torch::Tensor &freqs) {
  TORCH_CHECK(input.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(cu_seqlens.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(freqs.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(freqs.size(1) == 1 && freqs.size(2) == 1,
              "expected the second and third dims of the freqs tensor equal 1");
  TORCH_CHECK(input.size(2) >= freqs.size(3),
              "expected the last dim of the input tensor equals or is "
              "greater than the freqs tensor");
  TORCH_CHECK(freqs.scalar_type() == at::ScalarType::Float,
              "Dtype of the freqs tensor must be float");

  return fwd_thd_cuda(input, cu_seqlens, freqs);
}

torch::Tensor bwd_thd(const torch::Tensor &output_grads,
                      const torch::Tensor &cu_seqlens,
                      const torch::Tensor &freqs) {
  TORCH_CHECK(output_grads.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(cu_seqlens.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(freqs.dim() == 4, "expected 4D tensor");
  TORCH_CHECK(freqs.size(1) == 1 && freqs.size(2) == 1,
              "expected the second and third dims of the freqs tensor equal 1");
  TORCH_CHECK(output_grads.size(2) >= freqs.size(3),
              "expected the last dim of the output_grads tensor equals or is "
              "greater than the freqs tensor");
  TORCH_CHECK(freqs.scalar_type() == at::ScalarType::Float,
              "Dtype of the freqs tensor must be float");

  return bwd_thd_cuda(output_grads, cu_seqlens, freqs);
}

torch::Tensor fwd_2d(const torch::Tensor &input, const torch::Tensor &cos_h,
                     const torch::Tensor &sin_h, const torch::Tensor &cos_w,
                     const torch::Tensor &sin_w) {
  TORCH_CHECK(input.dim() == 5, "expected input to be 5D tensor");
  TORCH_CHECK(cos_h.dim() == 4, "expected cos_h to be 4D tensor");
  TORCH_CHECK(sin_h.dim() == 4, "expected sin_h to be 4D tensor");
  TORCH_CHECK(cos_w.dim() == 4, "expected cos_w to be 4D tensor");
  TORCH_CHECK(sin_w.dim() == 4, "expected sin_w to be 4D tensor");
  TORCH_CHECK(cos_h.size(2) == 1, "expected third dim of cos_h/sin_h equals 1");
  TORCH_CHECK(input.size(1) <= cos_h.size(1),
              "expected input's height <= cos_h/sin_h's");
  TORCH_CHECK(input.size(4) / 2 == cos_h.size(3),
              "expected cos_h/sin_h's head dim equals input's head dim / 2");
  TORCH_CHECK(cos_w.size(2) == 1, "expected third dim of cos_w/sin_w equals 1");
  TORCH_CHECK(input.size(2) <= cos_w.size(1),
              "expected input's width <= cos_w/sin_w's");
  TORCH_CHECK(input.size(4) / 2 == cos_w.size(3),
              "expected cos_w/sin_w's head dim equals input's head dim / 2");

  return fwd_2d_cuda(input, cos_h, sin_h, cos_w, sin_w);
}

torch::Tensor bwd_2d(const torch::Tensor &output_grads,
                     const torch::Tensor &cos_h, const torch::Tensor &sin_h,
                     const torch::Tensor &cos_w, const torch::Tensor &sin_w) {
  TORCH_CHECK(output_grads.dim() == 5, "expected output_grads to be 5D tensor");
  TORCH_CHECK(cos_h.dim() == 4, "expected cos_h to be 4D tensor");
  TORCH_CHECK(sin_h.dim() == 4, "expected sin_h to be 4D tensor");
  TORCH_CHECK(cos_w.dim() == 4, "expected cos_w to be 4D tensor");
  TORCH_CHECK(sin_w.dim() == 4, "expected sin_w to be 4D tensor");
  TORCH_CHECK(cos_h.size(2) == 1, "expected third dim of cos_h/sin_h equals 1");
  TORCH_CHECK(output_grads.size(1) <= cos_h.size(1),
              "expected output_grads' height <= cos_h/sin_h's");
  TORCH_CHECK(output_grads.size(4) / 2 == cos_h.size(3),
              "expected cos_h/sin_h's head dim equals output_grads' head dim / 2");
  TORCH_CHECK(cos_w.size(2) == 1, "expected third dim of cos_w/sin_w equals 1");
  TORCH_CHECK(output_grads.size(2) <= cos_w.size(1),
              "expected output_grads' width <= cos_w/sin_w's");
  TORCH_CHECK(output_grads.size(4) / 2 == cos_w.size(3),
              "expected cos_w/sin_w's head dim equals output_grads' head dim / 2");

  return bwd_2d_cuda(output_grads, cos_h, sin_h, cos_w, sin_w);
}

}  // end namespace fused_rope

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fused_rope::fwd,
        "Fused Rotary Positional Embedding -- Forward.");
  m.def("backward", &fused_rope::bwd,
        "Fused Rotary Positional Embedding -- Backward.");
  // cache sin/cos
  m.def("forward_cached", &fused_rope::fwd_cached,
        "Fused Rotary Positional Embedding Cached -- Forward.");
  m.def("backward_cached", &fused_rope::bwd_cached,
        "Fused Rotary Positional Embedding Cached -- Backward.");
  // thd
  m.def("forward_thd", &fused_rope::fwd_thd,
        "Fused Rotary Positional Embedding for thd layout -- Forward.");
  m.def("backward_thd", &fused_rope::bwd_thd,
        "Fused Rotary Positional Embedding for thd layout -- Backward.");
  // 2d
  m.def("forward_2d", &fused_rope::fwd_2d,
        "2D Fused Rotary Positional Embedding -- Forward.");
  m.def("backward_2d", &fused_rope::bwd_2d,
        "2D Fused Rotary Positional Embedding -- Backward.");
}
