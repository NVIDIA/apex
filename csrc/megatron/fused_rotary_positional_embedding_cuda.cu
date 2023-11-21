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

#include <ATen/ATen.h>

#include "fused_rotary_positional_embedding.h"
#include "type_shim.h"

namespace fused_rope {

torch::Tensor fwd_cuda(const torch::Tensor &input, const torch::Tensor &cos,
                       const torch::Tensor &sin, const bool transpose_output) {
  // input sizes: (s, b, h, d)
  // s: sequence length
  // b: batch size
  // h: head num
  // d: dim of each head
  const int s = input.size(0);
  const int b = input.size(1);
  const int h = input.size(2);
  const int d = input.size(3);
  // input strides
  const int stride_s = input.stride(0);
  const int stride_b = input.stride(1);
  const int stride_h = input.stride(2);
  const int stride_d = input.stride(3);
  // cos/sin's shape is always (s, 1, 1, d2), so the strides are same under
  // different memory formats
  const int d2 = cos.size(3);

  // output
  auto act_options = input.options().requires_grad(false);
  torch::Tensor output;
  if (transpose_output) {
    output = torch::empty({b, s, h, d}, act_options).transpose(0, 1);
  } else {
    output = torch::empty({s, b, h, d}, act_options);
  }
  // output strides
  const int o_stride_s = output.stride(0);
  const int o_stride_b = output.stride(1);
  const int o_stride_h = output.stride(2);
  const int o_stride_d = output.stride(3);

  DISPATCH_FLOAT_HALF_AND_BFLOAT(
      input.scalar_type(), 0, "dispatch_fused_rope_forward",
      dispatch_fused_rope_forward(
          s, b, h, d, d2, stride_s, stride_b, stride_h, stride_d, o_stride_s,
          o_stride_b, o_stride_h, o_stride_d, input.data_ptr<scalar_t_0>(),
          cos.data_ptr<scalar_t_0>(), sin.data_ptr<scalar_t_0>(),
          output.data_ptr<scalar_t_0>()););
  return output;
}

torch::Tensor bwd_cuda(const torch::Tensor &output_grads,
                       const torch::Tensor &cos, const torch::Tensor &sin,
                       const bool transpose_output) {
  // output_grads sizes: (s, b, h, d)
  // s: sequence length
  // b: batch size
  // h: head num
  // d: dim of each head
  const int s = output_grads.size(0);
  const int b = output_grads.size(1);
  const int h = output_grads.size(2);
  const int d = output_grads.size(3);
  // output_grads strides
  const int stride_s = output_grads.stride(0);
  const int stride_b = output_grads.stride(1);
  const int stride_h = output_grads.stride(2);
  const int stride_d = output_grads.stride(3);
  // cos/sin's shape is always (s, 1, 1, d2), so the strides are same under
  // different memory formats
  const int d2 = cos.size(3);

  auto act_options = output_grads.options().requires_grad(false);
  torch::Tensor input_grads;
  if (transpose_output) {
    input_grads = torch::empty({b, s, h, d}, act_options).transpose(0, 1);
  } else {
    input_grads = torch::empty({s, b, h, d}, act_options);
  }
  const int o_stride_s = input_grads.stride(0);
  const int o_stride_b = input_grads.stride(1);
  const int o_stride_h = input_grads.stride(2);
  const int o_stride_d = input_grads.stride(3);

  DISPATCH_FLOAT_HALF_AND_BFLOAT(
      output_grads.scalar_type(), 0, "dispatch_fused_rope_backward",
      dispatch_fused_rope_backward(
          s, b, h, d, d2, stride_s, stride_b, stride_h, stride_d, o_stride_s,
          o_stride_b, o_stride_h, o_stride_d,
          output_grads.data_ptr<scalar_t_0>(), cos.data_ptr<scalar_t_0>(),
          sin.data_ptr<scalar_t_0>(), input_grads.data_ptr<scalar_t_0>());)
  return input_grads;
}
}  // end namespace fused_rope
