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

torch::Tensor fwd_cuda(const torch::Tensor &input, const torch::Tensor &freqs,
                       const bool transpose_output) {
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
  // freqs' shape is always (s, 1, 1, d2), so the strides are same under
  // different memory formats
  const int d2 = freqs.size(3);

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
          freqs.data_ptr<float>(), output.data_ptr<scalar_t_0>()););
  return output;
}

torch::Tensor bwd_cuda(const torch::Tensor &output_grads,
                       const torch::Tensor &freqs,
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
  // freqs' shape is always (s, 1, 1, d2), so the strides are same under
  // different memory formats
  const int d2 = freqs.size(3);

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
          output_grads.data_ptr<scalar_t_0>(), freqs.data_ptr<float>(),
          input_grads.data_ptr<scalar_t_0>()););
  return input_grads;
}

#define DISPATCH_FUSED_ROPE_TYPES(TYPE1, TYPE2, NAME, ...)                     \
  switch (TYPE1) {                                                             \
  case at::ScalarType::Float: {                                                \
    using scalar_t_0 = float;                                                  \
    switch (TYPE2) {                                                           \
    case at::ScalarType::Float: {                                              \
      using scalar_t_1 = float;                                                \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      TORCH_CHECK(false, #NAME, " not supported for '", toString(TYPE1),       \
                  "' with '", toString(TYPE2), "'");                           \
    }                                                                          \
    break;                                                                     \
  }                                                                            \
  case at::ScalarType::Half: {                                                 \
    using scalar_t_0 = at::Half;                                               \
    switch (TYPE2) {                                                           \
    case at::ScalarType::Float: {                                              \
      using scalar_t_1 = float;                                                \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case at::ScalarType::Half: {                                               \
      using scalar_t_1 = at::Half;                                             \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      TORCH_CHECK(false, #NAME, " not supported for '", toString(TYPE1),       \
                  "' with '", toString(TYPE2), "'");                           \
    }                                                                          \
    break;                                                                     \
  }                                                                            \
  case at::ScalarType::BFloat16: {                                             \
    using scalar_t_0 = at::BFloat16;                                           \
    switch (TYPE2) {                                                           \
    case at::ScalarType::Float: {                                              \
      using scalar_t_1 = float;                                                \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case at::ScalarType::BFloat16: {                                           \
      using scalar_t_1 = at::BFloat16;                                         \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      TORCH_CHECK(false, #NAME, " not supported for '", toString(TYPE1),       \
                  "' with '", toString(TYPE2), "'");                           \
    }                                                                          \
    break;                                                                     \
  }                                                                            \
  default:                                                                     \
    TORCH_CHECK(false, #NAME, " not supported for '", toString(TYPE1),         \
                "' with '", toString(TYPE2), "'");                             \
  }

torch::Tensor fwd_cached_cuda(const torch::Tensor &input,
                              const torch::Tensor &cos,
                              const torch::Tensor &sin,
                              const bool transpose_output) {
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

  DISPATCH_FUSED_ROPE_TYPES(
      input.scalar_type(), cos.scalar_type(),
      "dispatch_fused_rope_cached_forward",
      dispatch_fused_rope_cached_forward(
          s, b, h, d, d2, stride_s, stride_b, stride_h, stride_d, o_stride_s,
          o_stride_b, o_stride_h, o_stride_d, input.data_ptr<scalar_t_0>(),
          cos.data_ptr<scalar_t_1>(), sin.data_ptr<scalar_t_1>(),
          output.data_ptr<scalar_t_0>()););
  return output;
}

torch::Tensor bwd_cached_cuda(const torch::Tensor &output_grads,
                              const torch::Tensor &cos,
                              const torch::Tensor &sin,
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

  DISPATCH_FUSED_ROPE_TYPES(
      output_grads.scalar_type(), cos.scalar_type(),
      "dispatch_fused_rope_cached_backward",
      dispatch_fused_rope_cached_backward(
          s, b, h, d, d2, stride_s, stride_b, stride_h, stride_d, o_stride_s,
          o_stride_b, o_stride_h, o_stride_d,
          output_grads.data_ptr<scalar_t_0>(), cos.data_ptr<scalar_t_1>(),
          sin.data_ptr<scalar_t_1>(), input_grads.data_ptr<scalar_t_0>()););
  return input_grads;
}

torch::Tensor fwd_thd_cuda(const torch::Tensor &input,
                           const torch::Tensor &cu_seqlens,
                           const torch::Tensor &freqs) {
  // input sizes: (t, h, d)
  // t: cumulative sum of sequence lengths
  // h: head num
  // d: dim of each head
  const int t = input.size(0);
  const int h = input.size(1);
  const int d = input.size(2);
  // input strides
  const int stride_t = input.stride(0);
  const int stride_h = input.stride(1);
  const int stride_d = input.stride(2);
  // batch size
  const int b = cu_seqlens.size(0) - 1;
  // freqs' shape is (max_s, 1, 1, d2)
  const int max_s = freqs.size(0);
  const int d2 = freqs.size(3);

  // output
  auto act_options = input.options().requires_grad(false);
  auto output = torch::empty({t, h, d}, act_options);
  // output strides
  const int o_stride_t = output.stride(0);
  const int o_stride_h = output.stride(1);
  const int o_stride_d = output.stride(2);

  DISPATCH_FLOAT_HALF_AND_BFLOAT(
      input.scalar_type(), 0, "dispatch_fused_rope_thd_forward",
      dispatch_fused_rope_thd_forward(
          max_s, b, h, d, d2, stride_t, stride_h, stride_d, o_stride_t,
          o_stride_h, o_stride_d, input.data_ptr<scalar_t_0>(),
          cu_seqlens.data_ptr<int>(), freqs.data_ptr<float>(),
          output.data_ptr<scalar_t_0>()););
  return output;
}

torch::Tensor bwd_thd_cuda(const torch::Tensor &output_grads,
                           const torch::Tensor &cu_seqlens,
                           const torch::Tensor &freqs) {
  // output_grads sizes: (t, h, d)
  // t: cumulative sum of sequence lengths
  // h: head num
  // d: dim of each head
  const int t = output_grads.size(0);
  const int h = output_grads.size(1);
  const int d = output_grads.size(2);
  // output_grads strides
  const int stride_t = output_grads.stride(0);
  const int stride_h = output_grads.stride(1);
  const int stride_d = output_grads.stride(2);
  // batch size
  const int b = cu_seqlens.size(0) - 1;
  // freqs' shape is (max_s, 1, 1, d2)
  const int max_s = freqs.size(0);
  const int d2 = freqs.size(3);

  auto act_options = output_grads.options().requires_grad(false);
  auto input_grads = torch::empty({t, h, d}, act_options);
  const int o_stride_t = input_grads.stride(0);
  const int o_stride_h = input_grads.stride(1);
  const int o_stride_d = input_grads.stride(2);

  DISPATCH_FLOAT_HALF_AND_BFLOAT(
      output_grads.scalar_type(), 0, "dispatch_fused_rope_thd_backward",
      dispatch_fused_rope_thd_backward(
          max_s, b, h, d, d2, stride_t, stride_h, stride_d, o_stride_t,
          o_stride_h, o_stride_d, output_grads.data_ptr<scalar_t_0>(),
          cu_seqlens.data_ptr<int>(), freqs.data_ptr<float>(),
          input_grads.data_ptr<scalar_t_0>()););
  return input_grads;
}

torch::Tensor fwd_2d_cuda(const torch::Tensor &input,
                          const torch::Tensor &cos_h,
                          const torch::Tensor &sin_h,
                          const torch::Tensor &cos_w,
                          const torch::Tensor &sin_w) {
  // input sizes: (b, ih, iw, h, d)
  // b: batch size
  // ih: image height
  // iw: image width
  // h: head num
  // d: dim of each head
  const int b = input.size(0);
  const int ih = input.size(1);
  const int iw = input.size(2);
  const int h = input.size(3);
  const int d = input.size(4);
  // input strides
  const int stride_b = input.stride(0);
  const int stride_ih = input.stride(1);
  const int stride_iw = input.stride(2);
  const int stride_h = input.stride(3);
  const int stride_d = input.stride(4);

  // output
  auto act_options = input.options().requires_grad(false);
  auto output = torch::empty({b, ih * iw, h, d}, act_options);
  // output strides
  const int o_stride_b = output.stride(0);
  const int o_stride_s = output.stride(1);
  const int o_stride_h = output.stride(2);
  const int o_stride_d = output.stride(3);

  DISPATCH_FUSED_ROPE_TYPES(
      input.scalar_type(), cos_h.scalar_type(),
      "dispatch_fused_rope_2d_forward",
      dispatch_fused_rope_2d_forward(
          b, ih, iw, h, d, stride_b, stride_ih, stride_iw, stride_h, stride_d,
          o_stride_b, o_stride_s, o_stride_h, o_stride_d,
          input.data_ptr<scalar_t_0>(), cos_h.data_ptr<scalar_t_1>(),
          sin_h.data_ptr<scalar_t_1>(), cos_w.data_ptr<scalar_t_1>(),
          sin_w.data_ptr<scalar_t_1>(), output.data_ptr<scalar_t_0>()););
  return output;
}

torch::Tensor bwd_2d_cuda(const torch::Tensor &output_grads,
                          const torch::Tensor &cos_h,
                          const torch::Tensor &sin_h,
                          const torch::Tensor &cos_w,
                          const torch::Tensor &sin_w) {
  // output_grads sizes: (b, ih, iw, h, d)
  // b: batch size
  // ih: image height
  // iw: image width
  // h: head num
  // d: dim of each head
  const int b = output_grads.size(0);
  const int ih = output_grads.size(1);
  const int iw = output_grads.size(2);
  const int h = output_grads.size(3);
  const int d = output_grads.size(4);
  // output_grads strides
  const int stride_b = output_grads.stride(0);
  const int stride_ih = output_grads.stride(1);
  const int stride_iw = output_grads.stride(2);
  const int stride_h = output_grads.stride(3);
  const int stride_d = output_grads.stride(4);

  auto act_options = output_grads.options().requires_grad(false);
  auto input_grads = torch::empty({b, ih * iw, h, d}, act_options);
  const int o_stride_b = input_grads.stride(0);
  const int o_stride_s = input_grads.stride(1);
  const int o_stride_h = input_grads.stride(2);
  const int o_stride_d = input_grads.stride(3);

  DISPATCH_FUSED_ROPE_TYPES(
      output_grads.scalar_type(), cos_h.scalar_type(),
      "dispatch_fused_rope_2d_backward",
      dispatch_fused_rope_2d_backward(
          b, ih, iw, h, d, stride_b, stride_ih, stride_iw, stride_h, stride_d,
          o_stride_b, o_stride_s, o_stride_h, o_stride_d,
          output_grads.data_ptr<scalar_t_0>(), cos_h.data_ptr<scalar_t_1>(),
          sin_h.data_ptr<scalar_t_1>(), cos_w.data_ptr<scalar_t_1>(),
          sin_w.data_ptr<scalar_t_1>(), input_grads.data_ptr<scalar_t_0>()););
  return input_grads;
}

}  // end namespace fused_rope
