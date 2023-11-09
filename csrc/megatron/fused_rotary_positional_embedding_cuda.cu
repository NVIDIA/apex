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
                       const torch::Tensor &sin) {
  const int sq = input.size(0);
  const int b = input.size(1);
  const int np = input.size(2);
  const int hn = input.size(3);
  const int hn2 = cos.size(3);

  // output
  auto act_options = input.options().requires_grad(false);
  torch::Tensor output = torch::empty({sq, b, np, hn}, act_options);

  DISPATCH_FLOAT_HALF_AND_BFLOAT(
      input.scalar_type(), 0, "dispatch_fused_rope_forward",
      dispatch_fused_rope_forward(
          sq, b, np, hn, hn2, input.data_ptr<scalar_t_0>(),
          cos.data_ptr<scalar_t_0>(), sin.data_ptr<scalar_t_0>(),
          output.data_ptr<scalar_t_0>()););
  return output;
}

torch::Tensor bwd_cuda(const torch::Tensor &output_grads,
                       const torch::Tensor &cos, const torch::Tensor &sin) {
  const int sq = output_grads.size(0);
  const int b = output_grads.size(1);
  const int np = output_grads.size(2);
  const int hn = output_grads.size(3);
  const int hn2 = cos.size(3);

  auto act_options = output_grads.options().requires_grad(false);
  torch::Tensor input_grads = torch::empty({sq, b, np, hn}, act_options);

  DISPATCH_FLOAT_HALF_AND_BFLOAT(
      output_grads.scalar_type(), 0, "dispatch_fused_rope_backward",
      dispatch_fused_rope_backward(
          sq, b, np, hn, hn2, output_grads.data_ptr<scalar_t_0>(),
          cos.data_ptr<scalar_t_0>(), sin.data_ptr<scalar_t_0>(),
          input_grads.data_ptr<scalar_t_0>());)
  return input_grads;
}
}  // end namespace fused_rope
