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
namespace scaled_masked_softmax {

at::Tensor fwd_cuda(at::Tensor const& input, at::Tensor const& mask, float scale_factor);

at::Tensor bwd_cuda(at::Tensor const& output_grads, at::Tensor const& softmax_results, float scale_factor);

int get_batch_per_block_cuda(int query_seq_len, int key_seq_len, int batches, int attn_heads);

at::Tensor fwd(at::Tensor const& input, at::Tensor const& mask, double scale_factor) {
  auto input_arg = input;
  auto mask_arg = mask;
  TORCH_CHECK(input_arg.dim() == 4, "expected 4D tensor");
  TORCH_CHECK((input_arg.scalar_type() == at::ScalarType::Half) || (input_arg.scalar_type() == at::ScalarType::BFloat16),
              "Only fp16 and bf16 are supported");
  TORCH_CHECK(mask_arg.dim() == 4, "expected 4D tensor");
  if (!input_arg.is_contiguous()) input_arg = input_arg.contiguous();
  if (!mask_arg.is_contiguous()) mask_arg = mask_arg.contiguous();

  return fwd_cuda(input_arg, mask_arg, static_cast<float>(scale_factor));
}

at::Tensor bwd(at::Tensor const& output_grads, at::Tensor const& softmax_results, double scale_factor) {
  auto output_grads_arg = output_grads;
  auto softmax_results_arg = softmax_results;
  TORCH_CHECK(output_grads_arg.dim() == 4, "expected 3D tensor");
  TORCH_CHECK(softmax_results_arg.dim() == 4, "expected 3D tensor");

  TORCH_CHECK((output_grads_arg.scalar_type() == at::ScalarType::Half) ||
                  (output_grads_arg.scalar_type() == at::ScalarType::BFloat16),
              "Only fp16 and bf16 are supported");
  TORCH_CHECK((softmax_results_arg.scalar_type() == at::ScalarType::Half) ||
                  (softmax_results_arg.scalar_type() == at::ScalarType::BFloat16),
              "Only fp16 and bf16 are supported");
  if (!output_grads_arg.is_contiguous()) output_grads_arg = output_grads_arg.contiguous();
  if (!softmax_results_arg.is_contiguous()) softmax_results_arg = softmax_results_arg.contiguous();

  return bwd_cuda(output_grads_arg, softmax_results_arg, static_cast<float>(scale_factor));
}

int64_t get_batch_per_block(int64_t query_seq_len, int64_t key_seq_len, int64_t batches, int64_t attn_heads) {
  return get_batch_per_block_cuda(static_cast<int>(query_seq_len), static_cast<int>(key_seq_len),
                                  static_cast<int>(batches), static_cast<int>(attn_heads));
}

}  // end namespace scaled_masked_softmax
}  // end namespace fused_softmax
}  // end namespace multihead_attn

TORCH_LIBRARY_FRAGMENT(apex, m) {
  m.def("scaled_masked_softmax_forward(Tensor input, Tensor mask, float scale_factor) -> Tensor");
  m.def("scaled_masked_softmax_backward(Tensor output_grads, Tensor softmax_results, float scale_factor) -> Tensor");
  m.def("scaled_masked_softmax_get_batch_per_block(int query_seq_len, int key_seq_len, int batches, "
        "int attn_heads) -> int");
}

TORCH_LIBRARY_IMPL(apex, CUDA, m) {
  m.impl("scaled_masked_softmax_forward", &multihead_attn::fused_softmax::scaled_masked_softmax::fwd);
  m.impl("scaled_masked_softmax_backward", &multihead_attn::fused_softmax::scaled_masked_softmax::bwd);
}

TORCH_LIBRARY_IMPL(apex, CompositeExplicitAutograd, m) {
  m.impl("scaled_masked_softmax_get_batch_per_block",
         &multihead_attn::fused_softmax::scaled_masked_softmax::get_batch_per_block);
}
