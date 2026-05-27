#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#if __has_include(<cuda_profiler_api.h>)
#include <cuda_profiler_api.h>
#endif
#include <cuda_runtime.h>
#include <math.h>
#include <ATen/ATen.h>

#include <iostream>
#include <vector>

#include "dropout.cuh"
#include "layer_norm.cuh"
#include "softmax.cuh"
#include "strided_batched_gemm.cuh"

namespace multihead_attn {
namespace encdec_norm_add {
namespace cublas_gemmex {

std::vector<at::Tensor> fwd_cuda(bool use_time_mask, bool is_training, int heads, at::Tensor const& inputs_q,
                                    at::Tensor const& inputs_kv, at::Tensor const& lyr_nrm_gamma_weights,
                                    at::Tensor const& lyr_nrm_beta_weights, at::Tensor const& input_weights_q,
                                    at::Tensor const& input_weights_kv, at::Tensor const& output_weights,
                                    const uint8_t* pad_mask, float dropout_prob) {
  const int embed_dim = inputs_q.size(2);
  const int sequences = inputs_q.size(1);
  const int q_seq_len = inputs_q.size(0);
  const int k_seq_len = inputs_kv.size(0);
  const int batches_q = sequences * q_seq_len;
  const int batches_kv = sequences * k_seq_len;
  const int total_tokens_q = batches_q * embed_dim;
  const int head_dim = embed_dim / heads;
  const int output_lin_q_dim = embed_dim;
  const int output_lin_kv_dim = 2 * embed_dim;
  const int attn_batches = heads * sequences;
  const int lead_dim_q = attn_batches * head_dim;
  const int lead_dim_kv = attn_batches * 2 * head_dim;
  const int batch_stride_q = head_dim;
  const int batch_stride_kv = 2 * head_dim;
  const int dropout_elems = attn_batches * q_seq_len * k_seq_len;
  const float alpha = 1.0;
  const float beta = 0.0;
  const float scale = 1.0 / sqrt(static_cast<float>(head_dim));

  // There is no reason to use more than one stream as every kernel is
  // sequentially dependent
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cublasSetStream(handle, stream);

  // 3 Intermediate Results + Output (Note: dropout intermediates are generated
  // by ATen library code)
  auto act_options = inputs_q.options().requires_grad(false);
  auto lyr_nrm_options = act_options.dtype(at::kFloat);
  auto mask_options = act_options.dtype(at::kByte);

  at::Tensor lyr_nrm_mean = at::empty({batches_q}, lyr_nrm_options);
  at::Tensor lyr_nrm_invvar = at::empty({batches_q}, lyr_nrm_options);
  at::Tensor lyr_nrm_results = at::empty_like(inputs_q, act_options);

  at::Tensor input_lin_q_results = at::empty({q_seq_len, sequences, output_lin_q_dim}, act_options);
  at::Tensor input_lin_kv_results = at::empty({k_seq_len, sequences, output_lin_kv_dim}, act_options);
  at::Tensor softmax_results = at::empty({attn_batches, q_seq_len, k_seq_len}, act_options);
  at::Tensor dropout_results = at::empty({attn_batches, q_seq_len, k_seq_len}, act_options);
  at::Tensor dropout_mask = at::empty({attn_batches, q_seq_len, k_seq_len}, mask_options);
  at::Tensor matmul2_results = at::empty({q_seq_len, attn_batches, head_dim}, act_options);
  at::Tensor output_lin_results = at::empty_like(inputs_q, act_options);
  at::Tensor dropout_add_mask = at::empty_like(inputs_q, mask_options);
  at::Tensor outputs = at::empty_like(inputs_q, act_options);

  // Input Linear Results Pointers to Q, K, and V of interviewed activations
  void* q_lin_results_ptr = static_cast<void*>(input_lin_q_results.data_ptr());
  void* k_lin_results_ptr = static_cast<void*>(input_lin_kv_results.data_ptr());
  void* v_lin_results_ptr = static_cast<void*>(static_cast<half*>(input_lin_kv_results.data_ptr()) + head_dim);

  // Softmax Intermediate Result Ptr (used by Matmul1 -> Softmax)
  void* softmax_results_ptr = static_cast<void*>(softmax_results.data_ptr());

  char a_layout_t{'t'};
  char a_layout_n{'n'};
  char b_layout_n{'n'};

  TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  // Layer Norm
  HostApplyLayerNorm<at::Half, float>(
      static_cast<at::Half*>(lyr_nrm_results.data_ptr()), static_cast<float*>(lyr_nrm_mean.data_ptr()),
      static_cast<float*>(lyr_nrm_invvar.data_ptr()), static_cast<const at::Half*>(inputs_q.data_ptr()),
      static_cast<int>(batches_q),  // n1
      static_cast<int>(embed_dim),  // n2
      1.0e-5, static_cast<const at::Half*>(lyr_nrm_gamma_weights.data_ptr()),
      static_cast<const at::Half*>(lyr_nrm_beta_weights.data_ptr()));

  // Input Linear Q Fwd
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, output_lin_q_dim, batches_q, embed_dim, static_cast<const void*>(&alpha),
      static_cast<const void*>(input_weights_q.data_ptr()), CUDA_R_16F, embed_dim,
      // static_cast<const void*>(inputs_q.data_ptr()),
      static_cast<const void*>(lyr_nrm_results.data_ptr()), CUDA_R_16F, embed_dim, static_cast<const void*>(&beta),
      q_lin_results_ptr, CUDA_R_16F, output_lin_q_dim, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Input Linear KV Fwd
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, output_lin_kv_dim, batches_kv, embed_dim, static_cast<const void*>(&alpha),
      static_cast<const void*>(input_weights_kv.data_ptr()), CUDA_R_16F, embed_dim,
      static_cast<const void*>(inputs_kv.data_ptr()), CUDA_R_16F, embed_dim, static_cast<const void*>(&beta),
      k_lin_results_ptr, CUDA_R_16F, output_lin_kv_dim, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // MatMul1 of Dot-Product Attention Plus scaling by 1/Sqrt(head size)
  gemm_switch_fp32accum(a_layout_t, b_layout_n, k_seq_len, q_seq_len, head_dim, scale,
                        static_cast<const half*>(k_lin_results_ptr), lead_dim_kv, batch_stride_kv,
                        static_cast<const half*>(q_lin_results_ptr), lead_dim_q, batch_stride_q, beta,
                        static_cast<half*>(softmax_results_ptr), k_seq_len, k_seq_len * q_seq_len, attn_batches);

  // Padded Softmax
  bool softmax_success = false;
  if (pad_mask == nullptr) {
    softmax_success = dispatch_softmax<half, half, float>(reinterpret_cast<half*>(softmax_results_ptr),
                                                          reinterpret_cast<const half*>(softmax_results_ptr), k_seq_len,
                                                          k_seq_len, attn_batches * q_seq_len);
  } else {
    if (use_time_mask) {
      softmax_success = dispatch_time_masked_softmax<half, half, float>(
          reinterpret_cast<half*>(softmax_results_ptr), reinterpret_cast<const half*>(softmax_results_ptr), pad_mask,
          k_seq_len, k_seq_len, attn_batches * q_seq_len, q_seq_len);
    } else {
      softmax_success = dispatch_masked_softmax<half, half, float>(
          reinterpret_cast<half*>(softmax_results_ptr), reinterpret_cast<const half*>(softmax_results_ptr), pad_mask,
          k_seq_len, k_seq_len, attn_batches * q_seq_len, attn_batches * q_seq_len / sequences);
    }
  }
  assert(softmax_success);

  if (is_training) {
    apex_fused_dropout_cuda<at::Half, float, uint32_t>(
        static_cast<at::Half const*>(softmax_results.data_ptr()), static_cast<at::Half*>(dropout_results.data_ptr()),
        static_cast<uint8_t*>(dropout_mask.data_ptr()), dropout_elems, (1.0f - dropout_prob));
  }

  // Matmul2
  gemm_switch_fp32accum(a_layout_n, b_layout_n, head_dim, q_seq_len, k_seq_len, alpha,
                        static_cast<const half*>(v_lin_results_ptr), lead_dim_kv, batch_stride_kv,
                        (is_training) ? static_cast<const half*>(dropout_results.data_ptr())
                                      : static_cast<const half*>(softmax_results.data_ptr()),
                        // static_cast<const half*>(dropout_results.data_ptr()),
                        k_seq_len, k_seq_len * q_seq_len, beta, static_cast<half*>(matmul2_results.data_ptr()),
                        head_dim * attn_batches, head_dim, attn_batches);

  // Output Linear
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, embed_dim, batches_q, embed_dim, static_cast<const void*>(&alpha),
      static_cast<const void*>(output_weights.data_ptr()), CUDA_R_16F, embed_dim,
      static_cast<const void*>(matmul2_results.data_ptr()), CUDA_R_16F, embed_dim, static_cast<const void*>(&beta),
      static_cast<void*>(output_lin_results.data_ptr()), CUDA_R_16F, embed_dim, CUDA_R_32F,
      // CUBLAS_GEMM_ALGO1_TENSOR_OP));
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // End-of-block Dropout-Add
  if (is_training) {
    apex_dropout_add_cuda<at::Half, float, uint32_t>(
        static_cast<at::Half const*>(output_lin_results.data_ptr()), static_cast<at::Half const*>(inputs_q.data_ptr()),
        static_cast<at::Half*>(outputs.data_ptr()), static_cast<uint8_t*>(dropout_add_mask.data_ptr()), total_tokens_q,
        (1.0f - dropout_prob));
  } else {
    apex_add_cuda<at::Half, float, uint32_t>(static_cast<at::Half const*>(output_lin_results.data_ptr()),
                                             static_cast<at::Half const*>(inputs_q.data_ptr()),
                                             static_cast<at::Half*>(outputs.data_ptr()), total_tokens_q);
  }

  TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

  return {lyr_nrm_results,      lyr_nrm_mean,     lyr_nrm_invvar,  input_lin_q_results,
          input_lin_kv_results, softmax_results,  dropout_results, dropout_mask,
          matmul2_results,      dropout_add_mask, outputs};
}

std::vector<at::Tensor> bwd_cuda(int heads, at::Tensor const& output_grads, at::Tensor const& matmul2_results,
                                    at::Tensor const& dropout_results, at::Tensor const& softmax_results,
                                    at::Tensor const& input_lin_q_results, at::Tensor const& input_lin_kv_results,
                                    at::Tensor const& lyr_nrm_results, at::Tensor const& lyr_nrm_mean,
                                    at::Tensor const& lyr_nrm_invvar, at::Tensor const& inputs_q,
                                    at::Tensor const& inputs_kv, at::Tensor const& lyr_nrm_gamma_weights,
                                    at::Tensor const& lyr_nrm_beta_weights, at::Tensor const& input_weights_q,
                                    at::Tensor const& input_weights_kv, at::Tensor const& output_weights,
                                    at::Tensor const& dropout_mask, at::Tensor const& dropout_add_mask,
                                    float dropout_prob) {
  const int embed_dim = inputs_q.size(2);
  const int sequences = inputs_q.size(1);
  const int q_seq_len = inputs_q.size(0);
  const int k_seq_len = inputs_kv.size(0);
  const int batches_q = sequences * q_seq_len;
  const int batches_kv = sequences * k_seq_len;
  const int total_tokens_q = batches_q * embed_dim;
  const int head_dim = embed_dim / heads;
  const int output_lin_q_dim = embed_dim;
  const int output_lin_kv_dim = 2 * embed_dim;
  const int attn_batches = heads * sequences;
  const int lead_dim_q = attn_batches * head_dim;
  const int lead_dim_kv = attn_batches * 2 * head_dim;
  const int batch_stride_q = head_dim;
  const int batch_stride_kv = 2 * head_dim;
  const int dropout_elems = attn_batches * q_seq_len * k_seq_len;
  const float alpha = 1.0;
  const float beta = 0.0;
  const float scale = 1.0 / sqrt(static_cast<float>(head_dim));

  // TODO: Streams can be used in Backprop but I haven't added more than one
  // in my first attempt to create the code
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cublasSetStream(handle, stream);

  // Output Tensor Allocations
  at::Tensor input_q_grads = at::empty_like(inputs_q);
  at::Tensor input_kv_grads = at::empty_like(inputs_kv);
  at::Tensor lyr_nrm_gamma_grads = at::empty_like(lyr_nrm_gamma_weights);
  at::Tensor lyr_nrm_beta_grads = at::empty_like(lyr_nrm_beta_weights);
  at::Tensor input_weight_q_grads = at::empty_like(input_weights_q);
  at::Tensor input_weight_kv_grads = at::empty_like(input_weights_kv);
  at::Tensor output_weight_grads = at::empty_like(output_weights);
  // Intermediate Tensor Allocations
  at::Tensor dropout_add_grads = at::empty_like(output_grads);
  at::Tensor output_lin_grads = at::empty_like(matmul2_results);
  at::Tensor matmul2_grads = at::empty_like(dropout_results);
  at::Tensor input_lin_q_output_grads = at::empty_like(input_lin_q_results);
  at::Tensor input_lin_kv_output_grads = at::empty_like(input_lin_kv_results);
  at::Tensor input_lin_q_grads = at::empty_like(inputs_q);

  auto q_lin_results_ptr = static_cast<half*>(input_lin_q_results.data_ptr());
  auto k_lin_results_ptr = static_cast<half*>(input_lin_kv_results.data_ptr());
  auto v_lin_results_ptr = static_cast<half*>(input_lin_kv_results.data_ptr()) + head_dim;

  auto q_lin_grads_ptr = static_cast<half*>(input_lin_q_output_grads.data_ptr());
  auto k_lin_grads_ptr = static_cast<half*>(input_lin_kv_output_grads.data_ptr());
  auto v_lin_grads_ptr = static_cast<half*>(input_lin_kv_output_grads.data_ptr()) + head_dim;

  char a_layout_n{'n'};
  char a_layout_t{'t'};
  char b_layout_n{'n'};
  char b_layout_t{'t'};

  TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  // Dropout Add Backward
  apex_masked_scale_cuda<at::Half, float, uint32_t>(
      static_cast<at::Half const*>(output_grads.data_ptr()), static_cast<at::Half*>(dropout_add_grads.data_ptr()),
      static_cast<uint8_t const*>(dropout_add_mask.data_ptr()), total_tokens_q, (1.0 / (1.0 - dropout_prob)));

  // Output Linear Dgrad
  TORCH_CUDABLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, embed_dim, batches_q, embed_dim,
                                    static_cast<const void*>(&alpha),
                                    static_cast<const void*>(output_weights.data_ptr()), CUDA_R_16F, embed_dim,
                                    static_cast<const void*>(dropout_add_grads.data_ptr()), CUDA_R_16F, embed_dim,
                                    static_cast<const void*>(&beta), static_cast<void*>(output_lin_grads.data_ptr()),
                                    CUDA_R_16F, embed_dim, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Output Linear Wgrad
  TORCH_CUDABLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, embed_dim, embed_dim, batches_q,
                                    static_cast<const void*>(&alpha),
                                    static_cast<const void*>(matmul2_results.data_ptr()), CUDA_R_16F, embed_dim,
                                    static_cast<const void*>(dropout_add_grads.data_ptr()), CUDA_R_16F, embed_dim,
                                    static_cast<const void*>(&beta), static_cast<void*>(output_weight_grads.data_ptr()),
                                    CUDA_R_16F, embed_dim, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // MatMul2 Dgrad1
  gemm_switch_fp32accum(a_layout_t, b_layout_n, k_seq_len, q_seq_len, head_dim, alpha,
                        static_cast<const half*>(v_lin_results_ptr), lead_dim_kv, batch_stride_kv,
                        static_cast<const half*>(output_lin_grads.data_ptr()), head_dim * attn_batches, head_dim, beta,
                        static_cast<half*>(matmul2_grads.data_ptr()), k_seq_len, k_seq_len * q_seq_len, attn_batches);

  // Matmul2 Dgrad2
  gemm_switch_fp32accum(a_layout_n, b_layout_t, head_dim, k_seq_len, q_seq_len, alpha,
                        static_cast<const half*>(output_lin_grads.data_ptr()), head_dim * attn_batches, head_dim,
                        static_cast<const half*>(dropout_results.data_ptr()), k_seq_len, k_seq_len * q_seq_len, beta,
                        v_lin_grads_ptr, lead_dim_kv, batch_stride_kv, attn_batches);

  // Apply Dropout Mask and Scale by Dropout Probability
  apex_masked_scale_cuda<at::Half, float, uint32_t>(
      static_cast<at::Half const*>(matmul2_grads.data_ptr()), static_cast<at::Half*>(matmul2_grads.data_ptr()),
      static_cast<uint8_t const*>(dropout_mask.data_ptr()), dropout_elems, (1.0 / (1.0 - dropout_prob)));

  // Softmax Grad
  bool softmax_success = false;
  softmax_success = dispatch_softmax_backward<half, half, float>(
      static_cast<half*>(matmul2_grads.data_ptr()), static_cast<half*>(matmul2_grads.data_ptr()),
      reinterpret_cast<half const*>(softmax_results.data_ptr()), k_seq_len, k_seq_len, attn_batches * q_seq_len);
  assert(softmax_success);

  // Matmul1 Dgrad1
  gemm_switch_fp32accum(a_layout_n, b_layout_n, head_dim, q_seq_len, k_seq_len, scale, k_lin_results_ptr, lead_dim_kv,
                        batch_stride_kv, static_cast<half*>(matmul2_grads.data_ptr()), k_seq_len, k_seq_len * q_seq_len,
                        beta, q_lin_grads_ptr, lead_dim_q, batch_stride_q, attn_batches);

  // Matmul1 Dgrad2
  gemm_switch_fp32accum(a_layout_n, b_layout_t, head_dim, k_seq_len, q_seq_len, scale, q_lin_results_ptr, lead_dim_q,
                        batch_stride_q, static_cast<half*>(matmul2_grads.data_ptr()), k_seq_len, k_seq_len * q_seq_len,
                        beta, k_lin_grads_ptr, lead_dim_kv, batch_stride_kv, attn_batches);

  // Input Linear Q Dgrad
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, embed_dim, batches_q, output_lin_q_dim, static_cast<const void*>(&alpha),
      static_cast<const void*>(input_weights_q.data_ptr()), CUDA_R_16F, embed_dim,
      static_cast<const void*>(q_lin_grads_ptr), CUDA_R_16F, output_lin_q_dim, static_cast<const void*>(&beta),
      // static_cast<void*>(input_q_grads.data_ptr()),
      static_cast<void*>(input_lin_q_grads.data_ptr()), CUDA_R_16F, embed_dim, CUDA_R_32F,
      // CUBLAS_GEMM_ALGO10_TENSOR_OP));
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Input Linear Q Wgrad
  TORCH_CUDABLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, embed_dim, output_lin_q_dim, batches_q,
                                    static_cast<const void*>(&alpha), static_cast<const void*>(inputs_q.data_ptr()),
                                    CUDA_R_16F, embed_dim, static_cast<const void*>(q_lin_grads_ptr), CUDA_R_16F,
                                    output_lin_q_dim, static_cast<const void*>(&beta),
                                    static_cast<void*>(input_weight_q_grads.data_ptr()), CUDA_R_16F, embed_dim,
                                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Input Linear KV Dgrad
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, embed_dim, batches_kv, output_lin_kv_dim, static_cast<const void*>(&alpha),
      static_cast<const void*>(input_weights_kv.data_ptr()), CUDA_R_16F, embed_dim,
      static_cast<const void*>(k_lin_grads_ptr), CUDA_R_16F, output_lin_kv_dim, static_cast<const void*>(&beta),
      static_cast<void*>(input_kv_grads.data_ptr()), CUDA_R_16F, embed_dim, CUDA_R_32F,
      // CUBLAS_GEMM_ALGO10_TENSOR_OP));
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Input Linear KV Wgrad
  TORCH_CUDABLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, embed_dim, output_lin_kv_dim, batches_kv,
                                    static_cast<const void*>(&alpha), static_cast<const void*>(inputs_kv.data_ptr()),
                                    CUDA_R_16F, embed_dim, static_cast<const void*>(k_lin_grads_ptr), CUDA_R_16F,
                                    output_lin_kv_dim, static_cast<const void*>(&beta),
                                    static_cast<void*>(input_weight_kv_grads.data_ptr()), CUDA_R_16F, embed_dim,
                                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // Fused Layer Norm Bwd with Residual Add
  HostLayerNormGradient<half, float>(
      static_cast<const half*>(input_lin_q_grads.data_ptr()), static_cast<half const*>(output_grads.data_ptr()),
      static_cast<const float*>(lyr_nrm_mean.data_ptr()), static_cast<const float*>(lyr_nrm_invvar.data_ptr()),
      inputs_q,
      static_cast<int>(batches_q),  // n1
      static_cast<int>(embed_dim),  // n2
      static_cast<const half*>(lyr_nrm_gamma_weights.data_ptr()),
      static_cast<const half*>(lyr_nrm_beta_weights.data_ptr()), 1.0e-5, static_cast<half*>(input_q_grads.data_ptr()),
      static_cast<half*>(lyr_nrm_gamma_grads.data_ptr()), static_cast<half*>(lyr_nrm_beta_grads.data_ptr()));

  TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

  return {input_q_grads,        input_kv_grads,        lyr_nrm_gamma_grads, lyr_nrm_beta_grads,
          input_weight_q_grads, input_weight_kv_grads, output_weight_grads};
}

}  // end namespace cublas_gemmex
}  // end namespace encdec_norm_add
}  // end namespace multihead_attn
