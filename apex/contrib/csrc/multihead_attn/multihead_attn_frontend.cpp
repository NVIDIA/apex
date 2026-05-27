#include <ATen/ATen.h>
#include <cuda_fp16.h>
#include <torch/library.h>

#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

namespace multihead_attn {
namespace fused_softmax {
namespace additive_mask_softmax_dropout {

std::vector<at::Tensor> fwd_cuda(bool is_training, int heads, at::Tensor const& input, const half* pad_mask,
                                 float dropout_prob);

at::Tensor bwd_cuda(int heads, at::Tensor const& output_grads, at::Tensor const& softmax_results,
                    at::Tensor const& dropout_mask, float dropout_prob);

std::vector<at::Tensor> fwd(bool use_mask, bool is_training, int heads, at::Tensor const& input,
                            at::Tensor const& pad_mask, float dropout_prob) {
  TORCH_CHECK(input.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  if (use_mask) {
    TORCH_CHECK(pad_mask.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(pad_mask.scalar_type() == at::ScalarType::Half, "Only BYTE is supported");
  }

  return fwd_cuda(is_training, heads, input, use_mask ? static_cast<const half*>(pad_mask.data_ptr()) : nullptr,
                  dropout_prob);
}

at::Tensor bwd(bool use_mask, int heads, at::Tensor const& output_grads, at::Tensor const& softmax_results,
               at::Tensor const& dropout_mask, float dropout_prob) {
  TORCH_CHECK(output_grads.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(softmax_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(dropout_mask.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(output_grads.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(softmax_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  //  TORCH_CHECK(dropout_mask.scalar_type()      == at::ScalarType::Byte,
  //  "Only BYTE is supported");

  return bwd_cuda(heads, output_grads, softmax_results, dropout_mask, dropout_prob);
}

}  // namespace additive_mask_softmax_dropout
namespace mask_softmax_dropout {

std::vector<at::Tensor> fwd_cuda(bool is_training, int heads, at::Tensor const& input, const uint8_t* pad_mask,
                                 float dropout_prob);

at::Tensor bwd_cuda(int heads, at::Tensor const& output_grads, at::Tensor const& softmax_results,
                    at::Tensor const& dropout_mask, const uint8_t* padding_mask, float dropout_prob);

std::vector<at::Tensor> fwd(bool use_mask, bool is_training, int heads, at::Tensor const& input,
                            at::Tensor const& pad_mask, float dropout_prob) {
  TORCH_CHECK(input.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half, "Only HALF is supported");

  if (use_mask) {
    TORCH_CHECK(pad_mask.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(pad_mask.scalar_type() == at::ScalarType::Byte, "Only BYTE is supported");
  }

  return fwd_cuda(is_training, heads, input, use_mask ? static_cast<const uint8_t*>(pad_mask.data_ptr()) : nullptr,
                  dropout_prob);
}

at::Tensor bwd(bool use_mask, int heads, at::Tensor const& output_grads, at::Tensor const& softmax_results,
               at::Tensor const& dropout_mask, at::Tensor const& padding_mask, float dropout_prob) {
  TORCH_CHECK(output_grads.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(softmax_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(dropout_mask.dim() == 3, "expected 3D tensor");

  TORCH_CHECK(output_grads.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(softmax_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  //  TORCH_CHECK(dropout_mask.scalar_type()      == at::ScalarType::Byte,
  //  "Only BYTE is supported");

  return bwd_cuda(heads, output_grads, softmax_results, dropout_mask,
                  use_mask ? static_cast<const uint8_t*>(padding_mask.data_ptr()) : nullptr, dropout_prob);
}

}  // end namespace mask_softmax_dropout
}  // end namespace fused_softmax

namespace encdec {
namespace cublas_gemmex {

std::vector<at::Tensor> fwd_cuda(bool use_time_mask, bool is_training, int heads, at::Tensor const& inputs_q,
                                 at::Tensor const& inputs_kv, at::Tensor const& input_weights_q,
                                 at::Tensor const& input_weights_kv, at::Tensor const& output_weights,
                                 const uint8_t* pad_mask, float dropout_prob);
std::vector<at::Tensor> bwd_cuda(int heads, at::Tensor const& output_grads, at::Tensor const& matmul2_results,
                                 at::Tensor const& dropout_results, at::Tensor const& softmax_results,
                                 at::Tensor const& input_lin_q_results, at::Tensor const& input_lin_kv_results,
                                 at::Tensor const& inputs_q, at::Tensor const& inputs_kv,
                                 at::Tensor const& input_weights_q, at::Tensor const& input_weights_kv,
                                 at::Tensor const& output_weights, at::Tensor const& dropout_mask, float dropout_prob);

std::vector<at::Tensor> fwd(bool use_mask, bool use_time_mask, bool is_training, int heads, at::Tensor const& inputs_q,
                            at::Tensor const& inputs_kv, at::Tensor const& input_weights_q,
                            at::Tensor const& input_weights_kv, at::Tensor const& output_weights,
                            at::Tensor const& pad_mask, float dropout_prob) {
  TORCH_CHECK(inputs_q.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(inputs_kv.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(input_weights_q.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(input_weights_kv.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(output_weights.dim() == 2, "expected 2D tensor");

  TORCH_CHECK(inputs_q.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(inputs_kv.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_weights_q.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_weights_kv.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(output_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");

  if (use_mask) {
    TORCH_CHECK(pad_mask.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(pad_mask.scalar_type() == at::ScalarType::Byte, "Only BYTE is supported");
  }

  return fwd_cuda(use_time_mask, is_training, heads, inputs_q, inputs_kv, input_weights_q, input_weights_kv,
                  output_weights, use_mask ? static_cast<const uint8_t*>(pad_mask.data_ptr()) : nullptr, dropout_prob);
}

std::vector<at::Tensor> bwd(int heads, at::Tensor const& output_grads, at::Tensor const& matmul2_results,
                            at::Tensor const& dropout_results, at::Tensor const& softmax_results,
                            at::Tensor const& input_lin_q_results, at::Tensor const& input_lin_kv_results,
                            at::Tensor const& inputs_q, at::Tensor const& inputs_kv, at::Tensor const& input_weights_q,
                            at::Tensor const& input_weights_kv, at::Tensor const& output_weights,
                            at::Tensor const& dropout_mask, float dropout_prob) {
  TORCH_CHECK(output_grads.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(matmul2_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(dropout_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(softmax_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(input_lin_q_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(input_lin_kv_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(inputs_q.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(inputs_kv.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(input_weights_q.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(input_weights_kv.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(output_weights.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(dropout_mask.dim() == 3, "expected 3D tensor");

  TORCH_CHECK(output_grads.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(matmul2_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(dropout_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(softmax_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_lin_q_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_lin_kv_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(inputs_q.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(inputs_kv.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_weights_q.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_weights_kv.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(output_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(dropout_mask.scalar_type() == at::ScalarType::Byte, "Only BYTE is supported");

  return bwd_cuda(heads, output_grads, matmul2_results, dropout_results, softmax_results, input_lin_q_results,
                  input_lin_kv_results, inputs_q, inputs_kv, input_weights_q, input_weights_kv, output_weights,
                  dropout_mask, dropout_prob);
}

}  // end namespace cublas_gemmex
}  // end namespace encdec

namespace encdec_norm_add {
namespace cublas_gemmex {

std::vector<at::Tensor> fwd_cuda(bool use_time_mask, bool is_training, int heads, at::Tensor const& inputs_q,
                                 at::Tensor const& inputs_kv, at::Tensor const& lyr_nrm_gamma_weights,
                                 at::Tensor const& lyr_nrm_beta_weights, at::Tensor const& input_weights_q,
                                 at::Tensor const& input_weights_kv, at::Tensor const& output_weights,
                                 const uint8_t* pad_mask, float dropout_prob);

std::vector<at::Tensor> bwd_cuda(int heads, at::Tensor const& output_grads, at::Tensor const& matmul2_results,
                                 at::Tensor const& dropout_results, at::Tensor const& softmax_results,
                                 at::Tensor const& input_lin_q_results, at::Tensor const& input_lin_kv_results,
                                 at::Tensor const& lyr_nrm_results, at::Tensor const& lyr_nrm_mean,
                                 at::Tensor const& lyr_nrm_invvar, at::Tensor const& inputs_q,
                                 at::Tensor const& inputs_kv, at::Tensor const& lyr_nrm_gamma_weights,
                                 at::Tensor const& lyr_nrm_beta_weights, at::Tensor const& input_weights_q,
                                 at::Tensor const& input_weights_kv, at::Tensor const& output_weights,
                                 at::Tensor const& dropout_mask, at::Tensor const& dropout_add_mask,
                                 float dropout_prob);

std::vector<at::Tensor> fwd(bool use_mask, bool use_time_mask, bool is_training, int heads, at::Tensor const& inputs_q,
                            at::Tensor const& inputs_kv, at::Tensor const& lyr_nrm_gamma_weights,
                            at::Tensor const& lyr_nrm_beta_weights, at::Tensor const& input_weights_q,
                            at::Tensor const& input_weights_kv, at::Tensor const& output_weights,
                            at::Tensor const& pad_mask, float dropout_prob) {
  TORCH_CHECK(inputs_q.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(inputs_kv.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(lyr_nrm_gamma_weights.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(lyr_nrm_beta_weights.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(input_weights_q.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(input_weights_kv.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(output_weights.dim() == 2, "expected 2D tensor");

  TORCH_CHECK(inputs_q.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(inputs_kv.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(lyr_nrm_gamma_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(lyr_nrm_beta_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_weights_q.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_weights_kv.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(output_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");

  if (use_mask) {
    TORCH_CHECK(pad_mask.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(pad_mask.scalar_type() == at::ScalarType::Byte, "Only BYTE is supported");
  }

  return fwd_cuda(use_time_mask, is_training, heads, inputs_q, inputs_kv, lyr_nrm_gamma_weights, lyr_nrm_beta_weights,
                  input_weights_q, input_weights_kv, output_weights,
                  use_mask ? static_cast<const uint8_t*>(pad_mask.data_ptr()) : nullptr, dropout_prob);
}

std::vector<at::Tensor> bwd(int heads, at::Tensor const& output_grads, at::Tensor const& matmul2_results,
                            at::Tensor const& dropout_results, at::Tensor const& softmax_results,
                            at::Tensor const& input_lin_q_results, at::Tensor const& input_lin_kv_results,
                            at::Tensor const& lyr_nrm_results, at::Tensor const& lyr_nrm_mean,
                            at::Tensor const& lyr_nrm_invvar, at::Tensor const& inputs_q, at::Tensor const& inputs_kv,
                            at::Tensor const& lyr_nrm_gamma_weights, at::Tensor const& lyr_nrm_beta_weights,
                            at::Tensor const& input_weights_q, at::Tensor const& input_weights_kv,
                            at::Tensor const& output_weights, at::Tensor const& dropout_mask,
                            at::Tensor const& dropout_add_mask, float dropout_prob) {
  TORCH_CHECK(output_grads.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(matmul2_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(dropout_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(softmax_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(input_lin_q_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(input_lin_kv_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(lyr_nrm_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(lyr_nrm_mean.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(lyr_nrm_invvar.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(inputs_q.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(inputs_kv.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(lyr_nrm_gamma_weights.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(lyr_nrm_beta_weights.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(input_weights_q.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(input_weights_kv.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(output_weights.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(dropout_mask.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(dropout_add_mask.dim() == 3, "expected 3D tensor");

  TORCH_CHECK(output_grads.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(matmul2_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(dropout_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(softmax_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_lin_q_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_lin_kv_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(lyr_nrm_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(lyr_nrm_mean.scalar_type() == at::ScalarType::Float, "Only FLOAT is supported");
  TORCH_CHECK(lyr_nrm_invvar.scalar_type() == at::ScalarType::Float, "Only FLOAT is supported");
  TORCH_CHECK(inputs_q.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(inputs_kv.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(lyr_nrm_gamma_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(lyr_nrm_beta_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_weights_q.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_weights_kv.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(output_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(dropout_mask.scalar_type() == at::ScalarType::Byte, "Only BYTE is supported");
  TORCH_CHECK(dropout_add_mask.scalar_type() == at::ScalarType::Byte, "Only BYTE is supported");

  return bwd_cuda(heads, output_grads, matmul2_results, dropout_results, softmax_results, input_lin_q_results,
                  input_lin_kv_results, lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, inputs_q, inputs_kv,
                  lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights_q, input_weights_kv, output_weights,
                  dropout_mask, dropout_add_mask, dropout_prob);
}

}  // end namespace cublas_gemmex
}  // end namespace encdec_norm_add

namespace self {
namespace cublas_gemmex {

std::vector<at::Tensor> fwd_cuda(bool use_time_mask, bool is_training, int heads, at::Tensor const& inputs,
                                 at::Tensor const& input_weights, at::Tensor const& output_weights,
                                 const uint8_t* pad_mask, float dropout_prob);

std::vector<at::Tensor> bwd_cuda(int heads, at::Tensor const& output_grads, at::Tensor const& matmul2_results,
                                 at::Tensor const& dropout_results, at::Tensor const& softmax_results,
                                 at::Tensor const& input_lin_results, at::Tensor const& inputs,
                                 at::Tensor const& input_weights, at::Tensor const& output_weights,
                                 at::Tensor const& dropout_mask, float dropout_prob);

std::vector<at::Tensor> fwd(bool use_mask, bool use_time_mask, bool is_training, int heads, at::Tensor const& inputs,
                            at::Tensor const& input_weights, at::Tensor const& output_weights,
                            at::Tensor const& pad_mask, float dropout_prob) {
  TORCH_CHECK(inputs.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(input_weights.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(output_weights.dim() == 2, "expected 2D tensor");

  TORCH_CHECK(inputs.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(output_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");

  if (use_mask) {
    TORCH_CHECK(pad_mask.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(pad_mask.scalar_type() == at::ScalarType::Byte, "Only BYTE is supported");
  }

  return fwd_cuda(use_time_mask, is_training, heads, inputs, input_weights, output_weights,
                  use_mask ? static_cast<const uint8_t*>(pad_mask.data_ptr()) : nullptr, dropout_prob);
}

std::vector<at::Tensor> bwd(int heads, at::Tensor const& output_grads, at::Tensor const& matmul2_results,
                            at::Tensor const& dropout_results, at::Tensor const& softmax_results,
                            at::Tensor const& input_lin_results, at::Tensor const& inputs,
                            at::Tensor const& input_weights, at::Tensor const& output_weights,
                            at::Tensor const& dropout_mask, float dropout_prob) {
  TORCH_CHECK(output_grads.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(matmul2_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(dropout_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(softmax_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(input_lin_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(inputs.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(input_weights.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(output_weights.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(dropout_mask.dim() == 3, "expected 3D tensor");

  TORCH_CHECK(output_grads.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(matmul2_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(dropout_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(softmax_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_lin_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(inputs.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(output_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(dropout_mask.scalar_type() == at::ScalarType::Byte, "Only BYTE is supported");

  return bwd_cuda(heads, output_grads, matmul2_results, dropout_results, softmax_results, input_lin_results, inputs,
                  input_weights, output_weights, dropout_mask, dropout_prob);
}

}  // end namespace cublas_gemmex
}  // end namespace self
namespace self_bias {
namespace cublas_gemmex {

std::vector<at::Tensor> fwd_cuda(bool use_time_mask, bool is_training, int heads, at::Tensor const& inputs,
                                 at::Tensor const& input_weights, at::Tensor const& output_weights,
                                 at::Tensor const& input_biases, at::Tensor const& output_biases,
                                 const uint8_t* pad_mask, float dropout_prob);

std::vector<at::Tensor> bwd_cuda(int heads, at::Tensor const& output_grads, at::Tensor const& matmul2_results,
                                 at::Tensor const& dropout_results, at::Tensor const& softmax_results,
                                 at::Tensor const& input_lin_results, at::Tensor const& inputs,
                                 at::Tensor const& input_weights, at::Tensor const& output_weights,
                                 // at::Tensor const& input_biases,
                                 // at::Tensor const& output_biases,
                                 at::Tensor const& dropout_mask, float dropout_prob);

std::vector<at::Tensor> fwd(bool use_mask, bool use_time_mask, bool is_training, int heads, at::Tensor const& inputs,
                            at::Tensor const& input_weights, at::Tensor const& output_weights,
                            at::Tensor const& input_biases, at::Tensor const& output_biases, at::Tensor const& pad_mask,
                            float dropout_prob) {
  TORCH_CHECK(inputs.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(input_weights.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(output_weights.dim() == 2, "expected 2D tensor");

  TORCH_CHECK(inputs.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(output_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");

  if (use_mask) {
    TORCH_CHECK(pad_mask.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(pad_mask.scalar_type() == at::ScalarType::Byte, "Only BYTE is supported");
  }

  return fwd_cuda(use_time_mask, is_training, heads, inputs, input_weights, output_weights, input_biases, output_biases,
                  use_mask ? static_cast<const uint8_t*>(pad_mask.data_ptr()) : nullptr, dropout_prob);
}

std::vector<at::Tensor> bwd(int heads, at::Tensor const& output_grads, at::Tensor const& matmul2_results,
                            at::Tensor const& dropout_results, at::Tensor const& softmax_results,
                            at::Tensor const& input_lin_results, at::Tensor const& inputs,
                            at::Tensor const& input_weights, at::Tensor const& output_weights,
                            at::Tensor const& dropout_mask, float dropout_prob) {
  TORCH_CHECK(output_grads.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(matmul2_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(dropout_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(softmax_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(input_lin_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(inputs.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(input_weights.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(output_weights.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(dropout_mask.dim() == 3, "expected 3D tensor");

  TORCH_CHECK(output_grads.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(matmul2_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(dropout_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(softmax_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_lin_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(inputs.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(output_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(dropout_mask.scalar_type() == at::ScalarType::Byte, "Only BYTE is supported");

  return bwd_cuda(heads, output_grads, matmul2_results, dropout_results, softmax_results, input_lin_results, inputs,
                  input_weights, output_weights, dropout_mask, dropout_prob);
}

}  // end namespace cublas_gemmex
}  // namespace self_bias
namespace self_bias_additive_mask {
namespace cublas_gemmex {

std::vector<at::Tensor> fwd_cuda(bool use_time_mask, bool is_training, int heads, at::Tensor const& inputs,
                                 at::Tensor const& input_weights, at::Tensor const& output_weights,
                                 at::Tensor const& input_biases, at::Tensor const& output_biases, const half* pad_mask,
                                 float dropout_prob);

std::vector<at::Tensor> bwd_cuda(int heads, at::Tensor const& output_grads, at::Tensor const& matmul2_results,
                                 at::Tensor const& dropout_results,
                                 // at::Tensor const& softmax_results,
                                 at::Tensor const& bmm1_results, at::Tensor const& pad_mask,
                                 at::Tensor const& input_lin_results, at::Tensor const& inputs,
                                 at::Tensor const& input_weights, at::Tensor const& output_weights,
                                 // at::Tensor const& input_biases,
                                 // at::Tensor const& output_biases,
                                 at::Tensor const& dropout_mask, float dropout_prob);

std::vector<at::Tensor> fwd(bool use_mask, bool use_time_mask, bool is_training, int heads, at::Tensor const& inputs,
                            at::Tensor const& input_weights, at::Tensor const& output_weights,
                            at::Tensor const& input_biases, at::Tensor const& output_biases, at::Tensor const& pad_mask,
                            float dropout_prob) {
  TORCH_CHECK(inputs.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(input_weights.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(output_weights.dim() == 2, "expected 2D tensor");

  TORCH_CHECK(inputs.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(output_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(use_mask, "no mask is not supported");

  if (use_mask) {
    TORCH_CHECK(pad_mask.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(pad_mask.scalar_type() == at::ScalarType::Half, "Only Half is supported");
  }

  return fwd_cuda(use_time_mask, is_training, heads, inputs, input_weights, output_weights, input_biases, output_biases,
                  use_mask ? static_cast<const half*>(pad_mask.data_ptr()) : nullptr, dropout_prob);
}

std::vector<at::Tensor> bwd(int heads, at::Tensor const& output_grads, at::Tensor const& matmul2_results,
                            at::Tensor const& dropout_results, at::Tensor const& bmm1_results,
                            at::Tensor const& pad_mask, at::Tensor const& input_lin_results, at::Tensor const& inputs,
                            at::Tensor const& input_weights, at::Tensor const& output_weights,
                            at::Tensor const& dropout_mask, float dropout_prob) {
  TORCH_CHECK(output_grads.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(matmul2_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(dropout_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(input_lin_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(inputs.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(input_weights.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(output_weights.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(dropout_mask.dim() == 3, "expected 3D tensor");

  TORCH_CHECK(output_grads.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(matmul2_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(dropout_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_lin_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(inputs.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(output_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(dropout_mask.scalar_type() == at::ScalarType::Byte, "Only BYTE is supported");

  return bwd_cuda(heads, output_grads, matmul2_results, dropout_results, bmm1_results, pad_mask, input_lin_results,
                  inputs, input_weights, output_weights, dropout_mask, dropout_prob);
}

}  // end namespace cublas_gemmex
}  // namespace self_bias_additive_mask

namespace self_norm_add {
namespace cublas_gemmex {

std::vector<at::Tensor> fwd_cuda(bool use_time_mask, bool is_training, int heads, at::Tensor const& inputs,
                                 at::Tensor const& lyr_nrm_gamma_weights, at::Tensor const& lyr_nrm_beta_weights,
                                 at::Tensor const& input_weights, at::Tensor const& output_weights,
                                 const uint8_t* pad_mask, float dropout_prob);

std::vector<at::Tensor> bwd_cuda(int heads, at::Tensor const& output_grads, at::Tensor const& matmul2_results,
                                 at::Tensor const& dropout_results, at::Tensor const& softmax_results,
                                 at::Tensor const& input_lin_results, at::Tensor const& lyr_nrm_results,
                                 at::Tensor const& lyr_nrm_mean, at::Tensor const& lyr_nrm_invvar,
                                 at::Tensor const& inputs, at::Tensor const& lyr_nrm_gamma_weights,
                                 at::Tensor const& lyr_nrm_beta_weights, at::Tensor const& input_weights,
                                 at::Tensor const& output_weights, at::Tensor const& dropout_mask,
                                 at::Tensor const& dropout_add_mask, float dropout_prob);

std::vector<at::Tensor> fwd(bool use_mask, bool use_time_mask, bool is_training, int heads, at::Tensor const& inputs,
                            at::Tensor const& lyr_nrm_gamma_weights, at::Tensor const& lyr_nrm_beta_weights,
                            at::Tensor const& input_weights, at::Tensor const& output_weights,
                            at::Tensor const& pad_mask, float dropout_prob) {
  TORCH_CHECK(inputs.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(lyr_nrm_gamma_weights.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(lyr_nrm_beta_weights.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(input_weights.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(output_weights.dim() == 2, "expected 2D tensor");

  TORCH_CHECK(inputs.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(lyr_nrm_gamma_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(lyr_nrm_beta_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(output_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");

  if (use_mask) {
    TORCH_CHECK(pad_mask.dim() == 2, "expected 2D tensor");
    TORCH_CHECK(pad_mask.scalar_type() == at::ScalarType::Byte, "Only BYTE is supported");
  }

  return fwd_cuda(use_time_mask, is_training, heads, inputs, lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights,
                  output_weights, use_mask ? static_cast<const uint8_t*>(pad_mask.data_ptr()) : nullptr, dropout_prob);
}

std::vector<at::Tensor> bwd(int heads, at::Tensor const& output_grads, at::Tensor const& matmul2_results,
                            at::Tensor const& dropout_results, at::Tensor const& softmax_results,
                            at::Tensor const& input_lin_results, at::Tensor const& lyr_nrm_results,
                            at::Tensor const& lyr_nrm_mean, at::Tensor const& lyr_nrm_invvar, at::Tensor const& inputs,
                            at::Tensor const& lyr_nrm_gamma_weights, at::Tensor const& lyr_nrm_beta_weights,
                            at::Tensor const& input_weights, at::Tensor const& output_weights,
                            at::Tensor const& dropout_mask, at::Tensor const& dropout_add_mask, float dropout_prob) {
  TORCH_CHECK(output_grads.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(matmul2_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(dropout_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(softmax_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(input_lin_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(lyr_nrm_results.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(lyr_nrm_mean.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(lyr_nrm_invvar.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(inputs.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(lyr_nrm_gamma_weights.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(lyr_nrm_beta_weights.dim() == 1, "expected 1D tensor");
  TORCH_CHECK(input_weights.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(output_weights.dim() == 2, "expected 2D tensor");
  TORCH_CHECK(dropout_mask.dim() == 3, "expected 3D tensor");
  TORCH_CHECK(dropout_add_mask.dim() == 3, "expected 3D tensor");

  TORCH_CHECK(output_grads.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(matmul2_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(dropout_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(softmax_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_lin_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(lyr_nrm_results.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(lyr_nrm_mean.scalar_type() == at::ScalarType::Float, "Only FLOAT is supported");
  TORCH_CHECK(lyr_nrm_invvar.scalar_type() == at::ScalarType::Float, "Only FLOAT is supported");
  TORCH_CHECK(inputs.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(lyr_nrm_gamma_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(lyr_nrm_beta_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(input_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(output_weights.scalar_type() == at::ScalarType::Half, "Only HALF is supported");
  TORCH_CHECK(dropout_mask.scalar_type() == at::ScalarType::Byte, "Only BYTE is supported");
  TORCH_CHECK(dropout_add_mask.scalar_type() == at::ScalarType::Byte, "Only BYTE is supported");

  return bwd_cuda(heads, output_grads, matmul2_results, dropout_results, softmax_results, input_lin_results,
                  lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, inputs, lyr_nrm_gamma_weights, lyr_nrm_beta_weights,
                  input_weights, output_weights, dropout_mask, dropout_add_mask, dropout_prob);
}

}  // end namespace cublas_gemmex
}  // end namespace self_norm_add
}  // end namespace multihead_attn

namespace {
int as_int(int64_t value) { return static_cast<int>(value); }

float as_float(double value) { return static_cast<float>(value); }

std::vector<at::Tensor> apex_fast_multihead_attn_additive_mask_softmax_dropout_forward(bool use_mask, bool is_training,
                                                                                       int64_t heads,
                                                                                       at::Tensor const& input,
                                                                                       at::Tensor const& pad_mask,
                                                                                       double dropout_prob) {
  return multihead_attn::fused_softmax::additive_mask_softmax_dropout::fwd(use_mask, is_training, as_int(heads), input,
                                                                           pad_mask, as_float(dropout_prob));
}

at::Tensor apex_fast_multihead_attn_additive_mask_softmax_dropout_backward(bool use_mask, int64_t heads,
                                                                           at::Tensor const& output_grads,
                                                                           at::Tensor const& softmax_results,
                                                                           at::Tensor const& dropout_mask,
                                                                           double dropout_prob) {
  return multihead_attn::fused_softmax::additive_mask_softmax_dropout::bwd(
      use_mask, as_int(heads), output_grads, softmax_results, dropout_mask, as_float(dropout_prob));
}

std::vector<at::Tensor> apex_fast_multihead_attn_mask_softmax_dropout_forward(bool use_mask, bool is_training,
                                                                              int64_t heads, at::Tensor const& input,
                                                                              at::Tensor const& pad_mask,
                                                                              double dropout_prob) {
  return multihead_attn::fused_softmax::mask_softmax_dropout::fwd(use_mask, is_training, as_int(heads), input, pad_mask,
                                                                  as_float(dropout_prob));
}

at::Tensor apex_fast_multihead_attn_mask_softmax_dropout_backward(bool use_mask, int64_t heads,
                                                                  at::Tensor const& output_grads,
                                                                  at::Tensor const& softmax_results,
                                                                  at::Tensor const& dropout_mask,
                                                                  at::Tensor const& padding_mask, double dropout_prob) {
  return multihead_attn::fused_softmax::mask_softmax_dropout::bwd(
      use_mask, as_int(heads), output_grads, softmax_results, dropout_mask, padding_mask, as_float(dropout_prob));
}

std::vector<at::Tensor> apex_fast_multihead_attn_encdec_multihead_attn_forward(
    bool use_mask, bool use_time_mask, bool is_training, int64_t heads, at::Tensor const& inputs_q,
    at::Tensor const& inputs_kv, at::Tensor const& input_weights_q, at::Tensor const& input_weights_kv,
    at::Tensor const& output_weights, at::Tensor const& pad_mask, double dropout_prob) {
  return multihead_attn::encdec::cublas_gemmex::fwd(use_mask, use_time_mask, is_training, as_int(heads), inputs_q,
                                                    inputs_kv, input_weights_q, input_weights_kv, output_weights,
                                                    pad_mask, as_float(dropout_prob));
}

std::vector<at::Tensor> apex_fast_multihead_attn_encdec_multihead_attn_backward(
    int64_t heads, at::Tensor const& output_grads, at::Tensor const& matmul2_results, at::Tensor const& dropout_results,
    at::Tensor const& softmax_results, at::Tensor const& input_lin_q_results, at::Tensor const& input_lin_kv_results,
    at::Tensor const& inputs_q, at::Tensor const& inputs_kv, at::Tensor const& input_weights_q,
    at::Tensor const& input_weights_kv, at::Tensor const& output_weights, at::Tensor const& dropout_mask,
    double dropout_prob) {
  return multihead_attn::encdec::cublas_gemmex::bwd(as_int(heads), output_grads, matmul2_results, dropout_results,
                                                    softmax_results, input_lin_q_results, input_lin_kv_results,
                                                    inputs_q, inputs_kv, input_weights_q, input_weights_kv,
                                                    output_weights, dropout_mask, as_float(dropout_prob));
}

std::vector<at::Tensor> apex_fast_multihead_attn_encdec_multihead_attn_norm_add_forward(
    bool use_mask, bool use_time_mask, bool is_training, int64_t heads, at::Tensor const& inputs_q,
    at::Tensor const& inputs_kv, at::Tensor const& lyr_nrm_gamma_weights, at::Tensor const& lyr_nrm_beta_weights,
    at::Tensor const& input_weights_q, at::Tensor const& input_weights_kv, at::Tensor const& output_weights,
    at::Tensor const& pad_mask, double dropout_prob) {
  return multihead_attn::encdec_norm_add::cublas_gemmex::fwd(
      use_mask, use_time_mask, is_training, as_int(heads), inputs_q, inputs_kv, lyr_nrm_gamma_weights,
      lyr_nrm_beta_weights, input_weights_q, input_weights_kv, output_weights, pad_mask, as_float(dropout_prob));
}

std::vector<at::Tensor> apex_fast_multihead_attn_encdec_multihead_attn_norm_add_backward(
    int64_t heads, at::Tensor const& output_grads, at::Tensor const& matmul2_results, at::Tensor const& dropout_results,
    at::Tensor const& softmax_results, at::Tensor const& input_lin_q_results, at::Tensor const& input_lin_kv_results,
    at::Tensor const& lyr_nrm_results, at::Tensor const& lyr_nrm_mean, at::Tensor const& lyr_nrm_invvar,
    at::Tensor const& inputs_q, at::Tensor const& inputs_kv, at::Tensor const& lyr_nrm_gamma_weights,
    at::Tensor const& lyr_nrm_beta_weights, at::Tensor const& input_weights_q, at::Tensor const& input_weights_kv,
    at::Tensor const& output_weights, at::Tensor const& dropout_mask, at::Tensor const& dropout_add_mask,
    double dropout_prob) {
  return multihead_attn::encdec_norm_add::cublas_gemmex::bwd(
      as_int(heads), output_grads, matmul2_results, dropout_results, softmax_results, input_lin_q_results,
      input_lin_kv_results, lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, inputs_q, inputs_kv, lyr_nrm_gamma_weights,
      lyr_nrm_beta_weights, input_weights_q, input_weights_kv, output_weights, dropout_mask, dropout_add_mask,
      as_float(dropout_prob));
}

std::vector<at::Tensor> apex_fast_multihead_attn_self_attn_forward(bool use_mask, bool use_time_mask, bool is_training,
                                                                   int64_t heads, at::Tensor const& inputs,
                                                                   at::Tensor const& input_weights,
                                                                   at::Tensor const& output_weights,
                                                                   at::Tensor const& pad_mask, double dropout_prob) {
  return multihead_attn::self::cublas_gemmex::fwd(use_mask, use_time_mask, is_training, as_int(heads), inputs,
                                                  input_weights, output_weights, pad_mask, as_float(dropout_prob));
}

std::vector<at::Tensor> apex_fast_multihead_attn_self_attn_backward(
    int64_t heads, at::Tensor const& output_grads, at::Tensor const& matmul2_results, at::Tensor const& dropout_results,
    at::Tensor const& softmax_results, at::Tensor const& input_lin_results, at::Tensor const& inputs,
    at::Tensor const& input_weights, at::Tensor const& output_weights, at::Tensor const& dropout_mask,
    double dropout_prob) {
  return multihead_attn::self::cublas_gemmex::bwd(as_int(heads), output_grads, matmul2_results, dropout_results,
                                                  softmax_results, input_lin_results, inputs, input_weights,
                                                  output_weights, dropout_mask, as_float(dropout_prob));
}

std::vector<at::Tensor> apex_fast_multihead_attn_self_attn_bias_forward(
    bool use_mask, bool use_time_mask, bool is_training, int64_t heads, at::Tensor const& inputs,
    at::Tensor const& input_weights, at::Tensor const& output_weights, at::Tensor const& input_biases,
    at::Tensor const& output_biases, at::Tensor const& pad_mask, double dropout_prob) {
  return multihead_attn::self_bias::cublas_gemmex::fwd(use_mask, use_time_mask, is_training, as_int(heads), inputs,
                                                       input_weights, output_weights, input_biases, output_biases,
                                                       pad_mask, as_float(dropout_prob));
}

std::vector<at::Tensor> apex_fast_multihead_attn_self_attn_bias_backward(
    int64_t heads, at::Tensor const& output_grads, at::Tensor const& matmul2_results, at::Tensor const& dropout_results,
    at::Tensor const& softmax_results, at::Tensor const& input_lin_results, at::Tensor const& inputs,
    at::Tensor const& input_weights, at::Tensor const& output_weights, at::Tensor const& dropout_mask,
    double dropout_prob) {
  return multihead_attn::self_bias::cublas_gemmex::bwd(as_int(heads), output_grads, matmul2_results, dropout_results,
                                                       softmax_results, input_lin_results, inputs, input_weights,
                                                       output_weights, dropout_mask, as_float(dropout_prob));
}

std::vector<at::Tensor> apex_fast_multihead_attn_self_attn_bias_additive_mask_forward(
    bool use_mask, bool use_time_mask, bool is_training, int64_t heads, at::Tensor const& inputs,
    at::Tensor const& input_weights, at::Tensor const& output_weights, at::Tensor const& input_biases,
    at::Tensor const& output_biases, at::Tensor const& pad_mask, double dropout_prob) {
  return multihead_attn::self_bias_additive_mask::cublas_gemmex::fwd(
      use_mask, use_time_mask, is_training, as_int(heads), inputs, input_weights, output_weights, input_biases,
      output_biases, pad_mask, as_float(dropout_prob));
}

std::vector<at::Tensor> apex_fast_multihead_attn_self_attn_bias_additive_mask_backward(
    int64_t heads, at::Tensor const& output_grads, at::Tensor const& matmul2_results, at::Tensor const& dropout_results,
    at::Tensor const& bmm1_results, at::Tensor const& pad_mask, at::Tensor const& input_lin_results,
    at::Tensor const& inputs, at::Tensor const& input_weights, at::Tensor const& output_weights,
    at::Tensor const& dropout_mask, double dropout_prob) {
  return multihead_attn::self_bias_additive_mask::cublas_gemmex::bwd(
      as_int(heads), output_grads, matmul2_results, dropout_results, bmm1_results, pad_mask, input_lin_results, inputs,
      input_weights, output_weights, dropout_mask, as_float(dropout_prob));
}

std::vector<at::Tensor> apex_fast_multihead_attn_self_attn_norm_add_forward(
    bool use_mask, bool use_time_mask, bool is_training, int64_t heads, at::Tensor const& inputs,
    at::Tensor const& lyr_nrm_gamma_weights, at::Tensor const& lyr_nrm_beta_weights, at::Tensor const& input_weights,
    at::Tensor const& output_weights, at::Tensor const& pad_mask, double dropout_prob) {
  return multihead_attn::self_norm_add::cublas_gemmex::fwd(use_mask, use_time_mask, is_training, as_int(heads), inputs,
                                                           lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights,
                                                           output_weights, pad_mask, as_float(dropout_prob));
}

std::vector<at::Tensor> apex_fast_multihead_attn_self_attn_norm_add_backward(
    int64_t heads, at::Tensor const& output_grads, at::Tensor const& matmul2_results, at::Tensor const& dropout_results,
    at::Tensor const& softmax_results, at::Tensor const& input_lin_results, at::Tensor const& lyr_nrm_results,
    at::Tensor const& lyr_nrm_mean, at::Tensor const& lyr_nrm_invvar, at::Tensor const& inputs,
    at::Tensor const& lyr_nrm_gamma_weights, at::Tensor const& lyr_nrm_beta_weights, at::Tensor const& input_weights,
    at::Tensor const& output_weights, at::Tensor const& dropout_mask, at::Tensor const& dropout_add_mask,
    double dropout_prob) {
  return multihead_attn::self_norm_add::cublas_gemmex::bwd(
      as_int(heads), output_grads, matmul2_results, dropout_results, softmax_results, input_lin_results,
      lyr_nrm_results, lyr_nrm_mean, lyr_nrm_invvar, inputs, lyr_nrm_gamma_weights, lyr_nrm_beta_weights, input_weights,
      output_weights, dropout_mask, dropout_add_mask, as_float(dropout_prob));
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(apex, m) {
  m.def(
      "fast_multihead_attn_additive_mask_softmax_dropout_forward(bool use_mask, bool is_training, int heads, "
      "Tensor input, Tensor pad_mask, float dropout_prob) -> Tensor[]");
  m.def(
      "fast_multihead_attn_additive_mask_softmax_dropout_backward(bool use_mask, int heads, Tensor output_grads, "
      "Tensor softmax_results, Tensor dropout_mask, float dropout_prob) -> Tensor");
  m.def(
      "fast_multihead_attn_mask_softmax_dropout_forward(bool use_mask, bool is_training, int heads, Tensor input, "
      "Tensor pad_mask, float dropout_prob) -> Tensor[]");
  m.def(
      "fast_multihead_attn_mask_softmax_dropout_backward(bool use_mask, int heads, Tensor output_grads, "
      "Tensor softmax_results, Tensor dropout_mask, Tensor padding_mask, float dropout_prob) -> Tensor");
  m.def(
      "fast_multihead_attn_encdec_multihead_attn_forward(bool use_mask, bool use_time_mask, bool is_training, "
      "int heads, Tensor inputs_q, Tensor inputs_kv, Tensor input_weights_q, Tensor input_weights_kv, "
      "Tensor output_weights, Tensor pad_mask, float dropout_prob) -> Tensor[]");
  m.def(
      "fast_multihead_attn_encdec_multihead_attn_backward(int heads, Tensor output_grads, Tensor matmul2_results, "
      "Tensor dropout_results, Tensor softmax_results, Tensor input_lin_q_results, Tensor input_lin_kv_results, "
      "Tensor inputs_q, Tensor inputs_kv, Tensor input_weights_q, Tensor input_weights_kv, Tensor output_weights, "
      "Tensor dropout_mask, float dropout_prob) -> Tensor[]");
  m.def(
      "fast_multihead_attn_encdec_multihead_attn_norm_add_forward(bool use_mask, bool use_time_mask, "
      "bool is_training, int heads, Tensor inputs_q, Tensor inputs_kv, Tensor lyr_nrm_gamma_weights, "
      "Tensor lyr_nrm_beta_weights, Tensor input_weights_q, Tensor input_weights_kv, Tensor output_weights, "
      "Tensor pad_mask, float dropout_prob) -> Tensor[]");
  m.def(
      "fast_multihead_attn_encdec_multihead_attn_norm_add_backward(int heads, Tensor output_grads, "
      "Tensor matmul2_results, Tensor dropout_results, Tensor softmax_results, Tensor input_lin_q_results, "
      "Tensor input_lin_kv_results, Tensor lyr_nrm_results, Tensor lyr_nrm_mean, Tensor lyr_nrm_invvar, "
      "Tensor inputs_q, Tensor inputs_kv, Tensor lyr_nrm_gamma_weights, Tensor lyr_nrm_beta_weights, "
      "Tensor input_weights_q, Tensor input_weights_kv, Tensor output_weights, Tensor dropout_mask, "
      "Tensor dropout_add_mask, float dropout_prob) -> Tensor[]");
  m.def(
      "fast_multihead_attn_self_attn_forward(bool use_mask, bool use_time_mask, bool is_training, int heads, "
      "Tensor inputs, Tensor input_weights, Tensor output_weights, Tensor pad_mask, float dropout_prob) "
      "-> Tensor[]");
  m.def(
      "fast_multihead_attn_self_attn_backward(int heads, Tensor output_grads, Tensor matmul2_results, "
      "Tensor dropout_results, Tensor softmax_results, Tensor input_lin_results, Tensor inputs, "
      "Tensor input_weights, Tensor output_weights, Tensor dropout_mask, float dropout_prob) -> Tensor[]");
  m.def(
      "fast_multihead_attn_self_attn_bias_forward(bool use_mask, bool use_time_mask, bool is_training, int heads, "
      "Tensor inputs, Tensor input_weights, Tensor output_weights, Tensor input_biases, Tensor output_biases, "
      "Tensor pad_mask, float dropout_prob) -> Tensor[]");
  m.def(
      "fast_multihead_attn_self_attn_bias_backward(int heads, Tensor output_grads, Tensor matmul2_results, "
      "Tensor dropout_results, Tensor softmax_results, Tensor input_lin_results, Tensor inputs, "
      "Tensor input_weights, Tensor output_weights, Tensor dropout_mask, float dropout_prob) -> Tensor[]");
  m.def(
      "fast_multihead_attn_self_attn_bias_additive_mask_forward(bool use_mask, bool use_time_mask, "
      "bool is_training, int heads, Tensor inputs, Tensor input_weights, Tensor output_weights, "
      "Tensor input_biases, Tensor output_biases, Tensor pad_mask, float dropout_prob) -> Tensor[]");
  m.def(
      "fast_multihead_attn_self_attn_bias_additive_mask_backward(int heads, Tensor output_grads, "
      "Tensor matmul2_results, Tensor dropout_results, Tensor bmm1_results, Tensor pad_mask, "
      "Tensor input_lin_results, Tensor inputs, Tensor input_weights, Tensor output_weights, Tensor dropout_mask, "
      "float dropout_prob) -> Tensor[]");
  m.def(
      "fast_multihead_attn_self_attn_norm_add_forward(bool use_mask, bool use_time_mask, bool is_training, "
      "int heads, Tensor inputs, Tensor lyr_nrm_gamma_weights, Tensor lyr_nrm_beta_weights, "
      "Tensor input_weights, Tensor output_weights, Tensor pad_mask, float dropout_prob) -> Tensor[]");
  m.def(
      "fast_multihead_attn_self_attn_norm_add_backward(int heads, Tensor output_grads, Tensor matmul2_results, "
      "Tensor dropout_results, Tensor softmax_results, Tensor input_lin_results, Tensor lyr_nrm_results, "
      "Tensor lyr_nrm_mean, Tensor lyr_nrm_invvar, Tensor inputs, Tensor lyr_nrm_gamma_weights, "
      "Tensor lyr_nrm_beta_weights, Tensor input_weights, Tensor output_weights, Tensor dropout_mask, "
      "Tensor dropout_add_mask, float dropout_prob) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(apex, CUDA, m) {
  m.impl("fast_multihead_attn_additive_mask_softmax_dropout_forward",
         &apex_fast_multihead_attn_additive_mask_softmax_dropout_forward);
  m.impl("fast_multihead_attn_additive_mask_softmax_dropout_backward",
         &apex_fast_multihead_attn_additive_mask_softmax_dropout_backward);
  m.impl("fast_multihead_attn_mask_softmax_dropout_forward", &apex_fast_multihead_attn_mask_softmax_dropout_forward);
  m.impl("fast_multihead_attn_mask_softmax_dropout_backward", &apex_fast_multihead_attn_mask_softmax_dropout_backward);
  m.impl("fast_multihead_attn_encdec_multihead_attn_forward", &apex_fast_multihead_attn_encdec_multihead_attn_forward);
  m.impl("fast_multihead_attn_encdec_multihead_attn_backward",
         &apex_fast_multihead_attn_encdec_multihead_attn_backward);
  m.impl("fast_multihead_attn_encdec_multihead_attn_norm_add_forward",
         &apex_fast_multihead_attn_encdec_multihead_attn_norm_add_forward);
  m.impl("fast_multihead_attn_encdec_multihead_attn_norm_add_backward",
         &apex_fast_multihead_attn_encdec_multihead_attn_norm_add_backward);
  m.impl("fast_multihead_attn_self_attn_forward", &apex_fast_multihead_attn_self_attn_forward);
  m.impl("fast_multihead_attn_self_attn_backward", &apex_fast_multihead_attn_self_attn_backward);
  m.impl("fast_multihead_attn_self_attn_bias_forward", &apex_fast_multihead_attn_self_attn_bias_forward);
  m.impl("fast_multihead_attn_self_attn_bias_backward", &apex_fast_multihead_attn_self_attn_bias_backward);
  m.impl("fast_multihead_attn_self_attn_bias_additive_mask_forward",
         &apex_fast_multihead_attn_self_attn_bias_additive_mask_forward);
  m.impl("fast_multihead_attn_self_attn_bias_additive_mask_backward",
         &apex_fast_multihead_attn_self_attn_bias_additive_mask_backward);
  m.impl("fast_multihead_attn_self_attn_norm_add_forward", &apex_fast_multihead_attn_self_attn_norm_add_forward);
  m.impl("fast_multihead_attn_self_attn_norm_add_backward", &apex_fast_multihead_attn_self_attn_norm_add_backward);
}

#undef CHECK_CUDA
#undef CHECK_CONTIGUOUS
#undef CHECK_INPUT
