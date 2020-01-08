#include <torch/extension.h>
#include <vector>

namespace multihead_attn {
namespace encdec {
namespace cublaslt_matmul {

void fwd_init_cuda(int heads, int embed_dim);
void bwd_init_cuda(int heads, int embed_dim);
void fwd_del_cuda();
void bwd_del_cuda();

std::vector<torch::Tensor> fwd_cuda(
                               bool                 use_time_mask,  
                               bool                 is_training,
                               int                  heads,
                               torch::Tensor const& inputs_q, 
                               torch::Tensor const& inputs_kv, 
                               torch::Tensor const& input_weights_q,
                               torch::Tensor const& input_weights_kv,
                               torch::Tensor const& output_weights,
                               const uint8_t*       pad_mask,
                               float                dropout_prob
                                                  );
std::vector<torch::Tensor> bwd_cuda(
                               int                  heads,
                               torch::Tensor const& output_grads, 
                               torch::Tensor const& matmul2_results,
                               torch::Tensor const& dropout_results,
                               torch::Tensor const& softmax_results,
                               torch::Tensor const& input_lin_q_results,
                               torch::Tensor const& input_lin_kv_results,
                               torch::Tensor const& inputs_q, 
                               torch::Tensor const& inputs_kv, 
                               torch::Tensor const& input_weights_q,
                               torch::Tensor const& input_weights_kv,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& dropout_mask,
                               float                dropout_prob
                                                  );

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> fwd(
                               bool                 use_mask,
                               bool                 use_time_mask,
                               bool                 is_training,
                               int                  heads,
                               torch::Tensor const& inputs_q, 
                               torch::Tensor const& inputs_kv, 
                               torch::Tensor const& input_weights_q,
                               torch::Tensor const& input_weights_kv,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& pad_mask,
                               float                dropout_prob
                                                 )
{
  AT_ASSERTM(inputs_q.dim()         == 3, "expected 3D tensor");
  AT_ASSERTM(inputs_kv.dim()        == 3, "expected 3D tensor");
  AT_ASSERTM(input_weights_q.dim()  == 2, "expected 2D tensor");
  AT_ASSERTM(input_weights_kv.dim() == 2, "expected 2D tensor");
  AT_ASSERTM(output_weights.dim()   == 2, "expected 2D tensor");

  AT_ASSERTM(inputs_q.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(inputs_kv.type().scalarType()        == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_weights_q.type().scalarType()  == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_weights_kv.type().scalarType() == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(output_weights.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
  
  if (use_mask) {
  	AT_ASSERTM(pad_mask.dim()       == 2, "expected 2D tensor");
  	AT_ASSERTM(pad_mask.type().scalarType()       == at::ScalarType::Byte, "Only BYTE is supported");
  }
  
  return fwd_cuda(
                                 use_time_mask,
                                 is_training,
                                 heads, 
                                 inputs_q, 
                                 inputs_kv, 
                                 input_weights_q, 
                                 input_weights_kv, 
                                 output_weights, 
                                 use_mask ? static_cast<const uint8_t*>(pad_mask.data_ptr()) : nullptr, 
                                 dropout_prob
                                );
}

std::vector<torch::Tensor> bwd(
                               int                  heads,
                               torch::Tensor const& output_grads, 
                               torch::Tensor const& matmul2_results,
                               torch::Tensor const& dropout_results,
                               torch::Tensor const& softmax_results,
                               torch::Tensor const& input_lin_q_results,
                               torch::Tensor const& input_lin_kv_results,
                               torch::Tensor const& inputs_q, 
                               torch::Tensor const& inputs_kv, 
                               torch::Tensor const& input_weights_q,
                               torch::Tensor const& input_weights_kv,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& dropout_mask,
                               float                dropout_prob
                                                  )
{
  AT_ASSERTM(output_grads.dim()         == 3, "expected 3D tensor");
  AT_ASSERTM(matmul2_results.dim()      == 3, "expected 3D tensor");
  AT_ASSERTM(dropout_results.dim()      == 3, "expected 3D tensor");
  AT_ASSERTM(softmax_results.dim()      == 3, "expected 3D tensor");
  AT_ASSERTM(input_lin_q_results.dim()  == 3, "expected 3D tensor");
  AT_ASSERTM(input_lin_kv_results.dim() == 3, "expected 3D tensor");
  AT_ASSERTM(inputs_q.dim()             == 3, "expected 3D tensor");
  AT_ASSERTM(inputs_kv.dim()            == 3, "expected 3D tensor");
  AT_ASSERTM(input_weights_q.dim()      == 2, "expected 2D tensor");
  AT_ASSERTM(input_weights_kv.dim()     == 2, "expected 2D tensor");
  AT_ASSERTM(output_weights.dim()       == 2, "expected 2D tensor");
  AT_ASSERTM(dropout_mask.dim()         == 3, "expected 3D tensor");
  
  AT_ASSERTM(output_grads.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(matmul2_results.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(dropout_results.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(softmax_results.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_lin_q_results.type().scalarType()  == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_lin_kv_results.type().scalarType() == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(inputs_q.type().scalarType()             == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(inputs_kv.type().scalarType()            == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_weights_q.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_weights_kv.type().scalarType()     == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(output_weights.type().scalarType()       == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(dropout_mask.type().scalarType()         == at::ScalarType::Byte, "Only BYTE is supported");
  
  return bwd_cuda(
                                 heads, 
                                 output_grads,
                                 matmul2_results,
                                 dropout_results,
                                 softmax_results, 
                                 input_lin_q_results, 
                                 input_lin_kv_results, 
                                 inputs_q, 
                                 inputs_kv, 
                                 input_weights_q,
                                 input_weights_kv,
                                 output_weights,
                                 dropout_mask, 
                                 dropout_prob
                                );
}

} // end namespace cublaslt_matmul
} // end namespace encdec 
} // end namespace multihead_attn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_init", &multihead_attn::encdec::cublaslt_matmul::fwd_init_cuda, "Multihead Attention Forward Init.");
  m.def("backward_init", &multihead_attn::encdec::cublaslt_matmul::bwd_init_cuda, "Multihead Attention Backward Init.");
  m.def("forward_del", &multihead_attn::encdec::cublaslt_matmul::fwd_del_cuda, "Multihead Attention Forward Del.");
  m.def("backward_del", &multihead_attn::encdec::cublaslt_matmul::bwd_del_cuda, "Multihead Attention Backward Del.");
  m.def("forward", &multihead_attn::encdec::cublaslt_matmul::fwd, "Encdec Multihead Attention Forward.");
  m.def("backward", &multihead_attn::encdec::cublaslt_matmul::bwd, "Encdec Multihead Attention Backward.");
}
