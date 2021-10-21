#include <torch/extension.h>
#include <vector>

namespace multihead_attn {
namespace self {
namespace cublas_gemmex {

std::vector<torch::Tensor> fwd_cuda(
                               bool                 use_time_mask,  
                               bool                 is_training,
                               int                  heads,
                               torch::Tensor const& inputs, 
                               torch::Tensor const& input_weights,
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
                               torch::Tensor const& input_lin_results,
                               torch::Tensor const& inputs, 
                               torch::Tensor const& input_weights,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& dropout_mask,
                               float                dropout_prob
                                                  );

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> fwd(
 							   bool 				use_mask,
                               bool                 use_time_mask,
                               bool                 is_training,
                               int                  heads,
                               torch::Tensor const& inputs, torch::Tensor const& input_weights,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& pad_mask,
                               float                dropout_prob
                                                 )
{
  AT_ASSERTM(inputs.dim()         == 3, "expected 3D tensor");
  AT_ASSERTM(input_weights.dim()  == 2, "expected 2D tensor");
  AT_ASSERTM(output_weights.dim() == 2, "expected 2D tensor");

  AT_ASSERTM(inputs.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_weights.type().scalarType()  == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(output_weights.type().scalarType() == at::ScalarType::Half, "Only HALF is supported");

  if (use_mask) {
  	AT_ASSERTM(pad_mask.dim()                     == 2,                    "expected 2D tensor");
  	AT_ASSERTM(pad_mask.type().scalarType()       == at::ScalarType::Byte, "Only BYTE is supported");
  }
  
  return fwd_cuda(
                                 use_time_mask,
                                 is_training,
                                 heads, 
                                 inputs, 
                                 input_weights, 
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
                               torch::Tensor const& input_lin_results,
                               torch::Tensor const& inputs, 
                               torch::Tensor const& input_weights,
                               torch::Tensor const& output_weights,
                               torch::Tensor const& dropout_mask,
                               float                dropout_prob
                                                  )
{
  AT_ASSERTM(output_grads.dim()      == 3, "expected 3D tensor");
  AT_ASSERTM(matmul2_results.dim()   == 3, "expected 3D tensor");
  AT_ASSERTM(dropout_results.dim()   == 3, "expected 3D tensor");
  AT_ASSERTM(softmax_results.dim()   == 3, "expected 3D tensor");
  AT_ASSERTM(input_lin_results.dim() == 3, "expected 3D tensor");
  AT_ASSERTM(inputs.dim()            == 3, "expected 3D tensor");
  AT_ASSERTM(input_weights.dim()     == 2, "expected 2D tensor");
  AT_ASSERTM(output_weights.dim()    == 2, "expected 2D tensor");
  AT_ASSERTM(dropout_mask.dim()      == 3, "expected 3D tensor");
  
  AT_ASSERTM(output_grads.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(matmul2_results.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(dropout_results.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(softmax_results.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_lin_results.type().scalarType() == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(inputs.type().scalarType()            == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(input_weights.type().scalarType()     == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(output_weights.type().scalarType()    == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(dropout_mask.type().scalarType()      == at::ScalarType::Byte, "Only BYTE is supported");
  
  return bwd_cuda(
                                 heads, 
                                 output_grads,
                                 matmul2_results,
                                 dropout_results,
                                 softmax_results, 
                                 input_lin_results, 
                                 inputs, 
                                 input_weights,
                                 output_weights,
                                 dropout_mask, 
                                 dropout_prob
                                );
}

} // end namespace cublas_gemmex
} // end namespace self
} // end namespace multihead_attn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &multihead_attn::self::cublas_gemmex::fwd, "Self Multihead Attention Forward.");
  m.def("backward", &multihead_attn::self::cublas_gemmex::bwd, "Self Multihead Attention Backward.");
}

