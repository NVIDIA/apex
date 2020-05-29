#include <torch/extension.h>
#include <vector>

namespace multihead_attn {
namespace fused_softmax {
namespace mask_softmax_dropout {

std::vector<torch::Tensor> fwd_cuda(
                               bool                 is_training,
                               int                  heads,
                               torch::Tensor const& input, 
                               const uint8_t*       pad_mask,
                               float                dropout_prob
                                                  );

torch::Tensor bwd_cuda(
		               int heads,
                               torch::Tensor const& output_grads, 
                               torch::Tensor const& softmax_results,
                               torch::Tensor const& dropout_mask,
                               const uint8_t *padding_mask,
                               float                dropout_prob
                                                  );

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> fwd(
 			       bool 				use_mask,
                               bool                 is_training,
                               int                  heads,
                               torch::Tensor const& input,
                               torch::Tensor const& pad_mask,
                               float                dropout_prob
                                                 )
{
  AT_ASSERTM(input.dim()         == 3, "expected 3D tensor");
  AT_ASSERTM(input.type().scalarType()         == at::ScalarType::Half, "Only HALF is supported");

  if (use_mask) {
  	AT_ASSERTM(pad_mask.dim()                     == 2,                    "expected 2D tensor");
  	AT_ASSERTM(pad_mask.type().scalarType()       == at::ScalarType::Byte, "Only BYTE is supported");
  }

  return fwd_cuda(
                                 is_training,
                                 heads, 
                                 input, 
                                 use_mask ? static_cast<const uint8_t*>(pad_mask.data_ptr()) : nullptr, 
                                 dropout_prob
                                );
}

torch::Tensor bwd(
		               bool use_mask,
		               int heads,
                               torch::Tensor const& output_grads, 
                               torch::Tensor const& softmax_results,
                               torch::Tensor const& dropout_mask,
                               torch::Tensor const& padding_mask,
                               float                dropout_prob
                                                  )
{
  AT_ASSERTM(output_grads.dim()      == 3, "expected 3D tensor");
  AT_ASSERTM(softmax_results.dim()   == 3, "expected 3D tensor");
  AT_ASSERTM(dropout_mask.dim()      == 3, "expected 3D tensor");

  AT_ASSERTM(output_grads.type().scalarType()      == at::ScalarType::Half, "Only HALF is supported");
  AT_ASSERTM(softmax_results.type().scalarType()   == at::ScalarType::Half, "Only HALF is supported");
//  AT_ASSERTM(dropout_mask.type().scalarType()      == at::ScalarType::Byte, "Only BYTE is supported");

  return bwd_cuda(
		                 heads,
                                 output_grads,
                                 softmax_results, 
                                 dropout_mask, 
                                 use_mask ? static_cast<const uint8_t*>(padding_mask.data_ptr()) : nullptr, 
                                 dropout_prob
                                );
}

} // end namespace mask_softmax_dropout
} // end namespace fused_softmax
} // end namespace multihead_attn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &multihead_attn::fused_softmax::mask_softmax_dropout::fwd, "Self Multihead Attention masked softmax dropout -- Forward.");
  m.def("backward", &multihead_attn::fused_softmax::mask_softmax_dropout::bwd, "Self Multihead Attention masked softmax dropout -- Backward.");
}

