#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>
#include <cublasLt.h>
#include <stdio.h>

at::Tensor linear_bias_forward(                      at::Tensor input,   at::Tensor weight,    at::Tensor bias);

std::vector<at::Tensor> linear_bias_backward(        at::Tensor input,   at::Tensor weight,    at::Tensor d_output);

std::vector<at::Tensor> linear_gelu_linear_forward(  at::Tensor input,   at::Tensor weight1,   at::Tensor bias1,     at::Tensor weight2,   at::Tensor bias2);

std::vector<at::Tensor> linear_gelu_linear_backward( at::Tensor input,   at::Tensor gelu_in,   at::Tensor output1,   at::Tensor weight1,   at::Tensor weight2, at::Tensor d_output2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_bias_forward",         &linear_bias_forward,         "linear bias forward");
  m.def("linear_bias_backward",        &linear_bias_backward,        "linear bias backward");
  m.def("linear_gelu_linear_forward",  &linear_gelu_linear_forward,  "linear gelu linear forward");
  m.def("linear_gelu_linear_backward", &linear_gelu_linear_backward, "linear gelu linear backward");
}

