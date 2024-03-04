#include <iostream>
#include <vector>

#include <torch/extension.h>

std::vector<at::Tensor> instance_norm_nvfuser_forward(
  at::Tensor input,
  at::Tensor weight,
  at::Tensor bias,
  at::Tensor run_mean,
  at::Tensor run_var,
  const bool use_input_stats,
  const float momentum,
  const float eps,
  const bool channels_last = false
  );

std::vector<at::Tensor> instance_norm_nvfuser_backward(
  at::Tensor input,
  at::Tensor grad_output,
  at::Tensor weight,
  at::Tensor running_mean,
  at::Tensor running_var,
  at::Tensor save_mean,
  at::Tensor save_invstd,
  const bool use_input_stats,
  const float eps,
  // const std::vector<bool>& output_mask,
  bool channels_last = false
  );

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &instance_norm_nvfuser_forward, "instance_norm forward (CUDA)");
  m.def("backward", &instance_norm_nvfuser_backward, "instance_norm backward (CUDA)");
}
