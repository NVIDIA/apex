#include <torch/extension.h>

void multi_tensor_lamb_compute_update_term_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor per_tensor_beta1,
  at::Tensor per_tensor_beta2,
  at::Tensor per_tensor_beta3,
  at::Tensor per_tensor_bias_correction,
  const int step,
  at::Tensor per_tensor_epsilon,
  const int mode,
  at::Tensor per_tensor_decay,
  const float grad_scale);

void multi_tensor_lamb_update_weights_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor per_tensor_param_norm,
  at::Tensor per_tensor_update_norm,
  const float learning_rate,
  at::Tensor per_tensor_decay,
  bool use_nvlamb);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multi_tensor_lamb_compute_update_term", &multi_tensor_lamb_compute_update_term_cuda,
        "Computes update term for LAMB optimizer");
  m.def("multi_tensor_lamb_update_weights", &multi_tensor_lamb_update_weights_cuda,
        "Applies update term for LAMB optimizer");
}
