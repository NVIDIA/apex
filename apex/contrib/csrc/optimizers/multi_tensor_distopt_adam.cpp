#include <torch/extension.h>

void multi_tensor_fused_adam_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor per_tensor_beta1,
  at::Tensor per_tensor_beta2,
  at::Tensor per_tensor_bias_correction,
  at::Tensor per_tensor_eps,
  at::Tensor per_tensor_weight_decay,
  float lr,
  float grad_scale,
  int step,
  int mode);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multi_tensor_fused_adam", &multi_tensor_fused_adam_cuda,
        "Multi tensor Adam optimized CUDA implementation.");
}
