#include <torch/extension.h>

void multi_tensor_lamb_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  const float lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  const int step,
  const int bias_correction,
  const float weight_decay,
  const int grad_averaging,
  const int mode,
  const float global_grad_norm,
  const float max_grad_norm);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("lamb", &multi_tensor_lamb_cuda, "Computes and apply update for LAMB optimizer");
}
