#include <torch/library.h>

void multi_tensor_fused_adam_cuda(int chunk_size, at::Tensor noop_flag,
                                  std::vector<std::vector<at::Tensor>> tensor_lists, at::Tensor grad_scale, float lr,
                                  float beta1, float beta2, float eps, int step, int mode, int bias_correction,
                                  float weight_decay);

void multi_tensor_fused_adam_capturable_cuda(int chunk_size, at::Tensor noop_flag,
                                             std::vector<std::vector<at::Tensor>> tensor_lists, at::Tensor grad_scale,
                                             at::Tensor lr, float beta1, float beta2, float eps, at::Tensor step,
                                             int mode, int bias_correction, float weight_decay);

void multi_tensor_fused_adam_with_param_remainders_cuda(int chunk_size, at::Tensor noop_flag,
                                                        std::vector<std::vector<at::Tensor>> tensor_lists,
                                                        at::Tensor grad_scale, float lr, float beta1, float beta2,
                                                        float eps, int step, int mode, int bias_correction,
                                                        float weight_decay);

void multi_tensor_fused_adam(int64_t chunk_size, at::Tensor noop_flag,
                             std::vector<std::vector<at::Tensor>> tensor_lists, at::Tensor grad_scale, double lr,
                             double beta1, double beta2, double eps, int64_t step, int64_t mode,
                             int64_t bias_correction, double weight_decay) {
  multi_tensor_fused_adam_cuda(static_cast<int>(chunk_size), noop_flag, tensor_lists, grad_scale,
                               static_cast<float>(lr), static_cast<float>(beta1), static_cast<float>(beta2),
                               static_cast<float>(eps), static_cast<int>(step), static_cast<int>(mode),
                               static_cast<int>(bias_correction), static_cast<float>(weight_decay));
}

void multi_tensor_fused_adam_capturable(int64_t chunk_size, at::Tensor noop_flag,
                                        std::vector<std::vector<at::Tensor>> tensor_lists, at::Tensor grad_scale,
                                        at::Tensor lr, double beta1, double beta2, double eps, at::Tensor step,
                                        int64_t mode, int64_t bias_correction, double weight_decay) {
  multi_tensor_fused_adam_capturable_cuda(static_cast<int>(chunk_size), noop_flag, tensor_lists, grad_scale, lr,
                                          static_cast<float>(beta1), static_cast<float>(beta2), static_cast<float>(eps),
                                          step, static_cast<int>(mode), static_cast<int>(bias_correction),
                                          static_cast<float>(weight_decay));
}

void multi_tensor_fused_adam_with_param_remainders(int64_t chunk_size, at::Tensor noop_flag,
                                                   std::vector<std::vector<at::Tensor>> tensor_lists,
                                                   at::Tensor grad_scale, double lr, double beta1, double beta2,
                                                   double eps, int64_t step, int64_t mode, int64_t bias_correction,
                                                   double weight_decay) {
  multi_tensor_fused_adam_with_param_remainders_cuda(
      static_cast<int>(chunk_size), noop_flag, tensor_lists, grad_scale, static_cast<float>(lr),
      static_cast<float>(beta1), static_cast<float>(beta2), static_cast<float>(eps), static_cast<int>(step),
      static_cast<int>(mode), static_cast<int>(bias_correction), static_cast<float>(weight_decay));
}

TORCH_LIBRARY_FRAGMENT(apex, m) {
  m.def(
      "distributed_adam_multi_tensor_fused_adam(int chunk_size, Tensor noop_flag, Tensor[][] tensor_lists, "
      "Tensor grad_scale, float lr, float beta1, float beta2, float eps, int step, int mode, "
      "int bias_correction, float weight_decay) -> ()");
  m.def(
      "distributed_adam_multi_tensor_fused_adam_capturable(int chunk_size, Tensor noop_flag, "
      "Tensor[][] tensor_lists, Tensor grad_scale, Tensor lr, float beta1, float beta2, float eps, "
      "Tensor step, int mode, int bias_correction, float weight_decay) -> ()");
  m.def(
      "distributed_adam_multi_tensor_fused_adam_with_param_remainders(int chunk_size, Tensor noop_flag, "
      "Tensor[][] tensor_lists, Tensor grad_scale, float lr, float beta1, float beta2, float eps, "
      "int step, int mode, int bias_correction, float weight_decay) -> ()");
}

TORCH_LIBRARY_IMPL(apex, CUDA, m) {
  m.impl("distributed_adam_multi_tensor_fused_adam", &multi_tensor_fused_adam);
  m.impl("distributed_adam_multi_tensor_fused_adam_capturable", &multi_tensor_fused_adam_capturable);
  m.impl("distributed_adam_multi_tensor_fused_adam_with_param_remainders",
         &multi_tensor_fused_adam_with_param_remainders);
}
