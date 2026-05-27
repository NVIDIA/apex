#include <torch/library.h>

void multi_tensor_lamb_compute_update_term_cuda(int chunk_size, at::Tensor noop_flag,
                                                std::vector<std::vector<at::Tensor>> tensor_lists,
                                                at::Tensor per_tensor_beta1, at::Tensor per_tensor_beta2,
                                                at::Tensor per_tensor_beta3, at::Tensor per_tensor_bias_correction,
                                                at::Tensor step, at::Tensor per_tensor_epsilon, const int mode,
                                                at::Tensor per_tensor_decay, at::Tensor global_scale,
                                                at::Tensor global_grad_norm, const float max_grad_norm);

void multi_tensor_lamb_update_weights_cuda(int chunk_size, at::Tensor noop_flag,
                                           std::vector<std::vector<at::Tensor>> tensor_lists,
                                           at::Tensor per_tensor_param_norm, at::Tensor per_tensor_update_norm,
                                           at::Tensor update_norm_offset, at::Tensor learning_rate,
                                           at::Tensor per_tensor_decay, at::Tensor global_grad_norm, bool use_nvlamb);

void multi_tensor_lamb_compute_update_term(int64_t chunk_size, at::Tensor noop_flag,
                                           std::vector<std::vector<at::Tensor>> tensor_lists,
                                           at::Tensor per_tensor_beta1, at::Tensor per_tensor_beta2,
                                           at::Tensor per_tensor_beta3, at::Tensor per_tensor_bias_correction,
                                           at::Tensor step, at::Tensor per_tensor_epsilon, int64_t mode,
                                           at::Tensor per_tensor_decay, at::Tensor global_scale,
                                           at::Tensor global_grad_norm, double max_grad_norm) {
  multi_tensor_lamb_compute_update_term_cuda(static_cast<int>(chunk_size), noop_flag, tensor_lists, per_tensor_beta1,
                                             per_tensor_beta2, per_tensor_beta3, per_tensor_bias_correction, step,
                                             per_tensor_epsilon, static_cast<int>(mode), per_tensor_decay, global_scale,
                                             global_grad_norm, static_cast<float>(max_grad_norm));
}

void multi_tensor_lamb_update_weights(int64_t chunk_size, at::Tensor noop_flag,
                                      std::vector<std::vector<at::Tensor>> tensor_lists,
                                      at::Tensor per_tensor_param_norm, at::Tensor per_tensor_update_norm,
                                      at::Tensor update_norm_offset, at::Tensor learning_rate,
                                      at::Tensor per_tensor_decay, at::Tensor global_grad_norm, bool use_nvlamb) {
  multi_tensor_lamb_update_weights_cuda(static_cast<int>(chunk_size), noop_flag, tensor_lists, per_tensor_param_norm,
                                        per_tensor_update_norm, update_norm_offset, learning_rate, per_tensor_decay,
                                        global_grad_norm, use_nvlamb);
}

TORCH_LIBRARY_FRAGMENT(apex, m) {
  m.def(
      "distributed_lamb_compute_update_term(int chunk_size, Tensor noop_flag, Tensor[][] tensor_lists, "
      "Tensor per_tensor_beta1, Tensor per_tensor_beta2, Tensor per_tensor_beta3, "
      "Tensor per_tensor_bias_correction, Tensor step, Tensor per_tensor_epsilon, int mode, "
      "Tensor per_tensor_decay, Tensor global_scale, Tensor global_grad_norm, float max_grad_norm) -> ()");
  m.def(
      "distributed_lamb_update_weights(int chunk_size, Tensor noop_flag, Tensor[][] tensor_lists, "
      "Tensor per_tensor_param_norm, Tensor per_tensor_update_norm, Tensor update_norm_offset, "
      "Tensor learning_rate, Tensor per_tensor_decay, Tensor global_grad_norm, bool use_nvlamb) -> ()");
}

TORCH_LIBRARY_IMPL(apex, CUDA, m) {
  m.impl("distributed_lamb_compute_update_term", &multi_tensor_lamb_compute_update_term);
  m.impl("distributed_lamb_update_weights", &multi_tensor_lamb_update_weights);
}
