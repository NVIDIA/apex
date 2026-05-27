#include <ATen/ATen.h>
#include <torch/library.h>

void multi_tensor_lamb_cuda(int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
                            const float lr, const float beta1, const float beta2, const float epsilon, const int step,
                            const int bias_correction, const float weight_decay, const int grad_averaging,
                            const int mode, const float global_grad_norm, const float max_grad_norm);

void multi_tensor_lamb_dispatch(int64_t chunk_size, at::Tensor noop_flag,
                                std::vector<std::vector<at::Tensor>> tensor_lists, double lr, double beta1,
                                double beta2, double epsilon, int64_t step, int64_t bias_correction,
                                double weight_decay, int64_t grad_averaging, int64_t mode, double global_grad_norm,
                                double max_grad_norm) {
  multi_tensor_lamb_cuda(static_cast<int>(chunk_size), noop_flag, tensor_lists, static_cast<float>(lr),
                         static_cast<float>(beta1), static_cast<float>(beta2), static_cast<float>(epsilon),
                         static_cast<int>(step), static_cast<int>(bias_correction), static_cast<float>(weight_decay),
                         static_cast<int>(grad_averaging), static_cast<int>(mode), static_cast<float>(global_grad_norm),
                         static_cast<float>(max_grad_norm));
}

TORCH_LIBRARY_FRAGMENT(apex, m) {
  m.def("fused_lamb_lamb(int chunk_size, Tensor noop_flag, Tensor[][] tensor_lists, float lr, float beta1, "
        "float beta2, float epsilon, int step, int bias_correction, float weight_decay, int grad_averaging, "
        "int mode, float global_grad_norm, float max_grad_norm) -> ()");
}

TORCH_LIBRARY_IMPL(apex, CUDA, m) {
  m.impl("fused_lamb_lamb", &multi_tensor_lamb_dispatch);
}
