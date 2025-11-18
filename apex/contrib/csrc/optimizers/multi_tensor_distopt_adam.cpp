#include <torch/extension.h>

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multi_tensor_fused_adam", &multi_tensor_fused_adam_cuda,
        "CUDA kernels for multi-tensor Adam, "
        "with param copy",
        py::call_guard<py::gil_scoped_release>());
  m.def("multi_tensor_fused_adam_capturable", &multi_tensor_fused_adam_capturable_cuda,
        "CUDA kernels for multi-tensor Adam, "
        "with param copy, capturable for CUDA graph",
        py::call_guard<py::gil_scoped_release>());
  m.def("multi_tensor_fused_adam_with_param_remainders", &multi_tensor_fused_adam_with_param_remainders_cuda,
        "CUDA kernel for multi-tensor Adam, "
        "with stored param remainders and param copy",
        py::call_guard<py::gil_scoped_release>());
}
