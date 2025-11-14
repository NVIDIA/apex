#include <torch/csrc/utils/tensor_flatten.h>
#include <torch/extension.h>
// https://github.com/pytorch/pytorch/blob/master/torch/csrc/utils/tensor_flatten.h

at::Tensor flatten(std::vector<at::Tensor> tensors) { return torch::utils::flatten_dense_tensors(tensors); }

std::vector<at::Tensor> unflatten(at::Tensor flat, std::vector<at::Tensor> tensors) {
  return torch::utils::unflatten_dense_tensors(flat, tensors);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flatten", &flatten, "Flatten dense tensors", py::call_guard<py::gil_scoped_release>());
  m.def("unflatten", &unflatten, "Unflatten dense tensors", py::call_guard<py::gil_scoped_release>());
}
