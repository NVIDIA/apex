// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

#include <gds.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <string>

//python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("load_data", [](torch::Tensor& tensor, std::string& filename) { apex::contrib::gds::load_data(tensor, filename); });
  m.def("save_data", [](torch::Tensor& tensor, std::string& filename) { apex::contrib::gds::save_data(tensor, filename); });
  m.def("load_data_no_gds", [](torch::Tensor& tensor, std::string& filename) { apex::contrib::gds::load_data_no_gds(tensor, filename); });
  m.def("save_data_no_gds", [](torch::Tensor& tensor, std::string& filename) { apex::contrib::gds::save_data_no_gds(tensor, filename); });
}
