// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

#include <gds.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <string>

//python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<
      apex::contrib::gds::File,
      std::shared_ptr<apex::contrib::gds::File>>(
      m, "_GDSFile")
  .def(py::init<>())
  .def(py::init<const std::string&, const std::string&>())
  .def("open", &apex::contrib::gds::File::open)
  .def("close", &apex::contrib::gds::File::close)
  .def("load_data", &apex::contrib::gds::File::load_data)
  .def("save_data", &apex::contrib::gds::File::save_data)
  .def("load_data_no_gds", &apex::contrib::gds::File::load_data_no_gds)
  .def("save_data_no_gds", &apex::contrib::gds::File::save_data_no_gds);
}
