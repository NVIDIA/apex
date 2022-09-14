// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
// This file is used to check the version of NCCL detected.
#include <tuple>

#include <torch/extension.h>

std::tuple<int, int> get_nccl_version();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_nccl_version", &get_nccl_version);
}
