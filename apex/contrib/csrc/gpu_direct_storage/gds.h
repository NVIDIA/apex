// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

#pragma once

#include <torch/torch.h>
#include <string>

namespace apex::contrib::gds {
  void load_data(const torch::Tensor& tensor, const std::string& filename);
  void save_data(const torch::Tensor& tensor, const std::string& filename);
  void load_data_no_gds(const torch::Tensor& tensor, const std::string& filename);
  void save_data_no_gds(const torch::Tensor& tensor, const std::string& filename);
}
