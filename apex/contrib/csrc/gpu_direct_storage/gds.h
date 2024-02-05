// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

#ifndef __TORCH_GDS_H__
#define __TORCH_GDS_H__

#include <torch/torch.h>
#include <string>

namespace apex::contrib::gds {
  void load_data(torch::Tensor& tensor, std::string& filename);
  void save_data(torch::Tensor& tensor, std::string& filename);
  void load_data_no_gds(torch::Tensor& tensor, std::string& filename);
  void save_data_no_gds(torch::Tensor& tensor, std::string& filename);
}

#endif  //__TORCH_GDS_H__
