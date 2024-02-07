// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

#pragma once

#include <string>
#include <cufile.h>
#include <torch/torch.h>

namespace apex::contrib::gds {
  class File {
    public:
    File();
    File(const std::string& filename, const std::string& mode);
    ~File();

    void open(const std::string& filename, const std::string& mode);
    void close();

    void load_data(const torch::Tensor& tensor);
    void save_data(const torch::Tensor& tensor);
    void load_data_no_gds(const torch::Tensor& tensor);
    void save_data_no_gds(const torch::Tensor& tensor);

    private:
    std::string filename;
    std::string mode;

    CUfileDescr_t cf_descr;
    CUfileHandle_t cf_handle;
    CUfileError_t status;

    int fd = -1;
    bool is_open = false;
    bool maybe_register = true;
  };
}
