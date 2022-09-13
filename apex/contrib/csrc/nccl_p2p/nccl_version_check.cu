// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

// This file is used to check the version of NCCL detected.
#include <tuple>
#include <nccl.h>


std::tuple<int, int> get_nccl_version() {
  return { int(NCCL_MAJOR), int(NCCL_MINOR) };
}
