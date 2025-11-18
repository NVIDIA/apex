// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

// This file is used to check the version of NCCL detected.
#include <nccl.h>

#include <tuple>

std::tuple<int, int> get_nccl_version() { return {int(NCCL_MAJOR), int(NCCL_MINOR)}; }
