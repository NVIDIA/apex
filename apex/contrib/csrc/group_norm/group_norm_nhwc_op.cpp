/***************************************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 *modification, are not permit- ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
 *INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "group_norm_nhwc.h"
#include "group_norm_nhwc_bwd_one_pass.h"
#include "group_norm_nhwc_fwd_one_pass.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUDA_STATUS(call)                                       \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(1);                                                        \
    }                                                                 \
  } while (0)

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CHANNELS_LAST(x)                                 \
  TORCH_CHECK(x.is_contiguous(at::MemoryFormat::ChannelsLast), \
              #x " must be channels last")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_NHWC_INPUT(x) \
  CHECK_CUDA(x);            \
  CHECK_CHANNELS_LAST(x)

enum Algo { OnePass, TwoPass };

static bool initialized = false;
static cudaDeviceProp props;

const std::unordered_set<int> supported_c_values = {
    128,  256,  320,  448,  512,  640,  768,  896,  960,  1024, 1280, 1344,
    1536, 1792, 1920, 2048, 2240, 2560, 2688, 3072, 3136, 3584, 4096};
const std::unordered_set<int> supported_groups_values = {16, 32};

std::vector<torch::Tensor> group_norm_fwd(torch::Tensor input, int groups,
                                          torch::Tensor weight,
                                          torch::Tensor bias, float eps,
                                          Algo algo, bool with_swish = false) {
  if (!initialized) {
    CHECK_CUDA_STATUS(cudaGetDeviceProperties(&props, 0));
    initialized = true;
  }
  CHECK_NHWC_INPUT(input);
  auto stream = at::cuda::getCurrentCUDAStream();

  // Achieve group norm arguments
  int n = input.size(0);
  int c = input.size(1);
  int h = input.size(2);
  int w = input.size(3);

  // Check kernel constraints
  TORCH_CHECK(supported_groups_values.count(groups),
              "`groups` of {16, 32} are only supported but ", groups,
              " is passed");
  TORCH_CHECK(supported_c_values.count(c), "`c` of ", c,
              " is not included in supported_c_values");

  // Allocate tensors
  auto options = at::TensorOptions(at::kCUDA);
  auto output = at::empty_like(input, at::MemoryFormat::Preserve);
  auto sums_d = at::empty({2 * n * groups}, options.dtype(at::kFloat));

  // Declare the parameters.
  Group_norm_nhwc_fwd_params params_fwd;
  memset(&params_fwd, 0, sizeof(params_fwd));

  // Initialize the parameters.
  params_fwd.y = reinterpret_cast<void *>(output.data_ptr());
  params_fwd.sums = reinterpret_cast<float2 *>(sums_d.data_ptr());
  params_fwd.x = const_cast<void *>(reinterpret_cast<void *>(input.data_ptr()));
  params_fwd.gamma =
      const_cast<void *>(reinterpret_cast<void *>(weight.data_ptr()));
  params_fwd.beta =
      const_cast<void *>(reinterpret_cast<void *>(bias.data_ptr()));
  params_fwd.epsilon = eps;
  params_fwd.n = n;
  params_fwd.h = h;
  params_fwd.w = w;
  params_fwd.c = c;
  params_fwd.groups = groups;
  params_fwd.with_swish = with_swish;

  PrecisionMode mode;
  if (input.dtype() == torch::kFloat32) {
    if (weight.dtype() == torch::kFloat16) {
      mode = PrecisionMode::FP32IOFP16W;
    } else if (weight.dtype() == torch::kBFloat16) {
      mode = PrecisionMode::FP32IOBF16W;
    } else {
      mode = PrecisionMode::FP32IOFP32W;
    }
  } else if (input.dtype() == torch::kBFloat16) {
    if (weight.dtype() == torch::kFloat16) {
      mode = PrecisionMode::BF16IOFP16W;
    } else if (weight.dtype() == torch::kBFloat16) {
      mode = PrecisionMode::BF16IOBF16W;
    } else {
      mode = PrecisionMode::BF16IOFP32W;
    }
  } else {
    if (weight.dtype() == torch::kFloat16) {
      mode = PrecisionMode::FP16IOFP16W;
    } else if (weight.dtype() == torch::kBFloat16) {
      mode = PrecisionMode::FP16IOBF16W;
    } else {
      mode = PrecisionMode::FP16IOFP32W;
    }
  }
  params_fwd.precision = mode;

  // The number of barriers.
  size_t barriers_elts = 0;
  // The number of elements in the reduction buffer.
  size_t red_buffer_elts = 0;
  // The number of elements in the reduction buffer that must be zeroed.
  size_t zeroed_red_buffer_elts = 0;

  // Finalize the parameters.
  dim3 grid;
  if (algo == OnePass) {
    group_norm_nhwc_fwd_one_pass_setup(params_fwd, barriers_elts,
                                       red_buffer_elts, grid, props);
  } else {
    group_norm_nhwc_fwd_two_passes_setup(params_fwd, zeroed_red_buffer_elts);
  }

  // Allocate on the device.
  auto red_buffer = at::empty({red_buffer_elts}, options.dtype(at::kFloat));
  params_fwd.red_buffer = red_buffer.data_ptr<float>();

  // Allocate the buffer if needed.
  auto barriers = at::zeros({barriers_elts}, options.dtype(at::kInt));
  params_fwd.barriers = barriers.data_ptr<int>();
  auto zeroed_red_buffer =
      at::zeros({zeroed_red_buffer_elts}, options.dtype(at::kFloat));
  params_fwd.zeroed_red_buffer = zeroed_red_buffer.data_ptr<float>();

  if (algo == OnePass) {
    group_norm_nhwc_fwd_one_pass_run(params_fwd, grid, stream);
  } else {
    group_norm_nhwc_fwd_two_passes_sum(params_fwd, stream);
    group_norm_nhwc_fwd_two_passes_scale(params_fwd, stream);
  }

  return {output, sums_d};
}

std::vector<torch::Tensor> group_norm_bwd(torch::Tensor grad_output,
                                          torch::Tensor sums,
                                          torch::Tensor input, int groups,
                                          torch::Tensor weight,
                                          torch::Tensor bias, float eps,
                                          Algo algo, bool with_swish = false) {
  if (!initialized) {
    CHECK_CUDA_STATUS(cudaGetDeviceProperties(&props, 0));
    initialized = true;
  }
  CHECK_NHWC_INPUT(grad_output);
  auto stream = at::cuda::getCurrentCUDAStream();

  // Achieve group norm arguments
  int n = input.size(0);
  int c = input.size(1);
  int h = input.size(2);
  int w = input.size(3);

  // Check kernel constraints
  TORCH_CHECK(supported_groups_values.count(groups),
              "`groups` of {16, 32} are only supported but ", groups,
              " is passed");
  TORCH_CHECK(supported_c_values.count(c), "`c` of ", c,
              " is not included in supported_c_values");

  // Allocate tensors
  auto options = at::TensorOptions(at::kCUDA);
  auto grad_input = at::empty_like(input, at::MemoryFormat::Preserve);
  auto grad_weight = at::empty_like(weight, at::MemoryFormat::Preserve);
  auto grad_bias = at::empty_like(bias, at::MemoryFormat::Preserve);
  auto sums_d = at::empty({2 * n * groups}, options.dtype(at::kFloat));

  // Declare the parameters.
  Group_norm_nhwc_bwd_params params_bwd;
  memset(&params_bwd, 0, sizeof(params_bwd));

  // Initialize the parameters.
  params_bwd.dx = reinterpret_cast<void *>(grad_input.data_ptr());
  params_bwd.dgamma = reinterpret_cast<void *>(grad_weight.data_ptr());
  params_bwd.dbeta = reinterpret_cast<void *>(grad_bias.data_ptr());
  params_bwd.sums =
      const_cast<float2 *>(reinterpret_cast<float2 *>(sums.data_ptr()));
  params_bwd.dy =
      const_cast<void *>(reinterpret_cast<void *>(grad_output.data_ptr()));
  params_bwd.x = const_cast<void *>(reinterpret_cast<void *>(input.data_ptr()));
  ;
  params_bwd.gamma =
      const_cast<void *>(reinterpret_cast<void *>(weight.data_ptr()));
  params_bwd.beta =
      const_cast<void *>(reinterpret_cast<void *>(bias.data_ptr()));
  ;
  params_bwd.epsilon = eps;
  params_bwd.n = n;
  params_bwd.h = h;
  params_bwd.w = w;
  params_bwd.c = c;
  params_bwd.groups = groups;
  params_bwd.with_swish = with_swish;

  PrecisionMode mode;
  if (input.dtype() == torch::kFloat32) {
    if (weight.dtype() == torch::kFloat16) {
      mode = PrecisionMode::FP32IOFP16W;
    } else if (weight.dtype() == torch::kBFloat16) {
      mode = PrecisionMode::FP32IOBF16W;
    } else {
      mode = PrecisionMode::FP32IOFP32W;
    }
  } else if (input.dtype() == torch::kBFloat16) {
    if (weight.dtype() == torch::kFloat16) {
      mode = PrecisionMode::BF16IOFP16W;
    } else if (weight.dtype() == torch::kBFloat16) {
      mode = PrecisionMode::BF16IOBF16W;
    } else {
      mode = PrecisionMode::BF16IOFP32W;
    }
  } else {
    if (weight.dtype() == torch::kFloat16) {
      mode = PrecisionMode::FP16IOFP16W;
    } else if (weight.dtype() == torch::kBFloat16) {
      mode = PrecisionMode::FP16IOBF16W;
    } else {
      mode = PrecisionMode::FP16IOFP32W;
    }
  }
  params_bwd.precision = mode;

  // The number of barriers.
  size_t barriers_elts = 0;
  // The number of elements in the reduction buffer.
  size_t red_buffer_elts = 0;
  // The number of elements in the reduction buffer that must be zeroed.
  size_t zeroed_red_buffer_elts = 0;

  // Finalize the parameters.
  dim3 grid;
  if (algo == OnePass) {
    group_norm_nhwc_bwd_one_pass_setup(params_bwd, barriers_elts,
                                       red_buffer_elts, zeroed_red_buffer_elts,
                                       grid, props);
  } else {
    group_norm_nhwc_bwd_two_passes_setup(params_bwd, zeroed_red_buffer_elts);
  }

  // Allocate on the device.
  auto red_buffer = at::empty({red_buffer_elts}, options.dtype(at::kFloat));
  params_bwd.red_buffer = red_buffer.data_ptr<float>();

  // Allocate the buffer if needed.
  auto barriers = at::zeros({barriers_elts}, options.dtype(at::kInt));
  params_bwd.barriers = barriers.data_ptr<int>();
  auto zeroed_red_buffer =
      at::zeros({zeroed_red_buffer_elts}, options.dtype(at::kFloat));
  params_bwd.zeroed_red_buffer = zeroed_red_buffer.data_ptr<float>();

  if (algo == OnePass) {
    group_norm_nhwc_bwd_one_pass_run(params_bwd, grid, stream);
  } else {
    group_norm_nhwc_bwd_two_passes_sum(params_bwd, stream);
    group_norm_nhwc_bwd_two_passes_scale(params_bwd, stream);
  }

  return {grad_input, grad_weight, grad_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::enum_<Algo>(m, "Algo")
      .value("OnePass", OnePass)
      .value("TwoPass", TwoPass)
      .export_values();
  m.def("forward", &group_norm_fwd, "NHWC group norm forward");
  m.def("backward", &group_norm_bwd, "NHWC group norm backward");
}
