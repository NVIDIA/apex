#include <torch/extension.h>

#include <cstdio>
#include <vector>

void wgrad_gemm_accum_fp32_cuda_stub(
  at::Tensor &input_2d,
  at::Tensor &d_output_2d,
  at::Tensor &d_weight
);

void wgrad_gemm_accum_fp16_cuda_stub(
  at::Tensor &input_2d,
  at::Tensor &d_output_2d,
  at::Tensor &d_weight
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wgrad_gemm_accum_fp32", &wgrad_gemm_accum_fp32_cuda_stub, "wgrad gemm accum in fp32");
    m.def("wgrad_gemm_accum_fp16", &wgrad_gemm_accum_fp16_cuda_stub, "wgrad gemm accum in fp16");
}
