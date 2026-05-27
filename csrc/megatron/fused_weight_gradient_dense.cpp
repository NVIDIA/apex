#include <ATen/ATen.h>
#include <torch/library.h>

#include <cstdio>
#include <vector>

void wgrad_gemm_accum_fp32_cuda_stub(at::Tensor& input_2d, at::Tensor& d_output_2d, at::Tensor& d_weight);

void wgrad_gemm_accum_fp16_cuda_stub(at::Tensor& input_2d, at::Tensor& d_output_2d, at::Tensor& d_weight);

void wgrad_gemm_accum_fp32_dispatch(const at::Tensor& input, const at::Tensor& d_output, const at::Tensor& d_weight) {
  at::Tensor input_arg = input;
  at::Tensor d_output_arg = d_output;
  at::Tensor d_weight_arg = d_weight;
  wgrad_gemm_accum_fp32_cuda_stub(input_arg, d_output_arg, d_weight_arg);
}

void wgrad_gemm_accum_fp16_dispatch(const at::Tensor& input, const at::Tensor& d_output, const at::Tensor& d_weight) {
  at::Tensor input_arg = input;
  at::Tensor d_output_arg = d_output;
  at::Tensor d_weight_arg = d_weight;
  wgrad_gemm_accum_fp16_cuda_stub(input_arg, d_output_arg, d_weight_arg);
}

TORCH_LIBRARY_FRAGMENT(apex, m) {
  m.def("fused_weight_gradient_mlp_wgrad_gemm_accum_fp32(Tensor input, Tensor d_output, Tensor(a!) d_weight) -> ()");
  m.def("fused_weight_gradient_mlp_wgrad_gemm_accum_fp16(Tensor input, Tensor d_output, Tensor(a!) d_weight) -> ()");
}

TORCH_LIBRARY_IMPL(apex, CUDA, m) {
  m.impl("fused_weight_gradient_mlp_wgrad_gemm_accum_fp32", &wgrad_gemm_accum_fp32_dispatch);
  m.impl("fused_weight_gradient_mlp_wgrad_gemm_accum_fp16", &wgrad_gemm_accum_fp16_dispatch);
}
