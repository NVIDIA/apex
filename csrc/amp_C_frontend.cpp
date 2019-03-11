#include <torch/extension.h>

void multi_tensor_scale_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  float scale);

void multi_tensor_sgd_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  float wd,
  float momentum,
  float dampening,
  float lr,
  bool nesterov,
  bool first_run,
  bool wd_after_momentum);

void scale_check_overflow_cuda(
  const at::Tensor& grads,
  float scale,
  const at::Tensor& d_buf,
  const at::Tensor& downscaled_grads);

void scale_check_overflow(
  at::Tensor grads,
  float scale,
  at::Tensor overflow_buf,
  at::Tensor downscaled_grads)
  // const at::optional<at::Tensor> downscaled_grads)
{ 
  AT_CHECK(grads.type().is_cuda(), "grads must be a CUDA tensor");
  AT_CHECK(grads.is_contiguous(), "grads must be contiguous");
  AT_CHECK(overflow_buf.type().is_cuda(), "overflow_buf must be a CUDA tensor");
  AT_CHECK(overflow_buf.is_contiguous(), "overflow_buf must be contiguous");
  AT_CHECK(downscaled_grads.type().is_cuda(), "downscaled_grads must be a CUDA tensor");
  AT_CHECK(downscaled_grads.is_contiguous(), "downscaled_grads must be contiguous");
  // Make sure we are downscaling the FP32 master grads
  AT_CHECK(downscaled_grads.type().scalarType() == at::ScalarType::Float,
    "The output grads supplied to scale_check_overflow should be fp32 (master grads).")
  AT_CHECK(grads.numel() == downscaled_grads.numel(), "Input and output grads must be the same size.");

  scale_check_overflow_cuda(grads, scale, overflow_buf, downscaled_grads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scale_check_overflow", &scale_check_overflow, "Fused overflow check + scale for FP32 tensors");
  m.def("multi_tensor_scale", &multi_tensor_scale_cuda,
        "Fused overflow check + scale for a list of contiguous tensors");
  m.def("multi_tensor_sgd", &multi_tensor_sgd_cuda,
        "Fused SGD optimizer for list of contiguous tensors");
}
