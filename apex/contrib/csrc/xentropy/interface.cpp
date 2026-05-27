#include <ATen/ATen.h>
#include <torch/library.h>

#include <string>

// CUDA forward declarations

std::vector<at::Tensor> softmax_xentropy_cuda(const at::Tensor& input, const at::Tensor& labels, const float smoothing,
                                              const bool half_to_float);

at::Tensor softmax_xentropy_backward_cuda(const at::Tensor& grad_loss, const at::Tensor& logits,
                                          const at::Tensor& max_log_sum_exp, const at::Tensor& labels,
                                          const float smoothing);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> softmax_xentropy_forward(const at::Tensor& input, const at::Tensor& labels,
                                                 const double smoothing, const bool half_to_float) {
  CHECK_CUDA(input);
  CHECK_INPUT(labels);

  return softmax_xentropy_cuda(input, labels, static_cast<float>(smoothing), half_to_float);
}

at::Tensor softmax_xentropy_backward(const at::Tensor& grad_loss, const at::Tensor& logits,
                                     const at::Tensor& max_log_sum_exp, const at::Tensor& labels,
                                     const double smoothing) {
  CHECK_CUDA(grad_loss);
  CHECK_CUDA(logits);
  CHECK_INPUT(max_log_sum_exp);
  CHECK_INPUT(labels);

  return softmax_xentropy_backward_cuda(grad_loss, logits, max_log_sum_exp, labels, static_cast<float>(smoothing));
}

std::string softmax_xentropy_version() {
#ifdef XENTROPY_VER
  return XENTROPY_VER;
#else
  return {};
#endif
}

TORCH_LIBRARY_FRAGMENT(apex, m) {
  m.def("xentropy_forward(Tensor input, Tensor labels, float smoothing, bool half_to_float) -> Tensor[]");
  m.def("xentropy_backward(Tensor grad_loss, Tensor logits, Tensor max_log_sum_exp, Tensor labels, "
        "float smoothing) -> Tensor");
  m.def("xentropy_version() -> str");
}

TORCH_LIBRARY_IMPL(apex, CUDA, m) {
  m.impl("xentropy_forward", &softmax_xentropy_forward);
  m.impl("xentropy_backward", &softmax_xentropy_backward);
}

TORCH_LIBRARY_IMPL(apex, CompositeExplicitAutograd, m) {
  m.impl("xentropy_version", &softmax_xentropy_version);
}
