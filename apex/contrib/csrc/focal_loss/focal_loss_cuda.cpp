#include <ATen/ATen.h>
#include <torch/library.h>

#include <cstdint>
#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> focal_loss_forward_cuda(const at::Tensor& cls_output, const at::Tensor& cls_targets_at_level,
                                                const at::Tensor& num_positives_sum, const int64_t num_real_classes,
                                                const float alpha, const float gamma, const float smoothing_factor);

at::Tensor focal_loss_backward_cuda(const at::Tensor& grad_output, const at::Tensor& partial_grad,
                                    const at::Tensor& num_positives_sum);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> focal_loss_forward(const at::Tensor& cls_output, const at::Tensor& cls_targets_at_level,
                                           const at::Tensor& num_positives_sum, const int64_t num_real_classes,
                                           const double alpha, const double gamma, const double smoothing_factor) {
  CHECK_INPUT(cls_output);
  CHECK_INPUT(cls_targets_at_level);
  CHECK_INPUT(num_positives_sum);

  return focal_loss_forward_cuda(cls_output, cls_targets_at_level, num_positives_sum, num_real_classes,
                                 static_cast<float>(alpha), static_cast<float>(gamma),
                                 static_cast<float>(smoothing_factor));
}

at::Tensor focal_loss_backward(const at::Tensor& grad_output, const at::Tensor& partial_grad,
                               const at::Tensor& num_positives_sum) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(partial_grad);

  return focal_loss_backward_cuda(grad_output, partial_grad, num_positives_sum);
}

TORCH_LIBRARY_FRAGMENT(apex, m) {
  m.def("focal_loss_forward(Tensor cls_output, Tensor cls_targets_at_level, Tensor num_positives_sum, "
        "int num_real_classes, float alpha, float gamma, float smoothing_factor) -> Tensor[]");
  m.def("focal_loss_backward(Tensor grad_output, Tensor partial_grad, Tensor num_positives_sum) -> Tensor");
}

TORCH_LIBRARY_IMPL(apex, CUDA, m) {
  m.impl("focal_loss_forward", &focal_loss_forward);
  m.impl("focal_loss_backward", &focal_loss_backward);
}
