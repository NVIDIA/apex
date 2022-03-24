#include <torch/torch.h>

#include <vector>
#include <cstdint>

// CUDA forward declarations

std::vector<at::Tensor> focal_loss_forward_cuda(
  const at::Tensor &cls_output,
  const at::Tensor &cls_targets_at_level,
  const at::Tensor &num_positives_sum,
  const int64_t num_real_classes,
  const float alpha,
  const float gamma,
  const float smoothing_factor);

at::Tensor focal_loss_backward_cuda(
  const at::Tensor &grad_output,
  const at::Tensor &partial_grad,
  const at::Tensor &num_positives_sum);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> focal_loss_forward(
  const at::Tensor &cls_output,
  const at::Tensor &cls_targets_at_level,
  const at::Tensor &num_positives_sum,
  const int64_t num_real_classes,
  const float alpha,
  const float gamma,
  const float smoothing_factor
) {
  CHECK_INPUT(cls_output);
  CHECK_INPUT(cls_targets_at_level);
  CHECK_INPUT(num_positives_sum);

  return focal_loss_forward_cuda(
    cls_output,
    cls_targets_at_level,
    num_positives_sum,
    num_real_classes,
    alpha,
    gamma,
    smoothing_factor);
}

at::Tensor focal_loss_backward(
  const at::Tensor &grad_output,
  const at::Tensor &partial_grad,
  const at::Tensor &num_positives_sum
) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(partial_grad);

  return focal_loss_backward_cuda(grad_output, partial_grad, num_positives_sum);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &focal_loss_forward,
        "Focal loss calculation forward (CUDA)");
  m.def("backward", &focal_loss_backward,
        "Focal loss calculation backward (CUDA)");
}
