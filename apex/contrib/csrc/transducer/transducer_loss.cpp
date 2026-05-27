#include <ATen/ATen.h>
#include <torch/library.h>

#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> transducer_loss_cuda_forward(at::Tensor x, at::Tensor label, at::Tensor audLen,
                                                        at::Tensor txtLen, at::Tensor batchOffset, int64_t maxFLen,
                                                        int64_t blankIdx, int64_t opt, bool packedInput);

at::Tensor transducer_loss_cuda_backward(at::Tensor x, at::Tensor lossGrad, at::Tensor alpha,
                                            at::Tensor beta, at::Tensor audLen, at::Tensor txtLen,
                                            at::Tensor label, at::Tensor batchOffset, int64_t maxFLen,
                                            int64_t blankIdx, int64_t opt, bool fuseSoftmaxBackward, bool packedInput);

std::vector<at::Tensor> transducer_loss_forward(at::Tensor x, at::Tensor label, at::Tensor fLen,
                                                   at::Tensor yLen, at::Tensor batchOffset, int64_t maxFLen,
                                                   int64_t blankIdx, int64_t opt, bool packedInput) {
  CHECK_INPUT(x);
  CHECK_INPUT(label);
  CHECK_INPUT(fLen);
  CHECK_INPUT(yLen);
  if (packedInput) CHECK_INPUT(batchOffset);
  return transducer_loss_cuda_forward(x, label, fLen, yLen, batchOffset, maxFLen, blankIdx, opt, packedInput);
}

at::Tensor transducer_loss_backward(at::Tensor x, at::Tensor lossGrad, at::Tensor alpha, at::Tensor beta,
                                       at::Tensor fLen, at::Tensor yLen, at::Tensor label,
                                       at::Tensor batchOffset, int64_t maxFLen, int64_t blankIdx, int64_t opt,
                                       bool fuseSoftmaxBackward, bool packedInput) {
  CHECK_INPUT(x);
  CHECK_INPUT(label);
  CHECK_INPUT(lossGrad);
  CHECK_INPUT(alpha);
  CHECK_INPUT(beta);
  CHECK_INPUT(fLen);
  CHECK_INPUT(yLen);
  if (packedInput) CHECK_INPUT(batchOffset);

  return transducer_loss_cuda_backward(x, lossGrad, alpha, beta, fLen, yLen, label, batchOffset, maxFLen, blankIdx, opt,
                                       fuseSoftmaxBackward, packedInput);
}

TORCH_LIBRARY_FRAGMENT(apex, m) {
  m.def("transducer_loss_forward(Tensor x, Tensor label, Tensor fLen, Tensor yLen, Tensor batchOffset, int maxFLen, "
        "int blankIdx, int opt, bool packedInput) -> Tensor[]");
  m.def("transducer_loss_backward(Tensor x, Tensor lossGrad, Tensor alpha, Tensor beta, Tensor fLen, Tensor yLen, "
        "Tensor label, Tensor batchOffset, int maxFLen, int blankIdx, int opt, bool fuseSoftmaxBackward, "
        "bool packedInput) -> Tensor");
}

TORCH_LIBRARY_IMPL(apex, CUDA, m) {
  m.impl("transducer_loss_forward", &transducer_loss_forward);
  m.impl("transducer_loss_backward", &transducer_loss_backward);
}
