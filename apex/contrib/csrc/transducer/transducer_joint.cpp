#include <ATen/ATen.h>
#include <torch/library.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> transducer_joint_cuda_forward(at::Tensor f, at::Tensor g, at::Tensor fLen, at::Tensor gLen,
                                                      at::Tensor batchOffset, int64_t packedBatch, int64_t opt,
                                                      bool packOutput, bool relu, bool dropout, double dropoutProb,
                                                      int64_t tileSize);

std::vector<at::Tensor> transducer_joint_cuda_backward(std::vector<at::Tensor> in, at::Tensor fLen, at::Tensor gLen,
                                                       at::Tensor batchOffset, int64_t maxFLen, int64_t maxGLen,
                                                       bool packOutput, double scale);

std::vector<at::Tensor> transducer_joint_forward(at::Tensor f, at::Tensor g, at::Tensor fLen, at::Tensor gLen,
                                                 at::Tensor batchOffset, int64_t packedBatch, int64_t opt,
                                                 bool packOutput, bool relu, bool dropout, double dropoutProb,
                                                 int64_t tileSize) {
  CHECK_INPUT(f);
  CHECK_INPUT(g);
  CHECK_INPUT(fLen);
  CHECK_INPUT(gLen);
  if (packOutput) CHECK_INPUT(batchOffset);
  return transducer_joint_cuda_forward(f, g, fLen, gLen, batchOffset, packedBatch, opt, packOutput, relu, dropout,
                                       dropoutProb, tileSize);
}

std::vector<at::Tensor> transducer_joint_backward(std::vector<at::Tensor> in, at::Tensor fLen, at::Tensor gLen,
                                                  at::Tensor batchOffset, int64_t maxFLen, int64_t maxGLen,
                                                  bool packOutput, double scale) {
  for (auto t : in) {
    CHECK_INPUT(t);
  }
  CHECK_INPUT(fLen);
  CHECK_INPUT(gLen);
  if (packOutput) CHECK_INPUT(batchOffset);
  return transducer_joint_cuda_backward(in, fLen, gLen, batchOffset, maxFLen, maxGLen, packOutput, scale);
}

TORCH_LIBRARY_FRAGMENT(apex, m) {
  m.def(
      "transducer_joint_forward(Tensor f, Tensor g, Tensor fLen, Tensor gLen, Tensor batchOffset, int packedBatch, "
      "int opt, bool packOutput, bool relu, bool dropout, float dropoutProb, int tileSize) -> Tensor[]");
  m.def(
      "transducer_joint_backward(Tensor[] input, Tensor fLen, Tensor gLen, Tensor batchOffset, int maxFLen, "
      "int maxGLen, bool packOutput, float scale) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(apex, CUDA, m) {
  m.impl("transducer_joint_forward", &transducer_joint_forward);
  m.impl("transducer_joint_backward", &transducer_joint_backward);
}
