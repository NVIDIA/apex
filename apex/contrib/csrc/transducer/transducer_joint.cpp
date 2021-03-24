#include <torch/extension.h>
#include <ATen/Functions.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor transducer_joint_cuda_forward(
    torch::Tensor f,
    torch::Tensor g,
    torch::Tensor fLen,
    torch::Tensor gLen,
    torch::Tensor batchOffset,
    int64_t packedBatch,
    int opt,
    bool packOutput,
    int tileSize);


std::vector<torch::Tensor> transducer_joint_cuda_backward(
    torch::Tensor grad,
    torch::Tensor fLen,
    torch::Tensor gLen,
    torch::Tensor batchOffset,
    int maxFLen,
    int maxGLen,
    bool packOutput);

torch::Tensor transducer_joint_forward(
    torch::Tensor f,
    torch::Tensor g,
    torch::Tensor fLen,
    torch::Tensor gLen,
    torch::Tensor batchOffset,
    int64_t packedBatch,
    int opt,
    bool packOutput,
    int tileSize) {
    CHECK_INPUT(f);
    CHECK_INPUT(g);
    CHECK_INPUT(fLen);
    CHECK_INPUT(gLen);
    if (packOutput)
        CHECK_INPUT(batchOffset);
    return transducer_joint_cuda_forward(
        f, 
        g, 
        fLen, 
        gLen,
        batchOffset,
        packedBatch,
        opt,
        packOutput,
        tileSize);
}

std::vector<torch::Tensor> transducer_joint_backward(
    torch::Tensor grad,
    torch::Tensor fLen,
    torch::Tensor gLen,
    torch::Tensor batchOffset,
    int maxFLen,
    int maxGLen,
    bool packOutput) {
    CHECK_INPUT(grad);
    CHECK_INPUT(fLen);
    CHECK_INPUT(gLen);
    if (packOutput)
        CHECK_INPUT(batchOffset);
    return transducer_joint_cuda_backward(
        grad, 
        fLen, 
        gLen,
        batchOffset,
        maxFLen,
        maxGLen,
        packOutput);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &transducer_joint_forward, "transducer joint forward (CUDA)");
  m.def("backward", &transducer_joint_backward, "transducer joint backward (CUDA)");
}