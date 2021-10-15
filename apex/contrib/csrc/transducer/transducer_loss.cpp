#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> transducer_loss_cuda_forward(
    torch::Tensor x,
    torch::Tensor label,
    torch::Tensor audLen,
    torch::Tensor txtLen,
    torch::Tensor batchOffset,
    int maxFLen,
    int blankIdx,
    int opt,
    bool packedInput);

torch::Tensor transducer_loss_cuda_backward(
    torch::Tensor x,
    torch::Tensor lossGrad,
    torch::Tensor alpha,
    torch::Tensor beta,
    torch::Tensor audLen,
    torch::Tensor txtLen,
    torch::Tensor label,
    torch::Tensor batchOffset,
    int maxFLen,
    int blankIdx,
    int opt,
    bool fuseSoftmaxBackward,
    bool packedInput);


std::vector<torch::Tensor> transducer_loss_forward(
    torch::Tensor x,
    torch::Tensor label,
    torch::Tensor fLen,
    torch::Tensor yLen,
    torch::Tensor batchOffset,
    int maxFLen,
    int blankIdx,
    int opt,
    bool packedInput
    ) {

    CHECK_INPUT(x);
    CHECK_INPUT(label);
    CHECK_INPUT(fLen);
    CHECK_INPUT(yLen);
    if (packedInput)
        CHECK_INPUT(batchOffset);
    return transducer_loss_cuda_forward(
        x, 
        label, 
        fLen, 
        yLen, 
        batchOffset,
        maxFLen,
        blankIdx, 
        opt,
        packedInput);
}

torch::Tensor transducer_loss_backward(
    torch::Tensor x,
    torch::Tensor lossGrad,
    torch::Tensor alpha,
    torch::Tensor beta,
    torch::Tensor fLen,
    torch::Tensor yLen,
    torch::Tensor label,
    torch::Tensor batchOffset,
    int maxFLen,
    int blankIdx,
    int opt,
    bool fuseSoftmaxBackward,
    bool packedInput){

    CHECK_INPUT(x);
    CHECK_INPUT(label);
    CHECK_INPUT(lossGrad);
    CHECK_INPUT(alpha);
    CHECK_INPUT(beta);
    CHECK_INPUT(fLen);
    CHECK_INPUT(yLen);
    if (packedInput)
        CHECK_INPUT(batchOffset);

    return transducer_loss_cuda_backward(
        x,
        lossGrad,
        alpha,
        beta,
        fLen,
        yLen,
        label,
        batchOffset,
        maxFLen,
        blankIdx,
        opt,
        fuseSoftmaxBackward,
        packedInput);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &transducer_loss_forward, "transducer loss forward (CUDA)");
  m.def("backward", &transducer_loss_backward, "transducer loss backward (CUDA)");
}
