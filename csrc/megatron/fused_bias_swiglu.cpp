#include <torch/extension.h>

// Function declarations
torch::Tensor fused_bias_swiglu_forward(torch::Tensor input, torch::Tensor bias);
torch::Tensor fused_bias_swiglu_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor bias);

// Register functions for PyTorch extension
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_bias_swiglu_forward, "Fused Bias SwiGLU Forward (CUDA)");
    m.def("backward", &fused_bias_swiglu_backward, "Fused Bias SwiGLU Backward (CUDA)");
}