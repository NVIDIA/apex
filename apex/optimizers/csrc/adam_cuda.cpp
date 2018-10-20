#include <torch/torch.h>

// CUDA forward declaration
void adam_cuda(at::Tensor & p, at::Tensor & p_copy, at::Tensor & m, at::Tensor & v, at::Tensor & g, float lr, float beta1, float beta2, float eps, int step, int mode);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// C++ interface
void adam(at::Tensor & p, at::Tensor & p_copy, at::Tensor & m, at::Tensor & v, at::Tensor & g, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode) {
        CHECK_INPUT(p);
        CHECK_INPUT(p_copy);
        CHECK_INPUT(m);
        CHECK_INPUT(v);
        CHECK_INPUT(g);
        return adam_cuda(p, p_copy, m, v, g, lr, beta1, beta2, eps, step, mode);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("adam", &adam, "Adam optimized CUDA implementation.");
}
