#include <torch/extension.h>

// CUDA forward declaration
void fused_adam_cuda(at::Tensor & p, at::Tensor & p_copy, at::Tensor & m, at::Tensor & v, at::Tensor & g, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction, float decay);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// C++ interface
void adam(at::Tensor & p, at::Tensor & p_copy, at::Tensor & m, at::Tensor & v, at::Tensor & g, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction, float decay) {
        CHECK_INPUT(p)
        if (p_copy.numel() > 0) CHECK_INPUT(p_copy);
        CHECK_INPUT(m);
        CHECK_INPUT(v);
        CHECK_INPUT(g);
        int64_t num_elem = p.numel();
        AT_ASSERTM(m.numel() == num_elem, "number of elements in m and p tensors should be equal");
        AT_ASSERTM(v.numel() == num_elem, "number of elements in v and p tensors should be equal");
        AT_ASSERTM(g.numel() == num_elem, "number of elements in g and p tensors should be equal");
        AT_ASSERTM(p_copy.numel() == num_elem || p_copy.numel() == 0, "number of elements in p_copy and p tensors should be equal, or p_copy should be empty");

        fused_adam_cuda(p, p_copy, m, v, g, lr, beta1, beta2, eps, grad_scale, step, mode, bias_correction, decay);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("adam", &adam, "Adam optimized CUDA implementation.");
}
