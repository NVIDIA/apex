#include <torch/extension.h>

// CUDA forward declaration
void fused_strided_check_finite(at::Tensor & noop, at::Tensor & p_copy, int stride, int clear_overflow_first);

void fused_adam_cuda_no_overflow_check(at::Tensor & p_in, at::Tensor & p_out, at::Tensor & p_copy, at::Tensor & m_in, at::Tensor & m_out, at::Tensor & v_in, at::Tensor & v_out, at::Tensor & g, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction, float decay);
void fused_adam_cuda(at::Tensor & p, at::Tensor & p_copy, at::Tensor & m, at::Tensor & v, at::Tensor & g, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction, float decay);
void fused_adam_undo_cuda(at::Tensor & p, at::Tensor & m, at::Tensor & v, at::Tensor & g, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction, float decay);

void fused_adam_cuda_mt(int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction, float decay);
void fused_adam_undo_cuda_mt(int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction, float decay);


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// C++ interface
void strided_check_finite(
		at::Tensor& noop,
		at::Tensor& p_copy,
		int stride,
		int clear_overflow_first
	 ) {
	CHECK_INPUT(p_copy);
	fused_strided_check_finite(noop, p_copy, stride, clear_overflow_first);
}
void adam_no_overflow_chedck(at::Tensor & p_in, at::Tensor & p_out, at::Tensor & p_copy, at::Tensor & m_in, at::Tensor & m_out, at::Tensor & v_in, at::Tensor & v_out, at::Tensor & g, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction, float decay) {
        CHECK_INPUT(p_in);
        CHECK_INPUT(p_out);
        if (p_copy.numel() > 0) CHECK_INPUT(p_copy);
        CHECK_INPUT(m_in);
        CHECK_INPUT(m_out);
        CHECK_INPUT(v_in);
        CHECK_INPUT(v_out);
        CHECK_INPUT(g);
        int64_t num_elem = p_in.numel();
        AT_ASSERTM(m_in.numel() == num_elem, "number of elements in m_in and p_in tensors should be equal");
        AT_ASSERTM(m_out.numel() == num_elem, "number of elements in m_out and p_in tensors should be equal");
        AT_ASSERTM(v_in.numel() == num_elem, "number of elements in v_in and p_in tensors should be equal");
        AT_ASSERTM(v_out.numel() == num_elem, "number of elements in v_out and p_in tensors should be equal");
        AT_ASSERTM(p_out.numel() == num_elem, "number of elements in p_out and p_in tensors should be equal");
        AT_ASSERTM(g.numel() == num_elem, "number of elements in g and p tensors should be equal");
        AT_ASSERTM(p_copy.numel() == num_elem || p_copy.numel() == 0, "number of elements in p_copy and p tensors should be equal, or p_copy should be empty");

        fused_adam_cuda_no_overflow_check(p_in, p_out, p_copy, m_in, m_out, v_in, v_out, g, lr, beta1, beta2, eps, grad_scale, step, mode, bias_correction, decay);
}
void adam(at::Tensor & p, at::Tensor & p_copy, at::Tensor & m, at::Tensor & v, at::Tensor & g, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction, float decay) {
        CHECK_INPUT(p);
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
void adam_undo(at::Tensor & p, at::Tensor & m, at::Tensor & v, at::Tensor & g, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction, float decay) {
        CHECK_INPUT(p);
        CHECK_INPUT(m);
        CHECK_INPUT(v);
        CHECK_INPUT(g);
        int64_t num_elem = p.numel();
        AT_ASSERTM(m.numel() == num_elem, "number of elements in m and p tensors should be equal");
        AT_ASSERTM(v.numel() == num_elem, "number of elements in v and p tensors should be equal");
        AT_ASSERTM(g.numel() == num_elem, "number of elements in g and p tensors should be equal");

        fused_adam_undo_cuda(p, m, v, g, lr, beta1, beta2, eps, grad_scale, step, mode, bias_correction, decay);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("strided_check_finite", &strided_check_finite, "Strided finite check.");
        m.def("adam_no_overflow_check", &adam, "Adam optimized CUDA implementation.");
        m.def("adam", &adam, "Adam optimized CUDA implementation.");
        m.def("adam_undo", &adam_undo, "Undo function for Adam optimized CUDA implementation.");
        m.def("adam_mt", &fused_adam_cuda_mt, "Multi tensor Adam optimized CUDA implementation.");
        m.def("adam_undo_mt", &fused_adam_undo_cuda_mt, "Multi tensor undo function for Adam optimized CUDA implementation.");
}
