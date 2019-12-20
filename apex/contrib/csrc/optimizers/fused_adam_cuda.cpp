#include <torch/extension.h>

// CUDA forward declaration
void fused_strided_check_finite(at::Tensor& noop, at::Tensor& p_copy, int stride, int clear_overflow_first);
void fused_adam_cuda(at::Tensor& noop, at::Tensor& p_in, at::Tensor& p_out, at::Tensor& p_copy, at::Tensor& m_in, at::Tensor& m_out, at::Tensor& v_in, at::Tensor& v_out, at::Tensor& g_in, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction, float decay);
void fused_adam_undo_cuda(at::Tensor& p_in, at::Tensor& p_out, at::Tensor& m_in, at::Tensor& m_out, at::Tensor& v_in, at::Tensor& v_out, at::Tensor& g_in, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction, float decay);
void fused_adam_cuda_mt(int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists, float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction, float decay);

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

void adam(
		at::Tensor& noop,
		at::Tensor& p_in, 
		at::Tensor& p_out, 
		at::Tensor& p_copy, 
		at::Tensor& m_in, 
		at::Tensor& m_out, 
		at::Tensor& v_in, 
		at::Tensor& v_out, 
		at::Tensor& g_in, 
		float lr, float beta1, float beta2, float eps, float grad_scale, 
		int step, int mode, int bias_correction, float decay
	 ) {
	CHECK_INPUT(p_in);
	CHECK_INPUT(p_out);
        int64_t num_elem = p_in.numel();
	AT_ASSERTM(num_elem == p_out.numel(), "number of elements in p_in and p_out tensors should be equal");
	if (p_copy.numel() > 0) CHECK_INPUT(p_copy);
	CHECK_INPUT(m_in);
	CHECK_INPUT(m_out);
	CHECK_INPUT(v_in);
	CHECK_INPUT(v_out);
	CHECK_INPUT(g_in);
        AT_ASSERTM(m_in.numel()   == num_elem, "number of elements in m_in and p_in tensors should be equal");
        AT_ASSERTM(m_out.numel()  == num_elem, "number of elements in m_out and p_in tensors should be equal");
        AT_ASSERTM(v_in.numel()   == num_elem, "number of elements in v_in and p_in tensors should be equal");
        AT_ASSERTM(v_out.numel()  == num_elem, "number of elements in v_out and p_in tensors should be equal");
        AT_ASSERTM(g_in.numel()   == num_elem, "number of elements in g_in and p_in tensors should be equal");
        AT_ASSERTM(p_copy.numel() == num_elem || p_copy.numel() == 0, "number of elements in p_copy and p_in tensors should be equal, or p_copy should be empty");

	fused_adam_cuda(noop, p_in, p_out, p_copy, m_in, m_out, v_in, v_out, g_in, lr, beta1, beta2, eps, grad_scale, step, mode, bias_correction, decay);
}

void adam_undo(
		at::Tensor& p_in, 
		at::Tensor& p_out, 
		at::Tensor& m_in, 
		at::Tensor& m_out, 
		at::Tensor& v_in, 
		at::Tensor& v_out, 
		at::Tensor& g_in, 
		float lr, float beta1, float beta2, float eps, float grad_scale, 
		int step, int mode, int bias_correction, float decay
	 ) {
	CHECK_INPUT(p_in);
	CHECK_INPUT(p_out);
        int64_t num_elem = p_in.numel();
	AT_ASSERTM(num_elem == p_out.numel(), "number of elements in p_in and p_out tensors should be equal");
	CHECK_INPUT(m_in);
	CHECK_INPUT(m_out);
	CHECK_INPUT(v_in);
	CHECK_INPUT(v_out);
	CHECK_INPUT(g_in);
        AT_ASSERTM(m_in.numel()   == num_elem, "number of elements in m_in and p_in tensors should be equal");
        AT_ASSERTM(m_out.numel()  == num_elem, "number of elements in m_out and p_in tensors should be equal");
        AT_ASSERTM(v_in.numel()   == num_elem, "number of elements in v_in and p_in tensors should be equal");
        AT_ASSERTM(v_out.numel()  == num_elem, "number of elements in v_out and p_in tensors should be equal");
        AT_ASSERTM(g_in.numel()   == num_elem, "number of elements in g_in and p_in tensors should be equal");

	fused_adam_undo_cuda(p_in, p_out, m_in, m_out, v_in, v_out, g_in, lr, beta1, beta2, eps, grad_scale, step, mode, bias_correction, decay);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("strided_check_finite", &strided_check_finite, "Strided finite check.");
        m.def("adam", &adam, "Adam optimized CUDA implementation.");
        m.def("adam_undo", &adam_undo, "Adam optimized CUDA implementation.");
        m.def("adam_mt", &fused_adam_cuda_mt, "Adam optimized multi tensor CUDA implementation.");
}
