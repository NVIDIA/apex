#include <torch/extension.h>

// CUDA forward declaration
void radix_decomp_cuda(at::Tensor& noop, at::Tensor& p_orig, at::Tensor& p_decomp, int largest_digit, int smallest_digit, double radix,	int clear_overflow_first);
void radix_comp_cuda(at::Tensor& noop, at::Tensor& p_orig, at::Tensor& p_decomp, int largest_digit, int smallest_digit, double radix,	int clear_overflow_first);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// C++ interface
void radix_decomp(
		at::Tensor& noop,
		at::Tensor& p_orig, 
		at::Tensor& p_decomp,
	        int largest_digit, int smallest_digit, double radix, int clear_overflow_first
	 ) {
	CHECK_INPUT(p_orig);
	CHECK_INPUT(p_decomp);
	AT_ASSERTM(largest_digit > smallest_digit, "largest_digit should be > smallest_digit");
	AT_ASSERTM(radix > 1.0, "radix should be > 1");
        int64_t orig_num_elem = p_orig.numel();
	int64_t n = largest_digit - smallest_digit + 1;
	int64_t decomp_num_elem = p_decomp.numel();
	AT_ASSERTM(orig_num_elem*n == decomp_num_elem, "p_orig.numel()*n and p_decomp.numel() should be equal");
	radix_decomp_cuda(noop, p_orig, p_decomp, largest_digit, smallest_digit, radix, clear_overflow_first);
}

void radix_comp(
		at::Tensor& noop,
		at::Tensor& p_orig, 
		at::Tensor& p_decomp,
	        int largest_digit, int smallest_digit, double radix, int clear_overflow_first
	 ) {
	CHECK_INPUT(p_orig);
	CHECK_INPUT(p_decomp);
	AT_ASSERTM(largest_digit > smallest_digit, "largest_digit should be > smallest_digit");
	AT_ASSERTM(radix > 1.0, "radix should be > 1");
        int64_t orig_num_elem = p_orig.numel();
	int64_t n = largest_digit - smallest_digit + 1;
	int64_t decomp_num_elem = p_decomp.numel();
	AT_ASSERTM(orig_num_elem*n == decomp_num_elem, "p_orig.numel()*n and p_decomp.numel() should be equal");
	radix_comp_cuda(noop, p_orig, p_decomp, largest_digit, smallest_digit, radix, clear_overflow_first);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("radix_decomp", &radix_decomp, "Radix decomposition CUDA implementation.");
	m.def("radix_comp", &radix_comp, "Composition of previously radix decomposed tensor.");
}
