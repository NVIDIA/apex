#include <torch/torch.h>

// Ideally, I'd like to call this file "weight_norm.cu" and put the interface and the implementation
// here, but I can't make nvcc play well with torch.h.  For now, use a layer of indirection 
// and separate .cu implementation files.

// If we want everything to be part of "apex._C", we need all the interface functions defined 
// in this file, or linker will complain about "multiple definitions of PyInit".
// TODO:  multiple modules?

// TODO:  modify fwd+bwd calls to return a tuple of Tensors.  This will require changing the
// Python client code as well.  For now, get things working with the same Python-side API.

void weight_norm_fwd_cuda
  (const at::Tensor& w,
   const at::Tensor& norms,
   const at::Tensor& v,
   const at::Tensor& g,
   int dim);

void weight_norm_fwd
  (at::Tensor w, 
   at::Tensor norms, 
   at::Tensor v, 
   at::Tensor g, 
   int dim) 
{
  weight_norm_fwd_cuda(w, norms, v, g, dim);
}

void weight_norm_bwd_cuda
  (const at::Tensor& pLpv,
   const at::Tensor& pLpg,
   const at::Tensor& pLpw,
   const at::Tensor& savedv,
   const at::Tensor& savedg,
   const at::Tensor& savedNorms,
   int dim);

void weight_norm_bwd
  (at::Tensor pLpv, 
   at::Tensor pLpg, 
   at::Tensor pLpw, 
   at::Tensor savedv, 
   at::Tensor savedg,
   at::Tensor savedNorms,
   int dim)
{
  weight_norm_bwd_cuda(pLpv, pLpg, pLpw, savedv, savedg, savedNorms, dim);
}

void scale_check_overflow_cuda
  (const at::Tensor& d_grads, 
   float scale,
   const at::Tensor& d_buf);

void scale_check_overflow
  (at::Tensor grads,
   float scale,
   at::Tensor overflow_buf)
{ 
  AT_CHECK(grads.type().is_cuda(), "x must be a CUDA tensor");
  AT_CHECK(overflow_buf.type().is_cuda(), "y must be a CUDA tensor");
  scale_check_overflow_cuda(grads, scale, overflow_buf);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("weight_norm_fwd", &weight_norm_fwd, "Fused weight norm, forward pass");
  m.def("weight_norm_bwd", &weight_norm_bwd, "Fused weight norm, backward pass");
  m.def("scale_check_overflow", &scale_check_overflow, "Fused overflow check + scale for FP32 tensors");
}
