#include <ATen/ATen.h>
#include <torch/library.h>

#include <cstdint>
#include <vector>

void index_mul_2d_float_foward_cuda(at::Tensor& out, const at::Tensor& in1, const at::Tensor& in2,
                                    const at::Tensor& idx1);

void index_mul_2d_float_backward_cuda(at::Tensor& grad_in1, at::Tensor& grad_in2, const at::Tensor& grad_out,
                                      const at::Tensor& in1, const at::Tensor& in2, const at::Tensor& idx1);

void index_mul_2d_float_backward_backward_cuda(at::Tensor& grad_grad_out, at::Tensor& grad_in1, at::Tensor& grad_in2,
                                               const at::Tensor& grad_out, const at::Tensor& grad_grad_in1,
                                               const at::Tensor& grad_grad_in2, const at::Tensor& in1,
                                               const at::Tensor& in2, const at::Tensor& idx1);

void index_mul_2d_half_foward_cuda(at::Tensor& out, const at::Tensor& in1, const at::Tensor& in2,
                                   const at::Tensor& idx1);

void index_mul_2d_half_backward_cuda(at::Tensor& grad_in1, at::Tensor& grad_in2, const at::Tensor& grad_out,
                                     const at::Tensor& in1, const at::Tensor& in2, const at::Tensor& idx1);

void index_mul_2d_half_backward_backward_cuda(at::Tensor& grad_grad_out, at::Tensor& grad_in1, at::Tensor& grad_in2,
                                              const at::Tensor& grad_out, const at::Tensor& grad_grad_in1,
                                              const at::Tensor& grad_grad_in2, const at::Tensor& in1,
                                              const at::Tensor& in2, const at::Tensor& idx1);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

void index_mul_2d_float_forward(at::Tensor& out, const at::Tensor& in1, const at::Tensor& in2, const at::Tensor& idx1) {
  return index_mul_2d_float_foward_cuda(out, in1, in2, idx1);
}

void index_mul_2d_float_backward(at::Tensor& grad_in1, at::Tensor& grad_in2, const at::Tensor& grad_out,
                                 const at::Tensor& in1, const at::Tensor& in2, const at::Tensor& idx1) {
  return index_mul_2d_float_backward_cuda(grad_in1, grad_in2, grad_out, in1, in2, idx1);
}

void index_mul_2d_float_backwrad_backward(at::Tensor& grad_grad_out, at::Tensor& grad_in1, at::Tensor& grad_in2,
                                          const at::Tensor& grad_out, const at::Tensor& grad_grad_in1,
                                          const at::Tensor& grad_grad_in2, const at::Tensor& in1, const at::Tensor& in2,
                                          const at::Tensor& idx1) {
  return index_mul_2d_float_backward_backward_cuda(grad_grad_out, grad_in1, grad_in2, grad_out, grad_grad_in1,
                                                   grad_grad_in2, in1, in2, idx1);
}

void index_mul_2d_half_forward(at::Tensor& out, const at::Tensor& in1, const at::Tensor& in2, const at::Tensor& idx1) {
  return index_mul_2d_half_foward_cuda(out, in1, in2, idx1);
}

void index_mul_2d_half_backward(at::Tensor& grad_in1, at::Tensor& grad_in2, const at::Tensor& grad_out,
                                const at::Tensor& in1, const at::Tensor& in2, const at::Tensor& idx1) {
  return index_mul_2d_half_backward_cuda(grad_in1, grad_in2, grad_out, in1, in2, idx1);
}

void index_mul_2d_half_backwrad_backward(at::Tensor& grad_grad_out, at::Tensor& grad_in1, at::Tensor& grad_in2,
                                         const at::Tensor& grad_out, const at::Tensor& grad_grad_in1,
                                         const at::Tensor& grad_grad_in2, const at::Tensor& in1, const at::Tensor& in2,
                                         const at::Tensor& idx1) {
  return index_mul_2d_half_backward_backward_cuda(grad_grad_out, grad_in1, grad_in2, grad_out, grad_grad_in1,
                                                  grad_grad_in2, in1, in2, idx1);
}

void index_mul_2d_float_forward_dispatch(const at::Tensor& out, const at::Tensor& in1, const at::Tensor& in2,
                                         const at::Tensor& idx1) {
  at::Tensor out_arg = out;
  index_mul_2d_float_forward(out_arg, in1, in2, idx1);
}

void index_mul_2d_float_backward_dispatch(const at::Tensor& grad_in1, const at::Tensor& grad_in2,
                                          const at::Tensor& grad_out, const at::Tensor& in1, const at::Tensor& in2,
                                          const at::Tensor& idx1) {
  at::Tensor grad_in1_arg = grad_in1;
  at::Tensor grad_in2_arg = grad_in2;
  index_mul_2d_float_backward(grad_in1_arg, grad_in2_arg, grad_out, in1, in2, idx1);
}

void index_mul_2d_float_backward_backward_dispatch(const at::Tensor& grad_grad_out, const at::Tensor& grad_in1,
                                                   const at::Tensor& grad_in2, const at::Tensor& grad_out,
                                                   const at::Tensor& grad_grad_in1, const at::Tensor& grad_grad_in2,
                                                   const at::Tensor& in1, const at::Tensor& in2,
                                                   const at::Tensor& idx1) {
  at::Tensor grad_grad_out_arg = grad_grad_out;
  at::Tensor grad_in1_arg = grad_in1;
  at::Tensor grad_in2_arg = grad_in2;
  index_mul_2d_float_backwrad_backward(grad_grad_out_arg, grad_in1_arg, grad_in2_arg, grad_out, grad_grad_in1,
                                       grad_grad_in2, in1, in2, idx1);
}

void index_mul_2d_half_forward_dispatch(const at::Tensor& out, const at::Tensor& in1, const at::Tensor& in2,
                                        const at::Tensor& idx1) {
  at::Tensor out_arg = out;
  index_mul_2d_half_forward(out_arg, in1, in2, idx1);
}

void index_mul_2d_half_backward_dispatch(const at::Tensor& grad_in1, const at::Tensor& grad_in2,
                                         const at::Tensor& grad_out, const at::Tensor& in1, const at::Tensor& in2,
                                         const at::Tensor& idx1) {
  at::Tensor grad_in1_arg = grad_in1;
  at::Tensor grad_in2_arg = grad_in2;
  index_mul_2d_half_backward(grad_in1_arg, grad_in2_arg, grad_out, in1, in2, idx1);
}

void index_mul_2d_half_backward_backward_dispatch(const at::Tensor& grad_grad_out, const at::Tensor& grad_in1,
                                                  const at::Tensor& grad_in2, const at::Tensor& grad_out,
                                                  const at::Tensor& grad_grad_in1, const at::Tensor& grad_grad_in2,
                                                  const at::Tensor& in1, const at::Tensor& in2,
                                                  const at::Tensor& idx1) {
  at::Tensor grad_grad_out_arg = grad_grad_out;
  at::Tensor grad_in1_arg = grad_in1;
  at::Tensor grad_in2_arg = grad_in2;
  index_mul_2d_half_backwrad_backward(grad_grad_out_arg, grad_in1_arg, grad_in2_arg, grad_out, grad_grad_in1,
                                      grad_grad_in2, in1, in2, idx1);
}

TORCH_LIBRARY_FRAGMENT(apex, m) {
  m.def("index_mul_2d_float_forward(Tensor(a!) out, Tensor in1, Tensor in2, Tensor idx1) -> ()");
  m.def(
      "index_mul_2d_float_backward(Tensor(a!) grad_in1, Tensor(b!) grad_in2, Tensor grad_out, Tensor in1, "
      "Tensor in2, Tensor idx1) -> ()");
  m.def(
      "index_mul_2d_float_backward_backward(Tensor(a!) grad_grad_out, Tensor(b!) grad_in1, "
      "Tensor(c!) grad_in2, Tensor grad_out, Tensor grad_grad_in1, Tensor grad_grad_in2, Tensor in1, "
      "Tensor in2, Tensor idx1) -> ()");
  m.def("index_mul_2d_half_forward(Tensor(a!) out, Tensor in1, Tensor in2, Tensor idx1) -> ()");
  m.def(
      "index_mul_2d_half_backward(Tensor(a!) grad_in1, Tensor(b!) grad_in2, Tensor grad_out, Tensor in1, "
      "Tensor in2, Tensor idx1) -> ()");
  m.def(
      "index_mul_2d_half_backward_backward(Tensor(a!) grad_grad_out, Tensor(b!) grad_in1, "
      "Tensor(c!) grad_in2, Tensor grad_out, Tensor grad_grad_in1, Tensor grad_grad_in2, Tensor in1, "
      "Tensor in2, Tensor idx1) -> ()");
}

TORCH_LIBRARY_IMPL(apex, CUDA, m) {
  m.impl("index_mul_2d_float_forward", &index_mul_2d_float_forward_dispatch);
  m.impl("index_mul_2d_float_backward", &index_mul_2d_float_backward_dispatch);
  m.impl("index_mul_2d_float_backward_backward", &index_mul_2d_float_backward_backward_dispatch);
  m.impl("index_mul_2d_half_forward", &index_mul_2d_half_forward_dispatch);
  m.impl("index_mul_2d_half_backward", &index_mul_2d_half_backward_dispatch);
  m.impl("index_mul_2d_half_backward_backward", &index_mul_2d_half_backward_backward_dispatch);
}
