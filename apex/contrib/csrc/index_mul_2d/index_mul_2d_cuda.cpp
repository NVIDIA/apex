#include <torch/torch.h>

#include <vector>
#include <cstdint>

void index_mul_2d_float_foward_cuda(at::Tensor &out,
                                 const at::Tensor &in1,
                                 const at::Tensor &in2,
                                 const at::Tensor &idx1);

void index_mul_2d_float_backward_cuda(at::Tensor &grad_in1,
                                   at::Tensor &grad_in2,
                                   const at::Tensor &grad_out,
                                   const at::Tensor &in1,
                                   const at::Tensor &in2,
                                   const at::Tensor &idx1);

void index_mul_2d_float_backward_backward_cuda(at::Tensor &grad_grad_out,
                                            at::Tensor &grad_in1,
                                            at::Tensor &grad_in2,
                                            const at::Tensor &grad_out,
                                            const at::Tensor &grad_grad_in1,
                                            const at::Tensor &grad_grad_in2,
                                            const at::Tensor &in1,
                                            const at::Tensor &in2,
                                            const at::Tensor &idx1);

void index_mul_2d_half_foward_cuda(at::Tensor &out,
                                const at::Tensor &in1,
                                const at::Tensor &in2,
                                const at::Tensor &idx1);

void index_mul_2d_half_backward_cuda(at::Tensor &grad_in1,
                                  at::Tensor &grad_in2,
                                  const at::Tensor &grad_out,
                                  const at::Tensor &in1,
                                  const at::Tensor &in2,
                                  const at::Tensor &idx1);

void index_mul_2d_half_backward_backward_cuda(at::Tensor &grad_grad_out,
                                           at::Tensor &grad_in1,
                                           at::Tensor &grad_in2,
                                           const at::Tensor &grad_out,
                                           const at::Tensor &grad_grad_in1,
                                           const at::Tensor &grad_grad_in2,
                                           const at::Tensor &in1,
                                           const at::Tensor &in2,
                                           const at::Tensor &idx1);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

void index_mul_2d_float_forward(
    at::Tensor &out,
    const at::Tensor &in1,
    const at::Tensor &in2,
    const at::Tensor &idx1)
{
  return index_mul_2d_float_foward_cuda(out, in1, in2, idx1);
}

void index_mul_2d_float_backward(
    at::Tensor &grad_in1,
    at::Tensor &grad_in2,
    const at::Tensor &grad_out,
    const at::Tensor &in1,
    const at::Tensor &in2,
    const at::Tensor &idx1)
{
  return index_mul_2d_float_backward_cuda(grad_in1, grad_in2, grad_out, in1, in2, idx1);
}

void index_mul_2d_float_backwrad_backward(
    at::Tensor &grad_grad_out,
    at::Tensor &grad_in1,
    at::Tensor &grad_in2,
    const at::Tensor &grad_out,
    const at::Tensor &grad_grad_in1,
    const at::Tensor &grad_grad_in2,
    const at::Tensor &in1,
    const at::Tensor &in2,
    const at::Tensor &idx1)
{
  return index_mul_2d_float_backward_backward_cuda(grad_grad_out, grad_in1, grad_in2, grad_out, grad_grad_in1, grad_grad_in2, in1, in2, idx1);
}

void index_mul_2d_half_forward(
    at::Tensor &out,
    const at::Tensor &in1,
    const at::Tensor &in2,
    const at::Tensor &idx1)
{
  return index_mul_2d_half_foward_cuda(out, in1, in2, idx1);
}

void index_mul_2d_half_backward(
    at::Tensor &grad_in1,
    at::Tensor &grad_in2,
    const at::Tensor &grad_out,
    const at::Tensor &in1,
    const at::Tensor &in2,
    const at::Tensor &idx1)
{
  return index_mul_2d_half_backward_cuda(grad_in1, grad_in2, grad_out, in1, in2, idx1);
}

void index_mul_2d_half_backwrad_backward(
    at::Tensor &grad_grad_out,
    at::Tensor &grad_in1,
    at::Tensor &grad_in2,
    const at::Tensor &grad_out,
    const at::Tensor &grad_grad_in1,
    const at::Tensor &grad_grad_in2,
    const at::Tensor &in1,
    const at::Tensor &in2,
    const at::Tensor &idx1)
{
  return index_mul_2d_half_backward_backward_cuda(grad_grad_out, grad_in1, grad_in2, grad_out, grad_grad_in1, grad_grad_in2, in1, in2, idx1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("float_forward", &index_mul_2d_float_forward,
        "index mul float calculation forward (CUDA)");
  m.def("float_backward", &index_mul_2d_float_backward,
        "index mul float calculation backward (CUDA)");
  m.def("float_backward_backward", &index_mul_2d_float_backwrad_backward,
        "index mul float calculation backward backward (CUDA)");
  m.def("half_forward", &index_mul_2d_half_forward,
        "index mul half calculation forward (CUDA)");
  m.def("half_backward", &index_mul_2d_half_backward,
        "index mul half calculation backward (CUDA)");
  m.def("half_backward_backward", &index_mul_2d_half_backwrad_backward,
        "index mul half calculation backward backward (CUDA)");
}

