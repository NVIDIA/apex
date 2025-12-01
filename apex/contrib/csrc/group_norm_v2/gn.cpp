#include "gn.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace group_norm_v2 {

torch::Tensor gn(torch::Tensor x, torch::Tensor w, torch::Tensor b, float eps, bool silu, int num_groups,
                 std::optional<torch::Tensor> mean_var_out, int sm_margin) {
  if (w.dtype() != b.dtype() || (mean_var_out.has_value() && mean_var_out->dtype() != torch::kFloat32)) {
    throw std::invalid_argument("gn dtype mismatch");
  }
  torch::Tensor out = torch::empty_like(x);
  float* ptr_mean_var_out = mean_var_out.has_value() ? mean_var_out->data_ptr<float>() : nullptr;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  int device_id = at::cuda::getCurrentCUDAStream().device().index();
  group_norm_v2::Meta meta;
  if (x.dtype() == torch::kHalf && w.dtype() == torch::kHalf) {
    group_norm_v2::gn_cuda((half*)out.data_ptr(), (half*)x.data_ptr(), (half*)w.data_ptr(), (half*)b.data_ptr(), eps,
                           silu, x.size(0), x.size(2) * x.size(3), num_groups, x.size(1) / num_groups, ptr_mean_var_out,
                           nullptr, nullptr, sm_margin, stream, device_id, &meta, true);
  } else if (x.dtype() == torch::kBFloat16 && w.dtype() == torch::kBFloat16) {
    group_norm_v2::gn_cuda((__nv_bfloat16*)out.data_ptr(), (__nv_bfloat16*)x.data_ptr(), (__nv_bfloat16*)w.data_ptr(),
                           (__nv_bfloat16*)b.data_ptr(), eps, silu, x.size(0), x.size(2) * x.size(3), num_groups,
                           x.size(1) / num_groups, ptr_mean_var_out, nullptr, nullptr, sm_margin, stream, device_id,
                           &meta, true);
  } else {
    throw std::invalid_argument("gn only supports half or bfloat16 input and weight");
  }
  torch::Tensor red_buffer =
      torch::empty({meta.red_buffer_size}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  thread_local torch::Tensor barrier;
  if (barrier.size(0) < meta.barrier_size) {
    barrier = torch::zeros({meta.barrier_size}, torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA));
  }
  if (x.dtype() == torch::kHalf && w.dtype() == torch::kHalf) {
    group_norm_v2::gn_cuda((half*)out.data_ptr(), (half*)x.data_ptr(), (half*)w.data_ptr(), (half*)b.data_ptr(), eps,
                           silu, x.size(0), x.size(2) * x.size(3), num_groups, x.size(1) / num_groups, ptr_mean_var_out,
                           red_buffer.data_ptr<float>(), barrier.data_ptr<unsigned>(), sm_margin, stream, device_id,
                           nullptr, false);
  } else if (x.dtype() == torch::kBFloat16 && w.dtype() == torch::kBFloat16) {
    group_norm_v2::gn_cuda((__nv_bfloat16*)out.data_ptr(), (__nv_bfloat16*)x.data_ptr(), (__nv_bfloat16*)w.data_ptr(),
                           (__nv_bfloat16*)b.data_ptr(), eps, silu, x.size(0), x.size(2) * x.size(3), num_groups,
                           x.size(1) / num_groups, ptr_mean_var_out, red_buffer.data_ptr<float>(),
                           barrier.data_ptr<unsigned>(), sm_margin, stream, device_id, nullptr, false);
  } else {
    throw std::invalid_argument("gn only supports half or bfloat16 input and weight");
  }
  return out;
}

auto gn_bwd(torch::Tensor grad_output, torch::Tensor x, torch::Tensor w, torch::Tensor b, torch::Tensor mean_var,
            float eps, bool silu, int num_groups, int sm_margin) {
  if (w.dtype() != b.dtype() || x.dtype() != grad_output.dtype() || mean_var.dtype() != torch::kFloat32) {
    throw std::invalid_argument("gn_bwd dtype mismatch");
  }
  torch::Tensor grad_input = torch::empty_like(x);
  torch::Tensor grad_weight = torch::empty_like(w);
  torch::Tensor grad_bias = torch::empty_like(w);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  int device_id = at::cuda::getCurrentCUDAStream().device().index();
  group_norm_v2::Meta meta;
  if (x.dtype() == torch::kHalf && w.dtype() == torch::kHalf) {
    group_norm_v2::gn_bwd_cuda((half*)grad_input.data_ptr(), (half*)grad_weight.data_ptr(), (half*)grad_bias.data_ptr(),
                               (half*)grad_output.data_ptr(), (half*)x.data_ptr(), (half*)w.data_ptr(),
                               (half*)b.data_ptr(), mean_var.data_ptr<float>(), eps, silu, x.size(0),
                               x.size(2) * x.size(3), num_groups, x.size(1) / num_groups, nullptr, nullptr, sm_margin,
                               stream, device_id, &meta, true);
  } else if (x.dtype() == torch::kBFloat16 && w.dtype() == torch::kBFloat16) {
    group_norm_v2::gn_bwd_cuda((__nv_bfloat16*)grad_input.data_ptr(), (__nv_bfloat16*)grad_weight.data_ptr(),
                               (__nv_bfloat16*)grad_bias.data_ptr(), (__nv_bfloat16*)grad_output.data_ptr(),
                               (__nv_bfloat16*)x.data_ptr(), (__nv_bfloat16*)w.data_ptr(), (__nv_bfloat16*)b.data_ptr(),
                               mean_var.data_ptr<float>(), eps, silu, x.size(0), x.size(2) * x.size(3), num_groups,
                               x.size(1) / num_groups, nullptr, nullptr, sm_margin, stream, device_id, &meta, true);
  } else {
    throw std::invalid_argument("gn only supports half or bfloat16 input and weight");
  }
  torch::Tensor red_buffer =
      torch::empty({meta.red_buffer_size}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
  thread_local torch::Tensor barrier;
  if (barrier.size(0) < meta.barrier_size) {
    barrier = torch::zeros({meta.barrier_size}, torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA));
  }
  if (x.dtype() == torch::kHalf && w.dtype() == torch::kHalf) {
    group_norm_v2::gn_bwd_cuda((half*)grad_input.data_ptr(), (half*)grad_weight.data_ptr(), (half*)grad_bias.data_ptr(),
                               (half*)grad_output.data_ptr(), (half*)x.data_ptr(), (half*)w.data_ptr(),
                               (half*)b.data_ptr(), mean_var.data_ptr<float>(), eps, silu, x.size(0),
                               x.size(2) * x.size(3), num_groups, x.size(1) / num_groups, red_buffer.data_ptr<float>(),
                               barrier.data_ptr<unsigned>(), sm_margin, stream, device_id, nullptr, false);
  } else if (x.dtype() == torch::kBFloat16 && w.dtype() == torch::kBFloat16) {
    group_norm_v2::gn_bwd_cuda((__nv_bfloat16*)grad_input.data_ptr(), (__nv_bfloat16*)grad_weight.data_ptr(),
                               (__nv_bfloat16*)grad_bias.data_ptr(), (__nv_bfloat16*)grad_output.data_ptr(),
                               (__nv_bfloat16*)x.data_ptr(), (__nv_bfloat16*)w.data_ptr(), (__nv_bfloat16*)b.data_ptr(),
                               mean_var.data_ptr<float>(), eps, silu, x.size(0), x.size(2) * x.size(3), num_groups,
                               x.size(1) / num_groups, red_buffer.data_ptr<float>(), barrier.data_ptr<unsigned>(),
                               sm_margin, stream, device_id, nullptr, false);
  } else {
    throw std::invalid_argument("gn only supports half or bfloat16 input and weight");
  }
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

}  // namespace group_norm_v2

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gn", &group_norm_v2::gn, py::arg("x"), py::arg("w"), py::arg("b"), py::arg("eps"), py::arg("silu"),
        py::arg("num_groups"), py::arg("mean_var_out") = py::none(), py::arg("sm_margin") = 0, "");
  m.def("gn_bwd", &group_norm_v2::gn_bwd, py::arg("grad_output"), py::arg("x"), py::arg("w"), py::arg("b"),
        py::arg("mean_var"), py::arg("eps"), py::arg("silu"), py::arg("num_groups"), py::arg("sm_margin") = 0, "");
}
