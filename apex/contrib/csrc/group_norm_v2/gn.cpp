#include "gn.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/library.h>

#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace group_norm_v2 {

at::Tensor gn(at::Tensor x, at::Tensor w, at::Tensor b, float eps, bool silu, int num_groups,
              std::optional<at::Tensor> mean_var_out, int sm_margin) {
  if (w.dtype() != b.dtype() || (mean_var_out.has_value() && mean_var_out->dtype() != at::kFloat)) {
    throw std::invalid_argument("gn dtype mismatch");
  }
  at::Tensor out = at::empty_like(x);
  float* ptr_mean_var_out = mean_var_out.has_value() ? mean_var_out->data_ptr<float>() : nullptr;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  int device_id = at::cuda::getCurrentCUDAStream().device().index();
  group_norm_v2::Meta meta;
  if (x.dtype() == at::kHalf && w.dtype() == at::kHalf) {
    group_norm_v2::gn_cuda((half*)out.data_ptr(), (half*)x.data_ptr(), (half*)w.data_ptr(), (half*)b.data_ptr(), eps,
                           silu, x.size(0), x.size(2) * x.size(3), num_groups, x.size(1) / num_groups, ptr_mean_var_out,
                           nullptr, nullptr, sm_margin, stream, device_id, &meta, true);
  } else if (x.dtype() == at::kBFloat16 && w.dtype() == at::kBFloat16) {
    group_norm_v2::gn_cuda((__nv_bfloat16*)out.data_ptr(), (__nv_bfloat16*)x.data_ptr(), (__nv_bfloat16*)w.data_ptr(),
                           (__nv_bfloat16*)b.data_ptr(), eps, silu, x.size(0), x.size(2) * x.size(3), num_groups,
                           x.size(1) / num_groups, ptr_mean_var_out, nullptr, nullptr, sm_margin, stream, device_id,
                           &meta, true);
  } else {
    throw std::invalid_argument("gn only supports half or bfloat16 input and weight");
  }
  at::Tensor red_buffer = at::empty({meta.red_buffer_size}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  thread_local at::Tensor barrier;
  if (barrier.size(0) < meta.barrier_size) {
    barrier = at::zeros({meta.barrier_size}, at::TensorOptions().dtype(at::ScalarType::UInt32).device(at::kCUDA));
  }
  if (x.dtype() == at::kHalf && w.dtype() == at::kHalf) {
    group_norm_v2::gn_cuda((half*)out.data_ptr(), (half*)x.data_ptr(), (half*)w.data_ptr(), (half*)b.data_ptr(), eps,
                           silu, x.size(0), x.size(2) * x.size(3), num_groups, x.size(1) / num_groups, ptr_mean_var_out,
                           red_buffer.data_ptr<float>(), barrier.data_ptr<unsigned>(), sm_margin, stream, device_id,
                           nullptr, false);
  } else if (x.dtype() == at::kBFloat16 && w.dtype() == at::kBFloat16) {
    group_norm_v2::gn_cuda((__nv_bfloat16*)out.data_ptr(), (__nv_bfloat16*)x.data_ptr(), (__nv_bfloat16*)w.data_ptr(),
                           (__nv_bfloat16*)b.data_ptr(), eps, silu, x.size(0), x.size(2) * x.size(3), num_groups,
                           x.size(1) / num_groups, ptr_mean_var_out, red_buffer.data_ptr<float>(),
                           barrier.data_ptr<unsigned>(), sm_margin, stream, device_id, nullptr, false);
  } else {
    throw std::invalid_argument("gn only supports half or bfloat16 input and weight");
  }
  return out;
}

auto gn_bwd(at::Tensor grad_output, at::Tensor x, at::Tensor w, at::Tensor b, at::Tensor mean_var, float eps, bool silu,
            int num_groups, int sm_margin) {
  if (w.dtype() != b.dtype() || x.dtype() != grad_output.dtype() || mean_var.dtype() != at::kFloat) {
    throw std::invalid_argument("gn_bwd dtype mismatch");
  }
  at::Tensor grad_input = at::empty_like(x);
  at::Tensor grad_weight = at::empty_like(w);
  at::Tensor grad_bias = at::empty_like(w);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  int device_id = at::cuda::getCurrentCUDAStream().device().index();
  group_norm_v2::Meta meta;
  if (x.dtype() == at::kHalf && w.dtype() == at::kHalf) {
    group_norm_v2::gn_bwd_cuda((half*)grad_input.data_ptr(), (half*)grad_weight.data_ptr(), (half*)grad_bias.data_ptr(),
                               (half*)grad_output.data_ptr(), (half*)x.data_ptr(), (half*)w.data_ptr(),
                               (half*)b.data_ptr(), mean_var.data_ptr<float>(), eps, silu, x.size(0),
                               x.size(2) * x.size(3), num_groups, x.size(1) / num_groups, nullptr, nullptr, sm_margin,
                               stream, device_id, &meta, true);
  } else if (x.dtype() == at::kBFloat16 && w.dtype() == at::kBFloat16) {
    group_norm_v2::gn_bwd_cuda((__nv_bfloat16*)grad_input.data_ptr(), (__nv_bfloat16*)grad_weight.data_ptr(),
                               (__nv_bfloat16*)grad_bias.data_ptr(), (__nv_bfloat16*)grad_output.data_ptr(),
                               (__nv_bfloat16*)x.data_ptr(), (__nv_bfloat16*)w.data_ptr(), (__nv_bfloat16*)b.data_ptr(),
                               mean_var.data_ptr<float>(), eps, silu, x.size(0), x.size(2) * x.size(3), num_groups,
                               x.size(1) / num_groups, nullptr, nullptr, sm_margin, stream, device_id, &meta, true);
  } else {
    throw std::invalid_argument("gn only supports half or bfloat16 input and weight");
  }
  at::Tensor red_buffer = at::empty({meta.red_buffer_size}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  thread_local at::Tensor barrier;
  if (barrier.size(0) < meta.barrier_size) {
    barrier = at::zeros({meta.barrier_size}, at::TensorOptions().dtype(at::ScalarType::UInt32).device(at::kCUDA));
  }
  if (x.dtype() == at::kHalf && w.dtype() == at::kHalf) {
    group_norm_v2::gn_bwd_cuda((half*)grad_input.data_ptr(), (half*)grad_weight.data_ptr(), (half*)grad_bias.data_ptr(),
                               (half*)grad_output.data_ptr(), (half*)x.data_ptr(), (half*)w.data_ptr(),
                               (half*)b.data_ptr(), mean_var.data_ptr<float>(), eps, silu, x.size(0),
                               x.size(2) * x.size(3), num_groups, x.size(1) / num_groups, red_buffer.data_ptr<float>(),
                               barrier.data_ptr<unsigned>(), sm_margin, stream, device_id, nullptr, false);
  } else if (x.dtype() == at::kBFloat16 && w.dtype() == at::kBFloat16) {
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

namespace {
at::Tensor apex_group_norm_v2_gn(at::Tensor x, at::Tensor w, at::Tensor b, double eps, bool silu, int64_t num_groups,
                                 const std::optional<at::Tensor>& mean_var_out, int64_t sm_margin) {
  return group_norm_v2::gn(x, w, b, static_cast<float>(eps), silu, static_cast<int>(num_groups), mean_var_out,
                           static_cast<int>(sm_margin));
}

std::vector<at::Tensor> apex_group_norm_v2_gn_bwd(at::Tensor grad_output, at::Tensor x, at::Tensor w, at::Tensor b,
                                                  at::Tensor mean_var, double eps, bool silu, int64_t num_groups,
                                                  int64_t sm_margin) {
  auto grads = group_norm_v2::gn_bwd(grad_output, x, w, b, mean_var, static_cast<float>(eps), silu,
                                     static_cast<int>(num_groups), static_cast<int>(sm_margin));
  return {std::get<0>(grads), std::get<1>(grads), std::get<2>(grads)};
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(apex, m) {
  m.def(
      "group_norm_v2_gn(Tensor x, Tensor w, Tensor b, float eps, bool silu, int num_groups, "
      "Tensor? mean_var_out, int sm_margin) -> Tensor");
  m.def(
      "group_norm_v2_gn_bwd(Tensor grad_output, Tensor x, Tensor w, Tensor b, Tensor mean_var, float eps, "
      "bool silu, int num_groups, int sm_margin) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(apex, CUDA, m) {
  m.impl("group_norm_v2_gn", &apex_group_norm_v2_gn);
  m.impl("group_norm_v2_gn_bwd", &apex_group_norm_v2_gn_bwd);
}
