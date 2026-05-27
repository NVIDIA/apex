#include <ATen/ATen.h>
#include <torch/library.h>

#include <iostream>
#include <map>
#include <vector>

#include "norm_sample.h"

// define this enum:
enum bn_type { BN_FWD, BN_BWD };

// this is a global variable
static std::map<std::vector<int64_t>, cudnn_frontend::ExecutionPlan> gbn_plan_cache;

at::Tensor gbn_forward(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& bias,
                       const at::Tensor& running_mean, const at::Tensor& running_var, const at::Tensor& minibatch_mean,
                       const at::Tensor& minibatch_inv_var, const float momentum, const float epsilon,
                       const int64_t bn_group, const int rank_id, const std::vector<int64_t>& peer_buffers) {
  int64_t N = x.size(0);
  int64_t C = x.size(1);
  int64_t H = x.size(2);
  int64_t W = x.size(3);

  int64_t tensorDims[] = {N, C, H, W};
  int64_t peerDims[] = {bn_group, 4 * C, 1, 1};
  int64_t perChannelDims[] = {1, C, 1, 1};
  int64_t epsilonDims[] = {1, 1, 1, 1};

  // Allocate output tensor
  at::Tensor y = at::empty_like(x);

  std::vector<void*> void_peer_buffers;
  for (int64_t addr : peer_buffers) {
    void_peer_buffers.push_back((void*)addr);
  }

  // we need the peer size for the buffer reset
  size_t peer_size = 1;
  for (size_t i = 0; i < 4; ++i) {
    peer_size *= peerDims[i];
  }

  // sanity check
  assert(bn_group == void_peer_buffers.size());

  // check if plan already exists
  std::vector<int64_t> fv = {(int64_t)BN_FWD, N, C, H, W, bn_group, (int64_t)CUDNN_DATA_HALF};
  if (gbn_plan_cache.find(fv) == gbn_plan_cache.end()) {
    auto plan = run_batch_norm_forward(tensorDims, perChannelDims, epsilonDims, peerDims, CUDNN_DATA_HALF);
    gbn_plan_cache.emplace(fv, std::move(plan));
  }

  // get plan and handle
  auto plan = gbn_plan_cache.find(fv)->second;

  // execute
  execute_batch_norm_forward(plan, x.data_ptr(), y.data_ptr(), scale.data_ptr(), bias.data_ptr(),
                             running_mean.data_ptr(), running_var.data_ptr(), running_mean.data_ptr(),
                             running_var.data_ptr(), minibatch_mean.data_ptr(), minibatch_inv_var.data_ptr(),
                             void_peer_buffers, static_cast<double>(epsilon), static_cast<double>(momentum), peer_size,
                             rank_id);

  return y;
}

std::vector<at::Tensor> gbn_backward(const at::Tensor& x, const at::Tensor& dy, const at::Tensor& scale,
                                     const at::Tensor& minibatch_mean, const at::Tensor& minibatch_inv_var,
                                     const float epsilon, const int64_t bn_group, const int rank_id,
                                     const std::vector<int64_t>& peer_buffers) {
  int64_t N = x.size(0);
  int64_t C = x.size(1);
  int64_t H = x.size(2);
  int64_t W = x.size(3);

  int64_t tensorDims[] = {N, C, H, W};
  int64_t peerDims[] = {bn_group, 4 * C, 1, 1};
  int64_t perChannelDims[] = {1, C, 1, 1};
  int64_t epsilonDims[] = {1, 1, 1, 1};

  // Allocate output tensor
  // outputs
  at::Tensor x_grad, scale_grad, bias_grad;

  // Allocate outputs
  x_grad = at::empty_like(x);
  scale_grad = at::empty_like(scale);
  bias_grad = at::empty_like(scale);

  std::vector<void*> void_peer_buffers;
  for (int64_t addr : peer_buffers) {
    void_peer_buffers.push_back((void*)addr);
  }

  // we need the peer size for the buffer reset
  size_t peer_size = 1;
  for (size_t i = 0; i < 4; ++i) {
    peer_size *= peerDims[i];
  }

  assert(bn_group == void_peer_buffers.size());

  std::vector<int64_t> fv = {(int64_t)BN_BWD, N, C, H, W, bn_group, (int64_t)CUDNN_DATA_HALF};
  if (gbn_plan_cache.find(fv) == gbn_plan_cache.end()) {
    auto plan = run_batch_norm_backward(tensorDims, perChannelDims, epsilonDims, peerDims, CUDNN_DATA_HALF);
    gbn_plan_cache.emplace(fv, std::move(plan));
  }

  // get plan and handle
  auto plan = gbn_plan_cache.find(fv)->second;

  // execute
  execute_batch_norm_backward(plan, x.data_ptr(), dy.data_ptr(), scale.data_ptr(), minibatch_mean.data_ptr(),
                              minibatch_inv_var.data_ptr(), void_peer_buffers, x_grad.data_ptr(), scale_grad.data_ptr(),
                              bias_grad.data_ptr(), static_cast<double>(epsilon), peer_size, rank_id);

  return std::vector<at::Tensor>{x_grad, scale_grad, bias_grad};
}

namespace {
std::vector<int64_t> to_int_vector(at::IntArrayRef values) {
  return std::vector<int64_t>(values.begin(), values.end());
}

at::Tensor apex_cudnn_gbn_forward(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& bias,
                                  const at::Tensor& running_mean, const at::Tensor& running_var,
                                  const at::Tensor& minibatch_mean, const at::Tensor& minibatch_inv_var,
                                  double momentum, double epsilon, int64_t bn_group, int64_t rank_id,
                                  at::IntArrayRef peer_buffers) {
  return gbn_forward(x, scale, bias, running_mean, running_var, minibatch_mean, minibatch_inv_var,
                     static_cast<float>(momentum), static_cast<float>(epsilon), bn_group, static_cast<int>(rank_id),
                     to_int_vector(peer_buffers));
}

std::vector<at::Tensor> apex_cudnn_gbn_backward(const at::Tensor& x, const at::Tensor& dy, const at::Tensor& scale,
                                                const at::Tensor& minibatch_mean, const at::Tensor& minibatch_inv_var,
                                                double epsilon, int64_t bn_group, int64_t rank_id,
                                                at::IntArrayRef peer_buffers) {
  return gbn_backward(x, dy, scale, minibatch_mean, minibatch_inv_var, static_cast<float>(epsilon), bn_group,
                      static_cast<int>(rank_id), to_int_vector(peer_buffers));
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(apex, m) {
  m.def(
      "cudnn_gbn_forward(Tensor x, Tensor scale, Tensor bias, Tensor running_mean, Tensor running_var, "
      "Tensor minibatch_mean, Tensor minibatch_inv_var, float momentum, float epsilon, int bn_group, int rank_id, "
      "int[] peer_buffers) -> Tensor");
  m.def(
      "cudnn_gbn_backward(Tensor x, Tensor dy, Tensor scale, Tensor minibatch_mean, Tensor minibatch_inv_var, "
      "float epsilon, int bn_group, int rank_id, int[] peer_buffers) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(apex, CUDA, m) {
  m.impl("cudnn_gbn_forward", &apex_cudnn_gbn_forward);
  m.impl("cudnn_gbn_backward", &apex_cudnn_gbn_backward);
}
