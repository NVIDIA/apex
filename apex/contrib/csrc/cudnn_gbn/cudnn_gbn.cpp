#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <vector>

#include <iostream>

#include "norm_sample.h"

at::Tensor gbn_forward(const at::Tensor& x,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_var,
                       const at::Tensor& minibatch_mean,
                       const at::Tensor& minibatch_inv_var,
                       const float momentum,
                       const float epsilon,
                       const int64_t bn_group,
                       const int rank_id,
                       const std::vector<int64_t> &peer_buffers) {

  int64_t N = x.size(0);
  int64_t C = x.size(1);
  int64_t H = x.size(2);
  int64_t W = x.size(3);

  int64_t tensorDims[]     = {N, C, H, W};
  int64_t peerDims[]       = {bn_group, 4*C, 1, 1};
  int64_t perChannelDims[] = {1, C, 1, 1};
  int64_t epsilonDims[]    = {1, 1, 1, 1};

  // Allocate output tensor
  at::Tensor y = at::empty_like(x);

  std::vector<void*> void_peer_buffers;
  for (int64_t addr : peer_buffers) {
    void_peer_buffers.push_back((void*)addr);
  }

  assert(bn_group == void_peer_buffers.size());
  run_batch_norm_forward(
    perChannelDims,
    epsilonDims,
    tensorDims,
    peerDims,
    x.data_ptr(),
    y.data_ptr(),
    scale.data_ptr(),
    bias.data_ptr(),
    running_mean.data_ptr(),
    running_var.data_ptr(),
    running_mean.data_ptr(),
    running_var.data_ptr(),
    minibatch_mean.data_ptr(),
    minibatch_inv_var.data_ptr(),
    void_peer_buffers,
    epsilon,
    momentum,
    rank_id
  );

  return y;
}

std::vector<at::Tensor> gbn_backward(
                       const at::Tensor& x,
                       const at::Tensor& dy,
                       const at::Tensor& scale,
                       const at::Tensor& minibatch_mean,
                       const at::Tensor& minibatch_inv_var,
                       const float epsilon,
                       const int64_t bn_group,
                       const int rank_id,
                       const std::vector<int64_t> &peer_buffers) {

  int64_t N = x.size(0);
  int64_t C = x.size(1);
  int64_t H = x.size(2);
  int64_t W = x.size(3);

  int64_t tensorDims[]     = {N, C, H, W};
  int64_t peerDims[]       = {bn_group, 4*C, 1, 1};
  int64_t perChannelDims[] = {1, C, 1, 1};
  int64_t epsilonDims[]    = {1, 1, 1, 1};

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

  assert(bn_group == void_peer_buffers.size());

  run_batch_norm_backward(
    perChannelDims,
    epsilonDims,
    tensorDims,
    peerDims,
    x.data_ptr(),
    dy.data_ptr(),
    scale.data_ptr(),
    minibatch_mean.data_ptr(),
    minibatch_inv_var.data_ptr(),
    x_grad.data_ptr(),
    scale_grad.data_ptr(),
    bias_grad.data_ptr(),
    void_peer_buffers,
    epsilon,
    rank_id);



  return std::vector<at::Tensor>{x_grad, scale_grad, bias_grad};
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gbn_forward, "Group batch norm forward");
  m.def("backward", &gbn_backward, "Group batch backward");
}
