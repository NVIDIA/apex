#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/ArrayRef.h>
#include <ATen/ScalarType.h>
#include "ATen/Scalar.h"
#ifndef VERSION_GE_1_1
#include "ATen/Type.h"
#endif
#include "ATen/Tensor.h"
#include "ATen/Storage.h"
#include "ATen/Generator.h"


namespace py = pybind11;

int64_t get_buffer_size(
                       const int bn_sync_steps);

void* get_data_ptr(
                       const at::Tensor& data);

void* get_remote_data_ptr(
                       const at::Tensor& handle,
                       const int64_t offset);

void close_remote_data(
                       const at::Tensor& handle);

at::Tensor nhwc_bn_fwd_train(
                       const at::Tensor& x,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const at::Tensor& minibatch_mean,
                       const at::Tensor& minibatch_inv_var,
                       const at::Tensor& ret_cta,
                       const float momentum,
                       const float epsilon,
                       const bool fuse_relu,
                       void* my_data,
                       void* pair_data,
                       void* pair_data2,
                       void* pair_data3,
                       const int bn_group,
                       const at::Tensor& magic_tensor,
                       const int occupancy,
                       const int grid_dim_x,
                       const bool coop);

at::Tensor nhwc_bn_fwd_eval(
                       const at::Tensor& x,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const at::Tensor& ret_cta,
                       const int bn_group,
                       const float momentum,
                       const float epsilon,
                       const bool fuse_relu);

std::vector<at::Tensor> nhwc_bn_bwd(
                       const at::Tensor& x,
                       const at::Tensor& dy,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const at::Tensor& minibatch_mean,
                       const at::Tensor& minibatch_inv_var,
                       const at::Tensor& ret_cta,
                       const float momentum,
                       const float epsilon,
                       const bool fuse_relu,
                       void* my_data,
                       void* pair_data,
                       void* pair_data2,
                       void* pair_data3,
                       const int bn_group,
                       const at::Tensor& magic_tensor,
                       const int occupancy,
                       const int grid_dim_x,
                       const bool coop);

at::Tensor nhwc_bn_addrelu_fwd_train(
                       const at::Tensor& x,
                       const at::Tensor& z,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const at::Tensor& minibatch_mean,
                       const at::Tensor& minibatch_inv_var,
                       const at::Tensor& bitmask,
                       const at::Tensor& ret_cta,
                       const float momentum,
                       const float epsilon,
                       void* my_data,
                       void* pair_data,
                       void* pair_data2,
                       void* pair_data3,
                       const int bn_group,
                       const at::Tensor& magic_tensor,
                       const int occupancy,
                       const int grid_dim_x,
                       const bool coop);

at::Tensor nhwc_bn_addrelu_fwd_eval(
                       const at::Tensor& x,
                       const at::Tensor& z,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const at::Tensor& ret_cta,
                       const int bn_group,
                       const float momentum,
                       const float epsilon);

std::vector<at::Tensor> nhwc_bn_addrelu_bwd(
                       const at::Tensor& x,
                       const at::Tensor& dy,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const at::Tensor& minibatch_mean,
                       const at::Tensor& minibatch_inv_var,
                       const at::Tensor& bitmask,
                       const at::Tensor& ret_cta,
                       const float momentum,
                       const float epsilon,
                       void* my_data,
                       void* pair_data,
                       void* pair_data2,
                       void* pair_data3,
                       const int bn_group,
                       const at::Tensor& magic_tensor,
                       const int occupancy,
                       const int grid_dim_x,
                       const bool coop);

int nhwc_bn_fwd_occupancy();
int nhwc_bn_bwd_occupancy();

int nhwc_bn_addrelu_fwd_occupancy();
int nhwc_bn_addrelu_bwd_occupancy();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("get_buffer_size", &get_buffer_size, "get_buffer_size");
  m.def("get_data_ptr", &get_data_ptr, "get_data_ptr");
  m.def("get_remote_data_ptr", &get_remote_data_ptr, "get_remote_data_ptr");
  m.def("close_remote_data", &close_remote_data, "close_remote_data");

  m.def("bn_fwd_nhwc", &nhwc_bn_fwd_train, "bn_fwd_nhwc");
  m.def("bn_fwd_eval_nhwc", &nhwc_bn_fwd_eval, "bn_fwd_eval_nhwc");
  m.def("bn_bwd_nhwc", &nhwc_bn_bwd, "bn_bwd_nhwc");

  m.def("bn_fwd_nhwc_occupancy", &nhwc_bn_fwd_occupancy, "bn_fwd_nhwc_occupancy");
  m.def("bn_bwd_nhwc_occupancy", &nhwc_bn_bwd_occupancy, "bn_bwd_nhwc_occupancy");

  m.def("bn_addrelu_fwd_nhwc", &nhwc_bn_addrelu_fwd_train, "bn_addrelu_fwd_nhwc");
  m.def("bn_addrelu_fwd_eval_nhwc", &nhwc_bn_addrelu_fwd_eval, "bn_addrelu_fwd_eval_nhwc");
  m.def("bn_addrelu_bwd_nhwc", &nhwc_bn_addrelu_bwd, "bn_addrelu_bwd_nhwc");

  m.def("bn_addrelu_fwd_nhwc_occupancy", &nhwc_bn_addrelu_fwd_occupancy, "bn_addrelu_fwd_nhwc_occupancy");
  m.def("bn_addrelu_bwd_nhwc_occupancy", &nhwc_bn_addrelu_bwd_occupancy, "bn_addrelu_bwd_nhwc_occupancy");
}

