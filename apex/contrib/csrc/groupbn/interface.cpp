#include <ATen/ATen.h>
#include <ATen/ArrayRef.h>
#include <ATen/ScalarType.h>
#include <torch/library.h>

#include "ATen/Generator.h"
#include "ATen/Scalar.h"
#include "ATen/Storage.h"
#include "ATen/Tensor.h"

int64_t get_buffer_size(const int bn_sync_steps);

void* get_data_ptr(const at::Tensor& data);

void* get_remote_data_ptr(const at::Tensor& handle, const int64_t offset);

void close_remote_data(const at::Tensor& handle);

at::Tensor nhwc_bn_fwd_train(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& bias,
                             const at::Tensor& running_mean, const at::Tensor& running_inv_var,
                             const at::Tensor& minibatch_mean, const at::Tensor& minibatch_inv_var,
                             const at::Tensor& ret_cta, const float momentum, const float epsilon, const bool fuse_relu,
                             void* my_data, void* pair_data, void* pair_data2, void* pair_data3, const int bn_group,
                             const at::Tensor& magic_tensor, const int occupancy, const int grid_dim_x,
                             const bool coop);

at::Tensor nhwc_bn_fwd_eval(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& bias,
                            const at::Tensor& running_mean, const at::Tensor& running_inv_var,
                            const at::Tensor& ret_cta, const int bn_group, const float momentum, const float epsilon,
                            const bool fuse_relu);

std::vector<at::Tensor> nhwc_bn_bwd(const at::Tensor& x, const at::Tensor& dy, const at::Tensor& scale,
                                    const at::Tensor& bias, const at::Tensor& running_mean,
                                    const at::Tensor& running_inv_var, const at::Tensor& minibatch_mean,
                                    const at::Tensor& minibatch_inv_var, const at::Tensor& ret_cta,
                                    const float momentum, const float epsilon, const bool fuse_relu, void* my_data,
                                    void* pair_data, void* pair_data2, void* pair_data3, const int bn_group,
                                    const at::Tensor& magic_tensor, const int occupancy, const int grid_dim_x,
                                    const bool coop);

at::Tensor nhwc_bn_addrelu_fwd_train(const at::Tensor& x, const at::Tensor& z, const at::Tensor& scale,
                                     const at::Tensor& bias, const at::Tensor& running_mean,
                                     const at::Tensor& running_inv_var, const at::Tensor& minibatch_mean,
                                     const at::Tensor& minibatch_inv_var, const at::Tensor& bitmask,
                                     const at::Tensor& ret_cta, const float momentum, const float epsilon,
                                     void* my_data, void* pair_data, void* pair_data2, void* pair_data3,
                                     const int bn_group, const at::Tensor& magic_tensor, const int occupancy,
                                     const int grid_dim_x, const bool coop);

at::Tensor nhwc_bn_addrelu_fwd_eval(const at::Tensor& x, const at::Tensor& z, const at::Tensor& scale,
                                    const at::Tensor& bias, const at::Tensor& running_mean,
                                    const at::Tensor& running_inv_var, const at::Tensor& ret_cta, const int bn_group,
                                    const float momentum, const float epsilon);

std::vector<at::Tensor> nhwc_bn_addrelu_bwd(const at::Tensor& x, const at::Tensor& dy, const at::Tensor& scale,
                                            const at::Tensor& bias, const at::Tensor& running_mean,
                                            const at::Tensor& running_inv_var, const at::Tensor& minibatch_mean,
                                            const at::Tensor& minibatch_inv_var, const at::Tensor& bitmask,
                                            const at::Tensor& ret_cta, const float momentum, const float epsilon,
                                            void* my_data, void* pair_data, void* pair_data2, void* pair_data3,
                                            const int bn_group, const at::Tensor& magic_tensor, const int occupancy,
                                            const int grid_dim_x, const bool coop);

int nhwc_bn_fwd_occupancy();
int nhwc_bn_bwd_occupancy();

int nhwc_bn_addrelu_fwd_occupancy();
int nhwc_bn_addrelu_bwd_occupancy();

namespace {
void* optional_ptr(const c10::optional<int64_t>& ptr) {
  return ptr.has_value() ? reinterpret_cast<void*>(ptr.value()) : nullptr;
}

int64_t apex_bnp_get_buffer_size(int64_t bn_sync_steps) { return get_buffer_size(static_cast<int>(bn_sync_steps)); }

int64_t apex_bnp_get_data_ptr(const at::Tensor& data) { return reinterpret_cast<int64_t>(get_data_ptr(data)); }

int64_t apex_bnp_get_remote_data_ptr(const at::Tensor& handle, int64_t offset) {
  return reinterpret_cast<int64_t>(get_remote_data_ptr(handle, offset));
}

int64_t apex_bnp_bn_fwd_nhwc_occupancy() { return nhwc_bn_fwd_occupancy(); }

int64_t apex_bnp_bn_bwd_nhwc_occupancy() { return nhwc_bn_bwd_occupancy(); }

int64_t apex_bnp_bn_addrelu_fwd_nhwc_occupancy() { return nhwc_bn_addrelu_fwd_occupancy(); }

int64_t apex_bnp_bn_addrelu_bwd_nhwc_occupancy() { return nhwc_bn_addrelu_bwd_occupancy(); }

at::Tensor apex_bnp_bn_fwd_nhwc(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& bias,
                                const at::Tensor& running_mean, const at::Tensor& running_inv_var,
                                const at::Tensor& minibatch_mean, const at::Tensor& minibatch_inv_var,
                                const at::Tensor& ret_cta, double momentum, double epsilon, bool fuse_relu,
                                c10::optional<int64_t> my_data, c10::optional<int64_t> pair_data,
                                c10::optional<int64_t> pair_data2, c10::optional<int64_t> pair_data3, int64_t bn_group,
                                const at::Tensor& magic_tensor, int64_t occupancy, int64_t grid_dim_x, bool coop) {
  return nhwc_bn_fwd_train(x, scale, bias, running_mean, running_inv_var, minibatch_mean, minibatch_inv_var, ret_cta,
                           static_cast<float>(momentum), static_cast<float>(epsilon), fuse_relu, optional_ptr(my_data),
                           optional_ptr(pair_data), optional_ptr(pair_data2), optional_ptr(pair_data3),
                           static_cast<int>(bn_group), magic_tensor, static_cast<int>(occupancy),
                           static_cast<int>(grid_dim_x), coop);
}

at::Tensor apex_bnp_bn_fwd_eval_nhwc(const at::Tensor& x, const at::Tensor& scale, const at::Tensor& bias,
                                     const at::Tensor& running_mean, const at::Tensor& running_inv_var,
                                     const at::Tensor& ret_cta, int64_t bn_group, double momentum, double epsilon,
                                     bool fuse_relu) {
  return nhwc_bn_fwd_eval(x, scale, bias, running_mean, running_inv_var, ret_cta, static_cast<int>(bn_group),
                          static_cast<float>(momentum), static_cast<float>(epsilon), fuse_relu);
}

std::vector<at::Tensor> apex_bnp_bn_bwd_nhwc(const at::Tensor& x, const at::Tensor& dy, const at::Tensor& scale,
                                             const at::Tensor& bias, const at::Tensor& running_mean,
                                             const at::Tensor& running_inv_var, const at::Tensor& minibatch_mean,
                                             const at::Tensor& minibatch_inv_var, const at::Tensor& ret_cta,
                                             double momentum, double epsilon, bool fuse_relu,
                                             c10::optional<int64_t> my_data, c10::optional<int64_t> pair_data,
                                             c10::optional<int64_t> pair_data2, c10::optional<int64_t> pair_data3,
                                             int64_t bn_group, const at::Tensor& magic_tensor, int64_t occupancy,
                                             int64_t grid_dim_x, bool coop) {
  return nhwc_bn_bwd(x, dy, scale, bias, running_mean, running_inv_var, minibatch_mean, minibatch_inv_var, ret_cta,
                     static_cast<float>(momentum), static_cast<float>(epsilon), fuse_relu, optional_ptr(my_data),
                     optional_ptr(pair_data), optional_ptr(pair_data2), optional_ptr(pair_data3),
                     static_cast<int>(bn_group), magic_tensor, static_cast<int>(occupancy),
                     static_cast<int>(grid_dim_x), coop);
}

at::Tensor apex_bnp_bn_addrelu_fwd_nhwc(const at::Tensor& x, const at::Tensor& z, const at::Tensor& scale,
                                        const at::Tensor& bias, const at::Tensor& running_mean,
                                        const at::Tensor& running_inv_var, const at::Tensor& minibatch_mean,
                                        const at::Tensor& minibatch_inv_var, const at::Tensor& bitmask,
                                        const at::Tensor& ret_cta, double momentum, double epsilon,
                                        c10::optional<int64_t> my_data, c10::optional<int64_t> pair_data,
                                        c10::optional<int64_t> pair_data2, c10::optional<int64_t> pair_data3,
                                        int64_t bn_group, const at::Tensor& magic_tensor, int64_t occupancy,
                                        int64_t grid_dim_x, bool coop) {
  return nhwc_bn_addrelu_fwd_train(x, z, scale, bias, running_mean, running_inv_var, minibatch_mean, minibatch_inv_var,
                                   bitmask, ret_cta, static_cast<float>(momentum), static_cast<float>(epsilon),
                                   optional_ptr(my_data), optional_ptr(pair_data), optional_ptr(pair_data2),
                                   optional_ptr(pair_data3), static_cast<int>(bn_group), magic_tensor,
                                   static_cast<int>(occupancy), static_cast<int>(grid_dim_x), coop);
}

at::Tensor apex_bnp_bn_addrelu_fwd_eval_nhwc(const at::Tensor& x, const at::Tensor& z, const at::Tensor& scale,
                                             const at::Tensor& bias, const at::Tensor& running_mean,
                                             const at::Tensor& running_inv_var, const at::Tensor& ret_cta,
                                             int64_t bn_group, double momentum, double epsilon) {
  return nhwc_bn_addrelu_fwd_eval(x, z, scale, bias, running_mean, running_inv_var, ret_cta, static_cast<int>(bn_group),
                                  static_cast<float>(momentum), static_cast<float>(epsilon));
}

std::vector<at::Tensor> apex_bnp_bn_addrelu_bwd_nhwc(
    const at::Tensor& x, const at::Tensor& dy, const at::Tensor& scale, const at::Tensor& bias,
    const at::Tensor& running_mean, const at::Tensor& running_inv_var, const at::Tensor& minibatch_mean,
    const at::Tensor& minibatch_inv_var, const at::Tensor& bitmask, const at::Tensor& ret_cta, double momentum,
    double epsilon, c10::optional<int64_t> my_data, c10::optional<int64_t> pair_data, c10::optional<int64_t> pair_data2,
    c10::optional<int64_t> pair_data3, int64_t bn_group, const at::Tensor& magic_tensor, int64_t occupancy,
    int64_t grid_dim_x, bool coop) {
  return nhwc_bn_addrelu_bwd(x, dy, scale, bias, running_mean, running_inv_var, minibatch_mean, minibatch_inv_var,
                             bitmask, ret_cta, static_cast<float>(momentum), static_cast<float>(epsilon),
                             optional_ptr(my_data), optional_ptr(pair_data), optional_ptr(pair_data2),
                             optional_ptr(pair_data3), static_cast<int>(bn_group), magic_tensor,
                             static_cast<int>(occupancy), static_cast<int>(grid_dim_x), coop);
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(apex, m) {
  m.def("bnp_get_buffer_size(int bn_sync_steps) -> int");
  m.def("bnp_get_data_ptr(Tensor data) -> int");
  m.def("bnp_get_remote_data_ptr(Tensor handle, int offset) -> int");
  m.def("bnp_close_remote_data(Tensor handle) -> ()");
  m.def(
      "bnp_bn_fwd_nhwc(Tensor x, Tensor scale, Tensor bias, Tensor running_mean, Tensor running_inv_var, "
      "Tensor minibatch_mean, Tensor minibatch_inv_var, Tensor ret_cta, float momentum, float epsilon, "
      "bool fuse_relu, int? my_data, int? pair_data, int? pair_data2, int? pair_data3, int bn_group, "
      "Tensor magic_tensor, int occupancy, int grid_dim_x, bool coop) -> Tensor");
  m.def(
      "bnp_bn_fwd_eval_nhwc(Tensor x, Tensor scale, Tensor bias, Tensor running_mean, Tensor running_inv_var, "
      "Tensor ret_cta, int bn_group, float momentum, float epsilon, bool fuse_relu) -> Tensor");
  m.def(
      "bnp_bn_bwd_nhwc(Tensor x, Tensor dy, Tensor scale, Tensor bias, Tensor running_mean, "
      "Tensor running_inv_var, Tensor minibatch_mean, Tensor minibatch_inv_var, Tensor ret_cta, float momentum, "
      "float epsilon, bool fuse_relu, int? my_data, int? pair_data, int? pair_data2, int? pair_data3, "
      "int bn_group, Tensor magic_tensor, int occupancy, int grid_dim_x, bool coop) -> Tensor[]");
  m.def("bnp_bn_fwd_nhwc_occupancy() -> int");
  m.def("bnp_bn_bwd_nhwc_occupancy() -> int");
  m.def(
      "bnp_bn_addrelu_fwd_nhwc(Tensor x, Tensor z, Tensor scale, Tensor bias, Tensor running_mean, "
      "Tensor running_inv_var, Tensor minibatch_mean, Tensor minibatch_inv_var, Tensor bitmask, Tensor ret_cta, "
      "float momentum, float epsilon, int? my_data, int? pair_data, int? pair_data2, int? pair_data3, "
      "int bn_group, Tensor magic_tensor, int occupancy, int grid_dim_x, bool coop) -> Tensor");
  m.def(
      "bnp_bn_addrelu_fwd_eval_nhwc(Tensor x, Tensor z, Tensor scale, Tensor bias, Tensor running_mean, "
      "Tensor running_inv_var, Tensor ret_cta, int bn_group, float momentum, float epsilon) -> Tensor");
  m.def(
      "bnp_bn_addrelu_bwd_nhwc(Tensor x, Tensor dy, Tensor scale, Tensor bias, Tensor running_mean, "
      "Tensor running_inv_var, Tensor minibatch_mean, Tensor minibatch_inv_var, Tensor bitmask, Tensor ret_cta, "
      "float momentum, float epsilon, int? my_data, int? pair_data, int? pair_data2, int? pair_data3, "
      "int bn_group, Tensor magic_tensor, int occupancy, int grid_dim_x, bool coop) -> Tensor[]");
  m.def("bnp_bn_addrelu_fwd_nhwc_occupancy() -> int");
  m.def("bnp_bn_addrelu_bwd_nhwc_occupancy() -> int");
}

TORCH_LIBRARY_IMPL(apex, CompositeExplicitAutograd, m) {
  m.impl("bnp_get_buffer_size", &apex_bnp_get_buffer_size);
  m.impl("bnp_get_data_ptr", &apex_bnp_get_data_ptr);
  m.impl("bnp_get_remote_data_ptr", &apex_bnp_get_remote_data_ptr);
  m.impl("bnp_close_remote_data", &close_remote_data);
  m.impl("bnp_bn_fwd_nhwc_occupancy", &apex_bnp_bn_fwd_nhwc_occupancy);
  m.impl("bnp_bn_bwd_nhwc_occupancy", &apex_bnp_bn_bwd_nhwc_occupancy);
  m.impl("bnp_bn_addrelu_fwd_nhwc_occupancy", &apex_bnp_bn_addrelu_fwd_nhwc_occupancy);
  m.impl("bnp_bn_addrelu_bwd_nhwc_occupancy", &apex_bnp_bn_addrelu_bwd_nhwc_occupancy);
}

TORCH_LIBRARY_IMPL(apex, CUDA, m) {
  m.impl("bnp_bn_fwd_nhwc", &apex_bnp_bn_fwd_nhwc);
  m.impl("bnp_bn_fwd_eval_nhwc", &apex_bnp_bn_fwd_eval_nhwc);
  m.impl("bnp_bn_bwd_nhwc", &apex_bnp_bn_bwd_nhwc);
  m.impl("bnp_bn_addrelu_fwd_nhwc", &apex_bnp_bn_addrelu_fwd_nhwc);
  m.impl("bnp_bn_addrelu_fwd_eval_nhwc", &apex_bnp_bn_addrelu_fwd_eval_nhwc);
  m.impl("bnp_bn_addrelu_bwd_nhwc", &apex_bnp_bn_addrelu_bwd_nhwc);
}
