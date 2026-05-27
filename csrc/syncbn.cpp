#include <ATen/ATen.h>
#include <torch/library.h>

#include <vector>

// returns {mean,biased_var}
// implemented using welford
std::vector<at::Tensor> welford_mean_var_CUDA(const at::Tensor input);

// reduces array of mean/var across processes
// returns global {mean,inv_std,biased_var}
// implemented using welford
std::vector<at::Tensor> welford_parallel_CUDA(const at::Tensor mean_feature_nodes,
                                              const at::Tensor var_biased_feature_nodes, const at::Tensor numel,
                                              const float eps);

// elementwise BN operation, returns output
// input/weight/shift should have identical data type;
// mean/inv_std have promoted data type (dtype==fp16?fp32:dtype)
at::Tensor batchnorm_forward_CUDA(const at::Tensor input, const at::Tensor mean, const at::Tensor inv_std,
                                  const at::optional<at::Tensor> weight, const at::optional<at::Tensor> shift);

// backward BN operation, returns {sum_dy, sum_dy_xmu, grad_weight, grad_bias}
// grad_output/input should have identical data type;
// mean/inv_std have promoted data type (dtype==fp16?fp32:dtype)
// implemented using kahan summation
std::vector<at::Tensor> reduce_bn_CUDA(const at::Tensor grad_output, const at::Tensor input, const at::Tensor mean,
                                       const at::Tensor inv_std, const at::optional<at::Tensor> weight);

// elementwise backward BN operation, returns grad_input
// grad_output/input/weight precision could be fp16/fp32;
// mean/inv_std/sum_dy/sum_dy_xmu precision is fp32
at::Tensor batchnorm_backward_CUDA(const at::Tensor grad_output, const at::Tensor input, const at::Tensor mean,
                                   const at::Tensor inv_std, const at::optional<at::Tensor> weight,
                                   const at::Tensor sum_dy, const at::Tensor sum_dy_xmu, const at::Tensor count);

// returns {mean, biased_var}
// implemented using welford
// expect data to be in n+c format (channel last) and applies CUDNN_BATCHNORM_SPATIAL
std::vector<at::Tensor> welford_mean_var_c_last_CUDA(const at::Tensor input);

// elementwise BN operation, returns output
// input/weight/shift should have identical data type;
// mean/inv_std have promoted data type (dtype==fp16?fp32:dtype)
// expect data to be in n+c format (channel last) and applies CUDNN_BATCHNORM_SPATIAL
at::Tensor batchnorm_forward_c_last_CUDA(const at::Tensor input, const at::optional<at::Tensor> z,
                                         const at::Tensor mean, const at::Tensor inv_std,
                                         const at::optional<at::Tensor> weight, const at::optional<at::Tensor> shift,
                                         const bool fuse_relu);

// backward BN operation, returns {sum_dy, sum_dy_xmu, grad_weight, grad_bias}
// grad_output/input should have identical data type;
// mean/inv_std have promoted data type (dtype==fp16?fp32:dtype)
// expect data to be in n+c format (channel last) and applies CUDNN_BATCHNORM_SPATIAL
std::vector<at::Tensor> reduce_bn_c_last_CUDA(const at::Tensor grad_output, const at::Tensor input,
                                              const at::Tensor mean, const at::Tensor inv_std,
                                              const at::optional<at::Tensor> weight);

// elementwise backward BN operation, returns grad_input
// grad_output/input/weight precision could be fp16/fp32;
// mean/inv_std/sum_dy/sum_dy_xmu precision is fp32
// expect data to be in n+c format (channel last) and applies CUDNN_BATCHNORM_SPATIAL
at::Tensor batchnorm_backward_c_last_CUDA(const at::Tensor grad_output, const at::Tensor input, const at::Tensor mean,
                                          const at::Tensor inv_std, const at::optional<at::Tensor> weight,
                                          const at::Tensor sum_dy, const at::Tensor sum_dy_xmu, const at::Tensor count);

at::Tensor relu_backward_c_last_CUDA(const at::Tensor grad_output, const at::Tensor input,
                                     const at::optional<at::Tensor> z, const at::Tensor mean, const at::Tensor inv_std,
                                     const at::optional<at::Tensor> weight, const at::optional<at::Tensor> shift);

namespace {
std::vector<at::Tensor> apex_welford_parallel_CUDA(const at::Tensor mean_feature_nodes,
                                                   const at::Tensor var_biased_feature_nodes, const at::Tensor numel,
                                                   double eps) {
  return welford_parallel_CUDA(mean_feature_nodes, var_biased_feature_nodes, numel, static_cast<float>(eps));
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(apex, m) {
  m.def("syncbn_welford_mean_var(Tensor input) -> Tensor[]");
  m.def("syncbn_welford_parallel(Tensor mean_feature_nodes, Tensor var_biased_feature_nodes, Tensor numel, float eps) "
        "-> Tensor[]");
  m.def("syncbn_batchnorm_forward(Tensor input, Tensor mean, Tensor inv_std, Tensor? weight, Tensor? shift) -> Tensor");
  m.def("syncbn_reduce_bn(Tensor grad_output, Tensor input, Tensor mean, Tensor inv_std, Tensor? weight) -> Tensor[]");
  m.def("syncbn_batchnorm_backward(Tensor grad_output, Tensor input, Tensor mean, Tensor inv_std, Tensor? weight, "
        "Tensor sum_dy, Tensor sum_dy_xmu, Tensor count) -> Tensor");
  m.def("syncbn_welford_mean_var_c_last(Tensor input) -> Tensor[]");
  m.def("syncbn_batchnorm_forward_c_last(Tensor input, Tensor? z, Tensor mean, Tensor inv_std, Tensor? weight, "
        "Tensor? shift, bool fuse_relu) -> Tensor");
  m.def("syncbn_reduce_bn_c_last(Tensor grad_output, Tensor input, Tensor mean, Tensor inv_std, Tensor? weight) "
        "-> Tensor[]");
  m.def("syncbn_batchnorm_backward_c_last(Tensor grad_output, Tensor input, Tensor mean, Tensor inv_std, "
        "Tensor? weight, Tensor sum_dy, Tensor sum_dy_xmu, Tensor count) -> Tensor");
  m.def("syncbn_relu_bw_c_last(Tensor grad_output, Tensor input, Tensor? z, Tensor mean, Tensor inv_std, "
        "Tensor? weight, Tensor? shift) -> Tensor");
}

TORCH_LIBRARY_IMPL(apex, CUDA, m) {
  m.impl("syncbn_welford_mean_var", &welford_mean_var_CUDA);
  m.impl("syncbn_welford_parallel", &apex_welford_parallel_CUDA);
  m.impl("syncbn_batchnorm_forward", &batchnorm_forward_CUDA);
  m.impl("syncbn_reduce_bn", &reduce_bn_CUDA);
  m.impl("syncbn_batchnorm_backward", &batchnorm_backward_CUDA);
  m.impl("syncbn_welford_mean_var_c_last", &welford_mean_var_c_last_CUDA);
  m.impl("syncbn_batchnorm_forward_c_last", &batchnorm_forward_c_last_CUDA);
  m.impl("syncbn_reduce_bn_c_last", &reduce_bn_c_last_CUDA);
  m.impl("syncbn_batchnorm_backward_c_last", &batchnorm_backward_c_last_CUDA);
  m.impl("syncbn_relu_bw_c_last", &relu_backward_c_last_CUDA);
}
