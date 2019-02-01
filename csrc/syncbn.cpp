#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>

// returns {mean,biased_var}
// implemented using welford 
std::vector<at::Tensor> welford_mean_var_CUDA(const at::Tensor input);

// reduces array of mean/var across processes
// returns global {mean,inv_std,biased_var}
// implemented using welford 
std::vector<at::Tensor> welford_parallel_CUDA(const at::Tensor mean_feature_nodes,
                                              const at::Tensor var_biased_feature_nodes,
                                              int numel,
                                              const float eps);

// elementwise BN operation, returns output
// input/weight/shift should have identical data type;
// mean/inv_std have promoted data type (dtype==fp16?fp32:dtype)
at::Tensor batchnorm_forward_CUDA(const at::Tensor input,
                                  const at::Tensor mean,
                                  const at::Tensor inv_std,
                                  const at::optional<at::Tensor> weight,
                                  const at::optional<at::Tensor> shift);

// backward BN operation, returns {mean_dy, mean_dy_xmu, grad_weight, grad_bias}
// grad_output/input should have identical data type;
// mean/inv_std have promoted data type (dtype==fp16?fp32:dtype)
// implemented using kahan summation
std::vector<at::Tensor> reduce_bn_CUDA(const at::Tensor grad_output,
                                           const at::Tensor input,
                                           const at::Tensor mean,
                                           const at::Tensor inv_std,
                                           const at::optional<at::Tensor> weight);

// elementwise backward BN operation, returns grad_input
// grad_output/input/weight precision could be fp16/fp32;
// mean/inv_std/mean_dy/mean_dy_xmu precision is fp32
at::Tensor batchnorm_backward_CUDA(const at::Tensor grad_output,
                                   const at::Tensor input,
                                   const at::Tensor mean,
                                   const at::Tensor inv_std,
                                   const at::optional<at::Tensor> weight,
                                   const at::Tensor mean_dy,
                                   const at::Tensor mean_dy_xmu);

// returns {mean, biased_var}
// implemented using welford 
// expect data to be in n+c format (channel last) and applies CUDNN_BATCHNORM_SPATIAL
std::vector<at::Tensor> welford_mean_var_c_last_CUDA(const at::Tensor input);

// elementwise BN operation, returns output
// input/weight/shift should have identical data type;
// mean/inv_std have promoted data type (dtype==fp16?fp32:dtype)
// expect data to be in n+c format (channel last) and applies CUDNN_BATCHNORM_SPATIAL
at::Tensor batchnorm_forward_c_last_CUDA(const at::Tensor input,
                                         const at::Tensor mean,
                                         const at::Tensor inv_std,
                                         const at::optional<at::Tensor> weight,
                                         const at::optional<at::Tensor> shift);

// backward BN operation, returns {mean_dy, mean_dy_xmu, grad_weight, grad_bias}
// grad_output/input should have identical data type;
// mean/inv_std have promoted data type (dtype==fp16?fp32:dtype)
// expect data to be in n+c format (channel last) and applies CUDNN_BATCHNORM_SPATIAL
std::vector<at::Tensor> reduce_bn_c_last_CUDA(const at::Tensor grad_output,
                                              const at::Tensor input,
                                              const at::Tensor mean,
                                              const at::Tensor inv_std,
                                              const at::optional<at::Tensor> weight);

// elementwise backward BN operation, returns grad_input
// grad_output/input/weight precision could be fp16/fp32;
// mean/inv_std/mean_dy/mean_dy_xmu precision is fp32
// expect data to be in n+c format (channel last) and applies CUDNN_BATCHNORM_SPATIAL
at::Tensor batchnorm_backward_c_last_CUDA(const at::Tensor grad_output,
                                          const at::Tensor input,
                                          const at::Tensor mean,
                                          const at::Tensor inv_std,
                                          const at::optional<at::Tensor> weight,
                                          const at::Tensor mean_dy,
                                          const at::Tensor mean_dy_xmu);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("welford_mean_var", &welford_mean_var_CUDA, "welford mean variance");
  m.def("welford_parallel", &welford_parallel_CUDA, "welford parallel reduce mean variance");
  m.def("batchnorm_forward", &batchnorm_forward_CUDA, "batchnorm forward");
  m.def("reduce_bn", &reduce_bn_CUDA, "batchnorm backward reduce grad sum and bias/weight grad");
  m.def("batchnorm_backward", &batchnorm_backward_CUDA, "batchnorm backward dgrad");
  m.def("welford_mean_var_c_last", &welford_mean_var_c_last_CUDA, "welford mean variance nhwc");
  m.def("batchnorm_forward_c_last", &batchnorm_forward_c_last_CUDA, "batchnorm forward nhwc");
  m.def("reduce_bn_c_last", &reduce_bn_c_last_CUDA, "batchnorm backwards reduce grad sum and bias/weight grad nhwc");
  m.def("batchnorm_backward_c_last", &batchnorm_backward_c_last_CUDA, "batchnorm backward dgrad nhwc");
}
