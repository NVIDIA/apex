#include <torch/torch.h>
#include <ATen/ATen.h>

#include <vector>


std::vector<at::Tensor> welford_mean_var_CUDA(const at::Tensor input);
std::vector<at::Tensor> welford_parallel_CUDA(const at::Tensor mean_feature_nodes, const at::Tensor var_biased_feature_nodes, int numel);

at::Tensor batchnorm_forward_CUDA(const at::Tensor input,
                                  const at::Tensor mean,
                                  const at::Tensor var,
                                  const at::Tensor weight,
                                  const at::Tensor shift,
                                  const float eps);

std::vector<at::Tensor> reduce_bn_CUDA(const at::Tensor grad_output,
                                           const at::Tensor input,
                                           const at::Tensor mean,
                                           const at::Tensor var,
                                           const float eps);

at::Tensor batchnorm_backward_CUDA(const at::Tensor grad_output,
                                   const at::Tensor input,
                                   const at::Tensor mean,
                                   const at::Tensor var,
                                   const at::Tensor weight,
                                   const at::Tensor shift,
                                   const at::Tensor mean_dy,
                                   const at::Tensor mean_dy_xmu,
                                   const float eps);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("welford_mean_var", &welford_mean_var_CUDA, "welford mean variance");
  m.def("welford_parallel", &welford_parallel_CUDA, "welford parallel reduce mean variance");
  m.def("batchnorm_forward", &batchnorm_forward_CUDA, "batchnorm forward");
  m.def("reduce_bn", &reduce_bn_CUDA, "batchnorm backward reduce grad sum and bias/weight gradient");
  m.def("batchnorm_backward", &batchnorm_backward_CUDA, "batchnorm backward dgrad");
}
