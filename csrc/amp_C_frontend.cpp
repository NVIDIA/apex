#include <torch/extension.h>

void multi_tensor_scale_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  float scale);

void multi_tensor_sgd_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  float wd,
  float momentum,
  float dampening,
  float lr,
  bool nesterov,
  bool first_run,
  bool wd_after_momentum,
  float scale);

void multi_tensor_axpby_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  float a,
  float b,
  int arg_to_check);

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::optional<bool> per_tensor_python);

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_mp_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::optional<bool> per_tensor_python);

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_scale_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  float scale,
  at::optional<bool> per_tensor_python);

void multi_tensor_lamb_stage1_cuda(
    int chunk_size,
    at::Tensor noop_flag,
    std::vector<std::vector<at::Tensor>> tensor_lists,
    at::Tensor per_tensor_decay,
    const int step,
    const float beta1,
    const float beta2,
    const float epsilon,
    at::Tensor global_grad_norm,
    const float max_global_grad_norm);

void multi_tensor_lamb_stage2_cuda(
    int chunk_size,
    at::Tensor noop_flag,
    std::vector<std::vector<at::Tensor>> tensor_lists,
    at::Tensor per_tensor_param_norm,
    at::Tensor per_tensor_update_norm,
    const float lr,
    const float weight_decay,
    at::optional<bool> use_nvlamb_python);

void multi_tensor_adam_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  const float lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  const int step,
  const int mode,
  const int bias_correction,
  const float weight_decay);


void multi_tensor_adagrad_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  const float lr,
  const float epsilon,
  const int mode,
  const float weight_decay);


void multi_tensor_novograd_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor grad_norms,
  const float lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  const int step,
  const int bias_correction,
  const float weight_decay,
  const int grad_averaging,
  const int mode,
  const int norm_type);

void multi_tensor_lamb_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  const float lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  const int step,
  const int bias_correction,
  const float weight_decay,
  const int grad_averaging,
  const int mode,
  at::Tensor global_grad_norm,
  const float max_grad_norm,
  at::optional<bool> use_nvlamb_python);

void multi_tensor_lamb_mp_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  at::Tensor step,
  const int bias_correction,
  const float weight_decay,
  const int grad_averaging,
  const int mode,
  at::Tensor global_grad_norm,
  at::Tensor max_grad_norm,
  at::optional<bool> use_nvlamb_python,
  at::Tensor found_inf,
  at::Tensor inv_scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multi_tensor_scale", &multi_tensor_scale_cuda,
        "Fused overflow check + scale for a list of contiguous tensors");
  m.def("multi_tensor_sgd", &multi_tensor_sgd_cuda,
        "Fused SGD optimizer for list of contiguous tensors");
  m.def("multi_tensor_axpby", &multi_tensor_axpby_cuda,
        "out = a*x + b*y for a list of contiguous tensors");
  m.def("multi_tensor_l2norm", &multi_tensor_l2norm_cuda,
        "Computes L2 norm for a list of contiguous tensors");
  m.def("multi_tensor_l2norm_mp", &multi_tensor_l2norm_mp_cuda,
        "Computes L2 norm for a list of contiguous tensors");
  m.def("multi_tensor_l2norm_scale", &multi_tensor_l2norm_scale_cuda,
        "Computes L2 norm for a list of contiguous tensors and does scaling");
  m.def("multi_tensor_lamb_stage1_cuda", &multi_tensor_lamb_stage1_cuda,
        "Computes update part of LAMB optimizer");
  m.def("multi_tensor_lamb_stage2_cuda", &multi_tensor_lamb_stage2_cuda,
        "Completes application of gradient to parameters for LAMB optimizer");
  m.def("multi_tensor_adam", &multi_tensor_adam_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer");
  m.def("multi_tensor_adagrad", &multi_tensor_adagrad_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer");
  m.def("multi_tensor_novograd", &multi_tensor_novograd_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer");
  m.def("multi_tensor_lamb", &multi_tensor_lamb_cuda,
        "Computes and apply update for LAMB optimizer");
  m.def("multi_tensor_lamb_mp", &multi_tensor_lamb_mp_cuda,
        "Computes and apply update for LAMB optimizer");
}
