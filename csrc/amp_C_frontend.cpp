#include <torch/library.h>

void multi_tensor_scale_cuda(int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
                             float scale);

void multi_tensor_sgd_cuda(int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
                           float wd, float momentum, float dampening, float lr, bool nesterov, bool first_run,
                           bool wd_after_momentum, float scale);

void multi_tensor_axpby_cuda(int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
                             float a, float b, int arg_to_check);

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_cuda(int chunk_size, at::Tensor noop_flag,
                                                            std::vector<std::vector<at::Tensor>> tensor_lists,
                                                            at::optional<bool> per_tensor_python);

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_mp_cuda(int chunk_size, at::Tensor noop_flag,
                                                               std::vector<std::vector<at::Tensor>> tensor_lists,
                                                               at::optional<bool> per_tensor_python);

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_scale_cuda(int chunk_size, at::Tensor noop_flag,
                                                                  std::vector<std::vector<at::Tensor>> tensor_lists,
                                                                  float scale, at::optional<bool> per_tensor_python);

std::tuple<at::Tensor, at::Tensor> multi_tensor_unscale_l2norm_cuda(int chunk_size, at::Tensor noop_flag,
                                                                    std::vector<std::vector<at::Tensor>> tensor_lists,
                                                                    at::Tensor inv_scale,
                                                                    at::optional<bool> per_tensor_python);

void multi_tensor_lamb_stage1_cuda(int chunk_size, at::Tensor noop_flag,
                                   std::vector<std::vector<at::Tensor>> tensor_lists, at::Tensor per_tensor_decay,
                                   const int step, const float beta1, const float beta2, const float epsilon,
                                   at::Tensor global_grad_norm, const float max_global_grad_norm);

void multi_tensor_lamb_stage2_cuda(int chunk_size, at::Tensor noop_flag,
                                   std::vector<std::vector<at::Tensor>> tensor_lists, at::Tensor per_tensor_param_norm,
                                   at::Tensor per_tensor_update_norm, const float lr, const float weight_decay,
                                   at::optional<bool> use_nvlamb_python);

void multi_tensor_adam_cuda(int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
                            const float lr, const float beta1, const float beta2, const float epsilon, const int step,
                            const int mode, const int bias_correction, const float weight_decay);

void multi_tensor_adam_capturable_cuda(int chunk_size, at::Tensor noop_flag,
                                       std::vector<std::vector<at::Tensor>> tensor_lists, at::Tensor lr,
                                       const float beta1, const float beta2, const float epsilon, at::Tensor step,
                                       const int mode, const int bias_correction, const float weight_decay,
                                       at::Tensor inv_scale);

void multi_tensor_adam_capturable_master_cuda(int chunk_size, at::Tensor noop_flag,
                                              std::vector<std::vector<at::Tensor>> tensor_lists, at::Tensor lr,
                                              const float beta1, const float beta2, const float epsilon,
                                              at::Tensor step, const int mode, const int bias_correction,
                                              const float weight_decay, at::Tensor inv_scale);

void multi_tensor_adagrad_cuda(int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
                               const float lr, const float epsilon, const int mode, const float weight_decay);

void multi_tensor_novograd_cuda(int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
                                at::Tensor grad_norms, const float lr, const float beta1, const float beta2,
                                const float epsilon, const int step, const int bias_correction,
                                const float weight_decay, const int grad_averaging, const int mode,
                                const int norm_type);

void multi_tensor_lamb_cuda(int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
                            const float lr, const float beta1, const float beta2, const float epsilon, const int step,
                            const int bias_correction, const float weight_decay, const int grad_averaging,
                            const int mode, at::Tensor global_grad_norm, const float max_grad_norm,
                            at::optional<bool> use_nvlamb_python);

void multi_tensor_lamb_mp_cuda(int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
                               at::Tensor lr, const float beta1, const float beta2, const float epsilon,
                               at::Tensor step, const int bias_correction, const float weight_decay,
                               const int grad_averaging, const int mode, at::Tensor global_grad_norm,
                               at::Tensor max_grad_norm, at::optional<bool> use_nvlamb_python, at::Tensor found_inf,
                               at::Tensor inv_scale);

at::Tensor update_scale_hysteresis_cuda(at::Tensor current_scale, at::Tensor growth_tracker,
                                        at::Tensor hysteresis_tracker, at::Tensor found_inf, const double growth_factor,
                                        const double backoff_factor, const int64_t growth_interval,
                                        const int hysteresis);

namespace {
void apex_multi_tensor_scale(int64_t chunk_size, at::Tensor noop_flag,
                             std::vector<std::vector<at::Tensor>> tensor_lists, double scale) {
  multi_tensor_scale_cuda(static_cast<int>(chunk_size), noop_flag, std::move(tensor_lists), static_cast<float>(scale));
}

void apex_multi_tensor_sgd(int64_t chunk_size, at::Tensor noop_flag,
                           std::vector<std::vector<at::Tensor>> tensor_lists, double wd, double momentum,
                           double dampening, double lr, bool nesterov, bool first_run, bool wd_after_momentum,
                           double scale) {
  multi_tensor_sgd_cuda(static_cast<int>(chunk_size), noop_flag, std::move(tensor_lists), static_cast<float>(wd),
                        static_cast<float>(momentum), static_cast<float>(dampening), static_cast<float>(lr), nesterov,
                        first_run, wd_after_momentum, static_cast<float>(scale));
}

void apex_multi_tensor_axpby(int64_t chunk_size, at::Tensor noop_flag,
                             std::vector<std::vector<at::Tensor>> tensor_lists, double a, double b,
                             int64_t arg_to_check) {
  multi_tensor_axpby_cuda(static_cast<int>(chunk_size), noop_flag, std::move(tensor_lists), static_cast<float>(a),
                          static_cast<float>(b), static_cast<int>(arg_to_check));
}

std::tuple<at::Tensor, at::Tensor> apex_multi_tensor_l2norm(
    int64_t chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
    at::optional<bool> per_tensor_python) {
  return multi_tensor_l2norm_cuda(static_cast<int>(chunk_size), noop_flag, std::move(tensor_lists), per_tensor_python);
}

std::tuple<at::Tensor, at::Tensor> apex_multi_tensor_l2norm_mp(
    int64_t chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
    at::optional<bool> per_tensor_python) {
  return multi_tensor_l2norm_mp_cuda(static_cast<int>(chunk_size), noop_flag, std::move(tensor_lists),
                                    per_tensor_python);
}

std::tuple<at::Tensor, at::Tensor> apex_multi_tensor_l2norm_scale(
    int64_t chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists, double scale,
    at::optional<bool> per_tensor_python) {
  return multi_tensor_l2norm_scale_cuda(static_cast<int>(chunk_size), noop_flag, std::move(tensor_lists),
                                       static_cast<float>(scale), per_tensor_python);
}

std::tuple<at::Tensor, at::Tensor> apex_multi_tensor_unscale_l2norm(
    int64_t chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists, at::Tensor inv_scale,
    at::optional<bool> per_tensor_python) {
  return multi_tensor_unscale_l2norm_cuda(static_cast<int>(chunk_size), noop_flag, std::move(tensor_lists), inv_scale,
                                         per_tensor_python);
}

void apex_multi_tensor_lamb_stage1_cuda(int64_t chunk_size, at::Tensor noop_flag,
                                        std::vector<std::vector<at::Tensor>> tensor_lists,
                                        at::Tensor per_tensor_decay, int64_t step, double beta1, double beta2,
                                        double epsilon, at::Tensor global_grad_norm, double max_global_grad_norm) {
  multi_tensor_lamb_stage1_cuda(static_cast<int>(chunk_size), noop_flag, std::move(tensor_lists), per_tensor_decay,
                                static_cast<int>(step), static_cast<float>(beta1), static_cast<float>(beta2),
                                static_cast<float>(epsilon), global_grad_norm, static_cast<float>(max_global_grad_norm));
}

void apex_multi_tensor_lamb_stage2_cuda(int64_t chunk_size, at::Tensor noop_flag,
                                        std::vector<std::vector<at::Tensor>> tensor_lists,
                                        at::Tensor per_tensor_param_norm, at::Tensor per_tensor_update_norm, double lr,
                                        double weight_decay, at::optional<bool> use_nvlamb_python) {
  multi_tensor_lamb_stage2_cuda(static_cast<int>(chunk_size), noop_flag, std::move(tensor_lists),
                                per_tensor_param_norm, per_tensor_update_norm, static_cast<float>(lr),
                                static_cast<float>(weight_decay), use_nvlamb_python);
}

void apex_multi_tensor_adam(int64_t chunk_size, at::Tensor noop_flag,
                            std::vector<std::vector<at::Tensor>> tensor_lists, double lr, double beta1, double beta2,
                            double epsilon, int64_t step, int64_t mode, int64_t bias_correction, double weight_decay) {
  multi_tensor_adam_cuda(static_cast<int>(chunk_size), noop_flag, std::move(tensor_lists), static_cast<float>(lr),
                         static_cast<float>(beta1), static_cast<float>(beta2), static_cast<float>(epsilon),
                         static_cast<int>(step), static_cast<int>(mode), static_cast<int>(bias_correction),
                         static_cast<float>(weight_decay));
}

void apex_multi_tensor_adam_capturable(int64_t chunk_size, at::Tensor noop_flag,
                                       std::vector<std::vector<at::Tensor>> tensor_lists, at::Tensor lr, double beta1,
                                       double beta2, double epsilon, at::Tensor step, int64_t mode,
                                       int64_t bias_correction, double weight_decay, at::Tensor inv_scale) {
  multi_tensor_adam_capturable_cuda(static_cast<int>(chunk_size), noop_flag, std::move(tensor_lists), lr,
                                    static_cast<float>(beta1), static_cast<float>(beta2), static_cast<float>(epsilon),
                                    step, static_cast<int>(mode), static_cast<int>(bias_correction),
                                    static_cast<float>(weight_decay), inv_scale);
}

void apex_multi_tensor_adam_capturable_master(int64_t chunk_size, at::Tensor noop_flag,
                                              std::vector<std::vector<at::Tensor>> tensor_lists, at::Tensor lr,
                                              double beta1, double beta2, double epsilon, at::Tensor step,
                                              int64_t mode, int64_t bias_correction, double weight_decay,
                                              at::Tensor inv_scale) {
  multi_tensor_adam_capturable_master_cuda(
      static_cast<int>(chunk_size), noop_flag, std::move(tensor_lists), lr, static_cast<float>(beta1),
      static_cast<float>(beta2), static_cast<float>(epsilon), step, static_cast<int>(mode),
      static_cast<int>(bias_correction), static_cast<float>(weight_decay), inv_scale);
}

void apex_multi_tensor_adagrad(int64_t chunk_size, at::Tensor noop_flag,
                               std::vector<std::vector<at::Tensor>> tensor_lists, double lr, double epsilon,
                               int64_t mode, double weight_decay) {
  multi_tensor_adagrad_cuda(static_cast<int>(chunk_size), noop_flag, std::move(tensor_lists), static_cast<float>(lr),
                            static_cast<float>(epsilon), static_cast<int>(mode), static_cast<float>(weight_decay));
}

void apex_multi_tensor_novograd(int64_t chunk_size, at::Tensor noop_flag,
                                std::vector<std::vector<at::Tensor>> tensor_lists, at::Tensor grad_norms, double lr,
                                double beta1, double beta2, double epsilon, int64_t step, int64_t bias_correction,
                                double weight_decay, int64_t grad_averaging, int64_t mode, int64_t norm_type) {
  multi_tensor_novograd_cuda(static_cast<int>(chunk_size), noop_flag, std::move(tensor_lists), grad_norms,
                             static_cast<float>(lr), static_cast<float>(beta1), static_cast<float>(beta2),
                             static_cast<float>(epsilon), static_cast<int>(step), static_cast<int>(bias_correction),
                             static_cast<float>(weight_decay), static_cast<int>(grad_averaging),
                             static_cast<int>(mode), static_cast<int>(norm_type));
}

void apex_multi_tensor_lamb(int64_t chunk_size, at::Tensor noop_flag,
                            std::vector<std::vector<at::Tensor>> tensor_lists, double lr, double beta1, double beta2,
                            double epsilon, int64_t step, int64_t bias_correction, double weight_decay,
                            int64_t grad_averaging, int64_t mode, at::Tensor global_grad_norm, double max_grad_norm,
                            at::optional<bool> use_nvlamb_python) {
  multi_tensor_lamb_cuda(static_cast<int>(chunk_size), noop_flag, std::move(tensor_lists), static_cast<float>(lr),
                         static_cast<float>(beta1), static_cast<float>(beta2), static_cast<float>(epsilon),
                         static_cast<int>(step), static_cast<int>(bias_correction), static_cast<float>(weight_decay),
                         static_cast<int>(grad_averaging), static_cast<int>(mode), global_grad_norm,
                         static_cast<float>(max_grad_norm), use_nvlamb_python);
}

void apex_multi_tensor_lamb_mp(int64_t chunk_size, at::Tensor noop_flag,
                               std::vector<std::vector<at::Tensor>> tensor_lists, at::Tensor lr, double beta1,
                               double beta2, double epsilon, at::Tensor step, int64_t bias_correction,
                               double weight_decay, int64_t grad_averaging, int64_t mode,
                               at::Tensor global_grad_norm, at::Tensor max_grad_norm,
                               at::optional<bool> use_nvlamb_python, at::Tensor found_inf, at::Tensor inv_scale) {
  multi_tensor_lamb_mp_cuda(static_cast<int>(chunk_size), noop_flag, std::move(tensor_lists), lr,
                            static_cast<float>(beta1), static_cast<float>(beta2), static_cast<float>(epsilon), step,
                            static_cast<int>(bias_correction), static_cast<float>(weight_decay),
                            static_cast<int>(grad_averaging), static_cast<int>(mode), global_grad_norm, max_grad_norm,
                            use_nvlamb_python, found_inf, inv_scale);
}

at::Tensor apex_update_scale_hysteresis(at::Tensor current_scale, at::Tensor growth_tracker,
                                        at::Tensor hysteresis_tracker, at::Tensor found_inf, double growth_factor,
                                        double backoff_factor, int64_t growth_interval, int64_t hysteresis) {
  return update_scale_hysteresis_cuda(current_scale, growth_tracker, hysteresis_tracker, found_inf, growth_factor,
                                      backoff_factor, growth_interval, static_cast<int>(hysteresis));
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(apex, m) {
  m.def("amp_multi_tensor_scale(int chunk_size, Tensor noop_flag, Tensor[][] tensor_lists, float scale) -> ()");
  m.def("amp_multi_tensor_sgd(int chunk_size, Tensor noop_flag, Tensor[][] tensor_lists, float wd, float momentum, "
        "float dampening, float lr, bool nesterov, bool first_run, bool wd_after_momentum, float scale) -> ()");
  m.def("amp_multi_tensor_axpby(int chunk_size, Tensor noop_flag, Tensor[][] tensor_lists, float a, float b, "
        "int arg_to_check) -> ()");
  m.def("amp_multi_tensor_l2norm(int chunk_size, Tensor noop_flag, Tensor[][] tensor_lists, bool? per_tensor_python) "
        "-> (Tensor, Tensor)");
  m.def("amp_multi_tensor_l2norm_mp(int chunk_size, Tensor noop_flag, Tensor[][] tensor_lists, "
        "bool? per_tensor_python) -> (Tensor, Tensor)");
  m.def("amp_multi_tensor_l2norm_scale(int chunk_size, Tensor noop_flag, Tensor[][] tensor_lists, float scale, "
        "bool? per_tensor_python) -> (Tensor, Tensor)");
  m.def("amp_multi_tensor_unscale_l2norm(int chunk_size, Tensor noop_flag, Tensor[][] tensor_lists, Tensor inv_scale, "
        "bool? per_tensor_python) -> (Tensor, Tensor)");
  m.def("amp_multi_tensor_lamb_stage1_cuda(int chunk_size, Tensor noop_flag, Tensor[][] tensor_lists, "
        "Tensor per_tensor_decay, int step, float beta1, float beta2, float epsilon, Tensor global_grad_norm, "
        "float max_global_grad_norm) -> ()");
  m.def("amp_multi_tensor_lamb_stage2_cuda(int chunk_size, Tensor noop_flag, Tensor[][] tensor_lists, "
        "Tensor per_tensor_param_norm, Tensor per_tensor_update_norm, float lr, float weight_decay, "
        "bool? use_nvlamb_python) -> ()");
  m.def("amp_multi_tensor_adam(int chunk_size, Tensor noop_flag, Tensor[][] tensor_lists, float lr, float beta1, "
        "float beta2, float epsilon, int step, int mode, int bias_correction, float weight_decay) -> ()");
  m.def("amp_multi_tensor_adam_capturable(int chunk_size, Tensor noop_flag, Tensor[][] tensor_lists, Tensor lr, "
        "float beta1, float beta2, float epsilon, Tensor step, int mode, int bias_correction, float weight_decay, "
        "Tensor inv_scale) -> ()");
  m.def("amp_multi_tensor_adam_capturable_master(int chunk_size, Tensor noop_flag, Tensor[][] tensor_lists, Tensor lr, "
        "float beta1, float beta2, float epsilon, Tensor step, int mode, int bias_correction, float weight_decay, "
        "Tensor inv_scale) -> ()");
  m.def("amp_multi_tensor_adagrad(int chunk_size, Tensor noop_flag, Tensor[][] tensor_lists, float lr, float epsilon, "
        "int mode, float weight_decay) -> ()");
  m.def("amp_multi_tensor_novograd(int chunk_size, Tensor noop_flag, Tensor[][] tensor_lists, Tensor grad_norms, "
        "float lr, float beta1, float beta2, float epsilon, int step, int bias_correction, float weight_decay, "
        "int grad_averaging, int mode, int norm_type) -> ()");
  m.def("amp_multi_tensor_lamb(int chunk_size, Tensor noop_flag, Tensor[][] tensor_lists, float lr, float beta1, "
        "float beta2, float epsilon, int step, int bias_correction, float weight_decay, int grad_averaging, "
        "int mode, Tensor global_grad_norm, float max_grad_norm, bool? use_nvlamb_python) -> ()");
  m.def("amp_multi_tensor_lamb_mp(int chunk_size, Tensor noop_flag, Tensor[][] tensor_lists, Tensor lr, float beta1, "
        "float beta2, float epsilon, Tensor step, int bias_correction, float weight_decay, int grad_averaging, "
        "int mode, Tensor global_grad_norm, Tensor max_grad_norm, bool? use_nvlamb_python, Tensor found_inf, "
        "Tensor inv_scale) -> ()");
  m.def("amp_update_scale_hysteresis(Tensor current_scale, Tensor growth_tracker, Tensor hysteresis_tracker, "
        "Tensor found_inf, float growth_factor, float backoff_factor, int growth_interval, int hysteresis) -> Tensor");
}

TORCH_LIBRARY_IMPL(apex, CUDA, m) {
  m.impl("amp_multi_tensor_scale", &apex_multi_tensor_scale);
  m.impl("amp_multi_tensor_sgd", &apex_multi_tensor_sgd);
  m.impl("amp_multi_tensor_axpby", &apex_multi_tensor_axpby);
  m.impl("amp_multi_tensor_l2norm", &apex_multi_tensor_l2norm);
  m.impl("amp_multi_tensor_l2norm_mp", &apex_multi_tensor_l2norm_mp);
  m.impl("amp_multi_tensor_l2norm_scale", &apex_multi_tensor_l2norm_scale);
  m.impl("amp_multi_tensor_unscale_l2norm", &apex_multi_tensor_unscale_l2norm);
  m.impl("amp_multi_tensor_lamb_stage1_cuda", &apex_multi_tensor_lamb_stage1_cuda);
  m.impl("amp_multi_tensor_lamb_stage2_cuda", &apex_multi_tensor_lamb_stage2_cuda);
  m.impl("amp_multi_tensor_adam", &apex_multi_tensor_adam);
  m.impl("amp_multi_tensor_adam_capturable", &apex_multi_tensor_adam_capturable);
  m.impl("amp_multi_tensor_adam_capturable_master", &apex_multi_tensor_adam_capturable_master);
  m.impl("amp_multi_tensor_adagrad", &apex_multi_tensor_adagrad);
  m.impl("amp_multi_tensor_novograd", &apex_multi_tensor_novograd);
  m.impl("amp_multi_tensor_lamb", &apex_multi_tensor_lamb);
  m.impl("amp_multi_tensor_lamb_mp", &apex_multi_tensor_lamb_mp);
  m.impl("amp_update_scale_hysteresis", &apex_update_scale_hysteresis);
}
