#include <ATen/ATen.h>
#include <torch/library.h>

// CUDA forward declaration
void fused_strided_check_finite(at::Tensor& overflow_flag, at::Tensor& p_copy, int stride, int clear_overflow_first);

void fused_adam_cuda(at::Tensor& p, at::Tensor& p_copy, at::Tensor& m, at::Tensor& v, at::Tensor& g, float lr,
                     float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction,
                     float decay);
void fused_reversible_adam_cuda(at::Tensor& p, at::Tensor& p_copy, at::Tensor& m, at::Tensor& v, at::Tensor& g,
                                float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode,
                                int bias_correction, float decay);
void fused_maybe_adam_undo_cuda(at::Tensor& overflow_flag, at::Tensor& p, at::Tensor& m, at::Tensor& v, at::Tensor& g,
                                float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode,
                                int bias_correction, float decay);

void fused_adam_cuda_mt(int chunk_size, at::Tensor overflow_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
                        float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode,
                        int bias_correction, float decay);

void maybe_cast_cuda(at::Tensor& overflow_flag, at::Tensor& p_in, at::Tensor& p_out);
void maybe_cast_cuda_mt(int chunk_size, at::Tensor overflow_flag, std::vector<std::vector<at::Tensor>> tensor_lists);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// C++ interface
void strided_check_finite(at::Tensor& overflow_flag, at::Tensor& p_copy, int stride, int clear_overflow_first) {
  CHECK_INPUT(p_copy);
  fused_strided_check_finite(overflow_flag, p_copy, stride, clear_overflow_first);
}
void adam(at::Tensor& p, at::Tensor& p_copy, at::Tensor& m, at::Tensor& v, at::Tensor& g, float lr, float beta1,
          float beta2, float eps, float grad_scale, int step, int mode, int bias_correction, float decay) {
  CHECK_INPUT(p);
  if (p_copy.numel() > 0) CHECK_INPUT(p_copy);
  CHECK_INPUT(m);
  CHECK_INPUT(v);
  CHECK_INPUT(g);
  int64_t num_elem = p.numel();
  TORCH_CHECK(m.numel() == num_elem, "number of elements in m and p tensors should be equal");
  TORCH_CHECK(v.numel() == num_elem, "number of elements in v and p tensors should be equal");
  TORCH_CHECK(g.numel() == num_elem, "number of elements in g and p tensors should be equal");
  TORCH_CHECK(p_copy.numel() == num_elem || p_copy.numel() == 0,
              "number of elements in p_copy and p tensors should be equal, or p_copy should be empty");

  fused_adam_cuda(p, p_copy, m, v, g, lr, beta1, beta2, eps, grad_scale, step, mode, bias_correction, decay);
}
void reversible_adam(at::Tensor& p, at::Tensor& p_copy, at::Tensor& m, at::Tensor& v, at::Tensor& g, float lr,
                     float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction,
                     float decay) {
  CHECK_INPUT(p);
  if (p_copy.numel() > 0) CHECK_INPUT(p_copy);
  CHECK_INPUT(m);
  CHECK_INPUT(v);
  CHECK_INPUT(g);
  int64_t num_elem = p.numel();
  TORCH_CHECK(m.numel() == num_elem, "number of elements in m and p tensors should be equal");
  TORCH_CHECK(v.numel() == num_elem, "number of elements in v and p tensors should be equal");
  TORCH_CHECK(g.numel() == num_elem, "number of elements in g and p tensors should be equal");
  TORCH_CHECK(p_copy.numel() == num_elem || p_copy.numel() == 0,
              "number of elements in p_copy and p tensors should be equal, or p_copy should be empty");

  fused_reversible_adam_cuda(p, p_copy, m, v, g, lr, beta1, beta2, eps, grad_scale, step, mode, bias_correction, decay);
}
void maybe_adam_undo(at::Tensor& overflow_flag, at::Tensor& p, at::Tensor& m, at::Tensor& v, at::Tensor& g, float lr,
                     float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction,
                     float decay) {
  CHECK_INPUT(p);
  CHECK_INPUT(m);
  CHECK_INPUT(v);
  CHECK_INPUT(g);
  int64_t num_elem = p.numel();
  TORCH_CHECK(m.numel() == num_elem, "number of elements in m and p tensors should be equal");
  TORCH_CHECK(v.numel() == num_elem, "number of elements in v and p tensors should be equal");
  TORCH_CHECK(g.numel() == num_elem, "number of elements in g and p tensors should be equal");

  fused_maybe_adam_undo_cuda(overflow_flag, p, m, v, g, lr, beta1, beta2, eps, grad_scale, step, mode, bias_correction,
                             decay);
}
void maybe_cast(at::Tensor& overflow_flag, at::Tensor& p_in, at::Tensor& p_out) {
  CHECK_INPUT(p_in);
  CHECK_INPUT(p_out);
  int64_t num_elem = p_in.numel();
  TORCH_CHECK(p_out.numel() == num_elem, "number of elements in p_in and p_out should be equal");

  maybe_cast_cuda(overflow_flag, p_in, p_out);
}

void strided_check_finite_dispatch(const at::Tensor& overflow_flag, const at::Tensor& p_copy, int64_t stride,
                                   int64_t clear_overflow_first) {
  at::Tensor overflow_flag_arg = overflow_flag;
  at::Tensor p_copy_arg = p_copy;
  strided_check_finite(overflow_flag_arg, p_copy_arg, static_cast<int>(stride), static_cast<int>(clear_overflow_first));
}

void adam_dispatch(const at::Tensor& p, const at::Tensor& p_copy, const at::Tensor& m, const at::Tensor& v,
                   const at::Tensor& g, double lr, double beta1, double beta2, double eps, double grad_scale,
                   int64_t step, int64_t mode, int64_t bias_correction, double decay) {
  at::Tensor p_arg = p;
  at::Tensor p_copy_arg = p_copy;
  at::Tensor m_arg = m;
  at::Tensor v_arg = v;
  at::Tensor g_arg = g;
  adam(p_arg, p_copy_arg, m_arg, v_arg, g_arg, static_cast<float>(lr), static_cast<float>(beta1),
       static_cast<float>(beta2), static_cast<float>(eps), static_cast<float>(grad_scale), static_cast<int>(step),
       static_cast<int>(mode), static_cast<int>(bias_correction), static_cast<float>(decay));
}

void reversible_adam_dispatch(const at::Tensor& p, const at::Tensor& p_copy, const at::Tensor& m, const at::Tensor& v,
                              const at::Tensor& g, double lr, double beta1, double beta2, double eps,
                              double grad_scale, int64_t step, int64_t mode, int64_t bias_correction, double decay) {
  at::Tensor p_arg = p;
  at::Tensor p_copy_arg = p_copy;
  at::Tensor m_arg = m;
  at::Tensor v_arg = v;
  at::Tensor g_arg = g;
  reversible_adam(p_arg, p_copy_arg, m_arg, v_arg, g_arg, static_cast<float>(lr), static_cast<float>(beta1),
                  static_cast<float>(beta2), static_cast<float>(eps), static_cast<float>(grad_scale),
                  static_cast<int>(step), static_cast<int>(mode), static_cast<int>(bias_correction),
                  static_cast<float>(decay));
}

void fused_adam_cuda_mt_dispatch(int64_t chunk_size, at::Tensor overflow_flag,
                                 std::vector<std::vector<at::Tensor>> tensor_lists, double lr, double beta1,
                                 double beta2, double eps, double grad_scale, int64_t step, int64_t mode,
                                 int64_t bias_correction, double decay) {
  fused_adam_cuda_mt(static_cast<int>(chunk_size), overflow_flag, tensor_lists, static_cast<float>(lr),
                     static_cast<float>(beta1), static_cast<float>(beta2), static_cast<float>(eps),
                     static_cast<float>(grad_scale), static_cast<int>(step), static_cast<int>(mode),
                     static_cast<int>(bias_correction), static_cast<float>(decay));
}

void maybe_adam_undo_dispatch(const at::Tensor& overflow_flag, const at::Tensor& p, const at::Tensor& m,
                              const at::Tensor& v, const at::Tensor& g, double lr, double beta1, double beta2,
                              double eps, double grad_scale, int64_t step, int64_t mode, int64_t bias_correction,
                              double decay) {
  at::Tensor overflow_flag_arg = overflow_flag;
  at::Tensor p_arg = p;
  at::Tensor m_arg = m;
  at::Tensor v_arg = v;
  at::Tensor g_arg = g;
  maybe_adam_undo(overflow_flag_arg, p_arg, m_arg, v_arg, g_arg, static_cast<float>(lr), static_cast<float>(beta1),
                  static_cast<float>(beta2), static_cast<float>(eps), static_cast<float>(grad_scale),
                  static_cast<int>(step), static_cast<int>(mode), static_cast<int>(bias_correction),
                  static_cast<float>(decay));
}

void maybe_cast_dispatch(const at::Tensor& overflow_flag, const at::Tensor& p_in, const at::Tensor& p_out) {
  at::Tensor overflow_flag_arg = overflow_flag;
  at::Tensor p_in_arg = p_in;
  at::Tensor p_out_arg = p_out;
  maybe_cast(overflow_flag_arg, p_in_arg, p_out_arg);
}

void maybe_cast_cuda_mt_dispatch(int64_t chunk_size, at::Tensor overflow_flag,
                                 std::vector<std::vector<at::Tensor>> tensor_lists) {
  maybe_cast_cuda_mt(static_cast<int>(chunk_size), overflow_flag, tensor_lists);
}

TORCH_LIBRARY_FRAGMENT(apex, m) {
  m.def("fused_adam_strided_check_finite(Tensor(a!) overflow_flag, Tensor p_copy, int stride, "
        "int clear_overflow_first) -> ()");
  m.def("fused_adam_adam(Tensor(a!) p, Tensor(b!) p_copy, Tensor(c!) m, Tensor(d!) v, Tensor g, "
        "float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, "
        "int bias_correction, float decay) -> ()");
  m.def("fused_adam_reversible_adam(Tensor(a!) p, Tensor(b!) p_copy, Tensor(c!) m, Tensor(d!) v, Tensor g, "
        "float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, "
        "int bias_correction, float decay) -> ()");
  m.def("fused_adam_adam_mt(int chunk_size, Tensor overflow_flag, Tensor[][] tensor_lists, float lr, "
        "float beta1, float beta2, float eps, float grad_scale, int step, int mode, int bias_correction, "
        "float decay) -> ()");
  m.def("fused_adam_maybe_adam_undo(Tensor overflow_flag, Tensor(a!) p, Tensor(b!) m, Tensor(c!) v, Tensor(d!) g, "
        "float lr, float beta1, float beta2, float eps, float grad_scale, int step, int mode, "
        "int bias_correction, float decay) -> ()");
  m.def("fused_adam_maybe_cast(Tensor overflow_flag, Tensor p_in, Tensor(a!) p_out) -> ()");
  m.def("fused_adam_maybe_cast_mt(int chunk_size, Tensor overflow_flag, Tensor[][] tensor_lists) -> ()");
}

TORCH_LIBRARY_IMPL(apex, CUDA, m) {
  m.impl("fused_adam_strided_check_finite", &strided_check_finite_dispatch);
  m.impl("fused_adam_adam", &adam_dispatch);
  m.impl("fused_adam_reversible_adam", &reversible_adam_dispatch);
  m.impl("fused_adam_adam_mt", &fused_adam_cuda_mt_dispatch);
  m.impl("fused_adam_maybe_adam_undo", &maybe_adam_undo_dispatch);
  m.impl("fused_adam_maybe_cast", &maybe_cast_dispatch);
  m.impl("fused_adam_maybe_cast_mt", &maybe_cast_cuda_mt_dispatch);
}
