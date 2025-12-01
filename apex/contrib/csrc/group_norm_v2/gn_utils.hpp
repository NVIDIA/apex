#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "gn.hpp"

// Definition of CUDA_CHECK macro
#define CUDA_CHECK(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err_ = call;                                                                                           \
    if (err_ != cudaSuccess) {                                                                                         \
      fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", __FILE__, __LINE__, err_, cudaGetErrorString(err_), \
              #call);                                                                                                  \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

#define GN_CUDA_HOST_PARAMS(T)                                                                                      \
  T *out, T *x, T *w, T *b, float eps, bool silu, int64_t n, int64_t hw, int num_groups, int channels_per_group,    \
      float *mean_var_out, float *red_buffer, unsigned *barrier, int sm_margin, cudaStream_t stream, int device_id, \
      Meta *meta_ptr, bool meta_only

#define GN_BWD_CUDA_HOST_PARAMS(T)                                                                                    \
  T *grad_input, T *grad_weight, T *grad_bias, T *grad_output, T *x, T *w, T *b, float *mean_var, float eps,          \
      bool silu, int64_t n, int64_t hw, int num_groups, int channels_per_group, float *red_buffer, unsigned *barrier, \
      int sm_margin, cudaStream_t stream, int device_id, Meta *meta_ptr, bool meta_only

#define GN_CUDA_HOST_ARGS                                                                                       \
  out, x, w, b, eps, silu, n, hw, num_groups, channels_per_group, mean_var_out, red_buffer, barrier, sm_margin, \
      stream, device_id, meta_ptr, meta_only

#define GN_BWD_CUDA_HOST_ARGS                                                                       \
  grad_input, grad_weight, grad_bias, grad_output, x, w, b, mean_var, eps, silu, n, hw, num_groups, \
      channels_per_group, red_buffer, barrier, sm_margin, stream, device_id, meta_ptr, meta_only

namespace group_norm_v2 {

cudaDeviceProp const& get_device_prop(int device_id);

#ifdef __CUDA_ARCH__

template <class... Ts>
__host__ __device__ inline int print_rank_0(char const* fmt, Ts&&... args) {
  if (threadIdx.x + threadIdx.y + threadIdx.z == 0 && blockIdx.x + blockIdx.y + blockIdx.z == 0) {
    return printf(fmt, std::forward<Ts>(args)...);
  }
  return 0;
}

#endif

}  // namespace group_norm_v2
