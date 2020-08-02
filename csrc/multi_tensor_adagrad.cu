#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
// Another possibility:
// #include <torch/all.h>

#include <assert.h>

#include "multi_tensor_apply.cuh"
#include "type_shim.h"

#define BLOCK_SIZE 1024
#define ILP 4

typedef enum {
  ADAGRAD_MODE_0 = 0, // L2 regularization mode.
  ADAGRAD_MODE_1 = 1, // AdamW-style weight decay.

} adagradMode_t;

using MATH_T = float;

template <typename T> struct AdagradFunctor {
  __device__ __forceinline__ void
  operator()(int chunk_size, volatile int *noop_gmem, TensorListMetadata<3> &tl,
             const float epsilon, const float lr, adagradMode_t mode,
             const float weight_decay) {
    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    T *g = (T *)tl.addresses[0][tensor_loc];
    g += chunk_idx * chunk_size;

    T *p = (T *)tl.addresses[1][tensor_loc];
    p += chunk_idx * chunk_size;

    T *h = (T *)tl.addresses[2][tensor_loc];
    h += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    // see note in multi_tensor_scale_kernel.cu
    for (int i_start = 0; i_start < n && i_start < chunk_size;
         i_start += blockDim.x * ILP) {
      MATH_T r_g[ILP];
      MATH_T r_p[ILP];
      MATH_T r_h[ILP];
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          r_g[ii] = g[i];
          r_p[ii] = p[i];
          r_h[ii] = h[i];
        } else {
          r_g[ii] = MATH_T(0);
          r_p[ii] = MATH_T(0);
          r_h[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        if (mode == ADAGRAD_MODE_0) { // L2
          r_g[ii] = r_g[ii] + weight_decay * r_p[ii];
          r_h[ii] = r_h[ii] + r_g[ii] * r_g[ii];
          r_p[ii] = r_p[ii] - lr * (r_g[ii] / (sqrtf(r_h[ii]) + epsilon));
        } else { // AdamW-style
          r_h[ii] = r_h[ii] + r_g[ii] * r_g[ii];
          r_p[ii] = r_p[ii] - lr * (r_g[ii] / (sqrtf(r_h[ii]) + epsilon) + weight_decay * r_p[ii]);
        }
      }
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          p[i] = r_p[ii];
          h[i] = r_h[ii];
        }
      }
    }
  }
};

void multi_tensor_adagrad_cuda(
    int chunk_size, at::Tensor noop_flag,
    std::vector<std::vector<at::Tensor>> tensor_lists, const float lr,
    const float epsilon, const int mode, const float weight_decay) {
  using namespace at;

  // Assume single type across p,g,h now
  DISPATCH_DOUBLE_FLOAT_AND_HALF(
      tensor_lists[0][0].scalar_type(), 0, "adagrad",
      multi_tensor_apply<3>(BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                            AdagradFunctor<scalar_t_0>(), epsilon, lr,
                            (adagradMode_t)mode, weight_decay);)

  AT_CUDA_CHECK(cudaGetLastError());
}
