#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
// Another possibility:
// #include <torch/all.h>

#include <assert.h>

#include "type_shim.h"
#include "multi_tensor_apply.cuh"

#define BLOCK_SIZE 512
#define ILP 4

typedef enum{
  MOMENT_MODE_0   =0, // Novograd paper mode, momentum caculation with denom then decay inside
  MOMENT_MODE_1   =1  // Decoupled weight decay mode
} momentMode_t;

void multi_tensor_norm_out_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor out,
  const float alpha,
  const float beta,
  const int norm_type);

using MATH_T = float;

template<typename T>
struct NovoGradFunctor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<3>& tl,
    const float beta1,
    const float beta2,
    const float beta3,
    const float beta1_correction,
    const float beta2_correction,
    const float epsilon,
    const float lr,
    momentMode_t m_mode,
    const float decay,
    const float* per_tensor_grad_norm)
  {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int tensor_num = tl.start_tensor_this_launch + tensor_loc;
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    float grad_norm = per_tensor_grad_norm[tensor_num];

    T* g = (T*)tl.addresses[0][tensor_loc];
    g += chunk_idx*chunk_size;

    T* p = (T*)tl.addresses[1][tensor_loc];
    p += chunk_idx*chunk_size;

    T* m = (T*)tl.addresses[2][tensor_loc];
    m += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    // see note in multi_tensor_scale_kernel.cu
    for(int i_start = 0;
            i_start < n && i_start < chunk_size;
            i_start += blockDim.x*ILP)
    {
      MATH_T r_g[ILP];
      MATH_T r_p[ILP];
      MATH_T r_m[ILP];
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          r_g[ii] = g[i];
          r_p[ii] = p[i];
          r_m[ii] = m[i];
        } else {
          r_g[ii] = MATH_T(0);
          r_p[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        if (m_mode == MOMENT_MODE_0) {
          MATH_T next_v_unbiased = grad_norm / beta2_correction;
          MATH_T denom = next_v_unbiased + epsilon;
          r_g[ii] = (r_g[ii] / denom) + (decay * r_p[ii]);
          r_m[ii] = beta1 * r_m[ii] + beta3 * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          r_p[ii] = r_p[ii] - (lr * next_m_unbiased);
        }
        else {
          r_m[ii] = beta1 * r_m[ii] + beta3 * r_g[ii];
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = grad_norm / beta2_correction;
          MATH_T denom = next_v_unbiased + epsilon;
          MATH_T update = (next_m_unbiased / denom) + (decay * r_p[ii]);
          r_p[ii] = r_p[ii] - (lr * update);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          p[i] = r_p[ii];
          m[i] = r_m[ii];
        }
      }
    }
  }
};

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
  const int moment_mode,
  const int norm_type)
{
  using namespace at;

  // Handle bias correction mode
  float bias_correction1 = 1.0f, bias_correction2 = 1.0f;
  if (bias_correction == 1) {
    bias_correction1 = 1 - std::pow(beta1, step);
    bias_correction2 = std::sqrt(1 - std::pow(beta2, step));
  }

  // Handle grad averaging mode
  float beta3 = 1;
  if (grad_averaging == 1) beta3 = 1 - beta1;

  std::vector<std::vector<at::Tensor>> grad_list(tensor_lists.begin(), tensor_lists.begin()+1);

  // Compute and update grad norm
  // Here use a per tensor norm, and blend new norm(n) and old norm(gn) by
  // L-2: gn = sqrt(a * gn^2 + b * n^2)
  // L-inf: gn = a * gn + b * n
  multi_tensor_norm_out_cuda(chunk_size, noop_flag, grad_list, grad_norms, beta2, (1.0f - beta2), norm_type);

  // Assume single type across p,g,m1,m2 now
  DISPATCH_DOUBLE_FLOAT_AND_HALF(
    tensor_lists[0][0].scalar_type(), 0, "novograd",
    multi_tensor_apply<3>(
      BLOCK_SIZE,
      chunk_size,
      noop_flag,
      tensor_lists,
      NovoGradFunctor<scalar_t_0>(),
      beta1,
      beta2,
      beta3, // 1-beta1 or 1 depends on averaging mode
      bias_correction1,
      bias_correction2,
      epsilon,
      lr,
      (momentMode_t) moment_mode,
      weight_decay,
      grad_norms.DATA_PTR<float>()); )

  AT_CUDA_CHECK(cudaGetLastError());

}
