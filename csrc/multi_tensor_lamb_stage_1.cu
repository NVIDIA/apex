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

// Step 1 computes the 'update' value of regular Adam optimizer.
template<typename GRAD_T, typename T, typename UPD_T>
struct LAMBStage1Functor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<5>& tl,
    const float* per_tensor_decay,
    const float beta1,
    const float beta2,
    const float beta1_correction,
    const float beta2_correction,
    const float epsilon,
    const float clipped_global_grad_norm)
  {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int tensor_num = tl.start_tensor_this_launch + tensor_loc;
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    float decay = per_tensor_decay[tensor_num];

    GRAD_T* g = (GRAD_T*)tl.addresses[0][tensor_loc];
    g += chunk_idx*chunk_size;

    T* p = (T*)tl.addresses[1][tensor_loc];
    p += chunk_idx*chunk_size;

    T* m = (T*)tl.addresses[2][tensor_loc];
    m += chunk_idx*chunk_size;

    T* v = (T*)tl.addresses[3][tensor_loc];
    v += chunk_idx*chunk_size;

    UPD_T* update = (UPD_T*)tl.addresses[4][tensor_loc];
    update += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    // see note in multi_tensor_scale_kernel.cu
    for(int i_start = 0;
            i_start < n && i_start < chunk_size;
            i_start += blockDim.x*ILP)
    {
      GRAD_T r_g[ILP];
      T r_p[ILP];
      T r_m[ILP];
      T r_v[ILP];
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          r_g[ii] = g[i];
          r_p[ii] = p[i];
          r_m[ii] = m[i];
          r_v[ii] = v[i];
        } else {
          r_g[ii] = GRAD_T(0);
          r_p[ii] = T(0);
          r_m[ii] = T(0);
          r_v[ii] = T(0);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        T scaled_grad = r_g[ii] / clipped_global_grad_norm;
        r_m[ii] = r_m[ii] * beta1 + (1-beta1) * scaled_grad;
        r_v[ii] = r_v[ii] * beta2 + (1-beta2) * scaled_grad * scaled_grad;
        T next_m_unbiased = r_m[ii] / beta1_correction;
        T next_v_unbiased = r_v[ii] / beta2_correction;
        T denom = std::sqrt(next_v_unbiased) + epsilon;
        r_p[ii] = (next_m_unbiased/denom) + (decay*r_p[ii]);
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          update[i] = (UPD_T)r_p[ii];
          m[i] = r_m[ii];
          v[i] = r_v[ii];
        }
      }
    }
  }
};

void multi_tensor_lamb_stage1_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor per_tensor_decay,
  const int step,
  const float beta1,
  const float beta2,
  const float epsilon,
  const float global_grad_norm,
  const float max_global_grad_norm)
{
  using namespace at;

  float clipped_global_grad_norm = global_grad_norm > max_global_grad_norm ? global_grad_norm / max_global_grad_norm : 1.0f;
  float next_step = float(step+1);
  float beta1_correction = 1.0f - std::pow(beta1, next_step);
  float beta2_correction = 1.0f - std::pow(beta2, next_step);
  DISPATCH_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(), 0, "lamb_stage_1",
    DISPATCH_FLOAT_AND_HALF(tensor_lists[1][0].scalar_type(), 1, "lamb_stage_1",
      DISPATCH_FLOAT_AND_HALF(tensor_lists[4][0].scalar_type(), 2, "lamb_stage_1",
        multi_tensor_apply<5>(
          BLOCK_SIZE,
          chunk_size,
          noop_flag,
          tensor_lists,
          LAMBStage1Functor<scalar_t_0, scalar_t_1, scalar_t_2>(),
          per_tensor_decay.DATA_PTR<float>(),
          beta1,
          beta2,
          beta1_correction,
          beta2_correction,
          epsilon,
          clipped_global_grad_norm); )))

  AT_CUDA_CHECK(cudaGetLastError());

  // AT_CUDA_CHECK(cudaDeviceSynchronize());
}
