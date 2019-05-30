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
template<typename GRAD_T, typename T>
struct LAMBStage1Functor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<5>& tl,
    const float* per_tensor_grad_norm,
    const float* per_tensor_decay,
    const float b1,
    const float b2,
    const float eps,
    const float grad_global_scale,
    adamMode_t mode)
  {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int tensor_num = tl.start_tensor_this_launch + tensor_loc;
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    float grad_scale = per_tensor_grad_norm[tensor_num];
    float decay = per_tensor_decay[tensor_num];

    GRAD_T* g = (GRAD_T*)tl.addresses[0][tensor_loc];
    g += chunk_idx*chunk_size;

    float* p = (float*)tl.addresses[1][tensor_loc];
    p += chunk_idx*chunk_size;

    float* m = (float*)tl.addresses[2][tensor_loc];
    m += chunk_idx*chunk_size;

    float* v = (float*)tl.addresses[3][tensor_loc];
    v += chunk_idx*chunk_size;

    float* out = (float*)tl.addresses[4][tensor_loc];
    out += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    // see note in multi_tensor_scale_kernel.cu
#pragma unroll
    for(int ii = 0; ii < ILP; ii++)
    {
      int i = i_start + threadIdx.x + ii*blockDim.x;
      if(i < n && i < chunk_size)
      {
        T scaled_grad = g[i]/grad_scale;
        m[i] = b1*m[i] + (1-b1)*scaled_grad;
        v[i] = b2*v[i] + (1-b2)*scaled_grad*scaled_grad;
        float denom;
        if (mode == ADAM_MODE_0)
          denom = sqrtf(v[i] + eps);
        else // Mode 1
          denom = sqrtf(v[i]) + eps;
        float update = (m[i]/denom) + (decay*p[i]);
        out[i] = update;
      }
    }
  }
};

void multi_tensor_lamb_stage1_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor per_tensor_grad_norm,
  at::Tensor per_tensor_decay,
  const float b1,
  const float b2,
  const float eps,
  const float grad_global_scale,
  adamMode_t mode)
{
  using namespace at;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensor_lists[0][0].scalar_type(), "lamb_stage_1", [&] {
      using accscalar_t = acc_type<scalar_t_0, true>;
      multi_tensor_apply<5>(
        BLOCK_SIZE,
        chunk_size,
        noop_flag,
        tensor_lists,
        LAMBStage1Functor<scalar_t_0, accscalar_t>(),
        per_tensor_grad_norm.data<float>(),
        per_tensor_decay.data<float>(),
        b1,
        b2,
        eps,
        grad_global_scale,
        mode); )

  AT_CUDA_CHECK(cudaGetLastError());

  // AT_CUDA_CHECK(cudaDeviceSynchronize());
}
