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

using MATH_T = float;

// Step 2 reads in 'update' value and per-tensor param_norm and update_norm.
// It computes new parameter value.
template<typename T, typename UPD_T>
struct LAMBStage2Functor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<2>& tl,
    const float* per_tensor_param_norm,
    const float* per_tensor_update_norm,
    const float learning_rate,
    const float decay,
    bool use_nvlamb)
  {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int tensor_num = tl.start_tensor_this_launch + tensor_loc;
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    MATH_T ratio = learning_rate;
    // nvlamb: apply adaptive learning rate to all parameters
    // otherwise, only apply to those with non-zero weight decay
    if (use_nvlamb || (decay != 0.0))
    {
      float param_norm = per_tensor_param_norm[tensor_num];
      float update_norm = per_tensor_update_norm[tensor_num];
      ratio = (update_norm != 0.0f && param_norm != 0.0f) ? learning_rate * (param_norm / update_norm) : learning_rate;
    }

    T* p = (T*)tl.addresses[0][tensor_loc];
    p += chunk_idx*chunk_size;

    UPD_T* update = (UPD_T*)tl.addresses[1][tensor_loc];
    update += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    for(int i_start = 0;
            i_start < n && i_start < chunk_size;
            i_start += blockDim.x*ILP)
    {
      T r_p[ILP];
      UPD_T r_update[ILP];
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          r_p[ii] = p[i];
          r_update[ii] = update[i];
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        r_p[ii] = r_p[ii] - (ratio*(T)r_update[ii]);
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          p[i] = r_p[ii];
        }
      }
    }
  }
};

void multi_tensor_lamb_stage2_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor per_tensor_param_norm,
  at::Tensor per_tensor_update_norm,
  const float lr,
  const float weight_decay,
  at::optional<bool> use_nvlamb_python)
{
  bool use_nvlamb = use_nvlamb_python.has_value() ? use_nvlamb_python.value() : false;

  using namespace at;

  DISPATCH_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(), 0, "lamb_stage_2",
    DISPATCH_FLOAT_AND_HALF(tensor_lists[1][0].scalar_type(), 1, "lamb_stage_2",
      multi_tensor_apply<2>(
        BLOCK_SIZE,
        chunk_size,
        noop_flag,
        tensor_lists,
        LAMBStage2Functor<scalar_t_0, scalar_t_1>(),
        per_tensor_param_norm.DATA_PTR<float>(),
        per_tensor_update_norm.DATA_PTR<float>(),
        lr,
	weight_decay,
	use_nvlamb); ))

  AT_CUDA_CHECK(cudaGetLastError());

  // AT_CUDA_CHECK(cudaDeviceSynchronize());
}
