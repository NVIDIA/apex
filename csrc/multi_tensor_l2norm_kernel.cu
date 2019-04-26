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

template<typename x_t>
struct L2NormFunctor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<1>& tl,
    float* output)
  {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    x_t* x = (x_t*)tl.addresses[0][tensor_loc];
    x += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    __shared__ float vals[512];

    // Non-divergent exit condition for __syncthreads, not necessary here
    float val = 0;
    for(int i = threadIdx.x; i < n && i < chunk_size; i += blockDim.x)
    {
      float next = static_cast<float>(x[i]);
      val += next*next;
    }

    float final = reduce_block_into_lanes(vals, val);

    if(threadIdx.x == 0)
    {
      if(!isfinite(final))
        *noop_gmem = 1; // Blindly fire off a write.  These will race but that's ok.
      output[blockIdx.x] += final;
    }
  }
};

__global__ void cleanup(float* x, float* ret)
{
  __shared__ float vals[512];

  float val = 0;
  if(threadIdx.x < 320)
    val = x[threadIdx.x];

  float final = reduce_block_into_lanes(vals, val);

  if(threadIdx.x == 0)
    *ret = sqrt(final);
}

at::Tensor multi_tensor_l2norm_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists)
{
  auto output = at::zeros({320}, tensor_lists[0][0].options().dtype(at::kFloat));

  DISPATCH_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(), 0, "multi_tensor_l2norm_cuda",
    multi_tensor_apply<1>(
      BLOCK_SIZE,
      chunk_size,
      noop_flag,
      tensor_lists,
      L2NormFunctor<scalar_t_0>(),
      output.data<float>());)

  AT_CUDA_CHECK(cudaGetLastError());

  // AT_CUDA_CHECK(cudaDeviceSynchronize());

  // This involves one more small kernel launches, but will be negligible end to end.
  // I could get rid of these by hacking the functor + multi tensor harness with persistence
  // logic, but keeping it simple for now
  auto ret = at::empty({1}, output.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  cleanup<<<1, 512, 0, stream>>>(output.data<float>(), ret.data<float>());
  return ret;
}
