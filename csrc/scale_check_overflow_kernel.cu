#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#include <assert.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define NBLOCKS 160*4
#define ILP 4

// It makes sense to lock the output type to fp32 because the downscaled
// grads should be master grads (and in the case of Amp, the params and their
// gradients should always be fp32).

template<typename in_t>
__global__ void scale_reduce_overflow(in_t* in,
                                      float* out,
                                      int n,
                                      float scale,
                                      volatile int* overflow_global)
{
  __shared__ int overflow;
  float incoming_vals[4];

  // Non-divergent exit condition for the __syncthreads
  for(int chunk_start = blockIdx.x*blockDim.x*ILP;
      chunk_start < n;
      chunk_start += gridDim.x*blockDim.x*ILP)
  {
    if(threadIdx.x == 0)
      overflow = *overflow_global;

    __syncthreads();

    if(overflow == 1)
      break;

    #pragma unroll
    for(int ii = 0; ii < ILP; ii++)
    {
      incoming_vals[ii] = 0;
      int i = chunk_start + threadIdx.x + ii*blockDim.x;
      if(i < n)
        incoming_vals[ii] = static_cast<float>(in[i]);
    }

    #pragma unroll
    for(int ii = 0; ii < ILP; ii++)
    {
      int i = chunk_start + threadIdx.x + ii*blockDim.x;
      if(i < n)
        if(isfinite(incoming_vals[ii]))
          out[i] = incoming_vals[ii]*scale;
        else
          *overflow_global = 1; // Blindly fire off a write.  These will race but that's ok.
    }    // This is NOT guaranteed to be seen immediately by thread 0 on the next iteration.
  }      // I wonder if there's a way we can rig the short-circuiting with only one syncthreads.
}        // It's possible we can just lean on the cache (no smem or syncs) and still be fast.


void scale_check_overflow_cuda
  (const at::Tensor& grads,
   float scale,
   const at::Tensor& overflow_buf,
   const at::Tensor& downscaled_grads)
{
  using namespace at;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  int n = grads.numel();

  // Lock the output (downscaled) type to float.
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grads.type(),
     "scale_check_overflow_cuda",
     [&]
     {
       // using accscalar_t = acc_type<scalar_t, true>;
       scale_reduce_overflow<<<NBLOCKS, BLOCK_SIZE, 0, stream>>>
         (grads.data<scalar_t>(),
          downscaled_grads.data<float>(),
          n,
          scale,
          overflow_buf.data<int>());
     });

  AT_CUDA_CHECK(cudaGetLastError());
}
