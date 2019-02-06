#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#include <assert.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024
#define NBLOCKS 160

// It makes sense to lock the output type to fp32 because the downscaled
// grads should be master grads (and in the case of Amp, the params and their
// gradients should always be fp32.

// This can be optimized with ILP but it's fine for now.
template<typename in_t>
__global__ void scale_reduce_overflow(in_t* in,
                                      float* out,
                                      int n,
                                      float scale,
                                      volatile int* overflow_global)
{
  __shared__ int overflow;

  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int stride = gridDim.x*blockDim.x;

  // Non-divergent exit condition for the __syncthreads
  for(int i = tid; i - threadIdx.x < n; i += stride)
  {
    if(threadIdx.x == 0)
      overflow = *overflow_global;

    __syncthreads();

    if(overflow == 1)
      break;

    if(i < n)
    {
      float incoming_val = static_cast<float>(in[i]);
      if(isfinite(incoming_val))
        out[i] = incoming_val*scale;
      else
        *overflow_global = 1; // Blindly fire off a write.  These will race but that's ok.
        // This is NOT guaranteed to be seen immediately by thread 0 on the next iteration.
        // I wonder if there's a way we can rig the short-circuiting with only one syncthreads.
        // It's possible we can just lean on the cache (no smem or syncs) and still be fast.
    }
  }
}


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
