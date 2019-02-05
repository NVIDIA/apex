#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#include <assert.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 1024
#define MAX_BLOCKS 1024

// It makes sense to lock the output type to fp32 because the downscaled
// grads should be master grads (and in the case of Amp, the params and their
// gradients should always be fp32.

template<typename in_t>
__global__ void scale_reduce_overflow(in_t* in,
                                      float* out, 
                                      size_t n, 
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
    
    if(tid < n)
    {
      float incoming_val = static_cast<float>(in[i]);
      if(isfinite(incoming_val))
        out[i] = incoming_val*scale;
      else
        *overflow_global = 1; // Blindly fire off a write.  These will race but that's ok.
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
  
  size_t n = grads.numel();

  int num_blks = 160;
 
  // Lock the output (downscaled) type to float.
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grads.type(),
     "scale_check_overflow_cuda",
     [&]
     {
       // using accscalar_t = acc_type<scalar_t, true>;
       scale_reduce_overflow<<<num_blks, BLOCK_SIZE, 0, stream>>>
         (grads.data<scalar_t>(), 
          downscaled_grads.data<float>(),
          n, 
          scale, 
          overflow_buf.data<int>());
     });

  AT_CUDA_CHECK(cudaGetLastError());
}
