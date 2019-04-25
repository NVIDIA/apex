#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
// Another possibility:
// #include <torch/all.h>

#include <assert.h>
// Stringstream is a big hammer, but I want to rely on operator<< for dtype.
#include <sstream>

#include "type_shim.h"
#include "multi_tensor_apply.cuh"

#define BLOCK_SIZE 512
#define ILP 4

template<typename in_t, typename out_t>
struct ScaleFunctor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<2>& tl,
    float scale)
  {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    in_t* in = (in_t*)tl.addresses[0][tensor_loc];
    in += chunk_idx*chunk_size;
   
    out_t* out = (out_t*)tl.addresses[1][tensor_loc];
    out += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    // Non-divergent exit condition for __syncthreads, not necessary here
    float incoming_vals[ILP];
    for(int i_start = 0;
        i_start < n && i_start < chunk_size;
        i_start += blockDim.x*ILP)
    {
      #pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        incoming_vals[ii] = 0;
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
          incoming_vals[ii] = static_cast<float>(in[i]);
      }

      // note for clarification to future michael:
      // From a pure memory dependency perspective, there's likely no point unrolling
      // the write loop, since writes just fire off once their LDGs arrive.
      // Put another way, the STGs are dependent on the LDGs, but not on each other.
      // There is still compute ILP benefit from unrolling the loop though.
      #pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          out[i] = static_cast<out_t>(incoming_vals[ii]*scale);
          if(!isfinite(incoming_vals[ii]))
            *noop_gmem = 1; // Blindly fire off a write.  These will race but that's ok.
        }
      }
    }
  }
};

void multi_tensor_scale_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  float scale)
{
  using namespace at;
  // The output (downscaled) type is always float.
  // If build times suffer, think about where to put this dispatch,
  // and what logic should be moved out of multi_tensor_apply.

  DISPATCH_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(), 0, "multi_tensor_scale_cuda",
    DISPATCH_FLOAT_AND_HALF(tensor_lists[1][0].scalar_type(), 1, "multi_tensor_scale_cuda",
      multi_tensor_apply<2>(
        BLOCK_SIZE,
        chunk_size,
        noop_flag,
        tensor_lists,
        ScaleFunctor<scalar_t_0, scalar_t_1>(),
        scale); ))
  AT_CUDA_CHECK(cudaGetLastError());

  // AT_CUDA_CHECK(cudaDeviceSynchronize());
}
