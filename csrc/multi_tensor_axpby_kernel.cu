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

template<typename x_t, typename y_t, typename out_t>
struct AxpbyFunctor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<3>& tl,
    float a,
    float b,
    int arg_to_check)
  {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    x_t* x = (x_t*)tl.addresses[0][tensor_loc];
    x += chunk_idx*chunk_size;

    y_t* y = (y_t*)tl.addresses[1][tensor_loc];
    y += chunk_idx*chunk_size;

    out_t* out = (out_t*)tl.addresses[2][tensor_loc];
    out += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    // Non-divergent exit condition for __syncthreads, not necessary here
    float xs[ILP];
    float ys[ILP];
    for(int i_start = 0;
        i_start < n && i_start < chunk_size;
        i_start += blockDim.x*ILP)
    {
      #pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        xs[ii] = 0;
        ys[ii] = 0;
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          xs[ii] = static_cast<float>(x[i]);
          ys[ii] = static_cast<float>(y[i]);
        }
      }

      // see note in multi_tensor_scale_kernel.cu
      #pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          out[i] = static_cast<out_t>(a*xs[ii] + b*ys[ii]);
          bool finite = true;
          if(arg_to_check == -1)
            finite = (isfinite(xs[ii]) && isfinite(ys[ii]));
          if(arg_to_check == 0)
            finite = isfinite(xs[ii]);
          if(arg_to_check == 1)
            finite = isfinite(ys[ii]);
          if(!finite)
            *noop_gmem = 1; // Blindly fire off a write.  These will race but that's ok.
        }
      }
    }
  }
};

void multi_tensor_axpby_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  float a,
  float b,
  int arg_to_check)
{
  using namespace at;
  // The output (downscaled) type is always float.
  // If build times suffer, think about where to put this dispatch,
  // and what logic should be moved out of multi_tensor_apply.

  DISPATCH_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(), 0, "multi_tensor_axpby_cuda",
    DISPATCH_FLOAT_AND_HALF(tensor_lists[1][0].scalar_type(), 1, "multi_tensor_axpby_cuda",
      DISPATCH_FLOAT_AND_HALF(tensor_lists[2][0].scalar_type(), 2, "multi_tensor_axpby_cuda",
           multi_tensor_apply<3>(
             BLOCK_SIZE,
             chunk_size,
             noop_flag,
             tensor_lists,
             AxpbyFunctor<scalar_t_0, scalar_t_1, scalar_t_2>(),
             a,
             b,
             arg_to_check); )))

  AT_CUDA_CHECK(cudaGetLastError());

  // AT_CUDA_CHECK(cudaDeviceSynchronize());
}
