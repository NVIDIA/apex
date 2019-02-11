#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include "multi_tensor_apply.h"

#include <assert.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define ILP 4

template<typename in_t>
struct UnscaleFunctor
{
  __device__ __forceinline__ void operator()(
    volatile int* noop_flag,
    int tensor_idx,
    int chunk_idx,
    int chunk_size,
    int n,
    void** addresses,
    int addresses_x,
    float scale)
  {
    __shared__ int noop;

    in_t* in = (in_t*)addresses[tensor_idx];
    in += chunk_idx*chunk_size;
   
    float* out = (float*)addresses[addresses_x + tensor_idx];
    out += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    // Non-divergent exit condition for the __syncthreads
    float incoming_vals[ILP];
    for(int i_start = 0;
        i_start < n && i_start < chunk_size;
        i_start += blockDim.x*ILP)
    {
      if(threadIdx.x == 0)
        noop = *noop_flag;

      __syncthreads();

      if(noop == 1)
        break;

      #pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        incoming_vals[ii] = 0;
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n)
          incoming_vals[ii] = static_cast<float>(in[i]);
      }

      #pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n)
          if(isfinite(incoming_vals[ii]))
            out[i] = incoming_vals[ii]*scale;
          else
            *noop_flag = 1; // Blindly fire off a write.  These will race but that's ok.
      }    // This is NOT guaranteed to be seen immediately by thread 0 on the next iteration.
    }      // I wonder if there's a way we can rig the short-circuiting with only one syncthreads.
  }        // It's possible we can just lean on the cache (no smem or syncs) and still be fast.
};


void multi_tensor_unscale_cuda(
  int nblocks,
  at::Tensor noop_flag,
  at::Tensor cpu_tensor_addresses,
  at::Tensor gpu_block_to_tensor,
  at::Tensor gpu_block_to_chunk,
  at::Tensor gpu_tensor_sizes,
  at::Tensor gpu_tensor_addresses,
  int chunk_size,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  float scale)
{
  using namespace at;

  int addresses_x = gpu_tensor_addresses.size(1);

  // <.< >.> i don't see any cops. i'm going to access the pointers directly.
  // auto addresses_a = cpu_tensor_addresses.accessor<int64_t, 2>();
  // This logic could be moved to prep_multi_tensor_launch, but we might need to
  // pick which kernel instantiation to launch based on the RTTI of tensor_lists,
  // so we may as well accept tensor_lists and extract the pointers here.
  void** addresses_a = (void**)cpu_tensor_addresses.data_ptr();

  int len0 = tensor_lists[0].size();
  for(unsigned int l = 0; l < tensor_lists.size(); l++)
  {
    AT_CHECK(tensor_lists[l].size() == len0, "Lengths of tensor lists do not match.");
    for(unsigned int t = 0; t < tensor_lists[l].size(); t++)
    {
      AT_CHECK(tensor_lists[l][t].numel() == tensor_lists[0][t].numel(),
        "Numel mismatch in corresponding tensors in different lists.");
      addresses_a[l*addresses_x + t] = tensor_lists[l][t].data_ptr();
      // addresses_a[l][t] = (void*)tensor_lists[l][t].data<float>();
    }
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  gpu_tensor_addresses.copy_(cpu_tensor_addresses, 1/*non_blocking*/);
 
  // Lock the output (downscaled) type to float.
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(tensor_lists[0][0].type(),
     "multi_tensor_unscale_cuda",
     [&]
     {
       // using accscalar_t = acc_type<scalar_t, true>;
       multi_tensor_apply_kernel<<<nblocks, BLOCK_SIZE, 0, stream>>>(
         noop_flag.data<int>(),
         gpu_block_to_tensor.data<int>(),
         gpu_block_to_chunk.data<int>(),
         gpu_tensor_sizes.data<int>(),
         chunk_size,
         (void**)gpu_tensor_addresses.data_ptr(),
         addresses_x,
         UnscaleFunctor<scalar_t>(),
         scale);
     });

  AT_CUDA_CHECK(cudaGetLastError());
  // AT_CUDA_CHECK(cudaDeviceSynchronize());
}
