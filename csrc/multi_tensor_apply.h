#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#include <assert.h>
#include <cuda_runtime.h>

template<typename T, typename... ArgTypes>
__global__ void multi_tensor_apply_kernel(
    volatile int* noop_flag,
    int* block_to_tensor,
    int* block_to_chunk, // could also get this from scan
    int* tensor_sizes,
    int chunk_size,
    void** addresses,
    int addresses_x,
    T callable,
    ArgTypes... args) // in_t** in, float** out, float scale
{
  __shared__ int noop;
  __shared__ int chunk_idx;
  __shared__ int tensor_idx;
  __shared__ int n;

  if(threadIdx.x == 0)
  {
    noop = *noop_flag;
    tensor_idx = block_to_tensor[blockIdx.x]; 
    chunk_idx = block_to_chunk[blockIdx.x];
    n = tensor_sizes[tensor_idx]; 
  }

  __syncthreads();

  if(noop == 1)
    return;

  // Hand the chunk information to the user-supplied functor to process however it likes.
  callable(
    noop_flag,
    tensor_idx,
    chunk_idx,
    chunk_size,
    n,
    addresses,
    addresses_x,
    args...);
}
