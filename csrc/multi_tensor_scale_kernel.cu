#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
// Another possibility:
// #include <torch/all.h>

#include <assert.h>
// Stringstream is a big hammer, but I want to rely on operator<< for dtype.
#include <sstream>

#include "multi_tensor_apply.cuh"
#include "type_shim.h"

#define BLOCK_SIZE 512
#define ILP 4

template <typename T>
__device__ __forceinline__ bool is_aligned(T* p) {
  return ((uint64_t)p) % (ILP * sizeof(T)) == 0;
}

template <typename T>
__device__ __forceinline__ void load_store(T* dst, T* src, int dst_offset, int src_offset) {
  typedef typename std::aligned_storage<ILP * sizeof(T), ILP * alignof(T)>::type LT;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

template <typename in_t, typename out_t, typename index_t>
struct ScaleFunctor {
  __device__ __forceinline__ void operator()(index_t chunk_size, volatile int* noop_gmem, TensorListMetadata<2>& tl,
                                             float scale) {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    index_t tensor_loc = tl.block_to_tensor[blockIdx.x];
    index_t chunk_idx = tl.block_to_chunk[blockIdx.x];
    index_t n = tl.sizes[tensor_loc];

    in_t* in = (in_t*)tl.addresses[0][tensor_loc];
    in += chunk_idx * chunk_size;

    out_t* out = (out_t*)tl.addresses[1][tensor_loc];
    out += chunk_idx * chunk_size;

    n -= chunk_idx * chunk_size;

    bool finite = true;
    in_t r_in[ILP];
    out_t r_out[ILP];

    // to make things simple, we put aligned case in a different code path
    if (n % ILP == 0 && chunk_size % ILP == 0 && is_aligned(in) && is_aligned(out)) {
      for (index_t i_start = threadIdx.x; i_start * ILP < n && i_start * ILP < chunk_size; i_start += blockDim.x) {
        // load
        load_store(r_in, in, 0, i_start);
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          r_out[ii] = static_cast<float>(r_in[ii]) * scale;
          finite = finite && isfinite(r_in[ii]);
        }
        // store
        load_store(out, r_out, i_start, 0);
      }
    } else {
      // Non-divergent exit condition for __syncthreads, not necessary here
      for (index_t i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          r_in[ii] = 0;
          index_t i = i_start + threadIdx.x + ii * blockDim.x;
          if (i < n && i < chunk_size) r_in[ii] = in[i];
        }
        // note for clarification to future michael:
        // From a pure memory dependency perspective, there's likely no point unrolling
        // the write loop, since writes just fire off once their LDGs arrive.
        // Put another way, the STGs are dependent on the LDGs, but not on each other.
        // There is still compute ILP benefit from unrolling the loop though.
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          r_out[ii] = static_cast<float>(r_in[ii]) * scale;
          finite = finite && isfinite(r_in[ii]);
        }
#pragma unroll
        for (int ii = 0; ii < ILP; ii++) {
          index_t i = i_start + threadIdx.x + ii * blockDim.x;
          if (i < n && i < chunk_size) out[i] = r_out[ii];
        }
      }
    }
    if (!finite) *noop_gmem = 1;  // Blindly fire off a write.  These will race but that's ok.
  }
};

void multi_tensor_scale_cuda(int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
                             float scale) {
  using namespace at;
  // The output (downscaled) type is always float.
  // If build times suffer, think about where to put this dispatch,
  // and what logic should be moved out of multi_tensor_apply.

  bool requires_64bit_indexing = false;
  for (auto it = tensor_lists.begin(); it != tensor_lists.end(); it++) {
    for (auto it2 = it->begin(); it2 != it->end(); it2++) {
      if (it2->numel() >= INT_MAX) {
        requires_64bit_indexing = true;
        break;
      }
    }
    if (requires_64bit_indexing) break;
  }

  if (requires_64bit_indexing) {
    DISPATCH_FLOAT_HALF_AND_BFLOAT(
        tensor_lists[0][0].scalar_type(), 0, "multi_tensor_scale_cuda",
        DISPATCH_FLOAT_HALF_AND_BFLOAT(
            tensor_lists[1][0].scalar_type(), 1, "multi_tensor_scale_cuda",
            multi_tensor_apply<2>((int64_t)BLOCK_SIZE, (int64_t)chunk_size, noop_flag, tensor_lists,
                                  ScaleFunctor<scalar_t_0, scalar_t_1, int64_t>(), scale);))
  } else {
    DISPATCH_FLOAT_HALF_AND_BFLOAT(
        tensor_lists[0][0].scalar_type(), 0, "multi_tensor_scale_cuda",
        DISPATCH_FLOAT_HALF_AND_BFLOAT(tensor_lists[1][0].scalar_type(), 1, "multi_tensor_scale_cuda",
                                       multi_tensor_apply<2>(BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                                                             ScaleFunctor<scalar_t_0, scalar_t_1, int32_t>(), scale);))
  }
  AT_CUDA_CHECK(cudaGetLastError());

  // AT_CUDA_CHECK(cudaDeviceSynchronize());
}
