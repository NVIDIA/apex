#include <cuda.h>
#include <cuda_runtime.h>

#if __CUDACC_VER_MAJOR__ >= 9
#define __SHFL_DOWN(var, delta)  __shfl_down_sync(0xffffffff, var, delta)
#else
#define __SHFL_DOWN(var, delta)  __shfl_down(var, delta)
#endif

#if __CUDACC_VER_MAJOR__ >= 9
#define __SYNCWARP __syncwarp()
#else
#define __SYNCWARP 
#endif

#ifdef VERSION_LE_04                                                        
#define USING_ACCSCALAR_T using accscalar_t = cuda::acc_type<cuda_scalar_t>;
#else                                                                        
#define USING_ACCSCALAR_T using accscalar_t = acc_type<cuda_scalar_t, true>; 
#endif                                                                       

#ifdef VERSION_LE_04                                    
#define REDUCE_ADD ReduceAdd<accscalar_t, accscalar_t>()
#else                                                   
#define REDUCE_ADD ReduceAdd<accscalar_t>()             
#endif                                                  


// Block size for weight_norm_*_first_dim_kernel.
// Currently, kernels are non-persistent.
// Dialing up the block size to, say 1024, can improve performance by
// increase the amount of cache available per block, which can improve cache hit rate.
// However, this is less efficient for short rows.  256 is pretty versatile. 
// May be worth implementing heuristics later.
#define BLOCK 256

// Block size for weight_norm_*_last_dim_kernel.
// This is tricker than the first_dim case because we must make blocks 
// at least 16 fast elements wide to ensure fully-coalesced half-precision accesses.
// Since output-element parallelism is along the fast dimension, this reduces the number of 
// blocks we can launch by 16X.  
#define TILE_W 16
// Somewhat versatile strategy: max out intra-block parallelism by extending
// blocks across the slow dimension up to the hardware-max block size of 1024.
#define TILE_H 64

// For reference, in THCTensorMathReduce.cuh:
// template <typename T>
// struct ReduceAdd {
//   inline __device__ T operator()(const T a, const T b) const {
//     return THCNumerics<T>::add(a, b);
//   }
// };

// lanes is intended to be <= 32.
template 
  <typename T, 
   typename ReduceOp>
__device__ __forceinline__ void reduce_block_into_lanes
  (T *x, 
   T val, 
   int lanes,
   ReduceOp reduceOp) 
{ 
  int tid = threadIdx.x + threadIdx.y*blockDim.x;
  int blockSize = blockDim.x*blockDim.y;

  if(blockSize >= 64)
  {
    x[tid] = val;
    __syncthreads();
  }
  
  #pragma unroll
  for(int i = (blockSize >> 1); i >= 64; i >>= 1) 
  {
    if(tid < i)
      x[tid] = reduceOp(x[tid], x[tid+i]);
    __syncthreads();
  }

  if(tid < 32) 
  {
    T final;
    if(blockSize >= 64)
      final = reduceOp(x[tid], x[tid+32]);
    else
      final = val;
    // __SYNCWARP();

    #pragma unroll
    for(int i = 16; i >= lanes; i >>= 1)
      final = reduceOp(final, __SHFL_DOWN(final, i));

    if(tid < lanes) 
      x[tid] = final; // EpilogueOp
  }

  // Make sure the smem result is visible to all warps.
  __syncthreads();
}
