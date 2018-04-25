#include "../include/kernel.h"

template<typename T> struct TtoInt { static const int test = -1; };
template<> struct TtoInt<float> { static const int test = 0; }; 
template<> struct TtoInt<half> { static const int test = 0; }; 
template<> struct TtoInt<double> { static const int test = 0; }; 

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

// Block size for weight_norm_*_first_dim_kernel.
// Currently, kernels are non-persistent.
// Dialing up the block size to, say 1024, can improve performance by
// increase the amount of cache available per block, which can improve cache hit rate.
// However, this is less efficient for short rows.  256 is pretty versatile. 
// Implement some heuristics later?
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

using namespace std;

// lanes is intended to be <= 32.
template <typename T>
__device__ __forceinline__ void reduce_block_into_lanes(T *x, T val, int lanes) 
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
      x[tid] += x[tid+i]; // JoinOp
    __syncthreads();
  }

  if(tid < 32) 
  {
    T final;
    if(blockSize >= 64)
      final = x[tid] + x[tid+32]; // JoinOp
    else
      final = val;
    // __SYNCWARP();

    #pragma unroll
    for(int i = 16; i >= lanes; i >>= 1)
      final += __SHFL_DOWN(final, i);

    if(tid < lanes) 
      x[tid] = final; // EpilogueOp
  }

  // Make sure the smem result is visible to all warps.
  __syncthreads();
}

template <typename T, typename IndexType>
__global__ void weight_norm_fwd_first_dim_kernel
(
  TensorInfo<T, IndexType> w,
  TensorInfo<float, IndexType> norms,
  TensorInfo<T, IndexType> v,
  TensorInfo<T, IndexType> g,
  IndexType rowSize
)
{
  // We are norming each slowest-dim row of the tensor separately.
  // For now, assign one block to each row.
  IndexType tid = threadIdx.x;
  IndexType row = blockIdx.x;
  IndexType stride = blockDim.x;

  // Logical index offset for this flattened row
  IndexType rowStart = row*rowSize;

  extern __shared__ float s[];
  
  float thread_sum = 0.f;
  for(IndexType i = tid; i < rowSize; i += stride ) 
  {
    float val_f = ScalarConvert<T, float>::to(DEVICE_LINEAR_GET(v, i + rowStart)); 
    thread_sum += val_f*val_f; // AccumOp, could do Kahan here
  }

  reduce_block_into_lanes(s, thread_sum, 1);
  float result = s[0];

  result = sqrtf(result);
  
  if(tid == 0)
    DEVICE_LINEAR_GET_F(norms, row) = result;

  // Broadcast load, could use shared memory instead.
  float g_this_row = ScalarConvert<T, float>::to(DEVICE_LINEAR_GET(g, row));

  float rnorm = 1.f/result; // for consistency with backward kernel

  // Write data to output
  for(IndexType i = tid; i < rowSize; i += stride ) 
  {
    float val_f = ScalarConvert<T, float>::to(DEVICE_LINEAR_GET(v, i + rowStart));
    DEVICE_LINEAR_GET(w, i + rowStart) = ScalarConvert<float,T>::to(g_this_row*val_f*rnorm);
  }
}

template <typename T, typename IndexType>
__global__ void weight_norm_fwd_last_dim_kernel
(
  TensorInfo<T, IndexType> w,
  TensorInfo<float, IndexType> norms,
  TensorInfo<T, IndexType> v,
  TensorInfo<T, IndexType> g,
  IndexType fast_dim_size,
  IndexType slower_dims_size
)
{
  IndexType fast_dim_location = threadIdx.x + blockIdx.x*blockDim.x;

  extern __shared__ float alloc[];
  float* s = &alloc[0];
  float* rnorms_this_block = &alloc[blockDim.x*blockDim.y];

  float thread_sum = 0.f;

  IndexType slower_dims_location = threadIdx.y;
  IndexType currentIdx = fast_dim_location + fast_dim_size*slower_dims_location;
  if(fast_dim_location < fast_dim_size)
    while(slower_dims_location < slower_dims_size)
    {
      float val_f = ScalarConvert<T, float>::to(DEVICE_LINEAR_GET(v, currentIdx)); 
      thread_sum += val_f*val_f; // AccumOp, could do Kahan here
      currentIdx += blockDim.y*fast_dim_size;
      slower_dims_location += blockDim.y; 
    }

  reduce_block_into_lanes(s, thread_sum, blockDim.x); 

  // Better to pass an EpilogueOp to reduce_block_into_lanes, can try later
  if(threadIdx.y == 0)
  {
    float result = s[threadIdx.x];
    float norm_this_col = sqrtf(result);
    DEVICE_LINEAR_GET_F(norms, fast_dim_location) = norm_this_col;
    rnorms_this_block[threadIdx.x] = 1.f/norm_this_col;
    // printf("blockIdx.x = %d, threadIdx.x = %d, norm_this_col  = %f\n", 
    //         blockIdx.x,      threadIdx.x,      norm_this_col);
  }
   
  __syncthreads(); 

  float g_this_col = ScalarConvert<T, float>::to(DEVICE_LINEAR_GET(g, fast_dim_location));     

  float rnorm = rnorms_this_block[threadIdx.x]; 

  slower_dims_location = threadIdx.y;
  currentIdx = fast_dim_location + fast_dim_size*slower_dims_location;
  if(fast_dim_location < fast_dim_size)
    while(slower_dims_location < slower_dims_size)
    {
      float val_f = ScalarConvert<T, float>::to(DEVICE_LINEAR_GET(v, currentIdx)); 
      DEVICE_LINEAR_GET(w, currentIdx) = ScalarConvert<float,T>::to(g_this_col*val_f*rnorm);
      currentIdx += blockDim.y*fast_dim_size;
      slower_dims_location += blockDim.y; 
    } 
}

template <typename T, typename IndexType>
__global__ void weight_norm_bwd_first_dim_kernel
(
  TensorInfo<T, IndexType> pLpv,
  TensorInfo<T, IndexType> pLpg,
  TensorInfo<T, IndexType> pLpw,
  TensorInfo<T, IndexType> savedv,
  TensorInfo<T, IndexType> savedg,
  TensorInfo<float, IndexType> savedNorms,
  IndexType rowSize
)
{
  // For now, assign one block to each row.
  IndexType tid = threadIdx.x;
  IndexType row = blockIdx.x;
  IndexType stride = blockDim.x;

  // Logical index offset for this flattened row
  IndexType rowStart = row*rowSize;

  extern __shared__ float s[];
  
  float thread_sum = 0.f;
  for(IndexType i = tid; i < rowSize; i += stride ) 
  {
    float pLpwi = ScalarConvert<T, float>::to(DEVICE_LINEAR_GET(pLpw, i + rowStart)); 
    float savedvi = ScalarConvert<T, float>::to(DEVICE_LINEAR_GET(savedv, i + rowStart)); 
    thread_sum += pLpwi*savedvi; // AccumOp, could do Kahan here
  }

  reduce_block_into_lanes(s, thread_sum, 1);
  float result = s[0];

  // Could choose to save reciprocal of norm instead I suppose, but norms is probably
  // more handy to keep around.
  // Broadcast load; could use shared memory instead.
  float rnorm = 1.f/DEVICE_LINEAR_GET_F(savedNorms, row);  
  float rnorm3 = rnorm*rnorm*rnorm;

  // Write g gradients.
  if(tid == 0)
    DEVICE_LINEAR_GET(pLpg, row) = ScalarConvert<float, T>::to(result*rnorm);

  // Broadcast load, could use shared memory instead.
  float g_this_row = ScalarConvert<T, float>::to(DEVICE_LINEAR_GET(savedg, row));
   
  // Write v gradients.  We are reusing values that were loaded earlier, so there 
  // is an optimization opportunity here (store values persistently).
  for(IndexType j = tid; j < rowSize; j += stride ) 
  {
    float pLpwj = ScalarConvert<T, float>::to(DEVICE_LINEAR_GET(pLpw, j + rowStart));  
    float savedvj = ScalarConvert<T, float>::to(DEVICE_LINEAR_GET(savedv, j + rowStart));  
    float pLpvj = g_this_row*(rnorm*pLpwj - rnorm3*savedvj*result);
    DEVICE_LINEAR_GET(pLpv, j + rowStart) = ScalarConvert<float,T>::to(pLpvj);
  }
}

template <typename T, typename IndexType>
__global__ void weight_norm_bwd_last_dim_kernel
(
  TensorInfo<T, IndexType> pLpv,
  TensorInfo<T, IndexType> pLpg,
  TensorInfo<T, IndexType> pLpw,
  TensorInfo<T, IndexType> savedv,
  TensorInfo<T, IndexType> savedg,
  TensorInfo<float, IndexType> savedNorms,
  IndexType fast_dim_size,
  IndexType slower_dims_size
)
{
  IndexType fast_dim_location = threadIdx.x + blockIdx.x*blockDim.x;

  extern __shared__ float s[];

  float thread_sum = 0.f;

  IndexType slower_dims_location = threadIdx.y;
  IndexType currentIdx = fast_dim_location + fast_dim_size*slower_dims_location;
  if(fast_dim_location < fast_dim_size)
    while(slower_dims_location < slower_dims_size)
    {
      float pLpwi = ScalarConvert<T, float>::to(DEVICE_LINEAR_GET(pLpw, currentIdx)); 
      float savedvi = ScalarConvert<T, float>::to(DEVICE_LINEAR_GET(savedv, currentIdx)); 
      thread_sum += pLpwi*savedvi; // AccumOp, could do Kahan here
      currentIdx += blockDim.y*fast_dim_size;
      slower_dims_location += blockDim.y; 
    }

  reduce_block_into_lanes(s, thread_sum, blockDim.x); 
  float result = s[threadIdx.x];

  // Broadcast load; could use shared memory instead.
  float rnorm = 1.f/DEVICE_LINEAR_GET_F(savedNorms, fast_dim_location);  
  float rnorm3 = rnorm*rnorm*rnorm;

  // Write g gradients.
  if(threadIdx.y == 0)
    DEVICE_LINEAR_GET(pLpg, fast_dim_location) = ScalarConvert<float, T>::to(result*rnorm);

  // Entire block pulls these values, could use shared memory instead.
  float g_this_col = ScalarConvert<T, float>::to(DEVICE_LINEAR_GET(savedg, fast_dim_location));

  // Write v gradients.
  slower_dims_location = threadIdx.y;
  currentIdx = fast_dim_location + fast_dim_size*slower_dims_location;
  if(fast_dim_location < fast_dim_size)
    while(slower_dims_location < slower_dims_size)
    {
      float pLpwj = ScalarConvert<T, float>::to(DEVICE_LINEAR_GET(pLpw, currentIdx));  
      float savedvj = ScalarConvert<T, float>::to(DEVICE_LINEAR_GET(savedv, currentIdx));  
      float pLpvj = g_this_col*(rnorm*pLpwj - rnorm3*savedvj*result);
      DEVICE_LINEAR_GET(pLpv, currentIdx) = ScalarConvert<float,T>::to(pLpvj);
      currentIdx += blockDim.y*fast_dim_size;
      slower_dims_location += blockDim.y; 
    } 
}

template<typename DataType, 
         typename AccumType,
         typename IndexType>
void send_to_fwd_wrapper::call
(
  vector<TensorInfo<void, idxType>>& tensors,
  int dim
)
{
#ifdef DEBUG_ANY
  cout << "hello from send_to_fwd with v.type = " << v.type << endl;
#endif

  auto w    (*((TensorInfo<DataType , idxType>*)&tensors[0]));
  auto norms(*((TensorInfo<AccumType, idxType>*)&tensors[1]));
  auto v    (*((TensorInfo<DataType , idxType>*)&tensors[2]));
  auto g    (*((TensorInfo<DataType , idxType>*)&tensors[3]));

  if(dim == 0) 
  {
    // Find logical size of each flattened slowest-dim row
    IndexType rowSize = 1;
    for(IndexType i = v.dims - 1; i > 0; i--)
      rowSize *= v.sizes[i];

    weight_norm_fwd_first_dim_kernel<<<v.sizes[0], BLOCK, BLOCK*sizeof(float)>>>
    (
      w, 
      norms,
      v,  
      g,  
      rowSize
    );
  }
  else if(dim == v.dims - 1)
  {
    // Precompute slower_dims_size and fast_dim_size because they involve dynamically indexing an array.
    IndexType slower_dims_size = 1;
    for(IndexType i = 0; i < v.dims - 1; i++)
      slower_dims_size *= v.sizes[i];

    int fast_dim_size = v.sizes[v.dims-1];

    weight_norm_fwd_last_dim_kernel<<<(fast_dim_size+TILE_W-1)/TILE_W,
                                       dim3(TILE_W,TILE_H), 
                                       (TILE_W*TILE_H + TILE_W)*sizeof(float)>>>
    (
      w, 
      norms,
      v,  
      g,  
      fast_dim_size,
      slower_dims_size 
    );
  }
  // else 
  // {
  //   intermediate dim kernel.  Error checking on the dim was already done in 
  //   Module.cpp:weight_norm_fwd.  Could put that logic here instead, if we include
  //   <python.h> in both files.
  // }

#ifdef DEBUG_PROFILE
  cudaDeviceSynchronize();
#endif
}

// template <typename T, typename IndexType>
template<typename DataType,
         typename AccumType,
         typename IndexType>
void send_to_bwd_wrapper::call
(
  vector<TensorInfo<void, idxType>>& tensors,
  int dim
)
{
#ifdef DEBUG_ANY
  cout << "Hello from send_to_bwd with pLpw.type = " << pLpw.type << endl;
#endif

  // this feels sinful
  auto pLpv      (*((TensorInfo<DataType , idxType>*)&tensors[0]));
  auto pLpg      (*((TensorInfo<DataType , idxType>*)&tensors[1]));
  auto pLpw      (*((TensorInfo<DataType , idxType>*)&tensors[2]));
  auto savedv    (*((TensorInfo<DataType , idxType>*)&tensors[3]));
  auto savedg    (*((TensorInfo<DataType , idxType>*)&tensors[4]));
  auto savedNorms(*((TensorInfo<AccumType, idxType>*)&tensors[5]));

  if(dim == 0) 
  {
    // Find logical size of each flattened slowest-dim row
    IndexType rowSize = 1;
    for(IndexType i = savedv.dims - 1; i > 0; i--)
      rowSize *= savedv.sizes[i];

    weight_norm_bwd_first_dim_kernel<<<pLpw.sizes[0], BLOCK, BLOCK*sizeof(float)>>>
    (
      pLpv,
      pLpg,
      pLpw,
      savedv,
      savedg,
      savedNorms,
      rowSize
    );
  }
  else if(dim == savedv.dims - 1)
  {
    // Precompute slower_dims_size and fast_dim_size because they involve dynamically indexing an array.
    IndexType slower_dims_size = 1;
    for(IndexType i = 0; i < savedv.dims - 1; i++)
      slower_dims_size *= savedv.sizes[i];

    int fast_dim_size = savedv.sizes[savedv.dims-1];

    weight_norm_bwd_last_dim_kernel<<<(fast_dim_size+TILE_W-1)/TILE_W,
                                       dim3(TILE_W,TILE_H), 
                                       (TILE_W*TILE_H + TILE_W)*sizeof(float)>>>
    (
      pLpv,
      pLpg,
      pLpw,
      savedv,
      savedg,
      savedNorms,
      fast_dim_size,
      slower_dims_size 
    );
  }
  // else 
  // {
  //   intermediate dim kernel.  Error checking on the dim was already done in 
  //   Module.cpp:weight_norm_bwd.  Could put that logic here instead, if we include
  //   <python.h> in both files.
  // }

#ifdef DEBUG_PROFILE
  cudaDeviceSynchronize();
#endif
}

#define INSTANTIATE_SEND_TO_FWD(DATATYPE, ACCUMTYPE, IDXTYPE)         \
template void send_to_fwd_wrapper::call<DATATYPE, ACCUMTYPE, IDXTYPE> \
(                                                                     \
  vector<TensorInfo<void, idxType>>&,                                 \
  int                                                                 \
);
INSTANTIATE_SEND_TO_FWD(float, float, idxType)
INSTANTIATE_SEND_TO_FWD(half, float, idxType)
#undef INSTANTIATE_SEND_TO_FWD

#define INSTANTIATE_SEND_TO_BWD(DATATYPE, ACCUMTYPE, IDXTYPE)         \
template void send_to_bwd_wrapper::call<DATATYPE, ACCUMTYPE, IDXTYPE> \
(                                                                     \
  vector<TensorInfo<void, idxType>>&,                                 \
  int                                                                 \
);                                                            
INSTANTIATE_SEND_TO_BWD(float, float, idxType)
INSTANTIATE_SEND_TO_BWD(half, float, idxType)
#undef INSTANTIATE_SEND_TO_BWD

#undef BLOCK
#undef TILE_W
#undef TILE_H
