#include "kernel_utils.cuh"

#include <ATen/ATen.h>

#ifdef VERSION_LE_04
#include "ATen/cuda/AccumulateType.cuh"
#else
#include "ATen/AccumulateType.h"
#endif

#include "ATen/cuda/CUDATensorMethods.cuh"
#include "ATen/cuda/CUDATypeConversion.cuh"
// #include <THC/THCTensorMathReduce.cuh>

template
  <typename scalar_t, 
   typename accscalar_t>
__global__ void weight_norm_fwd_first_dim_kernel
  (scalar_t* __restrict__ w,
   accscalar_t* __restrict__ norms,
   const scalar_t* __restrict__ v,
   const scalar_t* __restrict__ g,
   const int rowSize) 
{
  // We are norming each slowest-dim row of the tensor separately.
  // For now, assign one block to each row.
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const int stride = blockDim.x;

  // Logical index offset for this flattened row
  const int rowStart = row*rowSize;

  // Hack to get around nvcc complaining when an smem array is declared with the same name
  // but different types in different kernels (in this case different instantiations)
  // extern __shared__ accscalar_t s[]; // error: declaration is incompatible with previous "s"
  extern __shared__ char buf[];
  accscalar_t* s = (accscalar_t*)buf;
  
  accscalar_t thread_sum = 0.f;
  for(int i = tid; i < rowSize; i += stride ) 
  {
    accscalar_t val_f = scalar_cast<accscalar_t>(v[i+rowStart]); 
    thread_sum += val_f*val_f; // AccumOp, could do Kahan here
  }

  reduce_block_into_lanes(s, thread_sum, 1, ReduceAdd<accscalar_t>());
  accscalar_t result = s[0];

  result = sqrtf(result);
  
  if(tid == 0)
    norms[row] = result;

  // Broadcast load, could use shared memory instead.
  accscalar_t g_this_row = scalar_cast<accscalar_t>(g[row]);

  accscalar_t rnorm = 1.f/result; // for consistency with backward kernel

  // Write data to output
  for(int i = tid; i < rowSize; i += stride ) 
  {
    accscalar_t val_f = scalar_cast<accscalar_t>(v[i+rowStart]);
    w[i+rowStart] = scalar_cast<scalar_t>(g_this_row*val_f*rnorm);
  }
}

template
  <typename scalar_t, 
   typename accscalar_t>
__global__ void weight_norm_fwd_last_dim_kernel
(
  scalar_t* __restrict__ w,
  accscalar_t* __restrict__ norms,
  const scalar_t* __restrict__ v,
  const scalar_t* __restrict__ g,
  const int fast_dim_size,
  const int slower_dims_size
)
{
  const int fast_dim_location = threadIdx.x + blockIdx.x*blockDim.x;

  extern __shared__ char buf[];
  accscalar_t* alloc = (accscalar_t*)buf;
  accscalar_t* s = &alloc[0];
  accscalar_t* rnorms_this_block = &alloc[blockDim.x*blockDim.y];

  accscalar_t thread_sum = 0.f;

  int slower_dims_location = threadIdx.y;
  int currentIdx = fast_dim_location + fast_dim_size*slower_dims_location;
  if(fast_dim_location < fast_dim_size)
    while(slower_dims_location < slower_dims_size)
    {
      accscalar_t val_f = scalar_cast<accscalar_t>(v[currentIdx]); 
      thread_sum += val_f*val_f; // AccumOp, could do Kahan here
      currentIdx += blockDim.y*fast_dim_size;
      slower_dims_location += blockDim.y; 
    }

  reduce_block_into_lanes(s, thread_sum, blockDim.x, ReduceAdd<accscalar_t>()); 

  // Better to pass an EpilogueOp to reduce_block_into_lanes, implement later
  if(threadIdx.y == 0)
  {
    accscalar_t result = s[threadIdx.x];
    accscalar_t norm_this_col = sqrtf(result);
    norms[fast_dim_location] = norm_this_col;
    rnorms_this_block[threadIdx.x] = 1.f/norm_this_col;
  }
   
  __syncthreads(); 

  accscalar_t g_this_col = scalar_cast<accscalar_t>(g[fast_dim_location]);     
  accscalar_t rnorm = rnorms_this_block[threadIdx.x]; 

  slower_dims_location = threadIdx.y;
  currentIdx = fast_dim_location + fast_dim_size*slower_dims_location;
  if(fast_dim_location < fast_dim_size)
    while(slower_dims_location < slower_dims_size)
    {
      accscalar_t val_f = scalar_cast<accscalar_t>(v[currentIdx]); 
      w[currentIdx] = scalar_cast<scalar_t>(g_this_col*val_f*rnorm);
      currentIdx += blockDim.y*fast_dim_size;
      slower_dims_location += blockDim.y; 
    } 
}


void weight_norm_fwd_cuda
  (const at::Tensor& w,
   const at::Tensor& norms,
   const at::Tensor& v,
   const at::Tensor& g,
   int dim)
{
#ifdef DEBUG_ANY
  using namespace std;
  cout << "hello from send_to_fwd with v.type() = " << v.type() << endl;
#endif

  const int ndims = v.ndimension();

  if(dim == 0) 
  {
    // Find logical size of each flattened slowest-dim row
    int rowSize = 1;
    for(int i = ndims - 1; i > 0; i--)
      rowSize *= v.size(i);

    using namespace at;
    cudaStream_t stream = globalContext().getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF
      (v.type(), 
       "weight_norm_fwd_first_dim_kernel",  
       [&]
       {
         using cuda_scalar_t = apex::cuda::type<scalar_t>;
         USING_ACCSCALAR_T

         weight_norm_fwd_first_dim_kernel
           <<<v.size(0), 
              BLOCK, 
              BLOCK*sizeof(accscalar_t),
              stream>>>
           (w.data<cuda_scalar_t>(), 
            norms.data<accscalar_t>(),
            v.data<cuda_scalar_t>(),  
            g.data<cuda_scalar_t>(),  
            rowSize);
       });
  }
  else if(dim == ndims - 1)
  {
    // Precompute slower_dims_size and fast_dim_size
    int slower_dims_size = 1;
    for(int i = 0; i < ndims - 1; i++)
      slower_dims_size *= v.size(i);

    int fast_dim_size = v.size(ndims-1);
 
    using namespace at;
    cudaStream_t stream = globalContext().getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF
      (v.type(), 
       "weight_norm_fwd_last_dim_kernel",  
       [&]
       {
         using cuda_scalar_t = apex::cuda::type<scalar_t>;
         USING_ACCSCALAR_T
        
         // just trying this formatting out to see how it feels... 
         weight_norm_fwd_last_dim_kernel
           <<<(fast_dim_size+TILE_W-1)/TILE_W,
              dim3(TILE_W,TILE_H),
              (TILE_W*TILE_H + TILE_W)*sizeof(accscalar_t),
              stream>>>
           (w.data<cuda_scalar_t>(),
            norms.data<accscalar_t>(),
            v.data<cuda_scalar_t>(),
            g.data<cuda_scalar_t>(),
            fast_dim_size,
            slower_dims_size);
       });
  }
  // else 
  // {
  //   intermediate dim kernel.  Error checking on the dim was already done in 
  //   Module.cpp:weight_norm_fwd.  Could put that logic here instead, if we include
  //   <python.h> in both files.
  // }

  // The kernel execution is asynchronous, so this will only catch errors on the kernel launch,
  // not the kernel's execution.  Errors in kernel execution aren't guaranteed to be caught
  // until a later error check on a synchronizing CUDA call.  Unfortunately, without manually 
  // synchronizing here, this is the best we can do.
  THCudaCheck(cudaGetLastError());

#ifdef DEBUG_PROFILE
  THCudaCheck(cudaDeviceSynchronize());
#endif
}

