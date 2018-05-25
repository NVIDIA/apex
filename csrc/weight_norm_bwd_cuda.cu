#include "kernel_utils.cuh"

#include <ATen/ATen.h>
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDATensorMethods.cuh"
#include "ATen/cuda/CUDATypeConversion.cuh"
#include <THC/THCTensorMathReduce.cuh>

template
  <typename scalar_t, 
   typename accscalar_t>
__global__ void weight_norm_bwd_first_dim_kernel
  (scalar_t* __restrict__ pLpv,
   scalar_t* __restrict__ pLpg,
   const scalar_t* __restrict__ pLpw,
   const scalar_t* __restrict__ savedv,
   const scalar_t* __restrict__ savedg,
   const accscalar_t* __restrict__ savedNorms,
   const int rowSize)
{
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
    accscalar_t pLpwi = scalar_cast<accscalar_t>(pLpw[i+rowStart]); 
    accscalar_t savedvi = scalar_cast<accscalar_t>(savedv[i+rowStart]); 
    thread_sum += pLpwi*savedvi; // AccumOp, could do Kahan here
  }

  reduce_block_into_lanes(s, thread_sum, 1, ReduceAdd<accscalar_t>());
  accscalar_t result = s[0];

  // Could choose to save reciprocal of norm instead I suppose, but norms is probably
  // more handy to keep around.
  // Broadcast load; could use shared memory instead.
  accscalar_t rnorm = 1.f/savedNorms[row];  
  accscalar_t rnorm3 = rnorm*rnorm*rnorm;

  // Write g gradients.
  if(tid == 0)
    pLpg[row] = scalar_cast<scalar_t>(result*rnorm);

  // Broadcast load, could use shared memory instead.
  accscalar_t g_this_row = scalar_cast<accscalar_t>(savedg[row]);
   
  // Write v gradients.  We are reusing values that were loaded earlier, so there 
  // is an optimization opportunity here (store values persistently).
  for(int j = tid; j < rowSize; j += stride ) 
  {
    accscalar_t pLpwj = scalar_cast<accscalar_t>(pLpw[j+rowStart]);  
    accscalar_t savedvj = scalar_cast<accscalar_t>(savedv[j+rowStart]);  
    accscalar_t pLpvj = g_this_row*(rnorm*pLpwj - rnorm3*savedvj*result);
    pLpv[j+rowStart] = scalar_cast<scalar_t>(pLpvj);
  }
}

template 
  <typename scalar_t, 
   typename accscalar_t>
__global__ void weight_norm_bwd_last_dim_kernel
  (scalar_t* __restrict__ pLpv,
   scalar_t* __restrict__ pLpg,
   const scalar_t* __restrict__ pLpw,
   const scalar_t* __restrict__ savedv,
   const scalar_t* __restrict__ savedg,
   const accscalar_t* __restrict__ savedNorms,
   const int fast_dim_size,
   const int slower_dims_size)
{
  const int fast_dim_location = threadIdx.x + blockIdx.x*blockDim.x;

  extern __shared__ char buf[];
  accscalar_t* s = (accscalar_t*)buf;

  accscalar_t thread_sum = 0.f;

  int slower_dims_location = threadIdx.y;
  int currentIdx = fast_dim_location + fast_dim_size*slower_dims_location;
  if(fast_dim_location < fast_dim_size)
    while(slower_dims_location < slower_dims_size)
    {
      accscalar_t pLpwi = scalar_cast<accscalar_t>(pLpw[currentIdx]); 
      accscalar_t savedvi = scalar_cast<accscalar_t>(savedv[currentIdx]); 
      thread_sum += pLpwi*savedvi; // AccumOp, could do Kahan here
      currentIdx += blockDim.y*fast_dim_size;
      slower_dims_location += blockDim.y; 
    }

  reduce_block_into_lanes(s, thread_sum, blockDim.x, ReduceAdd<accscalar_t>()); 
  accscalar_t result = s[threadIdx.x];

  // Broadcast load; could use shared memory instead.
  accscalar_t rnorm = 1.f/savedNorms[fast_dim_location];  
  accscalar_t rnorm3 = rnorm*rnorm*rnorm;

  // Write g gradients.
  if(threadIdx.y == 0)
    pLpg[fast_dim_location] = scalar_cast<scalar_t>(result*rnorm);

  // Entire block pulls these values, could use shared memory instead.
  accscalar_t g_this_col = scalar_cast<accscalar_t>(savedg[fast_dim_location]);

  // Write v gradients.
  slower_dims_location = threadIdx.y;
  currentIdx = fast_dim_location + fast_dim_size*slower_dims_location;
  if(fast_dim_location < fast_dim_size)
    while(slower_dims_location < slower_dims_size)
    {
      accscalar_t pLpwj = scalar_cast<accscalar_t>(pLpw[currentIdx]);  
      accscalar_t savedvj = scalar_cast<accscalar_t>(savedv[currentIdx]);  
      accscalar_t pLpvj = g_this_col*(rnorm*pLpwj - rnorm3*savedvj*result);
      pLpv[currentIdx] = scalar_cast<scalar_t>(pLpvj);
      currentIdx += blockDim.y*fast_dim_size;
      slower_dims_location += blockDim.y; 
    } 
}

void weight_norm_bwd_cuda
  (const at::Tensor& pLpv,
   const at::Tensor& pLpg,
   const at::Tensor& pLpw,
   const at::Tensor& savedv,
   const at::Tensor& savedg,
   const at::Tensor& savedNorms,
   int dim)
{
#ifdef DEBUG_ANY
  using namespace std;
  cout << "Hello from send_to_bwd with pLpw.type = " << pLpw.type << endl;
#endif

  const int ndims = savedv.ndimension();

  if(dim == 0) 
  {
    // Find logical size of each flattened slowest-dim row
    int rowSize = 1;
    for(int i = ndims - 1; i > 0; i--)
      rowSize *= savedv.size(i);

    using namespace at;
    cudaStream_t stream = globalContext().getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF
      (savedv.type(), 
       "weight_norm_bwd_first_dim_kernel",  
       [&]
       {
         using cuda_scalar_t = cuda::type<scalar_t>;
         using accscalar_t = acc_type<cuda_scalar_t, true>;

	 weight_norm_bwd_first_dim_kernel
	   <<<pLpw.size(0), 
	      BLOCK, 
	      BLOCK*sizeof(accscalar_t),
              stream>>>
	   (pLpv.data<cuda_scalar_t>(),
	    pLpg.data<cuda_scalar_t>(),
	    pLpw.data<cuda_scalar_t>(),
	    savedv.data<cuda_scalar_t>(),
	    savedg.data<cuda_scalar_t>(),
	    savedNorms.data<accscalar_t>(),
	    rowSize);
       });
  }
  else if(dim == ndims - 1)
  {
    // Precompute slower_dims_size and fast_dim_size because they involve dynamically indexing an array.
    int slower_dims_size = 1;
    for(int i = 0; i < ndims - 1; i++)
      slower_dims_size *= savedv.size(i);

    int fast_dim_size = savedv.size(ndims-1);

    using namespace at;
    cudaStream_t stream = globalContext().getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF
      (savedv.type(), 
       "weight_norm_bwd_last_dim_kernel",  
       [&]
       {
         using cuda_scalar_t = cuda::type<scalar_t>;
         using accscalar_t = acc_type<cuda_scalar_t, true>;

         weight_norm_bwd_last_dim_kernel
           <<<(fast_dim_size+TILE_W-1)/TILE_W,
              dim3(TILE_W,TILE_H), 
              (TILE_W*TILE_H + TILE_W)*sizeof(accscalar_t),
              stream>>>
           (pLpv.data<cuda_scalar_t>(),
            pLpg.data<cuda_scalar_t>(),
            pLpw.data<cuda_scalar_t>(),
            savedv.data<cuda_scalar_t>(),
            savedg.data<cuda_scalar_t>(),
            savedNorms.data<accscalar_t>(),
            fast_dim_size,
            slower_dims_size);
       });
  }
  // else 
  // {
  //   intermediate dim kernel.  Error checking on the dim was already done in 
  //   Module.cpp:weight_norm_bwd.  Could put that logic here instead, if we include
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
