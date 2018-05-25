
#include <ATen/ATen.h>
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDATensorMethods.cuh"
#include "ATen/cuda/CUDATypeConversion.cuh"
#include <THC/THCTensorMathReduce.cuh>

#include <assert.h>

#define BLOCK_SIZE 1024
#define MAX_BLOCKS 1024

// It makes sense to lock the type to "float" here because the downscaling
// should only be applied to the FP32 master gradients.  Also, if "in" were 
// a different type, it would require divergent code for the vectorized load logic.
__global__ void scale_reduce_overflow
  (float *in, 
   size_t n, 
   float scale,
   uint8_t *overflow_out) 
{
    __shared__ uint8_t cta_overflow[BLOCK_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    uint8_t my_overflow = 0;
    for (int i = tid * 4; i < n; i+= stride * 4) {
        if (i < (n - 3)) {
            float4 f4 = ((float4*)in)[i / 4];
            if (isfinite(f4.x)) {
                f4.x *= scale;
            } else {
                my_overflow = 1;
            }
            if (isfinite(f4.y)) {
                f4.y *= scale;
            } else {
                my_overflow = 1;
            }
            if (isfinite(f4.z)) {
                f4.z *= scale;
            } else {
                my_overflow = 1;
            }
            if (isfinite(f4.w)) {
                f4.w *= scale;
            } else {
                my_overflow = 1;
            }
            ((float4*)in)[i / 4] = f4;
        } else {
            for (; i < n; ++i) {
                if (isfinite(in[i])) {
                    in[i] *= scale;
                } else {
                    my_overflow = 1;
                }
            }
        }
    }

    int tIdx = threadIdx.x;
    cta_overflow[tIdx] = my_overflow;
    __syncthreads();

    int participating = BLOCK_SIZE / 2;
    while (participating > 0) {
        if (tIdx < participating) {
            cta_overflow[tIdx] = max(cta_overflow[tIdx],
                                     cta_overflow[tIdx + participating]);
        }
        participating /= 2;
        __syncthreads();
    }
    if (tIdx == 0) {
        overflow_out[blockIdx.x] = max(cta_overflow[0],
                                       overflow_out[blockIdx.x]);
    }
}

void scale_check_overflow_cuda
  (const at::Tensor& d_grads, 
   float scale,
   const at::Tensor& d_buf) 
{
  using namespace at;
  cudaStream_t stream = globalContext().getCurrentCUDAStream();
  
  size_t n = d_grads.numel();
  size_t buf_n = d_buf.numel();

  int num_blks = min((int(n) + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     MAX_BLOCKS);
  assert(buf_n >= num_blks);
  scale_reduce_overflow<<<num_blks, BLOCK_SIZE, 0, stream>>>
    (d_grads.data<float>(), 
     n, 
     scale, 
     d_buf.data<uint8_t>());
  THCudaCheck(cudaGetLastError());
}

