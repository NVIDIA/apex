#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include "ATen/TensorUtils.h"
#include "ATen/Type.h"
#include "ATen/AccumulateType.h"
#include <THC/THCGeneral.h>

typedef enum{
    ADAM_MODE_0   =0, // eps under square root
    ADAM_MODE_1   =1  // eps outside square root
} adamMode_t;

template <typename T, typename GRAD_T>
__global__ void adam_cuda_kernel(
        T* __restrict__ p,
        GRAD_T* __restrict__ p_copy, // For mixed precision training, pass NULL if not needed
        T* __restrict__ m,
        T* __restrict__ v,
        const GRAD_T * __restrict__ g,
        const float b1,
        const float b2,
        const float eps,
        const float grad_scale,
        const float step_size,
        const size_t tsize,
        adamMode_t mode) {

        //Assuming 2D grids and 2D blocks
        const int blockId = gridDim.x * blockIdx.y + blockIdx.x;
        const int threadsPerBlock = blockDim.x * blockDim.y;
        const int threadIdInBlock = threadIdx.y * blockDim.x + threadIdx.x;
        const int i = (blockId * threadsPerBlock + threadIdInBlock);
        const int totThreads = gridDim.x*gridDim.y*threadsPerBlock;

        for (int j = i; j < tsize; j+=totThreads) {
                T scaled_grad = g[j]/grad_scale;
                m[j] = b1*m[j] + (1-b1)*scaled_grad;
                v[j] = b2*v[j] + (1-b2)*scaled_grad*scaled_grad;
                float denom;
                if (mode == ADAM_MODE_0)
                    denom = sqrtf(v[j] + eps);
                else // Mode 1
                    denom = sqrtf(v[j]) + eps;
                p[j] = p[j] - (step_size*m[j]/denom);
                if (p_copy != NULL) p_copy[j] = (GRAD_T) p[j];
        }
}

void fused_adam_cuda(
        at::Tensor & p,
        at::Tensor & p_copy,
        at::Tensor & m,
        at::Tensor & v,
        at::Tensor & g,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float grad_scale,
        int step,
        int mode) {

        //Get tensor size
        int tsize = 1;
        for (int i = 0; i < p.ndimension(); i++) {
                tsize *= p.size(i);
        }

        //Determine #threads and #blocks
        const int threadsPerBlock = 512;
        //elemPerThread = 1 actually works better.
        const int elemPerThread = std::min(16, (tsize+40959)/40960); //40960 = 80 SMs * 512 threads each
        const int elemPerBlock = threadsPerBlock*elemPerThread;
        const dim3 blocks((tsize+elemPerBlock-1)/elemPerBlock,1);

        //Constants
        const float bias_correction1 = 1 - std::pow(beta1, step);
        const float bias_correction2 = 1 - std::pow(beta2, step);
        const float step_size = lr * std::sqrt(bias_correction2)/bias_correction1;

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        if (g.type().scalarType() == at::ScalarType::Half) {
//all other values should be fp32 for half gradients
            AT_ASSERTM(p.type().scalarType() == at::ScalarType::Float, "message");
//dispatch is done on the gradient type 
            AT_DISPATCH_ALL_TYPES(g.type(), "adam_cuda_kernel", ([&] {
                using accscalar_t = at::acc_type<scalar_t, true>;
                adam_cuda_kernel<accscalar_t, scalar_t><<<blocks,threadsPerBlock, 0, stream>>>(
                        p.data<accscalar_t>(),
                        p_copy.numel() ? p_copy.data<scalar_t>() : NULL,
                        m.data<accscalar_t>(),
                        v.data<accscalar_t>(),
                        g.data<scalar_t>(),
                        beta1,
                        beta2,
                        eps,
                        grad_scale,
                        step_size,
                        tsize,
                        (adamMode_t) mode);
            }));
      } else {
            AT_DISPATCH_ALL_TYPES(g.type(), "adam_cuda_kernel", ([&] {
                adam_cuda_kernel<scalar_t, scalar_t><<<blocks,threadsPerBlock, 0, stream>>>(
                        p.data<scalar_t>(),
                        NULL, //don't output p_copy for fp32, it's wasted write
                        m.data<scalar_t>(),
                        v.data<scalar_t>(),
                        g.data<scalar_t>(),
                        beta1,
                        beta2,
                        eps,
                        grad_scale,
                        step_size,
                        tsize,
                        (adamMode_t) mode);
            }));
      }
      THCudaCheck(cudaGetLastError());

}
