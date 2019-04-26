#include "ATen/ATen.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/detail/IndexUtils.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include "ATen/TensorUtils.h"
#include "ATen/Type.h"
#include "ATen/AccumulateType.h"
#include <THC/THCGeneral.h>

#include "type_shim.h"

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
        adamMode_t mode,
        const float decay)
{
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
                float update = (m[j]/denom) + (decay*p[j]);
                p[j] = p[j] - (step_size*update);
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
        int mode,
        int bias_correction,
        float decay)
{
//        using namespace at;

        //Get tensor size
        int tsize = p.numel();
        //Determine #threads and #blocks
        const int threadsPerBlock = 512;
        const dim3 blocks((tsize+threadsPerBlock-1)/threadsPerBlock);
        AT_ASSERTM(at::cuda::detail::canUse32BitIndexMath(p), "parameter tensor is too large to be indexed with int32");
        //Constants
        float step_size = 0;
        if (bias_correction == 1) {
            const float bias_correction1 = 1 - std::pow(beta1, step);
            const float bias_correction2 = 1 - std::pow(beta2, step);
            step_size = lr * std::sqrt(bias_correction2)/bias_correction1;
        }
        else {
            step_size = lr;
        }
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        if (g.scalar_type() == at::ScalarType::Half) {
//all other values should be fp32 for half gradients
            AT_ASSERTM(p.scalar_type() == at::ScalarType::Float, "expected parameter to be of float type");
//dispatch is done on the gradient type
            using namespace at; // prevents "toString is undefined" errors
            DISPATCH_FLOAT_AND_HALF(g.scalar_type(), 0, "adam_cuda_kernel", 
                using accscalar_t = at::acc_type<scalar_t_0, true>;
                adam_cuda_kernel<accscalar_t, scalar_t_0><<<blocks,threadsPerBlock, 0, stream>>>(
                        p.data<accscalar_t>(),
                        p_copy.numel() ? p_copy.data<scalar_t_0>() : NULL,
                        m.data<accscalar_t>(),
                        v.data<accscalar_t>(),
                        g.data<scalar_t_0>(),
                        beta1,
                        beta2,
                        eps,
                        grad_scale,
                        step_size,
                        tsize,
                        (adamMode_t) mode,
                        decay);
                )
      } else {
            using namespace at;
            DISPATCH_DOUBLE_AND_FLOAT(g.scalar_type(), 0, "adam_cuda_kernel",
                adam_cuda_kernel<scalar_t_0, scalar_t_0><<<blocks,threadsPerBlock, 0, stream>>>(
                        p.data<scalar_t_0>(),
                        NULL, //don't output p_copy for fp32, it's wasted write
                        m.data<scalar_t_0>(),
                        v.data<scalar_t_0>(),
                        g.data<scalar_t_0>(),
                        beta1,
                        beta2,
                        eps,
                        grad_scale,
                        step_size,
                        tsize,
                        (adamMode_t) mode,
                        decay);
            );
      }
      THCudaCheck(cudaGetLastError());

}
