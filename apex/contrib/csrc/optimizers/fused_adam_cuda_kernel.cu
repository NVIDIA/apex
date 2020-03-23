#include "ATen/ATen.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/detail/IndexUtils.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include "ATen/TensorUtils.h"
// #include "ATen/Type.h"
#include "ATen/AccumulateType.h"
#include <THC/THCGeneral.h>
#include "multi_tensor_apply.cuh"

#define BLOCK_SIZE 512
#define ILP 4

#include "type_shim.h"

typedef enum{
    ADAM_MODE_0   =0, // eps under square root
    ADAM_MODE_1   =1  // eps outside square root
} adamMode_t;

template <typename GRAD_T>
__global__ void strided_check_finite_cuda_kernel(
        volatile int* noop_gmem,
        GRAD_T* __restrict__ p_copy,
        const size_t tsize,
        int stride,
        int clear_overflow_first)
{
    //Assuming 2D grids and 2D blocks
    const int blockId = gridDim.x * blockIdx.y + blockIdx.x;
    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int threadIdInBlock = threadIdx.y * blockDim.x + threadIdx.x;
    const int i = (blockId * threadsPerBlock + threadIdInBlock) * stride;
    const int totThreads = gridDim.x*gridDim.y*threadsPerBlock*stride;

    if (clear_overflow_first) {
        if (i == 0) {
            *noop_gmem = 0;
        }
        __syncthreads();
    }

    for (int j = i; j < tsize; j+=totThreads) {
        GRAD_T pi = p_copy[j];
        if (!isfinite(pi)) {
            *noop_gmem = 1;
        }
    }
}

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

    T mi[ILP];
    T vi[ILP];
    T pi[ILP];
    T gi[ILP];

    bool overflow = false;
    for(int j_start = 0;  j_start < tsize;  j_start+=totThreads*ILP) {
#pragma unroll
        for(int ii = 0; ii < ILP; ii++) {
            mi[ii] = T(0);
            vi[ii] = T(0);
            pi[ii] = T(0);
            gi[ii] = GRAD_T(0);

            int j = j_start + i + totThreads*ii;
            if (j < tsize) {
                pi[ii] = p[j];
                mi[ii] = m[j];
                vi[ii] = v[j];
                gi[ii] = static_cast<T>(g[j]);
            }
        }

#pragma unroll
        for(int ii = 0; ii < ILP; ii++) {
            T scaled_grad = gi[ii]/grad_scale;
            if (isfinite(scaled_grad)) {
                mi[ii] = b1*mi[ii] + (1-b1)*scaled_grad;
                vi[ii] = b2*vi[ii] + (1-b2)*scaled_grad*scaled_grad;
                float denom;
                if (mode == ADAM_MODE_0)
                    denom = sqrtf(vi[ii] + eps);
                else // Mode 1
                    denom = sqrtf(vi[ii]) + eps;
                float update = (mi[ii]/denom) + (decay*pi[ii]);
                pi[ii] = pi[ii] - (step_size*update);
            } else {
                overflow = true;
            }
        }

#pragma unroll
        for(int ii = 0; ii < ILP; ii++) {
            int j = j_start + i + totThreads*ii;
            if (j < tsize) {
                m[j] = mi[ii];
                v[j] = vi[ii];
                p[j] = pi[ii];
                if (p_copy != NULL) p_copy[j] = static_cast<GRAD_T>(pi[ii]);
            }
        }
    }

    if (p_copy != NULL) {
        __syncthreads();
        if (overflow) {
            p_copy[0] = INFINITY;
        }
    }
}

template <typename T, typename GRAD_T>
__global__ void adam_undo_cuda_kernel(
        T* __restrict__ p,
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

    T mi[ILP];
    T vi[ILP];
    T pi[ILP];
    T gi[ILP];

    for(int j_start = 0;  j_start < tsize;  j_start+=totThreads*ILP) {
#pragma unroll
        for(int ii = 0; ii < ILP; ii++) {
            mi[ii] = T(0);
            vi[ii] = T(0);
            pi[ii] = T(0);
            gi[ii] = GRAD_T(0);

            int j = j_start + i*ILP;
            if (j < tsize) {
                pi[ii] = p[j];
                mi[ii] = m[j];
                vi[ii] = v[j];
                gi[ii] = static_cast<T>(g[j]);
            }
        }

#pragma unroll
        for(int ii = 0; ii < ILP; ii++) {
            T scaled_grad = gi[ii]/grad_scale;
            if (isfinite(scaled_grad)) {
                float denom;
                if (mode == ADAM_MODE_0)
                    denom = sqrtf(vi[ii] + eps);
                else // Mode 1
                    denom = sqrtf(vi[ii]) + eps;
                pi[ii] = (pi[ii] + step_size*(mi[ii]/denom)) / (1.0f - step_size*decay);
                mi[ii] = (mi[ii] - (1-b1)*scaled_grad) / b1;
                vi[ii] = (vi[ii] - (1-b2)*scaled_grad*scaled_grad) / b2;
                // Make sure round off errors don't create (small) negative value.
                // This can happen if we have to revert the very first step.
                vi[ii] = vi[ii] >= 0.0f ? vi[ii] : 0.0f;
            }
        }

#pragma unroll
        for(int ii = 0; ii < ILP; ii++) {
            int j = j_start + i*ILP;
            if (j < tsize) {
                m[j] = mi[ii];
                v[j] = vi[ii];
                p[j] = pi[ii];
            }
        }
    }
}

template <int DEPTH, typename T, typename GRAD_T>
struct AdamFunctor
{
    __device__ __forceinline__ void operator()(
        int chunk_size,
        volatile int* noop_gmem,
        TensorListMetadata<DEPTH>& tl,
        const float b1,
        const float b2,
        const float eps,
        const float grad_scale,
        const float step_size,
        adamMode_t mode,
        const float decay)
    {
        int tensor_loc = tl.block_to_tensor[blockIdx.x];
        int chunk_idx = tl.block_to_chunk[blockIdx.x];
        int n = tl.sizes[tensor_loc];

        T* p = (T *)tl.addresses[0][tensor_loc];
        p += chunk_idx*chunk_size;
        T* m = (T *)tl.addresses[1][tensor_loc];
        m += chunk_idx*chunk_size;
        T* v = (T *)tl.addresses[2][tensor_loc];
        v += chunk_idx*chunk_size;
        GRAD_T* g = (GRAD_T *)tl.addresses[3][tensor_loc];
        g += chunk_idx*chunk_size;
        GRAD_T* p_copy = NULL;
        if (DEPTH == 5) {
            p_copy = (GRAD_T *)tl.addresses[4][tensor_loc];
            p_copy += chunk_idx*chunk_size;
        }

        n -= chunk_idx*chunk_size;
        int dim = chunk_size < n ? chunk_size : n;

        T mi[ILP];
        T vi[ILP];
        T pi[ILP];
        T gi[ILP];

        bool overflow = false;
        for(int j_start = 0;  j_start < dim;  j_start+=blockDim.x*ILP) {
#pragma unroll
            for(int ii = 0; ii < ILP; ii++) {
                mi[ii] = T(0);
                vi[ii] = T(0);
                pi[ii] = T(0);
                gi[ii] = GRAD_T(0);

                int j = j_start + threadIdx.x + ii*blockDim.x;
                if (j < dim) {
                    pi[ii] = p[j];
                    mi[ii] = m[j];
                    vi[ii] = v[j];
                    gi[ii] = static_cast<T>(g[j]);
                }
            }

#pragma unroll
            for(int ii = 0; ii < ILP; ii++) {
                T scaled_grad = gi[ii]/grad_scale;
                if (isfinite(scaled_grad)) {
                    mi[ii] = b1*mi[ii] + (1-b1)*scaled_grad;
                    vi[ii] = b2*vi[ii] + (1-b2)*scaled_grad*scaled_grad;
                    float denom;
                    if (mode == ADAM_MODE_0)
                        denom = sqrtf(vi[ii] + eps);
                    else // Mode 1
                        denom = sqrtf(vi[ii]) + eps;
                    float update = (mi[ii]/denom) + (decay*pi[ii]);
                    pi[ii] = pi[ii] - (step_size*update);
                } else {
                    overflow = true;
                }
            }

#pragma unroll
            for(int ii = 0; ii < ILP; ii++) {
                int j = j_start + threadIdx.x + ii*blockDim.x;
                if (j < dim) {
                    m[j] = mi[ii];
                    v[j] = vi[ii];
                    p[j] = pi[ii];
                    if (p_copy != NULL) p_copy[j] = static_cast<GRAD_T>(pi[ii]);
                }
            }
        }

        if (overflow) {
            *noop_gmem = 1;
        }
    }
};

template <int DEPTH, typename T, typename GRAD_T>
struct AdamUndoFunctor
{
    __device__ __forceinline__ void operator()(
        int chunk_size,
        volatile int* noop_gmem,
        TensorListMetadata<DEPTH>& tl,
        const float b1,
        const float b2,
        const float eps,
        const float grad_scale,
        const float step_size,
        adamMode_t mode,
        const float decay)
    {
        int tensor_loc = tl.block_to_tensor[blockIdx.x];
        int chunk_idx = tl.block_to_chunk[blockIdx.x];
        int n = tl.sizes[tensor_loc];

        T* p = (T *)tl.addresses[0][tensor_loc];
        p += chunk_idx*chunk_size;
        T* m = (T *)tl.addresses[1][tensor_loc];
        m += chunk_idx*chunk_size;
        T* v = (T *)tl.addresses[2][tensor_loc];
        v += chunk_idx*chunk_size;
        GRAD_T* g = (GRAD_T *)tl.addresses[3][tensor_loc];
        g += chunk_idx*chunk_size;

        n -= chunk_idx*chunk_size;
        int dim = chunk_size < n ? chunk_size : n;

        T mi[ILP];
        T vi[ILP];
        T pi[ILP];
        T gi[ILP];

        for(int j_start = 0;  j_start < dim;  j_start+=blockDim.x*ILP) {
#pragma unroll
            for(int ii = 0; ii < ILP; ii++) {
                mi[ii] = T(0);
                vi[ii] = T(0);
                pi[ii] = T(0);
                gi[ii] = GRAD_T(0);

                int j = j_start + threadIdx.x + ii*blockDim.x;
                if (j < dim) {
                    pi[ii] = p[j];
                    mi[ii] = m[j];
                    vi[ii] = v[j];
                    gi[ii] = static_cast<T>(g[j]);
                }
            }

#pragma unroll
            for(int ii = 0; ii < ILP; ii++) {
                T scaled_grad = gi[ii]/grad_scale;
                if (isfinite(scaled_grad)) {
                    float denom;
                    if (mode == ADAM_MODE_0)
                        denom = sqrtf(vi[ii] + eps);
                    else // Mode 1
                        denom = sqrtf(vi[ii]) + eps;
                    pi[ii] = (pi[ii] + step_size*(mi[ii]/denom)) / (1.0f - step_size*decay);
                    mi[ii] = (mi[ii] - (1-b1)*scaled_grad) / b1;
                    vi[ii] = (vi[ii] - (1-b2)*scaled_grad*scaled_grad) / b2;
                    // Make sure round off errors don't create (small) negative value.
                    // This can happen if we have to revert the very first step.
                    vi[ii] = vi[ii] >= 0.0f ? vi[ii] : 0.0f;
                }
            }

#pragma unroll
            for(int ii = 0; ii < ILP; ii++) {
                int j = j_start + threadIdx.x + ii*blockDim.x;
                if (j < dim) {
                    m[j] = mi[ii];
                    v[j] = vi[ii];
                    p[j] = pi[ii];
                }
            }
        }
    }
};

void fused_strided_check_finite(
	at::Tensor & noop,
        at::Tensor & p_copy,
        int stride,
	int clear_overflow_first)
{
	//Get tensor size
	int tsize = p_copy.numel();
	int niter = (tsize + stride - 1) / stride;

	//Determine #threads and #blocks
	const int threadsPerBlock = 512;
	const dim3 blocks((niter+threadsPerBlock-1)/threadsPerBlock);
	AT_ASSERTM(at::cuda::detail::canUse32BitIndexMath(p_copy), "parameter tensor is too large to be indexed with int32");

	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	using namespace at; // prevents "toString is undefined" errors
	DISPATCH_FLOAT_AND_HALF(p_copy.scalar_type(), 0, "check_finite_cuda_kernel",
			strided_check_finite_cuda_kernel<scalar_t_0><<<blocks,threadsPerBlock, 0, stream>>>(
				noop.DATA_PTR<int>(),
				p_copy.DATA_PTR<scalar_t_0>(),
				tsize,
				stride,
				clear_overflow_first);
			);
	THCudaCheck(cudaGetLastError());
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
                        p.DATA_PTR<accscalar_t>(),
                        p_copy.numel() ? p_copy.DATA_PTR<scalar_t_0>() : NULL,
                        m.DATA_PTR<accscalar_t>(),
                        v.DATA_PTR<accscalar_t>(),
                        g.DATA_PTR<scalar_t_0>(),
                        beta1,
                        beta2,
                        eps,
                        grad_scale,
                        step_size,
                        tsize,
                        (adamMode_t) mode,
                        decay);
                );
      } else {
            using namespace at;
            DISPATCH_DOUBLE_AND_FLOAT(g.scalar_type(), 0, "adam_cuda_kernel",
                adam_cuda_kernel<scalar_t_0, scalar_t_0><<<blocks,threadsPerBlock, 0, stream>>>(
                        p.DATA_PTR<scalar_t_0>(),
                        NULL, //don't output p_copy for fp32, it's wasted write
                        m.DATA_PTR<scalar_t_0>(),
                        v.DATA_PTR<scalar_t_0>(),
                        g.DATA_PTR<scalar_t_0>(),
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

void fused_adam_undo_cuda(
        at::Tensor & p,
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
                adam_undo_cuda_kernel<accscalar_t, scalar_t_0><<<blocks,threadsPerBlock, 0, stream>>>(
                        p.DATA_PTR<accscalar_t>(),
                        m.DATA_PTR<accscalar_t>(),
                        v.DATA_PTR<accscalar_t>(),
                        g.DATA_PTR<scalar_t_0>(),
                        beta1,
                        beta2,
                        eps,
                        grad_scale,
                        step_size,
                        tsize,
                        (adamMode_t) mode,
                        decay);
                );
      } else {
            using namespace at;
            DISPATCH_DOUBLE_AND_FLOAT(g.scalar_type(), 0, "adam_cuda_kernel",
                adam_undo_cuda_kernel<scalar_t_0, scalar_t_0><<<blocks,threadsPerBlock, 0, stream>>>(
                        p.DATA_PTR<scalar_t_0>(),
                        m.DATA_PTR<scalar_t_0>(),
                        v.DATA_PTR<scalar_t_0>(),
                        g.DATA_PTR<scalar_t_0>(),
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

void fused_adam_cuda_mt(
    int chunk_size,
    at::Tensor noop_flag,
    std::vector<std::vector<at::Tensor>> tensor_lists, // p, m, v, g, p_copy
    float lr,
    float beta1,
    float beta2,
    float eps,
    float grad_scale,
    int step,
    int mode,
    int bias_correction,
    float decay) {

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

    size_t tl_sz = tensor_lists.size();
    AT_ASSERTM(tl_sz == 4 || tl_sz == 5, "expected tensor lists of size 4 or 5");

    if (tensor_lists[3][0].scalar_type() == at::ScalarType::Half) {
//alher values should be fp32 for half gradients
        AT_ASSERTM(tensor_lists[0][0].scalar_type() == at::ScalarType::Float, "expected parameter to be of float type");
//dich is done on the gradient type
        if (tl_sz == 5) {
            DISPATCH_FLOAT_AND_HALF(tensor_lists[3][0].scalar_type(), 0, "adam_cuda_mt_kernel",
                    using accscalar_t = at::acc_type<scalar_t_0, true>;
                    multi_tensor_apply<5>(
                        BLOCK_SIZE,
                        chunk_size,
                        noop_flag,
                        tensor_lists,
                        AdamFunctor<5, accscalar_t, scalar_t_0>(),
                        beta1,
                        beta2,
                        eps,
                        grad_scale,
                        step_size,
                        (adamMode_t) mode,
                        decay);
                    );
        } else {
            DISPATCH_FLOAT_AND_HALF(tensor_lists[3][0].scalar_type(), 0, "adam_cuda_mt_kernel",
                    using accscalar_t = at::acc_type<scalar_t_0, true>;
                    multi_tensor_apply<4>(
                        BLOCK_SIZE,
                        chunk_size,
                        noop_flag,
                        tensor_lists,
                        AdamFunctor<4, accscalar_t, scalar_t_0>(),
                        beta1,
                        beta2,
                        eps,
                        grad_scale,
                        step_size,
                        (adamMode_t) mode,
                        decay);
                    );
        }
    } else {
        if (tl_sz == 5) {
            DISPATCH_DOUBLE_AND_FLOAT(tensor_lists[3][0].scalar_type(), 0, "adam_cuda_mt_kernel",
                    multi_tensor_apply<5>(
                        BLOCK_SIZE,
                        chunk_size,
                        noop_flag,
                        tensor_lists,
                        AdamFunctor<5, scalar_t_0, scalar_t_0>(),
                        beta1,
                        beta2,
                        eps,
                        grad_scale,
                        step_size,
                        (adamMode_t) mode,
                        decay);
                    );
        } else {
            DISPATCH_DOUBLE_AND_FLOAT(tensor_lists[3][0].scalar_type(), 0, "adam_cuda_mt_kernel",
                    multi_tensor_apply<4>(
                        BLOCK_SIZE,
                        chunk_size,
                        noop_flag,
                        tensor_lists,
                        AdamFunctor<4, scalar_t_0, scalar_t_0>(),
                        beta1,
                        beta2,
                        eps,
                        grad_scale,
                        step_size,
                        (adamMode_t) mode,
                        decay);
                    );
        }
    }
    THCudaCheck(cudaGetLastError());
}

void fused_adam_undo_cuda_mt(
    int chunk_size,
    at::Tensor noop_flag,
    std::vector<std::vector<at::Tensor>> tensor_lists, // p, m, v, g, p_copy
    float lr,
    float beta1,
    float beta2,
    float eps,
    float grad_scale,
    int step,
    int mode,
    int bias_correction,
    float decay) {

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

    size_t tl_sz = tensor_lists.size();
    AT_ASSERTM(tl_sz == 4, "expected tensor list of size 4");

    if (tensor_lists[3][0].scalar_type() == at::ScalarType::Half) {
        //alher values should be fp32 for half gradients
        AT_ASSERTM(tensor_lists[0][0].scalar_type() == at::ScalarType::Float, "expected parameter to be of float type");
        //dich is done on the gradient type
        DISPATCH_FLOAT_AND_HALF(tensor_lists[3][0].scalar_type(), 0, "adam_undo_cuda_mt_kernel",
                using accscalar_t = at::acc_type<scalar_t_0, true>;
                multi_tensor_apply<4>(
                    BLOCK_SIZE,
                    chunk_size,
                    noop_flag,
                    tensor_lists,
                    AdamUndoFunctor<4, accscalar_t, scalar_t_0>(),
                    beta1,
                    beta2,
                    eps,
                    grad_scale,
                    step_size,
                    (adamMode_t) mode,
                    decay);
                );
    } else {
        DISPATCH_DOUBLE_AND_FLOAT(tensor_lists[3][0].scalar_type(), 0, "adam_undo_cuda_mt_kernel",
                multi_tensor_apply<4>(
                    BLOCK_SIZE,
                    chunk_size,
                    noop_flag,
                    tensor_lists,
                    AdamUndoFunctor<4, scalar_t_0, scalar_t_0>(),
                    beta1,
                    beta2,
                    eps,
                    grad_scale,
                    step_size,
                    (adamMode_t) mode,
                    decay);
                );
    }
    THCudaCheck(cudaGetLastError());
}

