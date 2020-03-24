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
        volatile int* noop_gmem,
        T* __restrict__ p_in,
        T* __restrict__ p_out,
        GRAD_T* __restrict__ p_copy, // For mixed precision training, pass NULL if not needed
        T* __restrict__ m_in,
        T* __restrict__ m_out,
        T* __restrict__ v_in,
        T* __restrict__ v_out,
        const GRAD_T * __restrict__ g_in,
        const float b1,
        const float b2,
        const float eps,
        const float *p_grad_scale,
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
        float grad_scale = *p_grad_scale;

        for (int j = i; j < tsize; j+=totThreads) {
		T mi = m_in[j];
		T vi = v_in[j];
		T pi = p_in[j];
		GRAD_T gi = g_in[j];
                // don't put __syncthreads inside this loop, it will cause deadlock
		T scaled_grad = gi/grad_scale;
		if (isfinite(scaled_grad)) {
                	mi = b1*mi + (1-b1)*scaled_grad;
	                vi = b2*vi + (1-b2)*scaled_grad*scaled_grad;
        	        float denom;
                	if (mode == ADAM_MODE_0)
	                    denom = sqrtf(vi + eps);
        	        else // Mode 1
                	    denom = sqrtf(vi) + eps;
	                float update = (mi/denom) + (decay*pi);
        	        pi = pi - (step_size*update);
        	} else {
			*noop_gmem = 1;
		}
		m_out[j] = mi;
		v_out[j] = vi;
		p_out[j] = pi;
		if (p_copy != NULL) p_copy[j] = (GRAD_T) pi;
	}

	if (p_copy != NULL) {
		__syncthreads();
		if (i == 0 && *noop_gmem) {
			p_copy[0] = INFINITY;
		}
	}
}

template <typename T, typename GRAD_T>
__global__ void adam_undo_cuda_kernel(
        T* __restrict__ p_in,
        T* __restrict__ p_out,
        T* __restrict__ m_in,
        T* __restrict__ m_out,
        T* __restrict__ v_in,
        T* __restrict__ v_out,
        const GRAD_T * __restrict__ g_in,
        const float b1,
        const float b2,
        const float eps,
        const float *p_grad_scale,
        const float step_size,
        const size_t tsize,
        adamMode_t mode,
        const float decay,
        const float *found_inf)
{
        // If no overflow, then do nothing
        if (*found_inf == 0.f) { return; }

        //Assuming 2D grids and 2D blocks
        const int blockId = gridDim.x * blockIdx.y + blockIdx.x;
        const int threadsPerBlock = blockDim.x * blockDim.y;
        const int threadIdInBlock = threadIdx.y * blockDim.x + threadIdx.x;
        const int i = (blockId * threadsPerBlock + threadIdInBlock);
        const int totThreads = gridDim.x*gridDim.y*threadsPerBlock;
        float grad_scale = *p_grad_scale;

        for (int j = i; j < tsize; j+=totThreads) {
                T scaled_grad = g_in[j]/grad_scale;
		if (isfinite(scaled_grad)) {
	                float denom;
        	        if (mode == ADAM_MODE_0)
                	    denom = sqrtf(v_out[j] + eps);
	                else // Mode 1
        	            denom = sqrtf(v_out[j]) + eps;
			p_in[j] = (p_out[j] + step_size*(m_out[j]/denom)) / (1.0f - step_size*decay);
                	m_in[j] = (m_out[j] - (1-b1)*scaled_grad) / b1;
			v_in[j] = (v_out[j] - (1-b2)*scaled_grad*scaled_grad) / b2;
			// Make sure round off errors don't create (small) negative value.
			// This can happen if we have to revert the very first step.
			v_in[j] = v_in[j] >= 0.0f ? v_in[j] : 0.0f;
		} else {
			p_in[j] = p_out[j];
			m_in[j] = m_out[j];
			v_in[j] = v_out[j];
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

        T* p_in = (T *)tl.addresses[0][tensor_loc];
        p_in += chunk_idx*chunk_size;
        T* p_out = (T *)tl.addresses[1][tensor_loc];
        p_out += chunk_idx*chunk_size;
        T* m_in = (T *)tl.addresses[2][tensor_loc];
        m_in += chunk_idx*chunk_size;
        T* m_out = (T *)tl.addresses[3][tensor_loc];
        m_out += chunk_idx*chunk_size;
        T* v_in = (T *)tl.addresses[4][tensor_loc];
        v_in += chunk_idx*chunk_size;
        T* v_out = (T *)tl.addresses[5][tensor_loc];
        v_out += chunk_idx*chunk_size;
        GRAD_T* g_in = (GRAD_T *)tl.addresses[6][tensor_loc];
        g_in += chunk_idx*chunk_size;
        GRAD_T* p_copy = NULL;
        if (DEPTH == 8) {
            p_copy = (GRAD_T *)tl.addresses[7][tensor_loc];
            p_copy += chunk_idx*chunk_size;
        }

        n -= chunk_idx*chunk_size;

        T incoming_p[ILP];
        T incoming_m[ILP];
        T incoming_v[ILP];
        T incoming_g[ILP];

        for(int i_start = 0;
            i_start < n && i_start < chunk_size;
            i_start += blockDim.x*ILP) {

            #pragma unroll
            for(int ii = 0; ii < ILP; ii++) {
                incoming_p[ii] = 0;
                incoming_m[ii] = 0;
                incoming_v[ii] = 0;
                incoming_g[ii] = 0;

                int i = i_start + threadIdx.x + ii*blockDim.x;
                if (i < n && i < chunk_size) {
                    incoming_p[ii] = p_in[i];
                    incoming_m[ii] = m_in[i];
                    incoming_v[ii] = v_in[i];
                    incoming_g[ii] = static_cast<T>(g_in[i]);
                }
            }

            // note for clarification to future michael:
            // From a pure memory dependency perspective, there's likely no point unrolling
            // the write loop, since writes just fire off once their LDGs arrive.
            // Put another way, the STGs are dependent on the LDGs, but not on each other.
            // There is still compute ILP benefit from unrolling the loop though.
            #pragma unroll
            for(int ii = 0; ii < ILP; ii++) {
                int j = i_start + threadIdx.x + ii*blockDim.x;

                if(j < n && j < chunk_size) {
                    T scaled_grad = incoming_g[ii]/grad_scale;
                    m_out[j] = b1*incoming_m[ii] + (1-b1)*scaled_grad;
                    v_out[j] = b2*incoming_v[ii] + (1-b2)*scaled_grad*scaled_grad;
                    float denom;
                    if (mode == ADAM_MODE_0)
                        denom = sqrtf(v_out[j] + eps);
                    else // Mode 1
                        denom = sqrtf(v_out[j]) + eps;
                    float update = (m_out[j]/denom) + (decay*incoming_p[ii]);
                    p_out[j] = incoming_p[ii] - (step_size*update);
                    if (DEPTH == 8)  p_copy[j] = (GRAD_T) p_out[j];
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
	at::Tensor & noop,
        at::Tensor & p_in,
        at::Tensor & p_out,
        at::Tensor & p_copy,
        at::Tensor & m_in,
        at::Tensor & m_out,
        at::Tensor & v_in,
        at::Tensor & v_out,
        at::Tensor & g_in,
        float lr,
        float beta1,
        float beta2,
        float eps,
        at::Tensor & grad_scale,
        int step,
        int mode,
        int bias_correction,
        float decay)
{
//        using namespace at;

        //Get tensor size
        int tsize = p_in.numel();
        //Determine #threads and #blocks
        const int threadsPerBlock = 512;
        const dim3 blocks((tsize+threadsPerBlock-1)/threadsPerBlock);
        AT_ASSERTM(at::cuda::detail::canUse32BitIndexMath(p_in), "parameter tensor is too large to be indexed with int32");
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

        if (g_in.scalar_type() == at::ScalarType::Half) {
//all other values should be fp32 for half gradients
            AT_ASSERTM(p_in.scalar_type() == at::ScalarType::Float, "expected parameter to be of float type");
//dispatch is done on the gradient type
            using namespace at; // prevents "toString is undefined" errors
            DISPATCH_FLOAT_AND_HALF(g_in.scalar_type(), 0, "adam_cuda_kernel",
                using accscalar_t = at::acc_type<scalar_t_0, true>;
                adam_cuda_kernel<accscalar_t, scalar_t_0><<<blocks,threadsPerBlock, 0, stream>>>(
			noop.DATA_PTR<int>(),
                        p_in.DATA_PTR<accscalar_t>(),
                        p_out.DATA_PTR<accscalar_t>(),
                        p_copy.numel() ? p_copy.DATA_PTR<scalar_t_0>() : NULL,
                        m_in.DATA_PTR<accscalar_t>(),
                        m_out.DATA_PTR<accscalar_t>(),
                        v_in.DATA_PTR<accscalar_t>(),
                        v_out.DATA_PTR<accscalar_t>(),
                        g_in.DATA_PTR<scalar_t_0>(),
                        beta1,
                        beta2,
                        eps,
                        grad_scale.DATA_PTR<float>(),
                        step_size,
                        tsize,
                        (adamMode_t) mode,
                        decay);
                );
      } else {
            using namespace at;
            DISPATCH_DOUBLE_AND_FLOAT(g_in.scalar_type(), 0, "adam_cuda_kernel",
                adam_cuda_kernel<scalar_t_0, scalar_t_0><<<blocks,threadsPerBlock, 0, stream>>>(
			noop.DATA_PTR<int>(),
                        p_in.DATA_PTR<scalar_t_0>(),
                        p_out.DATA_PTR<scalar_t_0>(),
                        NULL, //don't output p_copy for fp32, it's wasted write
                        m_in.DATA_PTR<scalar_t_0>(),
                        m_out.DATA_PTR<scalar_t_0>(),
                        v_in.DATA_PTR<scalar_t_0>(),
                        v_out.DATA_PTR<scalar_t_0>(),
                        g_in.DATA_PTR<scalar_t_0>(),
                        beta1,
                        beta2,
                        eps,
                        grad_scale.DATA_PTR<float>(),
                        step_size,
                        tsize,
                        (adamMode_t) mode,
                        decay);
            );
      }
      THCudaCheck(cudaGetLastError());
}

void fused_adam_undo_cuda(
        at::Tensor & p_in,
        at::Tensor & p_out,
        at::Tensor & m_in,
        at::Tensor & m_out,
        at::Tensor & v_in,
        at::Tensor & v_out,
        at::Tensor & g_in,
        float lr,
        float beta1,
        float beta2,
        float eps,
        at::Tensor & grad_scale,
        int step,
        int mode,
        int bias_correction,
        float decay,
        at::Tensor & found_inf)
{
//        using namespace at;

        //Get tensor size
        int tsize = p_in.numel();
        //Determine #threads and #blocks
        const int threadsPerBlock = 512;
        const dim3 blocks((tsize+threadsPerBlock-1)/threadsPerBlock);
        AT_ASSERTM(at::cuda::detail::canUse32BitIndexMath(p_in), "parameter tensor is too large to be indexed with int32");
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

        if (g_in.scalar_type() == at::ScalarType::Half) {
//all other values should be fp32 for half gradients
            AT_ASSERTM(p_in.scalar_type() == at::ScalarType::Float, "expected parameter to be of float type");
//dispatch is done on the gradient type
            using namespace at; // prevents "toString is undefined" errors
            DISPATCH_FLOAT_AND_HALF(g_in.scalar_type(), 0, "adam_cuda_kernel",
                using accscalar_t = at::acc_type<scalar_t_0, true>;
                adam_undo_cuda_kernel<accscalar_t, scalar_t_0><<<blocks,threadsPerBlock, 0, stream>>>(
                        p_in.DATA_PTR<accscalar_t>(),
                        p_out.DATA_PTR<accscalar_t>(),
                        m_in.DATA_PTR<accscalar_t>(),
                        m_out.DATA_PTR<accscalar_t>(),
                        v_in.DATA_PTR<accscalar_t>(),
                        v_out.DATA_PTR<accscalar_t>(),
                        g_in.DATA_PTR<scalar_t_0>(),
                        beta1,
                        beta2,
                        eps,
                        grad_scale.DATA_PTR<float>(),
                        step_size,
                        tsize,
                        (adamMode_t) mode,
                        decay,
                        found_inf.DATA_PTR<float>());
                );
      } else {
            using namespace at;
            DISPATCH_DOUBLE_AND_FLOAT(g_in.scalar_type(), 0, "adam_cuda_kernel",
                adam_undo_cuda_kernel<scalar_t_0, scalar_t_0><<<blocks,threadsPerBlock, 0, stream>>>(
                        p_in.DATA_PTR<scalar_t_0>(),
                        p_out.DATA_PTR<scalar_t_0>(),
                        m_in.DATA_PTR<scalar_t_0>(),
                        m_out.DATA_PTR<scalar_t_0>(),
                        v_in.DATA_PTR<scalar_t_0>(),
                        v_out.DATA_PTR<scalar_t_0>(),
                        g_in.DATA_PTR<scalar_t_0>(),
                        beta1,
                        beta2,
                        eps,
                        grad_scale.DATA_PTR<float>(),
                        step_size,
                        tsize,
                        (adamMode_t) mode,
                        decay,
                        found_inf.DATA_PTR<float>());
            );
      }
      THCudaCheck(cudaGetLastError());
}

void fused_adam_cuda_mt(
    int chunk_size,
    at::Tensor noop_flag,
    std::vector<std::vector<at::Tensor>> tensor_lists, // p_in, p_out, m_in, m_out, v_in, v_out, g_in, p_copy
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
    AT_ASSERTM(tl_sz == 7 || tl_sz == 8, "expected tensor lists of size 7 or 8");

    if (tensor_lists[6][0].scalar_type() == at::ScalarType::Half) {
//alher values should be fp32 for half gradients
        AT_ASSERTM(tensor_lists[0][0].scalar_type() == at::ScalarType::Float, "expected parameter to be of float type");
//dich is done on the gradient type
        if (tl_sz == 8) {
            DISPATCH_FLOAT_AND_HALF(tensor_lists[6][0].scalar_type(), 0, "adam_cuda_mt_kernel",
                using accscalar_t = at::acc_type<scalar_t_0, true>;
                multi_tensor_apply<8>(
                    BLOCK_SIZE,
                    chunk_size,
                    noop_flag,
                    tensor_lists,
                    AdamFunctor<8, accscalar_t, scalar_t_0>(),
                    beta1,
                    beta2,
                    eps,
                    grad_scale,
                    step_size,
                    (adamMode_t) mode,
                    decay);
            );
        } else {
            DISPATCH_FLOAT_AND_HALF(tensor_lists[6][0].scalar_type(), 0, "adam_cuda_mt_kernel",
                using accscalar_t = at::acc_type<scalar_t_0, true>;
                multi_tensor_apply<7>(
                    BLOCK_SIZE,
                    chunk_size,
                    noop_flag,
                    tensor_lists,
                    AdamFunctor<7, accscalar_t, scalar_t_0>(),
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
        if (tl_sz == 8) {
            DISPATCH_DOUBLE_AND_FLOAT(tensor_lists[6][0].scalar_type(), 0, "adam_cuda_mt_kernel",
                multi_tensor_apply<8>(
                    BLOCK_SIZE,
                    chunk_size,
                    noop_flag,
                    tensor_lists,
                    AdamFunctor<8, scalar_t_0, scalar_t_0>(),
                    beta1,
                    beta2,
                    eps,
                    grad_scale,
                    step_size,
                    (adamMode_t) mode,
                    decay);
            );
        } else {
            DISPATCH_DOUBLE_AND_FLOAT(tensor_lists[6][0].scalar_type(), 0, "adam_cuda_mt_kernel",
                multi_tensor_apply<7>(
                    BLOCK_SIZE,
                    chunk_size,
                    noop_flag,
                    tensor_lists,
                    AdamFunctor<7, scalar_t_0, scalar_t_0>(),
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

__global__
void conditional_copy_kernel(at::Half *dst, at::Half *src, int n, float *found_inf) {
    if (*found_inf == 0.f) { return; }
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    half2 *h2_src = reinterpret_cast<half2*>(src);
    half2 *h2_dst = reinterpret_cast<half2*>(dst);

    for (int i = tid * 2; i < n; i += stride * 2) {
        if (i < (n - 1)) {
            h2_dst[i/2] = h2_src[i/2];
        } else {
            dst[i] = src[i];
        }
    }
}

void conditional_copy_cuda(at::Tensor& dst, at::Tensor& src, at::Tensor& found_inf) {
    const int block_size = 512;
    const int max_blocks = 1024;
    int n = dst.numel();
    int num_blocks = min((n + block_size - 1) / block_size, max_blocks);
    auto stream = at::cuda::getCurrentCUDAStream();
    conditional_copy_kernel<<<num_blocks, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
        dst.DATA_PTR<at::Half>(),
        src.DATA_PTR<at::Half>(),
        n,
        found_inf.DATA_PTR<float>());
    THCudaCheck(cudaGetLastError());
}
