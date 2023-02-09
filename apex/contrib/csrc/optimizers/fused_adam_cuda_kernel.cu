#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>

#include "ATen/ATen.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/detail/IndexUtils.cuh"
#include "ATen/TensorUtils.h"
// #include "ATen/Type.h"
#include "ATen/AccumulateType.h"

#include "multi_tensor_apply.cuh"

#define BLOCK_SIZE 512
#define ILP 4

template<typename T>
__device__ __forceinline__ bool is_aligned(T* p){
  return ((uint64_t)p) % (ILP*sizeof(T)) == 0;
}

template<typename T>
__device__ __forceinline__ void load_store(T* dst, T* src, int dst_offset, int src_offset){
  typedef typename std::aligned_storage<ILP*sizeof(T), ILP*alignof(T)>::type LT;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

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

        T incoming_p[ILP];
        T incoming_m[ILP];
        T incoming_v[ILP];
        T incoming_g[ILP];

        // to make things simple, we put aligned case in a different code path
        if(n % ILP == 0 &&
           chunk_size % ILP == 0 &&
           is_aligned(p) &&
           is_aligned(m) &&
           is_aligned(v) &&
           is_aligned(g) &&
           is_aligned(p_copy))
        {
          for(int i_start = threadIdx.x; i_start*ILP < n && i_start*ILP < chunk_size; i_start += blockDim.x)
          {
            // load
            GRAD_T tmp_g[ILP];
            load_store(incoming_p, p, 0, i_start);
            load_store(incoming_m, m, 0, i_start);
            load_store(incoming_v, v, 0, i_start);
            load_store(tmp_g, g, 0, i_start);
#pragma unroll
            for(int ii = 0; ii < ILP; ii++) {
              incoming_g[ii] = static_cast<T>(tmp_g[ii]);
              T scaled_grad = incoming_g[ii]/grad_scale;
              incoming_m[ii] = b1*incoming_m[ii] + (1-b1)*scaled_grad;
              incoming_v[ii] = b2*incoming_v[ii] + (1-b2)*scaled_grad*scaled_grad;
              float denom;
              if (mode == ADAM_MODE_0)
                denom = sqrtf(incoming_v[ii] + eps);
              else // Mode 1
                denom = sqrtf(incoming_v[ii]) + eps;
              float update = (incoming_m[ii]/denom) + (decay*incoming_p[ii]);
              incoming_p[ii] = incoming_p[ii] - (step_size*update);
              if (DEPTH == 5)  tmp_g[ii] = static_cast<GRAD_T>(incoming_p[ii]);
            }
            load_store(p, incoming_p, i_start, 0);
            load_store(m, incoming_m, i_start, 0);
            load_store(v, incoming_v, i_start, 0);
            if (DEPTH == 5) load_store(p_copy, tmp_g, i_start, 0);
          }
        }
        else
        {
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
                incoming_p[ii] = p[i];
                incoming_m[ii] = m[i];
                incoming_v[ii] = v[i];
                incoming_g[ii] = static_cast<T>(g[i]);
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
                m[j] = b1*incoming_m[ii] + (1-b1)*scaled_grad;
                v[j] = b2*incoming_v[ii] + (1-b2)*scaled_grad*scaled_grad;
                float denom;
                if (mode == ADAM_MODE_0)
                  denom = sqrtf(v[j] + eps);
                else // Mode 1
                  denom = sqrtf(v[j]) + eps;
                float update = (m[j]/denom) + (decay*incoming_p[ii]);
                p[j] = incoming_p[ii] - (step_size*update);
                if (DEPTH == 5)  p_copy[j] = (GRAD_T) p[j];
              }
            }
          }
        }
    }
};

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
        TORCH_CHECK(at::cuda::detail::canUse32BitIndexMath(p), "parameter tensor is too large to be indexed with int32");
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
            TORCH_CHECK(p.scalar_type() == at::ScalarType::Float, "expected parameter to be of float type");
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
      C10_CUDA_CHECK(cudaGetLastError());

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
    TORCH_CHECK(tl_sz == 4 || tl_sz == 5, "expected tensor lists of size 4 or 5");

    if (tensor_lists[3][0].scalar_type() == at::ScalarType::Half) {
//alher values should be fp32 for half gradients
        TORCH_CHECK(tensor_lists[0][0].scalar_type() == at::ScalarType::Float, "expected parameter to be of float type");
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
    C10_CUDA_CHECK(cudaGetLastError());
}

template <typename FROM_T, typename TO_T> 
__device__ void convert(const FROM_T vi, TO_T& vo)
{
    vo = static_cast<TO_T>(vi);
}

template <>
__device__ void convert(const float vi, uint8_t& vo)
{
    union S
    {
	float as_float;
	int as_int;
    };
    S s;
    s.as_float = vi;
    s.as_int = s.as_int & 0xFF800000;
    union T
    {
        at::Half as_half;
	uint8_t as_byte[2];
    };
    T t;
    t.as_half = static_cast<at::Half>(vi + s.as_float / 8.0f);
    vo = t.as_byte[1];
}

template <>
__device__ void convert(const uint8_t vi, float& vo)
{
    union T
    {
        at::Half as_half;
	uint8_t as_byte[2];
    };
    T t;
    t.as_byte[0] = 0;
    t.as_byte[1] = vi;
    vo = static_cast<float>(t.as_half);
}

template <>
__device__ void convert(const at::Half vi, uint8_t& vo)
{
    union S
    {
	float as_float;
	int as_int;
    };
    S s;
    s.as_float = static_cast<float>(vi);
    s.as_int = s.as_int & 0xFF800000;
    union T
    {
        at::Half as_half;
	uint8_t as_byte[2];
    };
    T t;
    t.as_half = static_cast<at::Half>(vi + s.as_float / 8.0f);
    vo = t.as_byte[1];
}

template <>
__device__ void convert(const uint8_t vi, at::Half& vo)
{
    union T
    {
        at::Half as_half;
	uint8_t as_byte[2];
    };
    T t;
    t.as_byte[0] = 0;
    t.as_byte[1] = vi;
    vo = t.as_half;
}

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
template <>
__global__ void strided_check_finite_cuda_kernel(
        volatile int* noop_gmem,
        uint8_t* __restrict__ p_copy,
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
        at::Half pi;
        convert(p_copy[j], pi);
        if (!isfinite(pi)) {
            *noop_gmem = 1;
        }
    }
}

template <typename FROM_T, typename TO_T> 
__global__ void maybe_cast_kernel(
        volatile int* overflow_flag,
        const FROM_T* p_in,
        TO_T* p_out,
        const size_t tsize)
{
    if (overflow_flag && *overflow_flag != 0) return;

    //Assuming 2D grids and 2D blocks
    const int blockId = gridDim.x * blockIdx.y + blockIdx.x;
    const int threadsPerBlock = blockDim.x * blockDim.y;
    const int threadIdInBlock = threadIdx.y * blockDim.x + threadIdx.x;
    const int i = (blockId * threadsPerBlock + threadIdInBlock);
    const int totThreads = gridDim.x*gridDim.y*threadsPerBlock;

    FROM_T pi[ILP];
    TO_T po[ILP];

    for(int j_start = 0;  j_start < tsize;  j_start+=totThreads*ILP) {
#pragma unroll
        for(int ii = 0; ii < ILP; ii++) {
            pi[ii] = 0;

            int j = j_start + i + totThreads*ii;
            if (j < tsize) {
                pi[ii] = p_in[j];
            }
        }

#pragma unroll
        for(int ii = 0; ii < ILP; ii++) {
            convert(pi[ii], po[ii]);
        }

#pragma unroll
        for(int ii = 0; ii < ILP; ii++) {
            int j = j_start + i + totThreads*ii;
            if (j < tsize) {
                p_out[j] = po[ii];
            }
        }
    }
}

template <typename T, typename GRAD_T, typename REDU_T>
__global__ void reversible_adam_cuda_kernel(
        T* __restrict__ p,
        REDU_T* __restrict__ p_copy, // For mixed precision training, pass NULL if not needed
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
                if (p_copy != NULL) {
                    convert(pi[ii], p_copy[j]);
                }
            }
        }
    }

    if (p_copy != NULL) {
        __syncthreads();
        if (overflow) {
            convert(float(INFINITY), p_copy[0]);
        }
    }
}

template <typename T, typename GRAD_T>
__global__ void maybe_adam_undo_cuda_kernel(
        volatile int* overflow_flag,
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
    // NB! Skip undo kernel when overflow flag is NOT set
    if (overflow_flag && *overflow_flag == 0) return;

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

template <int DEPTH, typename FROM_T, typename TO_T>
struct MaybeCastFunctor
{
    __device__ __forceinline__ void operator()(
        int chunk_size,
        volatile int* overflow_flag,
        TensorListMetadata<DEPTH>& tl)
    {
        if (overflow_flag && *overflow_flag != 0) return;

        int tensor_loc = tl.block_to_tensor[blockIdx.x];
        int chunk_idx = tl.block_to_chunk[blockIdx.x];
        int n = tl.sizes[tensor_loc];

        FROM_T* p_in = (FROM_T *)tl.addresses[0][tensor_loc];
        p_in += chunk_idx*chunk_size;
        TO_T* p_out = (TO_T *)tl.addresses[1][tensor_loc];
        p_out += chunk_idx*chunk_size;

        n -= chunk_idx*chunk_size;
        int dim = chunk_size < n ? chunk_size : n;

	FROM_T pi[ILP];
        TO_T po[ILP];

        for(int j_start = 0;  j_start < dim;  j_start+=blockDim.x*ILP) {
#pragma unroll
            for(int ii = 0; ii < ILP; ii++) {
                pi[ii] = FROM_T(0);
                int j = j_start + threadIdx.x + ii*blockDim.x;
                if (j < dim) {
                    pi[ii] = p_in[j];
                }
            }

#pragma unroll
            for(int ii = 0; ii < ILP; ii++) {
                convert(pi[ii], po[ii]);
            }

#pragma unroll
            for(int ii = 0; ii < ILP; ii++) {
                int j = j_start + threadIdx.x + ii*blockDim.x;
                if (j < dim) {
                    p_out[j] = po[ii];
                }
            }
        }
    }
};

void fused_strided_check_finite(
	at::Tensor & overflow_flag,
        at::Tensor & p_copy,
        int stride,
	int clear_overflow_first)
{
	//Get tensor size
	int tsize = p_copy.numel();
	int niter = (tsize + stride - 1) / stride;

	//Determine #threads and #blocks
	const int threadsPerBlock = 512;
	//In order to avoid race condition, blocks must be 1 when clear_overflow_first flag is set.
	const dim3 blocks(clear_overflow_first ? 1 : (niter+threadsPerBlock-1)/threadsPerBlock);
	TORCH_CHECK(at::cuda::detail::canUse32BitIndexMath(p_copy), "parameter tensor is too large to be indexed with int32");

	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        using namespace at; // prevents "toString is undefined" errors
        DISPATCH_FLOAT_HALF_AND_BYTE(p_copy.scalar_type(), 0, "check_finite_cuda_kernel",
                strided_check_finite_cuda_kernel<scalar_t_0><<<blocks,threadsPerBlock, 0, stream>>>(
                    overflow_flag.DATA_PTR<int>(),
                    p_copy.DATA_PTR<scalar_t_0>(),
                    tsize,
                    stride,
                    clear_overflow_first);
                );
	C10_CUDA_CHECK(cudaGetLastError());
}

void fused_reversible_adam_cuda(
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
//      using namespace at;

      //Get tensor size
      int tsize = p.numel();
      //Determine #threads and #blocks
      const int threadsPerBlock = 512;
      const dim3 blocks((tsize+threadsPerBlock-1)/threadsPerBlock);
      TORCH_CHECK(at::cuda::detail::canUse32BitIndexMath(p), "parameter tensor is too large to be indexed with int32");
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
          TORCH_CHECK(p.scalar_type() == at::ScalarType::Float, "expected parameter to be of float type");
          //dispatch is done on the gradient type
          using namespace at; // prevents "toString is undefined" errors
          if (p_copy.numel() == 0 || p_copy.scalar_type() == g.scalar_type()) {
              DISPATCH_FLOAT_AND_HALF(g.scalar_type(), 0, "adam_cuda_kernel",
                      using accscalar_t = at::acc_type<scalar_t_0, true>;
                      reversible_adam_cuda_kernel<accscalar_t, scalar_t_0, scalar_t_0><<<blocks,threadsPerBlock, 0, stream>>>(
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
              TORCH_CHECK(p_copy.scalar_type() == at::ScalarType::Byte, "expected parameter to be of byte type");
              DISPATCH_FLOAT_AND_HALF(g.scalar_type(), 0, "adam_cuda_e5m2_kernel",
                      using accscalar_t = at::acc_type<scalar_t_0, true>;
                      reversible_adam_cuda_kernel<accscalar_t, scalar_t_0, uint8_t><<<blocks,threadsPerBlock, 0, stream>>>(
                          p.DATA_PTR<accscalar_t>(),
                          p_copy.DATA_PTR<uint8_t>(),
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
          }
      } else {
          using namespace at;
          DISPATCH_DOUBLE_AND_FLOAT(g.scalar_type(), 0, "adam_cuda_kernel",
                  reversible_adam_cuda_kernel<scalar_t_0, scalar_t_0, scalar_t_0><<<blocks,threadsPerBlock, 0, stream>>>(
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
      C10_CUDA_CHECK(cudaGetLastError());
}

void maybe_cast_cuda(
        at::Tensor & overflow_flag,
        at::Tensor & p_in,
        at::Tensor & p_out)
{
      //Get tensor size
      int tsize = p_in.numel();
      TORCH_CHECK(tsize == p_out.numel(), "p_in.numel() must equal p_out.numel()");
      //Determine #threads and #blocks
      const int threadsPerBlock = 512;
      const dim3 blocks((tsize+threadsPerBlock-1)/threadsPerBlock);
      TORCH_CHECK(at::cuda::detail::canUse32BitIndexMath(p_in), "parameter tensor is too large to be indexed with int32");
      //Constants
      cudaStream_t stream = at::cuda::getCurrentCUDAStream();
      DISPATCH_FLOAT_HALF_AND_BYTE(p_in.scalar_type(), 0, "maybe_cast_cuda"
              DISPATCH_FLOAT_HALF_AND_BYTE(p_out.scalar_type(), 1, "maybe_cast_cuda",
                  maybe_cast_kernel<scalar_t_0,scalar_t_1><<<blocks,threadsPerBlock, 0, stream>>>(
                      overflow_flag.numel() ? overflow_flag.DATA_PTR<int>() : NULL,
                      p_in.DATA_PTR<scalar_t_0>(),
                      p_out.DATA_PTR<scalar_t_1>(),
                      tsize); ))
      C10_CUDA_CHECK(cudaGetLastError());
}

void maybe_cast_cuda_mt(
    int chunk_size,
    at::Tensor overflow_flag,
    std::vector<std::vector<at::Tensor>> tensor_lists) // p_in, p_out
{
    //Constants
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    size_t tl_sz = tensor_lists.size();
    TORCH_CHECK(tl_sz == 2, "expected tensor lists of size 2");

    DISPATCH_FLOAT_HALF_AND_BYTE(tensor_lists[0][0].scalar_type(), 0, "maybe_cast_cuda_mt_kernel",
            DISPATCH_FLOAT_HALF_AND_BYTE(tensor_lists[1][0].scalar_type(), 1, "maybe_cast_cuda_mt_kernel",
                multi_tensor_apply<2>(
                    BLOCK_SIZE,
                    chunk_size,
                    overflow_flag,
                    tensor_lists,
                    MaybeCastFunctor<2, scalar_t_0, scalar_t_1>()); ))
    C10_CUDA_CHECK(cudaGetLastError());
}

void fused_maybe_adam_undo_cuda(
        at::Tensor & overflow_flag,
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
    //Get tensor size
    int tsize = p.numel();
    //Determine #threads and #blocks
    const int threadsPerBlock = 512;
    const dim3 blocks((tsize+threadsPerBlock-1)/threadsPerBlock);
    TORCH_CHECK(at::cuda::detail::canUse32BitIndexMath(p), "parameter tensor is too large to be indexed with int32");
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
        TORCH_CHECK(p.scalar_type() == at::ScalarType::Float, "expected parameter to be of float type");
        //dispatch is done on the gradient type
        using namespace at; // prevents "toString is undefined" errors
        DISPATCH_FLOAT_AND_HALF(g.scalar_type(), 0, "adam_cuda_kernel",
                using accscalar_t = at::acc_type<scalar_t_0, true>;
                maybe_adam_undo_cuda_kernel<accscalar_t, scalar_t_0><<<blocks,threadsPerBlock, 0, stream>>>(
                    overflow_flag.numel() ? overflow_flag.DATA_PTR<int>() : NULL,
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
                maybe_adam_undo_cuda_kernel<scalar_t_0, scalar_t_0><<<blocks,threadsPerBlock, 0, stream>>>(
                    overflow_flag.numel() ? overflow_flag.DATA_PTR<int>() : NULL,
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
    C10_CUDA_CHECK(cudaGetLastError());
}
