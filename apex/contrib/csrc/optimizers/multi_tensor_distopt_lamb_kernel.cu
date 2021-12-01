#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
// Another possibility:
// #include <torch/all.h>

#include <assert.h>

#include "type_shim.h"
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

typedef enum{
  MOMENT_MODE_0   =0, // L2 regularization mode
  MOMENT_MODE_1   =1  // Decoupled weight decay mode
} adamMode_t;

template<typename T, typename GRAD_T, typename MATH_T>
struct DistOptLAMBStage1Functor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<5>& tl,
    const MATH_T* per_tensor_beta1,
    const MATH_T* per_tensor_beta2,
    const MATH_T* per_tensor_beta3,
    const int* per_tensor_bias_correction,
    const int* step,
    const MATH_T* per_tensor_epsilon,
    adamMode_t mode,
    const MATH_T* per_tensor_decay,
    const MATH_T* global_scale,
    const MATH_T* global_grad_norm,
    const float max_grad_norm)
  {
    // I'd like this kernel to propagate infs/nans.
    if (*noop_gmem == 1)
        return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int tensor_num = tl.start_tensor_this_launch + tensor_loc;
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    float combined_scale = *global_scale;
    if (max_grad_norm > 0) {
        combined_scale = max_grad_norm / (*global_grad_norm / *global_scale + 1e-6);
	combined_scale = *global_scale / std::min((float) 1.0, combined_scale);
    }

    MATH_T beta1 = per_tensor_beta1[tensor_num];
    MATH_T beta2 = per_tensor_beta2[tensor_num];
    MATH_T beta3 = 1 - beta1;
    MATH_T beta1_correction, beta2_correction;
    if (per_tensor_bias_correction[tensor_num] == 1) {
        beta1_correction = 1 - pow(beta1, *step);
        beta2_correction = 1 - pow(beta2, *step);
    } else {
        beta1_correction = (MATH_T) 1.0;
        beta2_correction = (MATH_T) 1.0;
    }
    MATH_T epsilon = per_tensor_epsilon[tensor_num];
    MATH_T decay = per_tensor_decay[tensor_num];

    GRAD_T* g = (GRAD_T*)tl.addresses[0][tensor_loc];
    g += chunk_idx*chunk_size;

    T* p = (T*)tl.addresses[1][tensor_loc];
    p += chunk_idx*chunk_size;

    T* m = (T*)tl.addresses[2][tensor_loc];
    m += chunk_idx*chunk_size;

    T* v = (T*)tl.addresses[3][tensor_loc];
    v += chunk_idx*chunk_size;

    MATH_T* u = (MATH_T*)tl.addresses[4][tensor_loc];
    u += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    MATH_T r_g[ILP];
    MATH_T r_p[ILP];
    MATH_T r_m[ILP];
    MATH_T r_v[ILP];
    // to make things simple, we put aligned case in a different code path
    if(n % ILP == 0 &&
       chunk_size % ILP == 0 &&
       is_aligned(g) &&
       is_aligned(p) &&
       is_aligned(m) &&
       is_aligned(v))
    {
      GRAD_T l_g[ILP];
      T l_p[ILP];
      T l_m[ILP];
      T l_v[ILP];
      for(int i_start = threadIdx.x; i_start*ILP < n && i_start*ILP < chunk_size; i_start += blockDim.x)
      {
        // load
        load_store(l_g, g, 0, i_start);
        if (decay != 0)
          load_store(l_p, p, 0, i_start);
        load_store(l_m, m, 0, i_start);
        load_store(l_v, v, 0, i_start);
        // unpack
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          r_g[ii] = l_g[ii];
          if (decay == 0) {
            r_p[ii] = MATH_T(0);
          }
          else {
            r_p[ii] = l_p[ii];
          }
          r_m[ii] = l_m[ii];
          r_v[ii] = l_v[ii];
        }
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          if (mode == MOMENT_MODE_0) {
            MATH_T scaled_grad = r_g[ii] / combined_scale;
            // L2 on scaled grad
            scaled_grad = scaled_grad + decay*r_p[ii];
            r_m[ii] = r_m[ii] * beta1 + beta3 * scaled_grad;
            r_v[ii] = r_v[ii] * beta2 + (1-beta2) * scaled_grad * scaled_grad;
            MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
            MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
            MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
            r_p[ii] = next_m_unbiased / denom;
          }
          else {
            MATH_T scaled_grad = r_g[ii] / combined_scale;
            r_m[ii] = r_m[ii] * beta1 + beta3 * scaled_grad;
            r_v[ii] = r_v[ii] * beta2 + (1-beta2) * scaled_grad * scaled_grad;
            MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
            MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
            MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
            r_p[ii] = (next_m_unbiased/denom) + (decay*r_p[ii]);
          }
        }
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          l_m[ii] = r_m[ii];
          l_v[ii] = r_v[ii];
        }
        // store
        load_store(u, r_p, i_start, 0);
        load_store(m, l_m, i_start, 0);
        load_store(v, l_v, i_start, 0);
      }
    }
    else
    {
      // see note in multi_tensor_scale_kernel.cu
      for(int i_start = 0;
          i_start < n && i_start < chunk_size;
          i_start += blockDim.x*ILP)
      {
        MATH_T r_g[ILP];
        MATH_T r_p[ILP];
        MATH_T r_m[ILP];
        MATH_T r_v[ILP];
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          int i = i_start + threadIdx.x + ii*blockDim.x;
          if(i < n && i < chunk_size)
          {
            r_g[ii] = g[i];
            // special ?optimization? for lamb stage 1
            if (decay == 0) {
              r_p[ii] = MATH_T(0);
            }
            else {
              r_p[ii] = p[i];
            }
            r_m[ii] = m[i];
            r_v[ii] = v[i];
          } else {
            r_g[ii] = MATH_T(0);
            r_p[ii] = MATH_T(0);
            r_m[ii] = MATH_T(0);
            r_v[ii] = MATH_T(0);
          }
        }
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          if (mode == MOMENT_MODE_0) {
            MATH_T scaled_grad = r_g[ii] / combined_scale;
            // L2 on scaled grad
            scaled_grad = scaled_grad + decay*r_p[ii];
            r_m[ii] = r_m[ii] * beta1 + beta3 * scaled_grad;
            r_v[ii] = r_v[ii] * beta2 + (1-beta2) * scaled_grad * scaled_grad;
            MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
            MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
            MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
            r_p[ii] = next_m_unbiased / denom;
          }
          else {
            MATH_T scaled_grad = r_g[ii] / combined_scale;
            r_m[ii] = r_m[ii] * beta1 + beta3 * scaled_grad;
            r_v[ii] = r_v[ii] * beta2 + (1-beta2) * scaled_grad * scaled_grad;
            MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
            MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
            MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
            r_p[ii] = (next_m_unbiased/denom) + (decay*r_p[ii]);
          }
        }
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          int i = i_start + threadIdx.x + ii*blockDim.x;
          if(i < n && i < chunk_size)
          {
            u[i] = r_p[ii];
            m[i] = r_m[ii];
            v[i] = r_v[ii];
          }
        }
      }
    }
  }
};

// Step 2 reads in 'update' value and per-tensor param_norm and update_norm.
// It computes new parameter value.
template<typename T, typename GRAD_T, typename MATH_T>
struct DistOptLAMBStage2Functor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<3>& tl,
    const MATH_T* per_tensor_param_norm,
    const MATH_T* per_tensor_update_norm,
    const long* update_norm_offset,
    const MATH_T* learning_rate,
    const MATH_T* per_tensor_decay,
    const MATH_T* global_grad_norm,
    bool use_nvlamb)
  {
    // I'd like this kernel to propagate infs/nans.
    if (*noop_gmem == 1)
        return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int tensor_num = tl.start_tensor_this_launch + tensor_loc;
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    MATH_T decay = per_tensor_decay[tensor_num];

    MATH_T ratio = *learning_rate;
    // nvlamb: apply adaptive learning rate to all parameters
    // otherwise, only apply to those with non-zero weight decay
    if (use_nvlamb || (decay != (MATH_T) 0.0))
    {
      MATH_T param_norm = per_tensor_param_norm[tensor_num];
      MATH_T update_norm = per_tensor_update_norm[update_norm_offset[tensor_num]];
      ratio = (update_norm != 0.0 && param_norm != 0.0) ? (*learning_rate) * (param_norm / update_norm) : (*learning_rate);
    }

    MATH_T* update = (MATH_T*)tl.addresses[0][tensor_loc];
    update += chunk_idx*chunk_size;

    T* p = (T*)tl.addresses[1][tensor_loc];
    p += chunk_idx*chunk_size;

    GRAD_T* p_copy = (GRAD_T*)tl.addresses[2][tensor_loc];
    p_copy += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    // to make things simple, we put aligned case in a different code path
    if(n % ILP == 0 &&
       chunk_size % ILP == 0 &&
       is_aligned(p) &&
       is_aligned(update))
    {
      T r_p[ILP];
      MATH_T r_update[ILP];
      GRAD_T r_p_copy[ILP];
      for(int i_start = threadIdx.x; i_start*ILP < n && i_start*ILP < chunk_size; i_start += blockDim.x)
      {
        // load
        load_store(r_p, p, 0, i_start);
        load_store(r_update, update, 0, i_start);
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
	  r_p[ii] = static_cast<MATH_T>(r_p[ii]) - (ratio * r_update[ii]);
          convert(r_p[ii], r_p_copy[ii]);
        }
        load_store(p, r_p, i_start, 0);
        load_store(p_copy, r_p_copy, i_start, 0);
      }
    }
    else
    {
      for(int i_start = 0;
          i_start < n && i_start < chunk_size;
          i_start += blockDim.x*ILP)
      {
        MATH_T r_p[ILP];
        MATH_T r_update[ILP];
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          int i = i_start + threadIdx.x + ii*blockDim.x;
          if(i < n && i < chunk_size)
          {
            r_p[ii] = p[i];
            r_update[ii] = update[i];
          }
        }
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          r_p[ii] = r_p[ii] - (ratio * r_update[ii]);
        }
#pragma unroll
        for(int ii = 0; ii < ILP; ii++)
        {
          int i = i_start + threadIdx.x + ii*blockDim.x;
          if(i < n && i < chunk_size)
          {
            p[i] = r_p[ii];
            convert(r_p[ii], p_copy[i]);
          }
        }
      }
    }
  }
};

void multi_tensor_lamb_compute_update_term_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor per_tensor_beta1,
  at::Tensor per_tensor_beta2,
  at::Tensor per_tensor_beta3,
  at::Tensor per_tensor_bias_correction,
  at::Tensor step,
  at::Tensor per_tensor_epsilon,
  const int mode,
  at::Tensor per_tensor_decay,
  at::Tensor global_scale,
  at::Tensor global_grad_norm,
  const float max_grad_norm)
{
  using namespace at;

  DISPATCH_FLOAT_AND_HALF(tensor_lists[1][0].scalar_type(), 0, "lamb_stage_1",
    DISPATCH_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(), 1, "lamb_stage_1",
      DISPATCH_FLOAT_AND_HALF(tensor_lists[4][0].scalar_type(), 2, "lamb_stage_1",
        multi_tensor_apply<5>(
          BLOCK_SIZE,
          chunk_size,
          noop_flag,
          tensor_lists,
          DistOptLAMBStage1Functor<scalar_t_0, scalar_t_1, scalar_t_2>(),
          per_tensor_beta1.DATA_PTR<scalar_t_2>(),
          per_tensor_beta2.DATA_PTR<scalar_t_2>(),
          per_tensor_beta3.DATA_PTR<scalar_t_2>(),
          per_tensor_bias_correction.DATA_PTR<int>(),
          step.DATA_PTR<int>(),
          per_tensor_epsilon.DATA_PTR<scalar_t_2>(),
          (adamMode_t) mode,
          per_tensor_decay.DATA_PTR<scalar_t_2>(),
          global_scale.DATA_PTR<scalar_t_2>(),
	  global_grad_norm.DATA_PTR<scalar_t_2>(),
	  max_grad_norm); )))

  AT_CUDA_CHECK(cudaGetLastError());
}

void multi_tensor_lamb_update_weights_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor per_tensor_param_norm,
  at::Tensor per_tensor_update_norm,
  at::Tensor update_norm_offset,
  at::Tensor learning_rate,
  at::Tensor per_tensor_decay,
  at::Tensor global_grad_norm,
  bool use_nvlamb)
{
  using namespace at;

  DISPATCH_FLOAT_AND_HALF(tensor_lists[1][0].scalar_type(), 0, "lamb_stage_2",
    DISPATCH_FLOAT_HALF_AND_BYTE(tensor_lists[2][0].scalar_type(), 1, "lamb_stage_2",
      DISPATCH_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(), 2, "lamb_stage_2",
        multi_tensor_apply<3>(
          BLOCK_SIZE,
          chunk_size,
          noop_flag,
          tensor_lists,
          DistOptLAMBStage2Functor<scalar_t_0, scalar_t_1, scalar_t_2>(),
          per_tensor_param_norm.DATA_PTR<scalar_t_2>(),
          per_tensor_update_norm.DATA_PTR<scalar_t_2>(),
          update_norm_offset.DATA_PTR<long>(),
	  learning_rate.DATA_PTR<scalar_t_2>(),
          per_tensor_decay.DATA_PTR<scalar_t_2>(),
	  global_grad_norm.DATA_PTR<scalar_t_2>(),
          use_nvlamb); )))

  AT_CUDA_CHECK(cudaGetLastError());
}
