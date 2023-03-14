#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
// Another possibility:
// #include <torch/all.h>

#include <assert.h>
#include <cmath>
#include "type_shim.h"
#include "multi_tensor_apply.cuh"

#define BLOCK_SIZE 512
#define ILP 4

template<typename T>
__device__ __forceinline__ bool is_aligned(const T* p){
  return ((uint64_t)p) % (ILP*sizeof(T)) == 0;
}

template<typename T>
__device__ __forceinline__ void load_store(
  T* dst,
  const T* src,
  int dst_offset = 0,
  int src_offset = 0){
  typedef typename std::aligned_storage<ILP*sizeof(T), ILP*alignof(T)>::type LT;
  ((LT*)dst)[dst_offset] = ((const LT*)src)[src_offset];
}

// (1-t)*x + t*y
__device__ __forceinline__ float lerp(float t, float x, float y) {
  // See https://developer.nvidia.com/blog/lerp-faster-cuda/
  return fma(t, y, fma(-t, x, x));
}

typedef enum{
  ADAM_MODE_0   =0, // L2 regularization mode
  ADAM_MODE_1   =1  // Decoupled weight decay mode(AdamW)
} adamMode_t;

/* Multi-tensor Adam
 *
 * Updates params in-place and outputs a copy with a desired datatype.
 */
template <typename T, typename GRAD_T, typename PARAM_OUT_T>
struct DistAdamFunctor
{
  // Vectorized local compute
  __device__ __forceinline__ static void local_step(
    T p[ILP],
    T m[ILP],
    T v[ILP],
    const GRAD_T g[ILP],
    const float grad_scale,
    const float beta1,
    const float beta2,
    const float beta1_correction,
    const float beta2_correction,
    const float eps,
    const float lr,
    adamMode_t mode,
    const float weight_decay) {
    if (mode == ADAM_MODE_0) { // L2
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        float scaled_grad = (g[ii] * grad_scale) + (weight_decay * p[ii]);
        float next_m = lerp(beta1, scaled_grad, m[ii]);
        float next_v = lerp(beta2, scaled_grad*scaled_grad, v[ii]);
        float next_m_unbiased = next_m / beta1_correction;
        float next_v_unbiased = next_v / beta2_correction;
        float denom = sqrtf(next_v_unbiased) + eps;
        float update = next_m_unbiased / denom;
        m[ii] = next_m;
        v[ii] = next_v;
        p[ii] -= lr * update;
      }
    } else { // weight decay
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        float scaled_grad = g[ii] * grad_scale;
        float next_m = lerp(beta1, scaled_grad, m[ii]);
        float next_v = lerp(beta2, scaled_grad*scaled_grad, v[ii]);
        float next_m_unbiased = next_m / beta1_correction;
        float next_v_unbiased = next_v / beta2_correction;
        float denom = sqrtf(next_v_unbiased) + eps;
        float update = (next_m_unbiased / denom) + (weight_decay * p[ii]);
        m[ii] = next_m;
        v[ii] = next_v;
        p[ii] -= lr * update;
      }
    }
  }

  __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<5>& tl,
    const float* grad_scale_ptr,
    const float beta1,
    const float beta2,
    const float beta1_correction,
    const float beta2_correction,
    const float eps,
    const float lr,
    adamMode_t mode,
    const float weight_decay) const
  {
    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    const float grad_scale = *grad_scale_ptr;

    T* p_in = (T *)tl.addresses[0][tensor_loc];
    p_in += chunk_idx*chunk_size;
    T* m = (T *)tl.addresses[1][tensor_loc];
    m += chunk_idx*chunk_size;
    T* v = (T *)tl.addresses[2][tensor_loc];
    v += chunk_idx*chunk_size;
    const GRAD_T* g = (GRAD_T *)tl.addresses[3][tensor_loc];
    g += chunk_idx*chunk_size;
    PARAM_OUT_T* p_out = (PARAM_OUT_T *)tl.addresses[4][tensor_loc];
    p_out += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;
    n = chunk_size < n ? chunk_size : n;

    const bool aligned = (n % ILP == 0 &&
                          is_aligned(p_in) &&
                          is_aligned(m) &&
                          is_aligned(v) &&
                          is_aligned(g) &&
                          is_aligned(p_out));

    for (int i_start = threadIdx.x*ILP; i_start < n; i_start += blockDim.x*ILP) {
      T local_p[ILP];
      T local_m[ILP];
      T local_v[ILP];
      GRAD_T local_g[ILP];
      PARAM_OUT_T local_p_out[ILP];

      // Load
      if (aligned) {
        load_store(local_p, p_in + i_start);
        load_store(local_m, m + i_start);
        load_store(local_v, v + i_start);
        load_store(local_g, g + i_start);
      } else {
#pragma unroll
        for (int ii = 0, i = i_start; ii < ILP; ii++, i++) {
          if (i < n) {
            local_p[ii] = p_in[i];
            local_m[ii] = m[i];
            local_v[ii] = v[i];
            local_g[ii] = g[i];
          } else {
            local_p[ii] = 0;
            local_m[ii] = 0;
            local_v[ii] = 0;
            local_g[ii] = 0;
          }
        }
      }

      // Local compute
      local_step(
        local_p, local_m, local_v, local_g, grad_scale,
        beta1, beta2, beta1_correction, beta2_correction,
        eps, lr, mode, weight_decay);
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        local_p_out[ii] = static_cast<PARAM_OUT_T>(local_p[ii]);
      }

      // Store
      if (aligned) {
        load_store(p_in + i_start, local_p);
        load_store(m + i_start, local_m);
        load_store(v + i_start, local_v);
        load_store(p_out + i_start, local_p_out);
      } else {
#pragma unroll
        for (int ii = 0, i = i_start; ii < ILP; ii++, i++) {
          if (i < n) {
            p_in[i] = local_p[ii];
            m[i] = local_m[ii];
            v[i] = local_v[ii];
	    p_out[i] = local_p_out[ii];
          }
        }
      }
    }
  }
};

/* Functor for multi-tensor Adam with implicit main params
 *
 * If params are BF16 and optimizer state is FP32, it is not necessary
 * to store FP32 main params. Instead, store 16-bit param remainder
 * and combine with BF16 param to reconstruct the FP32 main param.
 */
template <typename GRAD_T>
struct DistAdamWithParamRemaindersFunctor
{
  __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<6>& tl,
    const float* grad_scale_ptr,
    const float beta1,
    const float beta2,
    const float beta1_correction,
    const float beta2_correction,
    const float eps,
    const float lr,
    adamMode_t mode,
    const float weight_decay) const
  {
    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    const float grad_scale = *grad_scale_ptr;

    int16_t* p_in = (int16_t *)tl.addresses[0][tensor_loc];
    p_in += chunk_idx*chunk_size;
    int16_t* p_rem = (int16_t *)tl.addresses[1][tensor_loc];
    p_rem += chunk_idx*chunk_size;
    float* m = (float *)tl.addresses[2][tensor_loc];
    m += chunk_idx*chunk_size;
    float* v = (float *)tl.addresses[3][tensor_loc];
    v += chunk_idx*chunk_size;
    const GRAD_T* g = (GRAD_T *)tl.addresses[4][tensor_loc];
    g += chunk_idx*chunk_size;
    int16_t* p_out = (int16_t *)tl.addresses[5][tensor_loc];
    p_out += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;
    n = chunk_size < n ? chunk_size : n;

    const bool aligned = (n % ILP == 0 &&
                          is_aligned(p_in) &&
                          is_aligned(p_rem) &&
                          is_aligned(m) &&
                          is_aligned(v) &&
                          is_aligned(g) &&
                          is_aligned(p_out));

    for (int i_start = threadIdx.x*ILP; i_start < n; i_start += blockDim.x*ILP) {
      union fp32_or_int162 {
        float fp32;
        int16_t int16[2];
      };
      fp32_or_int162 local_p[ILP];
      int16_t local_p_bf16[ILP];
      int16_t local_p_rem[ILP];
      float local_m[ILP];
      float local_v[ILP];
      GRAD_T local_g[ILP];

      // Load
      if (aligned) {
        load_store(local_p_bf16, p_in + i_start);
        load_store(local_p_rem, p_rem + i_start);
        load_store(local_m, m + i_start);
        load_store(local_v, v + i_start);
        load_store(local_g, g + i_start);
      } else {
#pragma unroll
        for (int ii = 0, i = i_start; ii < ILP; ii++, i++) {
          if (i < n) {
            local_p_bf16[ii] = p_in[i];
            local_p_rem[ii] = p_rem[i];
            local_m[ii] = m[i];
            local_v[ii] = v[i];
            local_g[ii] = g[i];
          } else {
            local_p_bf16[ii] = 0;
            local_p_rem[ii] = 0;
            local_m[ii] = 0;
            local_v[ii] = 0;
            local_g[ii] = 0;
          }
        }
      }

      // Reconstruct FP32 params
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        if (local_p_rem[ii] < 0)
          local_p_bf16[ii]--; // Undo rounding
        local_p[ii].int16[1] = local_p_bf16[ii];
        local_p[ii].int16[0] = local_p_rem[ii];
      }

      // Local compute
      using LocalFunctor = DistAdamFunctor<float, GRAD_T, void>;
      LocalFunctor::local_step(
        reinterpret_cast<float *>(local_p), local_m, local_v, local_g, grad_scale,
        beta1, beta2, beta1_correction, beta2_correction,
        eps, lr, mode, weight_decay);

      // Split into BF16 params (rounded-to-nearest) and remainders
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        local_p_bf16[ii] = local_p[ii].int16[1];
        local_p_rem[ii] = local_p[ii].int16[0];
        if (local_p_rem[ii] < 0)
          local_p_bf16[ii]++; // Round up
      }

      // Store
      if (aligned) {
        load_store(p_rem + i_start, local_p_rem);
        load_store(m + i_start, local_m);
        load_store(v + i_start, local_v);
        load_store(p_out + i_start, local_p_bf16);
      } else {
#pragma unroll
        for (int ii = 0, i = i_start; ii < ILP; ii++, i++) {
          if (i < n) {
            p_rem[i] = local_p_rem[ii];
            m[i] = local_m[ii];
            v[i] = local_v[ii];
	    p_out[i] = local_p_bf16[ii];
          }
        }
      }
    }
  }
};

void multi_tensor_fused_adam_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,  // p_in, m, v, g, p_out
  at::Tensor grad_scale,
  float lr,
  float beta1,
  float beta2,
  float eps,
  int step,
  int mode,
  int bias_correction,
  float weight_decay)
{
  using namespace at;

  // Expect p_in, m, v, g, p_out
  size_t tl_sz = tensor_lists.size();
  TORCH_CHECK(tl_sz == 5, "expected tensor lists of size 5");
  const auto p_in_type = tensor_lists[0][0].scalar_type();
  const auto g_type = tensor_lists[3][0].scalar_type();
  const auto p_out_type = tensor_lists[4][0].scalar_type();

  float beta1_correction = 1.0f, beta2_correction = 1.0f;
  if (bias_correction == 1) {
    beta1_correction = 1 - std::pow(beta1, step);
    beta2_correction = 1 - std::pow(beta2, step);
  }

  DISPATCH_FLOAT_HALF_AND_BFLOAT(p_in_type, 0, "dist_adam_cuda_kernel",
    DISPATCH_FLOAT_HALF_AND_BFLOAT(g_type, 1, "dist_adam_cuda_kernel",
      DISPATCH_FLOAT_HALF_AND_BFLOAT(p_out_type, 2, "dist_adam_cuda_kernel",
        multi_tensor_apply<5>(
          BLOCK_SIZE,
          chunk_size,
          noop_flag,
          tensor_lists,
          DistAdamFunctor<scalar_t_0, scalar_t_1, scalar_t_2>(),
          grad_scale.data_ptr<float>(),
          beta1,
          beta2,
          beta1_correction,
          beta2_correction,
          eps,
          lr,
          (adamMode_t) mode,
          weight_decay);
  )));
  C10_CUDA_CHECK(cudaGetLastError());
}

void multi_tensor_fused_adam_with_param_remainders_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,  // p_in, p_rem, m, v, g, p_out
  at::Tensor grad_scale,
  float lr,
  float beta1,
  float beta2,
  float eps,
  int step,
  int mode,
  int bias_correction,
  float weight_decay)
{
  using namespace at;

  // Expect p_in, p_rem, m, v, g, p_out
  size_t tl_sz = tensor_lists.size();
  TORCH_CHECK(tl_sz == 6, "expected tensor lists of size 6");
  const auto g_type = tensor_lists[4][0].scalar_type();

  float beta1_correction = 1.0f, beta2_correction = 1.0f;
  if (bias_correction == 1) {
    beta1_correction = 1 - std::pow(beta1, step);
    beta2_correction = 1 - std::pow(beta2, step);
  }

  DISPATCH_FLOAT_HALF_AND_BFLOAT(g_type, 0, "dist_adam_with_param_remainders_cuda_kernel",
    multi_tensor_apply<6>(
      BLOCK_SIZE,
      chunk_size,
      noop_flag,
      tensor_lists,
      DistAdamWithParamRemaindersFunctor<scalar_t_0>(),
      grad_scale.data_ptr<float>(),
      beta1,
      beta2,
      beta1_correction,
      beta2_correction,
      eps,
      lr,
      (adamMode_t) mode,
      weight_decay);
  );
  C10_CUDA_CHECK(cudaGetLastError());
}
