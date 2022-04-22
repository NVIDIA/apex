#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
// Another possibility:
// #include <torch/all.h>

#include <curand_kernel.h>
#include <assert.h>

#include "type_shim.h"
#include "multi_tensor_apply.cuh"
#include "stochastic_round.cuh"

#define BLOCK_SIZE 512
#define ILP 4

typedef enum{
  MOMENT_MODE_0   =0, // L2 regularization mode
  MOMENT_MODE_1   =1  // Decoupled weight decay mode
} adamMode_t;

std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::optional<bool> per_tensor_python);

using MATH_T = float;
using MATH_T4 = float4;

template<typename T>
struct LANSStage1Functor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<5>& tl,
    const float beta1,
    const float beta2,
    const float beta3,
    const float beta1_correction,
    const float beta2_correction,
    const float epsilon,
    adamMode_t mode,
    const float decay,
    float* per_tensor_grad_norm,
    bool normalize_grad)
  {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int tensor_num = tl.start_tensor_this_launch + tensor_loc;
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    float grad_norm = per_tensor_grad_norm[tensor_num];

    T* g = (T*)tl.addresses[0][tensor_loc];
    g += chunk_idx*chunk_size;

    T* q = (T*)tl.addresses[1][tensor_loc];
    q += chunk_idx*chunk_size;

    T* p = (T*)tl.addresses[2][tensor_loc];
    p += chunk_idx*chunk_size;

    T* m = (T*)tl.addresses[3][tensor_loc];
    m += chunk_idx*chunk_size;

    T* v = (T*)tl.addresses[4][tensor_loc];
    v += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    // see note in multi_tensor_scale_kernel.cu
    for(int i_start = 0;
            i_start < n && i_start < chunk_size;
            i_start += blockDim.x*ILP)
    {
      MATH_T r_g[ILP];
      MATH_T r_q[ILP];
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
          r_q[ii] = q[i];
          // special ?optimization? for lans stage 1
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
          r_q[ii] = MATH_T(0);
          r_p[ii] = MATH_T(0);
          r_m[ii] = MATH_T(0);
          r_v[ii] = MATH_T(0);
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        MATH_T scaled_grad = r_g[ii];
        if (normalize_grad && grad_norm != 0.0f) {
           scaled_grad /= (grad_norm + epsilon);
        }
        if (mode == MOMENT_MODE_0) {
          // L2 on scaled grad
          scaled_grad = scaled_grad + decay*r_p[ii];
          r_m[ii] = r_m[ii] * beta1 + beta3 * scaled_grad;
          r_v[ii] = r_v[ii] * beta2 + (1-beta2) * scaled_grad * scaled_grad;
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          r_p[ii] = next_m_unbiased / denom;
          r_q[ii] = scaled_grad / denom;
        }
        else {
          r_m[ii] = r_m[ii] * beta1 + beta3 * scaled_grad;
          r_v[ii] = r_v[ii] * beta2 + (1-beta2) * scaled_grad * scaled_grad;
          MATH_T next_m_unbiased = r_m[ii] / beta1_correction;
          MATH_T next_v_unbiased = r_v[ii] / beta2_correction;
          MATH_T denom = sqrtf(next_v_unbiased) + epsilon;
          MATH_T scaled_p = decay * r_p[ii];
          r_p[ii] = (next_m_unbiased/denom) + scaled_p;
          r_q[ii] = (scaled_grad/denom) + scaled_p;
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          g[i] = r_p[ii];
          q[i] = r_q[ii];
          m[i] = r_m[ii];
          v[i] = r_v[ii];
        }
      }
    }
  }
};

// Step 2 reads in 'update' value and per-tensor param_norm and update_norm.
// It computes new parameter value.
template<typename T>
struct LANSStage2Functor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<3>& tl,
    const float beta1,
    const float beta3,
    const float* per_tensor_param_norm,
    const float* per_tensor_update_m_norm,
    const float* per_tensor_update_g_norm,
    const float learning_rate)
  {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int tensor_num = tl.start_tensor_this_launch + tensor_loc;
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    float param_norm = per_tensor_param_norm[tensor_num];
    float update_m_norm = per_tensor_update_m_norm[tensor_num];
    float update_g_norm = per_tensor_update_g_norm[tensor_num];
    MATH_T ratio_m = (update_m_norm != 0.0f && param_norm != 0.0f) ? learning_rate * (param_norm / update_m_norm) : learning_rate;
    MATH_T ratio_g = (update_g_norm != 0.0f && param_norm != 0.0f) ? learning_rate * (param_norm / update_g_norm) : learning_rate;
    ratio_m *= beta1;
    ratio_g *= beta3;

    T* update_m = (T*)tl.addresses[0][tensor_loc];
    update_m += chunk_idx*chunk_size;

    T* update_g = (T*)tl.addresses[1][tensor_loc];
    update_g += chunk_idx*chunk_size;

    T* p = (T*)tl.addresses[2][tensor_loc];
    p += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    for(int i_start = 0;
            i_start < n && i_start < chunk_size;
            i_start += blockDim.x*ILP)
    {
      MATH_T r_p[ILP];
      MATH_T r_update_m[ILP];
      MATH_T r_update_g[ILP];
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
       	int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          r_p[ii] = p[i];
          r_update_m[ii] = update_m[i];
          r_update_g[ii] = update_g[i];
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
       	r_p[ii] = r_p[ii] - (ratio_m * r_update_m[ii]) - (ratio_g * r_update_g[ii]);
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          p[i] = r_p[ii];
        }
      }
    }
  }

  __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<3>& tl,
    const float beta1,
    const float beta3,
    const float* per_tensor_param_norm,
    const float* per_tensor_update_m_norm,
    const float* per_tensor_update_g_norm,
    const float learning_rate,
    at::PhiloxCudaState philox_args)
  {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int tensor_num = tl.start_tensor_this_launch + tensor_loc;
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    float param_norm = per_tensor_param_norm[tensor_num];
    float update_m_norm = per_tensor_update_m_norm[tensor_num];
    float update_g_norm = per_tensor_update_g_norm[tensor_num];
    MATH_T ratio_m = (update_m_norm != 0.0f && param_norm != 0.0f) ? learning_rate * (param_norm / update_m_norm) : learning_rate;
    MATH_T ratio_g = (update_g_norm != 0.0f && param_norm != 0.0f) ? learning_rate * (param_norm / update_g_norm) : learning_rate;
    ratio_m *= beta1;
    ratio_g *= beta3;

    T* update_m = (T*)tl.addresses[0][tensor_loc];
    update_m += chunk_idx*chunk_size;

    T* update_g = (T*)tl.addresses[1][tensor_loc];
    update_g += chunk_idx*chunk_size;

    T* p = (T*)tl.addresses[2][tensor_loc];
    p += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

    curandStatePhilox4_32_10_t state;
    auto seeds = at::cuda::philox::unpack(philox_args);
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(
        std::get<0>(seeds),
        idx,
        std::get<1>(seeds),
        &state);

    for(int i_start = 0;
            i_start < n && i_start < chunk_size;
            i_start += blockDim.x*ILP)
    {
      MATH_T r_p[ILP];
      MATH_T r_update_m[ILP];
      MATH_T r_update_g[ILP];
      MATH_T4 rand = curand_uniform4(&state);
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          r_p[ii] = p[i];
          r_update_m[ii] = update_m[i];
          r_update_g[ii] = update_g[i];
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        r_p[ii] = r_p[ii] - (ratio_m * r_update_m[ii]) - (ratio_g * r_update_g[ii]);
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          p[i] = __stochastic_round<T>(r_p[ii], (&rand.x)[ii]);
        }
      }
    }
  }
};


void multi_tensor_lans_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  const float lr,
  const float beta1,
  const float beta2,
  const float epsilon,
  const int step,
  const int bias_correction,
  const float weight_decay,
  const int grad_averaging,
  const int mode,
  const bool normalize_grad,
  const bool stochastic_rounding)
{
  using namespace at;
  // Master weight and 32bit momentum(potentially changing) is not handled by this
  // So we assume every tensor are all in the same type

  // Handle bias correction mode
  float bias_correction1 = 1.0f, bias_correction2 = 1.0f;
  if (bias_correction == 1) {
    bias_correction1 = 1 - std::pow(beta1, step);
    bias_correction2 = 1 - std::pow(beta2, step);
  }

  // Handle grad averaging mode
  float beta3 = 1.0f;
  if (grad_averaging == 1) beta3 = 1 - beta1;

  std::vector<std::vector<at::Tensor>> grad_list(tensor_lists.begin(), tensor_lists.begin()+1);
  std::vector<std::vector<at::Tensor>> param_list(tensor_lists.begin()+2, tensor_lists.begin()+3);

  // Compute per-layer grad norm
  auto grad_norm_tuple = multi_tensor_l2norm_cuda(chunk_size, noop_flag, grad_list, true);

  // Compute per tensor param norm
  auto param_norm_tuple = multi_tensor_l2norm_cuda(chunk_size, noop_flag, param_list, true);

  // We now in-place modify grad to store update before compute its norm
  // Generally this is not a issue since people modify grad in step() method all the time
  // We can also grab list of empty tensor to avoid this, but I'd like to save space/cpu code
  DISPATCH_FLOAT_HALF_AND_BFLOAT(tensor_lists[0][0].scalar_type(), 0, "lans_stage_1",
      multi_tensor_apply<5>(
        BLOCK_SIZE,
        chunk_size,
        noop_flag,
        tensor_lists,
        LANSStage1Functor<scalar_t_0>(),
        beta1,
        beta2,
        beta3, // 1-beta1 or 1 depends on averaging mode
        bias_correction1,
        bias_correction2,
        epsilon,
        (adamMode_t) mode,
        weight_decay,
        std::get<1>(grad_norm_tuple).DATA_PTR<float>(),
        normalize_grad); )

  // Compute update norms
  auto update_m_norm_tuple = multi_tensor_l2norm_cuda(chunk_size, noop_flag, grad_list, true);

  std::vector<std::vector<at::Tensor>> q_list(tensor_lists.begin()+1, tensor_lists.begin()+2);
  auto update_g_norm_tuple = multi_tensor_l2norm_cuda(chunk_size, noop_flag, q_list, true);

  std::vector<std::vector<at::Tensor>> grad_q_param_list(tensor_lists.begin(), tensor_lists.begin()+3);

  if(stochastic_rounding)
  {
    auto gen = at::cuda::detail::getDefaultCUDAGenerator();
    at::PhiloxCudaState rng_engine_inputs;
    uint64_t counter_offset = (chunk_size - 1) / BLOCK_SIZE + 1;
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen.mutex());
      rng_engine_inputs = at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_cuda_state(counter_offset);
    }

    DISPATCH_FLOAT_HALF_AND_BFLOAT(tensor_lists[0][0].scalar_type(), 0, "lans_stage_2",
        multi_tensor_apply<3>(
          BLOCK_SIZE,
          chunk_size,
          noop_flag,
          grad_q_param_list,
          LANSStage2Functor<scalar_t_0>(),
          beta1,
          beta3,
          std::get<1>(param_norm_tuple).DATA_PTR<float>(),
          std::get<1>(update_m_norm_tuple).DATA_PTR<float>(),
          std::get<1>(update_g_norm_tuple).DATA_PTR<float>(),
          lr,
          rng_engine_inputs); )
  } else {
    DISPATCH_FLOAT_HALF_AND_BFLOAT(tensor_lists[0][0].scalar_type(), 0, "lans_stage_2",
        multi_tensor_apply<3>(
          BLOCK_SIZE,
          chunk_size,
          noop_flag,
          grad_q_param_list,
          LANSStage2Functor<scalar_t_0>(),
          beta1,
          beta3,
          std::get<1>(param_norm_tuple).DATA_PTR<float>(),
          std::get<1>(update_m_norm_tuple).DATA_PTR<float>(),
          std::get<1>(update_g_norm_tuple).DATA_PTR<float>(),
          lr); )
  }

  AT_CUDA_CHECK(cudaGetLastError());

}