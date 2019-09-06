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

template<typename T>
struct LAMBStage1Functor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<4>& tl,
    const float beta1,
    const float beta2,
    const float beta3,
    const float beta1_correction,
    const float beta2_correction,
    const float epsilon,
    adamMode_t mode,
    const float decay,
    float* global_grad_norm,
    float max_global_grad_norm)
  {
    // I'd like this kernel to propagate infs/nans.
    // if(*noop_gmem == 1)
    //   return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    float clipped_global_grad_norm = (*global_grad_norm) > max_global_grad_norm ? (*global_grad_norm) / max_global_grad_norm : 1.0f;

    T* g = (T*)tl.addresses[0][tensor_loc];
    g += chunk_idx*chunk_size;

    T* p = (T*)tl.addresses[1][tensor_loc];
    p += chunk_idx*chunk_size;

    T* m = (T*)tl.addresses[2][tensor_loc];
    m += chunk_idx*chunk_size;

    T* v = (T*)tl.addresses[3][tensor_loc];
    v += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

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
          MATH_T scaled_grad = r_g[ii] / clipped_global_grad_norm;
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
          MATH_T scaled_grad = r_g[ii] / clipped_global_grad_norm;
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
          g[i] = r_p[ii];
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
struct LAMBStage2Functor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<2>& tl,
    const float* per_tensor_param_norm,
    const float* per_tensor_update_norm,
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
    float update_norm = per_tensor_update_norm[tensor_num];
    MATH_T ratio = (update_norm != 0.0f && param_norm != 0.0f) ? learning_rate * (param_norm / update_norm) : learning_rate;

    T* update = (T*)tl.addresses[0][tensor_loc];
    update += chunk_idx*chunk_size;

    T* p = (T*)tl.addresses[1][tensor_loc];
    p += chunk_idx*chunk_size;

    n -= chunk_idx*chunk_size;

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
        }
      }
    }
  }
};


void multi_tensor_lamb_cuda(
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
  const float max_grad_norm)
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
  std::vector<std::vector<at::Tensor>> param_list(tensor_lists.begin()+1, tensor_lists.begin()+2);

  // Compute global grad norm
  auto grad_norm_tuple = multi_tensor_l2norm_cuda(chunk_size, noop_flag, grad_list, false);

  // Compute per tensor param norm
  auto param_norm_tuple = multi_tensor_l2norm_cuda(chunk_size, noop_flag, param_list, true);

  // We now in-place modify grad to store update before compute its norm
  // Generally this is not a issue since people modify grad in step() method all the time
  // We can also grab list of empty tensor to avoid this, but I'd like to save space/cpu code
  DISPATCH_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(), 0, "lamb_stage_1",
      multi_tensor_apply<4>(
        BLOCK_SIZE,
        chunk_size,
        noop_flag,
        tensor_lists,
        LAMBStage1Functor<scalar_t_0>(),
        beta1,
        beta2,
        beta3, // 1-beta1 or 1 depends on averaging mode
        bias_correction1,
        bias_correction2,
        epsilon,
        (adamMode_t) mode,
        weight_decay,
        std::get<0>(grad_norm_tuple).DATA_PTR<float>(),
        max_grad_norm); )

  // Compute update norms
  auto update_norm_tuple = multi_tensor_l2norm_cuda(chunk_size, noop_flag, grad_list, true);

  std::vector<std::vector<at::Tensor>> grad_param_list(tensor_lists.begin(), tensor_lists.begin()+2);

  DISPATCH_FLOAT_AND_HALF(tensor_lists[0][0].scalar_type(), 0, "lamb_stage_2",
      multi_tensor_apply<2>(
        BLOCK_SIZE,
        chunk_size,
       	noop_flag,
        grad_param_list,
        LAMBStage2Functor<scalar_t_0>(),
        std::get<1>(param_norm_tuple).DATA_PTR<float>(),
        std::get<1>(update_norm_tuple).DATA_PTR<float>(),
        lr); )

  AT_CUDA_CHECK(cudaGetLastError());

}
