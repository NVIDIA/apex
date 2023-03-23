#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#include "type_shim.h"
#include "compat.h"
#include "multi_tensor_apply.cuh"

#include <assert.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512
#define ILP 4

/**
 * Perform fused SGD on multiple buffers
 * N: number of tensors
 * tl[0] : gradients
 * tl[1] : weights
 * tl[2] : momentum buffers
 * tl[3] : fp16 weights (if appropriate)
 * wd : weight_decay (scalar)
 * momentum : momentum (scalar)
 * dampening : momentum dampening (scalar)
 * lr : learning rate (scalar)
 * nesterov : enable nesterov (bool)
 * first run : necessary for proper momentum handling & init
 * wd_after_momentum : apply weight decay _after_ momentum instead of before
 **/

template<int N, typename T_grad, typename T_weight>
struct LARSFunctor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorListMetadata<N>& tl,
    float *grad_norms,
    float *param_norms,
    float lr,
    float trust_coefficient,
    float epsilon,
    float weight_decay,
    float momentum,
    float dampening,
    bool nesterov,
    bool first_run,
    bool wd_after_momentum,
    float scale,
    const bool is_skipped) {
    
    // Early exit if we don't need to do anything
    if (*noop_gmem) return;
    	   
    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    n -= chunk_idx * chunk_size;
    //n = min(n, chunk_size);

    T_grad* grad_in = (T_grad*) tl.addresses[0][tensor_loc];
    grad_in += chunk_idx * chunk_size;

    T_weight* weight_in = (T_weight*) tl.addresses[1][tensor_loc];
    weight_in += chunk_idx * chunk_size;

    T_weight* mom_in = (T_weight*)tl.addresses[2][tensor_loc];
    mom_in += chunk_idx*chunk_size;

    at::Half *model_weights_out = nullptr;
    if(N == 4)
    {
      model_weights_out = (at::Half*)tl.addresses[3][tensor_loc];
      model_weights_out += chunk_idx*chunk_size;
    }

    float scaled_lr;
    if (is_skipped) {
      scaled_lr = lr;
    }
    else {
      int tensor_offset = tl.start_tensor_this_launch + tensor_loc;
      float p_norm = param_norms[tensor_offset];
      float trust_ratio = 1.0;
      float g_norm = grad_norms[tensor_offset];
      if (g_norm > 0.0f && p_norm > 0.0f) {
        trust_ratio = trust_coefficient * p_norm / (g_norm + p_norm * weight_decay + epsilon);
      }
      scaled_lr = lr * trust_ratio;
    }

    // Non-divergent exit condition for the __syncthreads
    float incoming_grads[ILP];
    float incoming_weights[ILP];
    float incoming_moms[ILP];
    for(int i_start = 0;
        i_start < n && i_start < chunk_size;
        i_start += blockDim.x*ILP)
    {
      #pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        incoming_grads[ii] = 0;
        incoming_weights[ii] = 0;
        incoming_moms[ii] = 0;
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          incoming_grads[ii] = static_cast<float>(grad_in[i]);
          incoming_weights[ii] = static_cast<float>(weight_in[i]);
          incoming_moms[ii] = static_cast<float>(mom_in[i]);
        }
      }

      // note for clarification to future michael:
      // From a pure memory dependency perspective, there's likely no point unrolling
      // the write loop, since writes just fire off once their LDGs arrive.
      // Put another way, the STGs are dependent on the LDGs, but not on each other.
      // There is still compute ILP benefit from unrolling the loop though.
      #pragma unroll
      for(int ii = 0; ii < ILP; ii++)
      {
        int i = i_start + threadIdx.x + ii*blockDim.x;
        if(i < n && i < chunk_size)
        {
          // apply weight decay before momentum
          incoming_grads[ii] += weight_decay * incoming_weights[ii];
          incoming_moms[ii] = incoming_moms[ii] * momentum - scaled_lr * incoming_grads[ii];

          // adjust the weight and write out
          if (nesterov) {
            incoming_weights[ii] += incoming_moms[ii] * momentum - scaled_lr * incoming_grads[ii];
          } else {
            incoming_weights[ii] += incoming_moms[ii];
          }

          weight_in[i] = static_cast<T_weight>(incoming_weights[ii]);
          
          // if necessary, write out an fp16 copy of the weights
          if(N == 4)
            model_weights_out[i] = static_cast<at::Half>(weight_in[i]);

          // also write out the new momentum
          //if(momentum != 0.f)
            mom_in[i] = static_cast<T_weight>(incoming_moms[ii]);
        }
      }
    }
  }
};

void multi_tensor_lars_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  at::Tensor grad_norms,
  at::Tensor param_norms,
  float lr,
  float trust_coefficient,
  float epsilon,
  float weight_decay,
  float momentum,
  float dampening,
  bool nesterov,
  bool first_run,
  bool wd_after_momentum,
  float scale,
  const bool is_skipped)
{
  auto num_tensors = tensor_lists.size();
  auto grad_type = tensor_lists[0][0].scalar_type();
  auto weight_type = tensor_lists[1][0].scalar_type();

  if(num_tensors == 4) {
    for(int i = 0; i < tensor_lists[3].size(); i++) {
        TORCH_CHECK(tensor_lists[3][i].scalar_type() == at::ScalarType::Half,
                 "Additional output tensors should always be fp16.");
    }
  }

  TORCH_CHECK(noop_flag.device() == tensor_lists[0][0].device(), "expected noop flag to be on the same device as tensors");

  // We have 3 possibilities to handle here, in terms of
  // grad_type, param_type, momentum_type, requires_fp16_copy
  // 1. fp16, fp16, fp16, No
  // 2. fp32, fp32, fp32, No
  // 3. fp16, fp32, fp32, Yes
  // 4. fp32, fp32, fp32, Yes // this is the materialize_master_grads=True case
  // 5. bfp16, bfp16, bfp16, No
  // 6. bfp16, fp32, fp32, Yes
  // It's easier to hardcode these possibilities than to use
  // switches etc. to handle the cross-product of cases where
  // we don't want the majority of them.

  // Case 1. fp16, fp16, fp16, No
  if(grad_type == at::ScalarType::Half &&
     weight_type == at::ScalarType::Half &&
     num_tensors == 3)
  {
    multi_tensor_apply<3>(
        BLOCK_SIZE,
        chunk_size,
        noop_flag,
        tensor_lists,
        LARSFunctor<3, at::Half, at::Half>(),
        grad_norms.DATA_PTR<float>(),
        param_norms.DATA_PTR<float>(),
        lr,
        trust_coefficient,
        epsilon,
        weight_decay,
        momentum,
        dampening,
        nesterov,
        first_run,
        wd_after_momentum,
        scale,
        is_skipped);
  }
  // Case 2. fp32, fp32, fp32, No
  else if(grad_type == at::ScalarType::Float &&
          weight_type == at::ScalarType::Float &&
          num_tensors == 3)
  {
    multi_tensor_apply<3>(
        BLOCK_SIZE,
        chunk_size,
        noop_flag,
        tensor_lists,
        LARSFunctor<3, float, float>(),
        grad_norms.DATA_PTR<float>(),
        param_norms.DATA_PTR<float>(),
        lr,
        trust_coefficient,
        epsilon,
        weight_decay,
        momentum,
        dampening,
        nesterov,
        first_run,
        wd_after_momentum,
        scale,
        is_skipped);
  }
  // Case 3. fp16, fp32, fp32, Yes
  else if(grad_type == at::ScalarType::Half &&
          weight_type == at::ScalarType::Float &&
          num_tensors == 4)
  {
    multi_tensor_apply<4>(
        BLOCK_SIZE,
        chunk_size,
        noop_flag,
        tensor_lists,
        LARSFunctor<4, at::Half, float>(),
        grad_norms.DATA_PTR<float>(),
        param_norms.DATA_PTR<float>(),
        lr,
        trust_coefficient,
        epsilon,
        weight_decay,
        momentum,
        dampening,
        nesterov,
        first_run,
        wd_after_momentum,
        scale,
        is_skipped);
  }
  // Case 4. fp32, fp32, fp32, Yes
  else if(grad_type == at::ScalarType::Float &&
          weight_type == at::ScalarType::Float &&
          num_tensors == 4)
  {
    multi_tensor_apply<4>(
        BLOCK_SIZE,
        chunk_size,
        noop_flag,
        tensor_lists,
        LARSFunctor<4, float, float>(),
        grad_norms.DATA_PTR<float>(),
        param_norms.DATA_PTR<float>(),
        lr,
        trust_coefficient,
        epsilon,
        weight_decay,
        momentum,
        dampening,
        nesterov,
        first_run,
        wd_after_momentum,
        scale,
        is_skipped);
  }
  // Case 5. bfp16, bfp16, bfp16, No
  else if(grad_type == at::ScalarType::BFloat16 &&
     weight_type == at::ScalarType::BFloat16 &&
     num_tensors == 3)
  {
    multi_tensor_apply<3>(
        BLOCK_SIZE,
        chunk_size,
        noop_flag,
        tensor_lists,
        LARSFunctor<3, at::BFloat16, at::BFloat16>(),
        grad_norms.DATA_PTR<float>(),
        param_norms.DATA_PTR<float>(),
        lr,
        trust_coefficient,
        epsilon,
        weight_decay,
        momentum,
        dampening,
        nesterov,
        first_run,
        wd_after_momentum,
        scale,
        is_skipped);
  }
  // Case 6. bfp16, fp32, fp32, Yes
  else if(grad_type == at::ScalarType::BFloat16 &&
          weight_type == at::ScalarType::Float &&
          num_tensors == 4)
  {
    multi_tensor_apply<4>(
        BLOCK_SIZE,
        chunk_size,
        noop_flag,
        tensor_lists,
        LARSFunctor<4, at::BFloat16, float>(),
        grad_norms.DATA_PTR<float>(),
        param_norms.DATA_PTR<float>(),
        lr,
        trust_coefficient,
        epsilon,
        weight_decay,
        momentum,
        dampening,
        nesterov,
        first_run,
        wd_after_momentum,
        scale,
        is_skipped);
  }
  else
  {
    AT_ERROR("multi_tensor_lars only supports some combinations of gradient & weight types. Given: ",
             "gradient: ", grad_type, ", weight: ", weight_type, ", num_lists: ", num_tensors);
  }

  AT_CUDA_CHECK(cudaGetLastError());
}
