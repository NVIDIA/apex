#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
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
 **/
template<int N, typename T_grad, typename T_weight>
struct SGDFunctor
{
   __device__ __forceinline__ void operator()(
    int chunk_size,
    volatile int* noop_gmem,
    TensorList<N>& tl,
    float wd,
    float momentum,
    float dampening,
    float lr,
    bool nesterov,
    bool first_run)
  {
    __shared__ int noop_smem;

    if(threadIdx.x == 0)
      noop_smem = *noop_gmem;
    __syncthreads();
    if(noop_smem == 1)
      return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    T_grad* grad_in = (T_grad*)tl.addresses[0][tensor_loc];
    grad_in += chunk_idx*chunk_size;
   
    T_weight* weight_in = (T_weight*)tl.addresses[1][tensor_loc];
    weight_in += chunk_idx*chunk_size;

    T_weight* mom_in = (T_weight*)tl.addresses[2][tensor_loc];
    mom_in += chunk_idx*chunk_size;

    at::Half *model_weights_out = nullptr;
    if (N == 4) {
      model_weights_out = (at::Half*)tl.addresses[3][tensor_loc];
      model_weights_out += chunk_idx*chunk_size;
    }

    n -= chunk_idx*chunk_size;

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
          incoming_grads[ii] = static_cast<float>(grad_in[i]);
          incoming_weights[ii] = static_cast<float>(weight_in[i]);
          incoming_moms[ii] = static_cast<float>(mom_in[i]);
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
        if(i < n && i < chunk_size) {
          // apply weight decay
          if (wd != 0.f) {
            incoming_grads[ii] += wd * incoming_weights[ii];
          }
          if (momentum != 0.f) {
            if (!first_run) {
              incoming_moms[ii] = incoming_moms[ii] * momentum + (1.f - dampening) * incoming_grads[ii];
            }

            if (nesterov) {
              incoming_grads[ii] += momentum * incoming_moms[ii];
            }
          }

          // adjust the weight and write out
          weight_in[i] += (-lr * incoming_grads[ii]);

          // if necessary, write out an fp16 copy of the weights
          if (N == 4) {
            model_weights_out[i] = static_cast<at::Half>(weight_in[i]);
          }

          // also write out the new momentum
          if (momentum != 0.f) {
            mom_in[i] = incoming_moms[ii];
          }
        }
      }

      // *noop_gmem = 1 is NOT guaranteed to be seen immediately by thread 0.  I wonder if
      // we can rig block-wide and grid-wide short-circuiting with only one syncthreads.
      // It's possible we can just lean on the cache (no smem or syncs) and still be fast.
      if(threadIdx.x == 0)
        noop_smem = *noop_gmem;
      __syncthreads();
      if(noop_smem == 1)
        break;
    }
  }
};

void multi_tensor_sgd_cuda(
  int chunk_size,
  at::Tensor noop_flag,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  float wd,
  float momentum,
  float dampening,
  float lr,
  bool nesterov,
  bool first_run)
{
  auto num_tensors = tensor_lists.size();
  auto grad_type = tensor_lists[0][0].type().scalarType();
  auto weight_type = tensor_lists[0][0].type().scalarType();

  // We have 4 potentials to handle here, in terms of
  // grad_type, param_type, momentum_type, requires_fp16_copy
  // 1. fp16, fp16, fp16, No
  // 2. fp16, fp32, fp32, No
  // 3. fp16, fp32, fp32, Yes
  // 4. fp32, fp32, fp32, No
  // It's easier to hardcode these possibilities than to use
  // switches etc. to handle the cross-product of cases where
  // we don't want the majority of them.

  // Case 1. fp16, fp16, fp16, No
  if (grad_type == at::ScalarType::Half &&
      weight_type == at::ScalarType::Half &&
      num_tensors == 3) {
    multi_tensor_apply<3>(
        BLOCK_SIZE,
        chunk_size,
        noop_flag,
        tensor_lists,
        SGDFunctor<3, at::Half, at::Half>(),
        wd,
        momentum,
        dampening,
        lr,
        nesterov,
        first_run);
  }
  // Case 2. fp16, fp32, fp32, No
  else if (grad_type == at::ScalarType::Half &&
           weight_type == at::ScalarType::Float &&
           num_tensors == 3) {
    multi_tensor_apply<3>(
        BLOCK_SIZE,
        chunk_size,
        noop_flag,
        tensor_lists,
        SGDFunctor<3, at::Half, float>(),
        wd,
        momentum,
        dampening,
        lr,
        nesterov,
        first_run);
  }
  // Case 3. fp16, fp32, fp32, Yes
  else if (grad_type == at::ScalarType::Half &&
           weight_type == at::ScalarType::Float &&
           num_tensors == 4) {
    multi_tensor_apply<4>(
        BLOCK_SIZE,
        chunk_size,
        noop_flag,
        tensor_lists,
        SGDFunctor<4, at::Half, float>(),
        wd,
        momentum,
        dampening,
        lr,
        nesterov,
        first_run);
  }
  // Case 4. fp32, fp32, fp32, No
  else if (grad_type == at::ScalarType::Float &&
      weight_type == at::ScalarType::Float &&
      num_tensors == 3) {
    multi_tensor_apply<3>(
        BLOCK_SIZE,
        chunk_size,
        noop_flag,
        tensor_lists,
        SGDFunctor<3, float, float>(),
        wd,
        momentum,
        dampening,
        lr,
        nesterov,
        first_run);
  }
  else {
    AT_ERROR("multi_tensor_sgd only supports some combinations of gradient & weight types. Given: ",
             "gradient: ", grad_type, ", weight: ", weight_type, ", num_lists: ", num_tensors);
  }

  AT_CUDA_CHECK(cudaGetLastError());
}
