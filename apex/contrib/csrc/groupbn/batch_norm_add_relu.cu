#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCNumerics.cuh>

#include "THC/THC.h"

#include "batch_norm_add_relu.h"

#include <cuda.h>

//FIXME move the common stuff to common h file
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

static size_t round_up_to_multiple(size_t x, int multiple) {
  return ((x + multiple - 1) / multiple) * multiple;
}

// TODO: Stop manually allocating CUDA memory; allocate an ATen byte
// tensor instead.
struct Workspace {
  Workspace(size_t size) : size(size), data(NULL) {
    data = THCudaMalloc(at::globalContext().lazyInitCUDA(), size);
  }
  Workspace(const Workspace&) = delete;
  Workspace(Workspace&&) = default;
  Workspace& operator=(Workspace&&) = default;
  ~Workspace() {
    if (data) {
      THCudaFree(at::globalContext().lazyInitCUDA(), data);
    }
  }

  size_t size;
  void* data;
};

// Return {y}
at::Tensor nhwc_bn_addrelu_fwd_train(
                       const at::Tensor& x,
                       const at::Tensor& z,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const at::Tensor& minibatch_mean,
                       const at::Tensor& minibatch_inv_var,
                       const at::Tensor& bitmask,
                       const float momentum,
                       const float epsilon,
                       void * my_data,
                       void * pair_data,
                       void * pair_data2,
                       const int bn_group,
                       const at::Tensor& magic_tensor,
                       const int max_cta_per_sm,
                       const int cta_launch_margin) {

  const int N = x.size(0);
  const int H = x.size(1);
  const int W = x.size(2);
  const int C = x.size(3);

  // generating new magic number and use that for sync
  int* magic = magic_tensor.data<int>();
  *magic = (*magic + 1) & 0xff;

  // Allocate output tensor
  at::Tensor y = at::empty({N, H, W, C}, x.options());

  // Create wrapper
  NhwcBatchNormAddRelu *bn = new NhwcBatchNormAddRelu();

  bn->setInputDescriptor(CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, N, C, H, W, bn_group);
  bn->setOutputDescriptor(CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, N, C, H, W);

  bn->setConstants(momentum, epsilon);

  // set pointers within the wrapper
  bn->setInputOutputPointers(x.data<at::Half>(),
                             nullptr,
                             y.data<at::Half>(),
                             nullptr,
                             z.data<at::Half>(),
                             nullptr);

  bn->setWeightPointers({scale.data<float>(), bias.data<float>()}, {nullptr, nullptr});
  bn->setParameterPointers({running_mean.data<float>(), running_inv_var.data<float>()});

  // deal with workspace(s)
  auto workspace_bytes = bn->numWorkspaceBytes();
  // We'll create explicit tensors for the first 2 workspace ptrs, then allocate & offset
  // an allocated workspace for the others
  size_t total_workspace_bytes = 0;
  std::vector<size_t> workspace_offsets;

  for (auto index = 4; index < workspace_bytes.size(); ++index) {
    total_workspace_bytes = round_up_to_multiple(total_workspace_bytes, 512);
    workspace_offsets.push_back(total_workspace_bytes);

    auto alloc_bytes = workspace_bytes[index];
    total_workspace_bytes += alloc_bytes;
  }

  // Allocate the workspace
  Workspace ws(total_workspace_bytes);

  std::vector<void *> workspace;
  workspace.push_back(minibatch_mean.data<float>());
  workspace.push_back(minibatch_inv_var.data<float>());
  workspace.push_back(bitmask.data<int32_t>());

  const int retired_cta_bytes = workspace_bytes[3];
  void* retired_ctas = THCudaMalloc(at::globalContext().lazyInitCUDA(), retired_cta_bytes); 
  cudaMemset(retired_ctas, 0, retired_cta_bytes); //FIXME: is this legit?
  workspace.push_back(retired_ctas);

  for (auto index = 4; index < workspace_bytes.size(); ++index) {
    void *ptr = reinterpret_cast<uint8_t*>(ws.data) + workspace_offsets[index-4];
    workspace.push_back(ptr);
  }

  bn->setWorkspacePointers(workspace, workspace_bytes);

  int device_id = 0; //FIXME when resource query is replaced with real functions
  // Don't fuse in ReLU for now at least
  bn->fwd(at::cuda::getCurrentCUDAStream().stream(), device_id, my_data, pair_data, pair_data2, bn_group, *magic, max_cta_per_sm, cta_launch_margin);

  THCudaFree(at::globalContext().lazyInitCUDA(), retired_ctas);
  return y;
}

at::Tensor nhwc_bn_addrelu_fwd_eval(
                       const at::Tensor& x,
                       const at::Tensor& z,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const int bn_group,
                       const float momentum,
                       const float epsilon) {

  const int N = x.size(0);
  const int H = x.size(1);
  const int W = x.size(2);
  const int C = x.size(3);

  // Allocate output tensor
  at::Tensor y = at::empty({N, H, W, C}, x.options());

  // Create wrapper
  NhwcBatchNormAddRelu *bn = new NhwcBatchNormAddRelu();

  bn->setInputDescriptor(CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, N, C, H, W, bn_group);
  bn->setOutputDescriptor(CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, N, C, H, W);

  bn->setConstants(momentum, epsilon);

  // set pointers within the wrapper
  bn->setInputOutputPointers(x.data<at::Half>(),
                             nullptr,
                             y.data<at::Half>(),
                             nullptr,
                             z.data<at::Half>(),
                             nullptr);

  bn->setWeightPointers({scale.data<float>(), bias.data<float>()}, {nullptr, nullptr});
  bn->setParameterPointers({running_mean.data<float>(), running_inv_var.data<float>()});

  // deal with workspace(s)
  auto workspace_bytes = bn->numWorkspaceBytes();
  // We'll create explicit tensors for the first 2 workspace ptrs, then allocate & offset
  // an allocated workspace for the others
  size_t total_workspace_bytes = 0;
  std::vector<size_t> workspace_offsets;

  for (auto index = 4; index < workspace_bytes.size(); ++index) {
    total_workspace_bytes = round_up_to_multiple(total_workspace_bytes, 512);
    workspace_offsets.push_back(total_workspace_bytes);

    auto alloc_bytes = workspace_bytes[index];
    total_workspace_bytes += alloc_bytes;
  }

  // Allocate the workspace
  Workspace ws(total_workspace_bytes);

  std::vector<void *> workspace;
  workspace.push_back(nullptr);
  workspace.push_back(nullptr);
  workspace.push_back(nullptr);

  const int retired_cta_bytes = workspace_bytes[3];
  void* retired_ctas = THCudaMalloc(at::globalContext().lazyInitCUDA(), retired_cta_bytes);
  cudaMemset(retired_ctas, 0, retired_cta_bytes); //FIXME: is this legit?
  workspace.push_back(retired_ctas);

  for (auto index = 4; index < workspace_bytes.size(); ++index) {
    void *ptr = reinterpret_cast<uint8_t*>(ws.data) + workspace_offsets[index-4];
    workspace.push_back(ptr);
  }

  bn->setWorkspacePointers(workspace, workspace_bytes);

  // Don't fuse in ReLU for now at least
  bn->fwdInference(at::cuda::getCurrentCUDAStream().stream());

  THCudaFree(at::globalContext().lazyInitCUDA(), retired_ctas);
  return y;

}

std::vector<at::Tensor> nhwc_bn_addrelu_bwd(
                       const at::Tensor& x,
                       const at::Tensor& dy,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const at::Tensor& minibatch_mean,
                       const at::Tensor& minibatch_inv_var,
                       const at::Tensor& bitmask,
                       const float momentum,
                       const float epsilon,
                       void * my_data,
                       void * pair_data, 
                       void * pair_data2, 
                       const int bn_group,
                       const at::Tensor& magic_tensor,
                       const int max_cta_per_sm,
                       const int cta_launch_margin) {
  // shape
  const int N = x.size(0);
  const int H = x.size(1);
  const int W = x.size(2);
  const int C = x.size(3);

  // generating new magic number and use that for sync
  int* magic = magic_tensor.data<int>();
  *magic = (*magic + 1) & 0xff;

  // outputs
  at::Tensor x_grad, z_grad, scale_grad, bias_grad;

  // Allocate outputs
  x_grad = at::empty_like(x);
  z_grad = at::empty_like(x);
  scale_grad = at::empty_like(scale);
  bias_grad = at::empty_like(bias);

  // Create wrapper
  NhwcBatchNormAddRelu *bn = new NhwcBatchNormAddRelu();

  bn->setInputDescriptor(CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, N, C, H, W, bn_group);
  bn->setOutputDescriptor(CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, N, C, H, W);

  bn->setConstants(momentum, epsilon);

  // set pointers within the wrapper
  bn->setInputOutputPointers(x.data<at::Half>(),
                             x_grad.data<at::Half>(),
                             nullptr,
                             dy.data<at::Half>(),
                             nullptr,
                             z_grad.data<at::Half>());

  bn->setWeightPointers({scale.data<float>(), bias.data<float>()}, {scale_grad.data<float>(), bias_grad.data<float>()});
  bn->setParameterPointers({running_mean.data<float>(), running_inv_var.data<float>()});

  // deal with workspace(s)
  auto workspace_bytes = bn->numWorkspaceBytes();
  // We'll create explicit tensors for the first 2 workspace ptrs, then allocate & offset
  // an allocated workspace for the others
  size_t total_workspace_bytes = 0;
  std::vector<size_t> workspace_offsets;

  for (auto index = 4; index < workspace_bytes.size(); ++index) {
    total_workspace_bytes = round_up_to_multiple(total_workspace_bytes, 512);
    workspace_offsets.push_back(total_workspace_bytes);

    auto alloc_bytes = workspace_bytes[index];
    total_workspace_bytes += alloc_bytes;
  }

  // Allocate the workspace
  Workspace ws(total_workspace_bytes);

  std::vector<void *> workspace;
  workspace.push_back(minibatch_mean.data<float>());
  workspace.push_back(minibatch_inv_var.data<float>());
  workspace.push_back(bitmask.data<int32_t>());

  const int retired_cta_bytes = workspace_bytes[3];
  void* retired_ctas = THCudaMalloc(at::globalContext().lazyInitCUDA(), retired_cta_bytes);
  cudaMemset(retired_ctas, 0, retired_cta_bytes); //FIXME: is this legit?
  workspace.push_back(retired_ctas);

  for (auto index = 4; index < workspace_bytes.size(); ++index) {
    void *ptr = reinterpret_cast<uint8_t*>(ws.data) + workspace_offsets[index-4];
    workspace.push_back(ptr);
  }

  bn->setWorkspacePointers(workspace, workspace_bytes);

  int device_id = 0; //FIXME when resource query is replaced with real functions
  bn->dgrad(at::cuda::getCurrentCUDAStream().stream(), device_id, my_data, pair_data, pair_data2, bn_group, *magic, max_cta_per_sm, cta_launch_margin);

  THCudaFree(at::globalContext().lazyInitCUDA(), retired_ctas);
  return std::vector<at::Tensor>{x_grad, z_grad, scale_grad, bias_grad};
}
