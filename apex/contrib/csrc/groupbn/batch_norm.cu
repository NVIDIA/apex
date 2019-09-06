#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCNumerics.cuh>

#include "THC/THC.h"

#include "batch_norm.h"

#include <cuda.h>

#include "compat.h"

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
at::Tensor nhwc_bn_fwd_train(
                       const at::Tensor& x,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const at::Tensor& minibatch_mean,
                       const at::Tensor& minibatch_inv_var,
                       const at::Tensor& ret_cta,
                       const float momentum,
                       const float epsilon,
                       const bool fuse_relu,
                       void * my_data,
                       void * pair_data,
                       void * pair_data2,
                       void * pair_data3,
                       const int bn_group,
                       const at::Tensor& magic_tensor,
                       const int occupancy,
                       const int grid_dim_x,
                       const bool coop) {

  const int N = x.size(0);
  const int H = x.size(1);
  const int W = x.size(2);
  const int C = x.size(3);

  // generating new magic number and use that for sync
  int* magic = magic_tensor.DATA_PTR<int>();
  *magic = (*magic + 1) & 0xff;

  // Allocate output tensor
  at::Tensor y = at::empty({N, H, W, C}, x.options());

  // Create wrapper
  NhwcBatchNorm *bn = new NhwcBatchNorm();

  bn->setInputDescriptor(CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, N, C, H, W, bn_group);
  bn->setOutputDescriptor(CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, N, C, H, W);

  bn->setConstants(momentum, epsilon);

  // set pointers within the wrapper
  bn->setInputOutputPointers(x.DATA_PTR<at::Half>(),
                             nullptr,
                             y.DATA_PTR<at::Half>(),
                             nullptr);

  bn->setWeightPointers({scale.DATA_PTR<float>(), bias.DATA_PTR<float>()}, {nullptr, nullptr});
  bn->setParameterPointers({running_mean.DATA_PTR<float>(), running_inv_var.DATA_PTR<float>()});

  // deal with workspace(s)
  auto workspace_bytes = bn->numWorkspaceBytes();
  // We'll create explicit tensors for the first 2 workspace ptrs, then allocate & offset
  // an allocated workspace for the others
  size_t total_workspace_bytes = 0;
  std::vector<size_t> workspace_offsets;

  for (auto index = 3; index < workspace_bytes.size(); ++index) {
    total_workspace_bytes = round_up_to_multiple(total_workspace_bytes, 512);
    workspace_offsets.push_back(total_workspace_bytes);

    auto alloc_bytes = workspace_bytes[index];
    total_workspace_bytes += alloc_bytes;
  }

  // Allocate the workspace
  Workspace ws(total_workspace_bytes);

  std::vector<void *> workspace;
  workspace.push_back(minibatch_mean.DATA_PTR<float>());
  workspace.push_back(minibatch_inv_var.DATA_PTR<float>());

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int retired_cta_bytes = workspace_bytes[2];
  void* retired_ctas = ret_cta.DATA_PTR<uint8_t>();
  assert(ret_cta.size(0)>=retired_cta_bytes);
  workspace.push_back(retired_ctas);

  for (auto index = 3; index < workspace_bytes.size(); ++index) {
    void *ptr = reinterpret_cast<uint8_t*>(ws.data) + workspace_offsets[index-3];
    workspace.push_back(ptr);
  }

  bn->setWorkspacePointers(workspace, workspace_bytes);

  // Don't fuse in ReLU for now at least
  bn->fwd(stream, fuse_relu, my_data, pair_data, pair_data2, pair_data3, bn_group, *magic, occupancy, grid_dim_x, coop);

  return y;
}

at::Tensor nhwc_bn_fwd_eval(
                       const at::Tensor& x,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const at::Tensor& ret_cta,
                       const int bn_group,
                       const float momentum,
                       const float epsilon,
                       const bool fuse_relu) {

  const int N = x.size(0);
  const int H = x.size(1);
  const int W = x.size(2);
  const int C = x.size(3);

  // Allocate output tensor
  at::Tensor y = at::empty({N, H, W, C}, x.options());

  // Create wrapper
  NhwcBatchNorm *bn = new NhwcBatchNorm();

  bn->setInputDescriptor(CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, N, C, H, W, bn_group);
  bn->setOutputDescriptor(CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, N, C, H, W);

  bn->setConstants(momentum, epsilon);

  // set pointers within the wrapper
  bn->setInputOutputPointers(x.DATA_PTR<at::Half>(),
                             nullptr,
                             y.DATA_PTR<at::Half>(),
                             nullptr);

  bn->setWeightPointers({scale.DATA_PTR<float>(), bias.DATA_PTR<float>()}, {nullptr, nullptr});
  bn->setParameterPointers({running_mean.DATA_PTR<float>(), running_inv_var.DATA_PTR<float>()});

  // deal with workspace(s)
  auto workspace_bytes = bn->numWorkspaceBytes();
  // We'll create explicit tensors for the first 2 workspace ptrs, then allocate & offset
  // an allocated workspace for the others
  size_t total_workspace_bytes = 0;
  std::vector<size_t> workspace_offsets;

  for (auto index = 3; index < workspace_bytes.size(); ++index) {
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

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int retired_cta_bytes = workspace_bytes[2];
  void* retired_ctas = ret_cta.DATA_PTR<uint8_t>();
  assert(ret_cta.size(0)>=retired_cta_bytes);
  workspace.push_back(retired_ctas);

  for (auto index = 3; index < workspace_bytes.size(); ++index) {
    void *ptr = reinterpret_cast<uint8_t*>(ws.data) + workspace_offsets[index-3];
    workspace.push_back(ptr);
  }

  bn->setWorkspacePointers(workspace, workspace_bytes);

  // Don't fuse in ReLU for now at least
  bn->fwdInference(stream, fuse_relu);

  return y;

}

std::vector<at::Tensor> nhwc_bn_bwd(
                       const at::Tensor& x,
                       const at::Tensor& dy,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const at::Tensor& minibatch_mean,
                       const at::Tensor& minibatch_inv_var,
                       const at::Tensor& ret_cta,
                       const float momentum,
                       const float epsilon,
                       const bool fuse_relu,
                       void * my_data,
                       void * pair_data, 
                       void * pair_data2, 
                       void * pair_data3, 
                       const int bn_group,
                       const at::Tensor& magic_tensor,
                       const int occupancy,
                       const int grid_dim_x,
                       const bool coop) {
  // shape
  const int N = x.size(0);
  const int H = x.size(1);
  const int W = x.size(2);
  const int C = x.size(3);

  // generating new magic number and use that for sync
  int* magic = magic_tensor.DATA_PTR<int>();
  *magic = (*magic + 1) & 0xff;

  // outputs
  at::Tensor x_grad, scale_grad, bias_grad;

  // Allocate outputs
  x_grad = at::empty_like(x);
  scale_grad = at::empty_like(scale);
  bias_grad = at::empty_like(bias);

  // Create wrapper
  NhwcBatchNorm *bn = new NhwcBatchNorm();

  bn->setInputDescriptor(CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, N, C, H, W, bn_group);
  bn->setOutputDescriptor(CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, N, C, H, W);

  bn->setConstants(momentum, epsilon);

  // set pointers within the wrapper
  bn->setInputOutputPointers(x.DATA_PTR<at::Half>(),
                             x_grad.DATA_PTR<at::Half>(),
                             nullptr,
                             dy.DATA_PTR<at::Half>());

  bn->setWeightPointers({scale.DATA_PTR<float>(), bias.DATA_PTR<float>()}, {scale_grad.DATA_PTR<float>(), bias_grad.DATA_PTR<float>()});
  bn->setParameterPointers({running_mean.DATA_PTR<float>(), running_inv_var.DATA_PTR<float>()});

  // deal with workspace(s)
  auto workspace_bytes = bn->numWorkspaceBytes();
  // We'll create explicit tensors for the first 2 workspace ptrs, then allocate & offset
  // an allocated workspace for the others
  size_t total_workspace_bytes = 0;
  std::vector<size_t> workspace_offsets;

  for (auto index = 3; index < workspace_bytes.size(); ++index) {
    total_workspace_bytes = round_up_to_multiple(total_workspace_bytes, 512);
    workspace_offsets.push_back(total_workspace_bytes);

    auto alloc_bytes = workspace_bytes[index];
    total_workspace_bytes += alloc_bytes;
  }

  // Allocate the workspace
  Workspace ws(total_workspace_bytes);

  std::vector<void *> workspace;
  workspace.push_back(minibatch_mean.DATA_PTR<float>());
  workspace.push_back(minibatch_inv_var.DATA_PTR<float>());

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int retired_cta_bytes = workspace_bytes[2];
  void* retired_ctas = ret_cta.DATA_PTR<uint8_t>();
  assert(ret_cta.size(0)>=retired_cta_bytes);
  workspace.push_back(retired_ctas);

  for (auto index = 3; index < workspace_bytes.size(); ++index) {
    void *ptr = reinterpret_cast<uint8_t*>(ws.data) + workspace_offsets[index-3];
    workspace.push_back(ptr);
  }

  bn->setWorkspacePointers(workspace, workspace_bytes);

  bn->dgrad(stream, fuse_relu, my_data, pair_data, pair_data2, pair_data3, bn_group, *magic, occupancy, grid_dim_x, coop);

  return std::vector<at::Tensor>{x_grad, scale_grad, bias_grad};
}

int nhwc_bn_fwd_occupancy() {
    int device_id=-1;
    cudaGetDevice(&device_id);

    //max occupancy supported by the code is 2
    return NhwcBatchNorm::smem_driven_fwd_occupancy(device_id, 2);
}

int nhwc_bn_bwd_occupancy() {
    int device_id=-1;
    cudaGetDevice(&device_id);
    
    //max occupancy supported by the code is 2
    return NhwcBatchNorm::smem_driven_bwd_occupancy(device_id, 2);
}


