#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "batch_norm_add_relu.h"

#include <cuda.h>

#include "compat.h"

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

struct Workspace {
  Workspace(size_t size) : size(size), data(NULL) {
    auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
    dataPtr = allocator.allocate(size);
    data = dataPtr.get();
  }
  Workspace(const Workspace&) = delete;
  Workspace(Workspace&&) = default;
  Workspace& operator=(Workspace&&) = default;
  ~Workspace() = default;

  size_t size;
  void* data;
  c10::DataPtr dataPtr;
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
                       const at::Tensor& ret_cta,
                       const float momentum,
                       const float epsilon,
                       void * my_data,
                       void * pair_data,
                       void * pair_data2,
                       void * pair_data3,
                       const int bn_group,
                       const at::Tensor& magic_tensor,
                       const int occupancy,
                       const int grid_dim_x,
                       const bool coop) {

  auto memory_format = x.suggest_memory_format();
  const bool check_channels_last = x.is_contiguous(at::MemoryFormat::ChannelsLast);
  const int N = x.size(0);
  const int H = check_channels_last ? x.size(2) : x.size(1);
  const int W = check_channels_last ? x.size(3) : x.size(2);
  const int C = check_channels_last ? x.size(1) : x.size(3);

  // generating new magic number and use that for sync
  int* magic = magic_tensor.DATA_PTR<int>();
  *magic = (*magic + 1) & 0xff;

  // Allocate output tensor
  at::Tensor y = check_channels_last? at::empty({N, C, H, W}, x.options().memory_format(memory_format)) : at::empty({N, H, W, C}, x.options());

  // Create wrapper
  NhwcBatchNormAddRelu *bn = new NhwcBatchNormAddRelu();

  bn->setInputDescriptor(DNN_TENSOR_FORMAT, DNN_DATA_HALF, N, C, H, W, bn_group);
  bn->setOutputDescriptor(DNN_TENSOR_FORMAT, DNN_DATA_HALF, N, C, H, W);

  bn->setConstants(momentum, epsilon);

  // set pointers within the wrapper
  bn->setInputOutputPointers(x.contiguous(memory_format).DATA_PTR<at::Half>(),
                             nullptr,
                             y.contiguous(memory_format).DATA_PTR<at::Half>(),
                             nullptr,
                             z.contiguous(memory_format).DATA_PTR<at::Half>(),
                             nullptr);

  bn->setWeightPointers({scale.contiguous().DATA_PTR<float>(),
                         bias.contiguous().DATA_PTR<float>()}, {nullptr, nullptr});
  bn->setParameterPointers({running_mean.contiguous().DATA_PTR<float>(),
                            running_inv_var.contiguous().DATA_PTR<float>()});

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
  workspace.push_back(minibatch_mean.contiguous().DATA_PTR<float>());
  workspace.push_back(minibatch_inv_var.contiguous().DATA_PTR<float>());
  workspace.push_back(bitmask.contiguous().DATA_PTR<bitmask_pyt_t>());

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int retired_cta_bytes = workspace_bytes[3];
  void* retired_ctas = ret_cta.contiguous().DATA_PTR<uint8_t>();
  assert(ret_cta.size(0)>=retired_cta_bytes);

  workspace.push_back(retired_ctas);

  for (auto index = 4; index < workspace_bytes.size(); ++index) {
    void *ptr = reinterpret_cast<uint8_t*>(ws.data) + workspace_offsets[index-4];
    workspace.push_back(ptr);
  }

  bn->setWorkspacePointers(workspace, workspace_bytes);

  // Don't fuse in ReLU for now at least
  bn->fwd(stream, my_data, pair_data, pair_data2, pair_data3, bn_group, *magic, occupancy, grid_dim_x, coop);

  return y.contiguous(memory_format);
}

at::Tensor nhwc_bn_addrelu_fwd_eval(
                       const at::Tensor& x,
                       const at::Tensor& z,
                       const at::Tensor& scale,
                       const at::Tensor& bias,
                       const at::Tensor& running_mean,
                       const at::Tensor& running_inv_var,
                       const at::Tensor& ret_cta,
                       const int bn_group,
                       const float momentum,
                       const float epsilon) {

  auto memory_format = x.suggest_memory_format();
  const bool check_channels_last = x.is_contiguous(at::MemoryFormat::ChannelsLast);
  const int N = x.size(0);
  const int H = check_channels_last ? x.size(2) : x.size(1);
  const int W = check_channels_last ? x.size(3) : x.size(2);
  const int C = check_channels_last ? x.size(1) : x.size(3);

  // Allocate output tensor
  at::Tensor y = check_channels_last? at::empty({N, C, H, W}, x.options().memory_format(memory_format)): at::empty({N, H, W, C}, x.options());

  // Create wrapper
  NhwcBatchNormAddRelu *bn = new NhwcBatchNormAddRelu();

  bn->setInputDescriptor(DNN_TENSOR_FORMAT, DNN_DATA_HALF, N, C, H, W, bn_group);
  bn->setOutputDescriptor(DNN_TENSOR_FORMAT, DNN_DATA_HALF, N, C, H, W);

  bn->setConstants(momentum, epsilon);

  // set pointers within the wrapper
  bn->setInputOutputPointers(x.contiguous(memory_format).DATA_PTR<at::Half>(),
                             nullptr,
                             y.contiguous(memory_format).DATA_PTR<at::Half>(),
                             nullptr,
                             z.contiguous(memory_format).DATA_PTR<at::Half>(),
                             nullptr);

  bn->setWeightPointers({scale.contiguous().DATA_PTR<float>(),
                         bias.contiguous().DATA_PTR<float>()}, {nullptr, nullptr});
  bn->setParameterPointers({running_mean.contiguous().DATA_PTR<float>(),
                            running_inv_var.contiguous().DATA_PTR<float>()});

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

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int retired_cta_bytes = workspace_bytes[3];
  void* retired_ctas = ret_cta.contiguous().DATA_PTR<uint8_t>();
  assert(ret_cta.size(0)>=retired_cta_bytes);
  workspace.push_back(retired_ctas);

  for (auto index = 4; index < workspace_bytes.size(); ++index) {
    void *ptr = reinterpret_cast<uint8_t*>(ws.data) + workspace_offsets[index-4];
    workspace.push_back(ptr);
  }

  bn->setWorkspacePointers(workspace, workspace_bytes);

  // Don't fuse in ReLU for now at least
  bn->fwdInference(stream);

  return y.contiguous(memory_format);

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
                       const at::Tensor& ret_cta,
                       const float momentum,
                       const float epsilon,
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
  auto memory_format = x.suggest_memory_format();
  const bool check_channels_last = x.is_contiguous(at::MemoryFormat::ChannelsLast);
  const int N = x.size(0);
  const int H = check_channels_last ? x.size(2) : x.size(1);
  const int W = check_channels_last ? x.size(3) : x.size(2);
  const int C = check_channels_last ? x.size(1) : x.size(3);

  // generating new magic number and use that for sync
  int* magic = magic_tensor.DATA_PTR<int>();
  *magic = (*magic + 1) & 0xff;

  // outputs
  at::Tensor x_grad, z_grad, scale_grad, bias_grad;

  // Allocate outputs
  x_grad = check_channels_last ? at::empty({N, C, H, W}, dy.options().memory_format(memory_format)) : at::empty_like(x);
  z_grad = check_channels_last ? at::empty({N, C, H, W}, dy.options().memory_format(memory_format)) : at::empty_like(x);
  scale_grad = at::empty_like(scale);
  bias_grad = at::empty_like(bias);

  // Create wrapper
  NhwcBatchNormAddRelu *bn = new NhwcBatchNormAddRelu();

  bn->setInputDescriptor(DNN_TENSOR_FORMAT, DNN_DATA_HALF, N, C, H, W, bn_group);
  bn->setOutputDescriptor(DNN_TENSOR_FORMAT, DNN_DATA_HALF, N, C, H, W);

  bn->setConstants(momentum, epsilon);

  // set pointers within the wrapper
  bn->setInputOutputPointers(x.contiguous(memory_format).DATA_PTR<at::Half>(),
                             x_grad.contiguous(memory_format).DATA_PTR<at::Half>(),
                             nullptr,
                             dy.contiguous(memory_format).DATA_PTR<at::Half>(),
                             nullptr,
                             z_grad.contiguous(memory_format).DATA_PTR<at::Half>());

  bn->setWeightPointers({scale.contiguous().DATA_PTR<float>(),
                         bias.contiguous().DATA_PTR<float>()},
                         {scale_grad.DATA_PTR<float>(), bias_grad.DATA_PTR<float>()});
  bn->setParameterPointers({running_mean.contiguous().DATA_PTR<float>(),
                            running_inv_var.contiguous().DATA_PTR<float>()});

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
  workspace.push_back(minibatch_mean.contiguous().DATA_PTR<float>());
  workspace.push_back(minibatch_inv_var.contiguous().DATA_PTR<float>());
  workspace.push_back(bitmask.contiguous().DATA_PTR<bitmask_pyt_t>());

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const int retired_cta_bytes = workspace_bytes[3];
  void* retired_ctas = ret_cta.contiguous().DATA_PTR<uint8_t>();
  assert(ret_cta.size(0)>=retired_cta_bytes);
  workspace.push_back(retired_ctas);

  for (auto index = 4; index < workspace_bytes.size(); ++index) {
    void *ptr = reinterpret_cast<uint8_t*>(ws.data) + workspace_offsets[index-4];
    workspace.push_back(ptr);
  }

  bn->setWorkspacePointers(workspace, workspace_bytes);

  bn->dgrad(stream, my_data, pair_data, pair_data2, pair_data3, bn_group, *magic, occupancy, grid_dim_x, coop);

  return std::vector<at::Tensor>{x_grad.contiguous(memory_format), z_grad.contiguous(memory_format), scale_grad, bias_grad};
}

int nhwc_bn_addrelu_fwd_occupancy() {
    int device_id=-1;
    cudaGetDevice(&device_id);
    
    //max occupancy supported by the code is 2
    return NhwcBatchNormAddRelu::smem_driven_fwd_occupancy(device_id, 2);
}

int nhwc_bn_addrelu_bwd_occupancy() {
    int device_id=-1;
    cudaGetDevice(&device_id);

    //max occupancy supported by the code is 2
    return NhwcBatchNormAddRelu::smem_driven_bwd_occupancy(device_id, 2);
}

