#include <torch/extension.h>

void multi_tensor_unscale_cuda(
  int nblocks,
  at::Tensor noop_flag,
  at::Tensor cpu_tensor_addresses,
  at::Tensor gpu_block_to_tensor,
  at::Tensor gpu_block_to_chunk,
  at::Tensor gpu_tensor_sizes,
  at::Tensor gpu_tensor_addresses,
  int chunk_size,
  std::vector<std::vector<at::Tensor>> tensor_lists,
  float scale);

std::vector<int> prep_multi_tensor_launch(
  at::Tensor cpu_block_to_tensor,
  at::Tensor cpu_block_to_chunk,
  at::Tensor cpu_tensor_sizes,
  at::Tensor gpu_block_to_tensor,
  at::Tensor gpu_block_to_chunk,
  at::Tensor gpu_tensor_sizes,
  int chunk_size,
  int max_depth,
  int max_tensors,
  int max_blocks,
  std::vector<std::vector<at::Tensor>> tensor_lists)
{
  int needs_reallocate = 0;

  if(tensor_lists.size() > max_depth || tensor_lists[0].size() > max_tensors)
    needs_reallocate = 1;

  auto cpu_tensor_sizes_a = cpu_tensor_sizes.accessor<int,1>();
  auto cpu_block_to_tensor_a = cpu_block_to_tensor.accessor<int,1>();
  auto cpu_block_to_chunk_a = cpu_block_to_chunk.accessor<int,1>();

  int nblocks = 0;
  for(int t = 0; t < tensor_lists[0].size(); t++)
  {
    int blocks_this_tensor = (tensor_lists[0][t].numel() + chunk_size - 1)/chunk_size;
    if(!needs_reallocate)
      cpu_tensor_sizes_a[t] = tensor_lists[0][t].numel();
    for(int chunk = 0; chunk < blocks_this_tensor; chunk++)
    {
      if(nblocks >= max_blocks)
        needs_reallocate = 1;
      if(!needs_reallocate)
      {
        cpu_block_to_tensor_a[nblocks] = t;
        cpu_block_to_chunk_a[nblocks] = chunk;
      }
      nblocks++;
    }
  }

  if(!needs_reallocate)
  {
    gpu_block_to_tensor.copy_(cpu_block_to_tensor, 1);
    gpu_block_to_chunk.copy_(cpu_block_to_chunk, 1);
    gpu_tensor_sizes.copy_(cpu_tensor_sizes, 1);
  }

  return std::vector<int>{needs_reallocate, nblocks};
}

void scale_check_overflow_cuda(const at::Tensor& grads,
                               float scale,
                               const at::Tensor& d_buf,
                               const at::Tensor& downscaled_grads);

void scale_check_overflow(at::Tensor grads,
                          float scale,
                          at::Tensor overflow_buf,
                          at::Tensor downscaled_grads)
                          // const at::optional<at::Tensor> downscaled_grads)
{ 
  AT_CHECK(grads.type().is_cuda(), "grads must be a CUDA tensor");
  AT_CHECK(grads.is_contiguous(), "grads must be contiguous");
  AT_CHECK(overflow_buf.type().is_cuda(), "overflow_buf must be a CUDA tensor");
  AT_CHECK(overflow_buf.is_contiguous(), "overflow_buf must be contiguous");
  AT_CHECK(downscaled_grads.type().is_cuda(), "downscaled_grads must be a CUDA tensor");
  AT_CHECK(downscaled_grads.is_contiguous(), "downscaled_grads must be contiguous");
  // Make sure we are downscaling the FP32 master grads
  AT_CHECK(downscaled_grads.type().scalarType() == at::ScalarType::Float,
    "The output grads supplied to scale_check_overflow should be fp32 (master grads).")
  AT_CHECK(grads.numel() == downscaled_grads.numel(), "Input and output grads must be the same size.");

  scale_check_overflow_cuda(grads, scale, overflow_buf, downscaled_grads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scale_check_overflow", &scale_check_overflow, "Fused overflow check + scale for FP32 tensors");
  m.def("prep_multi_tensor_launch", &prep_multi_tensor_launch, "Prepare multitensor launch");
  m.def("multi_tensor_unscale", &multi_tensor_unscale_cuda,
        "Fused overflow check + unscale for a list of contiguous tensors");
}
