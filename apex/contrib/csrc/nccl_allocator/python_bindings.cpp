#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/extension.h>
#include "NCCLAllocator.h"

namespace nccl_allocator::cuda {

int device_count = 1;
thread_local bool _use_nccl_mem;
std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> current_custom_allocator;

std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> getCurrentAllocator() {
  return current_custom_allocator;
}

void use_nccl_mem(bool flag) {
  _use_nccl_mem = flag;
}

void changeCurrentAllocator(std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> allocator) {
  TORCH_CHECK(
      !c10::cuda::CUDACachingAllocator::allocator.load()->initialized(),
      "Can't swap an already initialized allocator");
  c10::cuda::CUDACachingAllocator::allocator.store(allocator.get());
  current_custom_allocator = allocator;
}

void custom_raw_deleter(void* ptr) {
  current_custom_allocator->raw_delete(ptr);
}

} // namespace nccl_allocator::cuda

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_use_nccl_mem", [](bool flag) { nccl_allocator::cuda::use_nccl_mem(flag); });
  m.def(
      "_cuda_change_current_allocator",
      [](std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
             allocator) { nccl_allocator::cuda::changeCurrentAllocator(allocator); });
  py::class_<
      nccl_allocator::cuda::NCCLAllocator,
      c10::cuda::CUDACachingAllocator::CUDAAllocator,
      std::shared_ptr<nccl_allocator::cuda::NCCLAllocator>>(
      m, "_NCCLAllocator");
  m.def("_cuda_create_managed_allocator", []() {
    return nccl_allocator::cuda::createNCCLAllocator();
  });
};
