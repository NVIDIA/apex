#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/Exception.h>
#include <nccl.h>
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>
#include <torch/extension.h>

#define NCCL_CHECK(cmd)                                                                                     \
  do {                                                                                                      \
    ncclResult_t result = cmd;                                                                              \
    if (result != ncclSuccess) {                                                                            \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + ", " + \
                        std::string(ncclGetErrorString(result));                                            \
      TORCH_CHECK(false, err);                                                                              \
    }                                                                                                       \
  } while (0)

void* nccl_alloc_plug(size_t size, int device, void* stream) {
  void* ptr;
  NCCL_CHECK(ncclMemAlloc(&ptr, size));
  return ptr;
}

void nccl_free_plug(void* ptr, std::size_t size, int device, void* stream) { NCCL_CHECK(ncclMemFree(ptr)); }

std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> nccl_allocator;

void maybe_init() {
  if (!nccl_allocator) {
    nccl_allocator =
        std::make_shared<torch::cuda::CUDAPluggableAllocator::CUDAPluggableAllocator>(nccl_alloc_plug, nccl_free_plug);
  }
}

std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> get_nccl_allocator() {
  maybe_init();
  return nccl_allocator;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_nccl_allocator", []() { return get_nccl_allocator(); });
};
