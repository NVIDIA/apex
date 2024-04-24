#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/ApproximateClock.h>
#include <iostream>
#include <mutex>
#include <nccl.h>

#include "NCCLAllocator.h"

#if defined(NCCL_MAJOR) && \
    ((NCCL_MAJOR > 2) || (NCCL_MAJOR == 2) && (NCCL_MINOR >= 19))
#define NCCL_SUPPORTS_UB
#endif

#define C10_NCCL_CHECK(cmd)                                               \
  do {                                                                    \
    ncclResult_t result = cmd;                                            \
    if (result != ncclSuccess) {                                          \
      std::string err = "NCCL error in: " + std::string(__FILE__) + ":" + \
          std::to_string(__LINE__) + ", " +                               \
          std::string(ncclGetErrorString(result));                        \
      TORCH_CHECK(false, err);                                            \
    }                                                                     \
  } while (0)

namespace nccl_allocator::cuda {

extern int device_count;

extern thread_local bool _use_nccl_mem;

std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
createNCCLAllocator() {
  auto allocator = std::make_shared<NCCLAllocator>();
  allocator->init(device_count);
  return allocator;
}

void custom_raw_deleter(void* ptr);

_AllocationMetadata::_AllocationMetadata()
    : size(0), device_idx(-1), stream{}, ptr(nullptr), is_free(false) {}

_AllocationMetadata::_AllocationMetadata(
    size_t size,
    c10::DeviceIndex device_idx,
    cudaStream_t stream,
    void* ptr)
    : size(size), device_idx(device_idx), stream(stream), ptr(ptr), is_free(false) {}

NCCLAllocator::NCCLAllocator(NCCLAllocator& other)
    : init_fn_(other.init_fn_),
      memory_fraction_fn_(other.memory_fraction_fn_),
      base_alloc_fn_(other.base_alloc_fn_),
      record_stream_fn_(other.record_stream_fn_),
      begin_allocate_to_pool_fn_(other.begin_allocate_to_pool_fn_),
      end_allocate_to_pool_fn_(other.end_allocate_to_pool_fn_),
      relase_pool_fn_(other.relase_pool_fn_) {}

void* NCCLAllocator::malloc(size_t size, c10::DeviceIndex device, cudaStream_t stream) {
  void* r = nullptr;
#ifdef NCCL_SUPPORTS_UB
  if (_use_nccl_mem) {
    for (
      auto md = nccl_allocation_metadata_.begin();
           md != nccl_allocation_metadata_.end();
	   md++
    ) {
      // compare stream?
      if(md->second.is_free && md->second.size == size && md->second.device_idx == device) {
        md->second.is_free = false;
	r = md->second.ptr;
	return r;
      }
    }

    C10_NCCL_CHECK(ncclMemAlloc(&r, size));
    {
      const std::lock_guard<std::mutex> lock(allocator_mutex_);
      nccl_allocation_metadata_.emplace(r, _AllocationMetadata(size, device, stream, r));

      auto te = c10::cuda::CUDACachingAllocator::TraceEntry(
        c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_ALLOC,
        device,
        (int64_t)r,
        size,
        stream,
        c10::getApproximateTime(),
        nullptr);
      for (const auto& cb : trace_trackers_) {
        cb(te);
      }
    }
  }
#else
  if (_use_nccl_mem) {
    TORCH_WARN_ONCE("ncclMemAlloc is not supported. ",
                    "NCCL v2.19 and newer is required,",
                    "but found NCCL v", NCCL_MAJOR, ".", NCCL_MINOR, ". ",
                    "Falling back to cudaMalloc.");
  }
  if (false) {
    // goto cudaMalloc
  }
#endif
  else {
    if (r == nullptr) {
      // find allocation of the same size or slightly better
      // What is the best upper bound?
      auto start = cuda_rt_cache_metadata_.lower_bound(size);
      auto end = cuda_rt_cache_metadata_.upper_bound(size + 1024);
      for (auto md = start; md != end; ) {
        _AllocationMetadata metadata = md->second;
        // Should we check for stream?
        if (metadata.device_idx == device) {
          r = metadata.ptr;
          cuda_rt_cache_metadata_.erase(md++);
          break;
        }
        ++md;
      }
    }
    if (r == nullptr) {
      cudaError_t err = cudaMalloc(&r, size);
      if (cudaSuccess != err) {
        size_t freed = 0;
        for (auto md = cuda_rt_cache_metadata_.rbegin(); md != cuda_rt_cache_metadata_.rend(); ++md) {
          _AllocationMetadata metadata = md->second;
          freed = freed + metadata.size;
          cuda_rt_cache_metadata_.erase(--(md.base()));
	  C10_CUDA_CHECK(cudaFree(metadata.ptr));
          if (freed >= size) {
            break;
          }
        }
        C10_CUDA_CHECK(cudaMalloc(&r, size));
      }
    }
    {
      const std::lock_guard<std::mutex> lock(allocator_mutex_);
      cuda_rt_allocation_metadata_.emplace(r, _AllocationMetadata(size, device, stream, r));
    }
  }
  return r;
}

c10::DataPtr NCCLAllocator::allocate(size_t size) {
  c10::DeviceIndex device = -1;
  C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
  cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(static_cast<int>(device));
  void* r = this->malloc(size, device, stream);
  c10::DataPtr data_ptr = {
      r,
      r,
      raw_deleter(),
      c10::Device(
          c10::DeviceType::CUDA, device)};
  return data_ptr;
}

c10::DeleterFnPtr NCCLAllocator::raw_deleter() const {
  return &custom_raw_deleter;
}

void* NCCLAllocator::raw_alloc(size_t nbytes) {
  c10::DeviceIndex device = -1;
  C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
  cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(device);
  return malloc(nbytes, device, stream);
}

void* NCCLAllocator::raw_alloc_with_stream(
    size_t nbytes,
    cudaStream_t stream) {
  c10::DeviceIndex device = -1;
  C10_CUDA_CHECK(c10::cuda::GetDevice(&device));
  return malloc(nbytes, device, stream);
}

void NCCLAllocator::raw_delete(void* ptr) {
  cudaStream_t stream{};
  c10::DeviceIndex device_idx = -1;
  size_t size = 0;
  {
    const std::lock_guard<std::mutex> lock(allocator_mutex_);
    if (cuda_rt_allocation_metadata_.count(ptr)) {
      _AllocationMetadata& metadata = cuda_rt_allocation_metadata_[ptr];
      size = metadata.size;
      device_idx = metadata.device_idx;
      stream = metadata.stream;
      cuda_rt_allocation_metadata_.erase(ptr);
      cuda_rt_cache_metadata_.emplace(size, metadata);
    } else if (nccl_allocation_metadata_.count(ptr)) {
      nccl_allocation_metadata_[ptr].is_free = true;
    } else {
      TORCH_CHECK(false, "Trying to free a pointer not allocated here");
    }
  }
}

void NCCLAllocator::init(int device_count) {
  if (init_fn_) {
    init_fn_(device_count);
  }
  initialized_ = true;
}

bool NCCLAllocator::initialized() {
  return initialized_;
}

void NCCLAllocator::setMemoryFraction(double fraction, c10::DeviceIndex device) {
  if (memory_fraction_fn_) {
    memory_fraction_fn_(fraction, device);
  }
}

void NCCLAllocator::emptyCache() {
  void *ptr;
  cudaStream_t stream{};
  int device_idx = -1;
  size_t size = 0;
  for (
    auto md = cuda_rt_cache_metadata_.cbegin();
         md != cuda_rt_cache_metadata_.cend();
  ) {
    C10_CUDA_CHECK(cudaFree(md->second.ptr));
    cuda_rt_cache_metadata_.erase(md++);
  }
  for (
    auto md = nccl_allocation_metadata_.cbegin();
         md != nccl_allocation_metadata_.cend();
  ) {
    if (md->second.is_free) {
      ptr = md->second.ptr;
      size = md->second.size;
      device_idx = md->second.device_idx;
      stream = md->second.stream;
      nccl_allocation_metadata_.erase(md++);
      auto te = c10::cuda::CUDACachingAllocator::TraceEntry(
        c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_FREE,
        device_idx,
        (int64_t)ptr,
        size,
        stream,
        c10::getApproximateTime(),
        nullptr);
      for (const auto& cb : trace_trackers_) {
        cb(te);
      }
      C10_NCCL_CHECK(ncclMemFree(ptr));
    }
    else {
      ++md;
    }
  }
}

void NCCLAllocator::cacheInfo(c10::DeviceIndex dev_id, size_t* largestBlock) {
  TORCH_CHECK(
      false,
      "NCCLAllocator does not yet support cacheInfo. "
      "If you need it, please file an issue describing your use case.");
}

void* NCCLAllocator::getBaseAllocation(void* ptr, size_t* size) {
  if (base_alloc_fn_) {
    return base_alloc_fn_(ptr, size);
  } else {
    return ptr;
  }
}

void NCCLAllocator::recordStream(
    const c10::DataPtr& ptr,
    c10::cuda::CUDAStream stream) {
  if (record_stream_fn_) {
    record_stream_fn_(ptr.get(), stream);
  }
}

c10::cuda::CUDACachingAllocator::DeviceStats NCCLAllocator::getDeviceStats(
    c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "NCCLAllocator does not yet support getDeviceStats. "
      "If you need it, please file an issue describing your use case.");
}

void NCCLAllocator::resetAccumulatedStats(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "NCCLAllocator does not yet support resetAccumulatedStats. "
      "If you need it, please file an issue describing your use case.");
}

void NCCLAllocator::resetPeakStats(c10::DeviceIndex device) {
  TORCH_CHECK(
      false,
      "NCCLAllocator does not yet support resetPeakStats. "
      "If you need it, please file an issue describing your use case.");
}

c10::cuda::CUDACachingAllocator::SnapshotInfo NCCLAllocator::snapshot() {
  c10::cuda::CUDACachingAllocator::SnapshotInfo result;
  result.segments.reserve(nccl_allocation_metadata_.size());
  for (auto& da : nccl_allocation_metadata_) {
    c10::cuda::CUDACachingAllocator::SegmentInfo seg;

    seg.device = da.second.device_idx;
    seg.address = (int64_t)da.first;
    seg.total_size = da.second.size;
    seg.requested_size = da.second.size;
    seg.allocated_size = da.second.size;
    seg.active_size = da.second.size;
    seg.stream = da.second.stream;
    seg.is_large = true;
    seg.is_expandable = false;

    result.segments.push_back(seg);
  }
  return result;
}

std::shared_ptr<void> NCCLAllocator::getIpcDevPtr(std::string handle) {
  TORCH_CHECK(
      false,
      "NCCLAllocator does not yet support getIpcDevPtr. "
      "If you need it, please file an issue describing your use case.");
}

// CUDAGraph interactions
void NCCLAllocator::beginAllocateToPool(
    c10::DeviceIndex device,
    c10::cuda::MempoolId_t mempool_id,
    std::function<bool(cudaStream_t)> filter) {
  if (begin_allocate_to_pool_fn_) {
    begin_allocate_to_pool_fn_(device, mempool_id, std::move(filter));
  }
}

void NCCLAllocator::endAllocateToPool(
    c10::DeviceIndex device,
    c10::cuda::MempoolId_t mempool_id) {
  if (end_allocate_to_pool_fn_) {
    end_allocate_to_pool_fn_(device, mempool_id);
  }
}

void NCCLAllocator::releasePool(
    c10::DeviceIndex device,
    c10::cuda::MempoolId_t mempool_id) {
  if (relase_pool_fn_) {
    relase_pool_fn_(device, mempool_id);
  }
}

void NCCLAllocator::recordHistory(
    bool enabled,
    c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    c10::cuda::CUDACachingAllocator::RecordContext when) {
  TORCH_CHECK(
      false,
      "NCCLAllocator does not yet support recordHistory. "
      "If you need it, please file an issue describing your use case.");
}

void NCCLAllocator::attachOutOfMemoryObserver(
    c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) {
  TORCH_CHECK(
      false,
      "NCCLAllocator does not yet support attachOutOfMemoryObserver. "
      "If you need it, please file an issue describing your use case.");
}

void NCCLAllocator::attachAllocatorTraceTracker(
    c10::cuda::CUDACachingAllocator::AllocatorTraceTracker tracker) {
  const std::lock_guard<std::mutex> lock(allocator_mutex_);
  trace_trackers_.emplace_back(std::move(tracker));
}

std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState>
NCCLAllocator::getCheckpointState(c10::DeviceIndex device, at::cuda::MempoolId_t id) {
  TORCH_CHECK(
      false,
      "NCCLAllocator does not yet support getCheckpointState. "
      "If you need it, please file an issue describing your use case.");
}

c10::cuda::CUDACachingAllocator::CheckpointDelta NCCLAllocator::
    setCheckpointPoolState(
        c10::DeviceIndex device,
        std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> pps) {
  TORCH_CHECK(
      false,
      "NCCLAllocator does not yet support setCheckpointPoolState. "
      "If you need it, please file an issue describing your use case.");
}

void NCCLAllocator::enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) {
  c10::cuda::CUDAGuard device_guard(static_cast<int>(dev));
  cudaError_t err = cudaDeviceEnablePeerAccess(dev_to_access, 0);
  if (err == cudaErrorPeerAccessAlreadyEnabled) {
    // ignore and clear the error if access was already enabled
    (void)cudaGetLastError();
  } else {
    C10_CUDA_CHECK(err);
  }
}

cudaError_t NCCLAllocator::memcpyAsync(
    void* dst,
    int dstDevice,
    const void* src,
    int srcDevice,
    size_t count,
    cudaStream_t stream,
    bool p2p_enabled) {
  return cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
}

std::string NCCLAllocator::name() {
  return "pluggable";
}

void NCCLAllocator::copy_data(void* dest, const void* src, std::size_t count)
    const {
  C10_CUDA_CHECK(
      cudaMemcpy(dest, src, count, cudaMemcpyKind::cudaMemcpyDeviceToDevice));
}

} // namespace nccl_allocator::cuda
