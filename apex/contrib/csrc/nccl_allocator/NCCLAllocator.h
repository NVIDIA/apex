#pragma once

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <map>
#include <mutex>

namespace nccl_allocator::cuda {

std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
createNCCLAllocator();

struct _AllocationMetadata {
  _AllocationMetadata();
  _AllocationMetadata(size_t size, c10::DeviceIndex device_idx, cudaStream_t stream, void* ptr);
  size_t size;
  int device_idx;
  cudaStream_t stream;
  void* ptr;
  bool is_free;
};

struct NCCLAllocator
    : public c10::cuda::CUDACachingAllocator::CUDAAllocator {
  NCCLAllocator() = default;

  NCCLAllocator(NCCLAllocator& other);

  void* malloc(size_t size, c10::DeviceIndex device, cudaStream_t stream);

  c10::DataPtr allocate(size_t size) override;
  c10::DeleterFnPtr raw_deleter() const override;

  void* raw_alloc(size_t nbytes) override;
  void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override;
  void raw_delete(void* ptr) override;
  void init(int device_count) override;
  bool initialized() override;
  void setMemoryFraction(double fraction, c10::DeviceIndex device) override;
  void emptyCache() override;
  void cacheInfo(c10::DeviceIndex dev_id, size_t* largestBlock) override;
  void* getBaseAllocation(void* ptr, size_t* size) override;

  void recordStream(const c10::DataPtr&, c10::cuda::CUDAStream stream) override;

  c10::cuda::CUDACachingAllocator::DeviceStats getDeviceStats(
      c10::DeviceIndex device) override;
  void resetAccumulatedStats(c10::DeviceIndex device) override;
  void resetPeakStats(c10::DeviceIndex device) override;
  c10::cuda::CUDACachingAllocator::SnapshotInfo snapshot() override;
  void beginAllocateToPool(
      c10::DeviceIndex device,
      c10::cuda::MempoolId_t mempool_id,
      std::function<bool(cudaStream_t)>) override;
  void endAllocateToPool(c10::DeviceIndex device, c10::cuda::MempoolId_t mempool_id)
      override;
  void releasePool(c10::DeviceIndex device, c10::cuda::MempoolId_t mempool_id) override;
  std::shared_ptr<void> getIpcDevPtr(std::string handle) override;
  void recordHistory(
      bool enabled,
      c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      c10::cuda::CUDACachingAllocator::RecordContext when) override;
  void attachOutOfMemoryObserver(
      c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) override;
  void attachAllocatorTraceTracker(
      c10::cuda::CUDACachingAllocator::AllocatorTraceTracker tracker) override;
  std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState>
  getCheckpointState(c10::DeviceIndex device, at::cuda::MempoolId_t id) override;
  c10::cuda::CUDACachingAllocator::CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device,
      std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> pps)
      override;
  void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) override;
  cudaError_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      cudaStream_t stream,
      bool p2p_enabled) override;
  std::string name() override;
  void copy_data(void* dest, const void* src, std::size_t count) const;

 protected:
  std::function<void(int)> init_fn_;
  std::function<void(double, int)> memory_fraction_fn_;
  std::function<void*(void*, size_t*)> base_alloc_fn_;
  std::function<void(void* ptr, cudaStream_t stream)> record_stream_fn_;
  std::function<
      void(int, c10::cuda::MempoolId_t, std::function<bool(cudaStream_t)>)>
      begin_allocate_to_pool_fn_;
  std::function<void(int, c10::cuda::MempoolId_t)> end_allocate_to_pool_fn_;
  std::function<void(int, c10::cuda::MempoolId_t)> relase_pool_fn_;
  std::mutex allocator_mutex_;
  // We do the bookeeping here in order to simplify custom allocators
  std::unordered_map<void*, _AllocationMetadata> cuda_rt_allocation_metadata_;
  std::multimap<size_t, _AllocationMetadata> cuda_rt_cache_metadata_;
  std::unordered_map<void*, _AllocationMetadata> nccl_allocation_metadata_;

  bool initialized_ = false;
  std::vector<c10::cuda::CUDACachingAllocator::AllocatorTraceTracker> trace_trackers_;
}; // NCCLAllocator
} // namespace nccl_allocator::cuda
