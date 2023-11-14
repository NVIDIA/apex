#ifdef USE_ROCM
#include <ATen/hip/HIPContext.h>
#else
#include <ATen/cuda/CUDAContext.h>
#endif
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

namespace at {
namespace cuda {

namespace utils {

static inline int MaxSharedMemoryPerMultiprocessor(int device_id) {
#ifdef USE_ROCM
    return getDeviceProperties(device_id)->maxSharedMemoryPerMultiProcessor;
#else
    return getDeviceProperties(device_id)->sharedMemPerMultiprocessor;
#endif
}


}
}
}


#endif
