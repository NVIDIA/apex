#include <ATen/cuda/CUDAContext.h>
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

namespace at {
namespace cuda {

namespace utils {

//eventually should be replaced by real query functions
static inline int MultiprocessorCount(int device_id) {
    return getDeviceProperties(device_id)->multiProcessorCount;
}

static inline int SMArch(int device_id) {
    auto device_property = getDeviceProperties(device_id);
    int cc = device_property->major * 10 + device_property->minor;
    return cc;
}

static inline int MaxSharedMemoryPerMultiprocessor(int device_id) {
    return getDeviceProperties(device_id)->sharedMemPerMultiprocessor;
}


}
}
}


#endif
