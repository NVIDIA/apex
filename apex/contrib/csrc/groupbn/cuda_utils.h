#include <ATen/cuda/CUDAContext.h>
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

namespace at {
namespace cuda {

namespace utils {

static inline int MaxSharedMemoryPerMultiprocessor(int device_id) {
    return getDeviceProperties(device_id)->sharedMemPerMultiprocessor;
}


}
}
}


#endif
