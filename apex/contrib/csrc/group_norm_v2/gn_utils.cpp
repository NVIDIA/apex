#include "gn_utils.hpp"

#include <mutex>
#include <vector>

namespace group_norm_v2 {

cudaDeviceProp const& get_device_prop(int device_id) {
  static std::vector<cudaDeviceProp> device_props;
  static std::once_flag flag;
  std::call_once(flag, [&] {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    device_props.resize(count);
    for (int i = 0; i < count; i++) {
      CUDA_CHECK(cudaGetDeviceProperties(&device_props[i], i));
    }
  });
  return device_props.at(device_id);
}

}  // namespace group_norm_v2
