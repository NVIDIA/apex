#pragma once

#include <ATen/Half.h>
#include <ATen/cuda/CUDAHalf.cuh>

// Type traits to convert types to CUDA-specific types. Used primarily to
// convert at::Half to CUDA's half type. This makes the conversion explicit.

// Disambiguate from whatever is in aten
namespace apex { namespace cuda {
template <typename T>
struct TypeConversion {
  using type = T;
};

template <>
struct TypeConversion<at::Half> {
  using type = half;
};

template <typename T>
using type = typename TypeConversion<T>::type;
}} // namespace apex::cuda
