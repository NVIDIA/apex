#include "THCTensorInfo.cuh"
#include <iostream>
#include <cstdio>
#include <cassert>
#include <cuda.h>
#include <vector>

#define Dims -2
#define DEVICE_LINEAR_GET(D_TENSOR, INDEX) D_TENSOR.data[IndexToOffset<T, IndexType, Dims>::get(INDEX, D_TENSOR)]
#define DEVICE_LINEAR_GET_F(D_TENSOR, INDEX) D_TENSOR.data[IndexToOffset<float, IndexType, Dims>::get(INDEX, D_TENSOR)]

typedef int idxType;

struct send_to_fwd_wrapper
{
  template<typename DataType, 
           typename AccumType, 
           typename IndexType>
  static void call(std::vector<TensorInfo<void, idxType>>& tensors, int dim);
};

struct send_to_bwd_wrapper
{
  template<typename DataType,
           typename AccumType,
           typename IndexType>
  static void call(std::vector<TensorInfo<void, idxType>>& tensors, int dim);
};

template <typename In, typename Out>
struct ScalarConvert {
  static __host__ __device__ __forceinline__ Out to(const In v) { return (Out) v; }
};

#ifdef CUDA_HALF_TENSOR
template <typename Out>
struct ScalarConvert<half, Out> {
  static __host__ __device__ __forceinline__ Out to(const half v) {
#ifdef __CUDA_ARCH__
    return (Out) __half2float(v);
#else
    return (Out) THC_half2float(v);
#endif
  }
};

template <typename In>
struct ScalarConvert<In, half> {
  static __host__ __device__ __forceinline__ half to(const In v) {
#ifdef __CUDA_ARCH__
    return __float2half((float) v);
#else
    return THC_float2half((float) v);
#endif
  }
};

template <>
struct ScalarConvert<half, half> {
  static __host__ __device__ __forceinline__ half to(const half v) {
    return v;
  }
};

#endif
