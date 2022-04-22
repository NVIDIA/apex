#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>


template<typename T>
__device__ __forceinline__ T __stochastic_round(const float value,
						const float rand)
{
#if defined(USE_ROCM)
  if (value != value) {
#elif defined(_MSC_VER)
  if (isnan(value)) {
#else
  if (std::isnan(value)) {
#endif
    return static_cast<T>(UINT16_C(0x7FC0));
  } else {
    union {
      int32_t I32;
      float F32;
    };

    F32 = value;
    int32_t shifted_i32 = I32 >> 16;
    int32_t ru_bias = (shifted_i32 | 1) + INT32_C(0x7FFF);
    int32_t rd_bias = (shifted_i32 & 0) + INT32_C(0x7FFF);
    float ru = __int_as_float((I32 + ru_bias) & ~(((uint32_t)1 << 16) - 1));
    float rd = __int_as_float((I32 + rd_bias) & ~(((uint32_t)1 << 16) - 1));
    float p = (value - rd) / ((ru - rd) + 1e-8);
    float mask = rand <= p;
    return static_cast<T>(mask * ru + (1.0f - mask) * rd);
  }
}