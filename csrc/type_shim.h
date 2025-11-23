#ifdef TORCH_STABLE_ONLY
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/types.h>
#include "stable_abi_utils.h"

// Error macro for stable ABI
#define APEX_ERROR(...) apex::stable::STD_TORCH_CHECK(false, __VA_ARGS__)

// Namespace and type aliases for stable ABI
namespace apex_internal {
  using ScalarType = torch::headeronly::ScalarType;
  using Half = torch::headeronly::Half;
  using BFloat16 = torch::headeronly::BFloat16;

  inline std::string toString(ScalarType type) {
    return std::string(apex::stable::scalar_type_name(type));
  }
}

#else // !TORCH_STABLE_ONLY

#include <ATen/ATen.h>

// Error macro for traditional API
#define APEX_ERROR(...) AT_ERROR(__VA_ARGS__)

// Namespace and type aliases for traditional API
namespace apex_internal {
  using ScalarType = at::ScalarType;
  using Half = at::Half;
  using BFloat16 = at::BFloat16;

  inline std::string toString(at::ScalarType type) {
    return std::string(c10::toString(type));
  }
}

#endif // TORCH_STABLE_ONLY

// Forward/backward compatiblity hack around
// https://github.com/pytorch/pytorch/commit/3aeb78079bcd68282fe9117088e138b77318e288
// pending more future-proof guidance from upstream.
// struct TypeShim
// {
//   const at::Type& payload;
//   TypeShim(const at::Type& type) : payload(type) {}
//   // Enable trivial conversion to a const at::Type& for pre-3aeb78
//   operator const at::Type&(){ return payload; };
//   // Enable dispatch switch statements to take *this directly for  post-3aeb78
//   //operator at::ScalarType(){ return payload.; };
// };

#define DISPATCH_FLOAT_AND_HALF(TYPE, LEVEL, NAME, ...)               \
  switch (TYPE) {                                                     \
    case apex_internal::ScalarType::Float: {                          \
      using scalar_t_##LEVEL = float;                                 \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case apex_internal::ScalarType::Half: {                           \
      using scalar_t_##LEVEL = apex_internal::Half;                   \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    default:                                                          \
      APEX_ERROR(#NAME, " not implemented for '", apex_internal::toString(TYPE), "'"); \
  }

#define DISPATCH_FLOAT_HALF_AND_BFLOAT(TYPE, LEVEL, NAME, ...)        \
  switch (TYPE) {                                                     \
    case apex_internal::ScalarType::Float: {                          \
      using scalar_t_##LEVEL = float;                                 \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case apex_internal::ScalarType::Half: {                           \
      using scalar_t_##LEVEL = apex_internal::Half;                   \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case apex_internal::ScalarType::BFloat16: {                       \
      using scalar_t_##LEVEL = apex_internal::BFloat16;               \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    default:                                                          \
      APEX_ERROR(#NAME, " not implemented for '", apex_internal::toString(TYPE), "'"); \
  }

#define DISPATCH_FLOAT_HALF_AND_BYTE(TYPE, LEVEL, NAME, ...)          \
  switch (TYPE) {                                                     \
    case apex_internal::ScalarType::Float: {                          \
      using scalar_t_##LEVEL = float;                                 \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case apex_internal::ScalarType::Half: {                           \
      using scalar_t_##LEVEL = apex_internal::Half;                   \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case apex_internal::ScalarType::Byte: {                           \
      using scalar_t_##LEVEL = uint8_t;                               \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    default:                                                          \
      APEX_ERROR(#NAME, " not implemented for '", apex_internal::toString(TYPE), "'"); \
  }

#define DISPATCH_DOUBLE_FLOAT_AND_HALF(TYPE, LEVEL, NAME, ...)        \
  switch (TYPE) {                                                     \
    case apex_internal::ScalarType::Double: {                         \
      using scalar_t_##LEVEL = double;                                \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case apex_internal::ScalarType::Float: {                          \
      using scalar_t_##LEVEL = float;                                 \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case apex_internal::ScalarType::Half: {                           \
      using scalar_t_##LEVEL = apex_internal::Half;                   \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    default:                                                          \
      APEX_ERROR(#NAME, " not implemented for '", apex_internal::toString(TYPE), "'"); \
  }

#define DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT(TYPE, LEVEL, NAME, ...) \
  switch (TYPE) {                                                     \
    case apex_internal::ScalarType::Double: {                         \
      using scalar_t_##LEVEL = double;                                \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case apex_internal::ScalarType::Float: {                          \
      using scalar_t_##LEVEL = float;                                 \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case apex_internal::ScalarType::Half: {                           \
      using scalar_t_##LEVEL = apex_internal::Half;                   \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case apex_internal::ScalarType::BFloat16: {                       \
      using scalar_t_##LEVEL = apex_internal::BFloat16;               \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    default:                                                          \
      APEX_ERROR(#NAME, " not implemented for '", apex_internal::toString(TYPE), "'"); \
  }

#define DISPATCH_DOUBLE_AND_FLOAT(TYPE, LEVEL, NAME, ...)             \
  switch (TYPE) {                                                     \
    case apex_internal::ScalarType::Double: {                         \
      using scalar_t_##LEVEL = double;                                \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case apex_internal::ScalarType::Float: {                          \
      using scalar_t_##LEVEL = float;                                 \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    default:                                                          \
      APEX_ERROR(#NAME, " not implemented for '", apex_internal::toString(TYPE), "'"); \
  }

#define DISPATCH_HALF_AND_BFLOAT(TYPE, NAME, ...)                     \
  switch (TYPE) {                                                     \
    case apex_internal::ScalarType::Half: {                           \
      using scalar_t = apex_internal::Half;                           \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    case apex_internal::ScalarType::BFloat16: {                       \
      using scalar_t = apex_internal::BFloat16;                       \
      __VA_ARGS__;                                                    \
      break;                                                          \
    }                                                                 \
    default:                                                          \
      APEX_ERROR(#NAME, " not implemented for '", apex_internal::toString(TYPE), "'"); \
  }

#define DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(TYPEIN, TYPEOUT, NAME, ...) \
  switch (TYPEIN) {                                                            \
    case apex_internal::ScalarType::Float: {                                   \
      using scalar_t_in = float;                                               \
      switch (TYPEOUT) {                                                       \
        case apex_internal::ScalarType::Float: {                               \
          using scalar_t_out = float;                                          \
          __VA_ARGS__;                                                         \
          break;                                                               \
        }                                                                      \
        case apex_internal::ScalarType::Half: {                                \
          using scalar_t_out = apex_internal::Half;                            \
          __VA_ARGS__;                                                         \
          break;                                                               \
        }                                                                      \
        case apex_internal::ScalarType::BFloat16: {                            \
          using scalar_t_out = apex_internal::BFloat16;                        \
          __VA_ARGS__;                                                         \
          break;                                                               \
        }                                                                      \
        default:                                                               \
          APEX_ERROR(#NAME, " not implemented for '", apex_internal::toString(TYPEOUT), "'"); \
      }                                                                        \
      break;                                                                   \
    }                                                                          \
    case apex_internal::ScalarType::Half: {                                    \
      using scalar_t_in = apex_internal::Half;                                 \
      using scalar_t_out = apex_internal::Half;                                \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    case apex_internal::ScalarType::BFloat16: {                                \
      using scalar_t_in = apex_internal::BFloat16;                             \
      using scalar_t_out = apex_internal::BFloat16;                            \
      __VA_ARGS__;                                                             \
      break;                                                                   \
    }                                                                          \
    default:                                                                   \
      APEX_ERROR(#NAME, " not implemented for '", apex_internal::toString(TYPEIN), "'"); \
  }

#define DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(TYPEIN, TYPEOUT, NAME, ...) \
  switch (TYPEIN) {                                                                   \
    case apex_internal::ScalarType::Double: {                                         \
      using scalar_t_in = double;                                                     \
      switch (TYPEOUT) {                                                              \
        case apex_internal::ScalarType::Double: {                                     \
          using scalar_t_out = double;                                                \
          __VA_ARGS__;                                                                \
          break;                                                                      \
        }                                                                             \
        case apex_internal::ScalarType::Float: {                                      \
          using scalar_t_out = float;                                                 \
          __VA_ARGS__;                                                                \
          break;                                                                      \
        }                                                                             \
        case apex_internal::ScalarType::Half: {                                       \
          using scalar_t_out = apex_internal::Half;                                   \
          __VA_ARGS__;                                                                \
          break;                                                                      \
        }                                                                             \
        case apex_internal::ScalarType::BFloat16: {                                   \
          using scalar_t_out = apex_internal::BFloat16;                               \
          __VA_ARGS__;                                                                \
          break;                                                                      \
        }                                                                             \
        default:                                                                      \
          APEX_ERROR(#NAME, " not implemented for '", apex_internal::toString(TYPEOUT), "'"); \
      }                                                                               \
      break;                                                                          \
    }                                                                                 \
    case apex_internal::ScalarType::Float: {                                          \
      using scalar_t_in = float;                                                      \
      switch (TYPEOUT) {                                                              \
        case apex_internal::ScalarType::Float: {                                      \
          using scalar_t_out = float;                                                 \
          __VA_ARGS__;                                                                \
          break;                                                                      \
        }                                                                             \
        case apex_internal::ScalarType::Half: {                                       \
          using scalar_t_out = apex_internal::Half;                                   \
          __VA_ARGS__;                                                                \
          break;                                                                      \
        }                                                                             \
        case apex_internal::ScalarType::BFloat16: {                                   \
          using scalar_t_out = apex_internal::BFloat16;                               \
          __VA_ARGS__;                                                                \
          break;                                                                      \
        }                                                                             \
        default:                                                                      \
          APEX_ERROR(#NAME, " not implemented for '", apex_internal::toString(TYPEOUT), "'"); \
      }                                                                               \
      break;                                                                          \
    }                                                                                 \
    case apex_internal::ScalarType::Half: {                                           \
      using scalar_t_in = apex_internal::Half;                                        \
      using scalar_t_out = apex_internal::Half;                                       \
      __VA_ARGS__;                                                                    \
      break;                                                                          \
    }                                                                                 \
    case apex_internal::ScalarType::BFloat16: {                                       \
      using scalar_t_in = apex_internal::BFloat16;                                    \
      using scalar_t_out = apex_internal::BFloat16;                                   \
      __VA_ARGS__;                                                                    \
      break;                                                                          \
    }                                                                                 \
    default:                                                                          \
      APEX_ERROR(#NAME, " not implemented for '", apex_internal::toString(TYPEIN), "'"); \
  }

template <typename T>
__device__ __forceinline__ T reduce_block_into_lanes(T* x, T val, int lanes = 1,
                                                     bool share_result = false)  // lanes is intended to be <= 32.
{
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int blockSize = blockDim.x * blockDim.y;  // blockSize is intended to be a multiple of 32.

  if (blockSize >= 64) {
    x[tid] = val;
    __syncthreads();
  }

#pragma unroll
  for (int i = (blockSize >> 1); i >= 64; i >>= 1) {
    if (tid < i) x[tid] = x[tid] + x[tid + i];
    __syncthreads();
  }

  T final;

  if (tid < 32) {
    if (blockSize >= 64)
      final = x[tid] + x[tid + 32];
    else
      final = val;
    // __SYNCWARP();

#pragma unroll
    for (int i = 16; i >= lanes; i >>= 1) final = final + __shfl_down_sync(0xffffffff, final, i);
  }

  if (share_result) {
    if (tid < lanes) x[tid] = final;  // EpilogueOp
    // Make sure the smem result is visible to all warps.
  }
  __syncthreads();
  // Avoid potential write before read race when reduce_block_into_lanes is called back to back

  return final;
}

template <typename T>
__device__ __forceinline__ T
reduce_block_into_lanes_max_op(T* x, T val, int lanes = 1,
                               bool share_result = false)  // lanes is intended to be <= 32.
{
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int blockSize = blockDim.x * blockDim.y;  // blockSize is intended to be a multiple of 32.

  if (blockSize >= 64) {
    x[tid] = val;
    __syncthreads();
  }

#pragma unroll
  for (int i = (blockSize >> 1); i >= 64; i >>= 1) {
    if (tid < i) x[tid] = fmaxf(fabsf(x[tid]), fabsf(x[tid + i]));
    __syncthreads();
  }

  T final;

  if (tid < 32) {
    if (blockSize >= 64)
      final = fmaxf(fabsf(x[tid]), fabsf(x[tid + 32]));
    else
      final = val;
    // __SYNCWARP();

#pragma unroll
    for (int i = 16; i >= lanes; i >>= 1) final = fmaxf(fabsf(final), fabsf(__shfl_down_sync(0xffffffff, final, i)));
  }

  if (share_result) {
    if (tid < lanes) x[tid] = final;  // EpilogueOp
    // Make sure the smem result is visible to all warps.
    __syncthreads();
  }

  return final;
}
