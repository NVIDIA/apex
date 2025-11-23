#pragma once

#ifdef TORCH_STABLE_ONLY

// Stable ABI headers
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ivalue.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/dispatcher.h>
#include <torch/headeronly/types.h>

namespace apex {
namespace stable {

// ============================================================================
// MemoryFormat Contiguity Checking Workaround
// ============================================================================
// The stable ABI's Tensor::is_contiguous() doesn't support MemoryFormat
// parameter. This provides a workaround for checking different memory layouts.

enum class MemoryFormat {
  Contiguous,
  ChannelsLast,
  ChannelsLast3d,
  Preserve
};

// Check if a tensor is contiguous in a specific memory format
inline bool is_contiguous(const torch::stable::Tensor& tensor, MemoryFormat format) {
  using namespace torch::stable;

  // For standard contiguous check, use the stable ABI method
  if (format == MemoryFormat::Contiguous) {
    return tensor.is_contiguous();
  }

  // For ChannelsLast and ChannelsLast3d, we need custom logic
  // Get tensor properties
  auto sizes = tensor.sizes();
  auto strides = tensor.strides();
  int64_t ndim = tensor.dim();

  if (format == MemoryFormat::ChannelsLast) {
    // NCHW format requires ndim == 4
    if (ndim != 4) return false;

    // For ChannelsLast (NHWC), strides should follow: C=1, W=C, H=W*W_size, N=H*H_size
    // Expected stride order: strides[1] < strides[3] < strides[2] < strides[0]
    int64_t N = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];
    int64_t stride_c = strides[1];
    int64_t stride_w = strides[3];
    int64_t stride_h = strides[2];
    int64_t stride_n = strides[0];

    // Check if strides match NHWC layout
    return (stride_c == 1) &&
           (stride_w == C) &&
           (stride_h == W * C) &&
           (stride_n == H * W * C);
  }

  if (format == MemoryFormat::ChannelsLast3d) {
    // NCDHW format requires ndim == 5
    if (ndim != 5) return false;

    // For ChannelsLast3d (NDHWC), similar logic for 5D tensors
    int64_t N = sizes[0], C = sizes[1], D = sizes[2], H = sizes[3], W = sizes[4];
    int64_t stride_c = strides[1];
    int64_t stride_w = strides[4];
    int64_t stride_h = strides[3];
    int64_t stride_d = strides[2];
    int64_t stride_n = strides[0];

    // Check if strides match NDHWC layout
    return (stride_c == 1) &&
           (stride_w == C) &&
           (stride_h == W * C) &&
           (stride_d == H * W * C) &&
           (stride_n == D * H * W * C);
  }

  return false;
}

// ============================================================================
// Type Conversion Utilities
// ============================================================================

// Convert stable ScalarType to string for error messages
inline const char* scalar_type_name(torch::headeronly::ScalarType type) {
  using namespace torch::headeronly;
  switch (type) {
    case kByte: return "Byte";
    case kChar: return "Char";
    case kShort: return "Short";
    case kInt: return "Int";
    case kLong: return "Long";
    case kHalf: return "Half";
    case kFloat: return "Float";
    case kDouble: return "Double";
    case kBool: return "Bool";
    case kBFloat16: return "BFloat16";
    case kFloat8_e5m2: return "Float8_e5m2";
    case kFloat8_e4m3fn: return "Float8_e4m3fn";
    default: return "Unknown";
  }
}

// ============================================================================
// Error Checking Macros
// ============================================================================

#define STD_TORCH_CHECK(cond, ...) \
  do { \
    if (!(cond)) { \
      char buffer[1024]; \
      snprintf(buffer, sizeof(buffer), __VA_ARGS__); \
      throw std::runtime_error(buffer); \
    } \
  } while (0)

#define STD_TORCH_CHECK_EQ(a, b, ...) STD_TORCH_CHECK((a) == (b), __VA_ARGS__)
#define STD_TORCH_CHECK_NE(a, b, ...) STD_TORCH_CHECK((a) != (b), __VA_ARGS__)
#define STD_TORCH_CHECK_GT(a, b, ...) STD_TORCH_CHECK((a) > (b), __VA_ARGS__)
#define STD_TORCH_CHECK_GE(a, b, ...) STD_TORCH_CHECK((a) >= (b), __VA_ARGS__)
#define STD_TORCH_CHECK_LT(a, b, ...) STD_TORCH_CHECK((a) < (b), __VA_ARGS__)
#define STD_TORCH_CHECK_LE(a, b, ...) STD_TORCH_CHECK((a) <= (b), __VA_ARGS__)

// ============================================================================
// Boxed Calling Convention Helpers
// ============================================================================

// Helper to extract tensor from IValue stack
inline torch::stable::Tensor tensor_from_stack(torch::stable::StableIValue* stack, int idx) {
  return stack[idx].toTensor();
}

// Helper to extract int64 from IValue stack
inline int64_t int64_from_stack(torch::stable::StableIValue* stack, int idx) {
  return stack[idx].toInt();
}

// Helper to extract double from IValue stack
inline double double_from_stack(torch::stable::StableIValue* stack, int idx) {
  return stack[idx].toDouble();
}

// Helper to extract bool from IValue stack
inline bool bool_from_stack(torch::stable::StableIValue* stack, int idx) {
  return stack[idx].toBool();
}

// Helper to extract optional tensor from IValue stack
inline std::optional<torch::stable::Tensor> optional_tensor_from_stack(
    torch::stable::StableIValue* stack, int idx) {
  if (stack[idx].isNone()) {
    return std::nullopt;
  }
  return stack[idx].toTensor();
}

// Helper to extract tensor list from IValue stack
inline std::vector<torch::stable::Tensor> tensor_list_from_stack(
    torch::stable::StableIValue* stack, int idx) {
  auto list = stack[idx].toList();
  std::vector<torch::stable::Tensor> result;
  result.reserve(list.size());
  for (size_t i = 0; i < list.size(); ++i) {
    result.push_back(list.get(i).toTensor());
  }
  return result;
}

// Helper to put tensor to IValue stack
inline void tensor_to_stack(torch::stable::StableIValue* stack, int idx,
                            const torch::stable::Tensor& tensor) {
  stack[idx] = torch::stable::StableIValue::from(tensor);
}

// Helper to put tuple to IValue stack
inline void tuple_to_stack(torch::stable::StableIValue* stack, int idx,
                           const std::vector<torch::stable::Tensor>& tensors) {
  std::vector<torch::stable::StableIValue> ivalues;
  ivalues.reserve(tensors.size());
  for (const auto& t : tensors) {
    ivalues.push_back(torch::stable::StableIValue::from(t));
  }
  stack[idx] = torch::stable::StableIValue::fromTuple(ivalues);
}

// Helper to put list to IValue stack
inline void tensor_list_to_stack(torch::stable::StableIValue* stack, int idx,
                                  const std::vector<torch::stable::Tensor>& tensors) {
  std::vector<torch::stable::StableIValue> ivalues;
  ivalues.reserve(tensors.size());
  for (const auto& t : tensors) {
    ivalues.push_back(torch::stable::StableIValue::from(t));
  }
  stack[idx] = torch::stable::StableIValue::fromList(ivalues);
}

// ============================================================================
// Device and Stream Utilities
// ============================================================================

// Check if tensor is on CUDA
inline bool is_cuda(const torch::stable::Tensor& tensor) {
  return tensor.device().type() == torch::headeronly::kCUDA;
}

// Get CUDA device index
inline int64_t get_device_index(const torch::stable::Tensor& tensor) {
  STD_TORCH_CHECK(is_cuda(tensor), "Tensor must be on CUDA device");
  return tensor.device().index();
}

// ============================================================================
// Common Tensor Checks
// ============================================================================

inline void check_cuda(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(is_cuda(tensor), "%s must be a CUDA tensor", name);
}

inline void check_contiguous(const torch::stable::Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_contiguous(), "%s must be contiguous", name);
}

inline void check_same_device(const torch::stable::Tensor& t1,
                               const torch::stable::Tensor& t2,
                               const char* name1, const char* name2) {
  STD_TORCH_CHECK(t1.device() == t2.device(),
                  "%s and %s must be on the same device", name1, name2);
}

inline void check_same_dtype(const torch::stable::Tensor& t1,
                              const torch::stable::Tensor& t2,
                              const char* name1, const char* name2) {
  STD_TORCH_CHECK(t1.scalar_type() == t2.scalar_type(),
                  "%s and %s must have the same dtype, got %s and %s",
                  name1, name2,
                  scalar_type_name(t1.scalar_type()),
                  scalar_type_name(t2.scalar_type()));
}

} // namespace stable
} // namespace apex

#else // !TORCH_STABLE_ONLY

// When not using stable ABI, provide no-op definitions or traditional includes
#include <torch/extension.h>
#include <ATen/ATen.h>

namespace apex {
namespace stable {

// Map to traditional PyTorch MemoryFormat for non-stable builds
using MemoryFormat = at::MemoryFormat;

// Use traditional is_contiguous in non-stable builds
inline bool is_contiguous(const at::Tensor& tensor, at::MemoryFormat format) {
  return tensor.is_contiguous(format);
}

} // namespace stable
} // namespace apex

#endif // TORCH_STABLE_ONLY
