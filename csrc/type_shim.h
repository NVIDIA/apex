#include <ATen/ATen.h>

// Forward/backward compatiblity hack around
// https://github.com/pytorch/pytorch/commit/3aeb78079bcd68282fe9117088e138b77318e288
// pending more future-proof guidance from upstream.
struct TypeShim
{
  const at::Type& payload;
  TypeShim(const at::Type& type) : payload(type) {}
  // Enable trivial conversion to a const at::Type& for pre-3aeb78
  operator const at::Type&(){ return payload; };
  // Enable dispatch switch statements to take *this directly for  post-3aeb78
  operator at::ScalarType(){ return payload.scalarType(); };
};

#define DISPATCH_FLOAT_AND_HALF(TYPE, LEVEL, NAME, ...) \
  switch(TYPE) \
  { \
    case at::ScalarType::Float: \
    { \
      using scalar_t_##LEVEL = float; \
      __VA_ARGS__; \
      break; \
    } \
    case at::ScalarType::Half: \
    { \
      using scalar_t_##LEVEL = at::Half; \
      __VA_ARGS__; \
      break; \
    } \
    default: \
      AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");  \
  }
