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
