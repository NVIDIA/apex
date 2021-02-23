#pragma once

#include "torch/extension.h"
#include <ATen/cuda/Exceptions.h> // for CUDNN_CHECK

#define DIVUP(x, y) (((x) + ((y)-1)) / (y))

#define DISPATCH_FLOAT_AND_HALF(TYPE, NAME, ...)                               \
  [&] {                                                                        \
    const auto &the_type = TYPE;                                               \
    /* don't use TYPE again in case it is an expensive or side-effect op */    \
    at::ScalarType _st = ::detail::scalar_type(the_type);                      \
    switch (_st) {                                                             \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__)        \
    default:                                                                   \
      AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");           \
    }                                                                          \
  }()

template <int Bytes> struct Vec_type {};

template <> struct Vec_type<16> {
  using Type = uint4;
  static __device__ inline Type zero() { return make_uint4(0, 0, 0, 0); }
};
template <> struct Vec_type<8> {
  using Type = uint2;
  static __device__ inline Type zero() { return make_uint2(0, 0); }
};

template <> struct Vec_type<4> {
  using Type = uint32_t;
  static __device__ inline Type zero() { return 0; }
};

template <> struct Vec_type<2> {
  using Type = uint16_t;
  static __device__ inline Type zero() { return 0; }
};

template <typename T> struct TypeInfo {
  using base_t = T;
  using packed_t = T;
  using compute_t = float;
  using packed_compute_t = float;
};

template <> struct TypeInfo<half> {
  using base_t = half;
  using packed_t = half2;
  using compute_t = float;
  using packed_compute_t = float2;
};

template <typename dtype, int Bytes> struct Vec {

  using base_t = typename TypeInfo<dtype>::base_t;
  using packed_t = typename TypeInfo<dtype>::packed_t;
  using compute_t = typename TypeInfo<dtype>::compute_t;
  using packed_compute_t = typename TypeInfo<dtype>::packed_compute_t;

  static_assert(Bytes % sizeof(base_t) == 0, "");
  static_assert(Bytes % sizeof(packed_t) == 0, "");
  enum { BYTES_PER_THREAD = Bytes };
  enum { NUM_ELTS = Bytes / sizeof(base_t) };
  enum { NUM_PACKED = Bytes / sizeof(packed_t) };
  using vec_t = typename Vec_type<Bytes>::Type;
  using store_t = union {
    vec_t raw;
    base_t elt[NUM_ELTS];
    packed_t packed[NUM_PACKED];
  };
  store_t data;

  __device__ Vec() { data.raw = Vec_type<Bytes>::zero(); }

  __device__ inline void load_from(const char *ptr) {
    data.raw = *reinterpret_cast<const vec_t *>(ptr);
  }

  __device__ inline void load_or_zero(const char *ptr, const bool is_valid) {
    data.raw = is_valid ? *reinterpret_cast<const vec_t *>(ptr)
                        : Vec_type<Bytes>::zero();
  }

  __device__ inline void store_to(char *ptr) const {
    *reinterpret_cast<vec_t *>(ptr) = data.raw;
  }

  __device__ inline void store_valid(char *ptr, const bool is_valid) const {
    if (is_valid)
      *reinterpret_cast<vec_t *>(ptr) = data.raw;
  }
};
