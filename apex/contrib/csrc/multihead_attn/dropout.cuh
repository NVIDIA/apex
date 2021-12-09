#pragma once
#include <ATen/ATen.h>

#ifdef OLD_GENERATOR
#include <ATen/CUDAGenerator.h>
#else
#include <ATen/CUDAGeneratorImpl.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <curand_kernel.h>

namespace {

const int UNROLL = 4;

} // namespace anonymous

template <typename scalar_t, typename accscalar_t, typename IndexType>
__global__ void
apex_fused_dropout_kernel(scalar_t const *inputs, scalar_t *outputs,
                          uint8_t *mask, IndexType totalElements, accscalar_t p,
                          std::pair<uint64_t, uint64_t> seeds);

template <typename scalar_t, typename accscalar_t, typename IndexType>
__global__ void apex_dropout_add_kernel(scalar_t const *inputs,
                                        scalar_t const *add_inputs,
                                        scalar_t *outputs, uint8_t *mask,
                                        IndexType totalElements, accscalar_t p,
                                        std::pair<uint64_t, uint64_t> seeds);

template <typename scalar_t, typename accscalar_t, typename IndexType>
__global__ void apex_add_kernel(scalar_t const *inputs,
                                scalar_t const *add_inputs, scalar_t *outputs,
                                IndexType totalElements);

template <typename scalar_t, typename accscalar_t, typename IndexType>
__global__ void apex_masked_scale_kernel(scalar_t const *inputs,
                                         scalar_t *outputs, uint8_t const *mask,
                                         IndexType totalElements,
                                         accscalar_t scale);

template <typename scalar_t, typename accscalar_t, typename IndexType>
void apex_fused_dropout_cuda(scalar_t const *inputs, scalar_t *outputs,
                             uint8_t *mask, IndexType totalElements,
                             accscalar_t p);

template <typename scalar_t, typename accscalar_t, typename IndexType>
void apex_dropout_add_cuda(scalar_t const *inputs, scalar_t const *add_inputs,
                           scalar_t *outputs, uint8_t *mask,
                           IndexType totalElements, accscalar_t p);

template <typename scalar_t, typename accscalar_t, typename IndexType>
void apex_add_cuda(scalar_t const *inputs, scalar_t const *add_inputs,
                   scalar_t *outputs, IndexType totalElements);

template <typename scalar_t, typename accscalar_t, typename IndexType>
void apex_masked_scale_cuda(scalar_t const *inputs, scalar_t *outputs,
                            uint8_t const *mask, IndexType totalElements,
                            accscalar_t scale);
