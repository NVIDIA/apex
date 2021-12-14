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
constexpr int UNROLL = 4;
} // namespace

template <typename scalar_t, typename accscalar_t, typename IndexType>
__global__ void
apex_fused_dropout_kernel(scalar_t const *inputs, scalar_t *outputs,
                          uint8_t *mask, IndexType totalElements, accscalar_t p,
                          std::pair<uint64_t, uint64_t> seeds) {
  accscalar_t pinv = accscalar_t(1) / p;
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;

  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, idx, seeds.second, &state);

  IndexType rounded_size =
      ((totalElements - 1) / (blockDim.x * gridDim.x * UNROLL) + 1) *
      blockDim.x * gridDim.x * UNROLL;
  for (IndexType linearIndex = idx; linearIndex < rounded_size;
       linearIndex += gridDim.x * blockDim.x * UNROLL) {
    float4 rand = curand_uniform4(&state);
    scalar_t src[UNROLL];
    rand.x = rand.x <= p;
    rand.y = rand.y <= p;
    rand.z = rand.z <= p;
    rand.w = rand.w <= p;

    for (int ii = 0; ii < UNROLL; ii++) {
      IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
      if (li < totalElements) {
        src[ii] = inputs[li];
      }
    }
    for (int ii = 0; ii < UNROLL; ii++) {
      IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
      if (li < totalElements) {
        outputs[li] = src[ii] * (&rand.x)[ii] * pinv;
        mask[li] = (uint8_t)(&rand.x)[ii];
      }
    }
    __syncthreads();
  }
}

template <typename scalar_t, typename accscalar_t, typename IndexType>
__global__ void apex_dropout_add_kernel(scalar_t const *inputs,
                                        scalar_t const *add_inputs,
                                        scalar_t *outputs, uint8_t *mask,
                                        IndexType totalElements, accscalar_t p,
                                        std::pair<uint64_t, uint64_t> seeds) {
  accscalar_t pinv = accscalar_t(1) / p;
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;

  curandStatePhilox4_32_10_t state;
  curand_init(seeds.first, idx, seeds.second, &state);

  IndexType rounded_size =
      ((totalElements - 1) / (blockDim.x * gridDim.x * UNROLL) + 1) *
      blockDim.x * gridDim.x * UNROLL;
  for (IndexType linearIndex = idx; linearIndex < rounded_size;
       linearIndex += gridDim.x * blockDim.x * UNROLL) {
    float4 rand = curand_uniform4(&state);
    scalar_t src[UNROLL];
    scalar_t add_src[UNROLL];
    rand.x = rand.x <= p;
    rand.y = rand.y <= p;
    rand.z = rand.z <= p;
    rand.w = rand.w <= p;
    for (int ii = 0; ii < UNROLL; ii++) {
      IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
      if (li < totalElements) {
        src[ii] = inputs[li];
        add_src[ii] = add_inputs[li];
      }
    }
    for (int ii = 0; ii < UNROLL; ii++) {
      IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
      if (li < totalElements) {
        accscalar_t int1 = src[ii] * (&rand.x)[ii] * pinv;
        outputs[li] =
            static_cast<scalar_t>(static_cast<accscalar_t>(add_src[ii]) + int1);
        mask[li] = (uint8_t)(&rand.x)[ii];
      }
    }
    __syncthreads();
  }
}

template <typename scalar_t, typename accscalar_t, typename IndexType>
__global__ void apex_add_kernel(scalar_t const *inputs,
                                scalar_t const *add_inputs, scalar_t *outputs,
                                IndexType totalElements) {
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType rounded_size =
      ((totalElements - 1) / (blockDim.x * gridDim.x * UNROLL) + 1) *
      blockDim.x * gridDim.x * UNROLL;
  for (IndexType linearIndex = idx; linearIndex < rounded_size;
       linearIndex += gridDim.x * blockDim.x * UNROLL) {
    scalar_t src[UNROLL];
    scalar_t add_src[UNROLL];
    for (int ii = 0; ii < UNROLL; ii++) {
      IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
      if (li < totalElements) {
        src[ii] = inputs[li];
        add_src[ii] = add_inputs[li];
      }
    }
    for (int ii = 0; ii < UNROLL; ii++) {
      IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
      if (li < totalElements) {
        outputs[li] = src[ii] + add_src[ii];
      }
    }
    __syncthreads();
  }
}

template <typename scalar_t, typename accscalar_t, typename IndexType>
__global__ void apex_masked_scale_kernel(scalar_t const *inputs,
                                         scalar_t *outputs, uint8_t const *mask,
                                         IndexType totalElements,
                                         accscalar_t scale) {
  IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType rounded_size =
      ((totalElements - 1) / (blockDim.x * gridDim.x * UNROLL) + 1) *
      blockDim.x * gridDim.x * UNROLL;
  for (IndexType linearIndex = idx; linearIndex < rounded_size;
       linearIndex += gridDim.x * blockDim.x * UNROLL) {
    scalar_t src[UNROLL];
    scalar_t msk[UNROLL];
    for (int ii = 0; ii < UNROLL; ii++) {
      IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
      if (li < totalElements) {
        src[ii] = static_cast<scalar_t>(inputs[li]);
        msk[ii] = static_cast<scalar_t>(mask[li]);
      }
    }
    for (int ii = 0; ii < UNROLL; ii++) {
      IndexType li = linearIndex + blockDim.x * gridDim.x * ii;
      if (li < totalElements) {
        outputs[li] = static_cast<accscalar_t>(src[ii]) * scale *
                      static_cast<accscalar_t>(msk[ii]);
      }
    }
  }
}

template <typename scalar_t, typename accscalar_t, typename IndexType>
void apex_fused_dropout_cuda(scalar_t const *inputs, scalar_t *outputs,
                             uint8_t *mask, IndexType totalElements,
                             accscalar_t p) {
  auto gen = at::cuda::detail::getDefaultCUDAGenerator();

  int block_size = 256;
  dim3 dim_block(block_size);
  dim3 grid((totalElements + block_size - 1) / block_size);
  unsigned int blocks_per_sm =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor /
      block_size;
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()
                            ->multiProcessorCount *
                        blocks_per_sm,
                    grid.x);

  // number of times random will be generated per thread, to offset philox
  // counter in the random state
  int64_t counter_offset =
      ((totalElements - 1) / (block_size * grid.x * UNROLL) + 1) * UNROLL;
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
#ifdef OLD_GENERATOR
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(counter_offset);
#else
    std::lock_guard<std::mutex> lock(gen.mutex());
    rng_engine_inputs =
        at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_engine_inputs(
            counter_offset);
#endif
  }

  apex_fused_dropout_kernel<scalar_t, accscalar_t, IndexType>
      <<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
          inputs, outputs, mask, totalElements, p, rng_engine_inputs);
  C10_CUDA_CHECK(cudaGetLastError());
}

template <typename scalar_t, typename accscalar_t, typename IndexType>
void apex_dropout_add_cuda(scalar_t const *inputs, scalar_t const *add_inputs,
                           scalar_t *outputs, uint8_t *mask,
                           IndexType totalElements, accscalar_t p) {
  auto gen = at::cuda::detail::getDefaultCUDAGenerator();

  int block_size = 256;
  dim3 dim_block(block_size);
  dim3 grid((totalElements + block_size - 1) / block_size);
  unsigned int blocks_per_sm =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor /
      block_size;
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()
                            ->multiProcessorCount *
                        blocks_per_sm,
                    grid.x);

  // number of times random will be generated per thread, to offset philox
  // counter in the random state
  int64_t counter_offset =
      ((totalElements - 1) / (block_size * grid.x * UNROLL) + 1) * UNROLL;
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
#ifdef OLD_GENERATOR
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(counter_offset);
#else
    std::lock_guard<std::mutex> lock(gen.mutex());
    rng_engine_inputs =
        at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_engine_inputs(
            counter_offset);
#endif
  }

  apex_dropout_add_kernel<scalar_t, accscalar_t, IndexType>
      <<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
          inputs, add_inputs, outputs, mask, totalElements, p,
          rng_engine_inputs);
  C10_CUDA_CHECK(cudaGetLastError());
}

template <typename scalar_t, typename accscalar_t, typename IndexType>
void apex_add_cuda(scalar_t const *inputs, scalar_t const *add_inputs,
                   scalar_t *outputs, IndexType totalElements) {
  int block_size = 256;
  dim3 dim_block(block_size);
  dim3 grid((totalElements + block_size - 1) / block_size);
  unsigned int blocks_per_sm =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor /
      block_size;
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()
                            ->multiProcessorCount *
                        blocks_per_sm,
                    grid.x);

  apex_add_kernel<scalar_t, accscalar_t, IndexType>
      <<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
          inputs, add_inputs, outputs, totalElements);
  C10_CUDA_CHECK(cudaGetLastError());
}

template <typename scalar_t, typename accscalar_t, typename IndexType>
void apex_masked_scale_cuda(scalar_t const *inputs, scalar_t *outputs,
                            uint8_t const *mask, IndexType totalElements,
                            accscalar_t scale) {
  int block_size = 256;
  dim3 dim_block(block_size);
  dim3 grid((totalElements + block_size - 1) / block_size);
  unsigned int blocks_per_sm =
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor /
      block_size;
  grid.x = std::min((unsigned int)at::cuda::getCurrentDeviceProperties()
                            ->multiProcessorCount *
                        blocks_per_sm,
                    grid.x);

  apex_masked_scale_kernel<scalar_t, accscalar_t, IndexType>
      <<<grid, dim_block, 0, at::cuda::getCurrentCUDAStream()>>>(
          inputs, outputs, mask, totalElements, scale);
  C10_CUDA_CHECK(cudaGetLastError());
}
