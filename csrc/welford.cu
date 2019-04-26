#include <iostream>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "type_shim.h"


__device__ __forceinline__ int lastpow2(int n)
{
  int out = 1 << (31 - __clz(n));
  if(n == out)
    out >>= 1;
  return out;
}

__host__ __forceinline__ int h_next_pow2(unsigned int n) {
    n--;
    n |= (n >>  1);
    n |= (n >>  2);
    n |= (n >>  4);
    n |= (n >>  8);
    n |= (n >> 16);
    return ++n;
}

__host__ __forceinline__ int h_last_pow2(unsigned int n) {
    n |= (n >>  1);
    n |= (n >>  2);
    n |= (n >>  4);
    n |= (n >>  8);
    n |= (n >> 16);
    return n - (n >> 1);
}


#define WARP_SIZE 32

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val)
{
  #pragma unroll
  for(int i = WARP_SIZE/2; i > 0; i >>= 1)
    val = val + __shfl_down_sync(0xffffffff, val, i);
  return val;
}

template<typename T>
__device__ __forceinline__ T reduce_block(T *x, T val)
{
  int tid = threadIdx.y*blockDim.x + threadIdx.x;
  int blockSize = blockDim.x * blockDim.y;

  if (blockSize > 32) {
    val = warp_reduce_sum(val);
    if (tid % WARP_SIZE == 0)
      x[tid/WARP_SIZE] = val;

    __syncthreads();

    val = (tid < blockSize / WARP_SIZE? x[tid%WARP_SIZE] : T(0));
  }

  if(tid/WARP_SIZE==0) val = warp_reduce_sum(val);

  return val;
}

#define ELEMENTS_PER_ITER 4 // enables concurrency within each thread to hide latency
#define ELEMENTS_PER_THREAD 16
#define OPTIMAL_TILE_W 32
#define MAX_H_BLOCK 128
#define MAX_BLOCK_SIZE 512

__host__ int div_ru(int x, int y) {
  return h_last_pow2(1 + (x-1)/y);
}

__host__ void flexible_launch_configs(
      const int reduction,
      const int stride,
      dim3 &block,
      dim3 &grid,
      const bool coop_flag = false) {
  int block_x = std::min(h_last_pow2(stride), OPTIMAL_TILE_W);
  int block_y = std::min(h_last_pow2(div_ru(reduction , ELEMENTS_PER_THREAD)),
                         MAX_BLOCK_SIZE / block_x);
  if (block_x * block_y != MAX_BLOCK_SIZE) {
    block_x = std::min(h_last_pow2(stride), MAX_BLOCK_SIZE / block_y);
  }

  int grid_x = div_ru(stride, block_x);
  int grid_y = std::min(div_ru(reduction, block_y * ELEMENTS_PER_THREAD), MAX_H_BLOCK);
  if (coop_flag) {
    // it's not worth having a grid reduction if the reduction dimension is not big enough
    grid_y = grid_y < 8 ? 1 : grid_y;
  }

  block.x = block_x;
  block.y = block_y;
  block.z = 1;
  grid.x = grid_x;
  grid.y = grid_y;
  grid.z = 1;
}

template<typename T, typename C>
__device__ __forceinline__ void welford_merge_element(C& count,
                                                      T& mean,
                                                      T& m2n,
                                                      const C& num_new,
                                                      const T& mean_new,
                                                      const T& m2n_new) {
      T factor = T(1.0) / max(1, (count + num_new));
      T delta0 = mean - mean_new;
      mean = (mean_new * num_new + mean * count) * factor;
      m2n += m2n_new + delta0 * delta0 * num_new * count * factor;
      count += num_new;
}

template<typename T>
__device__ __forceinline__ void warp_reduce_mean_m2n(T &mean, T &m2n, int &num)
{
  #pragma unroll
  for(int i = WARP_SIZE/2; i > 0; i >>= 1) {
    auto num_new = __shfl_down_sync(0xffffffff, num, i);
    auto mean_new = __shfl_down_sync(0xffffffff, mean, i);
    auto m2n_new = __shfl_down_sync(0xffffffff, m2n, i);
    welford_merge_element(num, mean, m2n, num_new, mean_new, m2n_new);
  }
}

template <typename T>
__device__ void welford_reduce_mean_m2n(
      T* __restrict__ x,
      int* __restrict__ count,
      T &mean,
      T &m2n,
      int &num,
      int block_size,
      int thread_id)
{
  int lane = thread_id % WARP_SIZE;
  int wid = thread_id / WARP_SIZE;

  if (block_size > 32) {
    warp_reduce_mean_m2n(mean, m2n, num);
    if (lane == 0) {
      x[wid*2] = mean;
      x[wid*2+1] = m2n;
      count[wid] = num;
    }
    __syncthreads();

    if (wid == 0) {
      mean = (thread_id < block_size / WARP_SIZE)? x[lane*2] : T(0);
      m2n = (thread_id < block_size / WARP_SIZE)? x[lane*2+1] : T(0);
      num = (thread_id < block_size / WARP_SIZE)? count[lane] : int(0);
    }
  }

  if (wid==0) warp_reduce_mean_m2n(mean, m2n, num);

  return;
}

// return spatial size for NC+ Tensors
__host__ int get_tensor_spatial_size(const at::Tensor& input)
{
  auto space_size = input.size(2);
  for (int i = 3; i < input.ndimension(); i++) {
    space_size *= input.size(i);
  }
  return space_size;
}

// promote accumulation scalar type. promote half to float.
__host__ at::ScalarType promote_scalartype(const at::Tensor& input)
{
  return input.scalar_type() == at::ScalarType::Half ?
           at::ScalarType::Float : input.scalar_type();
}

// return single element size, optional accumulation type promotion.
__host__ size_t get_element_data_size(const at::Tensor& input, bool accumulation = false)
{
  auto scalar_type = accumulation ? promote_scalartype(input) : input.scalar_type();
  return at::elementSize(scalar_type);
}

template<typename T, typename C>
__device__ __forceinline__ void welford_merge_block_vertical(C& count,
                                                             T& mean,
                                                             T& m2n,
                                                             C* shmem_count,
                                                             T* shmem_mean,
                                                             T* shmem_m2n) {
  // write to shared memory
  auto address_base = threadIdx.x + threadIdx.y * blockDim.x;
  shmem_mean[address_base] = mean;
  shmem_m2n[address_base] = m2n;
  shmem_count[address_base] = count;

#pragma unroll
  for (int offset = blockDim.y/2; offset > 0; offset >>= 1) {
    __syncthreads();
    if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
      auto address = address_base + offset * blockDim.x;
      // read shared memory back to register for reduction
      auto num_new = shmem_count[address];
      auto mean_new = shmem_mean[address];
      auto m2n_new = shmem_m2n[address];

      welford_merge_element(count, mean, m2n, num_new, mean_new, m2n_new);

      // last write is not necessary
      shmem_mean[address_base] = mean;
      shmem_m2n[address_base] = m2n;
      shmem_count[address_base] = count;
    }
  }
}

template<typename T>
__device__ __forceinline__ void merge_block_vertical(T& sum_dy,
                                                     T& sum_dy_xmu,
                                                     T* shmem_sum_dy,
                                                     T* shmem_sum_dy_xmu) {
  // write to shared memory
  auto address_base = threadIdx.x + threadIdx.y * blockDim.x;
  shmem_sum_dy[address_base] = sum_dy;
  shmem_sum_dy_xmu[address_base] = sum_dy_xmu;

#pragma unroll
  for (int offset = blockDim.y/2; offset > 0; offset >>= 1) {
    __syncthreads();
    if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
      auto address = address_base + offset * blockDim.x;

      sum_dy += shmem_sum_dy[address];
      sum_dy_xmu += shmem_sum_dy_xmu[address];

      // last write is not necessary
      shmem_sum_dy[address_base] = sum_dy;
      shmem_sum_dy_xmu[address_base] = sum_dy_xmu;
    }
  }
}


// welford kernel calculating mean/biased_variance/unbiased_variance
template <typename scalar_t, typename accscalar_t, typename outscalar_t>
__global__ void welford_kernel(
      const scalar_t* __restrict__ input,
      outscalar_t* __restrict__ out_mean,
      outscalar_t* __restrict__ out_var_biased,
      const int bs,
      const int fs,
      const int ss) {
  int block_size = blockDim.x * blockDim.y;
  int count = 0;
  accscalar_t x_mean = accscalar_t(0);
  accscalar_t m_2_n = accscalar_t(0);

  int thread_id = threadIdx.y*blockDim.x + threadIdx.x;

  for (int batch_id = threadIdx.y; batch_id < bs; batch_id += blockDim.y) {
    int input_base = blockIdx.x*ss + batch_id*ss*fs;
    // sequential welford
    for (int offset = threadIdx.x; offset < ss ; offset += blockDim.x) {
      count++;
      auto x_n = static_cast<accscalar_t>(input[offset+input_base]);
      auto d = x_n - x_mean;
      x_mean += d / count;
      m_2_n += d * (x_n - x_mean);
    }
  }

  static __shared__ int s_mem[160];
  accscalar_t* s_mem_ac = (accscalar_t*) &s_mem[32];

  welford_reduce_mean_m2n<accscalar_t>(s_mem_ac, s_mem, x_mean, m_2_n, count, block_size, thread_id);

  if (thread_id == 0) {
    out_mean[blockIdx.x] = static_cast<outscalar_t>(x_mean);
    out_var_biased[blockIdx.x] = static_cast<outscalar_t>(m_2_n/count);
  }
}

// elementwise BN kernel
template <typename scalar_t, typename accscalar_t, typename layerscalar_t>
__global__ void batchnorm_forward_kernel(
      const scalar_t* __restrict__ input,
      const accscalar_t* __restrict__ mean,
      const accscalar_t* __restrict__ inv_std,
      const layerscalar_t* __restrict__ weight,
      const layerscalar_t* __restrict__ shift,
      scalar_t* __restrict__ out,
      const int ss,
      const int bs) {
  auto m_c = mean[blockIdx.x];
  auto inv_std_c = inv_std[blockIdx.x];
  auto w_c = weight == NULL ? accscalar_t(1.0) : static_cast<accscalar_t>(weight[blockIdx.x]);
  auto s_c = shift == NULL ? accscalar_t(0.0) : static_cast<accscalar_t>(shift[blockIdx.x]);

  for (int batch_offset = blockIdx.y*blockDim.y + threadIdx.y; batch_offset < bs; batch_offset += gridDim.y*blockDim.y) {
    int address_base = blockIdx.x*ss + batch_offset*gridDim.x*ss;
    for (int offset = threadIdx.x + blockIdx.z*blockDim.x; offset < ss ; offset+= gridDim.z*blockDim.x) {
      out[address_base+offset] = static_cast<scalar_t>(w_c * (static_cast<accscalar_t>(input[address_base+offset]) - m_c ) * inv_std_c + s_c);
    }
  }
}

// Backward BN kernel, calculates grad_bias, grad_weight as well as intermediate
// results to calculating grad_input.
// Breaking the grad_input to two step to support sync BN, which requires all
// reduce of the intermediate results across processes.
template <typename scalar_t, typename accscalar_t, typename layerscalar_t>
__global__ void reduce_bn_kernel(
      const scalar_t* __restrict__ input,
      const scalar_t* __restrict__ grad_output,
      const accscalar_t* __restrict__ mean,
      const accscalar_t* __restrict__ inv_std,
      accscalar_t* __restrict__ mean_dy,
      accscalar_t* __restrict__ mean_dy_xmu,
      layerscalar_t* __restrict__ grad_weight,
      layerscalar_t* __restrict__ grad_bias,
      const int bs,
      const int fs,
      const int ss) {
  static __shared__ int s_mem[64];
  int total_item_num = bs * ss;

  int thread_id = threadIdx.y*blockDim.x + threadIdx.x;

  auto r_mean = mean[blockIdx.x];
  auto factor = inv_std[blockIdx.x];

  // Kahan sum
  accscalar_t sum_dy = 0.0;
  accscalar_t sum_dy_xmu = 0.0;
  accscalar_t sum_dy_c = 0.0;
  accscalar_t sum_dy_xmu_c = 0.0;
  for (int batch_id = threadIdx.y; batch_id < bs; batch_id += blockDim.y) {
    int input_base = blockIdx.x*ss + batch_id*ss*fs;
    for (int offset = threadIdx.x; offset < ss ; offset += blockDim.x) {
      auto e_grad = static_cast<accscalar_t>(grad_output[offset+input_base]);
      auto e_input = static_cast<accscalar_t>(input[offset+input_base]);
      // calculating sum_dy
      auto sum_dy_y = e_grad - sum_dy_c;
      auto sum_dy_t = sum_dy + sum_dy_y;
      sum_dy_c = (sum_dy_t - sum_dy) - sum_dy_y;
      sum_dy = sum_dy_t;

      // calculating sum_dy_xmu
      auto sum_dy_xmu_y = e_grad * (e_input - r_mean) - sum_dy_xmu_c;
      auto sum_dy_xmu_t = sum_dy_xmu + sum_dy_xmu_y;
      sum_dy_xmu_c = (sum_dy_xmu_t - sum_dy_xmu) - sum_dy_xmu_y;
      sum_dy_xmu = sum_dy_xmu_t;
    }
  }

  sum_dy = reduce_block((accscalar_t*)s_mem, sum_dy);
  __syncthreads();
  sum_dy_xmu = reduce_block((accscalar_t*)s_mem, sum_dy_xmu);

  if (thread_id == 0) {
    if (grad_bias != NULL) {
      grad_bias[blockIdx.x] = static_cast<layerscalar_t>(sum_dy);
    }
    if (grad_weight != NULL) {
      grad_weight[blockIdx.x] = static_cast<layerscalar_t>(sum_dy_xmu * factor);
    }
    mean_dy[blockIdx.x] = sum_dy / total_item_num;
    mean_dy_xmu[blockIdx.x] = sum_dy_xmu / total_item_num;
  }
}

// elementwise backward BN kernel
template <typename scalar_t, typename accscalar_t, typename layerscalar_t>
__global__ void batchnorm_backward_kernel(
      const scalar_t* __restrict__ grad_output,
      const scalar_t* __restrict__ input,
      const accscalar_t* __restrict__ mean,
      const accscalar_t* __restrict__ inv_std,
      const layerscalar_t* __restrict__ weight,
      const accscalar_t* __restrict__ mean_dy,
      const accscalar_t* __restrict__ mean_dy_xmu,
      scalar_t* __restrict__ grad_input,
      const int ss,
      const int bs) {
  auto m_c = static_cast<accscalar_t>(mean[blockIdx.x]);
  auto m_dy_c = static_cast<accscalar_t>(mean_dy[blockIdx.x]);
  auto factor_1_c = inv_std[blockIdx.x];
  auto factor_2_c = (weight == NULL ? accscalar_t(1.0) : static_cast<accscalar_t>(weight[blockIdx.x])) * factor_1_c;
  factor_1_c = factor_1_c * factor_1_c * mean_dy_xmu[blockIdx.x];

  for (int batch_offset = blockIdx.y*blockDim.y+threadIdx.y; batch_offset < bs; batch_offset += gridDim.y*blockDim.y) {
    int address_base = blockIdx.x*ss + batch_offset*gridDim.x*ss;
    for (int offset = threadIdx.x + blockIdx.z*blockDim.x; offset < ss ; offset+= gridDim.z*blockDim.x) {
      grad_input[address_base+offset] = (static_cast<accscalar_t>(grad_output[address_base+offset]) - m_dy_c - (static_cast<accscalar_t>(input[address_base+offset]) - m_c) * factor_1_c) * factor_2_c;
    }
  }
}

// welford kernel for c last tensor calculating mean/biased_variance/unbiased_variance
template
   <typename scalar_t,
    typename accscalar_t,
    typename outscalar_t,
    int PARALLEL_LOADS>
__global__ void
welford_kernel_c_last(
      const scalar_t* __restrict__ input,
      outscalar_t* __restrict__ out_mean,
      outscalar_t* __restrict__ out_var_biased,
      volatile accscalar_t* staging_data,
      int* semaphores,
      const int reduction_size,
      const int stride) {
  // hide latency with concurrency
  accscalar_t x_mean[PARALLEL_LOADS];
  accscalar_t m_2_n[PARALLEL_LOADS];
  int count[PARALLEL_LOADS];

#pragma unroll
  for (int i = 0; i < PARALLEL_LOADS; i++) {
    x_mean[i] = accscalar_t(0);
    m_2_n[i] = accscalar_t(0);
    count[i] = accscalar_t(0);
  }
  // tensor dimension (m,c)

  // loop along m dimension
  int inner_loop_stride = blockDim.y * gridDim.y;

  // offset along m dimension
  int m_offset = blockIdx.y * blockDim.y + threadIdx.y;
  int c_offset = blockIdx.x * blockDim.x + threadIdx.x;

  int loop_count = 1 + (reduction_size - 1) / (inner_loop_stride * PARALLEL_LOADS);
  int address_base = m_offset * stride + c_offset;
  int address_increment = inner_loop_stride * stride;

  for (int i = 0; i < loop_count; i++) {
    accscalar_t x_math[PARALLEL_LOADS];
    accscalar_t x_count_inv[PARALLEL_LOADS];
    accscalar_t is_valid[PARALLEL_LOADS];

    // load multiple data in
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      if (c_offset < stride && m_offset < reduction_size) {
        x_math[j] = input[address_base];
        count[j]++;
        x_count_inv[j] = accscalar_t(1) / count[j];
        is_valid[j] = accscalar_t(1);
      } else {
        x_math[j] = accscalar_t(0);
        x_count_inv[j] = accscalar_t(0);
        is_valid[j] = accscalar_t(0);
      }
      m_offset += inner_loop_stride;
      address_base += address_increment;
    }

    // calculate mean/m2n with welford
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      accscalar_t delta0 = x_math[j] - x_mean[j];
      x_mean[j] += delta0 * x_count_inv[j];
      accscalar_t delta1 = x_math[j] - x_mean[j];
      m_2_n[j] += delta0 * delta1 * is_valid[j];
    }
  }

  // thread reduction to accumulate mean/m_2_n/count between PARALLEL_LOADS
#pragma unroll
  for (int j = 1; j < PARALLEL_LOADS; j++) {
    welford_merge_element(count[0], x_mean[0], m_2_n[0], count[j], x_mean[j], m_2_n[j]);
  }

  // release x_mean / m_2_n
  auto mean_th = x_mean[0];
  auto m2_th = m_2_n[0];
  auto count_th = count[0];

  // block-wise reduction with shared memory (since reduction cannot be done within a warp)
  static __shared__ accscalar_t shmem_mean[MAX_BLOCK_SIZE];
  static __shared__ accscalar_t shmem_m2n[MAX_BLOCK_SIZE];
  static __shared__ int shmem_count[MAX_BLOCK_SIZE];

  welford_merge_block_vertical(count_th, mean_th, m2_th, shmem_count, shmem_mean, shmem_m2n);

  // grid reduction if needed (coop launch used at the first place)
  if (gridDim.y > 1) {
    volatile accscalar_t* staging_mean = staging_data;
    volatile accscalar_t* staging_m2n = &staging_data[stride*gridDim.y];
    volatile int* staging_count = reinterpret_cast<volatile int*>(&staging_m2n[stride*gridDim.y]);

    address_base = c_offset + blockIdx.y * stride;
    // write data to staging_data;
    if (threadIdx.y == 0 && c_offset < stride) {
      staging_mean[address_base] = mean_th;
      staging_m2n[address_base] = m2_th;
      staging_count[address_base] = count_th;
    }

    __threadfence();
    __syncthreads(); // ensuring writes to staging_ is visible to all blocks

    __shared__ bool is_last_block_done;
    // mark block done
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int old = atomicAdd(&semaphores[blockIdx.x], 1);
      is_last_block_done = (old == (gridDim.y-1));
    }

    __syncthreads();

    // check that all data is now available in global memory
    if (is_last_block_done) {
      count_th = 0;
      mean_th = accscalar_t(0.0);
      m2_th = accscalar_t(0.0);

      for (int y = threadIdx.y; y < gridDim.y; y += blockDim.y) {
        address_base = c_offset + y * stride;
        int num_new = c_offset < stride ? staging_count[address_base] : 0;
        accscalar_t mean_new = c_offset < stride ? staging_mean[address_base] : accscalar_t(0.0);
        accscalar_t m2n_new = c_offset < stride ? staging_m2n[address_base] : accscalar_t(0.0);

        welford_merge_element(count_th, mean_th, m2_th, num_new, mean_new, m2n_new);
      }

      welford_merge_block_vertical(count_th, mean_th, m2_th, shmem_count, shmem_mean, shmem_m2n);
      if (threadIdx.y == 0 && c_offset < stride) {
        out_mean[c_offset] = static_cast<outscalar_t>(mean_th);
        out_var_biased[c_offset] = static_cast<outscalar_t>(m2_th / count_th);
      }
    }
  } else {
    if (blockIdx.y == 0 && threadIdx.y == 0 && c_offset < stride) {
      out_mean[c_offset] = static_cast<outscalar_t>(mean_th);
      out_var_biased[c_offset] = static_cast<outscalar_t>(m2_th / count_th);
    }
  }
}

// parallel welford kernel to further reduce mean / biased_var
// into mean / unbiased_var / inv_std across multiple processes.
template <typename scalar_t>
__global__ void welford_kernel_parallel(
      const scalar_t* __restrict__ mean,
      const scalar_t* __restrict__ var_biased,
      scalar_t* __restrict__ out_mean,
      scalar_t* __restrict__ out_var,
      scalar_t* __restrict__ inv_std,
      const int world_size,
      const int feature_size,
      const float eps,
      const int numel) {

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < feature_size; i += gridDim.x * blockDim.x) {
    // load data;
    int address = i;
    scalar_t x_mean = 0;
    scalar_t m_2_n = 0;
    int count = 0;
    for (int j = 0; j < world_size; j++) {
      welford_merge_element(count, x_mean, m_2_n, numel, mean[address], var_biased[address]*numel);
      address += feature_size;
    }
    out_mean[i] = x_mean;
    out_var[i] = m_2_n/ (count - 1);
    inv_std[i] = scalar_t(1) / sqrt(m_2_n/count + eps);
  }
}

// elementwise BN kernel
template <
    typename scalar_t,
    typename accscalar_t,
    typename layerscalar_t,
    int PARALLEL_LOADS>
__global__ void batchnorm_forward_c_last_kernel(
      const scalar_t* __restrict__ input,
      const accscalar_t* __restrict__ mean,
      const accscalar_t* __restrict__ inv_std,
      const layerscalar_t* __restrict__ weight,
      const layerscalar_t* __restrict__ shift,
      scalar_t* __restrict__ out,
      const int reduction_size,
      const int stride) {
  // tensor dimension (m,c)
  // loop along m dimension
  int inner_loop_stride = blockDim.y * gridDim.y;

  // offset along m dimension
  int m_offset = blockIdx.y * blockDim.y + threadIdx.y;
  int c_offset = blockIdx.x * blockDim.x + threadIdx.x;

  auto m_c = mean[c_offset];
  auto inv_std_c = static_cast<accscalar_t>(inv_std[c_offset]);
  auto w_c = weight == NULL ? accscalar_t(1.0) : static_cast<accscalar_t>(weight[c_offset]);
  auto s_c = shift == NULL ? accscalar_t(0.0) : static_cast<accscalar_t>(shift[c_offset]);

  int loop_count = 1 + (reduction_size - 1) / (inner_loop_stride * PARALLEL_LOADS);
  int address_base = m_offset * stride + c_offset;
  int address_increment = inner_loop_stride * stride;

  for (int i = 0; i < loop_count; i++) {
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      if (c_offset < stride && m_offset < reduction_size) {
        out[address_base] = static_cast<scalar_t>(
            w_c * (static_cast<accscalar_t>(input[address_base]) - m_c ) * inv_std_c + s_c
          );
      }
      m_offset += inner_loop_stride;
      address_base += address_increment;
    }
  }
}

// batchnorm backward kernel for c last tensor
template
   <typename scalar_t,
    typename accscalar_t,
    typename layerscalar_t,
    int PARALLEL_LOADS>
__global__ void reduce_bn_c_last_kernel(
      const scalar_t* __restrict__ input,
      const scalar_t* __restrict__ grad_output,
      const accscalar_t* __restrict__ mean,
      const accscalar_t* __restrict__ inv_std,
      accscalar_t* __restrict__ mean_dy,
      accscalar_t* __restrict__ mean_dy_xmu,
      layerscalar_t* __restrict__ grad_weight,
      layerscalar_t* __restrict__ grad_bias,
      volatile accscalar_t* staging_data,
      int* semaphores,
      const int reduction_size,
      const int stride) {

  // hide latency with concurrency
  accscalar_t sum_dy[PARALLEL_LOADS];
  accscalar_t sum_dy_xmu[PARALLEL_LOADS];

#pragma unroll
  for (int i = 0; i < PARALLEL_LOADS; i++) {
    sum_dy[i] = accscalar_t(0);
    sum_dy_xmu[i] = accscalar_t(0);
  }
  // tensor dimension (m,c)

  // loop along m dimension
  int inner_loop_stride = blockDim.y * gridDim.y;

  // offset along m dimension
  int m_offset = blockIdx.y * blockDim.y + threadIdx.y;
  int c_offset = blockIdx.x * blockDim.x + threadIdx.x;

  int loop_count = 1 + (reduction_size - 1) / (inner_loop_stride * PARALLEL_LOADS);
  int address_base = m_offset * stride + c_offset;
  int address_increment = inner_loop_stride * stride;

  auto r_mean = mean[c_offset];
  auto factor = inv_std[c_offset];

  for (int i = 0; i < loop_count; i++) {
    accscalar_t x_input[PARALLEL_LOADS];
    accscalar_t x_grad_output[PARALLEL_LOADS];

    // load multiple data in
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      if (c_offset < stride && m_offset < reduction_size) {
        x_input[j] = input[address_base];
        x_grad_output[j] = grad_output[address_base];
      } else {
        x_input[j] = accscalar_t(0);
        x_grad_output[j] = accscalar_t(0);
      }
      m_offset += inner_loop_stride;
      address_base += address_increment;
    }

    // calculate sum_dy / sum_dy_xmu
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      sum_dy[j] += x_grad_output[j];
      sum_dy_xmu[j] += x_grad_output[j] * (x_input[j] - r_mean);
    }
  }

  // thread reduction to accumulate sum_dy / sum_dy_xmu between PARALLEL_LOADS
#pragma unroll
  for (int j = 1; j < PARALLEL_LOADS; j++) {
    sum_dy[0] += sum_dy[j];
    sum_dy_xmu[0] += sum_dy_xmu[j];
  }

  // release array of registers
  auto sum_dy_th = sum_dy[0];
  auto sum_dy_xmu_th = sum_dy_xmu[0];

  // block-wise reduction with shared memory (since reduction cannot be done within a warp)
  static __shared__ accscalar_t shmem_sum_dy[MAX_BLOCK_SIZE];
  static __shared__ accscalar_t shmem_sum_dy_xmu[MAX_BLOCK_SIZE];

  merge_block_vertical(sum_dy_th, sum_dy_xmu_th, shmem_sum_dy, shmem_sum_dy_xmu);

  // grid reduction if needed (coop launch used at the first place)
  if (gridDim.y > 1) {
    volatile accscalar_t* staging_sum_dy = staging_data;
    volatile accscalar_t* staging_sum_dy_xmu = &staging_data[stride*gridDim.y];

    address_base = c_offset + blockIdx.y * stride;
    // write data to staging_data;
    if (threadIdx.y == 0 && c_offset < stride) {
      staging_sum_dy[address_base] = sum_dy_th;
      staging_sum_dy_xmu[address_base] = sum_dy_xmu_th;
    }

    __threadfence();
    __syncthreads(); // ensuring writes to staging_ is visible to all blocks

    __shared__ bool is_last_block_done;
    // mark block done
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      int old = atomicAdd(&semaphores[blockIdx.x], 1);
      is_last_block_done = (old == (gridDim.y-1));
    }

    __syncthreads();

    // check that all data is now available in global memory
    if (is_last_block_done) {
      sum_dy_th = accscalar_t(0.0);
      sum_dy_xmu_th = accscalar_t(0.0);

      for (int y = threadIdx.y; y < gridDim.y; y += blockDim.y) {
        address_base = c_offset + y * stride;
        sum_dy_th += (c_offset < stride ? staging_sum_dy[address_base] : accscalar_t(0.0));
        sum_dy_xmu_th += (c_offset < stride ? staging_sum_dy_xmu[address_base] : accscalar_t(0.0));
      }

      merge_block_vertical(sum_dy_th, sum_dy_xmu_th, shmem_sum_dy, shmem_sum_dy_xmu);
      if (threadIdx.y == 0 && c_offset < stride) {
        if (grad_bias != NULL) {
          grad_bias[c_offset] = static_cast<layerscalar_t>(sum_dy_th);
        }
        if (grad_weight != NULL) {
          grad_weight[c_offset] = static_cast<layerscalar_t>(sum_dy_xmu_th * factor);
        }
        mean_dy[c_offset] = sum_dy_th / reduction_size;
        mean_dy_xmu[c_offset] = sum_dy_xmu_th / reduction_size;
      }
    }
  } else {
    if (blockIdx.y == 0 && threadIdx.y == 0 && c_offset < stride) {
      if (grad_bias != NULL) {
        grad_bias[c_offset] = static_cast<layerscalar_t>(sum_dy_th);
      }
      if (grad_weight != NULL) {
        grad_weight[c_offset] = static_cast<layerscalar_t>(sum_dy_xmu_th * factor);
      }
      mean_dy[c_offset] = sum_dy_th / reduction_size;
      mean_dy_xmu[c_offset] = sum_dy_xmu_th / reduction_size;
    }
  }
}

// elementwise BN kernel
template <
    typename scalar_t,
    typename accscalar_t,
    typename layerscalar_t,
    int PARALLEL_LOADS>
__global__ void batchnorm_backward_c_last_kernel(
      const scalar_t* __restrict__ grad_output,
      const scalar_t* __restrict__ input,
      const accscalar_t* __restrict__ mean,
      const accscalar_t* __restrict__ inv_std,
      const layerscalar_t* __restrict__ weight,
      const accscalar_t* __restrict__ mean_dy,
      const accscalar_t* __restrict__ mean_dy_xmu,
      scalar_t* __restrict__ grad_input,
      const int reduction_size,
      const int stride) {
  // tensor dimension (m,c)
  // loop along m dimension
  int inner_loop_stride = blockDim.y * gridDim.y;

  // offset along m dimension
  int m_offset = blockIdx.y * blockDim.y + threadIdx.y;
  int c_offset = blockIdx.x * blockDim.x + threadIdx.x;

  auto m_c = mean[c_offset];
  auto m_dy_c = mean_dy[c_offset];
  auto factor_1_c = inv_std[c_offset];
  auto factor_2_c = (weight == NULL? accscalar_t(1.0) : static_cast<accscalar_t>(weight[c_offset])) * factor_1_c;
  factor_1_c = factor_1_c * factor_1_c * mean_dy_xmu[c_offset];

  int loop_count = 1 + (reduction_size - 1) / (inner_loop_stride * PARALLEL_LOADS);
  int address_base = m_offset * stride + c_offset;
  int address_increment = inner_loop_stride * stride;

  for (int i = 0; i < loop_count; i++) {
#pragma unroll
    for (int j = 0; j < PARALLEL_LOADS; j++) {
      if (c_offset < stride && m_offset < reduction_size) {
        grad_input[address_base] = static_cast<scalar_t>(
            (static_cast<accscalar_t>(grad_output[address_base]) - m_dy_c -
            (static_cast<accscalar_t>(input[address_base]) - m_c) * factor_1_c)
            * factor_2_c);
      }
      m_offset += inner_loop_stride;
      address_base += address_increment;
    }
  }
}

std::vector<at::Tensor> welford_mean_var_CUDA(const at::Tensor input) {
  const auto batch_size = input.size(0);
  const auto feature_size = input.size(1);

  auto space_size = get_tensor_spatial_size(input);
  auto scalar_type = promote_scalartype(input);

  at::Tensor out_var_biased = at::empty({feature_size}, input.options().dtype(scalar_type));
  at::Tensor out_mean = at::empty({feature_size}, input.options().dtype(scalar_type));

  int block_y = min(h_last_pow2(batch_size), int(MAX_BLOCK_SIZE / 32));
  int block_x = max(1, min(MAX_BLOCK_SIZE / block_y, h_last_pow2(space_size)));
  const dim3 block(block_x, block_y);
  const dim3 grid(feature_size);

  auto stream = at::cuda::getCurrentCUDAStream();

  {
    using namespace at;
    DISPATCH_FLOAT_AND_HALF(input.scalar_type(), 0, "welford_mean_var_kernel",
      using accscalar_t = at::acc_type<scalar_t_0, true>;
      welford_kernel<scalar_t_0, accscalar_t, accscalar_t><<<grid, block, 0, stream>>>(
          input.data<scalar_t_0>(),
          out_mean.data<accscalar_t>(),
          out_var_biased.data<accscalar_t>(),
          batch_size,
          feature_size,
          space_size);
    );
  }

  return {out_mean, out_var_biased};
}

at::Tensor batchnorm_forward_CUDA(
    const at::Tensor input,
    const at::Tensor mean,
    const at::Tensor inv_std,
    const at::optional<at::Tensor> weight,
    const at::optional<at::Tensor> shift) {
  const auto batch_size = input.size(0);
  const auto feature_size = input.size(1);
  at::Tensor out = at::empty_like(input);

  auto space_size = get_tensor_spatial_size(input);

  int block_x = max(32, min(MAX_BLOCK_SIZE, h_last_pow2(space_size)/4));
  int block_y = max(1, min(MAX_BLOCK_SIZE/block_x, h_last_pow2(batch_size)/4));
  const dim3 block(block_x, block_y);
  int grid_z = max(1, min(65535, h_last_pow2(space_size)/4/block_x));
  int batch_group_size = max(1, min(65535, h_last_pow2(batch_size)/block_y));
  const dim3 grid(feature_size, batch_group_size, grid_z);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (input.scalar_type() == at::ScalarType::Half
      && weight.has_value() &&
      weight.value().scalar_type() == at::ScalarType::Float) {
    using namespace at;
    DISPATCH_FLOAT_AND_HALF(input.scalar_type(), 0, "batchnorm_forward",
      using accscalar_t = at::acc_type<scalar_t_0, true>;
      batchnorm_forward_kernel<scalar_t_0, accscalar_t, accscalar_t><<<grid, block, 0, stream>>>(
          input.data<scalar_t_0>(),
          mean.data<accscalar_t>(),
          inv_std.data<accscalar_t>(),
          weight.has_value() ? weight.value().data<accscalar_t>() : NULL,
          shift.has_value() ? shift.value().data<accscalar_t>() : NULL,
          out.data<scalar_t_0>(),
          space_size,
          batch_size);
    );
  } else {
    if (weight.has_value()) {
      AT_CHECK(input.scalar_type() == weight.value().scalar_type(),
          "input.scalar_type() is not supported with weight.scalar_type()");
    }
    using namespace at;
    DISPATCH_FLOAT_AND_HALF(input.scalar_type(), 0, "batchnorm_forward",
      using accscalar_t = at::acc_type<scalar_t_0, true>;
      batchnorm_forward_kernel<scalar_t_0, accscalar_t, scalar_t_0><<<grid, block, 0, stream>>>(
          input.data<scalar_t_0>(),
          mean.data<accscalar_t>(),
          inv_std.data<accscalar_t>(),
          weight.has_value() ? weight.value().data<scalar_t_0>() : NULL,
          shift.has_value() ? shift.value().data<scalar_t_0>() : NULL,
          out.data<scalar_t_0>(),
          space_size,
          batch_size);
    );
  }
  return out;
}

std::vector<at::Tensor> reduce_bn_CUDA(
    const at::Tensor grad_output,
    const at::Tensor input,
    const at::Tensor mean,
    const at::Tensor inv_std,
    const at::optional<at::Tensor> weight)
{
  const auto batch_size = input.size(0);
  const auto feature_size = input.size(1);

  auto scalar_type = promote_scalartype(input);

  at::Tensor mean_dy = at::empty({feature_size}, mean.options());
  at::Tensor mean_dy_xmu = at::empty({feature_size}, mean.options());

  at::Tensor grad_weight;
  at::Tensor grad_bias;
  if (weight.has_value()) {
    grad_weight = at::empty({feature_size}, weight.value().options());
    grad_bias = at::empty({feature_size}, weight.value().options());
  } else {
    grad_weight = at::empty({0}, mean.options());
    grad_bias = at::empty({0}, mean.options());
  }

  auto space_size = get_tensor_spatial_size(input);

  int block_y = min(h_last_pow2(batch_size), int(MAX_BLOCK_SIZE/ 32));
  int block_x = max(1, min(MAX_BLOCK_SIZE/ block_y, h_last_pow2(space_size)));
  const dim3 block(block_x, block_y);
  const dim3 grid(feature_size);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (input.scalar_type() == at::ScalarType::Half
      && weight.has_value() &&
      weight.value().scalar_type() == at::ScalarType::Float) {
    using namespace at;
    DISPATCH_FLOAT_AND_HALF(input.scalar_type(), 0, "batchnorm_backward_reduce",
      using accscalar_t = at::acc_type<scalar_t_0, true>;
      reduce_bn_kernel<scalar_t_0, accscalar_t, accscalar_t><<<grid, block, 0, stream>>>(
          input.data<scalar_t_0>(),
          grad_output.data<scalar_t_0>(),
          mean.data<accscalar_t>(),
          inv_std.data<accscalar_t>(),
          mean_dy.data<accscalar_t>(),
          mean_dy_xmu.data<accscalar_t>(),
          weight.has_value() ? grad_weight.data<accscalar_t>() : NULL,
          weight.has_value() ? grad_bias.data<accscalar_t>() : NULL,
          batch_size,
          feature_size,
          space_size);
    );
  } else {
    if (weight.has_value()) {
        AT_CHECK(input.scalar_type() == weight.value().scalar_type(),
            "input.scalar_type() is not supported with weight.scalar_type()");
    }
    using namespace at;
    DISPATCH_FLOAT_AND_HALF(input.scalar_type(), 0, "batchnorm_backward_reduce",
      using accscalar_t = at::acc_type<scalar_t_0, true>;
      reduce_bn_kernel<scalar_t_0, accscalar_t, scalar_t_0><<<grid, block, 0, stream>>>(
          input.data<scalar_t_0>(),
          grad_output.data<scalar_t_0>(),
          mean.data<accscalar_t>(),
          inv_std.data<accscalar_t>(),
          mean_dy.data<accscalar_t>(),
          mean_dy_xmu.data<accscalar_t>(),
          weight.has_value() ? grad_weight.data<scalar_t_0>() : NULL,
          weight.has_value() ? grad_bias.data<scalar_t_0>() : NULL,
          batch_size,
          feature_size,
          space_size);
    );
  }

  return {mean_dy, mean_dy_xmu, grad_weight, grad_bias};
}

at::Tensor batchnorm_backward_CUDA(
    const at::Tensor grad_output,
    const at::Tensor input,
    const at::Tensor mean,
    const at::Tensor inv_std,
    const at::optional<at::Tensor> weight,
    const at::Tensor mean_dy,
    const at::Tensor mean_dy_xmu) {
  const auto batch_size = input.size(0);
  const auto feature_size = input.size(1);

  at::Tensor grad_input = at::empty_like(input);

  auto space_size = get_tensor_spatial_size(input);

  int block_x = max(32, min(MAX_BLOCK_SIZE, h_last_pow2(space_size)/4));
  int block_y = max(1, min(MAX_BLOCK_SIZE/block_x, h_last_pow2(batch_size)/4));
  const dim3 block(block_x, block_y);
  int grid_z = max(1, min(65535, h_last_pow2(space_size)/4/block_x));
  int batch_group_size = max(1, min(65535, h_last_pow2(batch_size)/block_y));
  const dim3 grid(feature_size, batch_group_size, grid_z);

  auto stream = at::cuda::getCurrentCUDAStream();

  if (input.scalar_type() == at::ScalarType::Half
      && weight.has_value() &&
      weight.value().scalar_type() == at::ScalarType::Float) {
    using namespace at;
    DISPATCH_FLOAT_AND_HALF(input.scalar_type(), 0, "batchnorm_backward",
      using accscalar_t = at::acc_type<scalar_t_0, true>;
      batchnorm_backward_kernel<scalar_t_0, accscalar_t, accscalar_t><<<grid, block, 0, stream>>>(
          grad_output.data<scalar_t_0>(),
          input.data<scalar_t_0>(),
          mean.data<accscalar_t>(),
          inv_std.data<accscalar_t>(),
          weight.has_value() ? weight.value().data<accscalar_t>() : NULL,
          mean_dy.data<accscalar_t>(),
          mean_dy_xmu.data<accscalar_t>(),
          grad_input.data<scalar_t_0>(),
          space_size,
          batch_size);
    );
  } else {
    if (weight.has_value()) {
      AT_CHECK(input.scalar_type() == weight.value().scalar_type(),
          "input.scalar_type() is not supported with weight.scalar_type()");
    }
    using namespace at;
    DISPATCH_FLOAT_AND_HALF(input.scalar_type(), 0, "batchnorm_backward",
      using accscalar_t = at::acc_type<scalar_t_0, true>;
      batchnorm_backward_kernel<scalar_t_0, accscalar_t, scalar_t_0><<<grid, block, 0, stream>>>(
          grad_output.data<scalar_t_0>(),
          input.data<scalar_t_0>(),
          mean.data<accscalar_t>(),
          inv_std.data<accscalar_t>(),
          weight.has_value() ? weight.value().data<scalar_t_0>() : NULL,
          mean_dy.data<accscalar_t>(),
          mean_dy_xmu.data<accscalar_t>(),
          grad_input.data<scalar_t_0>(),
          space_size,
          batch_size);
    );
  }

  return grad_input;
}

std::vector<at::Tensor> welford_parallel_CUDA(const at::Tensor mean_feature_nodes,
                                              const at::Tensor var_biased,
                                              int numel,
                                              const float eps) {
  const auto world_size = mean_feature_nodes.size(0);
  const auto feature_size = mean_feature_nodes.size(1);

  at::Tensor out_var = at::empty({feature_size}, var_biased.options());
  at::Tensor inv_std = at::empty_like(out_var);
  at::Tensor out_mean = at::empty_like(out_var);

  // TODO(jie): tile this for memory coalescing!
  const int block = std::min(h_last_pow2(feature_size), MAX_BLOCK_SIZE);
  const int grid = std::max<int>(1, feature_size / block);

  auto stream = at::cuda::getCurrentCUDAStream();

  {
    using namespace at;
    DISPATCH_FLOAT_AND_HALF(mean_feature_nodes.scalar_type(), 0, "welford_parallel_kernel",
      welford_kernel_parallel<scalar_t_0><<<grid, block, 0, stream>>>(
          mean_feature_nodes.data<scalar_t_0>(),
          var_biased.data<scalar_t_0>(),
          out_mean.data<scalar_t_0>(),
          out_var.data<scalar_t_0>(),
          inv_std.data<scalar_t_0>(),
          world_size,
          feature_size,
          eps,
          numel);
    );
  }

  return {out_mean, out_var, inv_std};
}

std::vector<at::Tensor> welford_mean_var_c_last_CUDA(const at::Tensor input) {
  const auto stride = input.size(input.ndimension()-1);
  const auto reduction_size = input.numel() / stride;

  auto scalar_type = promote_scalartype(input);
  auto option = input.options().dtype(scalar_type);

  at::Tensor out_var_biased = at::empty({stride}, option);
  at::Tensor out_mean = at::empty({stride}, option);

  dim3 block;
  dim3 grid;
  flexible_launch_configs(reduction_size, stride, block, grid, true);

  at::Tensor staging_data;
  at::Tensor semaphores;
  if (grid.y > 1) {
    staging_data = at::empty({4*stride*grid.y}, option);
    semaphores = at::zeros({grid.x}, input.options().dtype(at::kInt));
  }

  auto stream = at::cuda::getCurrentCUDAStream();

  {
    using namespace at;
    DISPATCH_FLOAT_AND_HALF(input.scalar_type(), 0, "welford_mean_var_c_last",
      using accscalar_t = at::acc_type<scalar_t_0, true>;
      accscalar_t* staging_data_ptr = grid.y > 1 ? staging_data.data<accscalar_t>() : nullptr;
      int* semaphores_ptr = grid.y > 1 ? semaphores.data<int>() : nullptr;
      welford_kernel_c_last<scalar_t_0, accscalar_t, accscalar_t, ELEMENTS_PER_ITER>
          <<<grid, block, 0, stream>>>(
          input.data<scalar_t_0>(),
          out_mean.data<accscalar_t>(),
          out_var_biased.data<accscalar_t>(),
          staging_data_ptr,
          semaphores_ptr,
          reduction_size,
          stride);
    );
  }

  return {out_mean, out_var_biased};
}

at::Tensor batchnorm_forward_c_last_CUDA(
    const at::Tensor input,
    const at::Tensor mean,
    const at::Tensor inv_std,
    const at::optional<at::Tensor> weight,
    const at::optional<at::Tensor> shift) {
  const auto stride = input.size(input.ndimension()-1);
  const auto reduction_size = input.numel() / stride;

  at::Tensor out = at::empty_like(input);

  dim3 block;
  dim3 grid;
  flexible_launch_configs(reduction_size, stride, block, grid);

  auto stream = at::cuda::getCurrentCUDAStream();

  if (input.scalar_type() == at::ScalarType::Half
      && weight.has_value() && weight.value().scalar_type() == at::ScalarType::Float) {
    using namespace at;
    DISPATCH_FLOAT_AND_HALF(input.scalar_type(), 0, "batchnorm_forward",
      using accscalar_t = at::acc_type<scalar_t_0, true>;
      batchnorm_forward_c_last_kernel<scalar_t_0, accscalar_t, accscalar_t, ELEMENTS_PER_ITER>
          <<<grid, block, 0, stream>>>(
          input.data<scalar_t_0>(),
          mean.data<accscalar_t>(),
          inv_std.data<accscalar_t>(),
          weight.has_value() ? weight.value().data<accscalar_t>() : NULL,
          shift.has_value() ? shift.value().data<accscalar_t>(): NULL,
          out.data<scalar_t_0>(),
          reduction_size,
          stride);
    );
  } else {
    if (weight.has_value()) {
      AT_CHECK(input.scalar_type() == weight.value().scalar_type(),
          "input.scalar_type() is not supported with weight.scalar_type()");
    }
    using namespace at;
    DISPATCH_FLOAT_AND_HALF(input.scalar_type(), 0, "batchnorm_forward",
      using accscalar_t = at::acc_type<scalar_t_0, true>;
      batchnorm_forward_c_last_kernel<scalar_t_0, accscalar_t, scalar_t_0, ELEMENTS_PER_ITER>
          <<<grid, block, 0, stream>>>(
          input.data<scalar_t_0>(),
          mean.data<accscalar_t>(),
          inv_std.data<accscalar_t>(),
          weight.has_value() ? weight.value().data<scalar_t_0>() : NULL,
          shift.has_value() ? shift.value().data<scalar_t_0>(): NULL,
          out.data<scalar_t_0>(),
          reduction_size,
          stride);
    );
  }
  return out;
}

std::vector<at::Tensor> reduce_bn_c_last_CUDA(
    const at::Tensor grad_output,
    const at::Tensor input,
    const at::Tensor mean,
    const at::Tensor inv_std,
    const at::optional<at::Tensor> weight) {
  const auto stride = input.size(input.ndimension()-1);
  const auto reduction_size = input.numel() / stride;

  at::Tensor mean_dy = at::empty({stride}, mean.options());
  at::Tensor mean_dy_xmu = at::empty({stride}, mean.options());

  at::Tensor grad_weight;
  at::Tensor grad_bias;
  if (weight.has_value()) {
    grad_weight = at::empty({stride}, weight.value().options());
    grad_bias = at::empty({stride}, weight.value().options());
  } else {
    // because I cannot return an uninitialized at::Tensor
    grad_weight = at::empty({0}, mean.options());
    grad_bias = at::empty({0}, mean.options());
  }

  dim3 block;
  dim3 grid;
  flexible_launch_configs(reduction_size, stride, block, grid, true);

  at::Tensor staging_data;
  at::Tensor semaphores;
  if (grid.y > 1) {
    staging_data = at::empty({2*stride*grid.y}, mean.options());
    semaphores = at::zeros({grid.x}, input.options().dtype(at::kInt));
  }
  auto stream = at::cuda::getCurrentCUDAStream();

  if (input.scalar_type() == at::ScalarType::Half
      && weight.has_value()
      && weight.value().scalar_type() == at::ScalarType::Float) {
    using namespace at;
    DISPATCH_FLOAT_AND_HALF(input.scalar_type(), 0, "batchnorm_backward_reduce",
      using accscalar_t = at::acc_type<scalar_t_0, true>;
      accscalar_t* staging_data_ptr = grid.y > 1 ? staging_data.data<accscalar_t>() : nullptr;
      int* semaphores_ptr = grid.y > 1 ? semaphores.data<int>() : nullptr;
      reduce_bn_c_last_kernel<scalar_t_0, accscalar_t, accscalar_t, ELEMENTS_PER_ITER>
          <<<grid, block, 0, stream>>>(
          input.data<scalar_t_0>(),
          grad_output.data<scalar_t_0>(),
          mean.data<accscalar_t>(),
          inv_std.data<accscalar_t>(),
          mean_dy.data<accscalar_t>(),
          mean_dy_xmu.data<accscalar_t>(),
          weight.has_value() ? grad_weight.data<accscalar_t>() : NULL,
          weight.has_value() ?grad_bias.data<accscalar_t>() : NULL,
          staging_data_ptr,
          semaphores_ptr,
          reduction_size,
          stride);
    );
  } else {
    if (weight.has_value()) {
      AT_CHECK(input.scalar_type() == weight.value().scalar_type(),
          "input.scalar_type() is not supported with weight.scalar_type()");
    }
    using namespace at;
    DISPATCH_FLOAT_AND_HALF(input.scalar_type(), 0, "batchnorm_backward_reduce",
      using accscalar_t = at::acc_type<scalar_t_0, true>;
      accscalar_t* staging_data_ptr = grid.y > 1 ? staging_data.data<accscalar_t>() : nullptr;
      int* semaphores_ptr = grid.y > 1 ? semaphores.data<int>() : nullptr;
      reduce_bn_c_last_kernel<scalar_t_0, accscalar_t, scalar_t_0, ELEMENTS_PER_ITER>
          <<<grid, block, 0, stream>>>(
          input.data<scalar_t_0>(),
          grad_output.data<scalar_t_0>(),
          mean.data<accscalar_t>(),
          inv_std.data<accscalar_t>(),
          mean_dy.data<accscalar_t>(),
          mean_dy_xmu.data<accscalar_t>(),
          weight.has_value() ? grad_weight.data<scalar_t_0>() : NULL,
          weight.has_value() ?grad_bias.data<scalar_t_0>() : NULL,
          staging_data_ptr,
          semaphores_ptr,
          reduction_size,
          stride);
    );
  }

  return {mean_dy, mean_dy_xmu, grad_weight, grad_bias};
}

at::Tensor batchnorm_backward_c_last_CUDA(
    const at::Tensor grad_output,
    const at::Tensor input,
    const at::Tensor mean,
    const at::Tensor inv_std,
    const at::optional<at::Tensor> weight,
    const at::Tensor mean_dy,
    const at::Tensor mean_dy_xmu) {
  const auto stride = input.size(input.ndimension()-1);
  const auto reduction_size = input.numel() / stride;

  at::Tensor grad_input = at::empty_like(input);

  dim3 block;
  dim3 grid;
  flexible_launch_configs(reduction_size, stride, block, grid);

  auto stream = at::cuda::getCurrentCUDAStream();

  if (input.scalar_type() == at::ScalarType::Half
      && weight.has_value() && weight.value().scalar_type() == at::ScalarType::Float) {
    using namespace at;
    DISPATCH_FLOAT_AND_HALF(input.scalar_type(), 0, "batchnorm_forward",
      using accscalar_t = at::acc_type<scalar_t_0, true>;
      batchnorm_backward_c_last_kernel<scalar_t_0, accscalar_t, accscalar_t, ELEMENTS_PER_ITER>
          <<<grid, block, 0, stream>>>(
          grad_output.data<scalar_t_0>(),
          input.data<scalar_t_0>(),
          mean.data<accscalar_t>(),
          inv_std.data<accscalar_t>(),
          weight.has_value() ? weight.value().data<accscalar_t>() : NULL,
          mean_dy.data<accscalar_t>(),
          mean_dy_xmu.data<accscalar_t>(),
          grad_input.data<scalar_t_0>(),
          reduction_size,
          stride);
    );
  } else {
    if (weight.has_value()) {
      AT_CHECK(input.scalar_type() == weight.value().scalar_type(),
          "input.scalar_type() is not supported with weight.scalar_type()");
    }
    using namespace at;
    DISPATCH_FLOAT_AND_HALF(input.scalar_type(), 0, "batchnorm_forward",
      using accscalar_t = at::acc_type<scalar_t_0, true>;
      batchnorm_backward_c_last_kernel<scalar_t_0, accscalar_t, scalar_t_0, ELEMENTS_PER_ITER>
          <<<grid, block, 0, stream>>>(
          grad_output.data<scalar_t_0>(),
          input.data<scalar_t_0>(),
          mean.data<accscalar_t>(),
          inv_std.data<accscalar_t>(),
          weight.has_value() ? weight.value().data<scalar_t_0>() : NULL,
          mean_dy.data<accscalar_t>(),
          mean_dy_xmu.data<accscalar_t>(),
          grad_input.data<scalar_t_0>(),
          reduction_size,
          stride);
    );
  }
 
  return grad_input;
}
