#include <iostream>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>


#include <cuda.h>
#include <cuda_runtime.h>


#include <vector>


__device__ __forceinline__ int lastpow2(int n)
{
  int out = 1 << (31 - __clz(n));
  if(n == out) 
    out >>= 1;
  return out;
}

__host__ __forceinline__ int h_next_pow2(unsigned int n) {
    n |= (n >>  1);
    n |= (n >>  2);
    n |= (n >>  4);
    n |= (n >>  8);
    n |= (n >> 16);
    return n + 1;
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

#define TILE_W 32
#define MAX_BLOCK_SIZE 256

template<typename T>
__device__ __forceinline__ void warp_reduce_mean_m2n(T &mean, T &m2n, int &num)
{
  #pragma unroll
  for(int i = WARP_SIZE/2; i > 0; i >>= 1) {
    auto num_new = __shfl_down_sync(0xffffffff, num, i);
    auto mean_new = __shfl_down_sync(0xffffffff, mean, i);
    auto m2n_new = __shfl_down_sync(0xffffffff, m2n, i);
    if (num_new != 0) {
      auto dif_mean = mean - mean_new;
      mean = (mean_new * num_new + mean * num) / (num + num_new);
      m2n += m2n_new + dif_mean*dif_mean*num*num_new/(num_new+num);
      num += num_new;
    }
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
  return input.type().scalarType() == at::ScalarType::Half ?
           at::ScalarType::Float : input.type().scalarType();
}

// return single element size, optional accumulation type promotion.
__host__ size_t get_element_data_size(const at::Tensor& input, bool accumulation = false)
{
  auto scalar_type = accumulation ? promote_scalartype(input) : input.type().scalarType();
  return at::elementSize(scalar_type);
}


// welford kernel calculating mean/biased_variance/unbiased_variance
template <typename scalar_t, typename accscalar_t, typename outscalar_t>
__global__ void welford_kernel(
      const scalar_t* __restrict__ input,
      outscalar_t* __restrict__ out_mean,
      outscalar_t* __restrict__ out_var,
      outscalar_t* __restrict__ out_var_biased,
      const int bs,
      const int fs,
      const int ss) {
  static __shared__ int s_mem[160];
  int block_size = blockDim.x * blockDim.y;

  accscalar_t* s_mem_ac = (accscalar_t*) &s_mem[32];

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
      auto x_mean_new = x_mean + (x_n - x_mean) / count;
      m_2_n = m_2_n + (x_n - x_mean_new) * (x_n - x_mean);
      x_mean = x_mean_new;
    }
  }

  welford_reduce_mean_m2n<accscalar_t>(s_mem_ac, s_mem, x_mean, m_2_n, count, block_size, thread_id);

  if (thread_id == 0) {
    out_mean[blockIdx.x] = static_cast<outscalar_t>(x_mean);
    out_var[blockIdx.x] = static_cast<outscalar_t>(m_2_n/(count-1));
    out_var_biased[blockIdx.x] = static_cast<outscalar_t>(m_2_n/count);
  }
}

// elementwise BN kernel
template <typename scalar_t, typename accscalar_t, typename layerscalar_t>
__global__ void batchnorm_forward_kernel(
      const scalar_t* __restrict__ input,
      const accscalar_t* __restrict__ mean,
      const accscalar_t* __restrict__ var,
      const layerscalar_t* __restrict__ weight,
      const layerscalar_t* __restrict__ shift,
      scalar_t* __restrict__ out,
      const int ss,
      const float eps) {
  int address_base = blockIdx.x*ss + blockIdx.y*gridDim.x*ss;

  auto m_c = mean[blockIdx.x];
  auto inv_std_c = static_cast<accscalar_t>(rsqrt(var[blockIdx.x] + eps));
  auto w_c = static_cast<accscalar_t>(weight[blockIdx.x]);
  auto s_c = static_cast<accscalar_t>(shift[blockIdx.x]);

  for (int offset = threadIdx.x; offset < ss ; offset+= blockDim.x) {
    out[address_base+offset] = static_cast<scalar_t>(w_c * (static_cast<accscalar_t>(input[address_base+offset]) - m_c ) * inv_std_c + s_c);
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
      const accscalar_t* __restrict__ var,
      accscalar_t* __restrict__ mean_dy,
      accscalar_t* __restrict__ mean_dy_xmu,
      layerscalar_t* __restrict__ grad_weight,
      layerscalar_t* __restrict__ grad_bias,
      const int bs,
      const int fs,
      const int ss,
      const float eps) {
  static __shared__ int s_mem[64];
  int total_item_num = bs * ss;

  int thread_id = threadIdx.y*blockDim.x + threadIdx.x;

  auto r_mean = mean[blockIdx.x];
  auto factor = accscalar_t(1.0) / (accscalar_t)sqrt(var[blockIdx.x] + eps);

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
    grad_bias[blockIdx.x] = static_cast<layerscalar_t>(sum_dy);
    grad_weight[blockIdx.x] = static_cast<layerscalar_t>(sum_dy_xmu * factor);
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
      const accscalar_t* __restrict__ var,
      const layerscalar_t* __restrict__ weight,
      const accscalar_t* __restrict__ mean_dy,
      const accscalar_t* __restrict__ mean_dy_xmu,
      scalar_t* __restrict__ grad_input,
      const int ss,
      const float eps) {
  int address_base = blockIdx.x*ss + blockIdx.y*gridDim.x*ss;

  auto m_c = static_cast<accscalar_t>(mean[blockIdx.x]);
  auto m_dy_c = static_cast<accscalar_t>(mean_dy[blockIdx.x]);
  auto factor_1_c = static_cast<accscalar_t>(var[blockIdx.x]) + eps;
  auto factor_2_c = static_cast<accscalar_t>(weight[blockIdx.x]) / sqrt(factor_1_c);
  factor_1_c /= static_cast<accscalar_t>(mean_dy_xmu[blockIdx.x]);

  for (int offset = threadIdx.x; offset < ss ; offset+= blockDim.x) {
    grad_input[address_base+offset] = (static_cast<accscalar_t>(grad_output[address_base+offset]) - m_dy_c - (static_cast<accscalar_t>(input[address_base+offset]) - m_c) / factor_1_c) * factor_2_c;
  }
}

// parallel welford kernel to further reduce mean / biased_var / unbiased_var
// across multiple processes.
template <typename scalar_t, typename accscalar_t>
__global__ void welford_kernel_parallel(
      const scalar_t* __restrict__ mean,
      const scalar_t* __restrict__ var_biased,
      scalar_t* __restrict__ out_mean,
      scalar_t* __restrict__ out_var,
      scalar_t* __restrict__ out_var_biased,
      const int ns,
      const int fs,
      const int numel) {
  static __shared__ int s_mem[160];
  int block_size = blockDim.x;

  accscalar_t* s_mem_ac = (accscalar_t*) &s_mem[32];

  int input_base = blockIdx.x*ns + threadIdx.x;
  int thread_id = threadIdx.x;

  // load data; 
  auto x_mean = static_cast<accscalar_t>(mean[input_base]);
  auto m_2_n = static_cast<accscalar_t>(var_biased[input_base]) * numel;
  auto count = numel;

  __syncthreads();

  welford_reduce_mean_m2n<accscalar_t>(s_mem_ac, s_mem, x_mean, m_2_n, count, block_size, thread_id);

  if (thread_id == 0) {
    out_mean[blockIdx.x] = static_cast<scalar_t>(x_mean);
    out_var[blockIdx.x] = static_cast<scalar_t>(m_2_n/(count-1));
    out_var_biased[blockIdx.x] = static_cast<scalar_t>(m_2_n/count);
  }
}
  

std::vector<at::Tensor> welford_mean_var_CUDA(const at::Tensor input) {
  const auto batch_size = input.size(0);
  const auto feature_size = input.size(1);

  auto space_size = get_tensor_spatial_size(input);
  auto scalar_type = promote_scalartype(input);

  at::Tensor out_var = at::empty({feature_size}, input.options().dtype(scalar_type));
  at::Tensor out_var_biased = at::empty({feature_size}, input.options().dtype(scalar_type));
  at::Tensor out_mean = at::empty({feature_size}, input.options().dtype(scalar_type));

  int block_x = TILE_W;
  int block_y = min(h_last_pow2(batch_size), int(MAX_BLOCK_SIZE / block_x));
  const dim3 block(block_x, block_y);
  const dim3 grid(feature_size);

  // shared memory used for reduce on mean, var, num_elements;
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "welford_mean_var_kernel", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    welford_kernel<scalar_t, accscalar_t, accscalar_t><<<grid, block, 0, stream>>>(
        input.data<scalar_t>(),
        out_mean.data<accscalar_t>(),
        out_var.data<accscalar_t>(),
        out_var_biased.data<accscalar_t>(),
        batch_size,
        feature_size,
        space_size);
  }));

  return {out_mean, out_var, out_var_biased};
}

at::Tensor batchnorm_forward_CUDA(
    const at::Tensor input,
    const at::Tensor mean,
    const at::Tensor var,
    const at::Tensor weight,
    const at::Tensor shift,
    const float eps) {
  const auto batch_size = input.size(0);
  const auto feature_size = input.size(1);
  at::Tensor out = at::empty_like(input);

  auto space_size = get_tensor_spatial_size(input);

  int block = min(MAX_BLOCK_SIZE, h_next_pow2(space_size)/4);
  // TODO(jie): should I do 1 block per feature?
  const dim3 grid(feature_size, batch_size);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (input.type().scalarType() == at::ScalarType::Half && weight.type().scalarType() == at::ScalarType::Float) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "batchnorm_forward", ([&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      batchnorm_forward_kernel<scalar_t, accscalar_t, accscalar_t><<<grid, block, 0, stream>>>(
          input.data<scalar_t>(),
          mean.data<accscalar_t>(),
          var.data<accscalar_t>(),
          weight.data<accscalar_t>(),
          shift.data<accscalar_t>(),
          out.data<scalar_t>(),
          space_size,
          eps);
    }));
  } else {
    AT_CHECK(input.type().scalarType() == weight.type().scalarType(), "input.type().scalarType() is not supported with weight.type().scalarType()"); 
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "batchnorm_forward", ([&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      batchnorm_forward_kernel<scalar_t, accscalar_t, scalar_t><<<grid, block, 0, stream>>>(
          input.data<scalar_t>(),
          mean.data<accscalar_t>(),
          var.data<accscalar_t>(),
          weight.data<scalar_t>(),
          shift.data<scalar_t>(),
          out.data<scalar_t>(),
          space_size,
          eps);
    }));
  }
  return out;
}

std::vector<at::Tensor> reduce_bn_CUDA(
    const at::Tensor grad_output,
    const at::Tensor input,
    const at::Tensor mean,
    const at::Tensor var,
    const at::Tensor weight,
    const float eps)
{
  const auto batch_size = input.size(0);
  const auto feature_size = input.size(1);

  auto scalar_type = promote_scalartype(input);

  at::Tensor mean_dy = at::empty({feature_size}, mean.options());
  at::Tensor mean_dy_xmu = at::empty({feature_size}, mean.options());
  at::Tensor grad_weight = at::empty({feature_size}, weight.options());
  at::Tensor grad_bias = at::empty({feature_size}, weight.options());

  auto space_size = get_tensor_spatial_size(input);

  int block_x = TILE_W;
  int block_y = min(h_last_pow2(batch_size), int(MAX_BLOCK_SIZE / block_x));
  const dim3 block(block_x, block_y);
  const dim3 grid(feature_size);
  // shared memory used for reduce on sum_dy, sum_dy_xmu;
  auto stream = at::cuda::getCurrentCUDAStream();

  if (input.type().scalarType() == at::ScalarType::Half && weight.type().scalarType() == at::ScalarType::Float) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "batchnorm_backward_reduce", ([&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      reduce_bn_kernel<scalar_t, accscalar_t, accscalar_t><<<grid, block, 0, stream>>>(
          input.data<scalar_t>(),
          grad_output.data<scalar_t>(),
          mean.data<accscalar_t>(),
          var.data<accscalar_t>(),
          mean_dy.data<accscalar_t>(),
          mean_dy_xmu.data<accscalar_t>(),
          grad_weight.data<accscalar_t>(),
          grad_bias.data<accscalar_t>(),
          batch_size,
          feature_size,
          space_size,
          eps);
    }));
  } else {
    AT_CHECK(input.type().scalarType() == weight.type().scalarType(), "input.type().scalarType() is not supported with weight.type().scalarType()"); 
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "batchnorm_backward_reduce", ([&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      reduce_bn_kernel<scalar_t, accscalar_t, scalar_t><<<grid, block, 0, stream>>>(
          input.data<scalar_t>(),
          grad_output.data<scalar_t>(),
          mean.data<accscalar_t>(),
          var.data<accscalar_t>(),
          mean_dy.data<accscalar_t>(),
          mean_dy_xmu.data<accscalar_t>(),
          grad_weight.data<scalar_t>(),
          grad_bias.data<scalar_t>(),
          batch_size,
          feature_size,
          space_size,
          eps);
    }));
  }
  
  return {mean_dy, mean_dy_xmu, grad_weight, grad_bias};
}

at::Tensor batchnorm_backward_CUDA(
    const at::Tensor grad_output,
    const at::Tensor input,
    const at::Tensor mean,
    const at::Tensor var,
    const at::Tensor weight,
    const at::Tensor mean_dy,
    const at::Tensor mean_dy_xmu,
    const float eps) {
  const auto batch_size = input.size(0);
  const auto feature_size = input.size(1);

  at::Tensor grad_input = at::empty_like(input);

  auto space_size = get_tensor_spatial_size(input);

  int block = min(MAX_BLOCK_SIZE, h_next_pow2(space_size)/4);
  // TODO(jie): should I do 1 block per feature?
  const dim3 grid(feature_size, batch_size);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (input.type().scalarType() == at::ScalarType::Half && weight.type().scalarType() == at::ScalarType::Float) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "batchnorm_backward", ([&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      batchnorm_backward_kernel<scalar_t, accscalar_t, accscalar_t><<<grid, block, 0, stream>>>(
          grad_output.data<scalar_t>(),
          input.data<scalar_t>(),
          mean.data<accscalar_t>(),
          var.data<accscalar_t>(),
          weight.data<accscalar_t>(),
          mean_dy.data<accscalar_t>(),
          mean_dy_xmu.data<accscalar_t>(),
          grad_input.data<scalar_t>(),
          space_size,
          eps);
    }));
  } else {
    AT_CHECK(input.type().scalarType() == weight.type().scalarType(), "input.type().scalarType() is not supported with weight.type().scalarType()"); 
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "batchnorm_backward", ([&] {
      using accscalar_t = at::acc_type<scalar_t, true>;
      batchnorm_backward_kernel<scalar_t, accscalar_t, scalar_t><<<grid, block, 0, stream>>>(
          grad_output.data<scalar_t>(),
          input.data<scalar_t>(),
          mean.data<accscalar_t>(),
          var.data<accscalar_t>(),
          weight.data<scalar_t>(),
          mean_dy.data<accscalar_t>(),
          mean_dy_xmu.data<accscalar_t>(),
          grad_input.data<scalar_t>(),
          space_size,
          eps);
    }));
  }
  
  return grad_input;
}

std::vector<at::Tensor> welford_parallel_CUDA(const at::Tensor mean_feature_nodes, const at::Tensor var_biased, int numel) {
  const auto feature_size = mean_feature_nodes.size(0);
  const auto world_size = mean_feature_nodes.size(1);

  at::Tensor out_var = at::empty({feature_size}, var_biased.options());
  at::Tensor out_var_biased = at::empty_like(out_var);
  at::Tensor out_mean = at::empty_like(out_var);

  // TODO(jie): tile this for memory coalescing!
  const dim3 block(world_size);
  const dim3 grid(feature_size);
  // shared memory used for reduce on mean, var, num_elements;
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(mean_feature_nodes.type(), "welford_parallel_kernel", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    welford_kernel_parallel<scalar_t, accscalar_t><<<grid, block, 0, stream>>>(
        mean_feature_nodes.data<scalar_t>(),
        var_biased.data<scalar_t>(),
        out_mean.data<scalar_t>(),
        out_var.data<scalar_t>(),
        out_var_biased.data<scalar_t>(),
        world_size,
        feature_size,
        numel);
  }));

  return {out_mean, out_var, out_var_biased};
}
