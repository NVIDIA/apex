#include <iostream>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>


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

template <typename scalar_t, typename accscalar_t>
__global__ void welford_kernel(
      const scalar_t* __restrict__ input,
      scalar_t* __restrict__ out_mean,
      scalar_t* __restrict__ out_var,
      scalar_t* __restrict__ out_var_biased,
      const int bs,
      const int fs,
      const int ss) {
  extern __shared__ int s_mem[];
  int block_size = blockDim.x * blockDim.y;

  float *mean_l = (float*) s_mem;
  float *m2n_l = (float*) &(s_mem[block_size]);
  int *num_item_l = (int*) &(s_mem[block_size*2]);

  int count = 0;
  float x_mean = 0;
  float m_2_n = 0;
  int input_base = blockIdx.x*ss + threadIdx.y*ss*fs;
  int thread_id = threadIdx.y*blockDim.x + threadIdx.x;

  // sequential welford
  for (int offset = threadIdx.x; offset < ss ; offset+= blockDim.x) {
    count++;
    auto x_n = static_cast<accscalar_t>(input[offset+input_base]);
    auto x_mean_new = x_mean + (x_n - x_mean) / count;
    m_2_n = m_2_n + (x_n - x_mean_new) * (x_n - x_mean);
    x_mean = x_mean_new;
  }

  // allow idle thread to write to shared memory
  mean_l[thread_id] = x_mean;
  m2n_l[thread_id] = m_2_n;
  num_item_l[thread_id] = count;
  __syncthreads();
  
  // parallel reduce with interleaved threads
  // TODO(jie): unroll this?
  // TODO(jie): maybe I should pad the blockDim.y to power of 2?
  for (int offset = lastpow2(block_size); offset > 0; offset>>=1) {
    // excluding idle threads, because /0!
    if (thread_id < offset && thread_id + offset < block_size && threadIdx.x < ss) {
      auto count = num_item_l[thread_id];
      auto val = mean_l[thread_id];
      auto count2 = num_item_l[thread_id+offset];
      auto val2 = mean_l[thread_id+offset];

      mean_l[thread_id] = (val * count + val2 * count2) / (count + count2);
      val = val - val2;
      m2n_l[thread_id] += m2n_l[thread_id + offset] + val*val*count*count2/(count+count2);
      num_item_l[thread_id] = count + count2;
    }
    __syncthreads();
  }

  if (thread_id == 0) {
    out_mean[blockIdx.x] = static_cast<scalar_t>(mean_l[0]);
    out_var[blockIdx.x] = static_cast<scalar_t>(m2n_l[0]/(num_item_l[0]-1));
    out_var_biased[blockIdx.x] = static_cast<scalar_t>(m2n_l[0]/num_item_l[0]);
  }
}

template <typename scalar_t, typename accscalar_t>
__global__ void batchnorm_forward_kernel(
      const scalar_t* __restrict__ input,
      const scalar_t* __restrict__ mean,
      const scalar_t* __restrict__ var,
      const scalar_t* __restrict__ weight,
      const scalar_t* __restrict__ shift,
      scalar_t* __restrict__ out,
      const int ss,
      const float eps) {
  int address_base = blockIdx.x*ss + blockIdx.y*gridDim.x*ss;

  auto m_c = static_cast<accscalar_t>(mean[blockIdx.x]);
  auto var_c = sqrt(static_cast<accscalar_t>(var[blockIdx.x]) + eps);
  auto w_c = static_cast<accscalar_t>(weight[blockIdx.x]);
  auto s_c = static_cast<accscalar_t>(shift[blockIdx.x]);

  // sequential welford
  for (int offset = threadIdx.x; offset < ss ; offset+= blockDim.x) {
    out[address_base+offset] = static_cast<scalar_t>((static_cast<accscalar_t>(input[address_base+offset]) - m_c ) / var_c * w_c + s_c);
  }
}

template <typename scalar_t, typename accscalar_t>
__global__ void reduce_bn_kernel(
      const scalar_t* __restrict__ input,
      const scalar_t* __restrict__ grad_output,
      const scalar_t* __restrict__ mean,
      const scalar_t* __restrict__ var,
      accscalar_t* __restrict__ mean_dy,
      accscalar_t* __restrict__ mean_dy_xmu,
      scalar_t* __restrict__ grad_weight,
      scalar_t* __restrict__ grad_bias,
      const int bs,
      const int fs,
      const int ss,
      const float eps) {
  extern __shared__ int s_mem[];
  int block_size = blockDim.x * blockDim.y;

  float *sum_dy_l = (float*) s_mem;
  float *sum_dy_xmu_l = (float*) &(s_mem[block_size]);
  int total_item_num = bs * ss;

  float s_dy = 0.0;
  float s_dy_xmu = 0.0;
  int input_base = blockIdx.x*ss + threadIdx.y*ss*fs;
  int thread_id = threadIdx.y*blockDim.x + threadIdx.x;

  auto r_mean = static_cast<accscalar_t>(mean[blockIdx.x]);
  auto factor = 1.0 / sqrt(static_cast<accscalar_t>(var[blockIdx.x]) + eps);

  // sequential welford
  for (int offset = threadIdx.x; offset < ss ; offset+= blockDim.x) {
    auto e_grad = static_cast<accscalar_t>(grad_output[offset+input_base]);
    auto e_input = static_cast<accscalar_t>(input[offset+input_base]);
    s_dy += e_grad;
    s_dy_xmu += e_grad * (e_input - r_mean);
  }

  sum_dy_l[thread_id] = s_dy;
  sum_dy_xmu_l[thread_id] = s_dy_xmu;
  __syncthreads();
  
  // parallel reduce with interleaved threads
  // TODO(jie): unroll this?
  // TODO(jie): maybe I should pad the blockDim.y to power of 2?
  for (int offset = lastpow2(block_size); offset > 0; offset>>=1) {
    if (thread_id < offset && thread_id + offset < block_size) {
      sum_dy_l[thread_id] += sum_dy_l[thread_id+offset];
      sum_dy_xmu_l[thread_id] += sum_dy_xmu_l[thread_id+offset];
    }
    __syncthreads();
  }

  if (thread_id == 0) {
    grad_bias[blockIdx.x] = static_cast<scalar_t>(sum_dy_l[0]);
    grad_weight[blockIdx.x] = static_cast<scalar_t>(sum_dy_xmu_l[0] * factor);
    mean_dy[blockIdx.x] = sum_dy_l[0] / total_item_num;
    mean_dy_xmu[blockIdx.x] = sum_dy_xmu_l[0] / total_item_num;
  }
}

template <typename scalar_t, typename accscalar_t>
__global__ void batchnorm_backward_kernel(
      const scalar_t* __restrict__ grad_output,
      const scalar_t* __restrict__ input,
      const scalar_t* __restrict__ mean,
      const scalar_t* __restrict__ var,
      const scalar_t* __restrict__ weight,
      const scalar_t* __restrict__ mean_dy,
      const scalar_t* __restrict__ mean_dy_xmu,
      scalar_t* __restrict__ grad_input,
      const int ss,
      const float eps) {
  int address_base = blockIdx.x*ss + blockIdx.y*gridDim.x*ss;

  auto m_c = static_cast<accscalar_t>(mean[blockIdx.x]);
  auto m_dy_c = static_cast<accscalar_t>(mean_dy[blockIdx.x]);
  auto factor_1_c = static_cast<accscalar_t>(var[blockIdx.x]) + eps;
  auto factor_2_c = static_cast<accscalar_t>(weight[blockIdx.x]) / sqrt(factor_1_c);
  factor_1_c /= static_cast<accscalar_t>(mean_dy_xmu[blockIdx.x]);

  // sequential welford
  for (int offset = threadIdx.x; offset < ss ; offset+= blockDim.x) {
    grad_input[address_base+offset] = (static_cast<accscalar_t>(grad_output[address_base+offset]) - m_dy_c - (static_cast<accscalar_t>(input[address_base+offset]) - m_c) / factor_1_c) * factor_2_c;
  }
}

std::vector<at::Tensor> welford_mean_var_CUDA(const at::Tensor input) {
  const auto batch_size = input.size(0);
  const auto feature_size = input.size(1);

  auto space_size = input.size(2);
  for (int i = 3; i < input.ndimension(); i++) {
    space_size *= input.size(i);
  }

  at::Tensor out_var = at::empty_like(input).resize_({feature_size});
  at::Tensor out_var_biased = at::empty_like(out_var);
  at::Tensor out_mean = at::empty_like(out_var);

  int block_x = 16;
  const dim3 block(block_x, batch_size);
  const dim3 grid(feature_size);
  // save current mean, var, num_elements;
  int smem_size = batch_size * block_x * 3 * sizeof(int);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "welford_mean_var_kernel", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    welford_kernel<scalar_t, accscalar_t><<<grid, block, smem_size>>>(
        input.data<scalar_t>(),
        out_mean.data<scalar_t>(),
        out_var.data<scalar_t>(),
        out_var_biased.data<scalar_t>(),
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

  auto space_size = input.size(2);
  for (int i = 3; i < input.ndimension(); i++) {
    space_size *= input.size(i);
  }

  const dim3 block(512);
  // TODO(jie): should I do 1 block per feature?
  const dim3 grid(feature_size, batch_size);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "batchnorm_forward", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    batchnorm_forward_kernel<scalar_t, accscalar_t><<<grid, block>>>(
        input.data<scalar_t>(),
        mean.data<scalar_t>(),
        var.data<scalar_t>(),
        weight.data<scalar_t>(),
        shift.data<scalar_t>(),
        out.data<scalar_t>(),
        space_size,
        eps);
  }));
  
  return out;
}

std::vector<at::Tensor> reduce_bn_CUDA(
    const at::Tensor grad_output,
    const at::Tensor input,
    const at::Tensor mean,
    const at::Tensor var,
    const float eps) {
  const auto batch_size = input.size(0);
  const auto feature_size = input.size(1);

  at::Tensor mean_dy = at::empty_like(mean);
  at::Tensor mean_dy_xmu = at::empty_like(mean);
  at::Tensor grad_weight = at::empty_like(mean);
  at::Tensor grad_bias = at::empty_like(mean);

  auto space_size = input.size(2);
  for (int i = 3; i < input.ndimension(); i++) {
    space_size *= input.size(i);
  }

  int block_x = 16;
  const dim3 block(block_x, batch_size);
  const dim3 grid(feature_size);
  // shared memory used for reduce;
  int smem_size = batch_size * block_x * 2 * sizeof(int);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "batchnorm_forward", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    reduce_bn_kernel<scalar_t, accscalar_t><<<grid, block, smem_size>>>(
        input.data<scalar_t>(),
        grad_output.data<scalar_t>(),
        mean.data<scalar_t>(),
        var.data<scalar_t>(),
        mean_dy.data<accscalar_t>(),
        mean_dy_xmu.data<accscalar_t>(),
        grad_weight.data<scalar_t>(),
        grad_bias.data<scalar_t>(),
        batch_size,
        feature_size,
        space_size,
        eps);
  }));
  
  return {mean_dy, mean_dy_xmu, grad_weight, grad_bias};
}

at::Tensor batchnorm_backward_CUDA(
    const at::Tensor grad_output,
    const at::Tensor input,
    const at::Tensor mean,
    const at::Tensor var,
    const at::Tensor weight,
    const at::Tensor shift,
    const at::Tensor mean_dy,
    const at::Tensor mean_dy_xmu,
    const float eps) {
  const auto batch_size = input.size(0);
  const auto feature_size = input.size(1);

  at::Tensor grad_input = at::empty_like(input);

  auto space_size = input.size(2);
  for (int i = 3; i < input.ndimension(); i++) {
    space_size *= input.size(i);
  }

  const dim3 block(512);
  // TODO(jie): should I do 1 block per feature?
  const dim3 grid(feature_size, batch_size);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "batchnorm_forward", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    batchnorm_backward_kernel<scalar_t, accscalar_t><<<grid, block>>>(
        grad_output.data<scalar_t>(),
        input.data<scalar_t>(),
        mean.data<scalar_t>(),
        var.data<scalar_t>(),
        weight.data<scalar_t>(),
        mean_dy.data<scalar_t>(),
        mean_dy_xmu.data<scalar_t>(),
        grad_input.data<scalar_t>(),
        space_size,
        eps);
  }));
  
  return grad_input;
}

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
  extern __shared__ int s_mem[];
  int block_size = blockDim.x;

  float *mean_l = (float*) s_mem;
  float *m2n_l = (float*) &(s_mem[block_size]);
  int *num_item_l = (int*) &(s_mem[block_size*2]);

  int input_base = blockIdx.x*ns + threadIdx.x;
  int thread_id = threadIdx.x;

  // load data; 
  mean_l[thread_id] = static_cast<accscalar_t>(mean[input_base]);
  m2n_l[thread_id] = static_cast<accscalar_t>(var_biased[input_base]) * numel;
  num_item_l[thread_id] = numel;

  __syncthreads();
  
  // parallel reduce with interleaved threads
  // TODO(jie): unroll this?
  for (int offset = lastpow2(block_size); offset > 0; offset>>=1) {
    if (thread_id < offset && thread_id + offset < block_size) {
      auto count = num_item_l[thread_id];
      auto val = mean_l[thread_id];
      auto count2 = num_item_l[thread_id+offset];
      auto val2 = mean_l[thread_id+offset];

      mean_l[thread_id] = (val * count + val2 * count2) / (count + count2);
      val = val - val2;
      m2n_l[thread_id] += m2n_l[thread_id + offset] + val*val*count*count2/(count+count2);
      num_item_l[thread_id] = count + count2;
    }
    __syncthreads();
  }

  if (thread_id == 0) {
    out_mean[blockIdx.x] = static_cast<scalar_t>(mean_l[0]);
    out_var[blockIdx.x] = static_cast<scalar_t>(m2n_l[0]/(num_item_l[0]-1));
    out_var_biased[blockIdx.x] = static_cast<scalar_t>(m2n_l[0]/num_item_l[0]);
  }
}
  
std::vector<at::Tensor> welford_parallel_CUDA(const at::Tensor mean_feature_nodes, const at::Tensor var_biased, int numel) {
  const auto feature_size = mean_feature_nodes.size(0);
  const auto node_size = mean_feature_nodes.size(1);

  // TODO(jie): how to properly construct empty tensor with shape?
  at::Tensor out_var = at::empty_like(var_biased).resize_({feature_size});
  at::Tensor out_var_biased = at::empty_like(out_var);
  at::Tensor out_mean = at::empty_like(out_var);

  // TODO(jie): 
  const dim3 block(node_size);
  const dim3 grid(feature_size);
  // save current mean, var, num_elements;
  int smem_size = node_size * 3 * sizeof(int);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(mean_feature_nodes.type(), "welford_parallel_kernel", ([&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    welford_kernel_parallel<scalar_t, accscalar_t><<<grid, block, smem_size>>>(
        mean_feature_nodes.data<scalar_t>(),
        var_biased.data<scalar_t>(),
        out_mean.data<scalar_t>(),
        out_var.data<scalar_t>(),
        out_var_biased.data<scalar_t>(),
        node_size,
        feature_size,
        numel);
  }));

  return {out_mean, out_var, out_var_biased};
}
