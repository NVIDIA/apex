//#define USE_HALF_SPECIALIZATIONS

#ifdef USE_HALF_SPECIALIZATIONS
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif

#include <ATen/ATen.h>
#include "ATen/cuda/CUDAContext.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<typename U> __device__
void cuWelfordOnlineSum(
  const U curr,
  U& mu,
  U& sigma2,
  U& count)
{
  count = count + U(1);
  U delta = curr - mu;
  U lmean = mu + delta / count;
  mu = lmean;
  U delta2 = curr - lmean;
  sigma2 = sigma2 + delta * delta2;
}

template<typename U> __device__
void cuChanOnlineSum(
  const U muB,
  const U sigma2B,
  const U countB,
  U& mu,
  U& sigma2,
  U& count)
{
  U delta = muB - mu;
  U nA = count;
  U nB = countB;
  count = count + countB;
  U nX = count;
  if (nX > U(0)) {
    nA = nA / nX;
    nB = nB / nX;
    mu = nA*mu + nB*muB;
    sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX;
  } else {
    mu = U(0);
    sigma2 = U(0);
  }
}

template<typename T, typename U> __device__
void cuWelfordMuSigma2(
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  U& mu,
  U& sigma2,
  U* buf) 
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  U count = U(0);
  mu= U(0);
  sigma2 = U(0);
  int i1 = blockIdx.y;
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const T* lvals = vals + i1*n2;
    int l = 4*thrx;
    for (;  l+3 < n2;  l+=4*numx) {
      for (int k = 0;  k < 4;  ++k) {
        U curr = static_cast<U>(lvals[l+k]);
        cuWelfordOnlineSum<U>(curr,mu,sigma2,count);
      }
    }
    for (;  l < n2;  ++l) {
      U curr = static_cast<U>(lvals[l]);
      cuWelfordOnlineSum<U>(curr,mu,sigma2,count);
    }
    // intra-warp reductions
    for (int l = 0;  l <= 4;  ++l) {
      int srcLaneB = (threadIdx.x+(1<<l))&31;
      U muB = __shfl_sync(0xffffffff, mu, srcLaneB);
      U countB = __shfl_sync(0xffffffff, count, srcLaneB);
      U sigma2B = __shfl_sync(0xffffffff, sigma2, srcLaneB);
      cuChanOnlineSum<U>(muB,sigma2B,countB,mu,sigma2,count);
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      U* ubuf = (U*)buf;
      U* ibuf = (U*)(ubuf + blockDim.y);
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2*wrt_y] = mu;
          ubuf[2*wrt_y+1] = sigma2;
          ibuf[wrt_y] = count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          U muB = ubuf[2*threadIdx.y];
          U sigma2B = ubuf[2*threadIdx.y+1];
          U countB = ibuf[threadIdx.y];
          cuChanOnlineSum<U>(muB,sigma2B,countB,mu,sigma2,count);
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        ubuf[0] = mu;
        ubuf[1] = sigma2;
      }
      __syncthreads();
      mu = ubuf[0];
      sigma2 = ubuf[1]/U(n2);
      // don't care about final value of count, we know count == n2
    } else {
      mu = __shfl_sync(0xffffffff, mu, 0);
      sigma2 = __shfl_sync(0xffffffff, sigma2/U(n2), 0);
    }
  }
}

template<> __device__
void cuWelfordMuSigma2(
  const at::Half* __restrict__ vals,
  const int n1,
  const int n2,
  float& mu,
  float& sigma2,
  float* buf) 
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  float count = 0.0f;
  mu= float(0);
  sigma2 = float(0);
  int i1 = blockIdx.y;
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const at::Half* lvals = vals + i1*n2;
    int l = 8*thrx;
    if ((((size_t)lvals)&3) != 0) {
      // 16 bit alignment
      // first thread consumes first point
      if (thrx == 0) {
        float curr = static_cast<float>(lvals[0]);
        cuWelfordOnlineSum(curr,mu,sigma2,count);
      }
      ++l;
    }
    // at this point, lvals[l] are 32 bit aligned for all threads.
    for (;  l+7 < n2;  l+=8*numx) {
      for (int k = 0;  k < 8;  k+=2) {
        float2 curr = __half22float2(*((__half2*)(lvals+l+k)));
        cuWelfordOnlineSum(curr.x,mu,sigma2,count);
	cuWelfordOnlineSum(curr.y,mu,sigma2,count);
      }
    }
    for (;  l < n2;  ++l) {
      float curr = static_cast<float>(lvals[l]);
      cuWelfordOnlineSum(curr,mu,sigma2,count);
    }
    // intra-warp reductions
    for (int l = 0;  l <= 4;  ++l) {
      int srcLaneB = (threadIdx.x+(1<<l))&31;
      float muB = __shfl_sync(0xffffffff, mu, srcLaneB);
      float countB = __shfl_sync(0xffffffff, count, srcLaneB);
      float sigma2B = __shfl_sync(0xffffffff, sigma2, srcLaneB);
      cuChanOnlineSum(muB,sigma2B,countB,mu,sigma2,count);
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      float* ubuf = (float*)buf;
      float* ibuf = (float*)(ubuf + blockDim.y);
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2*wrt_y] = mu;
          ubuf[2*wrt_y+1] = sigma2;
          ibuf[wrt_y] = count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          float muB = ubuf[2*threadIdx.y];
          float sigma2B = ubuf[2*threadIdx.y+1];
          float countB = ibuf[threadIdx.y];
          cuChanOnlineSum(muB,sigma2B,countB,mu,sigma2,count);
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        ubuf[0] = mu;
        ubuf[1] = sigma2;
      }
      __syncthreads();
      mu = ubuf[0];
      sigma2 = ubuf[1]/float(n2);
      // don't care about final value of count, we know count == n2
    } else {
      mu = __shfl_sync(0xffffffff, mu, 0);
      sigma2 = __shfl_sync(0xffffffff, sigma2/float(n2), 0);
    }
  }
}

template<typename U> U rsqrt(U v) {
  return U(1) / sqrt(v);
}
template<> float rsqrt(float v) {
  return rsqrtf(v);
}
template<> double rsqrt(double v) {
  return rsqrt(v);
}

namespace {
// This is the un-specialized struct.  Note that we prevent instantiation of this
// struct by putting an undefined symbol in the function body so it won't compile.
template <typename T>
struct SharedMemory
{
    // Ensure that we won't compile any un-specialized types
    __device__ T *getPointer()
    {
        extern __device__ void error(void);
        error();
        return NULL;
    }
};

template <>
struct SharedMemory <float>
{
    __device__ float *getPointer()
    {
        extern __shared__ float s_float[];
        return s_float;
    }
};

template <>
struct SharedMemory <double>
{
    __device__ double *getPointer()
    {
        extern __shared__ double s_double[];
        return s_double;
    }
};
}

template<typename T, typename U> __global__
void cuApplyLayerNorm(
  T* __restrict__ output_vals,
  U* __restrict__ mean,
  U* __restrict__ invvar,
  const T* __restrict__ vals,
  const int n1,
  const int n2,
  const U epsilon,
  const T* __restrict__ gamma,
  const T* __restrict__ beta
  ) 
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensors are contiguous
  //
  int i1 = blockIdx.y;
  if (i1 < n1) {
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    U mu,sigma2;
    cuWelfordMuSigma2(vals,n1,n2,mu,sigma2,buf);
    const T* lvals = vals + i1*n2;
    T* ovals = output_vals + i1*n2;
    U c_invvar = rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL && beta != NULL) {
      for (int i = thrx;  i < n2;  i+=numx) {
        U curr = static_cast<U>(lvals[i]);
        ovals[i] = gamma[i] * static_cast<T>(c_invvar * (curr - mu)) + beta[i];
      }
    } else {
      for (int i = thrx;  i < n2;  i+=numx) {
        U curr = static_cast<U>(lvals[i]);
        ovals[i] = static_cast<T>(c_invvar * (curr - mu));
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      mean[i1] = mu;
      invvar[i1] = c_invvar;
    }
  }
}

#ifdef USE_HALF_SPECIALIZATIONS
template<> __global__
void cuApplyLayerNorm(
  at::Half* __restrict__ output_vals,
  float* __restrict__ mean,
  float* __restrict__ invvar,
  const at::Half* __restrict__ vals,
  const int n1,
  const int n2,
  const float epsilon,
  const at::Half* __restrict__ gamma,
  const at::Half* __restrict__ beta
  ) 
{
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor are contiguous
  //
  int i1 = blockIdx.y;
  if (i1 < n1) {
    SharedMemory<float> shared;
    float* buf = shared.getPointer();
    float mu,sigma2;
    cuWelfordMuSigma2(vals,n1,n2,mu,sigma2,buf);
    const at::Half* lvals = vals + i1*n2;
    at::Half* ovals = output_vals + i1*n2;
    float c_invvar = rsqrtf(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    half muh = __float2half_rn(mu);
    __half2 muh2 = __half2half2(muh);
    half c_invvarh = __float2half_rn(c_invvar);
    __half2 c_invvarh2 = __half2half2(c_invvarh);
    if (gamma != NULL && beta != NULL) {
      int l_balign = (int)((size_t)lvals&3);
      int o_balign = (int)((size_t)ovals&3);
      if (l_balign == o_balign) {
        int i = 2*thrx;
        if (l_balign) {
          // 16 bit alignment
          // first thread consumes first point
          if (thrx == 0) {
            half curr = ((half*)lvals)[0];
            ovals[0] = ((half*)gamma)[0] * (c_invvarh * (curr - muh)) + ((half*)beta)[0];
          }
          ++i;
        }
        // at this point, lvals and ovals are 32 bit aligned
        for (;  i+1 < n2;  i+=2*numx) {
          __half2 curr = *((__half2*)(lvals+i));
          __half2 gamma0, beta0;
          gamma0.x = ((half*)gamma)[i];
          gamma0.y = ((half*)gamma)[i+1];
          beta0.x = ((half*)beta)[i];
          beta0.y = ((half*)beta)[i+1];
          __half2 ov = c_invvarh2 * (curr - muh2);
          *((__half2*)(ovals+i)) = gamma0 * ov + beta0;
        }
        for (;  i < n2;  ++i) {
          half curr = ((half*)lvals)[i];
          ovals[i] = ((half*)gamma)[i] * (c_invvarh * (curr - muh)) + ((half*)beta)[i];
        }
      } else {
        for (int i = thrx;  i < n2;  i+=numx) {
          half curr = ((half*)lvals)[i];
          ovals[i] = ((half*)gamma)[i] * (c_invvarh * (curr - muh)) + ((half*)beta)[i];
        }
      }
    } else {
      for (int i = thrx;  i < n2;  i+=numx) {
        half curr = ((half*)lvals)[i];
	ovals[i] = ((half*)gamma)[i] * (c_invvarh * (curr - muh)) + ((half*)beta)[i];
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      mean[i1] = mu;
      invvar[i1] = c_invvar;
    }
  }
}
#endif

template<typename T, typename U> __device__
void cuLoadWriteStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf1,
    U* warp_buf2,
    const T* input,
    const T* dout,
    const int i1_end,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar
    )
{
  int i1 = i1_block+thr_load_row_off;
  if (i1 < i1_end) {
    U curr_mean = mean[i1];
    U curr_invvar = invvar[i1];
    for (int k = 0;  k < blockDim.y;  ++k) {
      int i2 = i2_off + k;
      int load_idx = i1*n2+i2;
      int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
      if (i2<n2) {
        U curr_input = static_cast<U>(input[load_idx]);
	U curr_dout = static_cast<U>(dout[load_idx]);
	warp_buf1[write_idx] = curr_dout;
	warp_buf2[write_idx] = curr_dout * (curr_input - curr_mean) * curr_invvar;
      } else {
        warp_buf1[write_idx] = U(0);
        warp_buf2[write_idx] = U(0);
      }
    }
  } else {
    for (int k = 0;  k < blockDim.y;  ++k) {
      int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
      warp_buf1[write_idx] = U(0);
      warp_buf2[write_idx] = U(0);
    }
  }
}

template<typename T, typename U> __device__
void cuLoadAddStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf1,
    U* warp_buf2,
    const T* input,
    const T* dout,
    const int i1_end,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar
    )
{
  int i1 = i1_block+thr_load_row_off;
  if (i1 < i1_end) {
    U curr_mean = mean[i1];
    U curr_invvar = invvar[i1];
    for (int k = 0;  k < blockDim.y;  ++k) {
      int i2 = i2_off + k;
      int load_idx = i1*n2+i2;
      int write_idx = thr_load_row_off*row_stride+thr_load_col_off+k;
      if (i2<n2) {
        U curr_input = static_cast<U>(input[load_idx]);
	U curr_dout = static_cast<U>(dout[load_idx]);
	warp_buf1[write_idx] += curr_dout;
	warp_buf2[write_idx] += curr_dout * (curr_input - curr_mean) * curr_invvar;
      }
    }
  }
}

template<typename T, typename U> __global__
void cuComputePartGradGammaBeta(
    const T* __restrict__ dout,
    const T* __restrict__ input,
    const int n1,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    U epsilon,
    U* part_grad_gamma,
    U* part_grad_beta)
{
    const int numsegs_n1 = (n1+blockDim.y*blockDim.y-1) / (blockDim.y*blockDim.y);
    const int segs_per_block = (numsegs_n1 + gridDim.y - 1) / gridDim.y;
    const int i1_beg = blockIdx.y * segs_per_block * blockDim.y*blockDim.y;
    const int i1_beg_plus_one = (blockIdx.y+1) * segs_per_block * blockDim.y*blockDim.y;
    const int i1_end = i1_beg_plus_one < n1 ? i1_beg_plus_one : n1;
    const int row_stride = blockDim.x+1;
    const int thr_load_col_off = (threadIdx.x*blockDim.y)&(blockDim.x-1);
    const int thr_load_row_off = (threadIdx.x*blockDim.y)/blockDim.x + threadIdx.y*blockDim.y;
    const int i2_off = blockIdx.x * blockDim.x + thr_load_col_off;
    SharedMemory<U> shared;
    U* buf = shared.getPointer(); // buf has at least blockDim.x * blockDim.y * blockDim.y + (blockDim.y - 1)*(blockDim.x/blockDim.y) elements
    U* warp_buf1 = (U*)buf;
    U* warp_buf2 = warp_buf1 + blockDim.y * blockDim.y * row_stride;
    // compute partial sums from strided inputs
    // do this to increase number of loads in flight
    cuLoadWriteStridedInputs(i1_beg,thr_load_row_off,thr_load_col_off,i2_off,row_stride,warp_buf1,warp_buf2,input,dout,i1_end,n2,mean,invvar);
    for (int i1_block = i1_beg+blockDim.y*blockDim.y;  i1_block < i1_end;  i1_block+=blockDim.y*blockDim.y) {
      cuLoadAddStridedInputs(i1_block,thr_load_row_off,thr_load_col_off,i2_off,row_stride,warp_buf1,warp_buf2,input,dout,i1_end,n2,mean,invvar);
    }
    __syncthreads();
    // inter-warp reductions
    // sum within each warp
    U acc1 = U(0);
    U acc2 = U(0);
    for (int k = 0;  k < blockDim.y;  ++k) {
      int row1 = threadIdx.y + k*blockDim.y;
      int idx1 = row1*row_stride + threadIdx.x;
      acc1 += warp_buf1[idx1];
      acc2 += warp_buf2[idx1];
    }
    warp_buf1[threadIdx.y*row_stride+threadIdx.x] = acc1;
    warp_buf2[threadIdx.y*row_stride+threadIdx.x] = acc2;
    __syncthreads();
    // sum all warps
    for (int offset = blockDim.y/2;  offset > 1;  offset /= 2) {
      if (threadIdx.y < offset) {
        int row1 = threadIdx.y;
	int row2 = threadIdx.y + offset;
	int idx1 = row1*row_stride + threadIdx.x;
	int idx2 = row2*row_stride + threadIdx.x;
	warp_buf1[idx1] += warp_buf1[idx2];
	warp_buf2[idx1] += warp_buf2[idx2];
      }
      __syncthreads();
    }
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.y == 0 && i2 < n2) {
      int row1 = threadIdx.y;
      int row2 = threadIdx.y + 1;
      int idx1 = row1*row_stride + threadIdx.x;
      int idx2 = row2*row_stride + threadIdx.x;
      part_grad_beta[blockIdx.y*n2+i2] = warp_buf1[idx1] + warp_buf1[idx2];
      part_grad_gamma[blockIdx.y*n2+i2] = warp_buf2[idx1] + warp_buf2[idx2];
    }
}

template<typename T, typename U> __global__
void cuComputeGradGammaBeta(
    const U* part_grad_gamma,
    const U* part_grad_beta,
    const int part_size,
    const int n1,
    const int n2,
    T* grad_gamma,
    T* grad_beta)
{
    // sum partial gradients for gamma and beta
    SharedMemory<U> shared;
    U* buf = shared.getPointer(); 
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i2 < n2) {
      // each warp does sequential reductions until reduced part_size is num_warps
      int num_warp_reductions = part_size / blockDim.y;
      U sum_gamma = U(0);
      U sum_beta = U(0);
      const U* part_grad_gamma_ptr = part_grad_gamma + threadIdx.y * num_warp_reductions * n2 + i2;
      const U* part_grad_beta_ptr = part_grad_beta + threadIdx.y * num_warp_reductions * n2 + i2;
      for (int warp_offset = 0;  warp_offset < num_warp_reductions;  ++warp_offset) {
        sum_gamma += part_grad_gamma_ptr[warp_offset*n2];
        sum_beta += part_grad_beta_ptr[warp_offset*n2];
      }
      // inter-warp reductions
      const int nbsize3 = blockDim.x * blockDim.y / 2;
      for (int offset = blockDim.y/2;  offset >= 1;  offset /= 2) {
        // top half write to shared memory
        if (threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          buf[write_idx] = sum_gamma;
          buf[write_idx+nbsize3] = sum_beta;
        }
        __syncthreads();
        // bottom half sums
        if (threadIdx.y < offset) {
          const int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
          sum_gamma += buf[read_idx];
          sum_beta += buf[read_idx+nbsize3];
        }
        __syncthreads();
      }
      // write out fully summed gradients
      if (threadIdx.y == 0) {
        grad_gamma[i2] = sum_gamma;
        grad_beta[i2] = sum_beta;
      }
    }
}

template<typename T, typename U> __global__
void cuComputeGradInput(
    const T* __restrict__ dout,
    const T* __restrict__ input,
    const int n1,
    const int n2,
    const U* __restrict__ mean,
    const U* __restrict__ invvar,
    U epsilon,
    const T* gamma,
    T* grad_input)
{
  int i1 = blockIdx.y;
  if (i1 < n1) {
    U sum_loss1 = U(0);
    U sum_loss2 = U(0);
    const U c_mean = mean[i1];
    const U c_invvar = invvar[i1];
    const T* k_input = input + i1*n2;
    const T* k_dout = dout + i1*n2;
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL) {
      int l = 4*thrx;
      for (;  l+3 < n2;  l+=4*numx) {
        for (int k = 0;  k < 4;  ++k) {
          const U c_h = static_cast<U>(k_input[l+k]);
          const U c_loss = static_cast<U>(k_dout[l+k]);
          sum_loss1 += c_loss * gamma[l+k];
          sum_loss2 += c_loss * gamma[l+k] * (c_h - c_mean) * c_invvar;
        }
      }
      for (;  l < n2;  ++l) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        sum_loss1 += c_loss * gamma[l];
        sum_loss2 += c_loss * gamma[l] * (c_h - c_mean) * c_invvar;
      }
    } else {
      int l = 4*thrx;
      for (;  l+3 < n2;  l+=4*numx) {
        for (int k = 0;  k < 4;  ++k) {
          const U c_h = static_cast<U>(k_input[l+k]);
          const U c_loss = static_cast<U>(k_dout[l+k]);
          sum_loss1 += c_loss;
          sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
        }
      }
      for (;  l < n2;  ++l) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        sum_loss1 += c_loss;
        sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
      }
    }
    // intra-warp reductions
    for (int mask = blockDim.x/2;  mask > 0;  mask /= 2) {
      sum_loss1 += __shfl_xor_sync(0xffffffff, sum_loss1, mask);
      sum_loss2 += __shfl_xor_sync(0xffffffff, sum_loss2, mask);
    }
    // inter-warp reductions
    if (blockDim.y > 1) {
      SharedMemory<U> shared;
      U* buf = shared.getPointer(); 
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_i = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          buf[2*wrt_i] = sum_loss1;
          buf[2*wrt_i+1] = sum_loss2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.y < offset) {
          const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
          sum_loss1 += buf[2*read_i];
          sum_loss2 += buf[2*read_i+1];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        buf[2*threadIdx.x] = sum_loss1;
        buf[2*threadIdx.x+1] = sum_loss2;
      }
      __syncthreads();
      if (threadIdx.y !=0) {
        sum_loss1 = buf[2*threadIdx.x];
        sum_loss2 = buf[2*threadIdx.x+1];
      } 
    }
    // all threads now have the two sums over l
    U fH = (U)n2;
    U term1 = (U(1) / fH) * c_invvar;
    T* k_grad_input = grad_input + i1*n2;
    if (gamma != NULL) {
      for (int l = thrx;  l < n2;  l+=numx) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss * gamma[l];
        f_grad_input -= sum_loss1;
        f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    } else {
      for (int l = thrx;  l < n2;  l+=numx) {
        const U c_h = static_cast<U>(k_input[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss;
        f_grad_input -= sum_loss1;
        f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    }
  }
}

#ifdef USE_HALF_SPECIALIZATIONS
__device__
__half2 cuLoadHalf2(const half* p)
{
  return *((__half2*)p);
}

__device__
__half2 cuLoadHalf2Misaligned(const half* p)
{
  __half2 hv;
  hv.x = p[0];
  hv.y = p[1];
  return hv;
}

__device__
void cuWriteHalf2(half* p, const __half2 v)
{
  *((__half2*)p) = v;
}

__device__
void cuWriteHalf2Misaligned(half* p, const __half2 v)
{
  p[0] = v.x;
  p[1] = v.y;
}

__device__
__half2 cuGradientTwoPoints(
    const __half2 c_meanh2,
    const __half2 c_invvarh2,
    const __half2 fHh2,
    const __half2 term1h2,
    const __half2 sum_loss1h2,
    const __half2 sum_loss2h2,
    const __half2 c_h,
    const __half2 c_loss,
    const __half2 gamma0
    )
{
  __half2 h2_grad_input = fHh2 * c_loss * gamma0;
  h2_grad_input -= sum_loss1h2;
  h2_grad_input -= (c_h - c_meanh2) * c_invvarh2 * sum_loss2h2;
  h2_grad_input *= term1h2;
  return h2_grad_input;
}

__device__
__half2 cuGradientTwoPoints(
    const __half2 c_meanh2,
    const __half2 c_invvarh2,
    const __half2 fHh2,
    const __half2 term1h2,
    const __half2 sum_loss1h2,
    const __half2 sum_loss2h2,
    const __half2 c_h,
    const __half2 c_loss
    )
{
  __half2 h2_grad_input = fHh2 * c_loss;
  h2_grad_input -= sum_loss1h2;
  h2_grad_input -= (c_h - c_meanh2) * c_invvarh2 * sum_loss2h2;
  h2_grad_input *= term1h2;
  return h2_grad_input;
}

__device__
void cuComputeGradientOnePoint(
    const half c_meanh,
    const half c_invvarh,
    const half fHh,
    const half term1h,
    const half sum_loss1h,
    const half sum_loss2h,
    const int l,
    const half* k_input,
    const half* k_dout,
    const half* k_gamma,
    half* k_grad_input
    )
{
  half c_h = k_input[l];
  half c_loss = k_dout[l];
  half h_grad_input = fHh * c_loss * k_gamma[l];
  h_grad_input -= sum_loss1h;
  h_grad_input -= (c_h - c_meanh) * c_invvarh * sum_loss2h;
  h_grad_input *= term1h;
  ((half*)k_grad_input)[l] = h_grad_input;
}

__device__
void cuComputeGradientOnePoint(
    const half c_meanh,
    const half c_invvarh,
    const half fHh,
    const half term1h,
    const half sum_loss1h,
    const half sum_loss2h,
    const int l,
    const half* k_input,
    const half* k_dout,
    half* k_grad_input
    )
{
  half c_h = ((half*)k_input)[l];
  half c_loss = ((half*)k_dout)[l];
  half h_grad_input = fHh * c_loss;
  h_grad_input -= sum_loss1h;
  h_grad_input -= (c_h - c_meanh) * c_invvarh * sum_loss2h;
  h_grad_input *= term1h;
  ((half*)k_grad_input)[l] = h_grad_input;
}

__device__
void cuComputeGradientTwoPoints(
    const __half2 c_meanh2,
    const __half2 c_invvarh2,
    const __half2 fHh2,
    const __half2 term1h2,
    const __half2 sum_loss1h2,
    const __half2 sum_loss2h2,
    const int l,
    const half* k_input,
    const half* k_dout,
    const half* k_gamma,
    half* k_grad_input
    )
{
  __half2 c_h = cuLoadHalf2(k_input+l);
  __half2 c_loss = cuLoadHalf2(k_dout+l);
  __half2 gamma0 = cuLoadHalf2(k_gamma+l);
  __half2 h2_grad_input = cuGradientTwoPoints(c_meanh2,c_invvarh2,fHh2,term1h2,sum_loss1h2,sum_loss2h2,c_h,c_loss,gamma0);
  cuWriteHalf2(k_grad_input+l,h2_grad_input);
}

__device__
void cuComputeGradientTwoPoints(
    const __half2 c_meanh2,
    const __half2 c_invvarh2,
    const __half2 fHh2,
    const __half2 term1h2,
    const __half2 sum_loss1h2,
    const __half2 sum_loss2h2,
    const int l,
    const half* k_input,
    const half* k_dout,
    half* k_grad_input
    )
{
  __half2 c_h = cuLoadHalf2(k_input+l);
  __half2 c_loss = cuLoadHalf2(k_dout+l);
  __half2 h2_grad_input = cuGradientTwoPoints(c_meanh2,c_invvarh2,fHh2,term1h2,sum_loss1h2,sum_loss2h2,c_h,c_loss);
  cuWriteHalf2(k_grad_input+l,h2_grad_input);
}

__device__
void cuComputeGradientTwoPointsGammaMisaligned(
    const __half2 c_meanh2,
    const __half2 c_invvarh2,
    const __half2 fHh2,
    const __half2 term1h2,
    const __half2 sum_loss1h2,
    const __half2 sum_loss2h2,
    const int l,
    const half* k_input,
    const half* k_dout,
    const half* k_gamma,
    half* k_grad_input
    )
{
  __half2 c_h = cuLoadHalf2(k_input+l);
  __half2 c_loss = cuLoadHalf2(k_dout+l);
  __half2 gamma0 = cuLoadHalf2Misaligned(k_gamma+l);
  __half2 h2_grad_input = cuGradientTwoPoints(c_meanh2,c_invvarh2,fHh2,term1h2,sum_loss1h2,sum_loss2h2,c_h,c_loss,gamma0);
  cuWriteHalf2(k_grad_input+l,h2_grad_input);
}

__device__
void cuComputeGradientTwoPointsMisaligned(
    const __half2 c_meanh2,
    const __half2 c_invvarh2,
    const __half2 fHh2,
    const __half2 term1h2,
    const __half2 sum_loss1h2,
    const __half2 sum_loss2h2,
    const int l,
    const half* k_input,
    const half* k_dout,
    const half* k_gamma,
    half* k_grad_input
    )
{
  __half2 c_h = cuLoadHalf2Misaligned(k_input+l);
  __half2 c_loss = cuLoadHalf2Misaligned(k_dout+l);
  __half2 gamma0 = cuLoadHalf2Misaligned(k_gamma+l);
  __half2 h2_grad_input = cuGradientTwoPoints(c_meanh2,c_invvarh2,fHh2,term1h2,sum_loss1h2,sum_loss2h2,c_h,c_loss,gamma0);
  cuWriteHalf2Misaligned(k_grad_input+l,h2_grad_input);
}

__device__
void cuComputeGradientTwoPointsMisaligned(
    const __half2 c_meanh2,
    const __half2 c_invvarh2,
    const __half2 fHh2,
    const __half2 term1h2,
    const __half2 sum_loss1h2,
    const __half2 sum_loss2h2,
    const int l,
    const half* k_input,
    const half* k_dout,
    half* k_grad_input
    )
{
  __half2 c_h = cuLoadHalf2Misaligned(k_input+l);
  __half2 c_loss = cuLoadHalf2Misaligned(k_dout+l);
  __half2 h2_grad_input = cuGradientTwoPoints(c_meanh2,c_invvarh2,fHh2,term1h2,sum_loss1h2,sum_loss2h2,c_h,c_loss);
  cuWriteHalf2Misaligned(k_grad_input+l,h2_grad_input);
}

__device__
void cuComputeSumLossOnePoint(
    const half c_meanh,
    const half c_invvarh,
    const int l,
    const half* k_input,
    const half* k_dout,
    const half* k_gamma,
    float& sum_loss1,
    float& sum_loss2
    )
{
  half c_h = k_input[l];
  half c_loss = k_dout[l];
  sum_loss1 += (float)(c_loss * k_gamma[l]);
  sum_loss2 += (float)(c_loss * k_gamma[l] * (c_h - c_meanh) * c_invvarh);
}

__device__
void cuComputeSumLossOnePoint(
    const half c_meanh,
    const half c_invvarh,
    const int l,
    const half* k_input,
    const half* k_dout,
    float& sum_loss1,
    float& sum_loss2
    )
{
  half c_h = k_input[l];
  half c_loss = k_dout[l];
  sum_loss1 += (float)(c_loss);
  sum_loss2 += (float)(c_loss * (c_h - c_meanh) * c_invvarh);
}

__device__
void cuComputeSumLossTwoPoints(
    const __half2 c_meanh2,
    const __half2 c_invvarh2,
    const int l,
    const half* k_input,
    const half* k_dout,
    const half* k_gamma,
    float& sum_loss1,
    float& sum_loss2
    )
{
  __half2 c_h = cuLoadHalf2(k_input+l);
  __half2 c_loss = cuLoadHalf2(k_dout+l);
  __half2 gamma0 = cuLoadHalf2(k_gamma+l);
  __half2 term1h2 = c_loss * gamma0;
  __half2 term2h2 = term1h2 * (c_h - c_meanh2) * c_invvarh2;
  float2 term1f2 = __half22float2(term1h2);
  float2 term2f2 = __half22float2(term2h2);
  sum_loss1 += term1f2.x + term1f2.y;
  sum_loss2 += term2f2.x + term2f2.y;
}

__device__
void cuComputeSumLossTwoPoints(
    const __half2 c_meanh2,
    const __half2 c_invvarh2,
    const int l,
    const half* k_input,
    const half* k_dout,
    float& sum_loss1,
    float& sum_loss2
    )
{
  __half2 c_h = cuLoadHalf2(k_input+l);
  __half2 c_loss = cuLoadHalf2(k_dout+l);
  __half2 term1h2 = c_loss;
  __half2 term2h2 = term1h2 * (c_h - c_meanh2) * c_invvarh2;
  float2 term1f2 = __half22float2(term1h2);
  float2 term2f2 = __half22float2(term2h2);
  sum_loss1 += term1f2.x + term1f2.y;
  sum_loss2 += term2f2.x + term2f2.y;
}

__device__
void cuComputeSumLossTwoPointsGammaMisaligned(
    const __half2 c_meanh2,
    const __half2 c_invvarh2,
    const int l,
    const half* k_input,
    const half* k_dout,
    const half* k_gamma,
    float& sum_loss1,
    float& sum_loss2
    )
{
  __half2 c_h = cuLoadHalf2(k_input+l);
  __half2 c_loss = cuLoadHalf2(k_dout+l);
  __half2 gamma0 = cuLoadHalf2Misaligned(k_gamma+l);
  __half2 term1h2 = c_loss * gamma0;
  __half2 term2h2 = term1h2 * (c_h - c_meanh2) * c_invvarh2;
  float2 term1f2 = __half22float2(term1h2);
  float2 term2f2 = __half22float2(term2h2);
  sum_loss1 += term1f2.x + term1f2.y;
  sum_loss2 += term2f2.x + term2f2.y;
}

__device__
void cuComputeSumLossTwoPointsMisaligned(
    const __half2 c_meanh2,
    const __half2 c_invvarh2,
    const int l,
    const half* k_input,
    const half* k_dout,
    const half* k_gamma,
    float& sum_loss1,
    float& sum_loss2
    )
{
  __half2 c_h = cuLoadHalf2Misaligned(k_input+l);
  __half2 c_loss = cuLoadHalf2Misaligned(k_dout+l);
  __half2 gamma0 = cuLoadHalf2Misaligned(k_gamma+l);
  __half2 term1h2 = c_loss * gamma0;
  __half2 term2h2 = term1h2 * (c_h - c_meanh2) * c_invvarh2;
  float2 term1f2 = __half22float2(term1h2);
  float2 term2f2 = __half22float2(term2h2);
  sum_loss1 += term1f2.x + term1f2.y;
  sum_loss2 += term2f2.x + term2f2.y;
}

__device__
void cuComputeSumLossTwoPointsMisaligned(
    const __half2 c_meanh2,
    const __half2 c_invvarh2,
    const int l,
    const half* k_input,
    const half* k_dout,
    float& sum_loss1,
    float& sum_loss2
    )
{
  __half2 c_h = cuLoadHalf2Misaligned(k_input+l);
  __half2 c_loss = cuLoadHalf2Misaligned(k_dout+l);
  __half2 term1h2 = c_loss;
  __half2 term2h2 = term1h2 * (c_h - c_meanh2) * c_invvarh2;
  float2 term1f2 = __half22float2(term1h2);
  float2 term2f2 = __half22float2(term2h2);
  sum_loss1 += term1f2.x + term1f2.y;
  sum_loss2 += term2f2.x + term2f2.y;
}

#define TLP1 4
#define TLP2 2
template<> __global__
void cuComputeGradInput(
    const at::Half* __restrict__ dout,
    const at::Half* __restrict__ input,
    const int n1,
    const int n2,
    const float* __restrict__ mean,
    const float* __restrict__ invvar,
    float epsilon,
    const at::Half* gamma,
    at::Half* grad_input)
{
  int i1 = blockIdx.y;
  if (i1 < n1) {
    float sum_loss1 = 0.0f;
    float sum_loss2 = 0.0f;
    const half c_meanh = __float2half_rn(mean[i1]);
    const half c_invvarh = __float2half_rn(invvar[i1]);
    const __half2 c_meanh2 = __half2half2(c_meanh);
    const __half2 c_invvarh2 = __half2half2(c_invvarh);
    const half* k_input = (half*)(input + i1*n2);
    const half* k_dout = (half*)(dout + i1*n2);
    const half* k_gamma = (half*)gamma;
    half* k_grad_input = (half*)(grad_input + i1*n2);
    int i_align = (int)((size_t)k_input&3);
    int o_align = (int)((size_t)k_dout&3);
    int g_align = (int)((size_t)k_grad_input&3);
    int a_align = (int)((size_t)k_gamma&3);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (k_gamma != NULL) {
      int l = TLP1*thrx;
      if (i_align == o_align) {
        if (i_align != 0) {
          // 16 bit alignment
          // one thread consumes first point
          if (thrx == 0) {
            cuComputeSumLossOnePoint(
                c_meanh,c_invvarh,0,k_input,k_dout,k_gamma,
                sum_loss1,sum_loss2);
          }
          ++l;
        }
        if (i_align == a_align) {
          // at this point k_input[l] and k_dout[l] are 32 bit aligned
          for (;  l+TLP1 <= n2;  l+=TLP1*numx) {
            for (int k = 0;  k < TLP1;  k+=2) {
              cuComputeSumLossTwoPoints(
                  c_meanh2,c_invvarh2,l+k,k_input,k_dout,k_gamma,
                  sum_loss1,sum_loss2);
            }
  	  }
        } else {
          // at this point k_input[l] and k_dout[l] are 32 bit aligned
          for (;  l+TLP1 <= n2;  l+=TLP1*numx) {
            for (int k = 0;  k < TLP1;  k+=2) {
              cuComputeSumLossTwoPointsGammaMisaligned(
                  c_meanh2,c_invvarh2,l+k,k_input,k_dout,k_gamma,
                  sum_loss1,sum_loss2);
            }
  	  }
        }
      } else {
        // k_input and k_dout have different alignment,
        // not possible to safely read them as __half2.
        for (;  l+TLP1 <= n2;  l+=TLP1*numx) {
          for (int k = 0;  k < TLP1;  k+=2) {
            cuComputeSumLossTwoPointsMisaligned(
                c_meanh2,c_invvarh2,l+k,k_input,k_dout,k_gamma,
                sum_loss1,sum_loss2);
	  }
	}
      }
      for (;  l < n2;  ++l) {
        // finish any remaining points, will be 3 or less
        // and only affect a single thread.
        cuComputeSumLossOnePoint(
            c_meanh,c_invvarh,l,k_input,k_dout,k_gamma,
            sum_loss1,sum_loss2);
      }
    } else {
      int l = TLP1*thrx;
      if (i_align == o_align) {
        if (i_align != 0) {
          // 16 bit alignment
          // one thread consumes first point
          if (thrx == 0) {
            cuComputeSumLossOnePoint(
                c_meanh,c_invvarh,0,k_input,k_dout,
                sum_loss1,sum_loss2);
          }
          ++l;
        }
        // at this point k_input[l] and k_dout[l] are 32 bit aligned
        for (;  l+TLP1 <= n2;  l+=TLP1*numx) {
          for (int k = 0;  k < TLP1;  k+=2) {
            cuComputeSumLossTwoPoints(
                c_meanh2,c_invvarh2,l+k,k_input,k_dout,
                sum_loss1,sum_loss2);
          }
        }
      } else {
        // k_input and k_dout have different alignment,
        // not possible to safely read them as __half2.
        for (;  l+TLP1 <= n2;  l+=TLP1*numx) {
          for (int k = 0;  k < TLP1;  k+=2) {
            cuComputeSumLossTwoPointsMisaligned(
                c_meanh2,c_invvarh2,l+k,k_input,k_dout,
                sum_loss1,sum_loss2);
	  }
	}
      }
      for (;  l < n2;  ++l) {
        // finish any remaining points, will be 3 or less
        // and only affect a single thread.
        cuComputeSumLossOnePoint(
            c_meanh,c_invvarh,l,k_input,k_dout,k_gamma,
            sum_loss1,sum_loss2);
      }
    }
    // intra-warp reductions
    for (int mask = blockDim.x/2;  mask > 0;  mask /= 2) {
      sum_loss1 += __shfl_xor_sync(0xffffffff, sum_loss1, mask);
      sum_loss2 += __shfl_xor_sync(0xffffffff, sum_loss2, mask);
    }
    // inter-warp reductions
    if (blockDim.y > 1) {
      SharedMemory<float> shared;
      float* buf = shared.getPointer(); 
      for (int offset = blockDim.y/2;  offset > 0;  offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.y >= offset && threadIdx.y < 2*offset) {
          const int wrt_i = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          buf[2*wrt_i] = sum_loss1;
          buf[2*wrt_i+1] = sum_loss2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.y < offset) {
          const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
          sum_loss1 += buf[2*read_i];
          sum_loss2 += buf[2*read_i+1];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        buf[2*threadIdx.x] = sum_loss1;
        buf[2*threadIdx.x+1] = sum_loss2;
      }
      __syncthreads();
      if (threadIdx.y !=0) {
        sum_loss1 = buf[2*threadIdx.x];
        sum_loss2 = buf[2*threadIdx.x+1];
      } 
    }
    // all threads now have the two sums over l
    const half sum_loss1h = __float2half_rn(sum_loss1);
    const half sum_loss2h = __float2half_rn(sum_loss2);
    const __half2 sum_loss1h2 = __half2half2(sum_loss1h);
    const __half2 sum_loss2h2 = __half2half2(sum_loss2h);
    const half fHh = (half)n2;
    const half term1h = (half(1) / fHh) * c_invvarh;
    __half2 fHh2 = __half2half2(fHh);
    __half2 term1h2 = __half2half2(term1h);
    if (k_gamma != NULL) {
      int l = TLP2*thrx;
      if (i_align == o_align && i_align == g_align) {
        if (i_align != 0) {
          // 16 bit alignment
          // one thread consumes first point
          if (thrx == 0) {
            cuComputeGradientOnePoint(
                c_meanh,c_invvarh,fHh,term1h,sum_loss1h,sum_loss2h,
                0,k_input,k_dout,k_gamma,k_grad_input);
          }
          ++l;
        }
        if (i_align == a_align) {
          // k_input[l], k_dout[l], k_grad_input[l] and k_gamma[l] are 32 bit aligned
          for (;  l+TLP2 <= n2;  l+=TLP2*numx) {
            for (int k = 0;  k < TLP2;  k+=2) {
              cuComputeGradientTwoPoints(
                  c_meanh2,c_invvarh2,fHh2,term1h2,sum_loss1h2,sum_loss2h2,
                  l+k,k_input,k_dout,k_gamma,k_grad_input);
            }
          }
        } else {
          // k_input[l], k_dout[l] and k_grad_input[l] are 32 bit aligned
          for (;  l+TLP2 <= n2;  l+=TLP2*numx) {
            for (int k = 0;  k < TLP2;  k+=2) {
              cuComputeGradientTwoPointsGammaMisaligned(
                  c_meanh2,c_invvarh2,fHh2,term1h2,sum_loss1h2,sum_loss2h2,
                  l+k,k_input,k_dout,k_gamma,k_grad_input);
            }
          }
        }
      } else {
        for (;  l+TLP2 <= n2;  l+=TLP2*numx) {
          for (int k = 0;  k < TLP2;  k+=2) {
            cuComputeGradientTwoPointsMisaligned(
                c_meanh2,c_invvarh2,fHh2,term1h2,sum_loss1h2,sum_loss2h2,
                l+k,k_input,k_dout,k_gamma,k_grad_input);
          }
        }
      }
      for (;  l < n2;  ++l) {
        cuComputeGradientOnePoint(
            c_meanh,c_invvarh,fHh,term1h,sum_loss1h,sum_loss2h,
            l,k_input,k_dout,k_gamma,k_grad_input);
      }
    } else {
      int l = TLP2*thrx;
      if (i_align == o_align && i_align == g_align) {
        if (i_align != 0) {
          // 16 bit alignment
          // one thread consumes first point
          if (thrx == 0) {
            cuComputeGradientOnePoint(
                c_meanh,c_invvarh,fHh,term1h,sum_loss1h,sum_loss2h,
                0,k_input,k_dout,k_grad_input);
          }
          ++l;
        }
        // k_input[l], k_dout[l], k_grad_input[l] and k_gamma[l] are 32 bit aligned
        for (;  l+TLP2 <= n2;  l+=TLP2*numx) {
          for (int k = 0;  k < TLP2;  k+=2) {
            cuComputeGradientTwoPoints(
                c_meanh2,c_invvarh2,fHh2,term1h2,sum_loss1h2,sum_loss2h2,
                l+k,k_input,k_dout,k_grad_input);
          }
        }
      } else {
        for (;  l+TLP2 <= n2;  l+=TLP2*numx) {
          for (int k = 0;  k < TLP2;  k+=2) {
            cuComputeGradientTwoPointsMisaligned(
                c_meanh2,c_invvarh2,fHh2,term1h2,sum_loss1h2,sum_loss2h2,
                l+k,k_input,k_dout,k_grad_input);
          }
        }
      }
      for (;  l < n2;  ++l) {
        cuComputeGradientOnePoint(
            c_meanh,c_invvarh,fHh,term1h,sum_loss1h,sum_loss2h,
            l,k_input,k_dout,k_grad_input);
      }
    }
  }
}
#undef TLP2
#undef TLP1
#endif

template<typename T> 
void HostApplyLayerNorm(
    T* output,
    at::Tensor* mean,
    at::Tensor* invvar,
    const T* input,
    int n1,
    int n2,
    double epsilon,
    const T* gamma,
    const T* beta
    )
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const dim3 threads(32,4,1);
    const dim3 blocks(1,n1,1);
    int nshared = 
        threads.y > 1 ? 
	    threads.y*sizeof(float)+(threads.y/2)*sizeof(float) : 
	    0;
    cuApplyLayerNorm<<<blocks, threads, nshared, stream>>>(
		    output,
		    mean->data<float>(),
		    invvar->data<float>(),
		    input,
		    n1,n2,
		    (float)epsilon,
                    gamma,beta);
}
template<> 
void HostApplyLayerNorm(
    int64_t* output,
    at::Tensor* mean,
    at::Tensor* invvar,
    const int64_t* input,
    int n1,
    int n2,
    double epsilon,
    const int64_t* gamma,
    const int64_t* beta
    )
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const dim3 threads(32,4,1);
    const dim3 blocks(1,n1,1);
    int nshared = 
        threads.y > 1 ? 
	    threads.y*sizeof(double)+(threads.y/2)*sizeof(double) : 
	    0;
    cuApplyLayerNorm<<<blocks, threads, nshared, stream>>>(
		    output,
		    mean->data<double>(),
		    invvar->data<double>(),
		    input,
		    n1,n2,
		    (double)epsilon,
                    gamma,beta);
}
template<> 
void HostApplyLayerNorm(
    double* output,
    at::Tensor* mean,
    at::Tensor* invvar,
    const double* input,
    int n1,
    int n2,
    double epsilon,
    const double* gamma,
    const double* beta
    )
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const dim3 threads(32,4,1);
    const dim3 blocks(1,n1,1);
    int nshared = 
        threads.y > 1 ? 
	    threads.y*sizeof(double)+(threads.y/2)*sizeof(double) : 
	    0;
    cuApplyLayerNorm<<<blocks, threads, nshared, stream>>>(
		    output,
		    mean->data<double>(),
		    invvar->data<double>(),
		    input,
		    n1,n2,
		    (double)epsilon,
                    gamma,beta);
}

void cuda_layer_norm(
    at::Tensor* output,
    at::Tensor* mean,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    at::IntList normalized_shape,
    at::Tensor* gamma,
    at::Tensor* beta,
    double epsilon)
{
    AT_DISPATCH_ALL_TYPES_AND_HALF(input->type(), "layer_norm_cuda_kernel", ([&] {
        HostApplyLayerNorm(
            output->data<scalar_t>(),
	    mean,
	    invvar,
	    input->data<scalar_t>(),
	    n1,n2,
	    epsilon,
	    gamma != NULL ? gamma->data<scalar_t>() : NULL,
	    beta != NULL ? beta->data<scalar_t>() : NULL);
      }));
}

template<typename T> 
void HostLayerNormGradient(
    const T* dout,
    at::Tensor* mean,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    const T* gamma,
    const T* beta,
    double epsilon,
    at::Tensor* grad_input,
    at::Tensor* grad_gamma,
    at::Tensor* grad_beta
    )
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (gamma != NULL && beta != NULL) {
      // compute grad_gamma(j) and grad_beta(j)
      const int part_size = 16;
      const dim3 threads2(32,4,1);
      const dim3 blocks2((n2+threads2.x-1)/threads2.x,part_size,1);
      at::Tensor part_grad_gamma = at::empty({part_size,n2}, input->options().dtype(at::kFloat));
      at::Tensor part_grad_beta = at::empty({part_size,n2}, input->options().dtype(at::kFloat));
      const int nshared2_a = 2 * sizeof(float) * threads2.y * threads2.y * (threads2.x + 1);
      const int nshared2_b = threads2.x * threads2.y * sizeof(float);
      const int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;
      cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
		      dout,
		      input->data<T>(),
		      n1,n2,
		      mean->data<float>(),
		      invvar->data<float>(),
		      (float)epsilon,
		      part_grad_gamma.data<float>(),
		      part_grad_beta.data<float>());

      const dim3 threads3(32,8,1);
      const dim3 blocks3((n2+threads2.x-1)/threads2.x,1,1);
      const int nshared3 = threads3.x * threads3.y * sizeof(float);
      cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
		      part_grad_gamma.data<float>(),
		      part_grad_beta.data<float>(),
		      part_size,
		      n1,n2,
		      grad_gamma->data<T>(),
		      grad_beta->data<T>());
    }

    // compute grad_input
    const dim3 threads1(32,4,1);
    const dim3 blocks1(1,n1,1);
    int nshared =
	    threads1.y > 1 ?
	    threads1.y*threads1.x*sizeof(float) :
	    0;
    cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
            dout,
            input->data<T>(),
            n1,n2,
            mean->data<float>(),
            invvar->data<float>(),
            (float)epsilon,
            gamma,
            grad_input->data<T>());
}
template<> 
void HostLayerNormGradient(
    const int64_t* dout,
    at::Tensor* mean,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    const int64_t* gamma,
    const int64_t* beta,
    double epsilon,
    at::Tensor* grad_input,
    at::Tensor* grad_gamma,
    at::Tensor* grad_beta
    )
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (gamma != NULL && beta != NULL) {
      // compute grad_gamma(j) and grad_beta(j)
      const int part_size = 16;
      const dim3 threads2(32,4,1);
      const dim3 blocks2((n2+threads2.x-1)/threads2.x,part_size,1);
      at::Tensor part_grad_gamma = at::empty({part_size,n2}, input->options().dtype(at::kDouble));
      at::Tensor part_grad_beta = at::empty({part_size,n2}, input->options().dtype(at::kDouble));
      const int nshared2_a = 2 * sizeof(double) * threads2.y * threads2.y * (threads2.x + 1);
      const int nshared2_b = threads2.x * threads2.y * sizeof(double);
      const int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;
      cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
		      dout,
		      input->data<int64_t>(),
		      n1,n2,
		      mean->data<double>(),
		      invvar->data<double>(),
		      (double)epsilon,
		      part_grad_gamma.data<double>(),
		      part_grad_beta.data<double>());

      const dim3 threads3(32,8,1);
      const dim3 blocks3((n2+threads2.x-1)/threads2.x,1,1);
      const int nshared3 = threads3.x * threads3.y * sizeof(double);
      cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
		      part_grad_gamma.data<double>(),
		      part_grad_beta.data<double>(),
		      part_size,
		      n1,n2,
		      grad_gamma->data<int64_t>(),
		      grad_beta->data<int64_t>());
    }

    // compute grad_input
    const dim3 threads1(32,4,1);
    const dim3 blocks1(1,n1,1);
    int nshared =
	    threads1.y > 1 ?
	    threads1.y*threads1.x*sizeof(double) :
	    0;
    cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
            dout,
            input->data<int64_t>(),
            n1,n2,
            mean->data<double>(),
            invvar->data<double>(),
            (double)epsilon,
            gamma,
            grad_input->data<int64_t>());
}
template<> 
void HostLayerNormGradient(
    const double* dout,
    at::Tensor* mean,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    const double* gamma,
    const double* beta,
    double epsilon,
    at::Tensor* grad_input,
    at::Tensor* grad_gamma,
    at::Tensor* grad_beta
    )
{
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (gamma != NULL && beta != NULL) {
      // compute grad_gamma(j) and grad_beta(j)
      const int part_size = 16;
      const dim3 threads2(32,4,1);
      const dim3 blocks2((n2+threads2.x-1)/threads2.x,part_size,1);
      at::Tensor part_grad_gamma = at::empty({part_size,n2}, input->options().dtype(at::kDouble));
      at::Tensor part_grad_beta = at::empty({part_size,n2}, input->options().dtype(at::kDouble));
      const int nshared2_a = 2 * sizeof(double) * threads2.y * threads2.y * (threads2.x + 1);
      const int nshared2_b = threads2.x * threads2.y * sizeof(double);
      const int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;
      cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
		      dout,
		      input->data<double>(),
		      n1,n2,
		      mean->data<double>(),
		      invvar->data<double>(),
		      (double)epsilon,
		      part_grad_gamma.data<double>(),
		      part_grad_beta.data<double>());

      const dim3 threads3(32,8,1);
      const dim3 blocks3((n2+threads2.x-1)/threads2.x,1,1);
      const int nshared3 = threads3.x * threads3.y * sizeof(double);
      cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
		      part_grad_gamma.data<double>(),
		      part_grad_beta.data<double>(),
		      part_size,
		      n1,n2,
		      grad_gamma->data<double>(),
		      grad_beta->data<double>());
    }

    // compute grad_input
    const dim3 threads1(32,4,1);
    const dim3 blocks1(1,n1,1);
    int nshared =
	    threads1.y > 1 ?
	    threads1.y*threads1.x*sizeof(double) :
	    0;
    cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
            dout,
            input->data<double>(),
            n1,n2,
            mean->data<double>(),
            invvar->data<double>(),
            (double)epsilon,
            gamma,
            grad_input->data<double>());
}

void cuda_layer_norm_gradient(
    at::Tensor* dout,
    at::Tensor* mean,
    at::Tensor* invvar,
    at::Tensor* input,
    int n1,
    int n2,
    at::IntList normalized_shape,
    at::Tensor* gamma,
    at::Tensor* beta,
    double epsilon,
    at::Tensor* grad_input,
    at::Tensor* grad_gamma,
    at::Tensor* grad_beta)
{
    AT_DISPATCH_ALL_TYPES_AND_HALF(input->type(), "cuComputeGradInput", ([&] {
			    HostLayerNormGradient(
					    dout->data<scalar_t>(),
					    mean,invvar,
					    input,
					    n1,n2,
					    gamma->data<scalar_t>(),
					    beta->data<scalar_t>(),
					    epsilon,
					    grad_input,grad_gamma,grad_beta);
      }));
}
