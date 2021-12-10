#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/DeviceUtils.cuh>

template <typename U>
__device__ void cuWelfordOnlineSum(const U curr, U &mu, U &sigma2, U &count);

template <typename U>
__device__ void cuChanOnlineSum(const U muB, const U sigma2B, const U countB,
                                U &mu, U &sigma2, U &count);

template <typename T, typename U>
__device__ void cuWelfordMuSigma2(const T *__restrict__ vals, const int n1,
                                  const int n2, const int i1, U &mu, U &sigma2,
                                  U *buf);

template <typename T, typename U>
__global__ void
cuApplyLayerNorm(T *__restrict__ output_vals, U *__restrict__ mean,
                 U *__restrict__ invvar, const T *__restrict__ vals,
                 const int n1, const int n2, const U epsilon,
                 const T *__restrict__ gamma, const T *__restrict__ beta);

template <typename T, typename U>
__device__ void cuLoadWriteStridedInputs(
    const int i1_block, const int thr_load_row_off, const int thr_load_col_off,
    const int i2_off, const int row_stride, U *warp_buf1, U *warp_buf2,
    const T *input, const T *dout, const int i1_end, const int n2,
    const U *__restrict__ mean, const U *__restrict__ invvar);

template <typename T, typename U>
__device__ void cuLoadAddStridedInputs(
    const int i1_block, const int thr_load_row_off, const int thr_load_col_off,
    const int i2_off, const int row_stride, U *warp_buf1, U *warp_buf2,
    const T *input, const T *dout, const int i1_end, const int n2,
    const U *__restrict__ mean, const U *__restrict__ invvar);

template <typename T, typename U>
__global__ void cuComputePartGradGammaBeta(
    const T *__restrict__ dout, const T *__restrict__ input, const int n1,
    const int n2, const U *__restrict__ mean, const U *__restrict__ invvar,
    U epsilon, U *part_grad_gamma, U *part_grad_beta);

template <typename T, typename U>
__global__ void
cuComputeGradGammaBeta(const U *part_grad_gamma, const U *part_grad_beta,
                       const int part_size, const int n1, const int n2,
                       T *grad_gamma, T *grad_beta);

template <typename T, typename U>
__global__ void
cuComputeGradInput(const T *__restrict__ dout, const T *__restrict__ dout_resid,
                   const T *__restrict__ input, const int n1, const int n2,
                   const U *__restrict__ mean, const U *__restrict__ invvar,
                   U epsilon, const T *gamma, T *grad_input);

template <typename T, typename U>
void HostApplyLayerNorm(T *output, U *mean, U *invvar, const T *input, int n1,
                        int n2, double epsilon, const T *gamma, const T *beta);

template <typename T, typename U>
void HostLayerNormGradient(const T *dout, const T *dout_resid, const U *mean,
                           const U *invvar, const at::Tensor &input, int n1,
                           int n2, const T *gamma, const T *beta,
                           double epsilon, T *grad_input, T *grad_gamma,
                           T *grad_beta);
