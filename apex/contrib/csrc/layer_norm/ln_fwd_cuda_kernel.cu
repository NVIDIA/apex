#include "utils.cuh"
#include "ln_kernel_traits.h"
#include "ATen/cuda/CUDAContext.h"

template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::THREADS_PER_CTA) void ln_fwd_kernel(
    void *__restrict__ y_, void *__restrict__ mu_, void *__restrict__ rsigma_,
    const void *__restrict__ x_, const void *__restrict__ gamma_,
    const void *__restrict__ beta_, const float epsilon, int rows) {

  using Vec = typename Ktraits::Vec;

  using base_t = typename Ktraits::base_t;
  using compute_t = typename Ktraits::compute_t;
  enum { NUM_ELTS = Vec::NUM_ELTS };
  enum { WARPS_N = Ktraits::WARPS_N };
  enum { WARPS_M = Ktraits::WARPS_M };
  enum { ROWS_PER_CTA = Ktraits::ROWS_PER_CTA };

  enum { THREADS_PER_ROW = Ktraits::THREADS_PER_ROW };
  enum { BYTES_PER_LDG = Ktraits::BYTES_PER_LDG };
  static_assert(BYTES_PER_LDG == 16, "");

  enum { BYTES_PER_ROW = Ktraits::BYTES_PER_ROW };
  enum { LDGS = BYTES_PER_ROW / Ktraits::BYTES_PER_ROW_PER_CTA };
  static_assert(LDGS * Ktraits::BYTES_PER_ROW_PER_CTA == BYTES_PER_ROW, "");

  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;
  const int lane = tidx % THREADS_PER_WARP;
  const int warp = tidx / THREADS_PER_WARP;
  const int warp_n = warp % WARPS_N;
  const int warp_m = warp / WARPS_N;

  const int c = warp_n * THREADS_PER_WARP + lane;
  const int r = bidx * ROWS_PER_CTA + warp_m;

  const char *x_ptr = static_cast<const char *>(x_);

  const char *g_ptr = static_cast<const char *>(gamma_);
  const char *b_ptr = static_cast<const char *>(beta_);

  char *y_ptr = static_cast<char *>(y_);
  compute_t *mu_ptr = static_cast<compute_t *>(mu_);
  compute_t *rs_ptr = static_cast<compute_t *>(rsigma_);

  Vec gamma[LDGS];
  Vec beta[LDGS];
#pragma unroll
  for (int it = 0, col = c; it < LDGS; it++) {
    gamma[it].load_from(g_ptr + col * BYTES_PER_LDG);
    beta[it].load_from(b_ptr + col * BYTES_PER_LDG);
    col += THREADS_PER_ROW;
  }

  constexpr compute_t rn = 1.f / compute_t(Ktraits::COLS);
  for (int row = r; row < rows; row += gridDim.x * ROWS_PER_CTA) {
    Vec x[LDGS];
#pragma unroll
    for (int it = 0, col = c; it < LDGS; it++) {
      x[it].load_from(x_ptr + row * BYTES_PER_ROW + col * BYTES_PER_LDG);
      col += THREADS_PER_ROW;
    }
    compute_t xf[LDGS * NUM_ELTS];
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        xf[it * NUM_ELTS + jt] = compute_t(x[it].data.elt[jt]);
      }
    }

    compute_t mu_local = 0.f;

#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        mu_local += xf[it * NUM_ELTS + jt];
      }
    }

#pragma unroll
    for (int it = 1; it < THREADS_PER_WARP; it *= 2) {
      mu_local += __shfl_xor_sync(uint32_t(-1), mu_local, it);
    }
    mu_local *= rn;
    if(lane == 0){
    mu_ptr[row] = mu_local;
    }
    compute_t var_local = 0.f;

#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        compute_t diff = xf[it * NUM_ELTS + jt] - mu_local;
        var_local += diff * diff;
      }
    }

#pragma unroll
    for (int it = 1; it < THREADS_PER_WARP; it *= 2) {
      var_local += __shfl_xor_sync(uint32_t(-1), var_local, it);
    }
    compute_t rsigma = rsqrtf(var_local * rn + epsilon);
    if(lane == 0){
    rs_ptr[row] = rsigma;
    }

#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        base_t tmp = (rsigma * (xf[it * NUM_ELTS + jt] - mu_local));
        x[it].data.elt[jt] = gamma[it].data.elt[jt] *  tmp + beta[it].data.elt[jt];
      }
    }

#pragma unroll
    for (int it = 0, col = c; it < LDGS; it++) {
      x[it].store_to(y_ptr + row * BYTES_PER_ROW + col * BYTES_PER_LDG);
      col += THREADS_PER_ROW;
    }
  }
}
template<typename scalar_t>
void launch(
    at::Tensor & y, // BxSxhidden_size
    at::Tensor & mu,
    at::Tensor & rsigma,
    const at::Tensor & x, // BxSxhidden_size
    const at::Tensor & gamma,
    const at::Tensor & beta,
    const float epsilon,
    const int rows,
    const int cols,
    const int max_gridx,
    cudaStream_t stream
){

  if (cols == 1024) {
    using Ktraits = Kernel_traits<scalar_t, 1024, 4, 1>;
    const int grid =
        std::min<int>(DIVUP(rows, Ktraits::ROWS_PER_CTA), max_gridx);

    ln_fwd_kernel<Ktraits><<<grid, Ktraits::THREADS_PER_CTA, 0, stream>>>(
        y.data_ptr(), mu.data_ptr(), rsigma.data_ptr(), x.data_ptr(),
        gamma.data_ptr(), beta.data_ptr(), epsilon, rows);

  } else {
    assert(false && "Not implemented");
  }

  AT_CUDA_CHECK(cudaPeekAtLastError());
}

void ln_fwd_cuda(
    at::Tensor & y, // BxSxhidden_size
    at::Tensor & mu,
    at::Tensor & rsigma,
    const at::Tensor & x, // BxSxhidden_size
    const at::Tensor & gamma,
    const at::Tensor & beta,
    const float epsilon,
    const int rows, const int cols,
    cudaStream_t stream
){

  const auto dtype = x.scalar_type();
  const auto props = at::cuda::getCurrentDeviceProperties();
  const int max_gridx = props->maxGridSize[0];

  //TODO 
  // - Using dispatch macro costs 1% perf wtf?!?!
  // - Tune FP32 warps
  // - Add more sizes
  if (dtype == torch::kFloat16) {
    launch<half>(y, mu, rsigma, x, gamma, beta, epsilon, rows, cols, max_gridx, stream);
  } else if (dtype == torch::kFloat32) {
    launch<float>(y, mu, rsigma, x, gamma, beta, epsilon, rows, cols, max_gridx, stream);
  } else {
    assert(false && "Not implemented");
  }

}