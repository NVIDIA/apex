#include "utils.cuh"
#include "ln_kernel_traits.h"
#include "ATen/cuda/CUDAContext.h"

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::THREADS_PER_CTA) void ln_bwd_kernel(void * __restrict__ dx_,
                                                                          void * __restrict__ dg_,
                                                                          void * __restrict__ db_,
                                                                          const void * __restrict__ dw_,
                                                                          const void * __restrict__ x_,
                                                                          const void * __restrict__ mu_,
                                                                          const void * __restrict__ rs_,
                                                                          const void * __restrict__ g_,
                                                                          const int rows
                                                                        ){
  using Vec = typename Ktraits::Vec;

  enum { BYTES_PER_LDG = Ktraits::BYTES_PER_LDG };
  enum { ROWS_PER_CTA = Ktraits::ROWS_PER_CTA };
  enum { WARPS_M = Ktraits::WARPS_M };
  enum { WARPS_N = Ktraits::WARPS_N };
  enum { THREADS_PER_ROW = Ktraits::THREADS_PER_ROW };
  enum { COLS = Ktraits::COLS };
  enum { BYTES_PER_ROW = Ktraits::BYTES_PER_ROW };
  enum { LDGS = BYTES_PER_ROW / Ktraits::BYTES_PER_ROW_PER_CTA };
  static_assert(LDGS * Ktraits::BYTES_PER_ROW_PER_CTA == BYTES_PER_ROW, "");
  enum { NUM_ELTS = Vec::NUM_ELTS };
  using vec_t = typename Ktraits::vec_t;
  using base_t = typename Ktraits::base_t;
  using compute_t = typename Ktraits::compute_t;
  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;
  const int lane = tidx % THREADS_PER_WARP;
  const int warp = tidx / THREADS_PER_WARP;
  const int warp_m = warp / Ktraits::WARPS_N;
  const int warp_n = warp % Ktraits::WARPS_N;
  const int tid_r = warp_n * THREADS_PER_WARP + lane;

  const int r = bidx * Ktraits::ROWS_PER_CTA + warp_m;
  const int c = warp_n * THREADS_PER_WARP + lane;

  const char *dw_ptr = static_cast<const char *>(dw_);
  const char *x_ptr = static_cast<const char *>(x_);
  const char *g_ptr = static_cast<const char *>(g_);
  char *dx_ptr = static_cast<char *>(dx_);
  const compute_t *mu_ptr = static_cast<const compute_t *>(mu_);
  const compute_t *rs_ptr = static_cast<const compute_t *>(rs_);
  static_assert(COLS == THREADS_PER_ROW * LDGS * NUM_ELTS, "");

  // smem for final reduction
  //__shared__ compute_t smem_[ROWS_PER_CTA * COLS];
  extern __shared__ compute_t smem_[];
  // static_assert(sizeof(smem_dw_sum) == 32*1024,"");
  // Using the grid stride loop we can assign multiple rows to each thread
  // by using a number of CTAs smaller than rows / ROWS_PER_CTA
  // We accumulate them here, one in smem, one in registers, because the smem
  // capacity is limited compute_t * dw_sum = &smem_dw_sum[warp_m * COLS + tid_r
  // * LDGS * NUM_ELTS];
  compute_t dwy_sum[LDGS * NUM_ELTS];
  compute_t dw_sum[LDGS * NUM_ELTS];

  memset(dwy_sum, 0, sizeof(compute_t) * LDGS * NUM_ELTS);
  memset(dw_sum, 0, sizeof(compute_t) * LDGS * NUM_ELTS);
  // Debug 8 rows, 4B, 1024 cols

  __shared__ compute_t smem_mdy[ROWS_PER_CTA * WARPS_N];
  __shared__ compute_t smem_mdyy[ROWS_PER_CTA * WARPS_N];
  compute_t *mdy_shared = &smem_mdy[warp_m * WARPS_N];
  compute_t *mdyy_shared = &smem_mdyy[warp_m * WARPS_N];

  constexpr float rn = 1.f / float(COLS);
  Vec gamma[LDGS];
  int col = c;
#pragma unroll
  for (int it = 0; it < LDGS; it++) {
    gamma[it].load_from(g_ptr + col * BYTES_PER_LDG);
    col += Ktraits::THREADS_PER_ROW;
  }
  // TODO if ROWS_PER_CTA does not divice rows, we might get divergence in the
  // last blocks with syncthreads!
  // grid stride over rows
  #pragma unroll 1
  for (int row = r; row < rows; row += gridDim.x * ROWS_PER_CTA) {
    const compute_t mu_r = mu_ptr[row];
    const compute_t rs_r = rs_ptr[row];
    Vec dw[LDGS], x[LDGS], dx[LDGS];
    int col = c;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
      dw[it].load_from(dw_ptr + row * BYTES_PER_ROW + col * BYTES_PER_LDG);
      x[it].load_from(x_ptr + row * BYTES_PER_ROW + col * BYTES_PER_LDG);
      col += THREADS_PER_ROW;
    }
    // local reductions
    compute_t dy[LDGS * NUM_ELTS];
    compute_t y[LDGS * NUM_ELTS];

    compute_t mdy_local = 0.f;
    compute_t mdyy_local = 0.f;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < Vec::NUM_ELTS; jt++) {
        compute_t x_tmp = x[it].data.elt[jt];
        compute_t y_tmp = rs_r * (x_tmp - mu_r);
        compute_t dy_tmp = gamma[it].data.elt[jt] * dw[it].data.elt[jt];
        compute_t dw_tmp = dw[it].data.elt[jt];

        mdy_local += dy_tmp;
        mdyy_local += dy_tmp * y_tmp;

        dy[it * NUM_ELTS + jt] = dy_tmp;
        y[it * NUM_ELTS + jt] = y_tmp;

        dwy_sum[it * NUM_ELTS + jt] += dw_tmp * y_tmp;
        dw_sum[it * NUM_ELTS + jt] += dw_tmp;
      }
    }

    // reduction across row for mdy, mdyy
    if (WARPS_N == 1) { // no need to go through smem!
#pragma unroll
      for (int it = 1; it < THREADS_PER_WARP; it *= 2) {
        mdy_local += __shfl_xor_sync(uint32_t(-1), mdy_local, it);
        mdyy_local += __shfl_xor_sync(uint32_t(-1), mdyy_local, it);
      }

      mdy_local *= rn;
      mdyy_local *= rn;

    } else {

#pragma unroll
      for (int it = 16; it > 0; it /= 2) {
        mdy_local += __shfl_down_sync(uint32_t(-1), mdy_local, it);
        mdyy_local += __shfl_down_sync(uint32_t(-1), mdyy_local, it);
      } // lane 0 holds the result!

      if (lane == 0) {
        mdy_shared[warp_n] = mdy_local;
        mdyy_shared[warp_n] = mdyy_local;
      }

      __syncthreads();
      if (warp_n == 0 && lane == 0) {
        mdy_local = 0.f;
        mdyy_local = 0.f;
        for (int it = 0; it < WARPS_N; it++) {
          mdy_local += mdy_shared[it];
          mdyy_local += mdyy_shared[it];
        }
        mdy_shared[0] = mdy_local;
        mdyy_shared[0] = mdyy_local;
      }
      __syncthreads();

      mdy_local = mdy_shared[0] * rn;
      mdyy_local = mdyy_shared[0] * rn;
    }

#pragma unroll
    for (int it = 0; it < LDGS; it++) {
#pragma unroll
      for (int jt = 0; jt < NUM_ELTS; jt++) {
        compute_t dy_tmp = dy[it * NUM_ELTS + jt];
        compute_t y_tmp = y[it * NUM_ELTS + jt];
        compute_t dx_tmp =
            compute_t(rs_r) * (dy_tmp - mdyy_local * y_tmp - mdy_local);
        dx[it].data.elt[jt] = dx_tmp;
      }
    }

    col = c;
#pragma unroll
    for (int it = 0; it < LDGS; it++) {
      dx[it].store_to(dx_ptr + row * BYTES_PER_ROW + col * BYTES_PER_LDG);
      col += Ktraits::THREADS_PER_ROW;
    }

  } // end: grid stride loop

  // Finalize reduction of part dgamma and dbeta for this CTA
  // by reducing over the rows held across the WARPS_M warps

  enum { NUM_RES = COLS / Ktraits::THREADS_PER_CTA };
  static_assert(NUM_RES * Ktraits::THREADS_PER_CTA == COLS, "");

  compute_t *smem_write;

  smem_write = &smem_[warp_m * COLS + tid_r * NUM_ELTS];
#pragma unroll
  for (int it = 0; it < LDGS; it++) {
#pragma unroll
    for (int jt = 0; jt < NUM_ELTS; jt++) {
      smem_write[jt] = dw_sum[it * NUM_ELTS + jt];
    }
    smem_write += THREADS_PER_ROW * NUM_ELTS;
  }
  __syncthreads();
  compute_t cta_dw_sum[NUM_RES];
  memset(cta_dw_sum, 0, sizeof(compute_t) * NUM_RES);
  for (int it = 0; it < ROWS_PER_CTA; it++) {
    for (int jt = 0; jt < NUM_RES; jt++) {
      cta_dw_sum[jt] += smem_[it * COLS + tidx + jt * Ktraits::THREADS_PER_CTA];
    }
  }
  __syncthreads();

  smem_write = &smem_[warp_m * COLS + tid_r * NUM_ELTS];
#pragma unroll
  for (int it = 0; it < LDGS; it++) {
#pragma unroll
    for (int jt = 0; jt < NUM_ELTS; jt++) {
      smem_write[jt] = dwy_sum[it * NUM_ELTS + jt];
    }
    smem_write += THREADS_PER_ROW * NUM_ELTS;
  }
  __syncthreads();
  compute_t cta_dwy_sum[NUM_RES];
  memset(cta_dwy_sum, 0, sizeof(compute_t) * NUM_RES);
  for (int it = 0; it < ROWS_PER_CTA; it++) {
    for (int jt = 0; jt < NUM_RES; jt++) {
      cta_dwy_sum[jt] +=
          smem_[it * COLS + tidx + jt * Ktraits::THREADS_PER_CTA];
    }
  }

  compute_t *dgamma_part = static_cast<compute_t *>(dg_) + bidx * COLS + tidx;
  for (int jt = 0; jt < NUM_RES; jt++) {
    *dgamma_part = cta_dwy_sum[jt];
    dgamma_part += Ktraits::THREADS_PER_CTA;
  }

  compute_t *dbeta_part = static_cast<compute_t *>(db_) + bidx * COLS + tidx;
  for (int jt = 0; jt < NUM_RES; jt++) {
    *dbeta_part = cta_dw_sum[jt];
    dbeta_part += Ktraits::THREADS_PER_CTA;
  }
}

template<typename Ktraits, typename out_t>
__global__ __launch_bounds__(Ktraits::THREADS_PER_CTA) void ln_bwd_finalize_kernel(void * __restrict__ dg_,
                                                                                   void * __restrict__ db_,
                                                                                   const void * __restrict__ dg_part_,
                                                                                   const void * __restrict__ db_part_,
                                                                                   const int rows
                                                                                  ){
    using Vec = typename Ktraits::Vec;
    enum { NUM_ELTS = Vec::NUM_ELTS };


    using vec_t = typename Ktraits::vec_t;
    using base_t = typename Ktraits::base_t;
    using compute_t = typename Ktraits::compute_t;

    enum { BYTES_PER_LDG = Ktraits::BYTES_PER_LDG };
    enum { ROWS_PER_CTA = Ktraits::ROWS_PER_CTA };
    enum { WARPS_M = Ktraits::WARPS_M };
    enum { WARPS_N = Ktraits::WARPS_N };
    enum { THREADS_PER_ROW = Ktraits::THREADS_PER_ROW };
    enum { COLS = Ktraits::COLS };
    enum { BYTES_PER_ROW = Ktraits::BYTES_PER_ROW };
    enum {VEC_COLS = BYTES_PER_ROW / BYTES_PER_LDG};
    //dbg
    static_assert(VEC_COLS == COLS / NUM_ELTS, ""); 
    //static_assert(VEC_COLS == 1024,"");
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int lane = tidx % THREADS_PER_WARP;
    const int warp = tidx / THREADS_PER_WARP;
    const int warp_m = warp / Ktraits::WARPS_N;
    const int warp_n = warp % Ktraits::WARPS_N;
    const int tid_c = warp_n * THREADS_PER_WARP + lane;
    const int c =bidx * THREADS_PER_ROW + tid_c;
    const int r = warp_m;
    
    __shared__ compute_t smem_[(WARPS_M - 1) * THREADS_PER_ROW * NUM_ELTS];
    
    //Will probably run this with WARPS_N = 1 and grid = 1024 / (32*4) = 8, or NUM_ELTS=1 and grid = 32 
    // and WARPS_M = 4 (or 1??)
    for(int col = c; col < VEC_COLS; col += gridDim.x * THREADS_PER_ROW){
      const char* dg_part_ptr = static_cast<const char*>(dg_part_) + r * BYTES_PER_ROW + col * BYTES_PER_LDG;
      const char* db_part_ptr = static_cast<const char*>(db_part_) + r * BYTES_PER_ROW + col * BYTES_PER_LDG;

      compute_t dg_sum[NUM_ELTS];
      compute_t db_sum[NUM_ELTS];
      memset(dg_sum, 0, sizeof(compute_t) * NUM_ELTS);
      memset(db_sum, 0, sizeof(compute_t) * NUM_ELTS);
      #pragma unroll
      for(int row = r; row < rows;row += ROWS_PER_CTA){
        Vec dg;
        Vec db;
        dg.load_from(dg_part_ptr);
        db.load_from(db_part_ptr);
        dg_part_ptr += ROWS_PER_CTA * BYTES_PER_ROW;
        db_part_ptr += ROWS_PER_CTA * BYTES_PER_ROW;

        #pragma unroll
        for (int jt = 0; jt < NUM_ELTS; jt++) {
          dg_sum[jt] += dg.data.elt[jt];
          db_sum[jt] += db.data.elt[jt];
        }
      }

      // Finalize the reduction across rows of the CTA
      compute_t * smem_write;
      smem_write = smem_ + (warp_m -1) *THREADS_PER_ROW * NUM_ELTS + tid_c;

      if (warp_m > 0) {
#pragma unroll
        for (int jt = 0; jt < NUM_ELTS; jt++) {
          *smem_write = dg_sum[jt];
          smem_write+=THREADS_PER_ROW;
        }
      }
      __syncthreads();
      compute_t *smem_read ;
      smem_read = smem_ + tid_c ;
      if (warp_m == 0) {
#pragma unroll
        for (int it = 0; it < WARPS_M - 1; it++) {
#pragma unroll
          for (int jt = 0; jt < NUM_ELTS; jt++) {
            dg_sum[jt] += *smem_read;
            smem_read += THREADS_PER_ROW;
          }
        }
      }

      __syncthreads();

      smem_write = smem_ + (warp_m -1) *THREADS_PER_ROW * NUM_ELTS + tid_c;

      if (warp_m > 0) {
#pragma unroll
        for (int jt = 0; jt < NUM_ELTS; jt++) {
          *smem_write = db_sum[jt];
          smem_write+=THREADS_PER_ROW;
        }
      }
      __syncthreads();
      smem_read = smem_ + tid_c;
      if (warp_m == 0) {
#pragma unroll
        for (int it = 0; it < WARPS_M - 1; it++) {
#pragma unroll
          for (int jt = 0; jt < NUM_ELTS; jt++) {
            db_sum[jt] += *smem_read;
            smem_read += THREADS_PER_ROW;
          }
        }

        using vout_t = typename Vec_type<sizeof(out_t) * NUM_ELTS>::Type;
        union {
          vout_t raw;
          out_t elt[NUM_ELTS];
        } dg_out, db_out;

        // out_t dg_out[NUM_ELTS], db_out[NUM_ELTS];
#pragma unroll
        for (int jt = 0; jt < NUM_ELTS; jt++) {
          dg_out.elt[jt] = dg_sum[jt];
          db_out.elt[jt] = db_sum[jt];
        }
        vout_t *dg_ptr = reinterpret_cast<vout_t *>(dg_) + col ;
        vout_t *db_ptr = reinterpret_cast<vout_t *>(db_) + col ;
        *dg_ptr = dg_out.raw;
        *db_ptr = db_out.raw;
      }
    }
}

template<typename scalar_t>
void launch(at::Tensor &dx, at::Tensor &dgamma, at::Tensor &dbeta,
                 at::Tensor &dgamma_part, at::Tensor &dbeta_part,
                 const at::Tensor &dw, const at::Tensor &x,
                 const at::Tensor &mu, const at::Tensor &rsigma,
                 const at::Tensor &gamma, const int rows, const int cols, const int gridx, cudaStream_t stream){

  if (cols == 1024) {
    using Ktraits = Kernel_traits<scalar_t, 1024, 4, 1>;

    if (Ktraits::SMEM_BYTES >= 48 * 1024) {
      AT_CUDA_CHECK(cudaFuncSetAttribute(
          ln_bwd_kernel<Ktraits>, cudaFuncAttributeMaxDynamicSharedMemorySize,
          Ktraits::SMEM_BYTES));
    }

    ln_bwd_kernel<Ktraits>
        <<<gridx, Ktraits::THREADS_PER_CTA, Ktraits::SMEM_BYTES, stream>>>(
            dx.data_ptr(), dgamma_part.data_ptr(), dbeta_part.data_ptr(),
            dw.data_ptr(), x.data_ptr(), mu.data_ptr(), rsigma.data_ptr(),
            gamma.data_ptr(), rows);

    using Ktraits2 = Kernel_traits<float, 1024, 16, 1, 4>;

    constexpr int grid2 =
        DIVUP(1024, Ktraits2::THREADS_PER_ROW * Ktraits2::Vec::NUM_ELTS);

    ln_bwd_finalize_kernel<Ktraits2, scalar_t>
        <<<grid2, Ktraits2::THREADS_PER_CTA, 0, stream>>>(
            dgamma.data_ptr(), dbeta.data_ptr(), dgamma_part.data_ptr(),
            dbeta_part.data_ptr(), gridx);
  } else {
    assert(false && "Not implemented");
  }

  AT_CUDA_CHECK(cudaPeekAtLastError());
}

void ln_bwd_cuda(at::Tensor &dx, at::Tensor &dgamma, at::Tensor &dbeta,
                 const at::Tensor &dw, const at::Tensor &x,
                 const at::Tensor &mu, const at::Tensor &rsigma,
                 const at::Tensor &gamma, const int rows, const int cols, cudaStream_t stream) {


  const auto dtype = x.scalar_type();


  const auto props = at::cuda::getCurrentDeviceProperties();
  const int smCount = props->multiProcessorCount;
  // Launch 2 CTAs per SM 
  const int grid = 2 * smCount;

  //request workspace for two-step reduction. We always reduce in FP32.
  auto opts = x.options();
  auto dbeta_part = torch::empty({grid, cols}, opts.dtype(torch::kFloat32));
  auto dgamma_part = torch::empty({grid, cols}, opts.dtype(torch::kFloat32));

  if (dtype == torch::kFloat16) {
    launch<half>(dx, dgamma, dbeta, dgamma_part, dbeta_part, dw, x, mu, rsigma, gamma, rows, cols, grid, stream);
  } else if (dtype == torch::kFloat32) {
    launch<float>(dx, dgamma, dbeta, dgamma_part, dbeta_part, dw, x, mu, rsigma, gamma, rows, cols, grid, stream);
  } else {
    assert(false && "Not implemented");
  }

}