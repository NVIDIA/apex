/***************************************************************************************************
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR 
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND 
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE 
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <traits.h>
#include <group_norm_nhwc.h>
#include <group_norm_nhwc_fwd_one_pass.h>
#include <group_norm_nhwc_bwd_one_pass.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <type_traits>

template <typename T>
float inline unpack(const T& x) {
  return {};
}

template <>
float inline unpack(const __half& x) {
  return __half2float(x);
}


template <>
float inline unpack(const __nv_bfloat16& x) {
  return __bfloat162float(x);
}

template <>
float inline unpack(const float& x) {
  return x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void check_results(const char *name,
                   const T *out, 
                   const T *ref, 
                   size_t elts, 
                   float tol) {

  // The number of errors.
  int failed = 0;
  // The number of infinite value.
  int infs = 0;
  // The min/max values.
  float min_val = FLT_MAX, max_val = -FLT_MAX, max_err = 0.f;
  // The total sum of error.
  double sum_err = 0.0;

  // The case we are checking.
  printf("\e[1;34mchecking.....................: %s\e[0m\n", name);
  fflush(stdout);

  // Iterate over the different values.
  for( size_t ii = 0; ii < elts; ++ii ) {
    float a = unpack(out[ii]);
    float b = unpack(ref[ii]);

    // Compute the absolute norms.
    float abs_a = fabsf(a);
    float abs_b = fabsf(b);

    // Compute the error.
    float den = abs_a + abs_b; 
    // Is one of the quantities very small?
    bool is_small = abs_a <= tol || abs_b <= tol || den <= tol;
    // The error.
    float err = is_small ? fabsf(a-b) : fabsf(a-b) / den;
    // Is the result ok?
    bool ok = !isnan(a) && !isnan(b) && err <= tol;
    
    // Print the error.
    if( !ok && (failed < 10 || err > max_err) ) {

      fprintf(stderr, ">> invalid result for ii=%lu:\n", ii);
      if (std::is_same<T, __half>::value || std::is_same<T, __nv_bfloat16>::value) {
        // The data.
        fprintf(stderr, ">>   found...: 0x%04x (%10.6f)\n", reinterpret_cast<const uint16_t&>(out[ii]), a);
        fprintf(stderr, ">>   expected: 0x%04x (%10.6f)\n", reinterpret_cast<const uint16_t&>(ref[ii]), b);
      } else if (std::is_same<T, float>::value) {
        fprintf(stderr, ">>   found...: 0x%08x (%10.6f)\n", reinterpret_cast<const uint32_t&>(a), a);
        fprintf(stderr, ">>   expected: 0x%08x (%10.6f)\n", reinterpret_cast<const uint32_t&>(b), b);
      } else {
        fprintf(stderr, "\e[1;34mUnknown type of check_results\e[0m\n");
        exit(1);
      }
      fprintf(stderr, ">>   error...: %.6f\n", err);
    }

    // Update the number of failures.
    failed += ok ? 0 : 1;

    // Measure min/max errors.
    min_val = fminf(min_val, a);
    max_val = fmaxf(max_val, a);
    max_err = fmaxf(max_err, err);

    // Accumulate the sum.
    sum_err = sum_err + (double) err;

    infs += !isfinite(a);
    infs += !isfinite(b);
  }

  if( !failed && infs < 10 ) {
    printf("\e[1;32mcheck........................: OK\e[0m\n");
  } else {
    printf("\e[1;31mcheck........................: FAILED\e[0m\n");
  }

  printf("tested.......................: %lu\n", elts);
  printf("failures.....................: %d\n", failed);
  printf("failure rate.................: %.2lf%%\n", (double) failed * 100.0 / (double) elts);
  printf("infs.........................: %d\n", infs);
  printf("tolerance....................: %.8f\n", tol);
  printf("\n");

  printf("min. value...................: %.6f\n", min_val);
  printf("max. value...................: %.6f\n", max_val);
  printf("max. error...................: %.6f\n", max_err);
  printf("sum. error...................: %.6lf\n", sum_err);
  printf("avg. error...................: %.6lf\n", sum_err / (double) elts);
  printf("\n");
}

template void check_results(const char *name, const __half *out, const __half *ref, size_t elts, float tol);

template void check_results(const char *name, const __nv_bfloat16 *out, const __nv_bfloat16 *ref, size_t elts, float tol);

template void check_results(const char *name, const float *out, const float *ref, size_t elts, float tol);

////////////////////////////////////////////////////////////////////////////////////////////////////

static void group_norm_nhwc_bwd_(void *dx_h, 
                                 float *dgamma_h,
                                 float *dbeta_h,
                                 const void *dy_h, 
                                 const void *x_h, 
                                 const float *gamma_h,
                                 const float *beta_h,
                                 const float2 *sums_h, 
                                 float epsilon,
                                 int n, 
                                 int h, 
                                 int w, 
                                 int c, 
                                 int groups,
                                 bool with_swish,
                                 bool use_fp32,
                                 bool use_bf16) {

  // The number of channels in each group.
  int channels_per_group = c / groups;
  // The normalization term to compute the means.
  float rcp_hwc_per_group = 1.f / (float) (h * w * channels_per_group);

  // The array to compute gamma.
  float *dgamma = (float*) malloc(c * sizeof(float));
  // The array to compute beta.
  float *dbeta = (float*) malloc(c * sizeof(float));

  // Set gamma/beta to 0.
  memset(dgamma, 0, c * sizeof(float));
  memset(dbeta,  0, c * sizeof(float));

  // Normalize the activations.
  for( int ni = 0; ni < n; ++ni ) {
    for( int gi = 0; gi < groups; ++gi ) {

      // The sums from the fwd pass.
      float2 sums = sums_h[ni*groups + gi];
      // The mean of X (computed during the fwd pass -- one value per batch*group).
      float x_mean = sums.x;
      // The mean of squares of X (computed during the fwd pass -- one value per batch*group).
      float x_sq_mean = sums.y;
      // The variance.
      float x_var = x_sq_mean - x_mean * x_mean;
      // The reciprocal of the standard deviation (i.e. 1.f / sqrt(var + epsilon)).
      float rcp_x_stddev = x_var <= 0.f ? 1.f : 1.f / sqrtf(x_var + epsilon);

      // TODO: We should store rcp_x_stddev instead of the sums of squares.

      // The following nested loops compute 2 means.
      float mean_1 = 0.f, mean_2 = 0.f;

      // Iterate over the activations in the group.
      for( int hi = 0; hi < h; ++hi ) {
        for( int wi = 0; wi < w; ++wi ) {
          for( int ii = 0; ii < channels_per_group; ++ii ) {

            // The channel.
            int ci = gi * channels_per_group + ii;
            // Compute the src/dst offset.
            size_t offset = (size_t) ni*h*w*c + (size_t) hi*w*c + (size_t) wi*c + (size_t) ci;
            // Convert the element at that position to float.
            float x;
            if (use_fp32) {
              x = reinterpret_cast<const float*>(x_h)[offset];
            } else if (use_bf16) {
              x = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(x_h)[offset]);
            } else {
              x = __half2float(reinterpret_cast<const __half*>(x_h)[offset]);
            }
            // The output.
            float dy;
            if (use_fp32) {
              dy = reinterpret_cast<const float*>(dy_h)[offset];
            } else if (use_bf16) {
              dy = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(dy_h)[offset]);
            } else {
              dy = __half2float(reinterpret_cast<const __half*>(dy_h)[offset]);
            }

            // Gamma.
            float gamma = gamma_h[ci];

            // X - X_mean.
            float x_minus_x_mean = x - x_mean;
            // Normalize X.
            float x_norm = x_minus_x_mean * rcp_x_stddev;

            if( with_swish ) {
              // Beta
              float beta = beta_h[ci];

              float x_gn = x_norm * gamma + beta;
              float s = sigmoid(x_gn);
              dy = dy * s * (1.f + x_gn * (1.f - s));
            }

            // Compute the gradient for beta.
            dbeta[ci] += dy;

            // Compute the gradient for gamma.
            dgamma[ci] += dy * x_norm;

            // The gradient that enters the x_norm node.
            float dx_norm = dy * gamma;

            // Accumulators over 2 means
            mean_1 += x_norm * dx_norm;
            mean_2 += dx_norm;

          } // ii
        } // wi
      } // hi
      
      mean_1 *= rcp_hwc_per_group;
      mean_2 *= rcp_hwc_per_group;

      // Iterate over the activations in the group.
      for( int hi = 0; hi < h; ++hi ) {
        for( int wi = 0; wi < w; ++wi ) {
          for( int ii = 0; ii < channels_per_group; ++ii ) {

            // The channel.
            int ci = gi * channels_per_group + ii;
            // Compute the src/dst offset.
            size_t offset = (size_t) ni*h*w*c + (size_t) hi*w*c + (size_t) wi*c + (size_t) ci;
            float x;
            if (use_fp32) {
              x = reinterpret_cast<const float*>(x_h)[offset];
            } else if (use_bf16) {
              x = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(x_h)[offset]);
            } else {
              x = __half2float(reinterpret_cast<const __half*>(x_h)[offset]);
            }
            // The output.
            float dy;
            if (use_fp32) {
              dy = reinterpret_cast<const float*>(dy_h)[offset];
            } else if (use_bf16) {
              dy = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(dy_h)[offset]);
            } else {
              dy = __half2float(reinterpret_cast<const __half*>(dy_h)[offset]);
            }

            // Gamma.
            float gamma = gamma_h[ci];

            // X - X_mean.
            float x_minus_x_mean = x - x_mean;
            // Normalize X.
            float x_norm = x_minus_x_mean * rcp_x_stddev;

            if( with_swish ) {
              // Beta
              float beta = beta_h[ci];

              float x_gn = x_norm * gamma + beta;
              float s = sigmoid(x_gn);
              dy = dy * s * (1.f + x_gn * (1.f - s));
            }

            // The gradient that enters the x_norm node.
            float dx_norm = dy * gamma;

            // Input gradient
            float dx = (dx_norm - (x_norm * mean_1 + mean_2)) * rcp_x_stddev;

            // Set the output gradient.
            if (use_fp32) {
              reinterpret_cast<float*>(dx_h)[offset] = dx;
            } else if (use_bf16) {
              reinterpret_cast<__nv_bfloat16*>(dx_h)[offset] = __float2bfloat16_rn(dx);
            } else {
              reinterpret_cast<__half*>(dx_h)[offset] = __float2half_rn(dx);
            }

          } // ii
        } // wi
      } // hi

    } // gi
  } // ni

  // Store gamma/beta.
  for( int ci = 0; ci < c; ++ci ) {
    dgamma_h[ci] = dgamma[ci];
    dbeta_h [ci] = dbeta [ci];
  }

  // Release temporary memory.
  free(dgamma);
  free(dbeta);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static void group_norm_nhwc_fwd_(void *y_h, 
                                 const void *x_h, 
                                 const float *gamma_h, 
                                 const float *beta_h, 
                                 float epsilon,
                                 int n, 
                                 int h, 
                                 int w, 
                                 int c, 
                                 int groups,
                                 bool with_swish,
                                 bool use_fp32,
                                 bool use_bf16) {

  // The number of channels in each group.
  int channels_per_group = c / groups;

  // The normalization term to compute the means.
  float inv_hwcg = 1.f / (float) (h * w * channels_per_group);

  // Normalize the activations.
  for( int ni = 0; ni < n; ++ni ) {
    for( int gi = 0; gi < groups; ++gi ) {

      // The sums to compute the mean/variance for that group.
      float sum = 0.f, sum_sq = 0.f;

      // Iterate over the activations in the group.
      for( int hi = 0; hi < h; ++hi ) {
        for( int wi = 0; wi < w; ++wi ) {
          for( int ii = 0; ii < channels_per_group; ++ii ) {

            // The channel.
            int ci = gi * channels_per_group + ii;
            // Compute the src/dst offset.
            size_t offset = (size_t) ni*h*w*c + (size_t) hi*w*c + (size_t) wi*c + (size_t) ci;
            // Convert the element at that position to float.
            float x;
            if (use_fp32) {
              x = reinterpret_cast<const float*>(x_h)[offset];
            } else if (use_bf16) {
              x = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(x_h)[offset]);
            } else {
              x = __half2float(reinterpret_cast<const __half*>(x_h)[offset]);
            }

            // Update the sums.
            sum += x;
            sum_sq += x * x;

          } // ii
        } // wi
      } // hi

      // Compute the mean.
      float mean = sum * inv_hwcg;
      // Compute the average value for the squares.
      float mean_sq = sum_sq * inv_hwcg;
      // Compute the variance.
      float var = mean_sq - (mean * mean);
      // Invert the variance.
      float inv_stddev = var <= 0.f ? 1.f : (1.f / sqrtf(var + epsilon));

      // Iterate over the data to normalize the output.
      for( int hi = 0; hi < h; ++hi ) {
        for( int wi = 0; wi < w; ++wi ) {
          for( int ii = 0; ii < channels_per_group; ++ii ) {

            // The channel.
            int ci = gi * channels_per_group + ii;
            // Compute the src/dst offset.
            size_t offset = (size_t) ni*h*w*c + (size_t) hi*w*c + (size_t) wi*c + (size_t) ci;
            // Normalize.
            float x;
            if (use_fp32) {
              x = reinterpret_cast<const float*>(x_h)[offset];
            } else if (use_bf16) {
              x = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(x_h)[offset]);
            } else {
              x = __half2float(reinterpret_cast<const __half*>(x_h)[offset]);
            }
            float y = (x - mean) * inv_stddev;
            // Scale with gamma and add beta.
            y = y * gamma_h[ci] + beta_h[ci];
            // Apply swish (if needed).
            if( with_swish ) {
              y = y * sigmoid(y);
            }
            // Store the result.
            if (use_fp32) {
              reinterpret_cast<float*>(y_h)[offset] = y;
            } else if (use_bf16) {
              reinterpret_cast<__nv_bfloat16*>(y_h)[offset] = __float2bfloat16_rn(y);
            } else {
              reinterpret_cast<__half*>(y_h)[offset] = __float2half_rn(y);
            }

          } // ii
        } // wi
      } // hi
    } // gi
  } // ni
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void random_data(T *dst_h, size_t n, bool use_1s, int range = 3) {
  for( size_t ii = 0; ii < n; ++ii ) {
    float x = 1.f;
    if( !use_1s ) {
      x = (float) (rand() % range - (range / 2));
    }
    if (std::is_same<T, __half>::value) {
      dst_h[ii] = __float2half_rn(x);
    } else if (std::is_same<T, float>::value) {
      dst_h[ii] = x;
    } else if (std::is_same<T, __nv_bfloat16>::value) {
      dst_h[ii] = __float2bfloat16_rn(x);
    } else {
      fprintf(stderr, "\e[1;34mUnknown type of random_data\e[0m\n");
      exit(1);
    }
  }
}

template void random_data(float *dst_h, size_t n, bool use_1s, int range);

template void random_data(__half *dst_h, size_t n, bool use_1s, int range);

template void random_data(__nv_bfloat16 *dst_h, size_t n, bool use_1s, int range);

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class Mode { FWD_INFERENCE, FWD_TRAINING, BWD };

////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {

  // The tensor size.
  int n = 2, h = 64, w = 64, c = 320, groups = 32;
  // The default mode is inference.
  Mode mode = Mode::FWD_INFERENCE;
  // The constant epsilon for sqrt(var + epsilon).
  float epsilon = 1.e-5f;
  // Do we fuse with the Swish activation function?
  bool with_swish = false;
  // Do we use the one-pass kernel?
  bool use_one_pass = false;
  // The number of runs to time the code.
  int runs = 1;
  // Do we use 1s for the input data.
  bool use_1s = false;
  // The tolerance to check the results.
  float tol = 1.e-3f;
  // Do we skip the checks?
  bool skip_checks = false;
  // Do we output csv format only
  bool csv_output = false;
  // Use fp32 IO
  bool use_fp32 = false;
  // Use bf16 IO
  bool use_bf16 = false;

  // Parse the parameters.
  for( int ii = 1; ii < argc; ++ii ) {
    if( !strcmp(argv[ii], "-1s") ) {
      use_1s = true;
    } else if( !strcmp(argv[ii], "-bwd") ) {
      mode = Mode::BWD;
    } else if( !strcmp(argv[ii], "-c") && ++ii < argc ) {
      c = strtol(argv[ii], nullptr, 10);
    } else if( !strcmp(argv[ii], "-epsilon") && ++ii < argc ) {
      epsilon = (float) strtod(argv[ii], nullptr);
    } else if( !strcmp(argv[ii], "-fwd") ) {
      mode = Mode::FWD_INFERENCE;
    } else if( !strcmp(argv[ii], "-fwd-tr") ) {
      mode = Mode::FWD_TRAINING;
    } else if( !strcmp(argv[ii], "-groups") && ++ii < argc ) {
      groups = strtol(argv[ii], nullptr, 10);
    } else if( !strcmp(argv[ii], "-h") && ++ii < argc ) {
      h = strtol(argv[ii], nullptr, 10);
    } else if( !strcmp(argv[ii], "-n") && ++ii < argc ) {
      n = strtol(argv[ii], nullptr, 10);
    } else if( !strcmp(argv[ii], "-one-pass") ) {
      use_one_pass = true;
    } else if( !strcmp(argv[ii], "-runs") && ++ii < argc ) {
      runs = strtol(argv[ii], nullptr, 10);
    } else if( !strcmp(argv[ii], "-skip-checks") ) {
      skip_checks = true;
    } else if( !strcmp(argv[ii], "-tol") && ++ii < argc ) {
      tol = (float) strtod(argv[ii], nullptr);
    } else if( !strcmp(argv[ii], "-w") && ++ii < argc ) {
      w = strtol(argv[ii], nullptr, 10);
    } else if( !strcmp(argv[ii], "-with-swish") ) {
      with_swish = true;
    } else if( !strcmp(argv[ii], "-csv") ) {
      csv_output = true;
    } else if( !strcmp(argv[ii], "-fp32") ) {
      use_fp32 = true;
    } else if( !strcmp(argv[ii], "-bf16") ) {
      use_bf16 = true;
    } else if( ii < argc ) {
      fprintf(stderr, "Unknown argument: %s\n", argv[ii]);
      return 1;
    } else {
      fprintf(stderr, "Argument %s requires a value\n", argv[ii-1]);
      return 1;
    }
  }

  if (use_bf16 && use_fp32) {
    fprintf(stderr, "Can't use fp32 and bf16 IO at the same time\n");
    return 1;
  }

  // Header.
  if (!csv_output) {
    printf("\n");
    printf("#######################################################################\n");
    printf("# Group Norm NHWC + Swish kernel\n");
    printf("# --\n");
    printf("# Compiled on %s\n", __DATE__);
    printf("#######################################################################\n");
    printf("\n");
  }

  // GPU info.
  cudaDeviceProp props;
  CHECK_CUDA(cudaGetDeviceProperties(&props, 0));
  if (!csv_output) {
    printf("device.......................: %s\n", props.name);
    printf("cc...........................: %d.%d\n", props.major, props.minor);
    printf("# of sms.....................: %d\n", props.multiProcessorCount);
  }

  // Dram peak bandwidth.
  float dram_clock = props.memoryClockRate / 1.e6f;
  float dram_peak = 2.f * dram_clock * props.memoryBusWidth / 8.f;
  if (!csv_output) {
    printf("dram clock...................: %.3f GHz\n", dram_clock);
    printf("dram peak....................: %.3f TB/s\n", dram_peak * 1.e-3f);
    printf("\n");
  }

  // Output the problem size.
  if (!csv_output) {
    printf("n............................: %d\n", n);
    printf("h............................: %d\n", h);
    printf("w............................: %d\n", w);
    printf("c............................: %d\n", c);
    printf("groups.......................: %d\n", groups);
    printf("epsilon......................: %f\n", epsilon);
    printf("with swish...................: %s\n", with_swish ? "true" : "false");
    printf("channels per group...........: %d\n", c / groups);
    if( mode == Mode::BWD ) {
      printf("mode.........................: bwd\n");
    } else if( mode == Mode::FWD_INFERENCE ) {
      printf("mode.........................: fwd inference\n");
    } else if( mode == Mode::FWD_TRAINING ) { 
      printf("mode.........................: fwd training\n");
    } else {
      assert(false);
    }
    printf("\n");
  }

  // Compute the SOL.
  double bytes = 0;
  int32_t io_bytes = use_fp32 ? sizeof(float) : sizeof(__half);
  if( mode != Mode::BWD ) {
    bytes = (double) n * h * w * c * io_bytes + // src
            (double)             c * 4        + // gamma
            (double)             c * 4        + // beta
            (double) n * h * w * c * io_bytes;  // out
  } else {
    bytes = (double) n * h * w * c * io_bytes * 2 + // src, dsrc
            (double)             c * 4 * 2        + // gamma, dgamma
            (double)             c * 4 * 2        + // beta, dbeta
            (double) n * h * w * c * io_bytes * 1;  // dout
  }
  double gbytes = bytes * 1.e-9;
  double dram_sol = gbytes / dram_peak * 1.e3;
  if (!csv_output) {
    printf("bytes........................: %.3lfGB\n", gbytes);
    printf("dram sol.....................: %.6lfms\n", dram_sol);

    // The number of runs to measure performance.
    printf("runs.........................: %d\n", runs);
    printf("\n");
  }

  // The number of elements in the x tensor. The layout is N x H x W x C.
  size_t x_elts = (size_t) n * h * w * c;
  // The size of the src in bytes.
  size_t x_sz = x_elts * io_bytes;

  // Allocate the src/dst on the host.
  void *x_h = malloc(x_sz);
  void *y_h = malloc(x_sz);

  // Allocate src/dst on the device.
  void *x_d, *y_d;
  CHECK_CUDA(cudaMalloc((void**) &x_d, x_sz));
  CHECK_CUDA(cudaMalloc((void**) &y_d, x_sz));

  // The number of elements in the gamma/beta array.
  size_t gamma_elts = (size_t) c;
  // The size of the gamma/beta array in bytes.
  size_t gamma_sz = gamma_elts * sizeof(float);
  // Allocate gamma/beta on the host.
  float *gamma_h = (float*) malloc(gamma_sz);
  // Allocate gamma/beta on the device.
  float *gamma_d;
  CHECK_CUDA(cudaMalloc((void**) &gamma_d, gamma_sz));

  // Allocate gamma/beta on the host.
  float *beta_h = (float*) malloc(gamma_sz);
  // Allocate gamma/beta on the device.
  float *beta_d;
  CHECK_CUDA(cudaMalloc((void**) &beta_d, gamma_sz));

  // Allocate the reference on the host (to be computed on the host).
  void *y_ref_h = nullptr;
  if( !skip_checks ) {
    y_ref_h = malloc(x_sz);
  }

  // Allocate the src/dst on the host for the gradients (bwd).
  void *dx_h = nullptr, *dy_h = nullptr; 
  if( mode == Mode::BWD ) {
    dx_h = malloc(x_sz);
    dy_h = malloc(x_sz);
  }

  // Allocate src/dst on the device.
  void *dx_d = nullptr, *dy_d = nullptr;
  if( mode == Mode::BWD ) {
    CHECK_CUDA(cudaMalloc((void**) &dx_d, x_sz));
    CHECK_CUDA(cudaMalloc((void**) &dy_d, x_sz));
  }

  // The gradients for gamma and beta on the host.
  float *dgamma_h = nullptr, *dbeta_h = nullptr;
  if( mode == Mode::BWD ) {
    dgamma_h = (float*) malloc(gamma_sz);
    dbeta_h  = (float*) malloc(gamma_sz);
  }

  // The gradients for gamma and beta on the device.
  float *dgamma_d = nullptr, *dbeta_d = nullptr;
  if( mode == Mode::BWD ) {
    CHECK_CUDA(cudaMalloc((void**) &dgamma_d, gamma_sz));
    CHECK_CUDA(cudaMalloc((void**) &dbeta_d,  gamma_sz));
  }

  // The number of sums for the bwd pass.
  size_t sums_elts = mode == Mode::FWD_INFERENCE ? 0 : n * groups;
  // The size needed to store that array.
  size_t sums_sz = sums_elts * sizeof(float2);

  // The sums for the bwd pass on the host.
  float2 *sums_h = nullptr;
  if( sums_sz > 0 ) {
    sums_h = (float2*) malloc(sums_sz);
  }

  // The sums for the bwd pass on the device.
  float2 *sums_d = nullptr;
  if( sums_sz > 0 ) {
    CHECK_CUDA(cudaMalloc((void**) &sums_d, sums_sz));
  }

  // Allocate the reference on the host (to be computed on the host).
  void *dx_ref_h = nullptr;
  if( mode == Mode::BWD && !skip_checks ) {
    dx_ref_h = malloc(x_sz);
  }

  // Allocate the reference on the host (to be computed on the host).
  float *dgamma_ref_h = nullptr, *dbeta_ref_h = nullptr;
  if( mode == Mode::BWD && !skip_checks ) {
    dgamma_ref_h = (float*) malloc(gamma_sz);
    dbeta_ref_h  = (float*) malloc(gamma_sz);
  }

  // Generate random input data for the forward pass.
  if (use_fp32) {
    random_data<float>(reinterpret_cast<float*>(x_h),     x_elts,     use_1s);
  } else if (use_bf16) {
    random_data<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(x_h),     x_elts,     use_1s);
  } else {
    random_data<__half>(reinterpret_cast<__half*>(x_h),     x_elts,     use_1s);
  }
  random_data<float> (gamma_h, gamma_elts, use_1s);
  random_data<float> (beta_h,  gamma_elts, use_1s);

  // Generate the gradients for the bwd pass.
  if( mode == Mode::BWD ) {
    if (use_fp32) {
      random_data<float>(reinterpret_cast<float*>(dy_h), x_elts, use_1s);
    } else if (use_bf16) {
      random_data<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(dy_h), x_elts, use_1s);
    } else {
      random_data<__half>(reinterpret_cast<__half*>(dy_h), x_elts, use_1s);
    }
  }

  // Precompute the sums (from the fwd pass) for bwd.
  if( mode == Mode::BWD ) {
    // Clear the array of sums (all the elements are set to 0.f).
    memset(sums_h, 0, sums_sz);

    // The number of channels in each group.
    int channels_per_group = c / groups;
    // Iterate over the different groups.
    for( int ni = 0; ni < n; ++ni ) {
      for( int gi = 0; gi < groups; ++gi ) {
        for( int hi = 0; hi < h; ++hi ) {
          for( int wi = 0; wi < w; ++wi ) {
            for( int ii = 0; ii < channels_per_group; ++ii ) {
              // The position of the channel.
              int ci = gi*channels_per_group + ii;
              // The offset to the element.
              int64_t offset = (int64_t) ni*h*w*c + hi*w*c + wi*c + ci;
              // The element in float.
              float x;
              if (use_fp32) {
                x = reinterpret_cast<float*>(x_h)[offset];
              } else if (use_bf16) {
                x = __bfloat162float(reinterpret_cast<__nv_bfloat16*>(x_h)[offset]);
              } else {
                x = __half2float(reinterpret_cast<__half*>(x_h)[offset]);
              }

              // Update the sums (sum of X and sum of squares).
              sums_h[ni*groups + gi].x += x;
              sums_h[ni*groups + gi].y += x * x;
            }
          }
        }
      }
    }

    // The normalization term to compute the means.
    float rcp_hwc_per_group = 1.f / (float) (h * w * channels_per_group);
    // Normalize the sums.
    for( int ngi = 0; ngi < n * groups; ++ngi ) {
      sums_h[ngi].x *= rcp_hwc_per_group;
      sums_h[ngi].y *= rcp_hwc_per_group;
    }
  }

  // Compute the golden reference on the host.
  if (!skip_checks) {
    if( mode == Mode::BWD ) { 
      group_norm_nhwc_bwd_(dx_ref_h, 
                          dgamma_ref_h,
                          dbeta_ref_h,
                          dy_h,
                          x_h, 
                          gamma_h, 
                          beta_h,
                          sums_h,
                          epsilon, 
                          n, 
                          h, 
                          w, 
                          c, 
                          groups,
                          with_swish,
                          use_fp32,
                          use_bf16);
    } else {
      group_norm_nhwc_fwd_(y_ref_h, x_h, gamma_h, beta_h, epsilon, n, h, w, c, groups, with_swish, use_fp32, use_bf16);
    }
  }

  // Copy to the device.
  CHECK_CUDA(cudaMemcpyAsync(x_d, x_h, x_sz, cudaMemcpyHostToDevice, cudaStreamDefault));
  CHECK_CUDA(cudaMemcpyAsync(gamma_d, 
                             gamma_h, 
                             gamma_sz, 
                             cudaMemcpyHostToDevice, 
                             cudaStreamDefault));
  CHECK_CUDA(cudaMemcpyAsync(beta_d, 
                             beta_h, 
                             gamma_sz, 
                             cudaMemcpyHostToDevice, 
                             cudaStreamDefault));

  if( mode == Mode::BWD ) {
    CHECK_CUDA(cudaMemcpyAsync(dy_d, 
                               dy_h, 
                               x_sz, 
                               cudaMemcpyHostToDevice, 
                               cudaStreamDefault));

    // // DEBUG.
    // printf("sums_h[0] = %8.3f, %8.3f\n", sums_h[0].x, sums_h[0].y);
    // // END OF DEBUG.

    CHECK_CUDA(cudaMemcpyAsync(sums_d, 
                               sums_h, 
                               sums_sz, 
                               cudaMemcpyHostToDevice, 
                               cudaStreamDefault));
  }

  // Reset the output buffer with garbage to detect invalid results.
  if( mode == Mode::BWD ) {
    CHECK_CUDA(cudaMemsetAsync(dx_d,     0xdc, x_sz,     cudaStreamDefault));
    CHECK_CUDA(cudaMemsetAsync(dgamma_d, 0xdc, gamma_sz, cudaStreamDefault));
    CHECK_CUDA(cudaMemsetAsync(dbeta_d,  0xdc, gamma_sz, cudaStreamDefault));
  } else {
    CHECK_CUDA(cudaMemsetAsync(y_d,      0xdc, x_sz,     cudaStreamDefault));
  }

  // Declare the parameters.
  Group_norm_nhwc_fwd_params params_fwd;
  memset(&params_fwd, 0, sizeof(params_fwd));
  Group_norm_nhwc_bwd_params params_bwd;
  memset(&params_bwd, 0, sizeof(params_bwd));

  const auto precision = [&]() -> PrecisionMode {
    if (use_fp32) {
      return PrecisionMode::FP32IOFP32W;
    } else if (use_bf16) {
      return PrecisionMode::BF16IOFP32W;
    } else {
      return PrecisionMode::FP16IOFP32W;
    }
  }();

  // Initialize the parameters.
  if( mode == Mode::BWD ) { 
    params_bwd.dx = dx_d;
    params_bwd.dgamma = dgamma_d;
    params_bwd.dbeta = dbeta_d;
    params_bwd.sums = sums_d;
    params_bwd.dy = dy_d;
    params_bwd.x = x_d;
    params_bwd.gamma = gamma_d;
    params_bwd.beta = beta_d;
    params_bwd.epsilon = epsilon;
    params_bwd.n = n;
    params_bwd.h = h;
    params_bwd.w = w;
    params_bwd.c = c;
    params_bwd.groups = groups;
    params_bwd.with_swish = with_swish;
    params_bwd.precision = precision;
  } else {
    params_fwd.y = y_d;
    params_fwd.sums = sums_d;
    params_fwd.x = x_d;
    params_fwd.gamma = gamma_d;
    params_fwd.beta = beta_d;
    params_fwd.epsilon = epsilon;
    params_fwd.n = n;
    params_fwd.h = h;
    params_fwd.w = w;
    params_fwd.c = c;
    params_fwd.groups = groups;
    params_fwd.with_swish = with_swish;
    params_fwd.precision = precision;
  }

  // The number of barriers.
  size_t barriers_elts = 0;
  // The number of elements in the reduction buffer.
  size_t red_buffer_elts = 0; 
  // The number of elements in the reduction buffer that must be zeroed.
  size_t zeroed_red_buffer_elts = 0; 

  // Finalize the parameters.
  dim3 grid;
         if( mode == Mode::BWD && use_one_pass ) {
    group_norm_nhwc_bwd_one_pass_setup(params_bwd, 
                                       barriers_elts, 
                                       red_buffer_elts, 
                                       zeroed_red_buffer_elts,
                                       grid, 
                                       props);
  } else if( mode == Mode::BWD ) {
    group_norm_nhwc_bwd_two_passes_setup(params_bwd, 
                                         zeroed_red_buffer_elts);
  } else if( use_one_pass ) {
    group_norm_nhwc_fwd_one_pass_setup(params_fwd, 
                                       barriers_elts,
                                       red_buffer_elts, 
                                       grid, 
                                       props);
  } else {
    group_norm_nhwc_fwd_two_passes_setup(params_fwd, 
                                         zeroed_red_buffer_elts);
  }

  // The size in bytes for the reduction buffer.
  size_t red_buffer_sz = red_buffer_elts * sizeof(float);
  // Allocate on the device.
  if( red_buffer_sz > 0 ) {
    float **ptr = mode == Mode::BWD ? &params_bwd.red_buffer : &params_fwd.red_buffer;
    CHECK_CUDA(cudaMalloc((void**) ptr, red_buffer_sz));
  }

  // The size of the array of barriers.
  size_t barriers_sz = barriers_elts * sizeof(int);
  // The size in bytes for the reduction buffer that must be zeroed.
  size_t zeroed_red_buffer_sz = barriers_sz + zeroed_red_buffer_elts * sizeof(float);

  // Allocate the buffer if needed.
  void *zeroed_red_buffer_d_ = nullptr;
  if( zeroed_red_buffer_sz > 0 ) {
    CHECK_CUDA(cudaMalloc((void**) &zeroed_red_buffer_d_, zeroed_red_buffer_sz));
  }

  // The buffer of barriers. DO NOT CALL cudaFree on it!!!
  int *barriers_d = reinterpret_cast<int*>(zeroed_red_buffer_d_);
  // The zeroed red buffer. DO NOT CALL cudaFree on it!!!
  float *zeroed_red_buffer_d = reinterpret_cast<float*>(&barriers_d[barriers_elts]);
  // Must be aligned on 4B for floats. It obviously is (unless someone changes the code ;)).
  assert(reinterpret_cast<const int64_t&>(zeroed_red_buffer_d) % sizeof(float) == 0);

  // Set the barriers if needed.
  if( mode == Mode::BWD ) {
    params_bwd.barriers = barriers_d;
    params_bwd.zeroed_red_buffer = zeroed_red_buffer_d;
  } else {
    params_fwd.barriers = barriers_d;
    params_fwd.zeroed_red_buffer = zeroed_red_buffer_d;
  }

  // Create events to time the reference code.
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  // Time the reference code.
  CHECK_CUDA(cudaEventRecord(start));
  for( int ii = 0; ii < runs; ++ii ) {

    // Clear the zeroed buffer if needed.
    if( zeroed_red_buffer_sz > 0 ) {
      CHECK_CUDA(cudaMemsetAsync(zeroed_red_buffer_d_, 
                                 0, 
                                 zeroed_red_buffer_sz, 
                                 cudaStreamDefault));
    }
    if( use_one_pass && mode == Mode::BWD ) {
      group_norm_nhwc_bwd_one_pass_run(params_bwd, grid, cudaStreamDefault);
    } else if( use_one_pass ) {
      group_norm_nhwc_fwd_one_pass_run(params_fwd, grid, cudaStreamDefault);
    } else if( mode == Mode::BWD ) {
      group_norm_nhwc_bwd_two_passes_sum  (params_bwd, cudaStreamDefault);
      group_norm_nhwc_bwd_two_passes_scale(params_bwd, cudaStreamDefault);
    } else {
      group_norm_nhwc_fwd_two_passes_sum  (params_fwd, cudaStreamDefault);
      group_norm_nhwc_fwd_two_passes_scale(params_fwd, cudaStreamDefault);
    }
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Print the runtime.
  float elapsed = 0.f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
  if (!csv_output) {
    printf("elapsed......................: %.3fms\n", elapsed);
    printf("elapsed per run..............: %.3fms\n", elapsed / (float) runs);
    printf("efficiency...................: %.3lf%%\n", dram_sol * runs / elapsed * 100.0);
    printf("\n");
  }

  // Copy the results to the host.
  if( mode == Mode::BWD ) {
    CHECK_CUDA(cudaMemcpyAsync(dx_h, dx_d, x_sz, cudaMemcpyDeviceToHost, cudaStreamDefault));
    CHECK_CUDA(cudaMemcpyAsync(dgamma_h, 
                               dgamma_d, 
                               gamma_sz, 
                               cudaMemcpyDeviceToHost, 
                               cudaStreamDefault));
    CHECK_CUDA(cudaMemcpyAsync(dbeta_h, 
                               dbeta_d, 
                               gamma_sz, 
                               cudaMemcpyDeviceToHost, 
                               cudaStreamDefault));
  } else {
    CHECK_CUDA(cudaMemcpyAsync(y_h, y_d, x_sz, cudaMemcpyDeviceToHost, cudaStreamDefault));
  }

  // Make sure the data has been transferred.
  CHECK_CUDA(cudaStreamSynchronize(cudaStreamDefault));

  // Check the results.
  if (!csv_output) {
    if( mode == Mode::BWD && !skip_checks ) {
      if (use_fp32) {
        check_results<float>("dx", reinterpret_cast<float*>(dx_h), 
                             reinterpret_cast<float*>(dx_ref_h), x_elts, tol);
      } else if (use_bf16) {
        check_results<__nv_bfloat16>("dx", reinterpret_cast<__nv_bfloat16*>(dx_h),
                              reinterpret_cast<__nv_bfloat16*>(dx_ref_h), x_elts, tol);
      } else {
        check_results<__half>("dx", reinterpret_cast<__half*>(dx_h),
                              reinterpret_cast<__half*>(dx_ref_h), x_elts, tol);
      }
      check_results<float> ("dgamma", dgamma_h, dgamma_ref_h, gamma_elts, tol);
      check_results<float> ("dbeta",  dbeta_h,  dbeta_ref_h,  gamma_elts, tol);
    } else if( !skip_checks ) {
      if (use_fp32) {
        check_results<float>("y", reinterpret_cast<float*>(y_h), 
                             reinterpret_cast<float*>(y_ref_h), x_elts, tol);
      } else if (use_bf16) {
        check_results<__nv_bfloat16>("y", reinterpret_cast<__nv_bfloat16*>(y_h),
                              reinterpret_cast<__nv_bfloat16*>(y_ref_h), x_elts, tol);
      } else {
        check_results<__half>("y", reinterpret_cast<__half*>(y_h),
                              reinterpret_cast<__half*>(y_ref_h), x_elts, tol);
      }
    }
  } else {
    printf("%d,%d,%d,%d,%d,%d,%d,%f\n", n, h, w, c, groups, (uint32_t)use_one_pass, (uint32_t)mode, elapsed / (float) runs);
  }

  // Destroy the cuda events.
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  // Release device memory.
  CHECK_CUDA(cudaFree(x_d));
  CHECK_CUDA(cudaFree(y_d));
  CHECK_CUDA(cudaFree(gamma_d));
  CHECK_CUDA(cudaFree(beta_d));
  CHECK_CUDA(cudaFree(dx_d));
  CHECK_CUDA(cudaFree(dy_d));
  CHECK_CUDA(cudaFree(dgamma_d));
  CHECK_CUDA(cudaFree(dbeta_d));
  CHECK_CUDA(cudaFree(sums_d));
  CHECK_CUDA(cudaFree(zeroed_red_buffer_d_));
  CHECK_CUDA(cudaFree(params_bwd.red_buffer));
  CHECK_CUDA(cudaFree(params_fwd.red_buffer));

  // Release host memory.
  free(x_h);
  free(y_h);
  free(gamma_h);
  free(beta_h);
  free(dx_h);
  free(dy_h);
  free(dgamma_h);
  free(dbeta_h);
  free(sums_h);
  free(y_ref_h);
  free(dx_ref_h);
  free(dgamma_ref_h);
  free(dbeta_ref_h);

  // Release the GPU.
  CHECK_CUDA(cudaDeviceReset());
  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

