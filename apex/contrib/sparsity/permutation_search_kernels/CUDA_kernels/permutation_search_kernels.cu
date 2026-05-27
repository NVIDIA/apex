#include <stdio.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor_inl.h>
#include <torch/csrc/stable/tensor_struct.h>

#include <limits>

#define gpuErrchk(ans)                    \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert %d: %s %s %d\n", (int)code, cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

// find the magnitude after enforcing the 2:4 sparsity constraint on a group of 4 values
__device__ float group_2_to_4(float4 vals) {
  vals.x = fabs(vals.x);
  vals.y = fabs(vals.y);
  vals.z = fabs(vals.z);
  vals.w = fabs(vals.w);

  float sum0 = vals.x + vals.y;
  float sum1 = vals.x + vals.z;
  float sum2 = vals.x + vals.w;
  float sum3 = vals.y + vals.z;
  float sum4 = vals.y + vals.w;
  float sum5 = vals.z + vals.w;

  float best_sum0 = fmax(sum0, sum1);
  float best_sum1 = fmax(sum2, sum3);
  float best_sum2 = fmax(sum4, sum5);
  float best_sum = fmax(fmax(best_sum0, best_sum1), best_sum2);

  return best_sum;
}

inline void check_cpu_contiguous(torch::stable::Tensor const& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cpu(), name, " must be a CPU tensor");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline float* float_ptr_from_tensor(torch::stable::Tensor const& tensor, const char* name) {
  check_cpu_contiguous(tensor, name);
  STD_TORCH_CHECK(tensor.scalar_type() == torch::headeronly::ScalarType::Float, name, " must have dtype float32");
  return static_cast<float*>(tensor.mutable_data_ptr());
}

inline unsigned int* uint_ptr_from_tensor(torch::stable::Tensor const& tensor, const char* name) {
  check_cpu_contiguous(tensor, name);
  STD_TORCH_CHECK(tensor.element_size() == sizeof(unsigned int), name, " must have 32-bit elements");
  return reinterpret_cast<unsigned int*>(tensor.mutable_data_ptr());
}

inline unsigned int as_uint_arg(int64_t value, const char* name) {
  STD_TORCH_CHECK(value >= 0 && value <= std::numeric_limits<unsigned int>::max(), name, " is out of range");
  return static_cast<unsigned int>(value);
}

/**********************************************************
 *  Check for the best permutation for an entire matrix
 **********************************************************/
__global__ void permute_and_sum_after_2_to_4(float* matrix, unsigned int rows, unsigned int cols, unsigned int* stripes,
                                             unsigned int total_stripes, unsigned int* permutations, float* output) {
  // vectorize
  float4* mat4 = (float4*)matrix;
  cols /= 4;

  // each thread in a block takes some number of rows
  size_t num_rows = max((int)ceilf((float)rows / (float)blockDim.x), 1);
  size_t row_offset = num_rows * threadIdx.x;
  size_t num_stripes = total_stripes;  // total_stripes / gridDim.x;
  size_t stripe_offset = 0;            // num_stripes * blockIdx.x;
  unsigned int localStart = stripe_offset;
  unsigned int localEnd = localStart + num_stripes;

  // each block takes care of one permutation
  unsigned int p = blockIdx.x;
  unsigned int* permutation = &permutations[p * total_stripes * 4];

  float sum = 0.0f;
  extern __shared__ float s[32][32];
  float4* local_stripes = (float4*)&s[threadIdx.x];
  float* local_columns = (float*)&s[threadIdx.x];
  float4* permuted_local_stripes = (float4*)&local_stripes[num_stripes];
  float* permuted_local_columns = (float*)&local_columns[num_stripes * 4];

  for (unsigned int r = row_offset; r < row_offset + num_rows; ++r) {
    if (r >= rows) break;

    // load into smem
    for (unsigned int s = localStart; s < localEnd; ++s) {
      unsigned int stripe = stripes[s];
      local_stripes[s] = mat4[r * cols + stripe];
    }

// now permute
#pragma unroll 4
    for (unsigned int c = 0; c < num_stripes * 4; ++c) {
      permuted_local_columns[c] = local_columns[permutation[c]];
    }

    // now sum 2:4
    for (unsigned int s = 0; s < num_stripes; ++s) {
      sum += group_2_to_4(permuted_local_stripes[s]);
    }
  }

  atomicAdd(&output[p], sum);
}

void free_permutation_memory(float** dmatrix, unsigned int** dstripe_groups, unsigned int** dpermutations,
                             float** dresults, float** hresults) {
  cudaFree(*dmatrix);
  cudaFree(*dresults);
  cudaFree(*dpermutations);
  cudaFree(*dstripe_groups);
  free(*hresults);
}

int set_up_check_permutation_memory(float** dmatrix, unsigned int rows, unsigned int cols,
                                    unsigned int** dstripe_groups, unsigned int group_width, unsigned int num_groups,
                                    unsigned int** dpermutations, unsigned int num_permutations, float** dresults,
                                    float** hresults) {
  static unsigned int setupRows = 0;
  static unsigned int setupCols = 0;
  static unsigned int setupGroupWidth = 0;
  static unsigned int setupNumGroups = 0;
  static unsigned int setupNumPermutations = 0;
  static bool allocated = false;
  int fresh_alloc = 0;
  if (!allocated || setupRows != rows || setupCols != cols || setupGroupWidth != group_width ||
      setupNumGroups != num_groups || setupNumPermutations != num_permutations) {
    if (allocated) {
      free_permutation_memory(dmatrix, dstripe_groups, dpermutations, dresults, hresults);
    }

    gpuErrchk(cudaMalloc((void**)dmatrix, rows * cols * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)dstripe_groups, group_width * num_groups * sizeof(unsigned int)));
    gpuErrchk(cudaMalloc((void**)dpermutations, num_permutations * group_width * 4 * sizeof(unsigned int)));
    gpuErrchk(cudaMalloc((void**)dresults, num_permutations * sizeof(float)));
    *hresults = (float*)malloc(num_permutations * sizeof(float));

    allocated = true;
    setupRows = rows;
    setupCols = cols;
    setupGroupWidth = group_width;
    setupNumGroups = num_groups;
    setupNumPermutations = num_permutations;
    fresh_alloc = 1;
  }

  return fresh_alloc;
}

///////////////////////////////////////////////////////////

/**********************************************************
 * Get the magnitude of a matrix after applying 2:4
 **********************************************************/
// find the magnitude after enforcing the 2:4 sparsity constraint on a subset of the columns of an input matrix
__global__ void subset_sum_after_2_to_4(float* matrix, unsigned int rows, unsigned int cols, unsigned int start_col,
                                        unsigned int end_col, float* output) {
  // vectorize
  float4* mat4 = (float4*)matrix;
  cols /= 4;
  start_col /= 4;
  end_col /= 4;

  // each thread in a block takes some number of rows
  size_t num_rows = max((int)ceilf((float)rows / (float)blockDim.x), 1);
  size_t row_offset = num_rows * threadIdx.x;
  // each block takes some number of columns
  size_t num_cols = (end_col - start_col) / gridDim.x;
  size_t col_offset = num_cols * blockIdx.x;
  start_col += col_offset;
  end_col = start_col + num_cols;

  float sum = 0.0f;
  for (unsigned int r = row_offset; r < row_offset + num_rows; ++r) {
    if (r < rows) {
      for (unsigned int c = start_col; c < end_col; c++) {
        sum += group_2_to_4(mat4[r * cols + c]);
      }
    }
  }

  atomicAdd(output, sum);
}

// build the entire permute map at once
// each block handles one group of stripes
// each threads in the block handle all handle the same permutation at the same time on different rows before moving to
// the next permutation
__global__ void build_permute_map(float* matrix, unsigned int rows, unsigned int cols, unsigned int* stripes,
                                  unsigned int group_width, unsigned int* permutations, unsigned int num_permutations,
                                  unsigned int perm_length, float* output, unsigned int* best_indices) {
  // vectorize
  float4* mat4 = (float4*)matrix;
  cols /= 4;

  // each block handles a group of stripes
  unsigned int* stripe_group = (unsigned int*)&stripes[blockIdx.x * group_width];

  // shared memory: 32 threads each need 16*2
  extern __shared__ float pm_shared[32][32];
  float4* local_stripes = (float4*)&pm_shared[threadIdx.x];
  float* local_columns = (float*)&pm_shared[threadIdx.x];
  float4* permuted_stripes = (float4*)&local_stripes[4];
  float* permuted_columns = (float*)&local_columns[16];

  // each thread handles all permutations in the row before moving on to the next row
  size_t num_rows = max((int)ceilf((float)rows / (float)blockDim.x), 1);
  size_t row_offset = num_rows * threadIdx.x;

  for (unsigned int r = row_offset; r < row_offset + num_rows; ++r) {
    if (r >= rows) break;

    // load a row into smem
    for (unsigned int s = 0; s < group_width; ++s) {
      unsigned int const stripe = stripe_group[s];
      local_stripes[s] = mat4[r * cols + stripe];
    }

    for (unsigned int p = 0; p < num_permutations; ++p) {
      unsigned int* permutation = &permutations[p * perm_length];
      float sum = 0.0f;

// permute
#pragma unroll 4
      for (unsigned int c = 0; c < group_width * 4; ++c) {
        permuted_columns[c] = local_columns[permutation[c]];
      }

      // sum 2:4
      for (unsigned int s = 0; s < group_width; ++s) {
        sum += group_2_to_4(permuted_stripes[s]);
      }

      // update the running sum for this stripe group's permutation
      atomicAdd(&output[blockIdx.x * num_permutations + p], sum);
    }
  }

  // at this point, each permutation's sum in this stripe group has been calculated
  // now, find the best option
  __syncthreads();

  if (threadIdx.x == 0) {
    unsigned int best_permutation = 0;
    float best_magnitude = output[blockIdx.x * num_permutations];
    float base_magnitude = best_magnitude;

    // #pragma unroll 32
    for (unsigned int p = 1; p < num_permutations; ++p) {
      float magnitude = output[blockIdx.x * num_permutations + p];
      if (magnitude > best_magnitude) {
        best_permutation = p;
        best_magnitude = magnitude;
      }
    }

    output[blockIdx.x * num_permutations] = best_magnitude - base_magnitude;
    best_indices[blockIdx.x] = best_permutation;
  }
}

void free_sum_after_2_to_4_memory(float** dmatrix, float** dresult) {
  cudaFree(*dmatrix);
  cudaFree(*dresult);
}

int set_up_sum_after_2_to_4_memory(float** dmatrix, unsigned int rows, unsigned int cols, float** dresult) {
  static unsigned int setupRows = 0;
  static unsigned int setupCols = 0;
  static bool allocated = false;

  int fresh_allocation = 0;
  if (!allocated || setupRows != rows || setupCols != cols) {
    if (allocated) free_sum_after_2_to_4_memory(dmatrix, dresult);

    gpuErrchk(cudaMalloc((void**)dmatrix, rows * cols * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)dresult, sizeof(float)));

    setupRows = rows;
    setupCols = cols;

    fresh_allocation = 1;
  }

  allocated = true;

  return fresh_allocation;
}

void set_up_permute_map_memory(float** dmatrix, unsigned int rows, unsigned int cols, unsigned int** dstripes,
                               unsigned int num_groups, unsigned int group_width, unsigned int** dpermutations,
                               unsigned int num_permutations, unsigned int perm_length, float** doutput,
                               unsigned int** dindices, float** hresult, unsigned int** hindices) {
  static unsigned int setUpRows = 0;
  static unsigned int setUpCols = 0;
  static unsigned int setUpGroupWidth = 0;
  static unsigned int setUpNumGroups = 0;
  static unsigned int setUpNumPerms = 0;
  static unsigned int setUpPermLength = 0;

  if (setUpRows != rows || setUpCols != cols) {
    if (*dmatrix != NULL) {
      gpuErrchk(cudaFree(*dmatrix));
      *dmatrix = NULL;
    }
    gpuErrchk(cudaMalloc((void**)dmatrix, rows * cols * sizeof(float)));
  }

  if (setUpGroupWidth < group_width || setUpNumGroups < num_groups) {
    if (*dstripes != NULL) {
      gpuErrchk(cudaFree(*dstripes));
      *dstripes = NULL;
    }
    gpuErrchk(cudaMalloc((void**)dstripes, num_groups * group_width * sizeof(unsigned int)));

    if (setUpNumGroups < num_groups) {
      if (*dindices != NULL) {
        gpuErrchk(cudaFree(*dindices));
        *dindices = NULL;
      }
      gpuErrchk(cudaMalloc((void**)dindices, num_groups * sizeof(unsigned int)));
      if (*hindices != NULL) {
        free(*hindices);
        *hindices = NULL;
      }
      *hindices = (unsigned int*)malloc(num_groups * sizeof(unsigned int));
    }
  }

  if (setUpNumPerms < num_permutations || setUpPermLength < perm_length) {
    if (*dpermutations != NULL) {
      gpuErrchk(cudaFree(*dpermutations));
      *dpermutations = NULL;
    }
    gpuErrchk(cudaMalloc((void**)dpermutations, perm_length * num_permutations * sizeof(unsigned int)));
  }

  if (setUpNumPerms < num_permutations || setUpNumGroups < num_groups) {
    if (*doutput != NULL) {
      gpuErrchk(cudaFree(*doutput));
      *doutput = NULL;
    }
    gpuErrchk(cudaMalloc((void**)doutput, num_permutations * num_groups * sizeof(float)));
    if (*hresult != NULL) {
      free(*hresult);
      *hresult = NULL;
    }
    *hresult = (float*)malloc(num_permutations * num_groups * sizeof(float));
  }

  setUpRows = rows;
  setUpCols = cols;
  setUpGroupWidth = group_width;
  setUpNumGroups = num_groups;
  setUpNumPerms = num_permutations;
  setUpPermLength = perm_length;
}

/**********************************************************
 * Build the swap map for channel_swaps
 **********************************************************/
// find the magnitude improvement if some columns were swapped (check all pairs of columns in all the stripe_pairs)
__global__ void swap_columns_sum_after_2_to_4(float* matrix, unsigned int rows, unsigned int cols,
                                              unsigned int* stripe_pairs, float* output) {
  // vectorize
  float4* mat4 = (float4*)matrix;
  cols /= 4;

  // each thread takes some number of rows
  size_t const num_rows = max((int)ceilf((float)rows / (float)blockDim.x), 1);
  size_t const row_offset = num_rows * threadIdx.x;

  // each block is repsonsible for a pair of stripes
  unsigned int const stripe0 = stripe_pairs[2 * blockIdx.x];
  unsigned int const stripe1 = stripe_pairs[2 * blockIdx.x + 1];
  // space for 32 threads, 8 values (2 stripes) in use at a time, plus 16 partial sums and one base sum
  extern __shared__ float cs[32][32];
  float4* local_stripe0 = (float4*)&cs[threadIdx.x][0];
  float* local_cols0 = (float*)&cs[threadIdx.x][0];
  float4* local_stripe1 = (float4*)&cs[threadIdx.x][4];
  float* local_cols1 = (float*)&cs[threadIdx.x][4];
  float* local_psum = (float*)&cs[threadIdx.x][8];
  float* base_psum = (float*)&cs[threadIdx.x][24];

  *base_psum = 0.0f;
  for (unsigned int s = 0; s < 16; ++s) {
    local_psum[s] = 0.0f;
  }

  for (unsigned int r = row_offset; r < row_offset + num_rows; ++r) {
    if (r >= rows) break;
    *local_stripe0 = mat4[r * cols + stripe0];
    *local_stripe1 = mat4[r * cols + stripe1];
    *base_psum += group_2_to_4(*local_stripe0) + group_2_to_4(*local_stripe1);
    unsigned int swap_idx = 0;
    for (unsigned int c0 = 0; c0 < 4; ++c0) {
      for (unsigned int c1 = 0; c1 < 4; ++c1) {
        // swap c0 and c1
        float tmp = local_cols0[c0];
        local_cols0[c0] = local_cols1[c1];
        local_cols1[c1] = tmp;

        // grab the sum
        local_psum[swap_idx] += group_2_to_4(*local_stripe0) + group_2_to_4(*local_stripe1);

        // swap back
        local_cols1[c1] = local_cols0[c0];
        local_cols0[c0] = tmp;

        swap_idx++;
      }
    }
  }

  // reduce partial sums, store local diffs in the output
  __syncthreads();
  if (threadIdx.x == 0) {
    for (unsigned int t = 1; t < blockDim.x; ++t) {
      for (unsigned int swap = 0; swap < 16; ++swap) {
        local_psum[swap] += cs[t][8 + swap];
      }
      *base_psum += cs[t][24];
    }

    for (unsigned int swap = 0; swap < 16; ++swap) {
      atomicAdd(&output[blockIdx.x * 16 + swap], local_psum[swap] - (*base_psum));
    }
  }
}

void set_up_swap_map_memory(float** dmatrix, unsigned int rows, unsigned int cols, unsigned int** dstripe_pairs,
                            unsigned int num_pairs, float** dresult) {
  static unsigned int setupRows = 0;
  static unsigned int setupCols = 0;
  static unsigned int setupPairs = 0;

  if (*dmatrix == NULL || setupRows != rows || setupCols != cols) {
    if (*dmatrix != NULL) {
      gpuErrchk(cudaFree(*dmatrix));
      *dmatrix = NULL;
    }
    gpuErrchk(cudaMalloc((void**)dmatrix, rows * cols * sizeof(float)));
    setupRows = rows;
    setupCols = cols;
  }

  if (*dstripe_pairs == NULL || *dresult == NULL || setupPairs < num_pairs) {
    if (*dstripe_pairs != NULL) {
      gpuErrchk(cudaFree(*dstripe_pairs));
      *dstripe_pairs = NULL;
    }
    if (*dresult != NULL) {
      gpuErrchk(cudaFree(*dresult));
      *dresult = NULL;
    }
    gpuErrchk(cudaMalloc((void**)dstripe_pairs, num_pairs * 2 * sizeof(unsigned int)));
    gpuErrchk(cudaMalloc((void**)dresult, num_pairs * 16 * sizeof(float)));

    setupPairs = num_pairs;
  }
}

///////////////////////////////////////////////////////////

int64_t apex_permutation_search_check_permutations(torch::stable::Tensor const& matrix_tensor, int64_t rows_arg,
                                                   int64_t cols_arg, torch::stable::Tensor const& stripe_groups_tensor,
                                                   int64_t group_width_arg, int64_t num_groups_arg,
                                                   torch::stable::Tensor const& permutations_tensor,
                                                   int64_t num_permutations_arg,
                                                   torch::stable::Tensor const& improvement_tensor,
                                                   torch::stable::Tensor const& permutation_tensor) {
  static float* d_matrix;
  static unsigned int* d_permutations;
  static unsigned int* d_stripes;
  static float* d_results;
  static float* results;

  unsigned int rows = as_uint_arg(rows_arg, "rows");
  unsigned int cols = as_uint_arg(cols_arg, "cols");
  unsigned int group_width = as_uint_arg(group_width_arg, "group_width");
  unsigned int num_groups = as_uint_arg(num_groups_arg, "num_groups");
  unsigned int num_permutations = as_uint_arg(num_permutations_arg, "num_permutations");

  float* matrix = float_ptr_from_tensor(matrix_tensor, "matrix");
  unsigned int* stripe_groups = uint_ptr_from_tensor(stripe_groups_tensor, "stripe_groups");
  unsigned int* permutations = uint_ptr_from_tensor(permutations_tensor, "permutations");
  float* improvement = float_ptr_from_tensor(improvement_tensor, "improvement");
  unsigned int* permutation = uint_ptr_from_tensor(permutation_tensor, "permutation");

  int fresh_alloc = set_up_check_permutation_memory(&d_matrix, rows, cols, &d_stripes, group_width, num_groups,
                                                    &d_permutations, num_permutations, &d_results, &results);
  if (fresh_alloc == 1) {
    gpuErrchk(cudaMemcpy(d_permutations, permutations, num_permutations * group_width * 4 * sizeof(unsigned int),
                         cudaMemcpyHostToDevice));
    gpuErrchk(
        cudaMemcpy(d_stripes, stripe_groups, group_width * num_groups * sizeof(unsigned int), cudaMemcpyHostToDevice));
  }

  gpuErrchk(cudaMemset(d_results, 0, num_permutations * sizeof(float)));
  gpuErrchk(cudaMemcpy(d_matrix, matrix, rows * cols * sizeof(float), cudaMemcpyHostToDevice));

  permute_and_sum_after_2_to_4<<<num_permutations, 32, 32 * group_width * 4 * 2 * sizeof(float)>>>(
      d_matrix, rows, cols, d_stripes, group_width, d_permutations, d_results);
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(results, d_results, num_permutations * sizeof(float), cudaMemcpyDeviceToHost));

  unsigned int best_permutation = 0;
  float best_improvement = 0.0f;
  for (unsigned int p = 1; p < num_permutations; ++p) {
    float cur_improvement = results[p] - results[0];
    if (best_improvement < cur_improvement) {
      best_permutation = p;
      best_improvement = cur_improvement;
    }
  }

  *improvement = best_improvement;
  *permutation = best_permutation;
  return 0;
}

int64_t apex_permutation_search_sum_after_2_to_4(torch::stable::Tensor const& matrix_tensor, int64_t rows_arg,
                                                 int64_t cols_arg, int64_t start_col_arg, int64_t end_col_arg,
                                                 int64_t blocks_arg, int64_t threads_arg,
                                                 torch::stable::Tensor const& output_tensor) {
  static float* d_matrix;
  static float* d_result;

  unsigned int rows = as_uint_arg(rows_arg, "rows");
  unsigned int cols = as_uint_arg(cols_arg, "cols");
  unsigned int start_col = as_uint_arg(start_col_arg, "start_col");
  unsigned int end_col = as_uint_arg(end_col_arg, "end_col");
  unsigned int blocks = as_uint_arg(blocks_arg, "blocks");
  unsigned int threads = as_uint_arg(threads_arg, "threads");

  int fresh_allocation = set_up_sum_after_2_to_4_memory(&d_matrix, rows, cols, &d_result);
  (void)fresh_allocation;

  float* matrix = float_ptr_from_tensor(matrix_tensor, "matrix");
  float* output = float_ptr_from_tensor(output_tensor, "output");

  gpuErrchk(cudaMemcpy(d_matrix, matrix, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemset(d_result, 0, sizeof(float)));

  subset_sum_after_2_to_4<<<blocks, threads>>>(d_matrix, rows, cols, start_col, end_col, d_result);
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(output, d_result, sizeof(float), cudaMemcpyDeviceToHost));
  return 0;
}

int64_t apex_permutation_search_build_permute_map(torch::stable::Tensor const& matrix_tensor, int64_t rows_arg,
                                                  int64_t cols_arg, torch::stable::Tensor const& stripes_tensor,
                                                  int64_t num_groups_arg, int64_t group_width_arg,
                                                  torch::stable::Tensor const& permutations_tensor,
                                                  int64_t perm_length_arg,
                                                  torch::stable::Tensor const& improvements_tensor,
                                                  torch::stable::Tensor const& best_indices_tensor) {
  static float* d_matrix = NULL;
  static unsigned int* d_stripes = NULL;
  static unsigned int* d_permutations = NULL;
  static float* d_output = NULL;
  static unsigned int* d_indices = NULL;
  static float* hresult = NULL;
  static unsigned int* hindices = NULL;

  unsigned int rows = as_uint_arg(rows_arg, "rows");
  unsigned int cols = as_uint_arg(cols_arg, "cols");
  unsigned int num_groups = as_uint_arg(num_groups_arg, "num_groups");
  unsigned int group_width = as_uint_arg(group_width_arg, "group_width");
  unsigned int perm_length = as_uint_arg(perm_length_arg, "perm_length");
  STD_TORCH_CHECK(perm_length > 0, "perm_length must be positive");
  unsigned int num_permutations = as_uint_arg(permutations_tensor.numel() / perm_length, "num_permutations");

  const unsigned int MAX_GROUPS_PER_LAUNCH = num_permutations <= 5775 ? 1820 : 40;
  const unsigned int full_launches = num_groups / MAX_GROUPS_PER_LAUNCH;
  const unsigned int final_launch = num_groups % MAX_GROUPS_PER_LAUNCH;
  const unsigned int launches = full_launches + (final_launch != 0 ? 1 : 0);

  set_up_permute_map_memory(&d_matrix, rows, cols, &d_stripes, min(num_groups, MAX_GROUPS_PER_LAUNCH), group_width,
                            &d_permutations, num_permutations, perm_length, &d_output, &d_indices, &hresult, &hindices);

  float* matrix = float_ptr_from_tensor(matrix_tensor, "matrix");
  unsigned int* stripes = uint_ptr_from_tensor(stripes_tensor, "stripes");
  unsigned int* permutations = uint_ptr_from_tensor(permutations_tensor, "permutations");
  float* improvements = float_ptr_from_tensor(improvements_tensor, "improvements");
  unsigned int* best_indices = uint_ptr_from_tensor(best_indices_tensor, "best_indices");

  gpuErrchk(cudaMemcpy(d_matrix, matrix, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_permutations, permutations, num_permutations * perm_length * sizeof(unsigned int),
                       cudaMemcpyHostToDevice));

  unsigned int group_offset = 0;
  for (unsigned int l = 0; l < launches; ++l) {
    unsigned int groups_this_launch = (l < full_launches) ? MAX_GROUPS_PER_LAUNCH : final_launch;

    gpuErrchk(cudaMemcpy(d_stripes, &stripes[group_offset * group_width],
                         groups_this_launch * group_width * sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_output, 0, groups_this_launch * num_permutations * sizeof(float)));
    gpuErrchk(cudaMemset(d_indices, 0, groups_this_launch * sizeof(unsigned int)));

    unsigned int shmem = 32 * (32) * sizeof(float);
    build_permute_map<<<groups_this_launch, 32, shmem>>>(d_matrix, rows, cols, d_stripes, group_width, d_permutations,
                                                         num_permutations, perm_length, d_output, d_indices);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(
        cudaMemcpy(hresult, d_output, num_permutations * groups_this_launch * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(hindices, d_indices, groups_this_launch * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    for (unsigned int g = 0; g < groups_this_launch; ++g) {
      improvements[group_offset + g] = hresult[g * num_permutations];
      best_indices[group_offset + g] = hindices[g];
    }

    group_offset += groups_this_launch;
  }

  return 0;
}

int64_t apex_permutation_search_build_swap_map(torch::stable::Tensor const& matrix_tensor, int64_t rows_arg,
                                               int64_t cols_arg, torch::stable::Tensor const& stripe_pairs_tensor,
                                               torch::stable::Tensor const& output_tensor) {
  static float* d_matrix = NULL;
  static float* d_result = NULL;
  static unsigned int* d_stripe_pairs = NULL;

  unsigned int rows = as_uint_arg(rows_arg, "rows");
  unsigned int cols = as_uint_arg(cols_arg, "cols");
  unsigned int num_pairs = as_uint_arg(stripe_pairs_tensor.numel() / 2, "num_pairs");

  float* matrix = float_ptr_from_tensor(matrix_tensor, "matrix");
  unsigned int* stripe_pairs = uint_ptr_from_tensor(stripe_pairs_tensor, "stripe_pairs");
  float* output = float_ptr_from_tensor(output_tensor, "output");

  set_up_swap_map_memory(&d_matrix, rows, cols, &d_stripe_pairs, num_pairs, &d_result);
  gpuErrchk(cudaMemcpy(d_matrix, matrix, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_stripe_pairs, stripe_pairs, 2 * num_pairs * sizeof(unsigned int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemset(d_result, 0, num_pairs * 16 * sizeof(float)));

  unsigned int shmem = 32 * (32) * sizeof(float);
  swap_columns_sum_after_2_to_4<<<num_pairs, 32, shmem>>>(d_matrix, rows, cols, d_stripe_pairs, d_result);
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(output, d_result, num_pairs * 16 * sizeof(float), cudaMemcpyDeviceToHost));
  return 0;
}

STABLE_TORCH_LIBRARY_FRAGMENT(apex, m) {
  m.def(
      "permutation_search_sum_after_2_to_4(Tensor matrix, int rows, int cols, int start_col, int end_col, "
      "int blocks, int threads, Tensor(a!) output) -> int");
  m.def(
      "permutation_search_build_permute_map(Tensor matrix, int rows, int cols, Tensor stripes, int num_groups, "
      "int group_width, Tensor permutations, int perm_length, Tensor(a!) improvements, Tensor(b!) best_indices) "
      "-> int");
  m.def(
      "permutation_search_check_permutations(Tensor matrix, int rows, int cols, Tensor stripe_groups, "
      "int group_width, int num_groups, Tensor permutations, int num_permutations, Tensor(a!) improvement, "
      "Tensor(b!) permutation) -> int");
  m.def(
      "permutation_search_build_swap_map(Tensor matrix, int rows, int cols, Tensor stripe_pairs, "
      "Tensor(a!) output) -> int");
}

STABLE_TORCH_LIBRARY_IMPL(apex, CPU, m) {
  m.impl("permutation_search_sum_after_2_to_4", TORCH_BOX(&apex_permutation_search_sum_after_2_to_4));
  m.impl("permutation_search_build_permute_map", TORCH_BOX(&apex_permutation_search_build_permute_map));
  m.impl("permutation_search_check_permutations", TORCH_BOX(&apex_permutation_search_check_permutations));
  m.impl("permutation_search_build_swap_map", TORCH_BOX(&apex_permutation_search_build_swap_map));
}
