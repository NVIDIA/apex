#include <stdio.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert %d: %s %s %d\n", (int)code, cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


// find the magnitude after enforcing the 2:4 sparsity constraint on a group of 4 values
__device__ float group_2_to_4(float4 vals)
{
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

inline float* float_ptr_from_numpy(py::array_t<float>& py_float)
{
    return (float*)py_float.data();
}

inline unsigned int* uint_ptr_from_numpy(py::array_t<unsigned int>& py_uint)
{
    return (unsigned int*)py_uint.data();
}

/**********************************************************
*  Check for the best permutation for an entire matrix
**********************************************************/
__global__ void permute_and_sum_after_2_to_4(float* matrix,
                                             unsigned int rows,
                                             unsigned int cols,
                                             unsigned int* stripes,
                                             unsigned int total_stripes,
                                             unsigned int* permutations,
                                             float* output)
{
    // vectorize
    float4* mat4 = (float4*) matrix;
    cols /= 4;

    // each thread in a block takes some number of rows
    size_t num_rows = max((int)ceilf((float)rows / (float)blockDim.x), 1);
    size_t row_offset = num_rows * threadIdx.x;
    size_t num_stripes = total_stripes; // total_stripes / gridDim.x;
    size_t stripe_offset = 0; //num_stripes * blockIdx.x;
    unsigned int localStart = stripe_offset;
    unsigned int localEnd = localStart + num_stripes;

    // each block takes care of one permutation
    unsigned int p = blockIdx.x;
    unsigned int* permutation = &permutations[p*total_stripes*4];

    float sum = 0.0f;
    extern __shared__ float s[32][32];
    float4* local_stripes = (float4*)&s[threadIdx.x];
    float*  local_columns = (float*)&s[threadIdx.x];
    float4* permuted_local_stripes = (float4*)&local_stripes[num_stripes];
    float*  permuted_local_columns = (float*)&local_columns[num_stripes*4];

    for ( unsigned int r = row_offset; r < row_offset + num_rows; ++r) {
        if (r >= rows)
            break;

        // load into smem
        for ( unsigned int s = localStart; s < localEnd; ++s) {
            unsigned int stripe = stripes[s];
            local_stripes[s] = mat4[r*cols+stripe];
        }

        // now permute
        #pragma unroll 4
        for ( unsigned int c = 0; c < num_stripes*4; ++c) {
            permuted_local_columns[c] = local_columns[permutation[c]];
        }

        // now sum 2:4
        for ( unsigned int s = 0; s < num_stripes; ++s) {
            sum += group_2_to_4(permuted_local_stripes[s]);
        }
    }

    atomicAdd(&output[p], sum);
}

void free_permutation_memory(float** dmatrix,
                             unsigned int** dstripe_groups,
                             unsigned int** dpermutations,
                             float** dresults,
                             float** hresults)
{
    cudaFree(*dmatrix);
    cudaFree(*dresults);
    cudaFree(*dpermutations);
    cudaFree(*dstripe_groups);
    free(*hresults);
}

int set_up_check_permutation_memory(float** dmatrix,
                                    unsigned int rows,
                                    unsigned int cols,
                                    unsigned int** dstripe_groups,
                                    unsigned int group_width,
                                    unsigned int num_groups,
                                    unsigned int** dpermutations,
                                    unsigned int num_permutations,
                                    float** dresults,
                                    float** hresults)
{
    static unsigned int setupRows = 0;
    static unsigned int setupCols = 0;
    static unsigned int setupGroupWidth = 0;
    static unsigned int setupNumGroups = 0;
    static unsigned int setupNumPermutations = 0;
    static bool allocated = false;
    int fresh_alloc = 0;
    if (!allocated ||
        setupRows != rows ||
        setupCols != cols ||
        setupGroupWidth != group_width ||
        setupNumGroups != num_groups ||
        setupNumPermutations != num_permutations) {

        if (allocated) {
            free_permutation_memory(dmatrix, dstripe_groups, dpermutations, dresults, hresults);
        }

        gpuErrchk(cudaMalloc( (void**) dmatrix, rows*cols*sizeof(float)));
        gpuErrchk(cudaMalloc( (void**) dstripe_groups, group_width*num_groups*sizeof(unsigned int)));
        gpuErrchk(cudaMalloc( (void**) dpermutations, num_permutations*group_width*4*sizeof(unsigned int)));
        gpuErrchk(cudaMalloc( (void**) dresults, num_permutations*sizeof(float)));
        *hresults = (float*) malloc(num_permutations*sizeof(float));

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

int run_check_permutations(py::array_t<float>& py_matrix,
                           unsigned int rows,
                           unsigned int cols,
                           py::array_t<unsigned int>& py_stripe_groups, // groups of stripes, group_width = stripes per group, num_groups = groups in the array
                           unsigned int group_width,
                           unsigned int num_groups,
                           py::array_t<unsigned int>& py_permutations,  // array of permutations to try, group_width*4 values per each of num_permutations permutations
                           unsigned int num_permutations,
                           py::array_t<float>& py_improvement,          // improvment offered by the best permutation
                           py::array_t<unsigned int>& py_permutation   // the best permutation
                          )
{
    const unsigned int threads = 32;
    static float* d_matrix;
    static unsigned int* d_permutations;
    static unsigned int* d_stripes;
    static float* d_results;
    static float* results;

    float* matrix = float_ptr_from_numpy(py_matrix);
    unsigned int* stripe_groups = uint_ptr_from_numpy(py_stripe_groups);
    unsigned int* permutations = uint_ptr_from_numpy(py_permutations);
    float* improvement = float_ptr_from_numpy(py_improvement);
    unsigned int* permutation = uint_ptr_from_numpy(py_permutation);

    int fresh_alloc = set_up_check_permutation_memory(&d_matrix, rows, cols, &d_stripes, group_width, num_groups, &d_permutations, num_permutations, &d_results, &results);
    if (fresh_alloc == 1){
        gpuErrchk(cudaMemcpy( d_permutations, permutations, num_permutations*group_width*4*sizeof(unsigned int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy( d_stripes, stripe_groups, group_width*num_groups*sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

    // initialize results, new matrix
    gpuErrchk(cudaMemset (d_results, 0, num_permutations*sizeof(float)));
    gpuErrchk(cudaMemcpy (d_matrix, matrix, rows*cols*sizeof(float), cudaMemcpyHostToDevice ));

    // get results for all permutations
    permute_and_sum_after_2_to_4<<<num_permutations, threads, threads*group_width*4*2*sizeof(float)>>>(d_matrix, rows, cols, d_stripes, group_width, d_permutations, d_results);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy( results, d_results, num_permutations*sizeof(float), cudaMemcpyDeviceToHost ));

    // find the best permutation - could reduce on GPU
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

///////////////////////////////////////////////////////////

/**********************************************************
* Get the magnitude of a matrix after applying 2:4
**********************************************************/
// find the magnitude after enforcing the 2:4 sparsity constraint on a subset of the columns of an input matrix
__global__ void subset_sum_after_2_to_4(float* matrix,
                                        unsigned int rows,
                                        unsigned int cols,
                                        unsigned int start_col,
                                        unsigned int end_col,
                                        float* output)
{
    // vectorize
    float4* mat4 = (float4*) matrix;
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
    for ( unsigned int r = row_offset; r < row_offset + num_rows; ++r ) {
        if (r < rows) {
            for ( unsigned int c = start_col; c < end_col; c++ ) {
                sum += group_2_to_4(mat4[r * cols + c]);
            }
        }
    }

    atomicAdd(output, sum);
}


// build the entire permute map at once
// each block handles one group of stripes
// each threads in the block handle all handle the same permutation at the same time on different rows before moving to the next permutation
__global__ void build_permute_map(float* matrix,
                                  unsigned int rows,
                                  unsigned int cols,
                                  unsigned int* stripes,
                                  unsigned int group_width,
                                  unsigned int* permutations,
                                  unsigned int num_permutations,
                                  unsigned int perm_length,
                                  float* output,
                                  unsigned int* best_indices)
{
    // vectorize
    float4* mat4 = (float4*) matrix;
    cols /= 4;

    // each block handles a group of stripes
    unsigned int* stripe_group = (unsigned int*)&stripes[blockIdx.x*group_width];

    // shared memory: 32 threads each need 16*2
    extern __shared__ float pm_shared[32][32];
    float4* local_stripes = (float4*)&pm_shared[threadIdx.x];
    float*  local_columns = (float*) &pm_shared[threadIdx.x];
    float4* permuted_stripes = (float4*) &local_stripes[4];
    float*  permuted_columns = (float*) &local_columns[16];

    // each thread handles all permutations in the row before moving on to the next row
    size_t num_rows = max((int)ceilf((float)rows / (float)blockDim.x), 1);
    size_t row_offset = num_rows * threadIdx.x;

    for ( unsigned int r = row_offset; r < row_offset + num_rows; ++r) {
        if (r >= rows)
            break;

        // load a row into smem
        for ( unsigned int s = 0; s < group_width; ++s) {
            unsigned int const stripe = stripe_group[s];
            local_stripes[s] = mat4[r*cols+stripe];
        }

        for ( unsigned int p = 0; p < num_permutations; ++p) {
            unsigned int* permutation = &permutations[p*perm_length];
            float sum = 0.0f;

            // permute
            #pragma unroll 4
            for ( unsigned int c = 0; c < group_width*4; ++c) {
                permuted_columns[c] = local_columns[permutation[c]];
            }

            // sum 2:4
            for ( unsigned int s = 0; s < group_width; ++s) {
                sum += group_2_to_4(permuted_stripes[s]);
            }

            // update the running sum for this stripe group's permutation
            atomicAdd(&output[blockIdx.x*num_permutations + p], sum);
        }
    }

    // at this point, each permutation's sum in this stripe group has been calculated
    // now, find the best option
    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned int best_permutation = 0;
        float best_magnitude = output[blockIdx.x*num_permutations];
        float base_magnitude = best_magnitude;

        //#pragma unroll 32
        for (unsigned int p = 1; p < num_permutations; ++p) {
            float magnitude = output[blockIdx.x*num_permutations+p];
            if (magnitude > best_magnitude) {
                best_permutation = p;
                best_magnitude = magnitude;
            }
        }

        output[blockIdx.x*num_permutations] = best_magnitude - base_magnitude;
        best_indices[blockIdx.x] = best_permutation;
    }
}


void free_sum_after_2_to_4_memory(float** dmatrix,
                                  float** dresult)
{
    cudaFree(*dmatrix);
    cudaFree(*dresult);
}

int set_up_sum_after_2_to_4_memory(float** dmatrix,
                                   unsigned int rows,
                                   unsigned int cols,
                                   float** dresult)
{
    static unsigned int setupRows = 0;
    static unsigned int setupCols = 0;
    static bool allocated = false;

    int fresh_allocation = 0;
    if (!allocated ||
        setupRows != rows ||
        setupCols != cols)
    {
        if (allocated)
            free_sum_after_2_to_4_memory(dmatrix, dresult);

        gpuErrchk(cudaMalloc( (void**) dmatrix, rows*cols*sizeof(float)));
        gpuErrchk(cudaMalloc( (void**) dresult, sizeof(float)));

        setupRows = rows;
        setupCols = cols;

        fresh_allocation = 1;
    }

    allocated = true;

    return fresh_allocation;
}

int run_subset_sum_after_2_to_4(py::array_t<float>& py_matrix,
                                unsigned int rows,
                                unsigned int cols,
                                unsigned int start_col,
                                unsigned int end_col,
                                unsigned int blocks,
                                unsigned int threads,
                                py::array_t<float>& py_output)
{

    static float* d_matrix;
    static float* d_result;

    int fresh_allocation = set_up_sum_after_2_to_4_memory(&d_matrix, rows, cols, &d_result);

    float* matrix = float_ptr_from_numpy(py_matrix);
    float* output = float_ptr_from_numpy(py_output);

    gpuErrchk(cudaMemcpy( d_matrix, matrix, rows*cols*sizeof(float), cudaMemcpyHostToDevice ));
    gpuErrchk(cudaMemset( d_result, 0, sizeof(float)));

    subset_sum_after_2_to_4<<<blocks, threads>>>(d_matrix, rows, cols, start_col, end_col, d_result);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy( output, d_result, sizeof(float), cudaMemcpyDeviceToHost ));

    return 0;
}

void set_up_permute_map_memory(float** dmatrix,
                               unsigned int rows,
                               unsigned int cols,
                               unsigned int** dstripes,
                               unsigned int num_groups,
                               unsigned int group_width,
                               unsigned int** dpermutations,
                               unsigned int num_permutations,
                               unsigned int perm_length,
                               float** doutput,
                               unsigned int** dindices,
                               float** hresult,
                               unsigned int** hindices)
{
    static unsigned int setUpRows = 0;
    static unsigned int setUpCols = 0;
    static unsigned int setUpGroupWidth = 0;
    static unsigned int setUpNumGroups = 0;
    static unsigned int setUpNumPerms = 0;
    static unsigned int setUpPermLength = 0;

    if (setUpRows != rows ||
        setUpCols != cols) {
        if (*dmatrix != NULL) { gpuErrchk(cudaFree(*dmatrix)); *dmatrix = NULL; }
        gpuErrchk(cudaMalloc( (void**) dmatrix, rows*cols*sizeof(float)));
    }

    if (setUpGroupWidth < group_width ||
        setUpNumGroups < num_groups) {
        if (*dstripes != NULL) { gpuErrchk(cudaFree(*dstripes)); *dstripes = NULL; }
        gpuErrchk(cudaMalloc( (void**) dstripes, num_groups*group_width*sizeof(unsigned int)));

        if (setUpNumGroups < num_groups) {
            if (*dindices != NULL) { gpuErrchk(cudaFree(*dindices)); *dindices = NULL; }
            gpuErrchk(cudaMalloc( (void**) dindices, num_groups*sizeof(unsigned int)));
            if (*hindices != NULL) { free(*hindices); *hindices = NULL; }
            *hindices = (unsigned int*) malloc (num_groups*sizeof(unsigned int));
        }
    }

    if (setUpNumPerms < num_permutations ||
        setUpPermLength < perm_length) {
        if (*dpermutations != NULL) { gpuErrchk(cudaFree(*dpermutations)); *dpermutations = NULL; }
        gpuErrchk(cudaMalloc( (void**) dpermutations, perm_length*num_permutations*sizeof(unsigned int)));
    }

    if (setUpNumPerms < num_permutations ||
        setUpNumGroups < num_groups) {
        if (*doutput != NULL) { gpuErrchk(cudaFree(*doutput)); *doutput = NULL; }
        gpuErrchk(cudaMalloc( (void**) doutput, num_permutations*num_groups*sizeof(float)));
        if (*hresult != NULL) { free(*hresult); *hresult = NULL; }
        *hresult = (float*) malloc(num_permutations*num_groups*sizeof(float));
    }

    setUpRows = rows;
    setUpCols = cols;
    setUpGroupWidth = group_width;
    setUpNumGroups = num_groups;
    setUpNumPerms = num_permutations;
    setUpPermLength = perm_length;
}

int run_build_permute_map(py::array_t<float>& py_matrix,
                          unsigned int rows,
                          unsigned int cols,
                          py::array_t<unsigned int>& py_stripes,
                          unsigned int num_groups,
                          unsigned int group_width,
                          py::array_t<unsigned int>& py_permutations,
                          unsigned int perm_length,
                          py::array_t<float>& py_improvements,
                          py::array_t<unsigned int>& py_best_indices)
{
    static float* d_matrix = NULL;
    static unsigned int* d_stripes = NULL;
    static unsigned int* d_permutations = NULL;
    static float* d_output = NULL;
    static unsigned int* d_indices = NULL;
    static float* hresult = NULL;
    static unsigned int* hindices = NULL;

    const unsigned int num_permutations = py_permutations.size() / perm_length;

    const unsigned int MAX_GROUPS_PER_LAUNCH = num_permutations <= 5775 ? 1820 : 40;
    const unsigned int full_launches = num_groups / MAX_GROUPS_PER_LAUNCH;
    const unsigned int final_launch = num_groups % MAX_GROUPS_PER_LAUNCH;
    const unsigned int launches = full_launches + (final_launch != 0 ? 1 : 0);

    set_up_permute_map_memory(&d_matrix, rows, cols, &d_stripes, min(num_groups,MAX_GROUPS_PER_LAUNCH), group_width, &d_permutations, num_permutations, perm_length, &d_output, &d_indices, &hresult, &hindices);

    float* matrix = float_ptr_from_numpy(py_matrix);
    unsigned int* stripes = uint_ptr_from_numpy(py_stripes);
    unsigned int* permutations = uint_ptr_from_numpy(py_permutations);
    float* improvements = float_ptr_from_numpy(py_improvements);
    unsigned int* best_indices = uint_ptr_from_numpy(py_best_indices);

    gpuErrchk(cudaMemcpy( d_matrix, matrix, rows*cols*sizeof(float), cudaMemcpyHostToDevice ));
    gpuErrchk(cudaMemcpy( d_permutations, permutations, num_permutations*perm_length*sizeof(unsigned int), cudaMemcpyHostToDevice ));

    unsigned int group_offset = 0;
    for (unsigned int l = 0; l < launches; ++l) 
    {
        unsigned int groups_this_launch = (l < full_launches) ? MAX_GROUPS_PER_LAUNCH : final_launch;

        gpuErrchk(cudaMemcpy( d_stripes, &stripes[group_offset*group_width], groups_this_launch*group_width*sizeof(unsigned int), cudaMemcpyHostToDevice ));
        gpuErrchk(cudaMemset( d_output, 0, groups_this_launch*num_permutations*sizeof(float)));
        gpuErrchk(cudaMemset( d_indices, 0, groups_this_launch*sizeof(unsigned int)));

        unsigned int shmem = 32*(32)*sizeof(float);
        build_permute_map<<<groups_this_launch, 32, shmem>>>(d_matrix, rows, cols, d_stripes, group_width, d_permutations, num_permutations, perm_length, d_output, d_indices);
        gpuErrchk(cudaDeviceSynchronize());

        gpuErrchk(cudaMemcpy( hresult, d_output, num_permutations*groups_this_launch*sizeof(float), cudaMemcpyDeviceToHost ));
        gpuErrchk(cudaMemcpy( hindices, d_indices, groups_this_launch*sizeof(unsigned int), cudaMemcpyDeviceToHost ));

        // thread0 stuck the minimum in the first slot of each group
        for (unsigned int g = 0; g < groups_this_launch; ++g) {
            improvements[group_offset+g] = hresult[g*num_permutations];
            best_indices[group_offset+g] = hindices[g];
        }

        group_offset += groups_this_launch;
    }

    return 0;

}

/**********************************************************
* Build the swap map for channel_swaps
**********************************************************/
// find the magnitude improvement if some columns were swapped (check all pairs of columns in all the stripe_pairs)
__global__ void swap_columns_sum_after_2_to_4(float* matrix,
                                              unsigned int rows,
                                              unsigned int cols,
                                              unsigned int* stripe_pairs,
                                              float* output)
{
    // vectorize
    float4* mat4 = (float4*) matrix;
    cols /= 4;

    // each thread takes some number of rows
    size_t const num_rows = max((int)ceilf((float)rows / (float)blockDim.x), 1);
    size_t const row_offset = num_rows * threadIdx.x;

    // each block is repsonsible for a pair of stripes
    unsigned int const stripe0 = stripe_pairs[2*blockIdx.x];
    unsigned int const stripe1 = stripe_pairs[2*blockIdx.x+1];
    // space for 32 threads, 8 values (2 stripes) in use at a time, plus 16 partial sums and one base sum
    extern __shared__ float cs[32][32];
    float4* local_stripe0 = (float4*)&cs[threadIdx.x][0];
    float*  local_cols0   = (float*) &cs[threadIdx.x][0];
    float4* local_stripe1 = (float4*)&cs[threadIdx.x][4];
    float*  local_cols1   = (float*) &cs[threadIdx.x][4];
    float*  local_psum    = (float*) &cs[threadIdx.x][8];
    float*  base_psum     = (float*) &cs[threadIdx.x][24];

    *base_psum = 0.0f;
    for ( unsigned int s = 0; s < 16; ++s) {
        local_psum[s] = 0.0f;
    }

    for ( unsigned int r = row_offset; r < row_offset + num_rows; ++r) {
        if (r >= rows)
            break;
        *local_stripe0 = mat4[r*cols+stripe0];
        *local_stripe1 = mat4[r*cols+stripe1];
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
        for ( unsigned int t = 1; t < blockDim.x; ++t) {
            for (unsigned int swap = 0; swap < 16; ++swap) {
                local_psum[swap] += cs[t][8+swap];
            }
            *base_psum += cs[t][24];
        }

        for ( unsigned int swap = 0; swap < 16; ++swap) {
            atomicAdd(&output[blockIdx.x*16 + swap], local_psum[swap] - (*base_psum));
        }
    }
}

void set_up_swap_map_memory(float** dmatrix,
                            unsigned int rows,
                            unsigned int cols,
                            unsigned int** dstripe_pairs,
                            unsigned int num_pairs,
                            float** dresult)
{
    static unsigned int setupRows = 0;
    static unsigned int setupCols = 0;
    static unsigned int setupPairs = 0;

    if (*dmatrix == NULL ||
        setupRows != rows ||
        setupCols != cols)
    {
        if (*dmatrix != NULL) { gpuErrchk(cudaFree(*dmatrix)); *dmatrix = NULL; }
        gpuErrchk(cudaMalloc( (void**) dmatrix, rows*cols*sizeof(float)));
        setupRows = rows;
        setupCols = cols;
    }

    if (*dstripe_pairs == NULL ||
        *dresult == NULL ||
        setupPairs < num_pairs)
    {
        if (*dstripe_pairs != NULL) { gpuErrchk(cudaFree(*dstripe_pairs)); *dstripe_pairs = NULL; }
        if (*dresult != NULL) { gpuErrchk(cudaFree(*dresult)); *dresult = NULL; }
        gpuErrchk(cudaMalloc( (void**) dstripe_pairs, num_pairs*2*sizeof(unsigned int)));
        gpuErrchk(cudaMalloc( (void**) dresult, num_pairs*16*sizeof(float)));


        setupPairs = num_pairs;
    }
}


int run_build_swap_map(py::array_t<float>& py_matrix,
                       unsigned int rows,
                       unsigned int cols,
                       py::array_t<uint32_t>& py_stripe_pairs,
                       py::array_t<float>& py_output)
{
    static float* d_matrix = NULL;
    static float* d_result = NULL;
    static unsigned int* d_stripe_pairs = NULL;

    float* matrix = float_ptr_from_numpy(py_matrix);//(float*)py_matrix.data();
    unsigned int* stripe_pairs = uint_ptr_from_numpy(py_stripe_pairs);//(unsigned int*)py_stripe_pairs.data();
    float* output = float_ptr_from_numpy(py_output);//(float*)py_output.data();

    unsigned int num_pairs = py_stripe_pairs.size()/2;

    set_up_swap_map_memory(&d_matrix, rows, cols, &d_stripe_pairs, num_pairs, &d_result);
    gpuErrchk(cudaMemcpy( d_matrix, matrix, rows*cols*sizeof(float), cudaMemcpyHostToDevice ));
    gpuErrchk(cudaMemcpy( d_stripe_pairs, stripe_pairs, 2*num_pairs*sizeof(unsigned int), cudaMemcpyHostToDevice ));
    gpuErrchk(cudaMemset( d_result, 0, num_pairs*16*sizeof(float)));

    unsigned int shmem = 32*(32)*sizeof(float);
    swap_columns_sum_after_2_to_4<<<num_pairs, 32, shmem>>>(d_matrix, rows, cols, d_stripe_pairs, d_result);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy( output, d_result, num_pairs*16*sizeof(float), cudaMemcpyDeviceToHost ));

    return 0;
}
///////////////////////////////////////////////////////////



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sum_after_2_to_4", &run_subset_sum_after_2_to_4, "matrix sum after applying 2:4 (CUDA)");
    m.def("build_permute_map", &run_build_permute_map, "optimize stripe groups (CUDA)");
    m.def("check_permutations", &run_check_permutations, "exhaustively check all permutations (CUDA)");
    m.def("build_swap_map", &run_build_swap_map, "channel swaps (CUDA)");
}
