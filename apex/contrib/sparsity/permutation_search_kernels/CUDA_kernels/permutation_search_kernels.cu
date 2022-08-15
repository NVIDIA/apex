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
                          //unsigned int num_permutations,
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

    //const unsigned int cols = py_matrix.size() / rows;
    //const unsigned int num_groups = py_stripes.size() / group_width;
    //const unsigned int perm_length = group_width * 4; // 2:4 sparsity - each stripe in the group is 4 elements wide
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sum_after_2_to_4", &run_subset_sum_after_2_to_4, "matrix sum after applying 2:4 (CUDA)");
    m.def("build_permute_map", &run_build_permute_map, "optimize stripe groups (CUDA)");
}