//Currently, this module takes care of sparse 2D x 2D matmul. 
//TODO: handle case when weight tensor is 3D: (batch_idx, r, c)


#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>

/*
    Returns the library handle for the cuSparse API

    Params:
    None

    Return:
    cusparseLtHandle_t handle: Library handle (to be returned)

*/
cusparseLtHandle_t get_lib_handle() {
    cusparseLtHandle_t handle;
    cusparseLtInit(&handle);
    return handle;
}

/*
    Returns all required opaque data structures to carry out sparse matrix multiplication (input * weight)

    Params:
    at::Tensor weight:          The weight matrix
    at::Tensor input:           The input data matrix
    uint64_t weight_nr:         Number of rows in 'weight'
    uint64_t weight_nc:         Number of columns in 'weight'
    int64_t input_nr:           Number of rows in 'input'
    uint64_t input_nc:          Number of columns in 'input'
    void* workspace:            Pointer to location in gpu memory required to do the multiplication (Allocated here, to be returned) 
    void* compressed_weights:   Pointer to location in gpu memory containing the compressed weights (Allocated here, to be returned)

    Return:
    cusparseLtMatmulPlan_t plan: Opaque data structure required for cusparseLtMatmul. Encapsulates matmulDescriptor and algSelection
*/
cusparseLtMatmulPlan_t get_matmul_plan(at::Tensor weight, at::Tensor input, uint64_t weight_nr, uint64_t weight_nc, uint64_t input_nr, uint64_t input_nc, void* workspace, void* compressed_weights) {
    
    cusparseLtHandle_t handle;
    cusparseLtInit(&handle);

    cusparseLtMatDescriptor_t matA, matB, matC;

    //sparse matrix
    cusparseLtStructuredDescriptorInit(&handle, &matB, weight_nr, weight_nc, weight_nr, 16, // rows, cols, ld, alignment TODO: confirm ld and alignment
                                    CUDA_R_16F, CUSPARSE_ORDER_ROW,
                                    CUSPARSE_SPARSITY_50_PERCENT );

    //dense matrix
    cusparseLtDenseDescriptorInit(&handle, &matA, input_nr, input_nc, input_nr, 16,
                                CUDA_R_16F, CUSPARSE_ORDER_ROW);

    //dense matrix
    cusparseLtDenseDescriptorInit(&handle, &matC, input_nr, weight_nc, input_nr, 16,
                                CUDA_R_16F, CUSPARSE_ORDER_ROW);

    cusparseLtMatmulDescriptor_t matmulDescriptor;
    cusparseLtMatmulDescriptorInit(&handle,
                                &matmulDescriptor,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &matA, &matB, &matC, &matC, // D == C, outputs
                                CUSPARSE_COMPUTE_16F);

    cusparseLtMatmulAlgSelection_t algSelection;
    cusparseLtMatmulAlgSelectionInit(&handle, &algSelection,
                                    CUSPARSE_MATMUL_ALG_DEFAULT); 

    size_t workspaceSize;
    cusparseLtMatmulGetWorkspace(&handle, &matmulDescriptor, &algSelection,
                                &workspaceSize);

    cudaMalloc(&workspace, workspaceSize); //to return

    cusparseLtMatmulPlan_t plan;
    cusparseLtMatmulPlanInit(&handle, &plan, &matmulDescriptor, &algSelection,
                            workspaceSize);

    cudaStream_t stream = NULL;

    //TODO: confirm CUSPARSE_SPMMA_PRUNE_TILE vs CUSPARSE_SPMMA_PRUNE_STRIP
    cusparseLtSpMMAPrune(&handle, &matmulDescriptor, CUSPARSE_SPMMA_PRUNE_TILE,
                        weight, weight, stream); // prune in-place

    size_t compressedSize;
    cusparseLtSpMMACompressedSize(&handle, &plan, &compressedSize);

    cudaMalloc(&compressed_weights, compressedSize); //to return
    cusparseLtSpMMACompress(&handle, &plan, weight, compressed_weights, stream);

    return plan;
}

/*
    Carries out the sparse matrix-matrix multiplication (input * weight)

    Params:
    cusparseLtHandle_t handle:      Cusparse library handle
    at::Tensor input:               The input data matrix
    void* workspace:                Pointer to location in gpu memory required to do the multiplication
    void* compressed_weights:       Pointer to location in gpu memory containing the compressed weights
    cusparseLtMatmulPlan_t plan:    Opaque data structure required for cusparseLtMatmul. Encapsulates matmulDescriptor and algSelection 

    Return:
    None
*/

void sparse_matmul(cusparseLtHandle_t handle, cusparseLtMatmulPlan_t plan, void* compressed_weights, void* workspace, at::Tensor input, at::Tensor output) {
    cudaStream_t stream = NULL;
    const int32_t num_streams = 1;
    cusparseLtMatmul(&handle, &plan, 1.0, input, compressed_weights,
                    0.0, output, output, workspace, stream, num_streams);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_matmul_plan", &get_matmul_plan, "Return the cusparse 'plan' for matrix multiplication and return the workspace");
  m.def("get_lib_handle", &get_lib_handle, "Get cusparse library handle");
  m.def("sparse_matmul", &sparse_matmul, "Carry out the sparse_matrix multiplication");
}

