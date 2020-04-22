#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <torch/torch.h>

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define BIASADDRELU_FPROP_NUM_THREADS 128
#define BIASADDRELU_BPROP_NUM_THREADS 128

// move to a header later on
#define ILP 4
template<typename T>
__host__ __device__ __forceinline__ bool is_aligned(T* p){
  return ((uint64_t)p) % (ILP*sizeof(T)) == 0;
}

template<typename T>
__device__ __forceinline__ void load_store(T* dst, T* src, int dst_offset, int src_offset){
  typedef typename std::aligned_storage<ILP*sizeof(T), ILP*alignof(T)>::type LT;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}
template<typename T>
__device__ __forceinline__ void load_store(T* dst, volatile T* src, int dst_offset, int src_offset){
  typedef typename std::aligned_storage<ILP*sizeof(T), ILP*alignof(T)>::type LT;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}
template<typename T>
__device__ __forceinline__ void load_store(volatile T* dst, T* src, int dst_offset, int src_offset){
  typedef typename std::aligned_storage<ILP*sizeof(T), ILP*alignof(T)>::type LT;
  ((LT*)dst)[dst_offset] = ((LT*)src)[src_offset];
}

// Keep ReLU in float only. When using half, cast to float before calling.
__device__ __inline__ float relu(float a) {
  float retf = max(a, 0.f);
  return (retf);
}

// FP64 Wrapper around cublas GEMMEx
cublasStatus_t mlp_gemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    float* alpha,
    const double* A,
    int lda,
    const double* B,
    int ldb,
    const float* beta,
    double* C,
    int ldc) {
  return cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      CUDA_R_64F,
      lda,
      B,
      CUDA_R_64F,
      ldb,
      beta,
      C,
      CUDA_R_64F,
      ldc,
      CUDA_R_64F,
      CUBLAS_GEMM_DEFAULT);
}

// FP32 Wrapper around cublas GEMMEx
cublasStatus_t mlp_gemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    float* alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    const float* beta,
    float* C,
    int ldc) {
  return cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      CUDA_R_32F,
      lda,
      B,
      CUDA_R_32F,
      ldb,
      beta,
      C,
      CUDA_R_32F,
      ldc,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT);
}

// FP16 Tensor core wrapper around cublas GEMMEx
cublasStatus_t mlp_gemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    float* alpha,
    const at::Half* A,
    int lda,
    const at::Half* B,
    int ldb,
    float* beta,
    at::Half* C,
    int ldc) {
  return cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      CUDA_R_16F,
      lda,
      B,
      CUDA_R_16F,
      ldb,
      beta,
      C,
      CUDA_R_16F,
      ldc,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// Bias ADD + ReLU. Assume input X is [features x batch size], assume column major.
// Bias is one 'features' long vector, with implicit broadcast.
// Currently, activation support fuesed ReLU. Safe to call in-place.
template <typename T>
__global__ void biasAddRelu_fprop(T *X, T *b, uint batch_size, uint features) {
  T r_x[ILP];
  T r_b[ILP];
  if(is_aligned(X) && is_aligned(b) && features % ILP ==0) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (; tid*ILP < features * batch_size; tid += blockDim.x * gridDim.x) {
      int row = tid % (features / ILP);
      load_store(r_x, X, 0 , tid);
      load_store(r_b, b, 0 , row);
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        float bias_sum = static_cast<float>(r_x[ii]) + static_cast<float>(r_b[ii]);
        r_x[ii] = relu(bias_sum);
      }
      load_store(X, r_x, tid , 0);
    }
  } else {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (; tid < features * batch_size; tid += ILP * blockDim.x * gridDim.x) {
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        int idx = tid + ii * blockDim.x * gridDim.x;
        if(idx < features * batch_size) {
          int row = tid % features;
          r_x[ii] = X[idx];
          r_b[ii] = b[row];
        }
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        float bias_sum = static_cast<float>(r_x[ii]) + static_cast<float>(r_b[ii]);
        r_x[ii] = relu(bias_sum);
      }
#pragma unroll
      for(int ii = 0; ii < ILP; ii++) {
        int idx = tid + ii * blockDim.x * gridDim.x;
        if(idx < features * batch_size) {
          X[idx] = r_x[ii];
        }
      }
    }
  }
}

// Compute grid size for pointwise backward kernel.
// Some intelligence needed to determine number of splits for reduction.
void get_biasAddRelu_bprop_grid_size(
    int yfeat,
    int threadsPerBlock,
    int batch_size,
    int* grid_x,
    int* grid_y) {
  // Get number of SMs for efficient reduction.
  int num_SMs = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  // First preference, whole reduction in 1 CTA
  int nBlocks = (yfeat + threadsPerBlock - 1) / threadsPerBlock;

  // Figure out how many splits to divide reduction into. At least 32 elements per CTA.
  // we want grid_y as close to sqrt(batchsize)?
  int nRedSplits = std::sqrt(batch_size);
  // for batchsize <=64, just use 1 block
  if(batch_size < 64) nRedSplits = 1;
  // no need to go over occupancy
  nRedSplits = min((8*num_SMs)/nBlocks, nRedSplits);

  *grid_x = nBlocks;
  *grid_y = nRedSplits;
  return;
}

// Addition done deterministically via a 2-pass approach. Each CTA writes out partial
// sum, and the last CTA in grid Y dimension accumulates partials serially and writes to result.
template <typename T, int UNROLL_FACTOR>
__global__ void biasAddRelu_bprop(
    T* Y,
    T* dY,
    int features,
    int batch_size,
    T* dX,
    volatile float* intermediate,
    int* semaphores,
    T* db) {
  // The feature that this thread is responsible for
  int f = blockIdx.x * blockDim.x + threadIdx.x;

  // Compute the batch span this thread is responsible for
  int chunkSize = (batch_size + gridDim.y - 1) / gridDim.y;
  int nStart = blockIdx.y * chunkSize;
  int nSpan = min(batch_size, nStart + chunkSize) - nStart;
  volatile float* out = intermediate + blockIdx.y * features;

  // Flag to trigger last reduction.
  __shared__ bool isLastBlock;

  // Accumulate db in FP32 always
  float db_local = 0;
  if (f < features) {
    int nidx = 0;
    // Handle non-multiple of UNROLL_FACTOR residue
    for (; nidx < nSpan % UNROLL_FACTOR; nidx++) {
      int row, col, flat_idx;
      row = f;
      col = nStart + nidx;
      flat_idx = col * features + row;
      T y_val = Y[flat_idx];
      T dy_val = dY[flat_idx];
      T dx_val;
      if ((float)y_val > 0.f)
        dx_val = dy_val;
      else
        dx_val = 0;
      dX[flat_idx] = dx_val;
      db_local += (float)dx_val;
    }

    // Handle meat of work
    for (; (nidx + UNROLL_FACTOR - 1) < nSpan; nidx += UNROLL_FACTOR) {
      int row, col, flat_idx;
      row = f;
      col = nStart + nidx;
      flat_idx = col * features + row;
#pragma unroll 4
      for (int u = 0; u < UNROLL_FACTOR; u++) {
        T y_val = Y[flat_idx];
        T dy_val = dY[flat_idx];
        T dx_val;
        if ((float)y_val > 0.f)
          dx_val = dy_val;
        else
          dx_val = 0;
        dX[flat_idx] = dx_val;
        db_local += (float)dx_val;
        flat_idx += features;
      }
    }

    // Write out partial result
    out[f] = db_local;
  }
  __threadfence();
  __syncthreads();

  // Increment semaphore and check if this is the last CTA in
  // the grid_y dimension.
  if (threadIdx.x == 0 && f < features) {
    unsigned int sum_idx;
    sum_idx = atomicAdd(&(semaphores[blockIdx.x]), 1);
    isLastBlock = (sum_idx == (gridDim.y - 1));
  }
  __syncthreads();

  db_local = 0;
  if (isLastBlock && f < features) {
    for (int n = 0; n < gridDim.y; n++) {
      int row, col;
      row = f;
      col = n;
      db_local += (float)(intermediate[col * features + row]);
    }
    db[f] = (T)db_local;
  }
}

// Addition done deterministically via a 2-pass approach. Each CTA writes out partial
// sum, and the last CTA in grid Y dimension accumulates partials serially and writes to result.
template <typename T, int UNROLL_FACTOR>
__global__ void biasAddRelu_bprop_aligned(
    T* Y,
    T* dY,
    int features,
    int batch_size,
    T* dX,
    volatile float* intermediate,
    int* semaphores,
    T* db) {
  // The feature that this thread is responsible for
  int f = blockIdx.x * blockDim.x + threadIdx.x;

  // Compute the batch span this thread is responsible for
  int chunkSize = (batch_size + gridDim.y - 1) / gridDim.y;
  int nStart = blockIdx.y * chunkSize;
  int nSpan = min(batch_size, nStart + chunkSize) - nStart;
  volatile float* out = intermediate + blockIdx.y * features;

  // Flag to trigger last reduction.
  __shared__ bool isLastBlock;

  // Accumulate db in FP32 always
  float db_local[ILP];
  T r_y[ILP];
  T r_dy[ILP];
#pragma unroll
  for(int ii=0;ii<ILP;ii++){
    db_local[ii] = 0.f;
  }

  // f always <= features in this case
  //if (f < features) {
  int nidx = 0;

  // Handle non-multiple of UNROLL_FACTOR residue
  for (; nidx < nSpan % UNROLL_FACTOR; nidx++) {
    int row, col, flat_idx;
    row = f;
    col = nStart + nidx;
    flat_idx = col * features / ILP + row;

    load_store(r_y, Y, 0, flat_idx);
    load_store(r_dy, dY, 0, flat_idx);
#pragma unroll
    for(int ii=0;ii<ILP;ii++){
      if ((float)r_y[ii] <= 0.f)
        r_dy[ii] = 0;
      db_local[ii] += (float)r_dy[ii];
    }
    load_store(dX, r_dy, flat_idx, 0);
  }

  // Handle meat of work
  for (; (nidx + UNROLL_FACTOR - 1) < nSpan; nidx += UNROLL_FACTOR) {
    int row, col, flat_idx;
    row = f;
    col = nStart + nidx;
    flat_idx = col * features / ILP + row; // total threads in x == features/ILP
#pragma unroll
    for (int u = 0; u < UNROLL_FACTOR; u++) {
      load_store(r_y, Y, 0, flat_idx);
      load_store(r_dy, dY, 0, flat_idx);
#pragma unroll
      for(int ii=0;ii<ILP;ii++){
        if ((float)r_y[ii] <= 0.f)
          r_dy[ii] = 0;
        db_local[ii] += (float)r_dy[ii];
      }
      load_store(dX, r_dy, flat_idx, 0);
      flat_idx += features/ILP;
    }
  }

  if(gridDim.y == 1) {
#pragma unroll
    for(int ii=0;ii<ILP;ii++){
      r_dy[ii] = db_local[ii]; // reuse local dy buffer
    }
    load_store(db, r_dy, f, 0);
    return;
  }

  // Write out partial result
  load_store(out, db_local, f, 0);

  __threadfence();
  __syncthreads();

  // Increment semaphore and check if this is the last CTA in
  // the grid_y dimension.
  if (threadIdx.x == 0) {
    unsigned int sum_idx;
    sum_idx = atomicAdd(&(semaphores[blockIdx.x]), 1);
    isLastBlock = (sum_idx == (gridDim.y - 1));
  }
  __syncthreads();

#pragma unroll
  for(int ii=0;ii<ILP;ii++){
    db_local[ii] = 0.f;
  }
  float r_db[ILP];
  if (isLastBlock) {
    for (int n = 0; n < gridDim.y; n++) {
      int row, col;
      row = f;
      col = n;
      load_store(r_db, intermediate, 0, col * features / ILP + row);
#pragma unroll
      for(int ii=0;ii<ILP;ii++){
        db_local[ii] += r_db[ii];
      }
    }
#pragma unroll
    for(int ii=0;ii<ILP;ii++){
      r_dy[ii] = db_local[ii]; // reuse local dy buffer
    }
    load_store(db, r_dy, f, 0);
  }
}

// Lists where the num_layers-1 intermediate Y buffers start in reserved space on fprop, starting
// offset 0. The last Y value is, of course, stored in the user provided output buffer.
void get_y_offsets(
    int batch_size,
    int num_layers,
    const int* output_features,
    int* y_start_offsets) {
  y_start_offsets[0] = 0;
  for (int i = 1; i < num_layers; i++) {
    y_start_offsets[i] = y_start_offsets[i - 1] + batch_size * output_features[i - 1];
  }
}

// Returns the reserved space (in elements) needed for the MLP
size_t get_mlp_reserved_space(int batch_size, int num_layers, const int* output_features) {
  size_t res_space = 0;
  // Need to store output of every intermediate MLP - size equal to output_features[i] * batch_size
  // for all 'i' in [0, num_layers-1)
  for (int l = 0; l < num_layers; l++) {
    res_space += output_features[l] * batch_size;
  }
  return res_space;
}

// Returns the size of all fprop activations combined
size_t get_all_activations_size(int batch_size, int num_layers, const int* output_features) {
  size_t acts_size = 0;
  for (int l = 0; l < num_layers; l++) {
    acts_size += output_features[l] * batch_size;
  }
  return acts_size;
}

#if 0
// Returns the work space (in elements) needed for the MLP bprop.
size_t get_mlp_bp_workspace (int batch_size, int num_layers, const int* output_features) {
    /*
       Workspace is partitioned as
       DY_GEMMs : DX_GEMMs
    */
    size_t work_space = 0;

    // Store each intermediate dY explicitly. Need 2 dYs per MLP layer (one for o/p
    // of biasReLU_bp and one for o/p of dgrad GEMM).
    work_space += 2*get_all_activations_size(batch_size, num_layers, output_features);

    return work_space;
}
#endif

// Scratch space needed for reductions in number of elements
size_t get_reduction_scratch_space(int batch_size, int num_layers, const int* output_features) {
  size_t max_scratch_space = 0;
  // Loop over all layers to see which one needs the max scratch space
  for (int l = 0; l < num_layers; l++) {
    int tmp, num_splits;
    get_biasAddRelu_bprop_grid_size(
        output_features[l], BIASADDRELU_BPROP_NUM_THREADS, batch_size, &tmp, &num_splits);
    max_scratch_space = std::max(max_scratch_space, (size_t)(output_features[l] * num_splits));
  }

  return max_scratch_space;
}

// Buffer for semaphores
size_t get_semaphores_size(int num_layers, const int* output_features) {
  // Upper bound on semaphores is one per feature for the layer
  // with the most features.
  int max_features = 0;
  for (int l = 0; l < num_layers; l++) {
    max_features = std::max(max_features, output_features[l]);
  }
  return (size_t)max_features;
}

// Returns the work space (in elements) needed for the MLP bprop.
template <typename T>
size_t get_mlp_bp_workspace_in_bytes(int batch_size, int num_layers, const int* output_features) {
  size_t work_space = 0;

  // Store each intermediate dY explicitly. Need 2 dYs per MLP layer (one for o/p
  // of biasReLU_bp and one for o/p of dgrad GEMM).
  work_space += 2 * get_all_activations_size(batch_size, num_layers, output_features) * sizeof(T);
  work_space +=
      get_reduction_scratch_space(batch_size, num_layers, output_features) * sizeof(float);
  work_space += get_semaphores_size(num_layers, output_features) * sizeof(int);

  return work_space;
}

// Returns pointers to each segment of the workspace
template <typename T>
void partition_mlp_bp_workspace(
    int batch_size,
    int num_layers,
    const int* output_features,
    void* work_space,
    T** dy_gemms,
    T** dx_gemms,
    float** db_scratch,
    int** semaphores) {
  /*
     Workspace is partitioned as
     DY_GEMMs : DX_GEMMs : DB_SCRATCH : SEMAPHORES
  */
  // Start address where dy_gemm tensors are stored
  *dy_gemms = reinterpret_cast<T*>(work_space);
  // Start address where dx_gemm tensors are stored
  *dx_gemms = *dy_gemms + get_all_activations_size(batch_size, num_layers, output_features);
  // Start address where db intermediate tensors are stored
  *db_scratch = reinterpret_cast<float*>(
      *dx_gemms + get_all_activations_size(batch_size, num_layers, output_features));
  // Start address of semaphores
  *semaphores = reinterpret_cast<int*>(
      *db_scratch + get_reduction_scratch_space(batch_size, num_layers, output_features));

  return;
}

// Does a simple MLP fprop (GEMM+bias+ReLU).
// Can handle num_layers number of layers, each with its own shape. Output of layer i is assumed
// to be input of layer i+1. output_features, WPtr and BPtr are arrays of length num_layers, and
// must be in the same order i.e. WPtr[i] and BPtr[i] are respectively the weight and bias of layer
// 'i'.
template <typename T>
int mlp_fp(
    T* X,
    int input_features,
    int batch_size,
    T** WPtr,
    int num_layers,
    int* output_features,
    T** BPtr,
    T* Y,
    T* reserved_space) {
  T *weight, *input, *output, *bias;
  T *reserved_space_x, *reserved_space_y;
  reserved_space_x = NULL;
  reserved_space_y = reserved_space;

  // Get cublas handle from Pytorch
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  // Get the stream from cublas handle to reuse for biasReLU kernel.
  cudaStream_t stream;
  cublasGetStream(handle, &stream);

  for (int layer = 0; layer < num_layers; layer++) {
    weight = WPtr[layer];
    input = (layer == 0) ? X : reserved_space_x;
    output = (layer == num_layers - 1) ? Y : reserved_space_y;
    bias = BPtr[layer];
    int ifeat = (layer == 0) ? input_features : output_features[layer - 1];
    int ofeat = output_features[layer];

    float one = 1.f;
    float zero = 0.f;

    cublasStatus_t cublas_status;
    // Call GEMM: fprop is Y = W'X
    cublas_status = mlp_gemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        ofeat,
        batch_size,
        ifeat,
        &one,
        weight,
        ifeat,
        input,
        ifeat,
        &zero,
        output,
        ofeat);

    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      printf("GEMM fprop failed with %d\n", cublas_status);
      return 1;
    }

    // Call biasReLU
    const uint &input_size = ofeat;
    int num_blocks = 0;
    int num_SMs = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, biasAddRelu_fprop<T>, BIASADDRELU_FPROP_NUM_THREADS, 0);
    biasAddRelu_fprop<<<num_SMs*num_blocks, BIASADDRELU_FPROP_NUM_THREADS, 0, stream>>>(output, bias, batch_size, input_size);

    // Set current output as next layer input
    reserved_space_x = reserved_space_y;
    // Set next layer output
    reserved_space_y += ofeat * batch_size;
  }

  return 0;
}

// Does a simple MLP bprop (GEMM+bias+ReLU).
// Needs reserved space to come back exactly as it was populated in fprop.
// Does dgrad and wgrad sequentially.
template <typename T>
int mlp_bp(
    T* X,
    T* Y,
    int input_features,
    int batch_size,
    T** WPtr,
    int num_layers,
    int* output_features,
    T* dY,
    T* reserved_space,
    T* work_space,
    T* dX,
    T** dwPtr,
    T** dbPtr) {
  T* weight;
  T *dweight, *dx, *dy, *dbias;
  T *x, *y;

  // Where the dx of the biasReLU (== dy of gemm) is stored. Can be thrown away
  // after bp call.
  T* dy_gemm_base;
  // Where the dx after GEMM is stored.
  T* dx_gemm_base;
  // Where partial reduction results are stored.
  float* db_scratch;
  // Semaphores for reduction.
  int* semaphores;

  partition_mlp_bp_workspace<T>(
      batch_size,
      num_layers,
      output_features,
      work_space,
      &dy_gemm_base,
      &dx_gemm_base,
      &db_scratch,
      &semaphores);

  size_t semaphore_size = get_semaphores_size(num_layers, output_features) * sizeof(int);

  // Get cublas handle from Pytorch
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  // Get the stream from cublas handle to reuse for biasReLU kernel.
  cudaStream_t stream;
  cublasGetStream(handle, &stream);

  int* y_offsets = (int*)malloc(num_layers * sizeof(int));
  get_y_offsets(batch_size, num_layers, output_features, y_offsets);

  for (int layer = num_layers - 1; layer >= 0; layer--) {
    weight = WPtr[layer];
    dweight = dwPtr[layer];

    // x is read from reserved space
    x = (layer == 0) ? X : reserved_space + y_offsets[layer - 1];
    // dx is written in workspace for all but layer==0
    dx = (layer == 0) ? dX : dx_gemm_base + y_offsets[layer - 1];

    // y is read from reserved space
    y = (layer == num_layers - 1) ? Y : reserved_space + y_offsets[layer];
    // dx from layer+1
    dy = (layer == num_layers - 1) ? dY : dx_gemm_base + y_offsets[layer];
    // dy_gemm is written to and read immediately
    T* dy_gemm = dy_gemm_base + y_offsets[layer];

    dbias = dbPtr[layer];
    int xfeat = (layer == 0) ? input_features : output_features[layer - 1];
    int yfeat = output_features[layer];

    float one = 1.f;
    float zero = 0.f;

    // Call bias ReLU backprop - first implementation, 1 thread per bias element
    int threadsPerBlock = BIASADDRELU_BPROP_NUM_THREADS;
    int grid_x, grid_y;
    get_biasAddRelu_bprop_grid_size(yfeat, threadsPerBlock, batch_size, &grid_x, &grid_y);

    dim3 block(threadsPerBlock);

    cudaMemsetAsync(semaphores, 0, semaphore_size, stream);

    if(yfeat % (ILP * threadsPerBlock) == 0 &&
       is_aligned(y) &&
       is_aligned(dy) &&
       is_aligned(dy_gemm) &&
       is_aligned(dbias)){
      dim3 grid(grid_x/ILP, grid_y);
      biasAddRelu_bprop_aligned<T, 4><<<grid, block, 0, stream>>>(
        y, dy, yfeat, batch_size, dy_gemm, db_scratch, semaphores, dbias);
    } else {
      dim3 grid(grid_x, grid_y);
      biasAddRelu_bprop<T, 4><<<grid, block, 0, stream>>>(
        y, dy, yfeat, batch_size, dy_gemm, db_scratch, semaphores, dbias);
    }

    cublasStatus_t cublas_status;
    // Call GEMM dgrad
    cublas_status = mlp_gemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        xfeat,
        batch_size,
        yfeat,
        &one,
        weight,
        xfeat,
        dy_gemm,
        yfeat,
        &zero,
        dx,
        xfeat);

    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      printf("GEMM dgrad failed with %d\n", cublas_status);
      return 1;
    }

    // Call GEMM wgrad
    cublas_status = mlp_gemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        xfeat,
        yfeat,
        batch_size,
        &one,
        x,
        xfeat,
        dy_gemm,
        yfeat,
        &zero,
        dweight,
        xfeat);

    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      printf("GEMM wgrad failed with %d\n", cublas_status);
      return 1;
    }
  }

  return 0;
}

// Instantiate for floating point types
template int mlp_fp<float>(
    float* X,
    int input_features,
    int batch_size,
    float** WPtr,
    int num_layers,
    int* output_features,
    float** BPtr,
    float* Y,
    float* reserved_space);

template int mlp_bp<float>(
    float* X,
    float* Y,
    int input_features,
    int batch_size,
    float** WPtr,
    int num_layers,
    int* output_features,
    float* dY,
    float* reserved_space,
    float* work_space,
    float* dX,
    float** dwPtr,
    float** dbPtr);

template int mlp_fp<at::Half>(
    at::Half* X,
    int input_features,
    int batch_size,
    at::Half** WPtr,
    int num_layers,
    int* output_features,
    at::Half** BPtr,
    at::Half* Y,
    at::Half* reserved_space);

template int mlp_bp<at::Half>(
    at::Half* X,
    at::Half* Y,
    int input_features,
    int batch_size,
    at::Half** WPtr,
    int num_layers,
    int* output_features,
    at::Half* dY,
    at::Half* reserved_space,
    at::Half* work_space,
    at::Half* dX,
    at::Half** dwPtr,
    at::Half** dbPtr);

template int mlp_fp<double>(
    double* X,
    int input_features,
    int batch_size,
    double** WPtr,
    int num_layers,
    int* output_features,
    double** BPtr,
    double* Y,
    double* reserved_space);

template int mlp_bp<double>(
    double* X,
    double* Y,
    int input_features,
    int batch_size,
    double** WPtr,
    int num_layers,
    int* output_features,
    double* dY,
    double* reserved_space,
    double* work_space,
    double* dX,
    double** dwPtr,
    double** dbPtr);

template size_t get_mlp_bp_workspace_in_bytes<float>(
    int batch_size,
    int num_layers,
    const int* output_features);
template size_t get_mlp_bp_workspace_in_bytes<at::Half>(
    int batch_size,
    int num_layers,
    const int* output_features);
template size_t get_mlp_bp_workspace_in_bytes<double>(
    int batch_size,
    int num_layers,
    const int* output_features);
