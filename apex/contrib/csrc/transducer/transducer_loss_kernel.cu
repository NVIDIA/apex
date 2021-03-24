#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <THC/THC.h>
#include <ATen/cuda/CUDAContext.h>

template<typename scalar_t>
__device__ __forceinline__ scalar_t logSumExp(scalar_t a, scalar_t b) {
    // standard log-sum-exp trick is used here to provide better numerical stability
    return (a >= b) ? a + std::log1p(exp(b-a)) : b + std::log1p(exp(a-b));
}

// Vanilla transducer loss function (i.e. forward-backward algorithm)
// Detail of this loss function can be found in: 
// [1] Sequence Transduction with Recurrent Neural Networks.

// Forward (alpha) and backward (beta) path are launched together. Input is assumed to be converted
// into log scale by the preceding log_softmax layer
// Diagonal wavefront advancing usually used in dynamic programming is leveraged here. 
// alpha and beta are of acc_t type, as they are essentially accumulators.

// This loss function supports packed input where a tensor of shape [B, T, U, H] is packed into 
// [B_packed, H].
// Don't-care region (t > audLen) or (u > txtLen) is removed.
// To support the packed input, the starting offsets for each batch need to be specified with
// batchOffset.
template <typename scalar_t, typename acc_t>
__global__ void transducer_loss_forward(
    const scalar_t* x,
    const int* label,
    const int* audLen,
    const int* txtLen,
    const int64_t* batchOffset,
    int64_t dictSize,   // 64-bit indexing for data tensor
    int64_t blankIdx,
    int64_t maxFLen,
    int64_t maxGLen,
    bool packedInput,
    acc_t* alpha,
    acc_t* beta,
    scalar_t* loss) {

    const int batch = blockIdx.y;
    const int tid = threadIdx.x;
    const auto myFLen = audLen[batch];
    // Note that start of the sentence is added as 1 here
    const auto myGLen = txtLen[batch] + 1;  
    const auto myLabel = label + batch * (maxGLen-1);
    const int64_t myBatchOffset = packedInput ? (batch == 0 ? 0 : batchOffset[batch-1]) 
                                                : batch * maxFLen * maxGLen;
    const int64_t myStrideT = packedInput ? myGLen : maxGLen;
    const scalar_t* myX = x + myBatchOffset * dictSize; 
    int u  = tid;

    if (blockIdx.x == 0){
        // alpha path
        acc_t* myAlpha = alpha + batch*maxFLen*maxGLen;
        if (u == 0) 
            myAlpha[0] = 0;
        __syncthreads();

        for (int64_t step = 1; step < myFLen+myGLen-1; ++step){
            // Move along the diagonal wavefront to leverage available parallelism
            for (u = tid; u < myGLen; u += blockDim.x){
                int64_t t = step - u;
                if (t >= 0 and t < myFLen and u >= 0 and u < myGLen){
                    // Eq(16) in [1]
                    if (u == 0){
                        // alpha(t, u) = alpha(t-1, u) * null(t-1, u)
                        myAlpha[t*maxGLen + u] = myAlpha[(t-1)*maxGLen] 
                                                    + myX[((t-1)*myStrideT) * dictSize + blankIdx];
                    }
                    else if (t == 0){
                        // alpha(t, u-1) = alpha(t, u-1) * y(t, u-1)
                        myAlpha[u] = myAlpha[u - 1] + myX[(u - 1) * dictSize + myLabel[u - 1]];
                    }
                    else{
                        // alpha(t, u) = alpha(t-1, u) * null(t-1, u) + alpha(t, u-1) * y(t, u-1)
                        acc_t current = myAlpha[(t-1)*maxGLen + u] 
                                        + myX[((t-1)*myStrideT + u) * dictSize + blankIdx];
                        acc_t next = myAlpha[t*maxGLen + u - 1] 
                                        + myX[(t*myStrideT + u - 1) * dictSize + myLabel[u - 1]];
                        myAlpha[t*maxGLen + u] = logSumExp(next, current);
                    }
                }
            }
            __syncthreads();
        }
    }
    else if (blockIdx.x == 1){
        // beta path
        acc_t* myBeta = beta + batch*maxFLen*maxGLen;
        if (u == 0){
            myBeta[(myFLen-1)*maxGLen + myGLen - 1] = myX[((myFLen-1)*myStrideT 
                                                        + myGLen - 1) * dictSize + blankIdx];
        }
        __syncthreads();

        for (int64_t step = myFLen+myGLen - 3; step >= 0; --step){
            for (u = tid; u < myGLen; u += blockDim.x){
                int64_t t = step - u;
                if (t >= 0 and t < myFLen and u >=0 and u < myGLen){
                    // Eq(18) in [1]
                    if (u == myGLen - 1){
                        // beta(t, u) = beta(t+1, u) * null(t, u)
                        myBeta[t*maxGLen + u] = myBeta[(t+1)*maxGLen + u] 
                                                + myX[(t*myStrideT + u) * dictSize + blankIdx];
                    }
                    else if (t == myFLen - 1){
                        // beta(t, u) = beta(t, u+1) * y(t, u)
                        myBeta[t*maxGLen + u] = myBeta[t*maxGLen + u + 1] 
                                                + myX[(t*myStrideT + u) * dictSize + myLabel[u]];
                    }
                    else{
                        // beta(t, u) = beta(t+1, u)*null(t, u) + beta(t, u+1)*y(t, u)
                        acc_t current = myBeta[(t+1)*maxGLen + u] 
                                        + myX[(t*myStrideT + u) * dictSize + blankIdx];
                        acc_t next = myBeta[t*maxGLen + u + 1] 
                                        + myX[(t*myStrideT + u) * dictSize + myLabel[u]];
                        myBeta[t*maxGLen + u] = logSumExp(next, current);
                    }
                }
            }
            __syncthreads();
        }
        if (tid == 0)
            loss[batch] = -myBeta[0];   
    }

}

// transudcer loss function (i.e. forward-backward algorithm) with batch loading optimization.
// Compared to the vanilla version, there are two optimizations:
// 1. load x in batch through loop unrolling to reduce the latency.
// 2. Use registers and shared memory to hold alpha and beta values passed from one step the next.
// For simplicity, this kernel currently only supports U <= maxThread, which should be the common
// case. For cases where U > maxThread, the vanilla kernel is used as a fallback option.

// Detail of this loss function can be found in: 
// [1] Sequence Transduction with Recurrent Neural Networks.
// Forward (alpha) and backward (beta) path are launched together. Input is assumed to be converted
// into log scale by the preceding log_softmax layer
// Diagonal wavefront advancing usually used in dynamic programming is leveraged here.
// alpha and beta are of acc_t type, as they are essentially accumulators.

// This loss function supports packed input where a tensor of shape [B, T, U, H] is packed into 
// [B_packed, H].
// Don't-care region (t > audLen) or (u > txtLen) is removed.
// To support the packed input, the starting offsets for each batch need to be specified with
// batchOffset.
template <typename scalar_t, typename acc_t, int batchLdSize>
__global__ void transducer_loss_batch_load_forward(
    const scalar_t* x,
    const int* label,
    const int* audLen,
    const int* txtLen,
    const int64_t* batchOffset,
    int64_t dictSize,
    int64_t blankIdx,
    int64_t maxFLen,
    int64_t maxGLen,
    bool packedInput,
    acc_t* alpha,
    acc_t* beta,
    scalar_t* loss) {

    const int batch = blockIdx.y;
    int u  = threadIdx.x;
    const auto myFLen = audLen[batch];
    const auto myGLen = txtLen[batch] + 1;
    const int64_t myBatchOffset = packedInput ? (batch == 0 ? 0 : batchOffset[batch-1]) 
                                                : batch * maxFLen * maxGLen;
    const int64_t myStrideT = packedInput ? myGLen : maxGLen;
    const scalar_t* myX = x + myBatchOffset * dictSize; 
    scalar_t next[batchLdSize], current[batchLdSize];
    extern __shared__ char smem8[];
    auto smem = reinterpret_cast<acc_t*>(smem8);

    if (blockIdx.x == 0){
        // alpha path
        acc_t* myAlpha = alpha + batch*maxFLen*maxGLen;
        // two SMEM regions for double buffering read and write data to avoid data race
        acc_t * const sharedAlpha[2] = {smem, smem+maxGLen};

        sharedAlpha[0][u] = 0; 
        __syncthreads();

        if (u == 0)
            myAlpha[0] = 0;

        auto myAlphaLabel = (u == 0) ? 0 : label[batch*(maxGLen-1) + u - 1];
        // register used to pass value to the next step for the same thread
        acc_t prvStepAlpha = 0;
        for (int64_t step = 1; step < myFLen+myGLen-1+batchLdSize; step += batchLdSize){
            // Move along the diagonal wavefront to leverage available parallelism
            // Batch loading X through loop unrolling
            #pragma unroll
            for (int i = 0; i < batchLdSize; ++i){
                if (step+i<myFLen+myGLen-1){
                    // index computing
                    int64_t t = step + i - u;
                    int64_t currentId = ((t-1)*myStrideT + u) * dictSize + blankIdx;
                    int64_t nextId = (t*myStrideT + u - 1) * dictSize + myAlphaLabel;
                    // main loading loop
                    if (t >= 0 and t < myFLen and u >= 0 and u < myGLen){
                        if (u == 0){
                            current[i] = myX[currentId];
                        }
                        else if (t == 0){
                            next[i] = myX[nextId];
                        }
                        else{
                            current[i] = myX[currentId];
                            next[i] = myX[nextId];
                        }
                    }
                }
            }
            // main computing loop
            for (int i = 0; i < batchLdSize; ++i){
                // swap the pointer for double buffering
                auto sharedAlphaRd = sharedAlpha[(step+i-1)%2];
                auto sharedAlphaWr = sharedAlpha[(step+i)%2];
                if (step+i<myFLen+myGLen-1){
                    int64_t t = step + i - u;
                    if (t >= 0 and t < myFLen and u >= 0 and u < myGLen){
                        // Eq(16) in [1]
                        if (u == 0)
                            prvStepAlpha = prvStepAlpha+current[i];
                        else if (t == 0)
                            prvStepAlpha = sharedAlphaRd[u-1]+next[i];
                        else
                            prvStepAlpha = logSumExp(prvStepAlpha+current[i], sharedAlphaRd[u-1]
                                            + next[i]);
                        sharedAlphaWr[u] = prvStepAlpha;
                        myAlpha[t*maxGLen + u] = prvStepAlpha;
                    }
                }
                __syncthreads();
            }
        }
    }
    else if (blockIdx.x == 1){
        // beta path
        acc_t* myBeta = beta + batch*maxFLen*maxGLen;
        // two SMEM regions for double buffering read and write data to avoid data race
        acc_t * const sharedBeta[2] = {smem, smem + maxGLen};
        sharedBeta[0][u] = myX[((myFLen-1)*myStrideT + myGLen - 1) * dictSize + blankIdx];
        __syncthreads();

        auto myBetaLabel = (u == maxGLen - 1) ? 0 : label[batch*(maxGLen-1) + u];
        // register used to pass value to the next step for the same thread
        acc_t prvStepBeta = myX[((myFLen-1)*myStrideT + myGLen - 1) * dictSize + blankIdx];
        if (u == 0)
            myBeta[(myFLen-1)*maxGLen + myGLen - 1] = prvStepBeta;

        for (int64_t step = 1; step < myFLen+myGLen-1; step += batchLdSize){
            // Move along the diagonal wavefront to leverage available parallelism
            // Batch loading X
            #pragma unroll
            for (int i = 0; i < batchLdSize; ++i){
                if (step+i<myFLen+myGLen-1){
                    // index computing
                    int64_t t = myFLen+myGLen - (step + i) - 2 - u;
                    int64_t currentId = (t*myStrideT + u) * dictSize + blankIdx;
                    int64_t nextId = (t*myStrideT + u) * dictSize + myBetaLabel;
                    // main loading loop
                    if (t >= 0 and t < myFLen and u >= 0 and u < myGLen){
                        if (u == myGLen - 1){
                            current[i] = myX[currentId];
                        }
                        else if (t == myFLen - 1){
                            next[i] = myX[nextId];
                        }
                        else{
                            current[i] = myX[currentId];
                            next[i] = myX[nextId];
                        }
                    }
                }
            }
            // main computing loop
            for (int i = 0; i < batchLdSize; ++i){
                // swap the pointer for double buffering
                auto sharedBetaRd = sharedBeta[(step+i-1)%2];
                auto sharedBetaWr = sharedBeta[(step+i)%2];
                if (step+i<myFLen+myGLen-1){
                    int64_t t = myFLen+myGLen - (step + i) - 2 - u;
                    if (t >= 0 and t < myFLen and u >= 0 and u < myGLen){
                        // Eq(18) in [1]
                        if (u == myGLen - 1)
                            prvStepBeta = prvStepBeta+current[i];
                        else if (t == myFLen - 1)
                            prvStepBeta = sharedBetaRd[u+1]+next[i];
                        else
                            prvStepBeta = logSumExp(prvStepBeta+current[i], sharedBetaRd[u+1]
                                            + next[i]);
                        sharedBetaWr[u] = prvStepBeta;
                        myBeta[t*maxGLen + u] = prvStepBeta;
                    }
                    
                }
                __syncthreads();
            }
        }
        if (u == 0)
            loss[batch] = -prvStepBeta; 
    }

}

// Vanilla transudcer loss backward operation.
// Detail of this loss function can be found in: 
// [1] Sequence Transduction with Recurrent Neural Networks.
// For this backward kernel, bwd op for the preceding softmax is assumed to be handled elsewhere, 
// hence only Eq(20) in [1] is implemented in this kernel.

// Each thread block works on [batch, t, :, :] of data. Each thread works on a specific u at a time
// Since only gradients for the correct token and null token need to be updated, gradients at other
// locations are initialized to 0.

// To support the packed input, the starting offsets for each batch need to be specified with
// batchOffset.
template <typename scalar_t, typename acc_t>
__global__ void transducer_loss_backward(
    const scalar_t* x,
    const scalar_t* lossGrad,
    const int* audLen,
    const int* txtLen,
    const int* label,
    const acc_t* alpha,
    const acc_t* beta,
    const int64_t* batchOffset,
    int64_t dictSize,
    int64_t blankIdx,
    int64_t maxFLen,
    int64_t maxGLen,
    bool packedInput,
    scalar_t* xGrad) {

    const int tid = threadIdx.x;
    const int t = blockIdx.x;
    const int batch = blockIdx.y;
    const int64_t myFLen = audLen[batch];
    const int64_t myGLen = txtLen[batch] + 1;
    const int64_t myBatchOffset = packedInput ? (batch == 0 ? 0 : batchOffset[batch-1]) 
                                                : batch * maxFLen * maxGLen;
    const int64_t myStrideT = packedInput ? myGLen : maxGLen;
    auto myX = x + (myBatchOffset + t*myStrideT)*dictSize;
    auto myAlpha = alpha + batch*maxFLen*maxGLen;
    auto myBeta = beta + batch*maxFLen*maxGLen;
    auto myXGrad = xGrad + (myBatchOffset + t*myStrideT)*dictSize; 
    auto myLabel = label + batch*(maxGLen-1);

    int64_t u = tid;
    while (t < myFLen and u < myGLen){
        // Do the update
        // loss = -ln(Pr(y*|x))
        acc_t grad = std::log(lossGrad[batch]) + myAlpha[t*maxGLen + u] - myBeta[0];  
        if (u != myGLen - 1)
            myXGrad[u*dictSize + myLabel[u]] = -std::exp(grad + myBeta[t*maxGLen + u + 1] 
                                                + myX[u*dictSize + myLabel[u]]);
        if (t == myFLen - 1 and u == myGLen - 1)
            myXGrad[u*dictSize + blankIdx] = -std::exp(grad + myX[u*dictSize + blankIdx]);
        else if (t != myFLen - 1)
            myXGrad[u*dictSize + blankIdx] = -std::exp(grad + myBeta[(t+1)*maxGLen + u] 
                                                + myX[u*dictSize + blankIdx]); 

        u += blockDim.x;
    }
}

// Fused transudcer loss backward operation.
// Detail of this loss function can be found in: 
// [1] Sequence Transduction with Recurrent Neural Networks.
// The bwd op of the preceding softmax layer is fused in this kernel. 
// Each thread block works on [batch, t, u, :] of data. Each thread works on a specific h at a time

// To support the packed input, the starting offsets for each batch need to be specified with
// batchOffset.
template <typename scalar_t, typename acc_t>
__global__ void transducer_loss_fused_backward(
    const scalar_t* x,
    const scalar_t* lossGrad,
    const int* audLen,
    const int* txtLen,
    const int* label,
    const acc_t* alpha,
    const acc_t* beta,
    const int64_t* batchOffset,
    int64_t dictSize,
    int64_t blankIdx,
    int64_t maxFLen,
    int64_t maxGLen,
    bool packedInput,
    scalar_t* xGrad) {
    
    const int tid = threadIdx.x;
    const int u = blockIdx.x;
    const int t = blockIdx.y;
    const int batch = blockIdx.z;
    const int64_t myFLen = audLen[batch];
    const int64_t myGLen = txtLen[batch] + 1;
    const int64_t myBatchOffset = packedInput ? (batch == 0 ? 0 : batchOffset[batch-1]) 
                                                : batch * maxFLen * maxGLen;
    const int64_t myStrideT = packedInput ? myGLen : maxGLen;

    __shared__ acc_t commonFactor, myBetaTU;
    auto myXGrad = xGrad + (myBatchOffset + t*myStrideT +u)*dictSize; 

    if (t < myFLen and u < myGLen){ 
        auto myX = x + (myBatchOffset + t*myStrideT +u)*dictSize; 
        auto myAlpha = alpha + batch*maxFLen*maxGLen;
        auto myBeta = beta + batch*maxFLen*maxGLen;
        auto myLabel = label + batch*(maxGLen-1);

        // load and store shared variables in SMEM
        if (tid == 0){
            commonFactor = std::log(lossGrad[batch]) + myAlpha[t*maxGLen + u] - myBeta[0];
            myBetaTU = myBeta[t*maxGLen + u];
        }

        __syncthreads();

        for (int64_t h = tid; h < dictSize; h += blockDim.x){
            // Do the update
            acc_t grad = commonFactor + myX[h];  // loss = -ln(Pr(y*|x))
            acc_t myGrad = std::exp(grad + myBetaTU);
            if (u != myGLen - 1 and h == myLabel[u]){
                myGrad -= std::exp(grad + myBeta[t*maxGLen + u + 1]);
            }
            else if (h == blankIdx){
                if (t == myFLen - 1 and u == myGLen - 1)
                    myGrad -= std::exp(grad);
                else if (t != myFLen - 1)
                    myGrad -= std::exp(grad + myBeta[(t+1)*maxGLen + u]);
            }
            myXGrad[h] = myGrad;
        }
    }
    else if (!packedInput){
        // In non-pack mode, need to make sure the gradients for don't-care regions are zero.
        for (int64_t h = tid; h < dictSize; h += blockDim.x){
            myXGrad[h] = 0;
        }
    }
}



std::vector<torch::Tensor> transducer_loss_cuda_forward(
    torch::Tensor x,
    torch::Tensor label,
    torch::Tensor audLen,
    torch::Tensor txtLen,
    torch::Tensor batchOffset,
    int maxFLen,
    int blankIdx,
    int opt,
    bool packedInput){

    auto scalarType = x.scalar_type();
    auto tensorOpt = x.options();
    const int batchSize = label.size(0);
    const int maxGLen = label.size(1) + 1;
    const int dictSize = x.size(-1);

    TORCH_CHECK(blankIdx >= 0 and blankIdx < dictSize, 
                "Expected blank index to be in the range of 0 to ", 
                dictSize-1,
                ", but got ", 
                blankIdx);
    TORCH_CHECK(opt == -1 or opt == 0 or opt == 1, 
                "Got an invalid optimization level ", 
                opt);

    // The data type of alpha and beta will be resolved at dispatch time,
    // hence defined here and assigned later
    torch::Tensor alpha;    
    torch::Tensor beta;
    torch::Tensor loss = torch::empty({batchSize}, tensorOpt);
    const auto deviceProperties = at::cuda::getCurrentDeviceProperties();
    const auto maxThreadPerBlock = deviceProperties->maxThreadsPerBlock;
    const auto maxSmemPerBlock = deviceProperties->sharedMemPerBlock;
    const auto batchOffsetPtr = packedInput ? batchOffset.data_ptr<int64_t>() : nullptr;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(scalarType, "transducer_loss_cuda_forward", ([&] {
        // resolve accumulation type
        using acc_t = at::acc_type<scalar_t, true>;
        auto accType = c10::CppTypeToScalarType<acc_t>::value;
        auto accTensorOpt = tensorOpt.dtype(accType);
        alpha = torch::empty({batchSize, maxFLen, maxGLen}, accTensorOpt);
        beta = torch::empty({batchSize, maxFLen, maxGLen}, accTensorOpt);

        // decide what kernel to launch based on the problem size
        // if the required SMEM size or number threads exceeds the limit, fall back to the vanilla
        // kernel.
        const auto smemSize = 2*maxGLen*sizeof(acc_t);
        const auto optFallBack = (maxGLen > maxThreadPerBlock or smemSize > maxSmemPerBlock) ? 0 
                                    : (opt == -1) ? 1 : opt;
        const int threads = std::min(maxThreadPerBlock, maxGLen);
        const dim3 blocks(2, batchSize, 1);        

        if (optFallBack == 0)
            transducer_loss_forward<<<blocks, threads, 0, stream>>>(
                x.data_ptr<scalar_t>(), 
                label.data_ptr<int>(), 
                audLen.data_ptr<int>(), 
                txtLen.data_ptr<int>(), 
                batchOffsetPtr,
                dictSize, 
                blankIdx, 
                maxFLen,
                maxGLen,
                packedInput,
                alpha.data_ptr<acc_t>(), 
                beta.data_ptr<acc_t>(), 
                loss.data_ptr<scalar_t>());
        else if (optFallBack == 1)
            transducer_loss_batch_load_forward<scalar_t, acc_t, 4>
            <<<blocks, threads, smemSize, stream>>>(
                x.data_ptr<scalar_t>(), 
                label.data_ptr<int>(), 
                audLen.data_ptr<int>(), 
                txtLen.data_ptr<int>(), 
                batchOffsetPtr,
                dictSize, 
                blankIdx, 
                maxFLen,
                maxGLen,
                packedInput,
                alpha.data_ptr<acc_t>(), 
                beta.data_ptr<acc_t>(), 
                loss.data_ptr<scalar_t>());  

    }));
    THCudaCheck(cudaGetLastError());

    return {alpha, beta, loss};
}




torch::Tensor transducer_loss_cuda_backward(
    torch::Tensor x,
    torch::Tensor lossGrad,
    torch::Tensor alpha,
    torch::Tensor beta,
    torch::Tensor audLen,
    torch::Tensor txtLen,
    torch::Tensor label,
    torch::Tensor batchOffset,
    int maxFLen,
    int blankIdx,
    int opt,
    bool fuseSoftmaxBackward,
    bool packedInput){

    auto dtype = x.scalar_type();
    torch::Tensor xGrad;
    const int batchSize = label.size(0);
    const int maxGLen = label.size(1) + 1;
    const int dictSize = x.size(-1);
    const auto deviceProperties = at::cuda::getCurrentDeviceProperties();
    const int maxThreadPerBlock = deviceProperties->maxThreadsPerBlock;
    const int warpSize = deviceProperties->warpSize;
    const auto batchOffsetPtr = packedInput ? batchOffset.data_ptr<int64_t>() : nullptr;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    if (fuseSoftmaxBackward){
        // alloc empty tensors for performance, hence need to ensure zeros are writtern to 
        // don't-care region in the kernel.
        xGrad = torch::empty_like(x);

        // Would like each thread to work on 4 hidden units
        const int workPerThread = 4;  
        // Don't want to have more than 128 threads per thread block
        const int maxThreadPerElmt = std::min(128, maxThreadPerBlock);
        const int threads = std::min(maxThreadPerElmt, std::max(warpSize, 
                                    (dictSize+workPerThread-1)/workPerThread));
        const dim3 blocks(maxGLen, maxFLen, batchSize);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(dtype, "transducer_loss_cuda_backward", ([&] {
            using acc_t = at::acc_type<scalar_t, true>;
            transducer_loss_fused_backward<<<blocks, threads, 0, stream>>>(    
                x.data_ptr<scalar_t>(), 
                lossGrad.data_ptr<scalar_t>(),
                audLen.data_ptr<int>(), 
                txtLen.data_ptr<int>(), 
                label.data_ptr<int>(),
                alpha.data_ptr<acc_t>(), 
                beta.data_ptr<acc_t>(),  
                batchOffsetPtr,
                dictSize, 
                blankIdx, 
                maxFLen,
                maxGLen,
                packedInput,
                xGrad.data_ptr<scalar_t>());   
            
        }));
    }
    else{
        // for non-fused kernel, the gradients need to be writtern are very sparse, hence initialize
        // the tensor with all zeros.
        xGrad = torch::zeros_like(x);
        // don't launch more threads than needed.
        const int threads = std::min(maxThreadPerBlock, maxGLen);
        const dim3 blocks(maxFLen, batchSize);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(dtype, "transducer_loss_cuda_backward", ([&] {
            using acc_t = at::acc_type<scalar_t, true>;
            transducer_loss_backward<<<blocks, threads, 0, stream>>>(    
                x.data_ptr<scalar_t>(), 
                lossGrad.data_ptr<scalar_t>(),
                audLen.data_ptr<int>(), 
                txtLen.data_ptr<int>(), 
                label.data_ptr<int>(),
                alpha.data_ptr<acc_t>(), 
                beta.data_ptr<acc_t>(), 
                batchOffsetPtr, 
                dictSize, 
                blankIdx, 
                maxFLen,
                maxGLen,
                packedInput,
                xGrad.data_ptr<scalar_t>());
        }));
    }
    THCudaCheck(cudaGetLastError());
    
    return xGrad;
}
