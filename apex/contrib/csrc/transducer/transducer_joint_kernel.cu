#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/macros/Macros.h>
#include <THC/THC.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>


template <typename scalar_t>
__device__ __forceinline__ scalar_t warpReduce(scalar_t x, int width=C10_WARP_SIZE){
    for (unsigned offset = width/2; offset > 0; offset /= 2){
        x += __shfl_down_sync(0xffffffff, x, offset, width);   
    }
    return x;
}

inline int largestPowerOfTwo(int x){
    int y = 1;
    while (y <= x)
        y <<= 1;
    return y >> 1;
}

struct OffsetCalFwd{
    __device__ __forceinline__ OffsetCalFwd(
        int64_t batch, 
        const int64_t *batchOffset, 
        int64_t maxFLen, 
        int64_t maxGLen, 
        int64_t gLen,
        int64_t hiddenSize,
        bool packOutput) :
        batch(batch),
        batchOffset(batchOffset),
        maxFLen(maxFLen),
        maxGLen(maxGLen),
        gLen(gLen),
        hiddenSize(hiddenSize),
        packOutput(packOutput)
        {}
    
    int64_t batch;
    const int64_t *batchOffset;
    int64_t maxFLen;
    int64_t maxGLen;
    int64_t gLen;
    int64_t hiddenSize;
    bool packOutput;

    __device__ __forceinline__ int64_t getBatchOffset(){
        return packOutput ? ((batch==0) ? 0 : batchOffset[batch-1])*hiddenSize 
                            : batch*maxFLen*maxGLen*hiddenSize;
    }

    __device__ __forceinline__ int64_t getStrideF(){
        return packOutput ? gLen*hiddenSize : maxGLen*hiddenSize;
    }

    
};

struct OffsetCalBwd{
    __device__ __forceinline__ OffsetCalBwd(
        int64_t batch, 
        const int64_t *batchOffset, 
        const int *fLen, 
        const int *gLen,
        int64_t maxFLen, 
        int64_t maxGLen, 
        int64_t hiddenSize,
        bool packOutput,
        bool bwdFasterDim) :
        batch(batch),
        batchOffset(batchOffset),
        maxFLen(maxFLen),
        maxGLen(maxGLen),
        fLen(fLen),
        gLen(gLen),
        hiddenSize(hiddenSize),
        packOutput(packOutput),
        bwdFasterDim(bwdFasterDim)
        {}

    int64_t batch;
    const int64_t *batchOffset;
    const int *fLen;
    const int *gLen;
    int64_t maxFLen;
    int64_t maxGLen;
    int64_t hiddenSize;
    bool packOutput;
    bool bwdFasterDim;

    __device__ __forceinline__ int64_t getBatchOffset(){
        return packOutput ? ((batch==0) ? 0 : batchOffset[batch-1])*hiddenSize 
                            : batch*maxFLen*maxGLen*hiddenSize;
    }

    __device__ __forceinline__ int64_t getMaxXLen(){
        return bwdFasterDim ? maxGLen : maxFLen;
    }

    __device__ __forceinline__ auto getMyXLen() -> decltype(gLen[batch]){
        return bwdFasterDim ? gLen[batch] : fLen[batch];
    }

    __device__ __forceinline__ auto getMyYLen() -> decltype(gLen[batch]){
        return bwdFasterDim ? fLen[batch] : gLen[batch];
    }
    
    __device__ __forceinline__ int64_t getStrideX(){
        return bwdFasterDim ? hiddenSize : ((packOutput ? gLen[batch] : maxGLen) * hiddenSize);
    }

    __device__ __forceinline__ int64_t getStrideY(){
        return bwdFasterDim ? ((packOutput ? gLen[batch] : maxGLen) * hiddenSize) : hiddenSize;
    }
};



template <typename scalar_t, class OffsetCal>
__global__ void transducer_joint_forward(
    const scalar_t *f,
    const scalar_t *g,
    const int *fLen,
    const int *gLen,
    const int64_t *batchOffset,
    int64_t maxFLen,
    int64_t maxGLen,
    int64_t hiddenSize,
    bool packOutput,
    scalar_t *sum) {


    const int batch = blockIdx.z;
    const int t = blockIdx.y;
    const int u = blockIdx.x;
    const auto myFLen = fLen[batch];
    const auto myGLen = gLen[batch];

    OffsetCal offsetCal(batch, batchOffset, maxFLen, maxGLen, myGLen, hiddenSize, packOutput);
    const auto myBatchOffset = offsetCal.getBatchOffset();
    const auto strideF = offsetCal.getStrideF();
    scalar_t const *myF = f + batch*maxFLen*hiddenSize + t*hiddenSize;
    scalar_t const *myG = g + batch*maxGLen*hiddenSize + u*hiddenSize;
    scalar_t *mySum = sum + myBatchOffset + t*strideF + u * hiddenSize;

    if (t < myFLen and u < myGLen){
        #pragma unroll
        for (int h = threadIdx.x; h < hiddenSize; h += blockDim.x){
            if (h < hiddenSize){
                mySum[h] = myF[h] + myG[h];
            }
        }
    }
    else if (packOutput == false and t < maxFLen and u < maxGLen){
        #pragma unroll
        for (int h = threadIdx.x; h < hiddenSize; h += blockDim.x){
            if (h < hiddenSize){
                mySum[h] = -1;
            }
        }    
    }
}

template <typename scalar_t, int tileF, int tileG, class OffsetCal>
__global__ void transducer_joint_tiled_forward(
    const scalar_t *f,
    const scalar_t *g,
    const int *fLen,
    const int *gLen,
    const int64_t *batchOffset,
    int64_t maxFLen,
    int64_t maxGLen,
    int64_t hiddenSize,
    int64_t hiddenPerBlock,
    bool packOutput,
    scalar_t *sum) {


    const int batch = blockIdx.z;
    const int t = blockIdx.y * tileF;
    const int hiddenBlock = (hiddenSize + hiddenPerBlock - 1) / hiddenPerBlock;
    const int u = blockIdx.x / hiddenBlock * tileG;
    const int hOffset = (blockIdx.x % hiddenBlock) * hiddenPerBlock;
    const int h = threadIdx.x;
    const auto myFLen = fLen[batch];
    const auto myGLen = gLen[batch];

    OffsetCal offsetCal(batch, batchOffset, maxFLen, maxGLen, myGLen, hiddenSize, packOutput);
    const auto myBatchOffset = offsetCal.getBatchOffset();
    const auto strideF = offsetCal.getStrideF();

    scalar_t const *myF = f + batch*maxFLen*hiddenSize + t*hiddenSize + hOffset;
    scalar_t const *myG = g + batch*maxGLen*hiddenSize + u*hiddenSize + hOffset;
    scalar_t *mySum = sum + myBatchOffset + t*strideF + u*hiddenSize + hOffset;

    if (t < myFLen and u < myGLen and hOffset+h < hiddenSize){    
        scalar_t fBuffer[tileF], gBuffer[tileG];
        for (int i = 0; i < tileF; ++i){
            if (t + i < myFLen)
                fBuffer[i] = myF[i*hiddenSize + h];
        }
        for (int j = 0; j < tileG; ++j){
            if (u + j < myGLen)
                gBuffer[j] = myG[j*hiddenSize + h];
        }
        #pragma unroll
        for (int i = 0; i < tileF; ++i){
            if (t + i < myFLen){
                #pragma unroll
                for (int j = 0; j < tileG; ++j){
                    if (u + j < myGLen)
                        mySum[i*strideF + j*hiddenSize + h] = fBuffer[i] + gBuffer[j];
                    else if (packOutput == false and u + j < maxGLen)
                        mySum[i*strideF + j*hiddenSize + h] = -1;
                }
            }
            else if (packOutput == false and t + i < maxFLen){
                #pragma unroll
                for (int j = 0; j < tileG; ++j){
                    if (u + j < maxGLen)
                        mySum[i*strideF + j*hiddenSize + h] = -1;
                }
            }
        }
    }
    else if (packOutput == false and t < maxFLen and u < maxGLen and hOffset+h < hiddenSize){
        #pragma unroll
        for (int i = 0; i < tileF; ++i){
            if (t + i < maxFLen){
                #pragma unroll
                for (int j = 0; j < tileG; ++j){
                    if (u + j < maxGLen)
                        mySum[i*strideF + j*hiddenSize + h] = -1;
                }
            }
        }
    }
}


template <typename scalar_t, typename acc_t, class OffsetCal>
__device__ void transducer_joint_single_backward(
    const scalar_t *grad,
    const int *fLen,
    const int *gLen,
    const int64_t *batchOffset,
    int64_t maxFLen,
    int64_t maxGLen,
    int64_t hiddenSize,
    bool packOutput,
    bool bwdFasterDim,
    scalar_t *inGrad,
    int yBlockOffset=0) {


    const int batch = blockIdx.z;
    const int x = blockIdx.y-yBlockOffset;
    const int hOffset = blockIdx.x*C10_WARP_SIZE;
    const int wid = threadIdx.y;
    const int lid = threadIdx.x;
    const int numWarp = blockDim.y;
    extern __shared__ char smem8[];
    auto smem = reinterpret_cast<acc_t*>(smem8);

    OffsetCal offsetCal(batch, batchOffset, fLen, gLen, maxFLen, maxGLen, hiddenSize, packOutput, 
                        bwdFasterDim);
    const auto maxXLen = offsetCal.getMaxXLen();
    const auto myXLen = offsetCal.getMyXLen();
    const auto myYLen = offsetCal.getMyYLen();
    scalar_t *myInGrad = inGrad + batch*maxXLen*hiddenSize + x*hiddenSize + hOffset;
    
    if (x < myXLen){
        
        const auto myBatchOffset = offsetCal.getBatchOffset();
        const auto strideX = offsetCal.getStrideX();
        const auto strideY = offsetCal.getStrideY();
        scalar_t const *myGrad = grad + myBatchOffset + x*strideX + hOffset;
        
        acc_t warpSum = 0;
        auto numYPerWarp = (myYLen+numWarp-1)/numWarp;
        for (int warpY = 0; warpY < numYPerWarp; ++warpY){
            auto y = wid*numYPerWarp + warpY;
            if (y < myYLen and (hOffset+lid) < hiddenSize)
                warpSum += myGrad[y*strideY + lid];
        }
        smem[lid*numWarp + wid] = warpSum;
        __syncthreads();
        auto sum = smem[wid*C10_WARP_SIZE + lid];
        sum = warpReduce(sum, numWarp);

        if (hOffset+wid*C10_WARP_SIZE/numWarp+lid/numWarp < hiddenSize){
            if (lid % numWarp == 0){
                myInGrad[wid*C10_WARP_SIZE/numWarp + lid/numWarp] = sum;
            }
        }
    }
    else if (wid == 0 and hOffset + lid < hiddenSize){
        myInGrad[lid] = 0;
    }
}

template <typename scalar_t, typename acc_t, class OffsetCal>
__global__ void transducer_joint_combined_backward(
    const scalar_t *grad,
    const int *fLen,
    const int *gLen,
    const int64_t *batchOffset,
    int64_t maxFLen,
    int64_t maxGLen,
    int64_t hiddenSize,
    bool packOutput,
    scalar_t *fGrad,
    scalar_t *gGrad) {
    if (blockIdx.y < maxFLen){
        transducer_joint_single_backward<scalar_t, acc_t, OffsetCal>(
            grad,
            fLen,
            gLen,
            batchOffset,
            maxFLen,
            maxGLen,
            hiddenSize,
            packOutput,
            false,
            fGrad);
    }
    else{
        transducer_joint_single_backward<scalar_t, acc_t, OffsetCal>(
            grad,
            fLen,
            gLen,
            batchOffset,
            maxFLen,
            maxGLen,
            hiddenSize,
            packOutput,
            true,
            gGrad,
            maxFLen);
    }  
}

template <typename scalar_t, typename acc_t, typename vec_t, int V, class OffsetCal>
__device__ void transducer_joint_single_vec_backward(
    const scalar_t *grad,
    const int *fLen,
    const int *gLen,
    const int64_t *batchOffset,
    int64_t maxFLen,
    int64_t maxGLen,
    int64_t hiddenSize,
    bool packOutput,
    bool bwdFasterDim,
    scalar_t *inGrad,
    int yBlockOffset=0){

    const int batch = blockIdx.z;
    const int x = blockIdx.y - yBlockOffset;
    const int hOffset = blockIdx.x*C10_WARP_SIZE*V;
    const int wid = threadIdx.y;
    const int lid = threadIdx.x;
    const int numWarp = blockDim.y;

    OffsetCal offsetCal(batch, batchOffset, fLen, gLen, maxFLen, maxGLen, hiddenSize, packOutput, 
                        bwdFasterDim);
    const auto maxXLen = offsetCal.getMaxXLen();
    const auto myXLen = offsetCal.getMyXLen();
    const auto myYLen = offsetCal.getMyYLen();
    scalar_t *myInGrad = inGrad + batch*maxXLen*hiddenSize + x*hiddenSize + hOffset;
    extern __shared__ char smem8[];
    auto smem = reinterpret_cast<acc_t*>(smem8);

    acc_t warpSum[V];
    scalar_t inBuffer[V];
    scalar_t outBuffer[V];
    auto myInGradVec = reinterpret_cast<vec_t*>(myInGrad);
    auto outBufferVec = reinterpret_cast<vec_t*>(outBuffer);

    if (x < myXLen){
        const auto myBatchOffset = offsetCal.getBatchOffset();
        const auto strideX = offsetCal.getStrideX();
        const auto strideY = offsetCal.getStrideY();
        const scalar_t *myGrad = grad + myBatchOffset + x*strideX + hOffset;

        for (int i = 0; i < V; ++i)
            warpSum[i] = 0;

        auto numYPerWarp = (myYLen+numWarp-1)/numWarp;

        for (int warpY = 0; warpY < numYPerWarp; ++warpY){
            auto y = wid*numYPerWarp + warpY;
            auto myGradVec = reinterpret_cast<vec_t const *>(myGrad + y*strideY);
            auto inBufferVec = reinterpret_cast<vec_t*>(inBuffer);
            if (hOffset + lid*V < hiddenSize and y < myYLen){
                *inBufferVec = myGradVec[lid];
                #pragma unroll
                for (int i = 0; i < V; ++i){
                    warpSum[i] += inBuffer[i];
                }
            }
        }
        
        
        for (int i = 0; i < V; ++i){
            smem[lid*numWarp + wid] = warpSum[i];
            __syncthreads();
            auto sum = smem[wid*C10_WARP_SIZE + lid];

            if (hOffset+(wid*C10_WARP_SIZE/numWarp)*V < hiddenSize){
                sum = warpReduce(sum, numWarp);
                if (lid % numWarp == 0){
                    outBuffer[i] = sum;
                }
            }
            __syncthreads();
        }
        if (lid % numWarp == 0 and hOffset+(wid*C10_WARP_SIZE/numWarp + lid/numWarp)*V < hiddenSize)
            myInGradVec[wid*C10_WARP_SIZE/numWarp + lid/numWarp] = *outBufferVec;     
    }
    else if (wid == 0 and hOffset + lid*V < hiddenSize){
        myInGradVec[lid] = 0;
    }
}


template <typename scalar_t, typename acc_t, typename vec_t, int V, class OffsetCal>
__global__ void transducer_joint_combined_vec_backward(
    const scalar_t *grad,
    const int *fLen,
    const int *gLen,
    const int64_t *batchOffset,
    int64_t maxFLen,
    int64_t maxGLen,
    int64_t hiddenSize,
    bool packOutput,
    scalar_t *fGrad,
    scalar_t *gGrad) {
    if (blockIdx.y < maxFLen){
        transducer_joint_single_vec_backward<scalar_t, acc_t, vec_t, V, OffsetCal>(
            grad,
            fLen,
            gLen,
            batchOffset,
            maxFLen,
            maxGLen,
            hiddenSize,
            packOutput,
            false,
            fGrad);
    }
    else{
        transducer_joint_single_vec_backward<scalar_t, acc_t, vec_t, V, OffsetCal>(
            grad,
            fLen,
            gLen,
            batchOffset,
            maxFLen,
            maxGLen,
            hiddenSize,
            packOutput,
            true,
            gGrad,
            maxFLen);
    }  
}




torch::Tensor transducer_joint_cuda_forward(
    torch::Tensor f,
    torch::Tensor g,
    torch::Tensor fLen,
    torch::Tensor gLen,
    torch::Tensor batchOffset,
    int64_t packedBatch,
    int opt,
    bool packOutput,
    int tileSize){

    
    auto tensorOpt = f.options();
    auto dtype = f.scalar_type();
    const auto batchSize = f.size(0);
    const auto maxFLen = f.size(1);
    const auto maxGLen = g.size(1);
    const auto hiddenSize = f.size(2);
    
    int64_t *batchOffsetPtr = nullptr;
    torch::Tensor sum;
    if (!packOutput){
        sum = torch::empty({batchSize, maxFLen, maxGLen, hiddenSize}, tensorOpt);
        batchOffsetPtr = nullptr;
    }
    else{
        sum = torch::empty({packedBatch, hiddenSize}, tensorOpt);    
        batchOffsetPtr = batchOffset.data_ptr<int64_t>();
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    TORCH_CHECK(opt == 0 or opt == 1, "Got an invalid optimization level ", opt);
    // Simple heuristics
    const int numThread = std::min(128, (static_cast<int>(hiddenSize)+C10_WARP_SIZE-1)
                                        / C10_WARP_SIZE * C10_WARP_SIZE);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dtype, "transducer_joint_forward", ([&] {
        if (opt == 0){
            const int threads = numThread;
            const dim3 blocks(maxGLen, maxFLen, batchSize);

            transducer_joint_forward<scalar_t, OffsetCalFwd>
            <<<blocks, threads, 0, stream>>>(
                f.data_ptr<scalar_t>(), 
                g.data_ptr<scalar_t>(), 
                fLen.data_ptr<int>(), 
                gLen.data_ptr<int>(), 
                batchOffsetPtr,
                maxFLen,
                maxGLen,
                hiddenSize,
                packOutput,
                sum.data_ptr<scalar_t>());
        }
        if (opt == 1){

            const int threads = numThread;
            const int hiddenPerBlock = numThread;
            const int hiddenBlock = (hiddenSize + hiddenPerBlock - 1) / hiddenPerBlock;
            const dim3 blocks((maxGLen+tileSize-1)/tileSize * hiddenBlock, 
                                (maxFLen+tileSize-1)/tileSize, batchSize);

            TORCH_CHECK(tileSize == 1 or tileSize == 2 or tileSize == 4, 
                "Expected tileSize to be in [1, 2, 4], but got ", tileSize);
            switch (tileSize) {
                #define LAUNCH_TRANSDUCER_JOINT_TILED_FORWARD(tile) case tile:\
                    transducer_joint_tiled_forward<scalar_t, tile, tile, OffsetCalFwd>\
                    <<<blocks, threads, 0, stream>>>(\
                        f.data_ptr<scalar_t>(),\
                        g.data_ptr<scalar_t>(),\
                        fLen.data_ptr<int>(),\
                        gLen.data_ptr<int>(),\
                        batchOffsetPtr,\
                        maxFLen,\
                        maxGLen,\
                        hiddenSize,\
                        hiddenPerBlock,\
                        packOutput,\
                        sum.data_ptr<scalar_t>());\
                    break;
                LAUNCH_TRANSDUCER_JOINT_TILED_FORWARD(1);
                LAUNCH_TRANSDUCER_JOINT_TILED_FORWARD(2);
                LAUNCH_TRANSDUCER_JOINT_TILED_FORWARD(4);
            }

        }
    }));   
    THCudaCheck(cudaGetLastError());
    return sum;
}

std::vector<torch::Tensor> transducer_joint_cuda_backward(
    torch::Tensor grad,
    torch::Tensor fLen,
    torch::Tensor gLen,
    torch::Tensor batchOffset,
    int maxFLen,
    int maxGLen,
    bool packOutput){

    auto tensorOpt = grad.options();
    auto dtype = grad.scalar_type();
    const int batchSize = fLen.size(0);
    const int hiddenSize = grad.size(-1);

    const auto deviceProperties = at::cuda::getCurrentDeviceProperties();
    const int maxNumWarp = deviceProperties->maxThreadsPerBlock / C10_WARP_SIZE;

    torch::Tensor fGrad = torch::empty({batchSize, maxFLen, hiddenSize}, tensorOpt);
    torch::Tensor gGrad = torch::empty({batchSize, maxGLen, hiddenSize}, tensorOpt);

    int64_t *batchOffsetPtr = (!packOutput) ? nullptr : batchOffset.data_ptr<int64_t>(); 

    const int workPerThread = 32;
    int numWarp = largestPowerOfTwo((std::max(maxFLen, maxGLen) + workPerThread-1) / workPerThread);
    numWarp = std::max(2, numWarp);
    numWarp = std::min(maxNumWarp, numWarp); 
    const int smemSize = numWarp * C10_WARP_SIZE;
    const dim3 threads(C10_WARP_SIZE, numWarp, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dtype, "transducer_joint_cuda_backward_kernel", ([&] {
        auto gradPtr = grad.data_ptr<scalar_t>();
        auto fLenPtr = fLen.data_ptr<int>();
        auto gLenPtr = gLen.data_ptr<int>(); 
        auto fGradPtr = fGrad.data_ptr<scalar_t>();
        auto gGradPtr = gGrad.data_ptr<scalar_t>();

        using acc_t = at::acc_type<scalar_t, true>;
        using vec_t = uint64_t;

        constexpr int vectFactor = sizeof(vec_t) / sizeof(scalar_t);
        constexpr int vecAlignment = std::alignment_of<vec_t>::value;
        bool memAlign = (reinterpret_cast<uint64_t>(gradPtr) % vecAlignment == 0) 
                        and (reinterpret_cast<uint64_t>(fGradPtr) % vecAlignment == 0) 
                        and (reinterpret_cast<uint64_t>(gGradPtr) % vecAlignment == 0);

        if (vectFactor > 1 and hiddenSize%vectFactor == 0 and memAlign){
            const dim3 blocks((hiddenSize+C10_WARP_SIZE*vectFactor-1)
                                /(C10_WARP_SIZE*vectFactor), maxFLen+maxGLen, batchSize);
            transducer_joint_combined_vec_backward
            <scalar_t, acc_t, vec_t, vectFactor, OffsetCalBwd>
            <<<blocks, threads, smemSize*sizeof(acc_t)>>>(
                gradPtr,
                fLenPtr, 
                gLenPtr, 
                batchOffsetPtr, 
                maxFLen,
                maxGLen,
                hiddenSize,
                packOutput,
                fGradPtr,
                gGradPtr);
        }
        else{
            const dim3 blocks((hiddenSize+C10_WARP_SIZE-1)/C10_WARP_SIZE, 
                                maxFLen + maxGLen, batchSize);
            transducer_joint_combined_backward<scalar_t, acc_t, OffsetCalBwd>
            <<<blocks, threads, smemSize*sizeof(acc_t)>>>(
                gradPtr,
                fLenPtr, 
                gLenPtr, 
                batchOffsetPtr, 
                maxFLen,
                maxGLen,
                hiddenSize,
                packOutput,
                fGradPtr,
                gGradPtr);
        }
    }));   

    return {fGrad, gGrad};
}
