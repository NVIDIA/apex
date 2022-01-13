#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <ATen/AccumulateType.h>

#ifdef OLD_GENERATOR_PATH
#include <ATen/CUDAGeneratorImpl.h>
#else
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/macros/Macros.h>

#include "philox.cuh"

// Warp reduce kernels to reduce N groups of data into N numbers, where N = warpSize / width.
// width should be a power of 2 and should be less than warpSize.
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

/*
Figure out vectorization type for masks.
Similar to how PyTorch figures out acc_t here:
aten/src/ATen/AccumulateType.h 
*/
template <int V>
struct MaskVecType { };

template <> struct MaskVecType<1> { using type = uint8_t; };
template <> struct MaskVecType<2> { using type = uint16_t; };
template <> struct MaskVecType<4> { using type = uint32_t; };

template<int V>
using mvec_type = typename MaskVecType<V>::type;

// Helper class to calculate pointer offset that can be shared by different flavors of kernels.
// For fwd, batch offset and stride are different for packing and non-packing mode.
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

// Helper class to calculate pointer offset that can be shared by different flavors of kernels
// For bwd, batch offset and stride are different for packing and non-packing mode.
// The reducion is done for two input tensors. Therefore, generating two sets of offsets
// according to bwdFasterDim can lead to a unified implementation in the actual kernel.
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
    bool bwdFasterDim;  // whether doing bwd on the faster moving dimension

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


// Vanila transducer joint forward kernel
// Detail of this joint function can be found in: 
// [1] Sequence Transduction with Recurrent Neural Networks.

// f is a tensor of shape [batch, T, H]
// g is a tensor of shape [batch, U, H]
// the transducer joint does
// sum = f.unsqueeze(dim=2) + g.unsqueeze(dim=1)
// The resultant tensor is of shape [batch, T, U, H]
// Each thread block is working on one "batch" of data in the output tensor, [batch, t, u, :]

// This joint function can optionally pack the output where the output tensor with a shape of
// [B, T, U, H] is packed into [B_packed, H].
// Don't-care region (t > fLen) or (u > gLen) is removed.
// To enable packing, the starting offset for each batch need to be specified with batchOffset.
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
        // Need to write finite data to don't-care region because we instantiate the result tensor
        // with torch::empty for performance reasons. Even though it is don't-care region, the 
        // contents need to be finite, otherwise could lead to NaN in WGRAD.
        // In packing mode, this write is no longer necessary as we remove the don't-care region
        // from the output.
        // Picking -1 (over 0) here for ease of testing.
        #pragma unroll
        for (int h = threadIdx.x; h < hiddenSize; h += blockDim.x){
            if (h < hiddenSize){
                mySum[h] = -1;
            }
        }    
    }
}

/*
Tiled version of the joint forward kernel
Detail of this joint function can be found in: 
[1] Sequence Transduction with Recurrent Neural Networks.

f is a tensor of shape [batch, T, H]
g is a tensor of shape [batch, U, H]
the transducer joint does
sum = f.unsqueeze(dim=2) + g.unsqueeze(dim=1)
The resultant tensor is of shape [batch, T, U, H]
Each thread is working on a tile of the shape of tileF x tileG in the result tensor.
The input for the tile is first loaded in the register and is reused tileG and tileF times. 

This joint function can optionally pack the output where the output tensor with a shape of
[B, T, U, H] is packed into [B_packed, H].
Don't-care region (t > fLen) or (u > gLen) is removed.
To enable packing, the starting offset for each batch need to be specified with batchOffset.

Optionally this joint function performs ReLU and/or dropout on the joint output, which is 
controlled by arguments relu and dropout, respectively. philoxArgs is argument used for generating
pseudorandom number. When at least one of operations in ReLU and dropout is activated, the joint
function is a masked operation, which is controlled by the template argument masked. In this case, 
masks are saved to backward.
*/
template <typename scalar_t, int tileF, int tileG, int U, class OffsetCal, bool masked>
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
    bool relu, 
    bool dropout,
    float p,
    at::PhiloxCudaState philoxArgs,
    scalar_t *sum,
    uint8_t *mask) {

    static_assert(U == 4, "U has to be 4, as random numbers are generated in batch of 4");

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
    uint8_t *myMask = mask + myBatchOffset + t*strideF + u*hiddenSize + hOffset;

    // The following code is only needed for dropout. We try to bypass them as much as possible.
    auto seeds = masked ? at::cuda::philox::unpack(philoxArgs) 
                            : std::make_tuple(static_cast<uint64_t>(0), static_cast<uint64_t>(0));
    uint64_t tid = masked ? (static_cast<uint64_t>(blockIdx.z)*gridDim.y*gridDim.x + 
                        blockIdx.y*gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x
                            : 0;
    Philox ph(std::get<0>(seeds), tid, std::get<1>(seeds)); 
    scalar_t scale = masked ? ((p == 0) ? 0 : 1 / p) : 0;  
    bool dropoutMask[U];

    if (t < myFLen and u < myGLen and hOffset+h < hiddenSize){    
        // register buffers for tiled input reuse
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
                    int idx = i*tileG + j;
                    if (masked and dropout and idx % U == 0){
                        // For performance, generate 4 random numbers in one shot
                        // auto rand4 = curand_uniform4(&state);
                        auto rand4 = uniform4(ph());
                        dropoutMask[0] = rand4.x < p;
                        dropoutMask[1] = rand4.y < p;
                        dropoutMask[2] = rand4.z < p;
                        dropoutMask[3] = rand4.w < p;
                    }

                    if (u + j < myGLen){
                        scalar_t out = fBuffer[i] + gBuffer[j];
                        if (masked){
                            // Apply ReLU here when relu is True
                            bool localMask = relu ? (out>0) : 1;
                            localMask = dropout ? localMask & dropoutMask[idx%U] : localMask;
                            out = dropout ? out*localMask*scale : out*localMask;
                            myMask[i*strideF + j*hiddenSize + h] = static_cast<uint8_t>(localMask);
                        }
                        mySum[i*strideF + j*hiddenSize + h] = out;
                    }
                    else if (packOutput == false and u + j < maxGLen)
                        mySum[i*strideF + j*hiddenSize + h] = -1;
                }
            }
            else if (packOutput == false and t + i < maxFLen){
                // Again need to write finite data to don't-care region
                #pragma unroll
                for (int j = 0; j < tileG; ++j){
                    if (u + j < maxGLen)
                        mySum[i*strideF + j*hiddenSize + h] = -1;
                }
            }
        }
    }
    else if (packOutput == false and t < maxFLen and u < maxGLen and hOffset+h < hiddenSize){
        // Only need to ensure the finity in normal mode
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

/*
Bwd operation (reduction) on one input tensor. Since the operation performed for the two input
tensors are exactly the same, only one kernel is needed, and the different indexing offsets
and strides are handled by OffsetCalBwd.

When packing is enabled in the fwd op, unpacking is needed to restore the gradients in a 
non-packed form.

When ReLU and/or dropout are performed in the fwd pass, this operation becomes a masked operation,
and mask contains the mask information.
*/
template <typename scalar_t, typename acc_t, class OffsetCal, bool masked>
__device__ void transducer_joint_single_backward(
    const scalar_t *grad,
    const uint8_t *mask,
    const int *fLen,
    const int *gLen,
    const int64_t *batchOffset,
    int64_t maxFLen,
    int64_t maxGLen,
    int64_t hiddenSize,
    bool packOutput,
    bool bwdFasterDim,  // whether bwd on the faster moving dimension (u)
    float scale,
    scalar_t *inGrad,
    int yBlockOffset=0) {


    const int batch = blockIdx.z;
    // For the second input tensor, this offset need to be subtracted because the first yBlockOffset
    // sets of thread blocks are for the first input tensor.
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
        const scalar_t *myGrad = grad + myBatchOffset + x*strideX + hOffset;
        const uint8_t *myMask = masked ? mask + myBatchOffset + x*strideX + hOffset : nullptr;
        
        // Each warp reduces numYPerWarp "y" first
        acc_t warpSum = 0;
        auto numYPerWarp = (myYLen+numWarp-1)/numWarp;
        #pragma unroll
        for (int warpY = 0; warpY < numYPerWarp; ++warpY){
            auto y = wid*numYPerWarp + warpY;
            if (y < myYLen and (hOffset+lid) < hiddenSize)
                if (masked)
                    warpSum += static_cast<acc_t>(myGrad[y*strideY + lid]) * myMask[y*strideY + lid] * scale;
                else    
                    warpSum += myGrad[y*strideY + lid];
        }

        // transpose partial sum in SMEM and reduce further using warpReduce
        smem[lid*numWarp + wid] = warpSum;
        __syncthreads();
        auto sum = smem[wid*C10_WARP_SIZE + lid];
        sum = warpReduce(sum, numWarp);

        // a a b b c c d d
        // a a b b c c d d
        // a a b b c c d d
        // a a b b c c d d
        // example of 4 warps (a, b, c, d) with 8 threads per warp
        // Each warp need 8 / 4 = 2 threads to write the results.
        if (hOffset+wid*C10_WARP_SIZE/numWarp+lid/numWarp < hiddenSize){
            if (lid % numWarp == 0){
                myInGrad[wid*C10_WARP_SIZE/numWarp + lid/numWarp] = sum;
            }
        }
    }
    else if (wid == 0 and hOffset + lid < hiddenSize){
        // Need to ensure the grad is zero for don't care region
        myInGrad[lid] = 0;
    }
}

/*
Actual bwd (reduction) kernel get launched.
Call transducer_joint_single_backward twice on two input tensors. 
The two bwd ops are launched together, the first op uses blockIdx.y < maxFLen, and the second op 
uses the rest.
When ReLU and/or dropout are performed in the fwd pass, this operation becomes a masked operation,
and mask contains the mask information.
*/
template <typename scalar_t, typename acc_t, class OffsetCal, bool masked>
__global__ void transducer_joint_combined_backward(
    const scalar_t *grad,
    const uint8_t *mask,
    const int *fLen,
    const int *gLen,
    const int64_t *batchOffset,
    int64_t maxFLen,
    int64_t maxGLen,
    int64_t hiddenSize,
    bool packOutput,
    float scale,
    scalar_t *fGrad,
    scalar_t *gGrad) {
    if (blockIdx.y < maxFLen){
        transducer_joint_single_backward<scalar_t, acc_t, OffsetCal, masked>(
            grad,
            mask,
            fLen,
            gLen,
            batchOffset,
            maxFLen,
            maxGLen,
            hiddenSize,
            packOutput,
            false,
            scale,
            fGrad);
    }
    else{
        transducer_joint_single_backward<scalar_t, acc_t, OffsetCal, masked>(
            grad,
            mask,
            fLen,
            gLen,
            batchOffset,
            maxFLen,
            maxGLen,
            hiddenSize,
            packOutput,
            true,
            scale,
            gGrad,
            maxFLen);
    }  
}

/*
Vectorized version of transducer_joint_single_backward
Doing exact same operation as transducer_joint_single_backward except the load and store are
vectorized.
When packing is enabled in the fwd op, unpacking is needed to restore the gradients in a 
non-packed form.
When ReLU and/or dropout are performed in the fwd pass, this operation becomes a masked operation,
and mask contains the mask information.
*/
template <typename scalar_t, typename acc_t, typename vec_t, int V, class OffsetCal, bool masked>
__device__ void transducer_joint_single_vec_backward(
    const scalar_t *grad,
    const uint8_t *mask,
    const int *fLen,
    const int *gLen,
    const int64_t *batchOffset,
    int64_t maxFLen,
    int64_t maxGLen,
    int64_t hiddenSize,
    bool packOutput,
    bool bwdFasterDim,
    float scale,
    scalar_t *inGrad,
    int yBlockOffset=0){

    const int batch = blockIdx.z;
    const int x = blockIdx.y - yBlockOffset;
    const int hOffset = blockIdx.x*C10_WARP_SIZE*V;
    const int wid = threadIdx.y;
    const int lid = threadIdx.x;
    const int numWarp = blockDim.y;

    // Figure out the vectorization type for mask
    using mvec_t = mvec_type<V>;

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
    uint8_t maskBuffer[V];
    scalar_t outBuffer[V];
    auto myInGradVec = reinterpret_cast<vec_t*>(myInGrad);
    auto outBufferVec = reinterpret_cast<vec_t*>(outBuffer);

    if (x < myXLen){
        const auto myBatchOffset = offsetCal.getBatchOffset();
        const auto strideX = offsetCal.getStrideX();
        const auto strideY = offsetCal.getStrideY();
        const scalar_t *myGrad = grad + myBatchOffset + x*strideX + hOffset;
        const uint8_t *myMask = masked ? mask + myBatchOffset + x*strideX + hOffset
                                            :nullptr;

        for (int i = 0; i < V; ++i)
            warpSum[i] = 0;

        // Each warp reduces numYPerWarp "y" first
        auto numYPerWarp = (myYLen+numWarp-1)/numWarp;
        for (int warpY = 0; warpY < numYPerWarp; ++warpY){
            auto y = wid*numYPerWarp + warpY;
            auto myGradVec = reinterpret_cast<vec_t const *>(myGrad + y*strideY);
            auto myMaskVec = masked ? reinterpret_cast<mvec_t const *>(myMask + y*strideY)
                                        : nullptr;
            auto inBufferVec = reinterpret_cast<vec_t*>(inBuffer);
            auto maskBufferVec = reinterpret_cast<mvec_t*>(maskBuffer);
            if (hOffset + lid*V < hiddenSize and y < myYLen){
                *inBufferVec = myGradVec[lid];  // vectorized load
                if (masked){
                    *maskBufferVec = myMaskVec[lid];
                    #pragma unroll
                    for (int i = 0; i < V; ++i)
                        warpSum[i] += static_cast<acc_t>(inBuffer[i]) * maskBuffer[i] * scale;
                }
                else{
                    #pragma unroll
                    for (int i = 0; i < V; ++i)
                        warpSum[i] += inBuffer[i];
                }
            }
        }
        
        // transpose partial sum in SMEM and reduce further using warpReduce
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

        // a a b b c c d d
        // a a b b c c d d
        // a a b b c c d d
        // a a b b c c d d
        // example of 4 warps (a, b, c, d) with 8 threads per warp
        // Each warp need 8 / 4 = 2 threads to write the results.
        if (lid % numWarp == 0 and hOffset+(wid*C10_WARP_SIZE/numWarp + lid/numWarp)*V < hiddenSize)
            myInGradVec[wid*C10_WARP_SIZE/numWarp + lid/numWarp] = *outBufferVec;     
    }
    else if (wid == 0 and hOffset + lid*V < hiddenSize){
        // Need to ensure the grad is zero for don't care region
        myInGradVec[lid] = 0;
    }
}

/*
Vecotrized version of transducer_joint_combined_backward
Call transducer_joint_single_vec_backward twice on two input tensors. 
The two bwd ops are launched together, the first op uses blockIdx.y < maxFLen, and the second op 
uses the rest.
When ReLU and/or dropout are performed in the fwd pass, this operation becomes a masked operation,
and mask contains the mask information.
*/
template <typename scalar_t, typename acc_t, typename vec_t, int V, class OffsetCal, bool masked>
__global__ void transducer_joint_combined_vec_backward(
    const scalar_t *grad,
    const uint8_t *mask,
    const int *fLen,
    const int *gLen,
    const int64_t *batchOffset,
    int64_t maxFLen,
    int64_t maxGLen,
    int64_t hiddenSize,
    bool packOutput,
    float scale,
    scalar_t *fGrad,
    scalar_t *gGrad) {
    if (blockIdx.y < maxFLen){
        transducer_joint_single_vec_backward<scalar_t, acc_t, vec_t, V, OffsetCal, masked>(
            grad,
            mask,
            fLen,
            gLen,
            batchOffset,
            maxFLen,
            maxGLen,
            hiddenSize,
            packOutput,
            false,
            scale,
            fGrad);
    }
    else{
        transducer_joint_single_vec_backward<scalar_t, acc_t, vec_t, V, OffsetCal, masked>(
            grad,
            mask,
            fLen,
            gLen,
            batchOffset,
            maxFLen,
            maxGLen,
            hiddenSize,
            packOutput,
            true,
            scale,
            gGrad,
            maxFLen);
    }  
}




std::vector<torch::Tensor> transducer_joint_cuda_forward(
    torch::Tensor f,
    torch::Tensor g,
    torch::Tensor fLen,
    torch::Tensor gLen,
    torch::Tensor batchOffset,
    int64_t packedBatch,
    int opt,
    bool packOutput,
    bool relu,
    bool dropout,
    float dropoutProb,
    int tileSize){

    
    auto tensorOpt = f.options();
    auto dtype = f.scalar_type();
    const auto batchSize = f.size(0);
    const auto maxFLen = f.size(1);
    const auto maxGLen = g.size(1);
    const auto hiddenSize = f.size(2);
    bool masked = dropout or relu;
    
    int64_t *batchOffsetPtr = nullptr;
    torch::Tensor sum, mask;
    auto maskOpt = tensorOpt.dtype(torch::kUInt8);
    if (!packOutput){
        sum = torch::empty({batchSize, maxFLen, maxGLen, hiddenSize}, tensorOpt);
        batchOffsetPtr = nullptr;
        if (masked)
            mask = torch::empty({batchSize, maxFLen, maxGLen, hiddenSize}, maskOpt);
    }
    else{
        sum = torch::empty({packedBatch, hiddenSize}, tensorOpt);    
        batchOffsetPtr = batchOffset.data_ptr<int64_t>();
        if (masked)
            mask = torch::empty({packedBatch, hiddenSize}, maskOpt);
    }
    uint8_t *maskPtr = masked ? mask.data_ptr<uint8_t>() : nullptr;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    TORCH_CHECK(opt == 0 or opt == 1, "Got an invalid optimization level ", opt);
    // Simple heuristics
    const int numThread = std::min(128, (static_cast<int>(hiddenSize)+C10_WARP_SIZE-1)
                                        / C10_WARP_SIZE * C10_WARP_SIZE);
    
    if (opt == 0){
        // vanilla kernel
        const int threads = numThread;
        const dim3 blocks(maxGLen, maxFLen, batchSize);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(dtype, "transducer_joint_forward", ([&] {
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
        }));  
    }
    if (opt == 1){
        // tiled version. For simplicity, assume tileF == tileG, even though the kernel can 
        // support more general cases.
        const int threads = numThread;
        const int hiddenPerBlock = numThread;
        const int hiddenBlock = (hiddenSize + hiddenPerBlock - 1) / hiddenPerBlock;
        const dim3 blocks(  (maxGLen+tileSize-1)/tileSize * hiddenBlock, 
                            (maxFLen+tileSize-1)/tileSize, 
                            batchSize);

        TORCH_CHECK(tileSize == 1 or tileSize == 2 or tileSize == 4, 
                "Expected tileSize to be in [1, 2, 4], but got ", tileSize);

        at::PhiloxCudaState rng_engine_inputs;
        if (masked){
            // set up PRG when the input is masked. rng_engine_inputs will be used as a space filler 
            // for non-masked calls.
            // Therefore no need to initialize.
            c10::optional<at::Generator> gen_;
            auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(gen_, 
                                                    at::cuda::detail::getDefaultCUDAGenerator());
            // counterOffset records how many cuRAND calls each thread makes. For a tiled kernel, 
            // each thread processes tileF * tileG output elements. 
            int64_t counterOffset = tileSize * tileSize;
            {
                std::lock_guard<std::mutex> lock(gen->mutex_);
                rng_engine_inputs = gen->philox_cuda_state(counterOffset);
            }
        }

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(dtype, "transducer_joint_forward", ([&] {
            void(*kernel)(const scalar_t*, const scalar_t*, const int*, const int*, const int64_t*, 
                            int64_t, int64_t, int64_t, int64_t, bool, bool, bool, float, 
                            at::PhiloxCudaState, scalar_t*, uint8_t*);
            if (masked){
                switch (tileSize){
                    case 2:
                        kernel = &transducer_joint_tiled_forward<scalar_t, 2, 2, 4, OffsetCalFwd, 
                                                                    true>;
                        break;
                    case 4:
                        kernel = &transducer_joint_tiled_forward<scalar_t, 4, 4, 4, OffsetCalFwd, 
                                                                    true>;
                        break;
                }
            }
            else{
                switch (tileSize){
                    case 1:
                        kernel = &transducer_joint_tiled_forward<scalar_t, 1, 1, 4, OffsetCalFwd, 
                                                                    false>;
                        break;
                    case 2:
                        kernel = &transducer_joint_tiled_forward<scalar_t, 2, 2, 4, OffsetCalFwd, 
                                                                    false>;
                        break;
                    case 4:
                        kernel = &transducer_joint_tiled_forward<scalar_t, 4, 4, 4, OffsetCalFwd, 
                                                                    false>;
                        break;
                }
            }
            
            kernel<<<blocks, threads, 0, stream>>>(
                f.data_ptr<scalar_t>(),
                g.data_ptr<scalar_t>(),
                fLen.data_ptr<int>(),
                gLen.data_ptr<int>(),
                batchOffsetPtr,
                maxFLen,
                maxGLen,
                hiddenSize,
                hiddenPerBlock,
                packOutput,
                relu,
                dropout,
                1.0f - dropoutProb,
                rng_engine_inputs,
                sum.data_ptr<scalar_t>(),
                maskPtr);
        }));  
    }
 
    C10_CUDA_CHECK(cudaGetLastError());
    if (masked) 
        return {sum, mask};
    else
        return {sum};
}

std::vector<torch::Tensor> transducer_joint_cuda_backward(
    std::vector<torch::Tensor> in,
    torch::Tensor fLen,
    torch::Tensor gLen,
    torch::Tensor batchOffset,
    int maxFLen,
    int maxGLen,
    bool packOutput,
    float scale){

    auto grad = in[0];
    bool masked = (in.size() == 2);
    uint8_t *maskPtr = masked ? in[1].data_ptr<uint8_t>() : nullptr;

    auto tensorOpt = grad.options();
    auto dtype = grad.scalar_type();
    const int batchSize = fLen.size(0);
    const int hiddenSize = grad.size(-1);

    const auto deviceProperties = at::cuda::getCurrentDeviceProperties();
    const int maxNumWarp = deviceProperties->maxThreadsPerBlock / C10_WARP_SIZE;

    torch::Tensor fGrad = torch::empty({batchSize, maxFLen, hiddenSize}, tensorOpt);
    torch::Tensor gGrad = torch::empty({batchSize, maxGLen, hiddenSize}, tensorOpt);

    int64_t *batchOffsetPtr = (!packOutput) ? nullptr : batchOffset.data_ptr<int64_t>(); 

    // The number "y" I would like each thread to work on
    const int workPerThread = 32;   
    // Since the bwd for f and g have the same thread block size, we need to use the max of the two.
    int numWarp = largestPowerOfTwo((std::max(maxFLen, maxGLen) + workPerThread-1) / workPerThread);
    // Would like to have at least 2 warps 
    numWarp = std::max(2, numWarp);
    // cap on the maximum number of warps allowed
    numWarp = std::min(maxNumWarp, numWarp); 

    // Need smem for transposing the partial sum. The partial sum is in a matrix of the shape
    // numWarp x warpSize
    const int smemSize = numWarp * C10_WARP_SIZE;
    const dim3 threads(C10_WARP_SIZE, numWarp, 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dtype, "transducer_joint_cuda_backward_kernel", ([&] {
        auto gradPtr = grad.data_ptr<scalar_t>();
        auto fLenPtr = fLen.data_ptr<int>();
        auto gLenPtr = gLen.data_ptr<int>(); 
        auto fGradPtr = fGrad.data_ptr<scalar_t>();
        auto gGradPtr = gGrad.data_ptr<scalar_t>();

        // resolve the acc_t type
        using acc_t = at::acc_type<scalar_t, true>;
        using vec_t = uint64_t;

        constexpr int vectFactor = sizeof(vec_t) / sizeof(scalar_t);
        constexpr int vecAlignment = std::alignment_of<vec_t>::value;

        // if all input and output tensors meet the alignment requirement
        bool memAlign = (reinterpret_cast<uint64_t>(gradPtr) % vecAlignment == 0) 
                        and (reinterpret_cast<uint64_t>(fGradPtr) % vecAlignment == 0) 
                        and (reinterpret_cast<uint64_t>(gGradPtr) % vecAlignment == 0);

        if (vectFactor > 1 and hiddenSize%vectFactor == 0 and memAlign){
            // If vectorization helps and the alignment requirement is met, use the vectorized 
            // kernel. For simplicity, hiddenSize needs to be a multiple vecFactor.
            const dim3 blocks(  (hiddenSize+C10_WARP_SIZE*vectFactor-1)/(C10_WARP_SIZE*vectFactor), 
                                maxFLen+maxGLen, 
                                batchSize);
            if (masked){
                transducer_joint_combined_vec_backward
                    <scalar_t, acc_t, vec_t, vectFactor, OffsetCalBwd, true>
                    <<<blocks, threads, smemSize*sizeof(acc_t)>>>(
                    gradPtr,
                    maskPtr,
                    fLenPtr, 
                    gLenPtr, 
                    batchOffsetPtr, 
                    maxFLen,
                    maxGLen,
                    hiddenSize,
                    packOutput,
                    scale,
                    fGradPtr,
                    gGradPtr);
            }
            else{
                transducer_joint_combined_vec_backward
                <scalar_t, acc_t, vec_t, vectFactor, OffsetCalBwd, false>
                <<<blocks, threads, smemSize*sizeof(acc_t)>>>(
                    gradPtr,
                    maskPtr,
                    fLenPtr, 
                    gLenPtr, 
                    batchOffsetPtr, 
                    maxFLen,
                    maxGLen,
                    hiddenSize,
                    packOutput,
                    scale,
                    fGradPtr,
                    gGradPtr);    
            }
        }
        else{
            const dim3 blocks((hiddenSize+C10_WARP_SIZE-1)/C10_WARP_SIZE, 
                                maxFLen + maxGLen, batchSize);
            if (masked){
                transducer_joint_combined_backward<scalar_t, acc_t, OffsetCalBwd, true>
                <<<blocks, threads, smemSize*sizeof(acc_t)>>>(
                    gradPtr,
                    maskPtr,
                    fLenPtr, 
                    gLenPtr, 
                    batchOffsetPtr, 
                    maxFLen,
                    maxGLen,
                    hiddenSize,
                    packOutput,
                    scale,
                    fGradPtr,
                    gGradPtr);
            }
            else{
                transducer_joint_combined_backward<scalar_t, acc_t, OffsetCalBwd, false>
                <<<blocks, threads, smemSize*sizeof(acc_t)>>>(
                    gradPtr,
                    maskPtr,
                    fLenPtr, 
                    gLenPtr, 
                    batchOffsetPtr, 
                    maxFLen,
                    maxGLen,
                    hiddenSize,
                    packOutput,
                    scale,
                    fGradPtr,
                    gGradPtr);
            }
        }
    }));   

    return {fGrad, gGrad};
}
