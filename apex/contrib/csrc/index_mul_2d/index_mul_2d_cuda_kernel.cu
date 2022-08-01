#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Atomic.cuh>


__global__ void index_mul_2d_float_dim64(
    float *out, 
    const float *in1, 
    const float *in2, 
    const int64_t *idx1, 
    const int64_t size) 
{
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int bidx = blockIdx.x;
    const int start_idx = bidx * blockDim.y + tidy;
    constexpr int fea_dim = 64;

    if (start_idx < size) {
        int64_t vec_idx1 = (idx1[start_idx] * fea_dim) / 4 + tidx;
        int64_t vec_idx2 = (start_idx * fea_dim) / 4 + tidx;
        
        float4 res, src1, src2;
        src1 = reinterpret_cast<const float4 *>(in1)[vec_idx1];
        src2 = reinterpret_cast<const float4 *>(in2)[vec_idx2];
        res.x = src1.x * src2.x;
        res.y = src1.y * src2.y;
        res.z = src1.z * src2.z;
        res.w = src1.w * src2.w;
        reinterpret_cast<float4 *>(out)[vec_idx2] = res;
    }
}

__global__ void index_mul_2d_float(
    float *out, 
    const float *in1, 
    const float *in2, 
    const int64_t *idx1, 
    const int64_t size,
    const int64_t fea_dim) 
{
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int bidx = blockIdx.x;
    const int start_idx = bidx * blockDim.y + tidy;
    const int stride = blockDim.x;

    if (start_idx < size) {
        int64_t vec_idx1 = (idx1[start_idx] * fea_dim);
        int64_t vec_idx2 = (start_idx * fea_dim);
        
        for (int i = tidx; i < fea_dim; i += stride) {
            out[vec_idx2 + i] = in1[vec_idx1 + i] * in2[vec_idx2 + i];
        }
    }
}

__global__ void index_mul_2d_half(
    at::Half *out, 
    const at::Half *in1, 
    const at::Half *in2, 
    const int64_t *idx1, 
    const int64_t size,
    const int64_t fea_dim) 
{
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int bidx = blockIdx.x;
    const int start_idx = bidx * blockDim.y + tidy;
    const int stride = blockDim.x;

    if (start_idx < size) {
        int64_t vec_idx1 = (idx1[start_idx] * fea_dim);
        int64_t vec_idx2 = (start_idx * fea_dim);
        
        for (int i = tidx; i < fea_dim; i += stride) {
            out[vec_idx2 + i] = at::Half(static_cast<float>(in1[vec_idx1 + i]) * static_cast<float>(in2[vec_idx2 + i]));
        }
    }
}

__global__ void index_mul_2d_grad_float_dim64(
    float *grad_in1, 
    float *grad_in2,
    const float *grad_out, 
    const float *in1,
    const float *in2,
    const int64_t *idx1, 
    const int64_t size) 
{
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int bidx = blockIdx.x;
    const int start_idx = bidx * blockDim.y + tidy;
    constexpr int fea_dim = 64;

    if (start_idx < size) {
        int64_t vec_idx1 = (idx1[start_idx] * fea_dim) / 4 + tidx;
        int64_t vec_idx2 = (start_idx * fea_dim) / 4 + tidx;

        float4 src_in1, src_in2, src_grad_out, dst_grad_in2;
        src_grad_out = reinterpret_cast<const float4 *>(grad_out)[vec_idx2];
        src_in1 = reinterpret_cast<const float4 *>(in1)[vec_idx1];
        src_in2 = reinterpret_cast<const float4 *>(in2)[vec_idx2];
        int64_t grad_in1_base_idx = idx1[start_idx] * fea_dim + tidx * 4;
        gpuAtomicAdd(grad_in1 + grad_in1_base_idx + 0, src_grad_out.x * src_in2.x);
        gpuAtomicAdd(grad_in1 + grad_in1_base_idx + 1, src_grad_out.y * src_in2.y);
        gpuAtomicAdd(grad_in1 + grad_in1_base_idx + 2, src_grad_out.z * src_in2.z);
        gpuAtomicAdd(grad_in1 + grad_in1_base_idx + 3, src_grad_out.w * src_in2.w);
        dst_grad_in2.x = src_grad_out.x * src_in1.x;
        dst_grad_in2.y = src_grad_out.y * src_in1.y;
        dst_grad_in2.z = src_grad_out.z * src_in1.z;
        dst_grad_in2.w = src_grad_out.w * src_in1.w;
        reinterpret_cast<float4 *>(grad_in2)[vec_idx2] = dst_grad_in2; 
    }
}

__global__ void index_mul_2d_grad_float(
    float *grad_in1, 
    float *grad_in2,
    const float *grad_out, 
    const float *in1,
    const float *in2,
    const int64_t *idx1, 
    const int64_t size,
    const int64_t fea_dim) 
{
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int bidx = blockIdx.x;
    const int start_idx = bidx * blockDim.y + tidy;
    const int stride = blockDim.x;

    if (start_idx < size) {
        int64_t vec_idx1 = idx1[start_idx] * fea_dim;
        int64_t vec_idx2 = start_idx * fea_dim;

        for (int i = tidx; i < fea_dim; i += stride) {
            float src_in1 = in1[vec_idx1 + i];
            float src_in2 = in2[vec_idx2 + i];
            float src_grad_out = grad_out[vec_idx2 + i];
            grad_in2[vec_idx2 + i] = src_grad_out * src_in1;
            gpuAtomicAdd(grad_in1 + vec_idx1 + i, src_grad_out * src_in2);
        }
    }
}

__global__ void index_mul_2d_grad_half(
    at::Half *grad_in1, 
    at::Half *grad_in2,
    const at::Half *grad_out, 
    const at::Half *in1,
    const at::Half *in2,
    const int64_t *idx1, 
    const int64_t size,
    const int64_t fea_dim) 
{
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int bidx = blockIdx.x;
    const int start_idx = bidx * blockDim.y + tidy;
    const int stride = blockDim.x;

    if (start_idx < size) {
        int64_t vec_idx1 = idx1[start_idx] * fea_dim;
        int64_t vec_idx2 = start_idx * fea_dim;

        for (int i = tidx; i < fea_dim; i += stride) {
            float src_in1 = static_cast<float>(in1[vec_idx1 + i]);
            float src_in2 = static_cast<float>(in2[vec_idx2 + i]);
            float src_grad_out = static_cast<float>(grad_out[vec_idx2 + i]);
            grad_in2[vec_idx2 + i] = at::Half(src_grad_out * src_in1);
            gpuAtomicAdd(grad_in1 + vec_idx1 + i, at::Half(src_grad_out * src_in2));
        }
    }
}

__global__ void index_mul_2d_grad_grad_float_dim64(
    float *grad_grad_out,
    float *grad_in1,
    float *grad_in2,
    const float *grad_out,
    const float *grad_grad_in1,
    const float *grad_grad_in2,
    const float *in1,
    const float *in2,
    const int64_t *idx1,
    const int64_t size) 
{
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int bidx = blockIdx.x;
    const int start_idx = bidx * blockDim.y + tidy;
    constexpr int fea_dim = 64;

    if (start_idx < size) { 
        int64_t vec_idx1 = (idx1[start_idx] * fea_dim) / 4 + tidx;
        int64_t vec_idx2 = (start_idx * fea_dim) / 4 + tidx;

        float4 src_grad_grad_in1, src_in1, src_grad_grad_in2, src_in2, src_grad_out;
        float4 dst_grad_grad_out, dst_grad_in2;
        src_grad_grad_in1 = reinterpret_cast<const float4 *>(grad_grad_in1)[vec_idx1];
        src_in1 = reinterpret_cast<const float4 *>(in1)[vec_idx1];
        src_grad_grad_in2 = reinterpret_cast<const float4 *>(grad_grad_in2)[vec_idx2];
        src_in2 = reinterpret_cast<const float4 *>(in2)[vec_idx2];
        dst_grad_grad_out.x = src_grad_grad_in1.x * src_in2.x + src_grad_grad_in2.x * src_in1.x;
        dst_grad_grad_out.y = src_grad_grad_in1.y * src_in2.y + src_grad_grad_in2.y * src_in1.y;
        dst_grad_grad_out.z = src_grad_grad_in1.z * src_in2.z + src_grad_grad_in2.z * src_in1.z;
        dst_grad_grad_out.w = src_grad_grad_in1.w * src_in2.w + src_grad_grad_in2.w * src_in1.w;
        reinterpret_cast<float4 *>(grad_grad_out)[vec_idx2] = dst_grad_grad_out;
        src_grad_out = reinterpret_cast<const float4 *>(grad_out)[vec_idx2];
        int64_t grad_in1_base_idx = idx1[start_idx] * fea_dim + tidx * 4;
        gpuAtomicAdd(grad_in1 + grad_in1_base_idx + 0, src_grad_grad_in2.x * src_grad_out.x);
        gpuAtomicAdd(grad_in1 + grad_in1_base_idx + 1, src_grad_grad_in2.y * src_grad_out.y);
        gpuAtomicAdd(grad_in1 + grad_in1_base_idx + 2, src_grad_grad_in2.z * src_grad_out.z);
        gpuAtomicAdd(grad_in1 + grad_in1_base_idx + 3, src_grad_grad_in2.w * src_grad_out.w);
        dst_grad_in2.x = src_grad_grad_in1.x * src_grad_out.x;
        dst_grad_in2.y = src_grad_grad_in1.y * src_grad_out.y;
        dst_grad_in2.z = src_grad_grad_in1.z * src_grad_out.z;
        dst_grad_in2.w = src_grad_grad_in1.w * src_grad_out.w;
        reinterpret_cast<float4 *>(grad_in2)[vec_idx2] = dst_grad_in2;
    }
}

__global__ void index_mul_2d_grad_grad_float(
    float *grad_grad_out,
    float *grad_in1,
    float *grad_in2,
    const float *grad_out,
    const float *grad_grad_in1,
    const float *grad_grad_in2,
    const float *in1,
    const float *in2,
    const int64_t *idx1,
    const int64_t size,
    const int64_t fea_dim) 
{
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int bidx = blockIdx.x;
    const int start_idx = bidx * blockDim.y + tidy;
    const int stride = blockDim.x;
    
    if (start_idx < size) { 
        int64_t vec_idx1 = idx1[start_idx] * fea_dim;
        int64_t vec_idx2 = start_idx * fea_dim;

        for (int i = tidx; i < fea_dim; i += stride) {
            float src_grad_grad_in1 = grad_grad_in1[vec_idx1 + i];
            float src_grad_grad_in2 = grad_grad_in2[vec_idx2 + i];
            float src_in1 = in1[vec_idx1 + i];
            float src_in2 = in2[vec_idx2 + i];
            float src_grad_out = grad_out[vec_idx2 + i];
            grad_grad_out[vec_idx2 + i] = src_grad_grad_in1 * src_in2 + src_grad_grad_in2 * src_in1;
            grad_in2[vec_idx2 + i] = src_grad_grad_in1 * src_grad_out;
            gpuAtomicAdd(grad_in1 + vec_idx1 + i, src_grad_grad_in2 * src_grad_out);
        }
    }
}

__global__ void index_mul_2d_grad_grad_half(
    at::Half *grad_grad_out,
    at::Half *grad_in1,
    at::Half *grad_in2,
    const at::Half *grad_out,
    const at::Half *grad_grad_in1,
    const at::Half *grad_grad_in2,
    const at::Half *in1,
    const at::Half *in2,
    const int64_t *idx1,
    const int64_t size,
    const int64_t fea_dim) 
{
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int bidx = blockIdx.x;
    const int start_idx = bidx * blockDim.y + tidy;
    const int stride = blockDim.x;
    
    if (start_idx < size) { 
        int64_t vec_idx1 = idx1[start_idx] * fea_dim;
        int64_t vec_idx2 = start_idx * fea_dim;

        for (int i = tidx; i < fea_dim; i += stride) {
            float src_grad_grad_in1 = static_cast<float>(grad_grad_in1[vec_idx1 + i]);
            float src_grad_grad_in2 = static_cast<float>(grad_grad_in2[vec_idx2 + i]);
            float src_in1 = static_cast<float>(in1[vec_idx1 + i]);
            float src_in2 = static_cast<float>(in2[vec_idx2 + i]);
            float src_grad_out = static_cast<float>(grad_out[vec_idx2 + i]);
            grad_grad_out[vec_idx2 + i] = at::Half(src_grad_grad_in1 * src_in2 + src_grad_grad_in2 * src_in1);
            grad_in2[vec_idx2 + i] = at::Half(src_grad_grad_in1 * src_grad_out);
            gpuAtomicAdd(grad_in1 + vec_idx1 + i, at::Half(src_grad_grad_in2 * src_grad_out));
        }
    }
}

void index_mul_2d_float_foward_cuda(at::Tensor &out,
                                 const at::Tensor &in1,
                                 const at::Tensor &in2,
                                 const at::Tensor &idx1) {
    const int64_t size = in2.size(0);
    const int64_t fea_dim = in2.size(1);
    if (size < 0){
        return;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    if (fea_dim == 64) {
        const int BLOCK_THREADS_DIMX = 16;
        const int BLOCK_THREADS_DIMY = 16;
        const int BLOCK_NUMS = (size + BLOCK_THREADS_DIMY - 1) / BLOCK_THREADS_DIMY;

        index_mul_2d_float_dim64<<<BLOCK_NUMS, {BLOCK_THREADS_DIMX, BLOCK_THREADS_DIMY, 1}, 0, stream>>>(
            out.data_ptr<float>(), in1.data_ptr<float>(), in2.data_ptr<float>(), 
            idx1.data_ptr<int64_t>(), size);
    } else {
        const int BLOCK_THREADS_DIMX = 32;
        const int BLOCK_THREADS_DIMY = 8;
        const int BLOCK_NUMS = (size + BLOCK_THREADS_DIMY - 1) / BLOCK_THREADS_DIMY;

        index_mul_2d_float<<<BLOCK_NUMS, {BLOCK_THREADS_DIMX, BLOCK_THREADS_DIMY, 1}, 0, stream>>>(            
            out.data_ptr<float>(), in1.data_ptr<float>(), in2.data_ptr<float>(), 
            idx1.data_ptr<int64_t>(), size, fea_dim);
    }

    AT_CUDA_CHECK(cudaGetLastError());
}

void index_mul_2d_float_backward_cuda(at::Tensor &grad_in1,
                                   at::Tensor &grad_in2,
                                   const at::Tensor &grad_out,
                                   const at::Tensor &in1,
                                   const at::Tensor &in2,
                                   const at::Tensor &idx1) {
    const int64_t size = in2.size(0);
    const int64_t fea_dim = in2.size(1);
    if (size < 0){
        return;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (fea_dim == 64) {
        const int BLOCK_THREADS_DIMX = 16;
        const int BLOCK_THREADS_DIMY = 16;
        const int BLOCK_NUMS = (size + BLOCK_THREADS_DIMY - 1) / BLOCK_THREADS_DIMY;

        index_mul_2d_grad_float_dim64<<<BLOCK_NUMS, {BLOCK_THREADS_DIMX, BLOCK_THREADS_DIMY, 1}, 0, stream>>>(
            grad_in1.data_ptr<float>(), grad_in2.data_ptr<float>(), grad_out.data_ptr<float>(), 
            in1.data_ptr<float>(), in2.data_ptr<float>(), idx1.data_ptr<int64_t>(), size);

        AT_CUDA_CHECK(cudaGetLastError());
    } else {
        const int BLOCK_THREADS_DIMX = 32;
        const int BLOCK_THREADS_DIMY = 8;
        const int BLOCK_NUMS = (size + BLOCK_THREADS_DIMY - 1) / BLOCK_THREADS_DIMY;

        index_mul_2d_grad_float<<<BLOCK_NUMS, {BLOCK_THREADS_DIMX, BLOCK_THREADS_DIMY, 1}, 0, stream>>>(
            grad_in1.data_ptr<float>(), grad_in2.data_ptr<float>(), grad_out.data_ptr<float>(), 
            in1.data_ptr<float>(), in2.data_ptr<float>(), idx1.data_ptr<int64_t>(), size, fea_dim);
    }
}

void index_mul_2d_float_backward_backward_cuda(at::Tensor &grad_grad_out,
                                            at::Tensor &grad_in1,
                                            at::Tensor &grad_in2,
                                            const at::Tensor &grad_out,
                                            const at::Tensor &grad_grad_in1,
                                            const at::Tensor &grad_grad_in2,
                                            const at::Tensor &in1,
                                            const at::Tensor &in2,
                                            const at::Tensor &idx1) {
    const int64_t size = in2.size(0);
    const int64_t fea_dim = in2.size(1);
    if (size < 0){
        return;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (fea_dim == 64) {
        const int BLOCK_THREADS_DIMX = 16;
        const int BLOCK_THREADS_DIMY = 16;
        const int BLOCK_NUMS = (size + BLOCK_THREADS_DIMY - 1) / BLOCK_THREADS_DIMY;

        index_mul_2d_grad_grad_float_dim64<<<BLOCK_NUMS, {BLOCK_THREADS_DIMX, BLOCK_THREADS_DIMY, 1}, 0, stream>>>(
            grad_grad_out.data_ptr<float>(), grad_in1.data_ptr<float>(), grad_in2.data_ptr<float>(), 
            grad_out.data_ptr<float>(), grad_grad_in1.data_ptr<float>(), grad_grad_in2.data_ptr<float>(), 
            in1.data_ptr<float>(), in2.data_ptr<float>(), idx1.data_ptr<int64_t>(), size);
    } else {
        const int BLOCK_THREADS_DIMX = 32;
        const int BLOCK_THREADS_DIMY = 8;
        const int BLOCK_NUMS = (size + BLOCK_THREADS_DIMY - 1) / BLOCK_THREADS_DIMY;       

        index_mul_2d_grad_grad_float<<<BLOCK_NUMS, {BLOCK_THREADS_DIMX, BLOCK_THREADS_DIMY, 1}, 0, stream>>>(
            grad_grad_out.data_ptr<float>(), grad_in1.data_ptr<float>(), grad_in2.data_ptr<float>(), 
            grad_out.data_ptr<float>(), grad_grad_in1.data_ptr<float>(), grad_grad_in2.data_ptr<float>(), 
            in1.data_ptr<float>(), in2.data_ptr<float>(), idx1.data_ptr<int64_t>(), size, fea_dim); 
    }

    AT_CUDA_CHECK(cudaGetLastError());
}

void index_mul_2d_half_foward_cuda(at::Tensor &out,
                                const at::Tensor &in1,
                                const at::Tensor &in2,
                                const at::Tensor &idx1) {
    const int64_t size = in2.size(0);
    const int64_t fea_dim = in2.size(1);
    if (size < 0){
        return;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    const int BLOCK_THREADS_DIMX = 32;
    const int BLOCK_THREADS_DIMY = 8;
    const int BLOCK_NUMS = (size + BLOCK_THREADS_DIMY - 1) / BLOCK_THREADS_DIMY;

    index_mul_2d_half<<<BLOCK_NUMS, {BLOCK_THREADS_DIMX, BLOCK_THREADS_DIMY, 1}, 0, stream>>>(            
        out.data_ptr<at::Half>(), in1.data_ptr<at::Half>(), in2.data_ptr<at::Half>(), 
        idx1.data_ptr<int64_t>(), size, fea_dim);

    AT_CUDA_CHECK(cudaGetLastError());
}

void index_mul_2d_half_backward_cuda(at::Tensor &grad_in1,
                                 at::Tensor &grad_in2,
                                 const at::Tensor &grad_out,
                                 const at::Tensor &in1,
                                 const at::Tensor &in2,
                                 const at::Tensor &idx1) {
    const int64_t size = in2.size(0);
    const int64_t fea_dim = in2.size(1);
    if (size < 0){
        return;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int BLOCK_THREADS_DIMX = 32;
    const int BLOCK_THREADS_DIMY = 8;
    const int BLOCK_NUMS = (size + BLOCK_THREADS_DIMY - 1) / BLOCK_THREADS_DIMY;

    index_mul_2d_grad_half<<<BLOCK_NUMS, {BLOCK_THREADS_DIMX, BLOCK_THREADS_DIMY, 1}, 0, stream>>>(
        grad_in1.data_ptr<at::Half>(), grad_in2.data_ptr<at::Half>(), grad_out.data_ptr<at::Half>(), 
        in1.data_ptr<at::Half>(), in2.data_ptr<at::Half>(), idx1.data_ptr<int64_t>(), size, fea_dim);
}

void index_mul_2d_half_backward_backward_cuda(at::Tensor &grad_grad_out,
                                          at::Tensor &grad_in1,
                                          at::Tensor &grad_in2,
                                          const at::Tensor &grad_out,
                                          const at::Tensor &grad_grad_in1,
                                          const at::Tensor &grad_grad_in2,
                                          const at::Tensor &in1,
                                          const at::Tensor &in2,
                                          const at::Tensor &idx1) {
    const int64_t size = in2.size(0);
    const int64_t fea_dim = in2.size(1);
    if (size < 0){
        return;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int BLOCK_THREADS_DIMX = 32;
    const int BLOCK_THREADS_DIMY = 8;
    const int BLOCK_NUMS = (size + BLOCK_THREADS_DIMY - 1) / BLOCK_THREADS_DIMY;       

    index_mul_2d_grad_grad_half<<<BLOCK_NUMS, {BLOCK_THREADS_DIMX, BLOCK_THREADS_DIMY, 1}, 0, stream>>>(
        grad_grad_out.data_ptr<at::Half>(), grad_in1.data_ptr<at::Half>(), grad_in2.data_ptr<at::Half>(), 
        grad_out.data_ptr<at::Half>(), grad_grad_in1.data_ptr<at::Half>(), grad_grad_in2.data_ptr<at::Half>(), 
        in1.data_ptr<at::Half>(), in2.data_ptr<at::Half>(), idx1.data_ptr<int64_t>(), size, fea_dim); 

    AT_CUDA_CHECK(cudaGetLastError());
}