#include <torch/extension.h>
#include <c10/cuda/CUDAMathCompat.h>

// Swish (SiLU) activation function: SiLU(x) = x * sigmoid(x)
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// CUDA kernel for Fused Bias SwiGLU with chunking
template <typename T>
__global__ void fused_bias_swiglu_kernel(const T* __restrict__ input, 
                                         const T* __restrict__ bias, 
                                         T* __restrict__ output, 
                                         int half_dim, 
                                         int max_index) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = output_idx / half_dim;
    int input_idx = output_idx + row_idx * half_dim;
    int col_idx = output_idx - row_idx * half_dim;

    if (output_idx < max_index) {
        int other_chunk_idx = input_idx + half_dim;
        int other_col_idx = col_idx + half_dim;

        T x1 = input[input_idx] + bias[col_idx];
        T x2 = input[other_chunk_idx] + bias[other_col_idx];
        output[output_idx] = silu(x1) * x2;
    }
}

// CUDA Kernel: Computes the backward pass for fused bias SwiGLU
template <typename T>
__global__ void fused_bias_swiglu_backward_kernel(
    const T* __restrict__ grad_output, 
    const T* __restrict__ input, 
    const T* __restrict__ bias, 
    T* __restrict__ grad_input, 
    int half_dim, int max_index) {

    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row_idx = output_idx / half_dim;
    int input_idx = output_idx + row_idx * half_dim;
    int col_idx = output_idx - row_idx * half_dim;

    if (output_idx < max_index) {
        int other_chunk_idx = input_idx + half_dim;
        int other_col_idx = col_idx + half_dim;

        T y1 = input[input_idx] + bias[col_idx];
        T y2 = input[other_chunk_idx] + bias[other_col_idx];

        T sigmoid_y1 = 1.0f / (1.0f + expf(-y1));
        T silu_y1 = y1 * sigmoid_y1;

        T g = grad_output[output_idx];
        T d_y1 = g * sigmoid_y1 * (1.0f + y1 * (1.0f - sigmoid_y1)) * y2;
        T d_y2 = g * silu_y1;

        grad_input[input_idx] += d_y1;
        grad_input[other_chunk_idx] += d_y2;
    }
}

// PyTorch interface for CUDA kernel
torch::Tensor fused_bias_swiglu_forward(torch::Tensor input, torch::Tensor bias) {
    int batch_size = input.size(0);
    int hidden_dim = input.size(1);
    int half_dim = hidden_dim / 2;
    TORCH_CHECK(hidden_dim % 2 == 0, "Hidden dimension must be divisible by 2 for SwiGLU");
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA device");
    TORCH_CHECK(bias.is_cuda(), "Bias must be on CUDA device");

    input = input.contiguous();
    bias = bias.contiguous();

    auto output = torch::zeros({batch_size, hidden_dim / 2}, input.options());

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int threads = prop.maxThreadsPerBlock;
    int blocks = (batch_size * half_dim + threads - 1) / threads;
    blocks = min(blocks, prop.maxGridSize[0]);
    

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fused_bias_swiglu_forward", [&] {
        fused_bias_swiglu_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), half_dim, half_dim * batch_size
        );
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel execution error: " << cudaGetErrorString(err) << std::endl;
    }

    return output;
}

// PyTorch interface for backward pass
torch::Tensor fused_bias_swiglu_backward(
    torch::Tensor grad_output, torch::Tensor input, torch::Tensor bias) {

    int batch_size = input.size(0);
    int hidden_dim = input.size(1);
    int half_dim = hidden_dim / 2;

    TORCH_CHECK(hidden_dim % 2 == 0, "Hidden dimension must be divisible by 2 for SwiGLU");

    auto grad_input = torch::zeros_like(input);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int threads = prop.maxThreadsPerBlock;
    int blocks = (batch_size * half_dim + threads - 1) / threads;
    blocks = min(blocks, prop.maxGridSize[0]);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "fused_bias_swiglu_backward", [&] {
        fused_bias_swiglu_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            half_dim, half_dim * batch_size
        );
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel execution error: " << cudaGetErrorString(err) << std::endl;
    }

    return grad_input;
}