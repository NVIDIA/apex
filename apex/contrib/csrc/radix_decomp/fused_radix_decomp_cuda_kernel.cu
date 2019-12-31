#include "ATen/ATen.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/detail/IndexUtils.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cmath>
#include "ATen/TensorUtils.h"
// #include "ATen/Type.h"
#include "ATen/AccumulateType.h"
#include <THC/THCGeneral.h>
#include "multi_tensor_apply.cuh"

#define BLOCK_SIZE 512
#define ILP 4

#include "type_shim.h"

//
// Decompose values in p_orig into p_decomp.
// 
template <typename T, typename GRAD_T>
__global__ void radix_decomp_kernel(
	volatile int* noop_gmem,
	T* __restrict__ p_orig,
	GRAD_T* __restrict__ p_decomp,
	const int largest_digit,
	const int smallest_digit,
        const T radix,
	const size_t tsize,
	const int clear_overflow_first)
{
	const int n = largest_digit - smallest_digit + 1;
	
        //Assuming 2D grids and 2D blocks
        const int blockId = gridDim.x * blockIdx.y + blockIdx.x;
        const int threadsPerBlock = blockDim.x * blockDim.y;
        const int threadIdInBlock = threadIdx.y * blockDim.x + threadIdx.x;
        const int i = (blockId * threadsPerBlock + threadIdInBlock);
        const int totThreads = gridDim.x*gridDim.y*threadsPerBlock;

	if (clear_overflow_first) {
		if (i == 0) {
			*noop_gmem = 0;
		}
		__syncthreads();
	}

	for (int j = i; j < tsize; j+=totThreads) {
		T orig_val = p_orig[j];
		T sign_val = orig_val < static_cast<T>(0) ? static_cast<T>(-1) : static_cast<T>(1);
		T abs_val = orig_val * sign_val;
		GRAD_T* r = p_decomp + j*n;
		// To-Do: Store base as vector of const values.
		// To-Do: Look into reformulating this as a prescan algorithm.
		for (int k = largest_digit;  k >= smallest_digit;  --k) {
			T base = pow(radix, static_cast<T>(k));
			T digit = abs_val/base;
			// we truncate all digits to integers, except for the smallest (magnitude) one.
			if (k != smallest_digit) {
				digit = trunc(digit);
			}
			abs_val -= (digit * base);
			GRAD_T oval = static_cast<GRAD_T>(digit * sign_val);
			r[k-smallest_digit] = oval;
			if (!isfinite(oval)) {
				*noop_gmem = 1;
			}
			//if (i == 0) printf("orig_val=%e, sign_val=%e, abs_val=%e, base=%e, digit=%e, oval=%e\n",orig_val,sign_val,abs_val,base,digit,oval);
		}
	}
}

//
// Compose previosuly decomposed values in p_decomp into p_orig.
//
template <typename T, typename GRAD_T>
__global__ void radix_comp_kernel(
	volatile int* noop_gmem,
	T* __restrict__ p_orig,
	GRAD_T* __restrict__ p_decomp,
	const int largest_digit,
	const int smallest_digit,
        const T radix,
	const size_t tsize,
	const int clear_overflow_first)
{
	const int n = largest_digit - smallest_digit + 1;
	
        //Assuming 2D grids and 2D blocks
        const int blockId = gridDim.x * blockIdx.y + blockIdx.x;
        const int threadsPerBlock = blockDim.x * blockDim.y;
        const int threadIdInBlock = threadIdx.y * blockDim.x + threadIdx.x;
        const int i = (blockId * threadsPerBlock + threadIdInBlock);
        const int totThreads = gridDim.x*gridDim.y*threadsPerBlock;

	if (clear_overflow_first) {
		if (i == 0) {
			*noop_gmem = 0;
		}
		__syncthreads();
	}

	for (int j = i; j < tsize; j+=totThreads) {
		GRAD_T* r = p_decomp + j*n;
		T orig_val = static_cast<T>(0);
		for (int k = largest_digit;  k >= smallest_digit;  --k) {
			T base = pow(radix, static_cast<T>(k));
			T digit = static_cast<T>(r[k-smallest_digit]);
			orig_val += (digit * base);
		}
		p_orig[j] = orig_val;
	}
}

void radix_decomp_cuda(
	at::Tensor& noop,
	at::Tensor& p_orig,
	at::Tensor& p_decomp,
	int largest_digit,
	int smallest_digit,
	double radix,
	int clear_overflow_first)
{
        //Get tensor size
        int tsize = p_orig.numel();
        //Determine #threads and #blocks
        const int threadsPerBlock = 512;
        const dim3 blocks((tsize+threadsPerBlock-1)/threadsPerBlock);
        AT_ASSERTM(at::cuda::detail::canUse32BitIndexMath(p_orig), "parameter tensor is too large to be indexed with int32");
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        DISPATCH_DOUBLE_AND_FLOAT(p_orig.scalar_type(), 0, "radix_decomp_kernel",
	    DISPATCH_DOUBLE_FLOAT_AND_HALF(p_decomp.scalar_type(), 1, "radix_decomp_kernel",
                radix_decomp_kernel<scalar_t_0, scalar_t_1><<<blocks,threadsPerBlock, 0, stream>>>(
		    noop.DATA_PTR<int>(),
                    p_orig.DATA_PTR<scalar_t_0>(),
		    p_decomp.DATA_PTR<scalar_t_1>(),
		    largest_digit,smallest_digit,static_cast<scalar_t_0>(radix),
		    tsize,clear_overflow_first);
		);
	    );

      THCudaCheck(cudaGetLastError());
}

void radix_comp_cuda(
	at::Tensor& noop,
	at::Tensor& p_orig,
	at::Tensor& p_decomp,
	int largest_digit,
	int smallest_digit,
	double radix,
	int clear_overflow_first)
{
        //Get tensor size
        int tsize = p_orig.numel();
        //Determine #threads and #blocks
        const int threadsPerBlock = 512;
        const dim3 blocks((tsize+threadsPerBlock-1)/threadsPerBlock);
        AT_ASSERTM(at::cuda::detail::canUse32BitIndexMath(p_orig), "parameter tensor is too large to be indexed with int32");
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        DISPATCH_DOUBLE_AND_FLOAT(p_orig.scalar_type(), 0, "radix_comp_kernel",
	    DISPATCH_DOUBLE_FLOAT_AND_HALF(p_decomp.scalar_type(), 1, "radix_comp_kernel",
                radix_comp_kernel<scalar_t_0, scalar_t_1><<<blocks,threadsPerBlock, 0, stream>>>(
		    noop.DATA_PTR<int>(),
                    p_orig.DATA_PTR<scalar_t_0>(),
		    p_decomp.DATA_PTR<scalar_t_1>(),
		    largest_digit,smallest_digit,static_cast<scalar_t_0>(radix),
		    tsize,clear_overflow_first);
		);
	    );

      THCudaCheck(cudaGetLastError());
}

