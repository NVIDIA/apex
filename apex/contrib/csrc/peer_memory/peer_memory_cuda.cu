#include <torch/extension.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAContext.h>
#include <list>
#include <cstdio>
#include <cassert>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if( err != cudaSuccess ) {                        \
    char hostname[1024];                            \
    gethostname(hostname, 1024);                    \
    printf("%s: CUDA failure %s:%d '%s'\n",         \
         hostname,                                  \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
  }                                                 \
} while(0)

namespace {

/* Basic deleter function for from_blob function.
void deleter(void* ptr)
{
    printf("deleter(ptr=%p)\n",ptr);
    cudaFree(ptr);
}
*/

template<class T>
at::Tensor blob_view(T* raw_ptr, std::vector<int64_t> shape, const at::TensorOptions& options)
{
    std::vector<int64_t> strides(shape.size());
    size_t size = 1;
    int idx = strides.size();
    for (auto it = shape.rbegin();  it != shape.rend();  ++it)
    {
	strides[--idx] = size;
	size *= *it;
    }
    size *= sizeof(T);
    // TODO: Implement dynamic reuse of pooled peer memory.
    // We provide no deleter function because all peer memory allocations are static in this implementation.
    return torch::from_blob((void*)raw_ptr, shape, strides, 0L, options);
}

void tensor_shape(at::Tensor t, bool explicit_nhwc, int& N, int& C, int& H, int& W)
{
    if (t.dim() == 3) {
	N = 1;
        if (explicit_nhwc) {
            C = t.size(2);
            H = t.size(0);
            W = t.size(1);
        } else {
	    C = t.size(0);
    	    H = t.size(1);
    	    W = t.size(2);
        }
    } else if (t.dim() == 4) {
        if (explicit_nhwc) {
            N = t.size(0);
            C = t.size(3);
            H = t.size(1);
            W = t.size(2);
        } else {
            N = t.size(0);
            C = t.size(1);
            H = t.size(2);
            W = t.size(3);
        }
    } else {
        printf("%s;%d - t.dim() must be either 3 or 4 (was %d)\n",__FILE__,__LINE__,t.dim());
        assert(t.dim() == 3 || t.dim() == 4);
    }
}

void tensor_strides(at::Tensor t, bool explicit_nhwc, int& stride_N, int& stride_C, int& stride_H, int& stride_W)
{
    if (t.dim() == 3) {
        if (explicit_nhwc) {
            stride_C = t.stride(2);
            stride_H = t.stride(0);
            stride_W = t.stride(1);
        } else {
	    stride_C = t.stride(0);
    	    stride_H = t.stride(1);
    	    stride_W = t.stride(2);
        }
        stride_N = t.size(0)*t.size(1)*t.size(2);
    } else if (t.dim() == 4) {
        if (explicit_nhwc) {
            stride_N = t.stride(0);
            stride_C = t.stride(3);
            stride_H = t.stride(1);
            stride_W = t.stride(2);
        } else {
            stride_N = t.stride(0);
            stride_C = t.stride(1);
            stride_H = t.stride(2);
            stride_W = t.stride(3);
        }
    } else {
        printf("%s;%d - t.dim() must be either 3 or 4 (was %d)\n",__FILE__,__LINE__,t.dim());
        assert(t.dim() == 3 || t.dim() == 4);
    }
}

template<class T, bool is_HWC>
__device__ void strided_copy_kernel(
	T* dst, const int dst_stride_C, const int dst_stride_H, const int dst_stride_W, 
	const T* src, const int src_stride_C, const int src_stride_H, const int src_stride_W, 
	const int NC, const int NH, const int NW
	)
{
    size_t tot_num_threads = gridDim.x * blockDim.x;
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = NC*NH*NW;
    for (size_t i = thread_id;  i < count;  i += tot_num_threads)
    {
	size_t c,h,w;
	if (is_HWC) {
	    c = i % NC;
	    w = i / NC;
	    h = w / NW;
	    w = w % NW;
	}
	else {
	    w = i % NW;
	    h = i / NW;
	    c = h / NH;
            h = h % NH;
	}
	size_t dst_off = c*dst_stride_C + h*dst_stride_H + w*dst_stride_W;
	size_t src_off = c*src_stride_C + h*src_stride_H + w*src_stride_W;
	dst[dst_off] = src[src_off];
    }
}

__device__ void dual_signal_wait_clear(
	volatile int* signal1_flag, volatile int* wait1_flag,
	volatile int* signal2_flag, volatile int* wait2_flag,
	const int v1, const int v2, const int v3, const int v4,
	const bool clear
	)
{
    register int r1, r2, r3, r4, r5, r6, r7, r8;
    bool is_main_thread = (blockIdx.x == 0 && threadIdx.x == 0) ? true : false;
    // signal and wait
    if (is_main_thread) {
	asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(signal1_flag), "r"(v1), "r"(v2), "r"(v3), "r"(v4) : "memory");
	asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(signal2_flag), "r"(v1), "r"(v2), "r"(v3), "r"(v4) : "memory");
	do {
	    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4) : "l"(wait1_flag) : "memory");
	    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r5), "=r"(r6), "=r"(r7), "=r"(r8) : "l"(wait2_flag) : "memory");
	} while (r1 != v1 || r5 != v1 || r2 != v2 || r6 != v2 || r3 != v3 || r7 != v3 || r4 != v4 || r8 != v4);
    }
    cg::this_grid().sync();
    // optionally clear wait flag
    if (clear && is_main_thread) {
	r1 = 0;  r2 = 0;  r3 = 0;  r4 = 0;
	asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(wait1_flag), "r"(r1), "r"(r2), "r"(r3), "r"(r4) : "memory");
	asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(wait2_flag), "r"(r1), "r"(r2), "r"(r3), "r"(r4) : "memory");
    }
}

template<class T, bool is_HWC>
#if __CUDA_ARCH__ >= 700
__launch_bounds__(128, 16)
#endif
__global__ void push_pull_halos_1d_kernel(
        // top halo,
        const T* toh, int toh_stride_C, int toh_stride_H, int toh_stride_W,     // top output halo
        T* tox, int tox_stride_C, int tox_stride_H, int tox_stride_W,           // top tx buffer
        T* tih, int tih_stride_C, int tih_stride_H, int tih_stride_W,           // top input halo
        // btm halo
        const T* boh, int boh_stride_C, int boh_stride_H, int boh_stride_W,     // top output halo
        T* box, int box_stride_C, int box_stride_H, int box_stride_W,           // top tx buffer
        T* bih, int bih_stride_C, int bih_stride_H, int bih_stride_W,           // top input halo
        // dimensions
        int NC, int NH, int NW,
        // signals
        int* signal1_flag,
        int* signal2_flag,
        int* wait1_flag,
        int* wait2_flag
        )
{
    // push top output halo to transfer buffer
    strided_copy_kernel<T,is_HWC>(tox, tox_stride_C, tox_stride_H, tox_stride_W, toh, toh_stride_C, toh_stride_H, toh_stride_W, NC, NH, NW);
    // push btm output halo to transfer buffer
    strided_copy_kernel<T,is_HWC>(box, box_stride_C, box_stride_H, box_stride_W, boh, boh_stride_C, boh_stride_H, boh_stride_W, NC, NH, NW);
    // signal to top and btm neigbhbors that output halos are ready to be read
    // the choice of values for v1-v4 is arbitrary and does not matter, as long as all ranks use the same values
    dual_signal_wait_clear(signal1_flag, wait1_flag, signal2_flag, wait2_flag, -987751720, 840868300, -225529332, 281513358, true);
    // pull top halo from transfer buffer in peer memory to input
    strided_copy_kernel<T,is_HWC>(tox, tox_stride_C, tox_stride_H, tox_stride_W, tih, tih_stride_C, tih_stride_H, tih_stride_W, NC, NH, NW);
    // pull btm halo from transfer buffer in peer memory to input
    strided_copy_kernel<T,is_HWC>(box, box_stride_C, box_stride_H, box_stride_W, bih, bih_stride_C, bih_stride_H, bih_stride_W, NC, NH, NW);
}

}

namespace apex { namespace peer_memory {

int64_t allocate_raw(int64_t size)
{
    float* ptr = 0L;
    cudaMalloc(&ptr, size);
    return (int64_t)ptr;
}

void free_raw(int64_t raw)
{
    cudaFree((void*)raw);
}

at::Tensor get_raw_ipc_address(int64_t raw)
{
    cudaIpcMemHandle_t mem_handle;
    CUDACHECK( cudaIpcGetMemHandle(&mem_handle, (void*)raw) );
    const int n = sizeof(cudaIpcMemHandle_t);
    auto address_tensor = torch::empty({n}, torch::dtype(torch::kUInt8));
    auto address_tensor_p = address_tensor.data_ptr<uint8_t>();
    memcpy(address_tensor_p, (uint8_t*)&mem_handle, n);
    return address_tensor;
}

std::vector<int64_t> get_raw_peers(at::Tensor ipc_addresses, int peer_rank, int64_t raw)
{
    int peer_group_size = ipc_addresses.size(0);
    std::vector<int64_t> results(peer_group_size);
    for (int i = 0;  i < peer_group_size;  ++i) {
        if (i != peer_rank) {
            cudaIpcMemHandle_t mem_handle;
            memcpy(&mem_handle, ipc_addresses.index({i}).data_ptr<uint8_t>(), sizeof(cudaIpcMemHandle_t));
            void* p = 0L;
            CUDACHECK( cudaIpcOpenMemHandle((void**)&p, mem_handle, cudaIpcMemLazyEnablePeerAccess) );
            results[i] = (int64_t)p;
        } else {
            results[i] = (int64_t)raw;
        }
    }
    return results;
}

at::Tensor blob_view_half(int64_t raw, std::vector<int64_t> shape)
{
    return blob_view<at::Half>((at::Half*)raw, shape, torch::dtype(torch::kFloat16).device(torch::kCUDA));
}

at::Tensor blob_view_float(int64_t raw, std::vector<int64_t> shape)
{
    return blob_view<float>((float*)raw, shape, torch::dtype(torch::kFloat16).device(torch::kCUDA));
}

at::Tensor blob_view_int(int64_t raw, std::vector<int64_t> shape)
{
    return blob_view<int>((int*)raw, shape, torch::dtype(torch::kFloat16).device(torch::kCUDA));
}

void push_pull_halos_1d(
        bool explicit_nhwc,
        int numSM,                      // number of SMs to use
        at::Tensor top_out_halo,        // top output halo in sender device memory
        at::Tensor top_out_tx,          // top output transfer buffer in sender peer pool memory
        at::Tensor top_inp_halo,        // top input halo in receiver device memory
        at::Tensor btm_out_halo,        // btm output halo in sender device memory
        at::Tensor btm_out_tx,          // btm output transfer buffer in sender peer pool memory
        at::Tensor btm_inp_halo,        // btm input halo in receiver device memory
        at::Tensor top_signal,          // top input signal in receiver device memory
        at::Tensor btm_signal,          // btm input signal in receiver device memory
        at::Tensor waits                // top and btm signals for this rank
        )
{
    // basic checks of inputs
    TORCH_CHECK(top_out_halo.is_cuda());
    TORCH_CHECK(top_out_tx.is_cuda());
    TORCH_CHECK(top_inp_halo.is_cuda());
    TORCH_CHECK(btm_out_halo.is_cuda());
    TORCH_CHECK(btm_out_tx.is_cuda());
    TORCH_CHECK(btm_inp_halo.is_cuda());
    TORCH_CHECK(top_signal.is_cuda());
    TORCH_CHECK(btm_signal.is_cuda());
    TORCH_CHECK(waits.is_cuda());

    // shapes and strides
    int toh_N, toh_C, toh_H, toh_W;
    tensor_shape(top_out_halo, explicit_nhwc, toh_N, toh_C, toh_H, toh_W);
    int tox_N, tox_C, tox_H, tox_W;
    tensor_shape(top_out_tx, explicit_nhwc, tox_N, tox_C, tox_H, tox_W);
    int tih_N, tih_C, tih_H, tih_W;
    tensor_shape(top_inp_halo, explicit_nhwc, tih_N, tih_C, tih_H, tih_W);
    TORCH_CHECK(
            (toh_N == tox_N && tox_N == tih_N) &&
            (toh_C == tox_C && tox_C == tih_C) &&
            (toh_H == tox_H && tox_H == tih_H) &&
            (toh_W == tox_W && tox_W == tih_W));
    int boh_N, boh_C, boh_H, boh_W;
    tensor_shape(btm_out_halo, explicit_nhwc, boh_N, boh_C, boh_H, boh_W);
    int box_N, box_C, box_H, box_W;
    tensor_shape(btm_out_tx, explicit_nhwc, box_N, box_C, box_H, box_W);
    int bih_N, bih_C, bih_H, bih_W;
    tensor_shape(btm_inp_halo, explicit_nhwc, bih_N, bih_C, bih_H, bih_W);
    TORCH_CHECK(
            (boh_N == box_N && box_N == bih_N) &&
            (boh_C == box_C && box_C == bih_C) &&
            (boh_H == box_H && box_H == bih_H) &&
            (boh_W == box_W && box_W == bih_W));
    TORCH_CHECK(
	    (toh_N == boh_N) &&
	    (toh_C == boh_C) &&
	    (toh_H == boh_H) &&
	    (toh_W == boh_W));
    int NC=toh_C, NH=toh_H, NW=toh_W;

    int toh_stride_N, toh_stride_C, toh_stride_H, toh_stride_W;
    tensor_strides(top_out_halo, explicit_nhwc, toh_stride_N, toh_stride_C, toh_stride_H, toh_stride_W);
    int tox_stride_N, tox_stride_C, tox_stride_H, tox_stride_W;
    tensor_strides(top_out_tx, explicit_nhwc, tox_stride_N, tox_stride_C, tox_stride_H, tox_stride_W);
    int tih_stride_N, tih_stride_C, tih_stride_H, tih_stride_W;
    tensor_strides(top_inp_halo, explicit_nhwc, tih_stride_N, tih_stride_C, tih_stride_H, tih_stride_W);
    int boh_stride_N, boh_stride_C, boh_stride_H, boh_stride_W;
    tensor_strides(btm_out_halo, explicit_nhwc, boh_stride_N, boh_stride_C, boh_stride_H, boh_stride_W);
    int box_stride_N, box_stride_C, box_stride_H, box_stride_W;
    tensor_strides(btm_out_tx, explicit_nhwc, box_stride_N, box_stride_C, box_stride_H, box_stride_W);
    int bih_stride_N, bih_stride_C, bih_stride_H, bih_stride_W;
    tensor_strides(btm_inp_halo, explicit_nhwc, bih_stride_N, bih_stride_C, bih_stride_H, bih_stride_W);

    // determine if nhwc
    auto is_nhwc = (toh_stride_C == 1) ? true : false;

    // figure out launch parameters
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    assert(numSM > 0 && numSM <= prop.multiProcessorCount);
    auto current_stream = at::cuda::getCurrentCUDAStream();
    const int numThreads = 128;
    dim3 block(numThreads,1,1);
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, top_out_halo.scalar_type(), "push_pull_halos_1d_kernel", [&]{
            scalar_t* toh_p = top_out_halo.data_ptr<scalar_t>();
            scalar_t* tox_p = top_out_tx.data_ptr<scalar_t>();
            scalar_t* tih_p = top_inp_halo.data_ptr<scalar_t>();
            scalar_t* boh_p = btm_out_halo.data_ptr<scalar_t>();
            scalar_t* box_p = btm_out_tx.data_ptr<scalar_t>();
            scalar_t* bih_p = btm_inp_halo.data_ptr<scalar_t>();
	    int* top_signal_p = top_signal.data_ptr<int>();
	    int* btm_signal_p = btm_signal.data_ptr<int>() + 4;
	    int* top_wait_p = waits.data_ptr<int>();
	    int* btm_wait_p = waits.data_ptr<int>() + 4;

            // do int4 vector loads if channel count permits
            int elem_size_in_bytes = toh_C * sizeof(scalar_t);
            int elem_size_in_int4 = (elem_size_in_bytes / 16);
            if (is_nhwc && elem_size_in_int4*16 == elem_size_in_bytes) {
                // can do int4 transfers
	        int divisor = elem_size_in_bytes / elem_size_in_int4;
		toh_stride_N /= divisor;   toh_stride_H /= divisor;    toh_stride_W /= divisor;
		tox_stride_N /= divisor;   tox_stride_H /= divisor;    tox_stride_W /= divisor;
		tih_stride_N /= divisor;   tih_stride_H /= divisor;    tih_stride_W /= divisor;
		boh_stride_N /= divisor;   boh_stride_H /= divisor;    boh_stride_W /= divisor;
		box_stride_N /= divisor;   box_stride_H /= divisor;    box_stride_W /= divisor;
		bih_stride_N /= divisor;   bih_stride_H /= divisor;    bih_stride_W /= divisor;
		void *kernelArgs[] = {
		    (int4**)&toh_p, &toh_stride_C, &toh_stride_H, &toh_stride_W,
		    (int4**)&tox_p, &tox_stride_C, &tox_stride_H, &tox_stride_W,
		    (int4**)&tih_p, &tih_stride_C, &tih_stride_H, &tih_stride_W,
		    (int4**)&boh_p, &boh_stride_C, &boh_stride_H, &boh_stride_W,
		    (int4**)&box_p, &box_stride_C, &box_stride_H, &box_stride_W,
		    (int4**)&bih_p, &bih_stride_C, &bih_stride_H, &bih_stride_W,
		    &NC, &NH, &NW,
		    &top_signal_p, &btm_signal_p, &top_wait_p, &btm_wait_p
		};
            	int numBlocksPerSm;
	        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, push_pull_halos_1d_kernel<int4,true>, numThreads, 0);
	        dim3 grid(numSM*numBlocksPerSm,1,1);
	        cudaLaunchCooperativeKernel((void*)push_pull_halos_1d_kernel<int4,true>, grid, block, kernelArgs, 0, current_stream);
            } else {
                // cannot do int4 transfers
		void *kernelArgs[] = {
		    &toh_p, &toh_stride_C, &toh_stride_H, &toh_stride_W,
		    &tox_p, &tox_stride_C, &tox_stride_H, &tox_stride_W,
		    &tih_p, &tih_stride_C, &tih_stride_H, &tih_stride_W,
		    &boh_p, &boh_stride_C, &boh_stride_H, &boh_stride_W,
		    &box_p, &box_stride_C, &box_stride_H, &box_stride_W,
		    &bih_p, &bih_stride_C, &bih_stride_H, &bih_stride_W,
		    &NC, &NH, &NW,
		    &top_signal_p, &btm_signal_p, &top_wait_p, &btm_wait_p
		};
                int numBlocksPerSm;
                if (is_nhwc) {
	            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, push_pull_halos_1d_kernel<scalar_t,true>, numThreads, 0);
	            dim3 grid(numSM*numBlocksPerSm,1,1);
	            cudaLaunchCooperativeKernel((void*)push_pull_halos_1d_kernel<scalar_t,true>, grid, block, kernelArgs, 0, current_stream);
                } else {
	            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, push_pull_halos_1d_kernel<scalar_t,false>, numThreads, 0);
	            dim3 grid(numSM*numBlocksPerSm,1,1);
	            cudaLaunchCooperativeKernel((void*)push_pull_halos_1d_kernel<scalar_t,false>, grid, block, kernelArgs, 0, current_stream);
                }
	    }
        } );
}

} }

