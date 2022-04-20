#include <torch/extension.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAContext.h>
#include <list>
#include <cstdio>
#include <cassert>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>
#include "nccl.h"
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
at::Tensor blob_view(T* raw_ptr, std::vector<int64_t> shape, const at::TensorOptions& options, bool channels_last)
{
    size_t size = 1;
    std::vector<int64_t> strides(shape.size());
    if (channels_last) {
        assert(shape.size() == 4);
        strides[0] = shape[1]*shape[2]*shape[3];
        strides[1] = 1;
        strides[2] = shape[1]*shape[3];
        strides[3] = shape[1];
    } else {
        int idx = strides.size();
        for (auto it = shape.rbegin();  it != shape.rend();  ++it)
        {
	    strides[--idx] = size;
	    size *= *it;
        }
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

__device__ void checked_signal(
	volatile int* signal1_flag, volatile int* signal2_flag,
	const int v1, const int v2, const int v3, const int v4
	)
{
    cg::this_grid().sync();
    bool is_main_thread = (blockIdx.x == 0 && threadIdx.x == 0) ? true : false;
    if (is_main_thread) {
	// flush all writes to global memory
	__threadfence_system();
	// wait for top or bottom neighbor to clear signal
	register int r1, r2, r3, r4;
	bool top_zeroed=false, btm_zeroed=false, top_done=false, btm_done=false;
	do {
	    do {
		if (!top_zeroed) {
		    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4) : "l"(signal1_flag) : "memory");
		    if (r1 != v1 || r2 != v2 || r3 != v3 || r4 != v4) top_zeroed = true;
		}
		if (!btm_zeroed) {
		    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4) : "l"(signal2_flag) : "memory");
		    if (r1 != v1 || r2 != v2 || r3 != v3 || r4 != v4) btm_zeroed = true;
		}
	    } while((top_zeroed == top_done) && (btm_zeroed == btm_done));
	    if (!top_done && top_zeroed) {
		// signal to top neighbor my output is ready
		asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(signal1_flag), "r"(v1), "r"(v2), "r"(v3), "r"(v4) : "memory");
		top_done = true;
	    }
	    if (!btm_done && btm_zeroed) {
		// signal to bottom neighbor my output is ready
		asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(signal2_flag), "r"(v1), "r"(v2), "r"(v3), "r"(v4) : "memory");
		btm_done = true;
	    }
	} while (!top_done || !btm_done);
    }
}

__device__ void wait_for(
	volatile int* wait_flag,
	const int v1, const int v2, const int v3, const int v4
	)
{
    bool is_main_thread = (blockIdx.x == 0 && threadIdx.x == 0) ? true : false;
    if (is_main_thread) {
    	register int r1, r2, r3, r4;
	// wait for senders to signal their output is read
	do {
	    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4) : "l"(wait_flag) : "memory");
	} while (r1 != v1 || r2 != v2 || r3 != v3 || r4 != v4);
    }
    cg::this_grid().sync();  // all threads wait for main
}


__device__ void clear_flag(
	volatile int* wait_flag
	)
{
    cg::this_grid().sync();  // wait for all threads in kernel to finish
    bool is_main_thread = (blockIdx.x == 0 && threadIdx.x == 0) ? true : false;
    if (is_main_thread) {
	register int r1, r2, r3, r4;
	r1 = 0;  r2 = 0;  r3 = 0;  r4 = 0;
	asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(wait_flag), "r"(r1), "r"(r2), "r"(r3), "r"(r4) : "memory");
    }
}

template<class T, bool is_HWC>
#if __CUDA_ARCH__ >= 700
__launch_bounds__(128, 16)
#endif
__global__ void push_pull_halos_1d_kernel(
        // top halo,
        const T* toh, int toh_stride_C, int toh_stride_H, int toh_stride_W,     // top output halo
        T* tox, int tox_stride_C, int tox_stride_H, int tox_stride_W,           // top output tx buffer
        T* tix, int tix_stride_C, int tix_stride_H, int tix_stride_W,           // top input tx buffer
        T* tih, int tih_stride_C, int tih_stride_H, int tih_stride_W,           // top input halo
        // btm halo
        const T* boh, int boh_stride_C, int boh_stride_H, int boh_stride_W,     // btm output halo
        T* box, int box_stride_C, int box_stride_H, int box_stride_W,           // btm output tx buffer
        T* bix, int bix_stride_C, int bix_stride_H, int bix_stride_W,           // btm input tx buffer
        T* bih, int bih_stride_C, int bih_stride_H, int bih_stride_W,           // btm input halo
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
    checked_signal(signal1_flag, signal2_flag, -987751720, 840868300, -225529332, 281513358);
    // pull top halo from transfer buffer in peer memory to input
    wait_for(wait1_flag, -987751720, 840868300, -225529332, 281513358);
    strided_copy_kernel<T,is_HWC>(tih, tih_stride_C, tih_stride_H, tih_stride_W, tix, tix_stride_C, tix_stride_H, tix_stride_W, NC, NH, NW);
    clear_flag(wait1_flag);
    // pull btm halo from transfer buffer in peer memory to input
    wait_for(wait2_flag, -987751720, 840868300, -225529332, 281513358);
    strided_copy_kernel<T,is_HWC>(bih, bih_stride_C, bih_stride_H, bih_stride_W, bix, bix_stride_C, bix_stride_H, bix_stride_W, NC, NH, NW);
    clear_flag(wait2_flag);
}

__global__ void delay_kernel(int delay_nanoseconds, int* counter)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // waste time while doing something compiler can't predict, thus preventing it from optimizing away this code.
        int new_counter = 0;
        double elapsed = 0;
        clock_t start = clock();
        do {
            clock_t now = clock();
            elapsed = (double)(now - start)*1e9 / CLOCKS_PER_SEC;
            ++new_counter;
        } while (elapsed < (double)delay_nanoseconds);
        *counter = new_counter;
    }
}

}

namespace apex { namespace contrib { namespace peer_memory {

int64_t allocate_raw(int64_t size)
{
    float* ptr = 0L;
    cudaMalloc(&ptr, size);
    cudaMemset(ptr, 0, size);
    return (int64_t)ptr;
}

void free_raw(int64_t raw)
{
    cudaFree((void*)raw);
}

void zero(int64_t raw, int64_t size)
{
    cudaMemset((void*)raw, 0, size);
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

at::Tensor blob_view_half(int64_t raw, std::vector<int64_t> shape, bool channels_last)
{
    return blob_view<at::Half>((at::Half*)raw, shape, torch::dtype(torch::kFloat16).device(torch::kCUDA), channels_last);
}

at::Tensor blob_view_float(int64_t raw, std::vector<int64_t> shape, bool channels_last)
{
    return blob_view<float>((float*)raw, shape, torch::dtype(torch::kFloat32).device(torch::kCUDA), channels_last);
}

at::Tensor blob_view_int(int64_t raw, std::vector<int64_t> shape, bool channels_last)
{
    return blob_view<int>((int*)raw, shape, torch::dtype(torch::kInt32).device(torch::kCUDA), channels_last);
}

void push_pull_halos_1d(
	bool diagnostics,
        bool explicit_nhwc,
        int numSM,                      // number of SMs to use
        at::Tensor top_out_halo,        // top output halo in sender device memory
        at::Tensor top_out_tx,          // top output transfer buffer in sender peer pool memory
	at::Tensor top_inp_tx,		// top input transfer buffer in top neighbor peer pool memory
        at::Tensor top_inp_halo,        // top input halo in receiver device memory
        at::Tensor btm_out_halo,        // btm output halo in sender device memory
        at::Tensor btm_out_tx,          // btm output transfer buffer in sender peer pool memory
	at::Tensor btm_inp_tx,		// btm input transfer buffer in btm neighbor peer pool memory
        at::Tensor btm_inp_halo,        // btm input halo in receiver device memory
        at::Tensor top_signal,          // top input signal in receiver device memory
        at::Tensor btm_signal,          // btm input signal in receiver device memory
        at::Tensor waits                // top and btm signals for this rank
        )
{
    // basic checks of inputs
    TORCH_CHECK(top_out_halo.is_cuda());
    TORCH_CHECK(top_out_tx.is_cuda());
    TORCH_CHECK(top_inp_tx.is_cuda());
    TORCH_CHECK(top_inp_halo.is_cuda());
    TORCH_CHECK(btm_out_halo.is_cuda());
    TORCH_CHECK(btm_out_tx.is_cuda());
    TORCH_CHECK(btm_inp_tx.is_cuda());
    TORCH_CHECK(btm_inp_halo.is_cuda());
    TORCH_CHECK(top_signal.is_cuda());
    TORCH_CHECK(btm_signal.is_cuda());
    TORCH_CHECK(waits.is_cuda());

    // shapes and strides
    int toh_N, toh_C, toh_H, toh_W;
    tensor_shape(top_out_halo, explicit_nhwc, toh_N, toh_C, toh_H, toh_W);
    int tox_N, tox_C, tox_H, tox_W;
    tensor_shape(top_out_tx, explicit_nhwc, tox_N, tox_C, tox_H, tox_W);
    int tix_N, tix_C, tix_H, tix_W;
    tensor_shape(top_inp_tx, explicit_nhwc, tix_N, tix_C, tix_H, tix_W);
    int tih_N, tih_C, tih_H, tih_W;
    tensor_shape(top_inp_halo, explicit_nhwc, tih_N, tih_C, tih_H, tih_W);
    TORCH_CHECK(
            (toh_N == tox_N && tox_N == tix_N && tix_N == tih_N) &&
            (toh_C == tox_C && tox_C == tix_C && tix_C == tih_C) &&
            (toh_H == tox_H && tox_H == tix_H && tix_H == tih_H) &&
            (toh_W == tox_W && tox_W == tix_W && tix_W == tih_W));
    int boh_N, boh_C, boh_H, boh_W;
    tensor_shape(btm_out_halo, explicit_nhwc, boh_N, boh_C, boh_H, boh_W);
    int box_N, box_C, box_H, box_W;
    tensor_shape(btm_out_tx, explicit_nhwc, box_N, box_C, box_H, box_W);
    int bix_N, bix_C, bix_H, bix_W;
    tensor_shape(btm_inp_tx, explicit_nhwc, bix_N, bix_C, bix_H, bix_W);
    int bih_N, bih_C, bih_H, bih_W;
    tensor_shape(btm_inp_halo, explicit_nhwc, bih_N, bih_C, bih_H, bih_W);
    TORCH_CHECK(
            (boh_N == box_N && box_N == bix_N && bix_N == bih_N) &&
            (boh_C == box_C && box_C == bix_C && bix_C == bih_C) &&
            (boh_H == box_H && box_H == bix_H && bix_H == bih_H) &&
            (boh_W == box_W && box_W == bix_W && bix_W == bih_W));
    TORCH_CHECK(
	    (toh_N == boh_N) &&
	    (toh_C == boh_C) &&
	    (toh_H == boh_H) &&
	    (toh_W == boh_W));
    int NC=toh_C, NH=toh_H, NW=toh_W;
    if (diagnostics) printf("NC=%d, NH=%d, NW=%d\n",NC,NH,NW);

    int toh_stride_N, toh_stride_C, toh_stride_H, toh_stride_W;
    tensor_strides(top_out_halo, explicit_nhwc, toh_stride_N, toh_stride_C, toh_stride_H, toh_stride_W);
    int tox_stride_N, tox_stride_C, tox_stride_H, tox_stride_W;
    tensor_strides(top_out_tx, explicit_nhwc, tox_stride_N, tox_stride_C, tox_stride_H, tox_stride_W);
    int tix_stride_N, tix_stride_C, tix_stride_H, tix_stride_W;
    tensor_strides(top_inp_tx, explicit_nhwc, tix_stride_N, tix_stride_C, tix_stride_H, tix_stride_W);
    int tih_stride_N, tih_stride_C, tih_stride_H, tih_stride_W;
    tensor_strides(top_inp_halo, explicit_nhwc, tih_stride_N, tih_stride_C, tih_stride_H, tih_stride_W);
    int boh_stride_N, boh_stride_C, boh_stride_H, boh_stride_W;
    tensor_strides(btm_out_halo, explicit_nhwc, boh_stride_N, boh_stride_C, boh_stride_H, boh_stride_W);
    int box_stride_N, box_stride_C, box_stride_H, box_stride_W;
    tensor_strides(btm_out_tx, explicit_nhwc, box_stride_N, box_stride_C, box_stride_H, box_stride_W);
    int bix_stride_N, bix_stride_C, bix_stride_H, bix_stride_W;
    tensor_strides(btm_inp_tx, explicit_nhwc, bix_stride_N, bix_stride_C, bix_stride_H, bix_stride_W);
    int bih_stride_N, bih_stride_C, bih_stride_H, bih_stride_W;
    tensor_strides(btm_inp_halo, explicit_nhwc, bih_stride_N, bih_stride_C, bih_stride_H, bih_stride_W);

    // determine if nhwc
    auto is_nhwc = (toh_stride_C == 1) ? true : false;
    if (diagnostics) printf("is_nhwc = %s\n",is_nhwc?"true":"false");

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
	    if (diagnostics) printf("size(scalar_t) = %ld\n",sizeof(scalar_t));
            scalar_t* toh_p = top_out_halo.data_ptr<scalar_t>();
            scalar_t* tox_p = top_out_tx.data_ptr<scalar_t>();
            scalar_t* tix_p = top_inp_tx.data_ptr<scalar_t>();
            scalar_t* tih_p = top_inp_halo.data_ptr<scalar_t>();
            scalar_t* boh_p = btm_out_halo.data_ptr<scalar_t>();
            scalar_t* box_p = btm_out_tx.data_ptr<scalar_t>();
            scalar_t* bix_p = btm_inp_tx.data_ptr<scalar_t>();
            scalar_t* bih_p = btm_inp_halo.data_ptr<scalar_t>();
	    if (diagnostics) printf("waypoint1\n");
	    int* top_signal_p = top_signal.data_ptr<int>() + 4;
	    int* btm_signal_p = btm_signal.data_ptr<int>();
	    int* top_wait_p = waits.data_ptr<int>();
	    int* btm_wait_p = waits.data_ptr<int>() + 4;
	    if (diagnostics) printf("waypoint2\n");

            // do int4 vector loads if channel count permits
            int elem_size_in_bytes = toh_C * sizeof(scalar_t);
            int elem_size_in_int4 = (elem_size_in_bytes / 16);
	    if (diagnostics) printf("elem_size_in_bytes = %d, elem_size_in_int4 = %d\n",elem_size_in_bytes,elem_size_in_int4);
            if (is_nhwc && elem_size_in_int4*16 == elem_size_in_bytes) {
                // can do int4 transfers
	        int divisor = toh_C / elem_size_in_int4;
		if (diagnostics) printf("CAN DO INT4 :: divisor = %d\n",divisor);
		toh_stride_N /= divisor;   toh_stride_H /= divisor;    toh_stride_W /= divisor;
		tox_stride_N /= divisor;   tox_stride_H /= divisor;    tox_stride_W /= divisor;
		tix_stride_N /= divisor;   tix_stride_H /= divisor;    tix_stride_W /= divisor;
		tih_stride_N /= divisor;   tih_stride_H /= divisor;    tih_stride_W /= divisor;
		boh_stride_N /= divisor;   boh_stride_H /= divisor;    boh_stride_W /= divisor;
		box_stride_N /= divisor;   box_stride_H /= divisor;    box_stride_W /= divisor;
		bix_stride_N /= divisor;   bix_stride_H /= divisor;    bix_stride_W /= divisor;
		bih_stride_N /= divisor;   bih_stride_H /= divisor;    bih_stride_W /= divisor;
		NC /= divisor;
		if (diagnostics) {
                    printf("divisor=%d\n",divisor);
                    printf("toh_stride :: N=%d, C=%d, H=%d, W=%d\n",toh_stride_N,toh_stride_C,toh_stride_H,toh_stride_W);
                    printf("tox_stride :: N=%d, C=%d, H=%d, W=%d\n",tox_stride_N,tox_stride_C,tox_stride_H,tox_stride_W);
                    printf("tix_stride :: N=%d, C=%d, H=%d, W=%d\n",tix_stride_N,tix_stride_C,tix_stride_H,tix_stride_W);
                    printf("tih_stride :: N=%d, C=%d, H=%d, W=%d\n",tih_stride_N,tih_stride_C,tih_stride_H,tih_stride_W);
                    printf("boh_stride :: N=%d, C=%d, H=%d, W=%d\n",boh_stride_N,boh_stride_C,boh_stride_H,boh_stride_W);
                    printf("box_stride :: N=%d, C=%d, H=%d, W=%d\n",box_stride_N,box_stride_C,box_stride_H,box_stride_W);
                    printf("bix_stride :: N=%d, C=%d, H=%d, W=%d\n",bix_stride_N,bix_stride_C,bix_stride_H,bix_stride_W);
                    printf("bih_stride :: N=%d, C=%d, H=%d, W=%d\n",bih_stride_N,bih_stride_C,bih_stride_H,bih_stride_W);
                    printf("NC=%d, NH=%d, NW=%d\n",NC,NH,NW);
                }
		void *kernelArgs[] = {
		    (int4**)&toh_p, &toh_stride_C, &toh_stride_H, &toh_stride_W,
		    (int4**)&tox_p, &tox_stride_C, &tox_stride_H, &tox_stride_W,
		    (int4**)&tix_p, &tix_stride_C, &tix_stride_H, &tix_stride_W,
		    (int4**)&tih_p, &tih_stride_C, &tih_stride_H, &tih_stride_W,
		    (int4**)&boh_p, &boh_stride_C, &boh_stride_H, &boh_stride_W,
		    (int4**)&box_p, &box_stride_C, &box_stride_H, &box_stride_W,
		    (int4**)&bix_p, &bix_stride_C, &bix_stride_H, &bix_stride_W,
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
		if (diagnostics) printf("CAN NOT DO INT4\n");
		void *kernelArgs[] = {
		    &toh_p, &toh_stride_C, &toh_stride_H, &toh_stride_W,
		    &tox_p, &tox_stride_C, &tox_stride_H, &tox_stride_W,
		    &tix_p, &tix_stride_C, &tix_stride_H, &tix_stride_W,
		    &tih_p, &tih_stride_C, &tih_stride_H, &tih_stride_W,
		    &boh_p, &boh_stride_C, &boh_stride_H, &boh_stride_W,
		    &box_p, &box_stride_C, &box_stride_H, &box_stride_W,
		    &bix_p, &bix_stride_C, &bix_stride_H, &bix_stride_W,
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

} } }

