#include <torch/extension.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAContext.h>
#include <list>
#include <cstdio>
#include <cassert>
#include <cuda_runtime_api.h>
#include "nccl.h"

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

constexpr int THREADS_PER_CTA = 128;

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
        printf("%s;%d - t.dim() must be either 3 or 4 (was %d)\n",__FILE__,__LINE__,int(t.dim()));
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

template<class T>
inline __device__ void __zero(T* dst)
{
    *dst = T(0);
}

inline __device__ void __zero(int2* dst)
{
    *dst = {0, 0};
}

template<class T, bool contiguous>
inline __device__ void zero_tensor(
	const int dim0,
        const int dim1,
        const int dim2,
        T* __restrict__ data,
        const int data_stride0,
        const int data_stride1,
        const int data_stride2,
        const int thread_id,
        const int block_id,
        const int num_blocks
        )
{
    const int global_id = thread_id + block_id * THREADS_PER_CTA;
    const int num_threads = num_blocks * THREADS_PER_CTA;
    const int count = dim0 * dim1 * dim2;
    for (int i = global_id; i < count; i += num_threads) {
        int offset;
        if (contiguous) {
            offset = i;
        } else {
            const int j2 = i % dim2;
            const int k = i / dim2;
            const int j1 = k % dim1;
            const int j0 = k / dim1;
            offset = j0 * data_stride0 + j1 * data_stride1 + j2 * data_stride2;
        }
        __zero(data + offset);
    }
}

template<class T, bool contiguous>
inline __device__ void push_pull_tensor(
	const int dim0,
        const int dim1,
        const int dim2,
        const T* __restrict__ data_in,
        const int data_in_stride0,
        const int data_in_stride1,
        const int data_in_stride2,
	T* __restrict__ data_out,
        const int data_out_stride0,
        const int data_out_stride1,
        const int data_out_stride2,
        int4* local_peer,
        int4* remote_peer,
        const int thread_id,
        const int block_id,
        const int num_blocks
	)
{
    // 128b=16B NVLink flit
    // Note: Use last 4B as a semaphore
    static_assert(sizeof(T) <= 12);
    union Flit {
        T payload;
        uint uints[4];
    };
    // Communication bit indicates whether flit has been received from
    // a remote GPU
    constexpr uint communication_mask = 1 << 0;
    // Status bit is used to choose the active peer buffer in an
    // alternating double buffer scheme. We use buffer 1 if the bits
    // match, use buffer 2 if the bits differ, and invert the bit
    // after finishing with a buffer.
    constexpr uint status_mask = 1 << 1;

    // Split peer memory into two sets of buffers
    // Note: Each block owns a THREADS_PER_CTA*2*16B chunk of peer
    // memory
    const int peer_offset1 = block_id * THREADS_PER_CTA * 2 + thread_id;
    const int peer_offset2 = peer_offset1 + THREADS_PER_CTA;
    volatile int* local_peer1 = reinterpret_cast<volatile int*>(local_peer + peer_offset1);
    volatile int* local_peer2 = reinterpret_cast<volatile int*>(local_peer + peer_offset2);
    volatile int* remote_peer1 = reinterpret_cast<volatile int*>(remote_peer + peer_offset1);
    volatile int* remote_peer2 = reinterpret_cast<volatile int*>(remote_peer + peer_offset2);

    // Iterate through tensor entries
    const int num_threads = num_blocks * THREADS_PER_CTA;
    const int count = dim0 * dim1 * dim2;
    for (int i0 = block_id * THREADS_PER_CTA; i0 < count; i0 += num_threads) {
        const int i = i0 + thread_id;
        const bool has_data = i < count;

        // Calculate buffer positions
        int data_in_offset, data_out_offset;
        if (contiguous) {
            data_in_offset = i;
            data_out_offset = i;
        } else {
            const int j2 = i % dim2;
            const int k = i / dim2;
            const int j1 = k % dim1;
            const int j0 = k / dim1;
            data_in_offset = j0 * data_in_stride0 + j1 * data_in_stride1 + j2 * data_in_stride2;
            data_out_offset = j0 * data_out_stride0 + j1 * data_out_stride1 + j2 * data_out_stride2;
        }

        // Determine which peer memory buffer to use
        // Note: The status bit is not affected by asynchronous
        // communication from the remote GPU.
        Flit local_message1, local_message2;
        asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" :
                     "=r"(local_message1.uints[0]),
                     "=r"(local_message1.uints[1]),
                     "=r"(local_message1.uints[2]),
                     "=r"(local_message1.uints[3])
                     : "l"(local_peer1) : "memory");
        asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" :
                     "=r"(local_message2.uints[0]),
                     "=r"(local_message2.uints[1]),
                     "=r"(local_message2.uints[2]),
                     "=r"(local_message2.uints[3])
                     : "l"(local_peer2) : "memory");
        const uint status1 = local_message1.uints[3] & status_mask;
        const uint status2 = local_message2.uints[3] & status_mask;
        const bool peer1_is_active = (status1 ^ status2) == 0;
        volatile int* ox = peer1_is_active ? remote_peer1 : remote_peer2;
        volatile int* ix = peer1_is_active ? local_peer1 : local_peer2;
        const uint status = peer1_is_active ? status1 : status2;
        Flit recv_message = peer1_is_active ? local_message1 : local_message2;

        // Send flit to remote GPU
        // Note: Set communication bit and keep status bit
        Flit send_message;
        if (has_data) {
            send_message.payload = data_in[data_in_offset];
        }
        send_message.uints[3] = communication_mask | status;
        asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::
                     "l"(ox),
                     "r"(send_message.uints[0]),
                     "r"(send_message.uints[1]),
                     "r"(send_message.uints[2]),
                     "r"(send_message.uints[3])
                     : "memory");

        // Recieve flit from peer
        while ((recv_message.uints[3] & communication_mask) == 0) {
            asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" :
                         "=r"(recv_message.uints[0]),
                         "=r"(recv_message.uints[1]),
                         "=r"(recv_message.uints[2]),
                         "=r"(recv_message.uints[3])
                         : "l"(ix) : "memory");
        }
        if (has_data) {
            data_out[data_out_offset] = recv_message.payload;
        }

        // Reset semaphore
        // Note: Clear communication bit and invert status bit
        uint flag = ~status & status_mask;
        asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::
                     "l"(ix),
                     "n"(0),
                     "n"(0),
                     "n"(0),
                     "r"(flag)
                     : "memory");
        if (i0 + num_threads < count) {
            __threadfence_system();
        }
    }
}

template<class T, bool contiguous, bool top_zero, bool btm_zero>
#if __CUDA_ARCH__ >= 700
__launch_bounds__(THREADS_PER_CTA)
#endif
__global__ void push_pull_halos_1d_kernel(
        // top halo,
        T* toh, int toh_stride0, int toh_stride1, int toh_stride2,              // top output halo (local)
        const T* tih, int tih_stride0, int tih_stride1, int tih_stride2,        // top input halo (local)
        int4* tox,                                                              // top output transfer buffer (remote peer)
        int4* tix,                                                              // top input transfer buffer (local peer)
        // btm halo
        T* boh, int boh_stride0, int boh_stride1, int boh_stride2,              // btm output halo (local)
        const T* bih, int bih_stride0, int bih_stride1, int bih_stride2,        // btm input halo (local)
        int4* box,                                                              // btm output transfer buffer (remote peer)
        int4* bix,                                                              // btm input transfer buffer (local peer)
        // dimensions
        int dim0, int dim1, int dim2,
        bool top_first                                                          // whether to launch communicate top halo first
        )
{
    const int num_blocks_side = gridDim.x / 2;
    const int block_id_side = (blockIdx.x < num_blocks_side
                               ? blockIdx.x
                               : blockIdx.x - num_blocks_side);
    const bool in_top_block = top_first == (blockIdx.x < num_blocks_side);
    if (in_top_block) {
        if (top_zero) {
            zero_tensor<T,contiguous>(
                dim0, dim1, dim2,
                toh, toh_stride0, toh_stride1, toh_stride2,
                threadIdx.x, block_id_side, num_blocks_side);
        } else {
            push_pull_tensor<T,contiguous>(
                dim0, dim1, dim2,
                tih, tih_stride0, tih_stride1, tih_stride2,
                toh, toh_stride0, toh_stride1, toh_stride2,
                tix, tox,
                threadIdx.x, block_id_side, num_blocks_side);
        }
    } else {
        if (btm_zero) {
            zero_tensor<T,contiguous>(
                dim0, dim1, dim2,
                boh, boh_stride0, boh_stride1, boh_stride2,
                threadIdx.x, block_id_side, num_blocks_side);
        } else {
            push_pull_tensor<T,contiguous>(
                dim0, dim1, dim2,
                bih, bih_stride0, bih_stride1, bih_stride2,
                boh, boh_stride0, boh_stride1, boh_stride2,
                bix, box,
                threadIdx.x, block_id_side, num_blocks_side);
        }
    }
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
        int numSM,                      // number of SMs to use (zero corresponds to all SMs)
        int rank,                       // rank in spatial parallel group
	bool top_zero,			// if top halo should be zeroed
        at::Tensor top_in_halo,         // top input halo buffer (in local device memory, sent to top neighbor)
	at::Tensor top_in_transfer,	// top input transfer buffer (in local peer memory)
        at::Tensor top_out_transfer,    // top output transfer buffer (in top neighbor peer memory)
        at::Tensor top_out_halo,        // top output halo buffer (in local device memory, received from top neighbor)
	bool btm_zero,			// if btm halo should be zeroed
        at::Tensor btm_in_halo,         // btm input halo buffer (in local device memory, sent to btm neighbor)
	at::Tensor btm_in_transfer,	// btm input transfer buffer (in local peer memory)
        at::Tensor btm_out_transfer,    // btm output transfer buffer (in btm neighbor peer memory)
        at::Tensor btm_out_halo         // btm output halo buffer (in local device memory, received from btm neighbor)
        )
{
    // basic checks of inputs
    TORCH_CHECK(!(top_zero && btm_zero));
    TORCH_CHECK(top_in_halo.is_cuda());
    TORCH_CHECK(top_out_transfer.is_cuda());
    TORCH_CHECK(top_in_transfer.is_cuda());
    TORCH_CHECK(top_out_halo.is_cuda());
    TORCH_CHECK(btm_in_halo.is_cuda());
    TORCH_CHECK(btm_out_transfer.is_cuda());
    TORCH_CHECK(btm_in_transfer.is_cuda());
    TORCH_CHECK(btm_out_halo.is_cuda());

    // tensor shapes
    int tih_N, tih_C, tih_H, tih_W;
    tensor_shape(top_in_halo, explicit_nhwc, tih_N, tih_C, tih_H, tih_W);
    int toh_N, toh_C, toh_H, toh_W;
    tensor_shape(top_out_halo, explicit_nhwc, toh_N, toh_C, toh_H, toh_W);
    int bih_N, bih_C, bih_H, bih_W;
    tensor_shape(btm_in_halo, explicit_nhwc, bih_N, bih_C, bih_H, bih_W);
    int boh_N, boh_C, boh_H, boh_W;
    tensor_shape(btm_out_halo, explicit_nhwc, boh_N, boh_C, boh_H, boh_W);
    TORCH_CHECK(toh_N == tih_N && tih_N == boh_N && boh_N == bih_N &&
                toh_C == tih_C && tih_C == boh_C && boh_C == bih_C &&
                toh_H == tih_H && tih_H == boh_H && boh_H == bih_H &&
                toh_W == tih_W && tih_W == boh_W && boh_W == bih_W);
    int NN=toh_N, NC=toh_C, NH=toh_H, NW=toh_W;
    if (diagnostics) {
        printf("rank %d: NN=%d, NC=%d, NH=%d, NW=%d\n", rank, NN, NC, NH, NW);
    }
    TORCH_CHECK(NN == 1);

    // tensor strides
    int tih_stride_N, tih_stride_C, tih_stride_H, tih_stride_W;
    tensor_strides(top_in_halo, explicit_nhwc, tih_stride_N, tih_stride_C, tih_stride_H, tih_stride_W);
    int toh_stride_N, toh_stride_C, toh_stride_H, toh_stride_W;
    tensor_strides(top_out_halo, explicit_nhwc, toh_stride_N, toh_stride_C, toh_stride_H, toh_stride_W);
    int bih_stride_N, bih_stride_C, bih_stride_H, bih_stride_W;
    tensor_strides(btm_in_halo, explicit_nhwc, bih_stride_N, bih_stride_C, bih_stride_H, bih_stride_W);
    int boh_stride_N, boh_stride_C, boh_stride_H, boh_stride_W;
    tensor_strides(btm_out_halo, explicit_nhwc, boh_stride_N, boh_stride_C, boh_stride_H, boh_stride_W);
    if (diagnostics) {
        printf("rank %d: tih_stride :: N=%d, C=%d, H=%d, W=%d\n",
               rank, tih_stride_N, tih_stride_C, tih_stride_H, tih_stride_W);
        printf("rank %d: toh_stride :: N=%d, C=%d, H=%d, W=%d\n",
               rank, toh_stride_N, toh_stride_C, toh_stride_H, toh_stride_W);
        printf("rank %d: bih_stride :: N=%d, C=%d, H=%d, W=%d\n",
               rank, bih_stride_N, bih_stride_C, bih_stride_H, bih_stride_W);
        printf("rank %d: boh_stride :: N=%d, C=%d, H=%d, W=%d\n",
               rank, boh_stride_N, boh_stride_C, boh_stride_H, boh_stride_W);
    }

    // determine if nhwc
    bool is_nhwc = (toh_stride_C == 1);
    if (diagnostics) {
        printf("rank %d: is_nhwc = %s\n", rank, is_nhwc ? "true" : "false");
    }

    // determine if contiguous
    bool contiguous = true;
    if ((NN-1)*toh_stride_N + (NC-1)*toh_stride_C +
        (NH-1)*toh_stride_H + (NW-1)*toh_stride_W
        != NN*NC*NH*NW - 1) {
        contiguous = false;
    }
    if ((NN-1)*boh_stride_N + (NC-1)*boh_stride_C +
        (NH-1)*boh_stride_H + (NW-1)*boh_stride_W
        != NN*NC*NH*NW - 1) {
        contiguous = false;
    }
    if (!top_zero) {
        if (toh_stride_N != tih_stride_N || toh_stride_C != tih_stride_C ||
            toh_stride_H != tih_stride_H || toh_stride_W != tih_stride_W) {
            contiguous = false;
        }
    }
    if (!btm_zero) {
        if (boh_stride_N != bih_stride_N || boh_stride_C != bih_stride_C ||
            boh_stride_H != bih_stride_H || boh_stride_W != bih_stride_W) {
            contiguous = false;
        }
    }
    if (diagnostics) {
        printf("rank %d: contiguous = %s\n", rank, contiguous ? "true" : "false");
    }

    // determine whether to communicate top halo first
    bool top_first = rank % 2 != 0;
    if (diagnostics) {
        printf("rank %d: top_first = %s\n", rank, top_first ? "true" : "false");
    }

    // peer memory buffers
    int tox_size = top_out_transfer.numel() * top_out_transfer.element_size();
    int tix_size = top_in_transfer.numel() * top_in_transfer.element_size();
    int box_size = btm_out_transfer.numel() * btm_out_transfer.element_size();
    int bix_size = btm_in_transfer.numel() * btm_in_transfer.element_size();
    if (!top_zero) {
        TORCH_CHECK(top_out_transfer.is_contiguous());
        TORCH_CHECK(top_in_transfer.is_contiguous());
        TORCH_CHECK(tox_size == tix_size);
    }
    if (!btm_zero) {
        TORCH_CHECK(btm_out_transfer.is_contiguous());
        TORCH_CHECK(btm_in_transfer.is_contiguous());
        TORCH_CHECK(box_size == bix_size);
    }

    // figure out launch parameters
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (numSM <= 0 || numSM > prop.multiProcessorCount) {
      numSM = prop.multiProcessorCount;
    }
    auto current_stream = at::cuda::getCurrentCUDAStream();
    dim3 block(THREADS_PER_CTA, 1, 1);

    // helper macros to launch templated kernel
#define LAUNCH_PUSH_PULL_HALO_KERNEL_BASE(T, CONTIGUOUS, TOP_ZERO, BTM_ZERO, KERNEL_ARGS, NUM_ELEMENTS) \
    do {                                                                \
        /* kernel configuration */                                      \
        int numBlocksPerSm;                                             \
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(                  \
            &numBlocksPerSm,                                            \
            push_pull_halos_1d_kernel<T,CONTIGUOUS,TOP_ZERO,BTM_ZERO>,  \
            THREADS_PER_CTA,                                            \
            0);                                                         \
        dim3 grid(numSM*numBlocksPerSm,1,1);                            \
        if (grid.x % 2 != 0) {                                          \
            /* require even number of blocks (half for top, half for bottom) */ \
            grid.x -= 1;                                                \
        }                                                               \
        if ((grid.x / 2) * THREADS_PER_CTA > NUM_ELEMENTS) {            \
            /* only need enough blocks to cover top and bottom halo elements */ \
            grid.x = 2 * ((NUM_ELEMENTS + THREADS_PER_CTA - 1) / THREADS_PER_CTA); \
        }                                                               \
        if (!TOP_ZERO) {                                                \
            /* require 2*128b=32B peer memory per thread */             \
            if ((grid.x / 2) * THREADS_PER_CTA * 32 > tox_size) {       \
                grid.x = 2 * (tox_size / (THREADS_PER_CTA * 32));       \
            }                                                           \
        }                                                               \
        if (!BTM_ZERO) {                                                \
            /* require 2*128b=32B peer memory per thread */             \
            if ((grid.x / 2) * THREADS_PER_CTA * 32 > box_size) {       \
                grid.x = 2 * (box_size / (THREADS_PER_CTA * 32));       \
            }                                                           \
        }                                                               \
        TORCH_CHECK(grid.x >= 2);                                       \
                                                                        \
        /* launch kernel */                                             \
        cudaLaunchCooperativeKernel(                                    \
            (void*)push_pull_halos_1d_kernel<T,CONTIGUOUS,TOP_ZERO,BTM_ZERO>, \
            grid,                                                       \
            block,                                                      \
            KERNEL_ARGS,                                                \
            0,                                                          \
            current_stream);                                            \
    } while (false)
#define LAUNCH_PUSH_PULL_HALO_KERNEL(T, CONTIGUOUS, KERNEL_ARGS, NUM_ELEMENTS) \
    do {                                                                \
        if (top_zero) {                                                 \
            LAUNCH_PUSH_PULL_HALO_KERNEL_BASE(T, CONTIGUOUS, true, false, KERNEL_ARGS, NUM_ELEMENTS); \
        } else if (btm_zero) {                                          \
            LAUNCH_PUSH_PULL_HALO_KERNEL_BASE(T, CONTIGUOUS, false, true, KERNEL_ARGS, NUM_ELEMENTS); \
        } else {                                                        \
            LAUNCH_PUSH_PULL_HALO_KERNEL_BASE(T, CONTIGUOUS, false, false, KERNEL_ARGS, NUM_ELEMENTS); \
        }                                                               \
    } while (false)

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, top_out_halo.scalar_type(), "push_pull_halos_1d_kernel", [&]{
	if (diagnostics) {
            printf("rank %d: size(scalar_t) = %ld\n", rank, sizeof(scalar_t));
        }
        scalar_t* toh_p = top_out_halo.data_ptr<scalar_t>();
        scalar_t* tih_p = top_in_halo.data_ptr<scalar_t>();
        int4* tox_p = reinterpret_cast<int4*>(top_out_transfer.data_ptr<scalar_t>());
        int4* tix_p = reinterpret_cast<int4*>(top_in_transfer.data_ptr<scalar_t>());
        scalar_t* boh_p = btm_out_halo.data_ptr<scalar_t>();
        scalar_t* bih_p = btm_in_halo.data_ptr<scalar_t>();
        int4* box_p = reinterpret_cast<int4*>(btm_out_transfer.data_ptr<scalar_t>());
        int4* bix_p = reinterpret_cast<int4*>(btm_in_transfer.data_ptr<scalar_t>());
        if (diagnostics) printf("rank %d: choosing halo exchange kernel\n", rank);

        // do int2 vector loads if channel count permits
        if (contiguous &&
            (NN*NH*NW*NC * sizeof(scalar_t)) % sizeof(int2) == 0) {
            // can do contiguous int2 transfers
            if (diagnostics) {
            }
            toh_stride_N = toh_stride_H = toh_stride_W = toh_stride_C = 1;
            tih_stride_N = tih_stride_H = tih_stride_W = tih_stride_C = 1;
            boh_stride_N = boh_stride_H = boh_stride_W = boh_stride_C = 1;
            bih_stride_N = bih_stride_H = bih_stride_W = bih_stride_C = 1;
            NC = (NN*NH*NW*NC * sizeof(scalar_t)) / sizeof(int2);
            NN = NH = NW = 1;
            if (diagnostics) {
                printf("rank %d: launching contiguous int2 halo exchange kernel\n",
                       rank);
                printf("rank %d: NC=%d, NH=%d, NW=%d\n", rank, NC, NH, NW);
            }
            void *kernel_args[] = {
                (int2**)&toh_p, &toh_stride_H, &toh_stride_W, &toh_stride_C,
                (int2**)&tih_p, &tih_stride_H, &tih_stride_W, &tih_stride_C,
                &tox_p, &tix_p,
                (int2**)&boh_p, &boh_stride_H, &boh_stride_W, &boh_stride_C,
                (int2**)&bih_p, &bih_stride_H, &bih_stride_W, &bih_stride_C,
                &box_p, &bix_p,
                &NH, &NW, &NC,
                &top_first
            };
            int num_elem = NN*NH*NW*NC;
            LAUNCH_PUSH_PULL_HALO_KERNEL(int2, true, kernel_args, num_elem);
        } else if (is_nhwc && (NC * sizeof(scalar_t)) % sizeof(int2) == 0) {
            // can do strided int2 transfers
            int divisor = sizeof(int2) / sizeof(scalar_t);
            if (diagnostics) {
                printf("rank %d: launching strided int2 halo exchange kernel\n",
                       rank);
            }
            toh_stride_N /= divisor;    toh_stride_H /= divisor;    toh_stride_W /= divisor;
            tih_stride_N /= divisor;    tih_stride_H /= divisor;    tih_stride_W /= divisor;
            boh_stride_N /= divisor;    boh_stride_H /= divisor;    boh_stride_W /= divisor;
            bih_stride_N /= divisor;    bih_stride_H /= divisor;    bih_stride_W /= divisor;
            NC /= divisor;
            if (diagnostics) {
                printf("rank %d: divisor=%d\n", rank, divisor);
                printf("rank %d: tih_stride :: N=%d, C=%d, H=%d, W=%d\n",
                       rank, tih_stride_N, tih_stride_C, tih_stride_H, tih_stride_W);
                printf("rank %d: toh_stride :: N=%d, C=%d, H=%d, W=%d\n",
                       rank, toh_stride_N, toh_stride_C, toh_stride_H, toh_stride_W);
                printf("rank %d: bih_stride :: N=%d, C=%d, H=%d, W=%d\n",
                       rank, bih_stride_N, bih_stride_C, bih_stride_H, bih_stride_W);
                printf("rank %d: boh_stride :: N=%d, C=%d, H=%d, W=%d\n",
                       rank, boh_stride_N, boh_stride_C, boh_stride_H, boh_stride_W);
                printf("rank %d: NC=%d, NH=%d, NW=%d\n", rank, NC, NH, NW);
            }
            void *kernel_args[] = {
                (int2**)&toh_p, &toh_stride_H, &toh_stride_W, &toh_stride_C,
                (int2**)&tih_p, &tih_stride_H, &tih_stride_W, &tih_stride_C,
                &tox_p, &tix_p,
                (int2**)&boh_p, &boh_stride_H, &boh_stride_W, &boh_stride_C,
                (int2**)&bih_p, &bih_stride_H, &bih_stride_W, &bih_stride_C,
                &box_p, &bix_p,
                &NH, &NW, &NC,
                &top_first
            };
            int num_elem = NH*NW*NC;
            LAUNCH_PUSH_PULL_HALO_KERNEL(int2, false, kernel_args, num_elem);
        } else {
            // cannot do int2 transfers
            if (diagnostics) {
                printf("rank %d: launching non-int2 halo exchange kernel\n",
                       rank);
            }
            int num_elem = NC*NH*NW;
            if (is_nhwc) {
                void *kernel_args[] = {
                    &toh_p, &toh_stride_H, &toh_stride_W, &toh_stride_C,
                    &tih_p, &tih_stride_H, &tih_stride_W, &tih_stride_C,
                    &tox_p, &tix_p,
                    &boh_p, &boh_stride_H, &boh_stride_W, &boh_stride_C,
                    &bih_p, &bih_stride_H, &bih_stride_W, &bih_stride_C,
                    &box_p, &bix_p,
                    &NH, &NW, &NC,
                    &top_first
                };
                LAUNCH_PUSH_PULL_HALO_KERNEL(scalar_t, false, kernel_args, num_elem);
            } else {
                void *kernel_args[] = {
                    &toh_p, &toh_stride_C, &toh_stride_H, &toh_stride_W,
                    &tih_p, &tih_stride_C, &tih_stride_H, &tih_stride_W,
                    &tox_p, &tix_p,
                    &boh_p, &boh_stride_C, &boh_stride_H, &boh_stride_W,
                    &bih_p, &bih_stride_C, &bih_stride_H, &bih_stride_W,
                    &box_p, &bix_p,
                    &NC, &NH, &NW,
                    &top_first
                };
                LAUNCH_PUSH_PULL_HALO_KERNEL(scalar_t, false, kernel_args, num_elem);
            }
        }
    } );

#undef LAUNCH_PUSH_PULL_HALO_KERNEL_BASE
#undef LAUNCH_PUSH_PULL_HALO_KERNEL
}

} } }
