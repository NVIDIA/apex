#include <torch/extension.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAContext.h>
#include <list>
#include <cstdio>
#include <ctime>
#include <cassert>
#include "nccl.h"

/*
 * This file implements a crude but effective mechanism for copying data between tenors owned by different ranks
 * on the same machine using cudaMemcpyAsync peer-to-peer transfers.
 */

namespace {

__global__ void AddDelay_kernel(const int delay, int* counter) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // waste time while doing something compiler can't predict, thus preventing it from optimizing away this code.
        int new_counter = 0;
        double elapsed = 0;
        clock_t start = clock();
        do {
            clock_t now = clock();
            elapsed = (double)(now - start)*1e9 / CLOCKS_PER_SEC;
            ++new_counter;
        } while (elapsed < (double)delay);
        *counter = new_counter;
    }
}

class NcclCommWrapper
{
    private:
        ncclComm_t comm;
        int rank, world_size;

        ncclDataType_t get_nccl_type(at::Tensor input)
        {
            switch (input.scalar_type())
            {
                case at::ScalarType::Half:
                    return ncclFloat16;
                case at::ScalarType::Float:
                    return ncclFloat32;
                case at::ScalarType::Double:
                    return ncclFloat64;
                case at::ScalarType::Byte:
                    return ncclUint8;
                case at::ScalarType::Char:
                    return ncclInt8;
                case at::ScalarType::Int:
                    return ncclInt32;
                case at::ScalarType::Long:
                    return ncclInt64;
                case at::ScalarType::BFloat16:
                    return ncclBfloat16;
                default:
                    assert(false);
            }
        }

    public:
        NcclCommWrapper()
        {
            memset(&comm, 0, sizeof(ncclComm_t));
            rank = 0;
            world_size = 0;
        }
        NcclCommWrapper(ncclUniqueId id, int my_rank, int num_ranks)
        {
            ncclCommInitRank(&comm, num_ranks, id, my_rank);
            rank = my_rank;
            world_size = num_ranks;
        }

        ~NcclCommWrapper()
        {
            printf("ncclCommDestroy()\n");
            ncclCommDestroy(comm);
        }

        void send(at::Tensor input, int destination)
        {
            ncclDataType_t ncclType = get_nccl_type(input);
            AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half, input.scalar_type(), "nccl_send", [&]() {
                size_t count = sizeof(scalar_t) * torch::numel(input);
                auto input_ptr = input.data_ptr<scalar_t>();
                ncclSend(input_ptr, count, ncclType, destination, comm, at::cuda::getCurrentCUDAStream());
            });
        }

        void recv(at::Tensor input, int sender)
        {
            ncclDataType_t ncclType = get_nccl_type(input);
            AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half, input.scalar_type(), "nccl_send", [&]() {
                size_t count = sizeof(scalar_t) * torch::numel(input);
                auto input_ptr = input.data_ptr<scalar_t>();
                ncclRecv(input_ptr, count, ncclType, sender, comm, at::cuda::getCurrentCUDAStream());
            });
        }

	void left_right_halo_exchange_inplace(bool left_zero, bool right_zero, at::Tensor left_output_halo, at::Tensor right_output_halo, at::Tensor left_input_halo, at::Tensor right_input_halo, int group_size)
	{
            auto stream = at::cuda::getCurrentCUDAStream();
            ncclGroupStart();
            ncclDataType_t ncclType = get_nccl_type(left_output_halo);
            // we use wrap-around ranks, so left_input_halo of rank 0 has right_output_halo of rank world_size-1 after exchange etc.
            // this is technically speaking wasteful, but there is no benefit in having the edge ranks do less work than internal ranks.
            int group_rank = rank % group_size;
            int group_index = rank / group_size;
            int prev_rank = (group_rank + group_size - 1) % group_size;
            int next_rank = (group_rank + 1) % group_size;
            prev_rank = prev_rank + group_index * group_size;
            next_rank = next_rank + group_index * group_size;
            size_t left_n = torch::numel(left_output_halo);
            size_t right_n = torch::numel(right_output_halo);
            if (group_rank > 0) {
                AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half, left_output_halo.scalar_type(), "left_halo_exch", [&]() {
                    // send left (to my_rank - 1)
                    ncclSend(left_output_halo.data_ptr<scalar_t>(), left_n, ncclType, prev_rank, comm, stream);
                    // receive left (from my_rank - 1)
                    ncclRecv(left_input_halo.data_ptr<scalar_t>(), right_n, ncclType, prev_rank, comm, stream);
                });
            }
            if (group_rank < group_size-1) {
                AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half, right_output_halo.scalar_type(), "right_halo_exch", [&]() {
                    // send right (to my_rank + 1 )
                    ncclSend(right_output_halo.data_ptr<scalar_t>(), right_n, ncclType, next_rank, comm, stream);
                    // receive right (from my_rank + 1)
                    ncclRecv(right_input_halo.data_ptr<scalar_t>(), left_n, ncclType, next_rank, comm, stream);
                });
            }
            ncclGroupEnd();
	    if (left_zero) left_input_halo.zero_();
	    if (right_zero) right_input_halo.zero_();
	}

        std::vector<at::Tensor> left_right_halo_exchange(bool left_zero, bool right_zero, at::Tensor left_output_halo, at::Tensor right_output_halo, int group_size)
        {
            // after halo exchange:
            // left_output_halo of rank+1 ends up in right_input_halo of rank
            // right_output_halo of rank-1 ends up in left_input_halo of rank
            auto right_input_halo = torch::empty_like(left_output_halo);
            auto left_input_halo = torch::empty_like(right_output_halo);
	    left_right_halo_exchange_inplace(left_zero, right_zero, left_output_halo, right_output_halo, left_input_halo, right_input_halo, group_size);
	    return {left_input_halo, right_input_halo};
        }
};

std::vector<NcclCommWrapper> nccl_comms;

} // end anonymous namespace

namespace apex { namespace contrib { namespace nccl_p2p {

at::Tensor get_unique_nccl_id(int n)
{
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    auto id_tensor = torch::empty({n*(int)sizeof(ncclUniqueId)}, torch::dtype(torch::kUInt8).device(torch::kCPU).requires_grad(false));
    auto id_ptr = id_tensor.data_ptr<uint8_t>();
    size_t offset = 0;
    for (int i = 0;  i < n;  ++i)
    {
        ncclUniqueId id;
        ncclGetUniqueId(&id);
        memcpy(id_ptr+offset, &id, sizeof(ncclUniqueId));
        offset += sizeof(ncclUniqueId);
    }
    return id_tensor;
}

int init_nccl_comm(at::Tensor unique_nccl_id, int my_rank, int num_ranks)
{
    ncclUniqueId id;
    auto unique_nccl_id_ptr = unique_nccl_id.data_ptr<uint8_t>();
    memcpy(&id, unique_nccl_id_ptr, sizeof(ncclUniqueId));
    NcclCommWrapper* comm = new NcclCommWrapper(id, my_rank, num_ranks);
    int handle = nccl_comms.size();
    nccl_comms.push_back(*comm);
    comm = 0L;
    return handle;
}

void nccl_send(int handle, at::Tensor input, int destination)
{
    assert(handle >= 0 && handle < nccl_comms.size());
    class NcclCommWrapper communicator = nccl_comms[handle];
    communicator.send(input, destination);
}

void nccl_recv(int handle, at::Tensor input, int sender)
{
    assert(handle >= 0 && handle < nccl_comms.size());
    class NcclCommWrapper communicator = nccl_comms[handle];
    communicator.recv(input, sender);
}

void left_right_halo_exchange_inplace(int handle, bool left_zero, bool right_zero, at::Tensor left_output_halo, at::Tensor right_output_halo, at::Tensor left_input_halo, at::Tensor right_input_halo, int group_size)
{
    assert(handle >= 0 && handle < nccl_comms.size());
    class NcclCommWrapper& communicator = nccl_comms[handle];
    return communicator.left_right_halo_exchange_inplace(left_zero, right_zero, left_output_halo, right_output_halo, left_input_halo, right_input_halo, group_size);
}

std::vector<at::Tensor> left_right_halo_exchange(int handle, bool left_zero, bool right_zero, at::Tensor left_output_halo, at::Tensor right_output_halo, int group_size)
{
    assert(handle >= 0 && handle < nccl_comms.size());
    class NcclCommWrapper& communicator = nccl_comms[handle];
    return communicator.left_right_halo_exchange(left_zero, right_zero, left_output_halo, right_output_halo, group_size);
}

void add_delay(int delay)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    auto t = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    AddDelay_kernel<<<1,1,0,stream>>>(delay, t.data_ptr<int>());
}

}}}
