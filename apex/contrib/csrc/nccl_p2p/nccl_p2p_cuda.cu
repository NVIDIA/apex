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

	void left_right_halo_exchange_inplace(int left_rank, int right_rank, at::Tensor left_output_halo, at::Tensor right_output_halo, at::Tensor left_input_halo, at::Tensor right_input_halo)
	{
            auto stream = at::cuda::getCurrentCUDAStream();
            ncclGroupStart();
            ncclDataType_t ncclType = get_nccl_type(left_output_halo);
	    bool left_zero = (left_rank < 0);
	    bool right_zero = (right_rank < 0);
            size_t left_n = torch::numel(left_output_halo);
            size_t right_n = torch::numel(right_output_halo);
	    assert(left_n > 0 && left_n == right_n);
	    if (left_zero) {
		left_input_halo.zero_();
	    } else {
                AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half, left_output_halo.scalar_type(), "left_halo_exch", [&]() {
                    // send left (to my_rank - 1)
                    ncclSend(left_output_halo.data_ptr<scalar_t>(), left_n, ncclType, left_rank, comm, stream);
                    // receive left (from my_rank - 1)
                    ncclRecv(left_input_halo.data_ptr<scalar_t>(), right_n, ncclType, left_rank, comm, stream);
                });
            }
            if (right_zero) {
		right_input_halo.zero_();
	    } else {
                AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Bool, at::ScalarType::BFloat16, at::ScalarType::Half, right_output_halo.scalar_type(), "right_halo_exch", [&]() {
                    // send right (to my_rank + 1 )
                    ncclSend(right_output_halo.data_ptr<scalar_t>(), right_n, ncclType, right_rank, comm, stream);
                    // receive right (from my_rank + 1)
                    ncclRecv(right_input_halo.data_ptr<scalar_t>(), left_n, ncclType, right_rank, comm, stream);
                });
            }
            ncclGroupEnd();
	}

        std::vector<at::Tensor> left_right_halo_exchange(int left_rank, int right_rank, at::Tensor left_output_halo, at::Tensor right_output_halo)
        {
            // after halo exchange:
            // left_output_halo of rank+1 ends up in right_input_halo of rank
            // right_output_halo of rank-1 ends up in left_input_halo of rank
            auto right_input_halo = torch::empty_like(left_output_halo);
            auto left_input_halo = torch::empty_like(right_output_halo);
	    left_right_halo_exchange_inplace(left_rank, right_rank, left_output_halo, right_output_halo, left_input_halo, right_input_halo);
	    return {left_input_halo, right_input_halo};
        }
};

class ManagedObjects
{
    public:
	ManagedObjects()
	{
	}
	~ManagedObjects()
	{
	    for (auto it = _nccl_comms.begin(); it != _nccl_comms.end();  ++it)
	    {
		delete *it;
	    }
	}

	int add_comm(NcclCommWrapper* comm)
	{
	    int handle = _nccl_comms.size();
	    _nccl_comms.push_back(comm);
	    return handle;
	}

	NcclCommWrapper& get_comm(int handle)
	{
            assert(handle >= 0 && handle < _nccl_comms.size());
	    return *_nccl_comms[handle];
	}

    private:
	std::vector<NcclCommWrapper*> _nccl_comms;
};
class ManagedObjects mo;

} // end anonymous namespace

namespace apex { namespace contrib { namespace nccl_p2p {

at::Tensor get_unique_nccl_id(int n)
{
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    auto id_tensor = torch::empty({n,(int)sizeof(ncclUniqueId)}, torch::dtype(torch::kUInt8).device(torch::kCPU).requires_grad(false));
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
    int handle = mo.add_comm(comm);
    comm = 0L;
    return handle;
}

void left_right_halo_exchange_inplace(int handle, int left_rank, int right_rank, at::Tensor left_output_halo, at::Tensor right_output_halo, at::Tensor left_input_halo, at::Tensor right_input_halo)
{
    class NcclCommWrapper& communicator = mo.get_comm(handle);
    return communicator.left_right_halo_exchange_inplace(left_rank, right_rank, left_output_halo, right_output_halo, left_input_halo, right_input_halo);
}

std::vector<at::Tensor> left_right_halo_exchange(int handle, int left_rank, int right_rank, at::Tensor left_output_halo, at::Tensor right_output_halo)
{
    class NcclCommWrapper& communicator = mo.get_comm(handle);
    return communicator.left_right_halo_exchange(left_rank, right_rank, left_output_halo, right_output_halo);
}

void add_delay(int delay)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    auto t = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    AddDelay_kernel<<<1,1,0,stream>>>(delay, t.data_ptr<int>());
}

}}}
