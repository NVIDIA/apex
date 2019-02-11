import torch
from amp_C import prep_multi_tensor_launch

class MultiTensorApply(object):
    def __init__(self, max_blocks, max_tensors, max_depth, chunk_size):
        self.chunk_size = chunk_size
        self.reallocate(max_blocks, max_tensors, max_depth)

    def __call__(self, op, noop_flag_buffer, tensor_lists, *args):
        self.assign_blocks(tensor_lists)

        # print(self.gpu_block_to_tensor)
        # print(self.gpu_block_to_chunk)
        # print(self.gpu_tensor_sizes)

        return op(self.nblocks,
                  noop_flag_buffer,
                  self.cpu_tensor_addresses,
                  self.gpu_block_to_tensor, 
                  self.gpu_block_to_chunk,
                  self.gpu_tensor_sizes,
                  self.gpu_tensor_addresses,
                  self.chunk_size, 
                  tensor_lists,
                  *args)

        # print()
        # print([[p.data_ptr() for p in l] for l in tensor_lists])
        # print()
        # print(self.gpu_tensor_addresses)

    def assign_blocks(self, tensor_lists):
        needs_reallocate = False

        # Currently, this loop appears prohibitively expensive.
        # Need to move to c++.
        torch.cuda.nvtx.range_push("assign_blocks loop")
        # list0 = tensor_lists[0]
        # self.nblocks = 0
        # for t, tensor in enumerate(list0):
        #     blocks_this_tensor = (tensor.numel() + 
        #                           self.chunk_size - 1)//self.chunk_size
        #     if not needs_reallocate:
        #         self.cpu_tensor_sizes[t] = tensor.numel()
        #     for chunk in range(blocks_this_tensor):
        #         if self.nblocks >= self.max_blocks:
        #             needs_reallocate = True
        #         if not needs_reallocate:
        #             self.cpu_block_to_tensor[self.nblocks] = t
        #             self.cpu_block_to_chunk[self.nblocks] = chunk
        #         self.nblocks += 1
        needs_reallocate, self.nblocks = prep_multi_tensor_launch(self.cpu_block_to_tensor, 
                                                                  self.cpu_block_to_chunk,
                                                                  self.cpu_tensor_sizes,
                                                                  self.gpu_block_to_tensor, 
                                                                  self.gpu_block_to_chunk,
                                                                  self.gpu_tensor_sizes,
                                                                  self.chunk_size,
                                                                  self.max_depth,
                                                                  self.max_tensors,
                                                                  self.max_blocks,
                                                                  tensor_lists)
        torch.cuda.nvtx.range_pop()

        print(self.nblocks)

        if self.nblocks > self.max_blocks:
            self.max_blocks = self.nblocks
        if len(tensor_lists) > self.max_depth:
            self.max_depth = len(tensor_lists)
        if len(tensor_lists[0]) > self.max_tensors:
            self.max_tensors = len(tensor_lists[0])

        if needs_reallocate:
            self.reallocate(self.max_blocks, self.max_tensors, self.max_depth)
            needs_reallocate, self.nblocks = prep_multi_tensor_launch(self.cpu_block_to_tensor, 
                                                                      self.cpu_block_to_chunk,
                                                                      self.cpu_tensor_sizes,
                                                                      self.gpu_block_to_tensor, 
                                                                      self.gpu_block_to_chunk,
                                                                      self.gpu_tensor_sizes,
                                                                      self.chunk_size,
                                                                      self.max_depth,
                                                                      self.max_tensors,
                                                                      self.max_blocks,
                                                                      tensor_lists)
            assert needs_reallocate == 0, "Should not need reallocate on second attempt."
            assert self.nblocks <= self.max_blocks, "Should not need to increase blocks again."

    def reallocate(self, max_blocks, max_tensors, max_depth):
        self.max_blocks = max_blocks
        self.max_tensors = max_tensors 
        self.max_depth = max_depth

        self.cpu_block_to_tensor = torch.IntTensor(max_blocks).pin_memory()
        self.cpu_block_to_chunk = torch.IntTensor(max_blocks).pin_memory()
        self.cpu_tensor_sizes = torch.IntTensor(max_tensors).pin_memory()
        self.cpu_tensor_addresses = torch.LongTensor(max_depth, max_tensors).pin_memory()

        self.gpu_block_to_tensor = torch.cuda.IntTensor(max_blocks)
        self.gpu_block_to_chunk = torch.cuda.IntTensor(max_blocks)
        self.gpu_tensor_sizes = torch.cuda.IntTensor(max_tensors)
        self.gpu_tensor_addresses = torch.cuda.LongTensor(max_depth, max_tensors)
