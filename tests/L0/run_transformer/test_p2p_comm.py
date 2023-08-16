import logging
import unittest

import torch
from torch.testing._internal import common_utils

logging.getLogger("torch").setLevel(logging.WARNING)

from apex.transformer import parallel_state
from apex.transformer.pipeline_parallel import p2p_communication
from apex.transformer.testing.distributed_test_base import NcclDistributedTestBase
from apex.transformer.testing.distributed_test_base import UccDistributedTestBase

logging.getLogger("apex").setLevel(logging.DEBUG)


# [P2P Ops Involved in Pipeline Model Parallel forward/backward]
# **forward_backward_pipelining_without_interleaving**
# - send_forward  / recv_forward
# - send_backward / recv_backward
# - send_forward_recv_backward
# - send_backward_recv_forward
# **forward_backward_pipelining_with_interleaving**
# - send_backward_recv_backward
# - recv_backward
# - recv_forward
# - send_forward_backward_recv_forward_backward
# - send_forward_recv_forward
class P2PCommTestBase:

    numel = 4
    shape = (2, 2)
    dtype = torch.float32

    @property
    def world_size(self):
        return min(2, torch.cuda.device_count())

    def _init_model_parallel(self):
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size_=1,
            pipeline_model_parallel_size_=self.world_size,
            virtual_pipeline_model_parallel_size_=None,
        )

    def create_tensor(self, value: int = None):
        return torch.tensor(
            [value] * self.numel).view(self.shape).to(device="cuda", dtype=self.dtype)

    # Brief: Simulate warm-up.
    # Brief: test `recv_forward` & `send_forward`.
    def test_no_interleaving_warmup(self):
        self.assertEqual(self.world_size, 2)
        self._init_model_parallel()
        input_tensor = None
        if parallel_state.is_pipeline_first_stage():
            tensor = self.create_tensor(self.rank)
            print(tensor)
            p2p_communication.send_forward(output_tensor=tensor, tensor_shape=self.shape, dtype=self.dtype)
        else:
            input_tensor = p2p_communication.recv_forward(tensor_shape=self.shape, dtype=self.dtype)

        if parallel_state.is_pipeline_first_stage():
            self.assertIsNone(input_tensor)
        else:
            expected_input_tensor = self.create_tensor(self.rank - 1)
            self.assertEqual(input_tensor, expected_input_tensor)

    # Brief: test `send_forward`, `send_forward_recv_forward`, and `recv_forward`.
    def test_send_forward_recv_forward(self):
        self._init_model_parallel()
        prev_tensor = None
        tensor = self.create_tensor(self.rank)
        if parallel_state.is_pipeline_first_stage():
            p2p_communication.send_forward(output_tensor=tensor, tensor_shape=self.shape, dtype=self.dtype)
        elif parallel_state.is_pipeline_last_stage():
            prev_tensor = p2p_communication.recv_forward(tensor_shape=self.shape, dtype=self.dtype)
        else:
            prev_tensor = p2p_communication.send_forward_recv_forward(
                output_tensor=tensor,
                recv_prev=True,
                tensor_shape=self.shape,
                dtype=self.dtype,
            )

        if parallel_state.is_pipeline_first_stage():
            self.assertIsNone(prev_tensor)
        else:
            expected_prev_tensor = self.create_tensor(self.rank - 1)
            self.assertEqual(prev_tensor, expected_prev_tensor)

    # Brief: test `send_backward`, `send_backward_recv_backward`, and `recv_backward`.
    def test_send_backward_recv_backward(self):
        self._init_model_parallel()
        tensor = self.create_tensor(self.rank)

        next_tensor = None
        if parallel_state.is_pipeline_first_stage():
            next_tensor = p2p_communication.recv_backward(tensor_shape=self.shape, dtype=self.dtype)
        elif parallel_state.is_pipeline_last_stage():
            p2p_communication.send_backward(input_tensor_grad=tensor, tensor_shape=self.shape, dtype=self.dtype)
        else:
            next_tensor = p2p_communication.send_backward_recv_backward(
                input_tensor_grad=tensor,
                recv_next=True,
                tensor_shape=self.shape,
                dtype=self.dtype,
            )

        if parallel_state.is_pipeline_last_stage():
            self.assertIsNone(next_tensor)
        else:
            expected_next_tensor = self.create_tensor(self.rank + 1)
            self.assertEqual(next_tensor, expected_next_tensor)


# n.b.(mkozuki): Intentionally skip NCCL backend tests as I trust pytorch/pytorch repo.
@unittest.skipIf(torch.cuda.device_count() < 2, "Requires >= 2 GPUs")
class UccP2PCommTest(P2PCommTestBase, UccDistributedTestBase): pass


if __name__ == "__main__":
    common_utils.run_tests()
