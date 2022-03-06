import os
import unittest

import torch
from torch import distributed as dist, tensor
from torch.testing._internal.distributed.distributed_test import TestDistBackend

from apex.transformer import parallel_state


os.environ["BACKEND"] = "NCCL"


def calc_expected_tensor_model_paralell_rank(rank: int, tensor_model_parallel_world_size: int) -> int:
    return rank % tensor_model_parallel_world_size


class ParallelStateTest(TestDistBackend):

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        parallel_state.destroy_model_parallel()
        super().tearDown()

    def test_initialize_model_parallel(self) -> None:
        world_size: int = dist.get_world_size()
        
        for tensor_model_parallel_world_size in range(1, world_size + 1):
            with self.subTest(tensor_model_parallel_size=tensor_model_parallel_world_size):
                self.assertFalse(parallel_state.model_parallel_is_initialized())
                parallel_state.initialize_model_parallel(tensor_model_parallel_size_=tensor_model_parallel_world_size)
                self.assertTrue(parallel_state)
                self.assertEqual(tensor_model_parallel_world_size, parallel_state.get_tensor_model_parallel_world_size())
                expected_tensor_model_parallel_rank = calc_expected_tensor_model_paralell_rank(dist.get_rank(), tensor_model_parallel_world_size)
                self.assertEqual(expected_tensor_model_parallel_rank, parallel_state.get_tensor_model_parallel_rank())

                parallel_state.destroy_model_parallel()
                self.assertFalse(parallel_state.model_parallel_is_initialized())


if __name__ == "__main__":
    unittest.main()
