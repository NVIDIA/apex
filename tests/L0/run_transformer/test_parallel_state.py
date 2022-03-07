import os
import sys

import torch
from torch import distributed as dist
from torch.testing._internal import common_utils
from torch.testing._internal import common_distributed
# from torch.testing._internal.distributed.distributed_test import TestDistBackend

from apex.transformer import parallel_state


os.environ["BACKEND"] = "NCCL"


def calc_expected_tensor_model_paralell_rank(
        rank: int,
        tensor_model_parallel_world_size: int,
) -> int:
    return rank % tensor_model_parallel_world_size


class ParallelStateTest(common_distributed.MultiProcessTestCase):

    BACKEND_NCCL = "nccl"

    def setUp(self) -> None:
        super().setUp()
        print("Called TestDistBackend.setUp")
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return min(torch.cuda.device_count(), super().world_size)

    # TODO (mkozuki): Check if this is seriously needed.
    @property
    def init_method(self):
        return f"{common_utils.FILE_SCHEMA}{self.file_name}"

    @classmethod
    def _run(cls, rank, test_name, file_name, pipe):
        self = cls(test_name)
        self.assertTrue(torch.cuda.is_available())
        self.rank = rank
        self.file_name = file_name

        print(f"[dist init] rank = {self.rank}, world_size = {self.world_size}")

        backend = os.environ.get("BACKEND", None)
        if backend is None:
            back_end = ParallelStateTest.BACKEND_NCCL

        try:
            dist.init_process_group(
                init_method=self.init_method,
                backend=backend,
                world_size=int(self.world_size),
                rank=self.rank,
            )
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                print(f"Backend of {back_end} not available")
                sys.exit(0)
            raise

        torch.cuda.set_device(self.rank % torch.cuda.device_count())

        dist.barrier()
        self.run_test(test_name, pipe)
        dist.barrier()

        dist.destroy_process_group()
        sys.exit(0)

    def test_initialize_model_parallel(self) -> None:
        
        self.assertFalse(parallel_state.model_parallel_is_initialized())

        parallel_state.initialize_model_parallel(tensor_model_parallel_size_=self.world_size)
        self.assertTrue(parallel_state)
        self.assertEqual(
            self.world_size,
            parallel_state.get_tensor_model_parallel_world_size(),
        )
        expected_tensor_model_parallel_rank = calc_expected_tensor_model_paralell_rank(
            self.rank, self.world_size)
        self.assertEqual(
            expected_tensor_model_parallel_rank,
            parallel_state.get_tensor_model_parallel_rank(),
        )

        parallel_state.destroy_model_parallel()
        self.assertFalse(parallel_state.model_parallel_is_initialized())


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests
    run_tests()
