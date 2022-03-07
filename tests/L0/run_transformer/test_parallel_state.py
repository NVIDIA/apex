import os
import sys

import torch
from torch import distributed as dist
from torch.testing._internal import common_utils
from torch.testing._internal import common_distributed

from apex.transformer import parallel_state


os.environ["BACKEND"] = "NCCL"
DATA_PARALLEL_WORLD_SIZE: int = 1


def calc_expected_tensor_model_paralell_rank(
    rank: int,
    tensor_model_parallel_world_size: int,
) -> int:
    return rank % tensor_model_parallel_world_size


class ParallelStateTest(common_distributed.MultiProcessTestCase):

    BACKEND_NCCL = "nccl"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._world_size = min(torch.cuda.device_count(), super().world_size)

    def setUp(self) -> None:
        super().setUp()
        self._world_size = min(torch.cuda.device_count(), super().world_size)
        self._spawn_processes()

    def tearDown(self) -> None:
        super().tearDown()
        self._world_size = None

    # N.B. (mkozuki): From the perspective of execution time, I think
    # faster tests are preferred to longer ones, but in some cases,
    # we want to run longer for coverage.
    # So, preparing a knob to set world_size > 4.
    @property
    def world_size(self) -> int:
        return self._world_size

    @world_size.setter
    def world_size(self, new_world_size: int) -> None:
        self._world_size = new_world_size

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

        for tensor_model_parallel_world_size in range(1, self.world_size + 1):
            with self.subTest(
                tensor_model_parallel_world_size=tensor_model_parallel_world_size
            ):
                if self.world_size % tensor_model_parallel_world_size:
                    continue

                pipeline_model_parallel_world_size = (
                    self.world_size // tensor_model_parallel_world_size
                )

                parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size_=tensor_model_parallel_world_size,
                    pipeline_model_parallel_size_=pipeline_model_parallel_world_size,
                )
                self.assertEqual(
                    tensor_model_parallel_world_size,
                    parallel_state.get_tensor_model_parallel_world_size(),
                )
                expected_tensor_model_parallel_rank = (
                    calc_expected_tensor_model_paralell_rank(
                        self.rank, tensor_model_parallel_world_size
                    )
                )
                self.assertEqual(
                    expected_tensor_model_parallel_rank,
                    parallel_state.get_tensor_model_parallel_rank(),
                )

                expected_tensor_model_parallel_src_rank = (
                    self.rank // tensor_model_parallel_world_size
                ) * tensor_model_parallel_world_size
                self.assertEqual(
                    expected_tensor_model_parallel_src_rank,
                    parallel_state.get_tensor_model_parallel_src_rank(),
                )

                parallel_state.destroy_model_parallel()
                self.assertFalse(parallel_state.model_parallel_is_initialized())

    def test_initialize_model_parallel_with_virtual_and_split(self) -> None:
        if self.world_size < 4:
            self.skipTest("requires >= 4 GPUs")
        self.assertFalse(parallel_state.model_parallel_is_initialized())

        tensor_model_parallel_world_size = 1 + int(self.world_size > 4)
        pipeline_model_parallel_world_size = (
            self.world_size // tensor_model_parallel_world_size
        )
        virtual_pipeline_model_parallel_world_size = 2
        pipeline_model_parallel_split_rank = pipeline_model_parallel_world_size // 2

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size_=tensor_model_parallel_world_size,
            pipeline_model_parallel_size_=pipeline_model_parallel_world_size,
            virtual_pipeline_model_parallel_size_=virtual_pipeline_model_parallel_world_size,
            pipeline_model_parallel_split_rank_=pipeline_model_parallel_split_rank,
        )
        self.assertEqual(
            calc_expected_tensor_model_paralell_rank(
                self.rank, tensor_model_parallel_world_size
            ),
            parallel_state.get_tensor_model_parallel_rank(),
        )
        self.assertEqual(
            pipeline_model_parallel_world_size,
            parallel_state.get_pipeline_model_parallel_world_size(),
        )
        self.assertEqual(
            virtual_pipeline_model_parallel_world_size,
            parallel_state.get_virtual_pipeline_model_parallel_world_size(),
        )

        expected_pipeline_rank = (
            self.rank - (self.rank % tensor_model_parallel_world_size)
        ) % pipeline_model_parallel_world_size
        self.assertEqual(
            expected_pipeline_rank,
            parallel_state.get_pipeline_model_parallel_rank(),
        )
        # virtual pipeline model parallel rank is lazily set, i.e., right after the call of
        # `initialize_model_parallel`, it's set to 0.
        self.assertEqual(
            0,
            parallel_state.get_virtual_pipeline_model_parallel_rank(),
        )
        self.assertEqual(
            pipeline_model_parallel_split_rank,
            parallel_state.get_pipeline_model_parallel_split_rank(),
        )

        fake_split_rank = 77
        parallel_state.set_pipeline_model_parallel_split_rank(fake_split_rank)
        self.assertEqual(
            fake_split_rank, parallel_state.get_pipeline_model_parallel_split_rank()
        )

        parallel_state.destroy_model_parallel()


if __name__ == "__main__":
    common_utils.run_tests()
