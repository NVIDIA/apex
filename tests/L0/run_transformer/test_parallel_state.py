import logging
import os

from torch.testing._internal import common_utils

logging.getLogger("torch").setLevel(logging.WARNING)

from apex.transformer import parallel_state
from apex.transformer.testing.distributed_test_base import NcclDistributedTestBase
from apex.transformer.testing.distributed_test_base import UccDistributedTestBase

logging.getLogger("apex").setLevel(logging.WARNING)


os.environ["BACKEND"] = "NCCL"
DATA_PARALLEL_WORLD_SIZE: int = 1


def calc_expected_tensor_model_paralell_rank(
    rank: int, tensor_model_parallel_world_size: int,
) -> int:
    return rank % tensor_model_parallel_world_size


class ParallelStateTestBase:
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
                expected_tensor_model_parallel_rank = calc_expected_tensor_model_paralell_rank(
                    self.rank, tensor_model_parallel_world_size
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
            expected_pipeline_rank, parallel_state.get_pipeline_model_parallel_rank(),
        )
        # virtual pipeline model parallel rank is lazily set, i.e., right after the call of
        # `initialize_model_parallel`, it's set to 0.
        self.assertEqual(
            0, parallel_state.get_virtual_pipeline_model_parallel_rank(),
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

        # relative position embedding groups check
        self.assertEqual(
           expected_pipeline_rank < pipeline_model_parallel_split_rank,
           parallel_state.is_rank_in_encoder_relative_position_embedding_group(),
        )
        self.assertEqual(
           expected_pipeline_rank >= pipeline_model_parallel_split_rank,
           parallel_state.is_rank_in_decoder_relative_position_embedding_group(),
        )

        parallel_state.destroy_model_parallel()

    def test_initialize_model_parallel_decoder_only(self) -> None:
        """Initialize model parallelism for decoder-only Transformers like GPT-3"""

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
                    pipeline_model_parallel_split_rank_=0,
                )
                self.assertEqual(
                    tensor_model_parallel_world_size,
                    parallel_state.get_tensor_model_parallel_world_size(),
                )
                expected_tensor_model_parallel_rank = calc_expected_tensor_model_paralell_rank(
                    self.rank, tensor_model_parallel_world_size
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


class NcclParallelStateTest(ParallelStateTestBase, NcclDistributedTestBase): pass
class UccParallelStateTest(ParallelStateTestBase, UccDistributedTestBase): pass


if __name__ == "__main__":
    common_utils.run_tests()
