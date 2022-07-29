import logging

import torch
from torch.testing._internal import common_utils

from apex.transformer import parallel_state
from apex.transformer.tensor_parallel import mappings
from apex.transformer.testing.distributed_test_base import NcclDistributedTestBase
from apex.transformer.testing.distributed_test_base import UccDistributedTestBase


logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("apex").setLevel(logging.WARNING)


class MappingTestBase:
    def test_reduce(self):
        for tensor_model_paralell_world_size in range(1, self.world_size + 1):
            if self.world_size % tensor_model_paralell_world_size > 0:
                continue
            with self.subTest(
                tensor_model_paralell_world_size=tensor_model_paralell_world_size
            ):
                parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size_=tensor_model_paralell_world_size
                )
                t = torch.full((10, 10, 10, 10), 50, device=f"cuda:{self.rank}")
                expected = torch.full(
                    (10, 10, 10, 10),
                    50 * tensor_model_paralell_world_size,
                    device=f"cuda:{self.rank}",
                )
                self.assertTrue(torch.equal(mappings._reduce(t), expected))
                parallel_state.destroy_model_parallel()

    def test_split(self):
        for tensor_model_paralell_world_size in range(1, self.world_size + 1):
            if self.world_size % tensor_model_paralell_world_size > 0:
                continue
            with self.subTest(
                tensor_model_paralell_world_size=tensor_model_paralell_world_size
            ):
                parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size_=tensor_model_paralell_world_size
                )

                tensors = [
                    torch.randn(10, 1)
                    for rank in range(tensor_model_paralell_world_size)
                ]
                x = torch.cat(tensors, 1)
                out = mappings._split_along_last_dim(x)
                self.assertTrue(
                    torch.equal(
                        out, tensors[parallel_state.get_tensor_model_parallel_rank()]
                    )
                )
                parallel_state.destroy_model_parallel()

    def test_gather(self):
        for tensor_model_paralell_world_size in range(1, self.world_size + 1):
            if self.world_size % tensor_model_paralell_world_size > 0:
                continue
            with self.subTest(
                tensor_model_paralell_world_size=tensor_model_paralell_world_size
            ):
                parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size_=tensor_model_paralell_world_size
                )
                device = f"cuda:{self.rank}"
                gathered = mappings._gather_along_last_dim(
                    torch.tensor(
                        [parallel_state.get_tensor_model_parallel_rank()], device=device
                    )
                )
                expected = torch.tensor(
                    [rank for rank in range(tensor_model_paralell_world_size)],
                    device=device,
                )
                self.assertTrue(torch.equal(gathered, expected))
                parallel_state.destroy_model_parallel()


class NcclMappingTest(MappingTestBase, NcclDistributedTestBase): pass
class UccMappingTest(MappingTestBase, UccDistributedTestBase): pass


if __name__ == "__main__":
    common_utils.run_tests()
