import logging

import torch
from torch.testing._internal import common_utils

logging.getLogger("torch").setLevel(logging.WARNING)

from apex.transformer import parallel_state
from apex.transformer.tensor_parallel import utils
from apex.transformer.testing.distributed_test_base import DistributedTestBase

logging.getLogger("apex").setLevel(logging.WARNING)


class TransformerUtilsTest(DistributedTestBase):
    def test_split_tensor_along_last_dim(self):
        for tensor_model_paralell_world_size in range(1, self.world_size + 1):
            if self.world_size % tensor_model_paralell_world_size > 0:
                continue
            with self.subTest(
                tensor_model_paralell_world_size=tensor_model_paralell_world_size
            ):
                parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size_=tensor_model_paralell_world_size
                )

                device = "cpu"
                input_tensor = torch.randn((100, 100, 100), device=device)
                splits = utils.split_tensor_along_last_dim(input_tensor, 10)
                last_dim_shapes = torch.tensor(
                    [int(split.size()[-1]) for split in splits]
                )

                self.assertTrue(torch.equal(last_dim_shapes, torch.full((10,), 10),))

                parallel_state.destroy_model_parallel()


if __name__ == "__main__":
    common_utils.run_tests()
