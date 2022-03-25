import logging

import torch
from torch.testing._internal import common_utils

logging.getLogger("torch").setLevel(logging.WARNING)

from apex.transformer import parallel_state
from apex.transformer import tensor_parallel
from apex.transformer.testing.distributed_test_base import DistributedTestBase

logging.getLogger("apex").setLevel(logging.WARNING)


class TransformerRandomTest(DistributedTestBase):
    def test_set_cuda_rng_state(self):
        for tensor_model_parallel_world_size in range(1, self.world_size + 1):
            if self.world_size % tensor_model_parallel_world_size:
                continue
            with self.subTest(
                tensor_model_parallel_world_size=tensor_model_parallel_world_size
            ):
                parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size_=tensor_model_parallel_world_size
                )

                size, seed = 123, 1234
                torch.cuda.manual_seed(seed)
                tensor = torch.cuda.FloatTensor(size)

                rng_state = torch.cuda.get_rng_state()
                rng_state_clone = rng_state.clone()

                for _ in range(5):
                    torch.randn(size, out=tensor)
                result_1 = tensor.clone()

                self.assertEqual(rng_state.sub(rng_state_clone).max(), 0)
                self.assertGreater(
                    torch.cuda.get_rng_state().sub(rng_state_clone).max(), 0
                )

                new_rng_state = torch.cuda.get_rng_state()
                self.assertGreater(new_rng_state.sub(rng_state).max(), 0)

                tensor_parallel.random._set_cuda_rng_state(rng_state)
                for _ in range(5):
                    torch.randn(size, out=tensor)
                tensor_parallel.random._set_cuda_rng_state(rng_state)
                for _ in range(5):
                    torch.randn(size, out=tensor)
                result_2 = tensor.clone()

                torch.testing.assert_close(result_2, result_1)

                self.assertEqual(rng_state.sub(rng_state_clone).max(), 0)

                parallel_state.destroy_model_parallel()

    def test_cuda_rng_tracker(self):
        for tensor_model_parallel_world_size in range(1, self.world_size + 1):
            if self.world_size % tensor_model_parallel_world_size:
                continue
            with self.subTest(
                tensor_model_parallel_world_size=tensor_model_parallel_world_size
            ):
                parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size_=tensor_model_parallel_world_size
                )

                seed_1, seed_2, size = 1234, 4321, [12, 21]
                tensor = torch.cuda.FloatTensor(size)

                torch.cuda.manual_seed(seed_1)
                torch.randn(size, out=tensor)
                target_11 = tensor.clone()
                torch.randn(size, out=tensor)
                target_12 = tensor.clone()

                torch.cuda.manual_seed(seed_2)
                torch.randn(size, out=tensor)
                targt_21 = tensor.clone()
                torch.randn(size, out=tensor)
                target_22 = tensor.clone()

                torch.cuda.manual_seed(seed_1)
                tensor_parallel.random.get_cuda_rng_tracker().add("test", seed_2)

                torch.randn(size, out=tensor)
                result_11 = tensor.clone()

                with tensor_parallel.random.get_cuda_rng_tracker().fork("test"):
                    torch.randn(size, out=tensor)
                    result_21 = tensor.clone()

                torch.randn(size, out=tensor)
                result_12 = tensor.clone()

                with tensor_parallel.random.get_cuda_rng_tracker().fork("test"):
                    torch.randn(size, out=tensor)
                    result_22 = tensor.clone()

                self.assertEqual(target_11, result_11)
                self.assertEqual(target_12, result_12)
                self.assertEqual(targt_21, result_21)
                self.assertEqual(target_22, result_22)
                self.assertNotEqual(result_11, result_21)
                self.assertNotEqual(result_21, result_22)

                tensor_parallel.random.get_cuda_rng_tracker().reset()
                parallel_state.destroy_model_parallel()


if __name__ == "__main__":
    common_utils.run_tests()
