import itertools
from importlib_metadata import requires

import torch
import torch.nn as nn
from torch.testing._internal import common_utils

from apex.transformer import parallel_state
from apex.transformer import tensor_parallel
from apex.transformer.tensor_parallel import layers
from apex.transformer.testing.commons import set_random_seed
from apex.transformer.testing.distributed_test_base import DistributedTestBase


class TensorParallelLayerTest(DistributedTestBase):

    BATCH_SIZE: int = 17
    SEQUENCE_LENGTH: int = 23
    VOCAB_SIZE: int = 48
    HIDDEN_SIZE: int = 16
    INPUT_SIZE_COEFF: int = 13
    OUTPUT_SIZE_COEFF: int = 17
    SEED: int = 123

    def test_parallel_embedding(self) -> None:
        for tensor_model_parallel_world_size in range(1, self.world_size + 1):
            if self.world_size % tensor_model_parallel_world_size:
                continue
            with self.subTest(
                tensor_model_parallel_world_size=tensor_model_parallel_world_size
            ):
                parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size_=tensor_model_parallel_world_size,
                )
                set_random_seed(TensorParallelLayerTest.SEED + 1)
                input_tensor = torch.randint(0, TensorParallelLayerTest.VOCAB_SIZE, (TensorParallelLayerTest.BATCH_SIZE, TensorParallelLayerTest.SEQUENCE_LENGTH), device="cuda")
                loss_weight = torch.randn((TensorParallelLayerTest.BATCH_SIZE, TensorParallelLayerTest.SEQUENCE_LENGTH, TensorParallelLayerTest.HIDDEN_SIZE), device="cuda")

                set_random_seed(TensorParallelLayerTest.SEED)
                embedding_torch = nn.Embedding(TensorParallelLayerTest.VOCAB_SIZE, TensorParallelLayerTest.HIDDEN_SIZE).cuda()
                output_torch = embedding_torch(input_tensor)
                loss_torch = torch.mul(output_torch, loss_weight).sum()
                loss_torch.backward()

                set_random_seed(TensorParallelLayerTest.SEED)
                embedding_vocab_parallel = layers.VocabParallelEmbedding(TensorParallelLayerTest.VOCAB_SIZE, TensorParallelLayerTest.HIDDEN_SIZE, init_method=nn.init.normal_).cuda()
                output_vocab_parallel = embedding_vocab_parallel(input_tensor)
                loss_vocab_parallel = torch.mul(output_vocab_parallel, loss_weight).sum()
                loss_vocab_parallel.backward()

                self.assertEqual(output_torch, output_vocab_parallel)
                self.assertEqual(loss_torch, loss_vocab_parallel)

                splitted_weight_torch = torch.split(embedding_torch.weight.grad, TensorParallelLayerTest.VOCAB_SIZE // tensor_model_parallel_world_size, 0)[parallel_state.get_tensor_model_parallel_rank()]
                self.assertEqual(splitted_weight_torch, embedding_vocab_parallel.weight.grad)

                parallel_state.destroy_model_parallel()

    def _affine_weight_init_test_impl(self, is_column_parallel: bool) -> None:
        dim = int(not is_column_parallel)
        for init_device, tensor_model_parallel_world_size in itertools.product(("cpu", "gpu"), range(1, self.world_size + 1)):
            if self.world_size % tensor_model_parallel_world_size:
                continue
            with self.subTest(
                init_device=init_device,
                tensor_model_parallel_world_size=tensor_model_parallel_world_size
            ):
                input_size: int = TensorParallelLayerTest.INPUT_SIZE_COEFF * tensor_model_parallel_world_size
                output_size: int = TensorParallelLayerTest.OUTPUT_SIZE_COEFF * tensor_model_parallel_world_size

                weight_shape = (TensorParallelLayerTest.OUTPUT_SIZE_COEFF, input_size) if is_column_parallel else (output_size, TensorParallelLayerTest.INPUT_SIZE_COEFF)
                weight = torch.empty(weight_shape)
                set_random_seed(TensorParallelLayerTest.SEED)

                sharding_dim_size = TensorParallelLayerTest.OUTPUT_SIZE_COEFF if is_column_parallel else TensorParallelLayerTest.INPUT_SIZE_COEFF

                if init_device == "cpu":
                    layers._initialize_affine_weight_cpu(weight, output_size, input_size, sharding_dim_size, dim, nn.init.normal_, params_dtype=torch.float32)
                else:
                    layers._initialize_affine_weight_gpu(weight, torch.nn.init.normal_, dim)
                # Target
                set_random_seed(TensorParallelLayerTest.SEED)
                main_weight = torch.empty(output_size, input_size)
                nn.init.normal_(main_weight)
                curr_weight = torch.split(main_weight, sharding_dim_size, dim=dim)[parallel_state.get_tensor_model_parallel_rank()]
                self.assertEqual(curr_weight, weight)
                parallel_state.destroy_model_parallel()

    def test_affine_weight_init_column_parallel(self) -> None:
        self._affine_weight_init_test_impl(is_column_parallel=True)

    def test_affine_weight_init_row_parallel(self) -> None:
        self._affine_weight_init_test_impl(is_column_parallel=False)

    def test_row_parallel_linear(self) -> None:
        for tensor_model_parallel_world_size in range(1, self.world_size + 1):
            if self.world_size % tensor_model_parallel_world_size:
                continue
            with self.subTest(
                tensor_model_parallel_world_size=tensor_model_parallel_world_size
            ):
                parallel_state.initialize_model_parallel(tensor_model_parallel_size_=tensor_model_parallel_world_size)

                input_size: int = TensorParallelLayerTest.INPUT_SIZE_COEFF * tensor_model_parallel_world_size
                output_size: int = TensorParallelLayerTest.OUTPUT_SIZE_COEFF * tensor_model_parallel_world_size

                linear_layer = layers.RowParallelLinear(input_size, output_size, keep_master_weight_for_test=True, params_dtype=torch.float32, use_cpu_initialization=True).cuda()
                loss_weight = torch.randn((TensorParallelLayerTest.BATCH_SIZE, output_size)).cuda()

                # Forward and backward
                input_tensor = torch.randn(TensorParallelLayerTest.BATCH_SIZE, input_size, requires_grad=True).cuda()
                output, _ = linear_layer(input_tensor)
                loss = torch.mul(output, loss_weight).sum()
                loss.backward()

                parallel_state.destroy_model_parallel()


if __name__ == "__main__":
    common_utils.run_tests()