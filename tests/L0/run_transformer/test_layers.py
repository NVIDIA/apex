import logging

import torch
import torch.nn as nn
from torch.testing._internal import common_utils

logging.getLogger("torch").setLevel(logging.WARNING)

from apex.transformer import parallel_state
from apex.transformer.tensor_parallel import layers
from apex.transformer.testing.commons import set_random_seed
from apex.transformer.testing.distributed_test_base import DistributedTestBase

logging.getLogger("apex").setLevel(logging.WARNING)


# N.B. (mkozuki): Disable TF32 matrix multiply.
# Matrices used in this test are so small that TF32 matmul
# can be less precise so that `self.assertEqual` raises.
torch.backends.cuda.matmul.allow_tf32 = False


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
                input_tensor = torch.randint(
                    0,
                    TensorParallelLayerTest.VOCAB_SIZE,
                    (
                        TensorParallelLayerTest.BATCH_SIZE,
                        TensorParallelLayerTest.SEQUENCE_LENGTH,
                    ),
                    device="cuda",
                )
                loss_weight = torch.randn(
                    (
                        TensorParallelLayerTest.BATCH_SIZE,
                        TensorParallelLayerTest.SEQUENCE_LENGTH,
                        TensorParallelLayerTest.HIDDEN_SIZE,
                    ),
                    device="cuda",
                )

                set_random_seed(TensorParallelLayerTest.SEED)
                embedding_torch = nn.Embedding(
                    TensorParallelLayerTest.VOCAB_SIZE,
                    TensorParallelLayerTest.HIDDEN_SIZE,
                ).cuda()
                output_torch = embedding_torch(input_tensor)
                loss_torch = torch.mul(output_torch, loss_weight).sum()
                loss_torch.backward()

                # N.B. (mkozuki): With affine weight initialization on GPU,
                # it's super difficult to keep the consistency with nn.Embedding.
                # Thus, turning on `use_cpu_initialization`.
                set_random_seed(TensorParallelLayerTest.SEED)
                embedding_vocab_parallel = layers.VocabParallelEmbedding(
                    TensorParallelLayerTest.VOCAB_SIZE,
                    TensorParallelLayerTest.HIDDEN_SIZE,
                    init_method=nn.init.normal_,
                    use_cpu_initialization=True,
                ).cuda()
                output_vocab_parallel = embedding_vocab_parallel(input_tensor)
                loss_vocab_parallel = torch.mul(
                    output_vocab_parallel, loss_weight
                ).sum()
                loss_vocab_parallel.backward()

                self.assertEqual(output_torch, output_vocab_parallel)
                self.assertEqual(loss_torch, loss_vocab_parallel)

                splitted_weight_torch = torch.split(
                    embedding_torch.weight.grad,
                    TensorParallelLayerTest.VOCAB_SIZE
                    // tensor_model_parallel_world_size,
                    0,
                )[parallel_state.get_tensor_model_parallel_rank()]
                self.assertEqual(
                    splitted_weight_torch, embedding_vocab_parallel.weight.grad
                )

                parallel_state.destroy_model_parallel()

    def _affine_weight_init_test_impl(
        self, init_device: str, is_column_parallel: bool
    ) -> None:
        dim = int(not is_column_parallel)
        for tensor_model_parallel_world_size in range(1, self.world_size + 1):
            if self.world_size % tensor_model_parallel_world_size:
                continue
            with self.subTest(
                tensor_model_parallel_world_size=tensor_model_parallel_world_size
            ):
                parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size_=tensor_model_parallel_world_size
                )
                input_size: int = TensorParallelLayerTest.INPUT_SIZE_COEFF * tensor_model_parallel_world_size
                output_size: int = TensorParallelLayerTest.OUTPUT_SIZE_COEFF * tensor_model_parallel_world_size

                weight_shape = (
                    (TensorParallelLayerTest.OUTPUT_SIZE_COEFF, input_size)
                    if is_column_parallel
                    else (output_size, TensorParallelLayerTest.INPUT_SIZE_COEFF)
                )
                weight = torch.empty(weight_shape)
                set_random_seed(TensorParallelLayerTest.SEED)

                sharding_dim_size = (
                    TensorParallelLayerTest.OUTPUT_SIZE_COEFF
                    if is_column_parallel
                    else TensorParallelLayerTest.INPUT_SIZE_COEFF
                )

                if init_device == "cpu":
                    layers._initialize_affine_weight_cpu(
                        weight,
                        output_size,
                        input_size,
                        sharding_dim_size,
                        dim,
                        nn.init.normal_,
                        params_dtype=torch.float32,
                    )
                else:
                    layers._initialize_affine_weight_gpu(
                        weight, torch.nn.init.normal_, dim
                    )
                # Target
                set_random_seed(TensorParallelLayerTest.SEED)
                if init_device == "cpu":
                    main_weight = torch.empty(output_size, input_size)
                    nn.init.normal_(main_weight)
                    curr_weight = torch.split(main_weight, sharding_dim_size, dim=dim)[
                        parallel_state.get_tensor_model_parallel_rank()
                    ]
                else:
                    curr_weight = torch.empty(*weight_shape)
                    nn.init.normal_(curr_weight)
                self.assertEqual(curr_weight, weight)
                parallel_state.destroy_model_parallel()

    def test_affine_weight_init_column_parallel_cpu(self) -> None:
        self._affine_weight_init_test_impl(init_device="cpu", is_column_parallel=True)

    def test_affine_weight_init_column_parallel_gpu(self) -> None:
        self._affine_weight_init_test_impl(init_device="gpu", is_column_parallel=True)

    def test_affine_weight_init_row_parallel_cpu(self) -> None:
        self._affine_weight_init_test_impl(init_device="cpu", is_column_parallel=False)

    def test_affine_weight_init_row_parallel_gpu(self) -> None:
        self._affine_weight_init_test_impl(init_device="gpu", is_column_parallel=False)

    def test_row_parallel_linear(self) -> None:
        for tensor_model_parallel_world_size in range(1, self.world_size + 1):
            if self.world_size % tensor_model_parallel_world_size:
                continue
            with self.subTest(
                tensor_model_parallel_world_size=tensor_model_parallel_world_size
            ):
                parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size_=tensor_model_parallel_world_size
                )

                input_size: int = TensorParallelLayerTest.INPUT_SIZE_COEFF * tensor_model_parallel_world_size
                output_size: int = TensorParallelLayerTest.OUTPUT_SIZE_COEFF * tensor_model_parallel_world_size

                set_random_seed(TensorParallelLayerTest.SEED)
                linear_layer = layers.RowParallelLinear(
                    input_size,
                    output_size,
                    keep_master_weight_for_test=True,
                    params_dtype=torch.float32,
                    use_cpu_initialization=True,
                ).cuda()
                loss_weight = torch.randn(
                    (TensorParallelLayerTest.BATCH_SIZE, output_size)
                ).cuda()

                # Forward and backward
                input_tensor = torch.randn(
                    TensorParallelLayerTest.BATCH_SIZE, input_size, requires_grad=True
                ).cuda()
                input_tensor.retain_grad()
                output, _ = linear_layer(input_tensor)
                loss = torch.mul(output, loss_weight).sum()
                loss.backward()
                self.assertIsNotNone(input_tensor.grad)

                with torch.no_grad():
                    dldy = loss_weight.clone()
                    x = input_tensor.clone()
                    a = linear_layer.master_weight.cuda()
                dlda = torch.matmul(dldy.t(), x)
                dldb = torch.matmul(
                    torch.ones(TensorParallelLayerTest.BATCH_SIZE, 1).cuda().t(), dldy
                ).view(-1)
                dldx = torch.matmul(dldy, a)

                with torch.no_grad():
                    curr_dlda = torch.split(
                        dlda, TensorParallelLayerTest.INPUT_SIZE_COEFF, dim=1
                    )[parallel_state.get_tensor_model_parallel_rank()].clone()
                self.assertEqual(linear_layer.weight.grad, curr_dlda)
                self.assertEqual(input_tensor.grad, dldx)
                self.assertEqual(linear_layer.bias.grad, dldb)

                parallel_state.destroy_model_parallel()

    def test_column_parallel_linear(self):
        self._column_parallel_linear_test_impl(False, False)

    def test_column_parallel_linear_no_async(self):
        self._column_parallel_linear_test_impl(True, False)

    def test_column_parallel_linear_gradient_accumulation_fusion(self):
        self._column_parallel_linear_test_impl(False, True)

    def _column_parallel_linear_test_impl(
        self,
        no_async_tensor_model_parallel_allreduce: bool,
        gradient_accumulation_fusion: bool,
    ):
        for tensor_model_parallel_world_size in range(1, self.world_size + 1):
            print(
                f"tensor_model_parallel_world_size={tensor_model_parallel_world_size}"
            )
            with self.subTest(
                tensor_model_parallel_world_size=tensor_model_parallel_world_size
            ):
                if self.world_size % tensor_model_parallel_world_size:
                    continue
                parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size_=tensor_model_parallel_world_size,
                )

                feature_size_coeff = TensorParallelLayerTest.INPUT_SIZE_COEFF
                feature_size = feature_size_coeff * tensor_model_parallel_world_size
                hidden_size = feature_size

                set_random_seed(TensorParallelLayerTest.SEED)
                input_tensor = torch.randn(
                    TensorParallelLayerTest.BATCH_SIZE,
                    hidden_size,
                    feature_size,
                    device="cuda",
                    requires_grad=True,
                )
                input_tensor.retain_grad()
                loss_weight = torch.randn(
                    (TensorParallelLayerTest.BATCH_SIZE, hidden_size, feature_size,),
                    device="cuda",
                )
                linear = layers.ColumnParallelLinear(
                    feature_size,
                    feature_size,
                    bias=False,
                    keep_master_weight_for_test=True,
                    params_dtype=torch.float32,
                    use_cpu_initialization=True,
                    no_async_tensor_model_parallel_allreduce=no_async_tensor_model_parallel_allreduce,
                    gradient_accumulation_fusion=gradient_accumulation_fusion,
                ).cuda()
                if gradient_accumulation_fusion:
                    with torch.no_grad():
                        linear.weight.main_grad = torch.randn_like(linear.weight)
                output, _ = linear(input_tensor)
                self.assertEqual(
                    output.shape,
                    (TensorParallelLayerTest.BATCH_SIZE, hidden_size, feature_size,),
                )
                loss = torch.mul(output, loss_weight).sum()
                loss.backward()

                with torch.no_grad():
                    dldy = loss_weight.clone()
                    x = input_tensor.clone()
                    a = linear.master_weight.cuda().clone()
                dldx = torch.matmul(dldy, a)
                self.assertEqual(input_tensor.grad, dldx)
                # TODO (mkozuki): Cover the other cases.
                if (
                    tensor_model_parallel_world_size == 1
                    and not gradient_accumulation_fusion
                ):
                    dlda = torch.matmul(torch.transpose(dldy, 1, 2), x).sum(dim=0)
                    curr_dlda = torch.split(dlda, feature_size_coeff, dim=0)[
                        parallel_state.get_tensor_model_parallel_rank()
                    ]
                    self.assertEqual(linear.weight.grad, curr_dlda)

                parallel_state.destroy_model_parallel()


if __name__ == "__main__":
    common_utils.run_tests()
