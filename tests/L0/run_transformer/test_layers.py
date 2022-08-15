import logging
import unittest
import typing

import torch
import torch.nn as nn
from torch.testing._internal import common_utils

from apex.transformer import parallel_state
from apex.transformer.tensor_parallel import layers
from apex.transformer.testing.commons import set_random_seed
from apex.transformer.testing.distributed_test_base import NcclDistributedTestBase
from apex.transformer.testing.distributed_test_base import UccDistributedTestBase


logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("apex").setLevel(logging.WARNING)


# N.B.(mkozuki): Disable TF32 matrix multiply.
# Matrices used in this test are so small that TF32 matmul
# can be less precise so that `self.assertEqual` raises.
torch.backends.cuda.matmul.allow_tf32 = False


class TensorParallelLayerTestBase:

    BATCH_SIZE: int = 8
    SEQUENCE_LENGTH: int = 128
    VOCAB_SIZE: int = 1024
    HIDDEN_SIZE: int = 256
    INPUT_SIZE_COEFF: int = 256
    OUTPUT_SIZE_COEFF: int = 256
    SEED: int = 123456

    @property
    def tensor_shape(self) -> typing.Sequence[int]:
        return [self.SEQUENCE_LENGTH, self.BATCH_SIZE, self.HIDDEN_SIZE]

    @torch.no_grad()
    @unittest.skipIf(torch.cuda.device_count() < 2, "Requires >=2 GPUs")
    def test_all_gather_parity(self) -> None:
        if self.DISTRIBUTED_BACKEND == "ucc":
            self.skipTest("torch_ucc does NOT support `torch.distributed._all_gather_base` as of 2022/06/15")
        from torch.distributed.distributed_c10d import all_gather, _all_gather_base  # NOQA

        for tensor_model_parallel_world_size in range(1, self.world_size + 1):
            if self.world_size % tensor_model_parallel_world_size:
                continue
            with self.subTest(tensor_model_parallel_world_size=tensor_model_parallel_world_size):
                parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size_=tensor_model_parallel_world_size,
                )
                tensor_model_parallel_rank = parallel_state.get_tensor_model_parallel_rank()
                cur_tensor_model_device = torch.device(f"cuda:{tensor_model_parallel_rank}")
                with torch.no_grad():
                    tensor = tensor_model_parallel_rank * torch.ones(
                        self.tensor_shape, dtype=torch.float32, device=cur_tensor_model_device)
                numel = tensor.numel()
                numel_gathered = tensor_model_parallel_world_size * numel
                gathered = torch.empty(
                    torch.Size((numel_gathered,)),
                    device=cur_tensor_model_device,
                    dtype=torch.float32,
                    requires_grad=False,
                )
                chunks = [
                    gathered[i * numel : (i + 1) * numel]
                    for i in range(tensor_model_parallel_world_size)
                ]
                all_gather(chunks, tensor, group=parallel_state.get_tensor_model_parallel_group())

                gathered_for_base = torch.empty(
                    torch.Size((numel_gathered,)),
                    device=cur_tensor_model_device,
                    dtype=torch.float32,
                    requires_grad=False,
                )
                _all_gather_base(
                    gathered_for_base,
                    tensor,
                    group=parallel_state.get_tensor_model_parallel_group(),
                )

                self.assertEqual(gathered, gathered_for_base)
                parallel_state.destroy_model_parallel()

    @torch.no_grad()
    @unittest.skipIf(torch.cuda.device_count() < 2, "Requires >=2 GPUs")
    def test_reduce_scatter_parity(self) -> None:
        if self.DISTRIBUTED_BACKEND == "ucc":
            self.skipTest("torch_ucc does NOT support `torch.distributed._reduce_scatter_base` as of 2022/06/15")
        from torch.distributed.distributed_c10d import reduce_scatter, _reduce_scatter_base  # NOQA

        for tensor_model_parallel_world_size in range(2, self.world_size + 1):
            if self.world_size % tensor_model_parallel_world_size:
                continue
            with self.subTest(tensor_model_parallel_world_size=tensor_model_parallel_world_size):
                parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size_=tensor_model_parallel_world_size,
                )
                tensor_model_parallel_rank = parallel_state.get_tensor_model_parallel_rank()
                cur_tensor_model_device = torch.device(f"cuda:{tensor_model_parallel_rank}")
                with torch.no_grad():
                    input = torch.cat([
                        i * torch.ones(self.tensor_shape, dtype=torch.float32, device=cur_tensor_model_device)
                        for i in range(tensor_model_parallel_world_size)
                    ])
                    input_list = [t.clone() for t in input.chunk(tensor_model_parallel_world_size)]
                output = torch.empty(
                    self.tensor_shape,
                    device=cur_tensor_model_device,
                    dtype=torch.float32,
                    requires_grad=False,
                )
                reduce_scatter(
                    output, input_list,
                    group=parallel_state.get_tensor_model_parallel_group(),
                )

                output_for_base = torch.empty(
                    self.tensor_shape,
                    device=cur_tensor_model_device,
                    dtype=torch.float32,
                    requires_grad=False,
                )
                _reduce_scatter_base(
                    output_for_base,
                    input,
                    group=parallel_state.get_tensor_model_parallel_group(),
                )

                self.assertEqual(output, output_for_base)
                self.assertEqual(input, torch.cat(input_list))
                parallel_state.destroy_model_parallel()

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
                set_random_seed(self.SEED + 1)
                input_tensor = torch.randint(
                    0,
                    self.VOCAB_SIZE,
                    (
                        self.BATCH_SIZE,
                        self.SEQUENCE_LENGTH,
                    ),
                    device="cuda",
                )
                loss_weight = torch.randn(
                    (
                        self.BATCH_SIZE,
                        self.SEQUENCE_LENGTH,
                        self.HIDDEN_SIZE,
                    ),
                    device="cuda",
                )

                set_random_seed(self.SEED)
                embedding_torch = nn.Embedding(
                    self.VOCAB_SIZE,
                    self.HIDDEN_SIZE,
                ).cuda()
                output_torch = embedding_torch(input_tensor)
                loss_torch = torch.mul(output_torch, loss_weight).sum()
                loss_torch.backward()

                # N.B.(mkozuki): With affine weight initialization on GPU,
                # it's super difficult to keep the consistency with nn.Embedding.
                # Thus, turning on `use_cpu_initialization`.
                set_random_seed(self.SEED)
                embedding_vocab_parallel = layers.VocabParallelEmbedding(
                    self.VOCAB_SIZE,
                    self.HIDDEN_SIZE,
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
                    self.VOCAB_SIZE
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
                input_size: int = self.INPUT_SIZE_COEFF * tensor_model_parallel_world_size
                output_size: int = self.OUTPUT_SIZE_COEFF * tensor_model_parallel_world_size

                weight_shape = (
                    (self.OUTPUT_SIZE_COEFF, input_size)
                    if is_column_parallel
                    else (output_size, self.INPUT_SIZE_COEFF)
                )
                weight = torch.empty(weight_shape)
                set_random_seed(self.SEED)

                sharding_dim_size = (
                    self.OUTPUT_SIZE_COEFF
                    if is_column_parallel
                    else self.INPUT_SIZE_COEFF
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
                set_random_seed(self.SEED)
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
        self._row_parallel_linear_test_impl(False, False, False)

    def test_row_parallel_linear_gradient_accumulation_fusion(self) -> None:
        self._row_parallel_linear_test_impl(True, False, False)

    def test_row_parallel_linear_gradient_accumulation_fusion_in_fp16(self) -> None:
        self._row_parallel_linear_test_impl(True, True, False)

    @unittest.skipIf(torch.cuda.device_count() < 2, "Sequence Parallel requires >=2 GPUs")
    def test_row_parallel_linear_sequence_parallel(self) -> None:
        self._row_parallel_linear_test_impl(False, False, True)

    # TODO(mkozuki): Merge this with `_column_parallel_linear_test_impl`
    # Note that `input_is_parallel` is unique to `RowParallelLinear` which could make the merge complicated.
    def _row_parallel_linear_test_impl(
        self,
        gradient_accumulation_fusion: bool,
        accumulation_in_fp16: bool,
        sequence_parallel_enabled: bool,
    ) -> None:
        tensor_shape = (
            self.SEQUENCE_LENGTH,
            self.BATCH_SIZE,
            self.HIDDEN_SIZE,
        )
        for tensor_model_parallel_world_size in range(
            1 + int(sequence_parallel_enabled), self.world_size + 1
        ):
            if self.world_size % tensor_model_parallel_world_size:
                continue
            with self.subTest(
                tensor_model_parallel_world_size=tensor_model_parallel_world_size,
            ):
                parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size_=tensor_model_parallel_world_size,
                )
                set_random_seed(self.SEED)

                linear = layers.RowParallelLinear(
                    self.HIDDEN_SIZE,
                    self.HIDDEN_SIZE,
                    keep_master_weight_for_test=True,
                    params_dtype=torch.float32,
                    use_cpu_initialization=True,
                    gradient_accumulation_fusion=gradient_accumulation_fusion,
                    accumulation_in_fp16=accumulation_in_fp16,
                    sequence_parallel_enabled=sequence_parallel_enabled,
                    # n.b.(mkozuki): RowParallelLinear is constructed with `input_is_parallel=True`
                    # by default, e.g. https://github.com/NVIDIA/NeMo/blob/782b4e1652aaa43c8be390d9\
                    # db0dc89544afa080/nemo/collections/nlp/modules/common/megatron/transformer.py#L204
                    input_is_parallel=True,
                ).cuda()
                if accumulation_in_fp16:
                    linear = linear.half()
                # Simulate the situation where fusion of weight grad calculation and gradient accumulation is enabled.
                if gradient_accumulation_fusion:
                    with torch.no_grad():
                        linear.weight.main_grad = torch.zeros_like(linear.weight)

                with torch.no_grad():
                    orig_input_tensor = torch.randn(tensor_shape, requires_grad=True, device="cuda")
                    orig_loss_weight = torch.randn(tensor_shape, device="cuda")
                    input_tensor = orig_input_tensor.chunk(
                        chunks=tensor_model_parallel_world_size,
                        dim=2,
                    )[parallel_state.get_tensor_model_parallel_rank()].contiguous()
                    if sequence_parallel_enabled:
                        loss_weight = orig_loss_weight.chunk(
                            chunks=tensor_model_parallel_world_size,
                            dim=0,
                        )[parallel_state.get_tensor_model_parallel_rank()]
                    else:
                        loss_weight = orig_loss_weight
                    if accumulation_in_fp16:
                        orig_input_tensor = orig_input_tensor.half()
                        input_tensor = input_tensor.half()
                        loss_weight = loss_weight.half()
                input_tensor.requires_grad_()
                output, _ = linear(input_tensor)
                loss = torch.mul(output, loss_weight).sum()
                loss.backward()
                self.assertIsNotNone(input_tensor.grad)

                ref_linear = nn.Linear(
                    in_features=self.HIDDEN_SIZE,
                    out_features=self.HIDDEN_SIZE,
                    bias=False,
                    device="cuda",
                )
                with torch.no_grad():
                    dldy = orig_loss_weight.clone()
                    x = orig_input_tensor.clone()
                    ref_linear.weight.copy_(linear.master_weight)
                    if accumulation_in_fp16:
                        ref_linear = ref_linear.half()
                x.requires_grad_()
                expected_output = ref_linear(x)
                expected_loss = torch.mul(expected_output, dldy).sum()
                expected_loss.backward()

                if not accumulation_in_fp16:
                    if sequence_parallel_enabled:
                        self.assertEqual(
                            x=output,
                            y=expected_output.chunk(
                                chunks=tensor_model_parallel_world_size,
                                dim=0,
                            )[parallel_state.get_tensor_model_parallel_rank()],
                        )
                    else:
                        self.assertEqual(
                            x=output,
                            y=expected_output,
                        )

                grad_attr_name = "main_grad" if gradient_accumulation_fusion else "grad"
                # NOTE(mkozuki): Numerical errors seems to be enlarged by tensor model parallel.
                if tensor_model_parallel_world_size == 1:
                    self.assertEqual(
                        x=getattr(linear.weight, grad_attr_name),
                        y=ref_linear.weight.grad.chunk(
                            chunks=tensor_model_parallel_world_size,
                            dim=0,
                        )[parallel_state.get_tensor_model_parallel_rank()],
                    )

                parallel_state.destroy_model_parallel()

    def test_column_parallel_linear(self):
        self._column_parallel_linear_test_impl(False, False, False, False)

    def test_column_parallel_linear_async(self):
        self._column_parallel_linear_test_impl(True, False, False, False)

    def test_column_parallel_linear_gradient_accumulation_fusion(self):
        self._column_parallel_linear_test_impl(False, True, False, False)

    def test_column_parallel_linear_gradient_accumulation_fusion_in_fp16(self):
        self._column_parallel_linear_test_impl(False, True, True, False)

    def test_column_parallel_linear_sequence_parallel(self):
        if self.DISTRIBUTED_BACKEND == "ucc":
            self.skipTest("Backward's reduce_scatter fails. as of 2022/06/15")
        self._column_parallel_linear_test_impl(False, False, False, True)

    @unittest.skipIf(torch.cuda.device_count() < 2, "Sequence Parallel requires >= 2 GPUs")
    def test_column_parallel_linear_exception(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "`async_tensor_model_parallel_allreduce` and `sequence_parallel_enabled` cannot be enabled at the same time.",
        ):
            self._column_parallel_linear_test_impl(True, False, False, True)

    def _column_parallel_linear_test_impl(
        self,
        async_tensor_model_parallel_allreduce: bool,
        gradient_accumulation_fusion: bool,
        accumulation_in_fp16: bool,
        sequence_parallel_enabled: bool,
    ):
        for tensor_model_parallel_world_size in range(1, self.world_size + 1):
            if async_tensor_model_parallel_allreduce and sequence_parallel_enabled:
                if tensor_model_parallel_world_size == 1:
                    continue
            with self.subTest(tensor_model_parallel_world_size=tensor_model_parallel_world_size):
                if self.world_size % tensor_model_parallel_world_size:
                    continue
                parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size_=tensor_model_parallel_world_size,
                )

                input_tensor_shape = self.tensor_shape
                expected_output_shape = self.tensor_shape
                # When sequence parallel, `gather_output` is disabled, i.e.,
                # output of matmul isn't gathered in dimension of feature/hidden (last dim).
                if sequence_parallel_enabled:
                    expected_output_shape[-1] //= tensor_model_parallel_world_size

                # tensor's shape is [sequence length, batch size, hidden size]
                set_random_seed(self.SEED)
                linear = layers.ColumnParallelLinear(
                    self.HIDDEN_SIZE,
                    self.HIDDEN_SIZE,
                    bias=False,
                    keep_master_weight_for_test=True,
                    params_dtype=torch.float32,
                    use_cpu_initialization=True,
                    gather_output=not sequence_parallel_enabled,
                    no_async_tensor_model_parallel_allreduce=not async_tensor_model_parallel_allreduce,
                    gradient_accumulation_fusion=gradient_accumulation_fusion,
                    accumulation_in_fp16=accumulation_in_fp16,
                    sequence_parallel_enabled=sequence_parallel_enabled,
                ).cuda()
                if accumulation_in_fp16:
                    linear = linear.half()

                # Simulate the situation where fusion of weight grad calculation and gradient accumulation happens.
                if gradient_accumulation_fusion:
                    with torch.no_grad():
                        linear.weight.main_grad = torch.zeros_like(linear.weight)

                orig_input_tensor = torch.randn(input_tensor_shape, device="cuda", requires_grad=True)
                if accumulation_in_fp16:
                    orig_input_tensor = orig_input_tensor.half()
                if sequence_parallel_enabled:
                    input_tensor = list(
                        orig_input_tensor.chunk(tensor_model_parallel_world_size, dim=0)
                    )[parallel_state.get_tensor_model_parallel_rank()]
                else:
                    input_tensor = orig_input_tensor
                output, _ = linear(input_tensor)
                # The order of dimension is expected to be (sequence, batch, hidden)
                self.assertEqual(output.shape, expected_output_shape)

                orig_loss_weight = torch.randn(input_tensor_shape, device="cuda")
                if accumulation_in_fp16:
                    orig_loss_weight = orig_loss_weight.half()
                if sequence_parallel_enabled:
                    loss_weight = orig_loss_weight.chunk(
                        tensor_model_parallel_world_size, dim=2,
                    )[parallel_state.get_tensor_model_parallel_rank()]
                else:
                    loss_weight = orig_loss_weight
                loss = torch.mul(output, loss_weight).sum()
                loss.backward()

                with torch.no_grad():
                    dldy = orig_loss_weight.clone()
                    x = orig_input_tensor.clone()
                    ref_linear = nn.Linear(
                        in_features=self.HIDDEN_SIZE,
                        out_features=self.HIDDEN_SIZE,
                        bias=False,
                        device="cuda",
                    )
                    if accumulation_in_fp16:
                        ref_linear = ref_linear.half()
                    # NOTE(mkozuki): `master_weight` is available because `keep_master_weight_for_test` is set.
                    ref_linear.weight.copy_(linear.master_weight)
                x.requires_grad_()
                expected_output = ref_linear(x)
                if sequence_parallel_enabled:
                    chunk = expected_output.chunk(
                        tensor_model_parallel_world_size,
                        dim=2,
                    )[parallel_state.get_tensor_model_parallel_rank()]
                    self.assertEqual(
                        x=output,
                        y=chunk,
                    )
                else:
                    self.assertEqual(
                        x=output,
                        y=expected_output,
                    )

                expected_loss = torch.mul(expected_output, dldy).sum()
                expected_loss.backward()
                grad_attr_name = "main_grad" if gradient_accumulation_fusion else "grad"
                # NOTE(mkozuki): Numerical errors seems to be enlarged by tensor model parallel.
                if tensor_model_parallel_world_size == 1:
                    self.assertEqual(
                        x=getattr(linear.weight, grad_attr_name),
                        y=ref_linear.weight.grad.chunk(
                            chunks=tensor_model_parallel_world_size,
                            dim=0,
                        )[parallel_state.get_tensor_model_parallel_rank()],
                    )

                parallel_state.destroy_model_parallel()


class NcclTensorParallelLayerTest(TensorParallelLayerTestBase, NcclDistributedTestBase):
    pass


class UccTensorParallelLayerTest(TensorParallelLayerTestBase, UccDistributedTestBase):
    pass


if __name__ == "__main__":
    common_utils.run_tests()
