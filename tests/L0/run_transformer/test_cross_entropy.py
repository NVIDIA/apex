import logging
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.testing._internal import common_utils

logging.getLogger("torch").setLevel(logging.WARNING)

from apex.transformer import parallel_state
from apex.transformer import tensor_parallel
from apex.transformer.tensor_parallel import cross_entropy
from apex.transformer.testing.commons import set_random_seed, IdentityLayer
from apex.transformer.testing.distributed_test_base import NcclDistributedTestBase
from apex.transformer.testing.distributed_test_base import UccDistributedTestBase

logging.getLogger("apex").setLevel(logging.WARNING)


def torch_cross_entropy(
    batch_size: int, seq_length: int, vocab_size: int, logits_scale: float, seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    set_random_seed(seed)
    identity = IdentityLayer(
        (batch_size, seq_length, vocab_size), scale=logits_scale
    ).cuda()
    logits = identity()
    target = torch.cuda.LongTensor(size=(batch_size, seq_length)).random_(0, vocab_size)
    loss = (
        F.cross_entropy(
            logits.view(-1, logits.size()[-1]), target.view(-1), reduction="none"
        )
        .view_as(target)
        .mean()
    )
    loss.backward()
    return loss, identity.weight.grad


def tensor_sharded_cross_entropy(
    batch_size, seq_length, vocab_size, logits_scale, seed
):
    set_random_seed(seed)
    identity = IdentityLayer(
        (batch_size, seq_length, vocab_size), scale=logits_scale
    ).cuda()
    logits = identity()
    logits_parallel = tensor_parallel.scatter_to_tensor_model_parallel_region(logits)
    target = torch.cuda.LongTensor(size=(batch_size, seq_length)).random_(0, vocab_size)
    logits_parallel_ = logits_parallel.clone().detach()
    loss = cross_entropy.vocab_parallel_cross_entropy(logits_parallel, target).mean()
    loss.backward()
    # check for mutation
    assert torch.equal(logits_parallel_, logits_parallel)
    return loss, identity.weight.grad


class VocabParallelCrossEntropyTestBase:
    def test_cross_entropy(self):
        batch_size, sequence_length, vocab_size_per_partition = 13, 17, 11
        logits_scale = 1000.0
        seed = 1234
        for tensor_model_parallel_world_size in range(1, self.world_size + 1):
            if self.world_size % tensor_model_parallel_world_size:
                continue
            with self.subTest(
                tensor_model_parallel_world_size=tensor_model_parallel_world_size
            ):
                parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size_=tensor_model_parallel_world_size,
                )
                vocab_size = vocab_size_per_partition * tensor_model_parallel_world_size
                loss_torch, grad_torch = torch_cross_entropy(
                    batch_size, sequence_length, vocab_size, logits_scale, seed
                )
                (
                    loss_tensor_parallel,
                    grad_tensor_parallel,
                ) = tensor_sharded_cross_entropy(
                    batch_size, sequence_length, vocab_size, logits_scale, seed
                )

                self.assertEqual(loss_torch, loss_tensor_parallel)
                self.assertEqual(grad_torch, grad_tensor_parallel)

                parallel_state.destroy_model_parallel()


class NcclVocabParallelCrossEntropyTest(VocabParallelCrossEntropyTestBase, NcclDistributedTestBase): pass
class UccVocabParallelCrossEntropyTest(VocabParallelCrossEntropyTestBase, UccDistributedTestBase): pass


if __name__ == "__main__":
    common_utils.run_tests()
