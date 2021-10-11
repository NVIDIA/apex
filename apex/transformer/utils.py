import torch

from apex.transformer.parallel_state import get_tensor_model_parallel_rank
from apex.transformer.parallel_state import get_tensor_model_parallel_world_size


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_into_1d_equal_chunks(tensor):
    """Break a tensor into equal 1D chunks."""
    data = tensor.view(-1)
    partition_size = torch.numel(data) // get_tensor_model_parallel_world_size()
    start_index = partition_size * get_tensor_model_parallel_rank()
    end_index = start_index + partition_size
    return data[start_index:end_index]
