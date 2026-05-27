import torch

from apex._custom_ops import load_custom_op_library, scalar_int


load_custom_op_library("_permutation_search_cuda", __file__)


def _as_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    return torch.from_numpy(value)


def _op(op_name):
    return getattr(torch.ops.apex, op_name)


def sum_after_2_to_4(matrix, rows, cols, start_col, end_col, blocks, threads, output):
    return _op("permutation_search_sum_after_2_to_4")(
        _as_tensor(matrix),
        scalar_int(rows),
        scalar_int(cols),
        scalar_int(start_col),
        scalar_int(end_col),
        scalar_int(blocks),
        scalar_int(threads),
        _as_tensor(output),
    )


def build_permute_map(matrix, rows, cols, stripes, num_groups, group_width, permutations, perm_length, improvements, best_indices):
    return _op("permutation_search_build_permute_map")(
        _as_tensor(matrix),
        scalar_int(rows),
        scalar_int(cols),
        _as_tensor(stripes),
        scalar_int(num_groups),
        scalar_int(group_width),
        _as_tensor(permutations),
        scalar_int(perm_length),
        _as_tensor(improvements),
        _as_tensor(best_indices),
    )


def check_permutations(
    matrix,
    rows,
    cols,
    stripe_groups,
    group_width,
    num_groups,
    permutations,
    num_permutations,
    improvement,
    permutation,
):
    return _op("permutation_search_check_permutations")(
        _as_tensor(matrix),
        scalar_int(rows),
        scalar_int(cols),
        _as_tensor(stripe_groups),
        scalar_int(group_width),
        scalar_int(num_groups),
        _as_tensor(permutations),
        scalar_int(num_permutations),
        _as_tensor(improvement),
        _as_tensor(permutation),
    )


def build_swap_map(matrix, rows, cols, stripe_pairs, output):
    return _op("permutation_search_build_swap_map")(
        _as_tensor(matrix), scalar_int(rows), scalar_int(cols), _as_tensor(stripe_pairs), _as_tensor(output)
    )
