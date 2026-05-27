from pathlib import Path

import torch

_loaded_libraries = set()


def load_custom_op_library(extension_name, anchor_file):
    base_dir = Path(anchor_file).resolve().parent
    search_dirs = [base_dir, base_dir.parent, base_dir.parent.parent]
    candidates = sorted(
        {
            candidate
            for directory in search_dirs
            for candidate in directory.glob(f"{extension_name}*.so")
        },
        key=lambda path: (".cpython-" in path.name, path.name),
    )
    if not candidates:
        raise ImportError(f"Could not find shared library for {extension_name!r} next to {anchor_file}")

    library = str(candidates[0])
    if library not in _loaded_libraries:
        torch.ops.load_library(library)
        _loaded_libraries.add(library)
    return library


def scalar_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def scalar_int(value):
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def tensor_list_arg(value):
    return [list(tensors) for tensors in value]
