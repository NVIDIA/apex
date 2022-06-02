from typing import Sequence, Tuple, Optional

import torch

TensorListsT = Sequence[Sequence[torch.Tensor]]

def multi_tensor_scale(
    chunk_size: int, noop_flag: torch.Tensor, tensor_lists: TensorListsT, scale: float
) -> None:
    ...
def multi_tensor_sgd(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: TensorListsT,
    wd: float,
    momentum: float,
    dampening: float,
    lr: float,
    nesterov: bool,
    first_run: bool,
    wd_after_momentum: bool,
    scale: float,
) -> None: ...
def multi_tensor_axpby(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: TensorListsT,
    a: float,
    b: float,
    arg_to_check: int,
) -> None: ...
def multi_tensor_l2norm(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: TensorListsT,
    per_tensor_python: Optional[bool],
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def multi_tensor_l2norm_mp(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: TensorListsT,
    per_tensor_python: Optional[bool],
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def multi_tensor_l2norm_scale(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: TensorListsT,
    scale: float,
    per_tensor_python: Optional[bool],
) -> Tuple[torch.Tensor, torch.Tensor]: ...
def multi_tensor_lamb_stage1_cuda(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: TensorListsT,
    per_tensor_decay: torch.Tensor,
    beta1: float,
    beta2: float,
    epsilon: float,
    global_grad_norm: torch.Tensor,
    max_global_grad_norm: float,
) -> None: ...
def multi_tensor_lamb_stage2_cuda(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: TensorListsT,
    per_tensor_param_norm: float,
    per_tensor_update_norm: float,
    lr: float,
    weight_decay: float,
    use_nvlamb_python: Optional[bool],
) -> None: ...
def multi_tensor_adam(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: TensorListsT,
    lr: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    step: int,
    mode: int,
    bias_correction: int,
    weight_decay: float,
) -> None: ...
def multi_tensor_adagrad(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: TensorListsT,
    lr: float,
    epsilon: float,
    mode: int,
    weight_decay: float,
) -> None: ...
def multi_tensor_novograd(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: TensorListsT,
    lr: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    step: int,
    bias_correction: int,
    weight_decay: float,
    grad_averaging: int,
    mode: int,
    norm_type: int,
) -> None: ...
def multi_tensor_lamb(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: TensorListsT,
    lr: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    step: int,
    bias_correction: int,
    weight_decay: float,
    grad_averaging: int,
    mode: int,
    global_grad_norm: torch.Tensor,
    max_grad_norm: float,
    use_nvlamb_python: Optional[bool],
) -> bool: ...
def multi_tensor_lamb_mp(
    chunk_size: int,
    noop_flag: torch.Tensor,
    tensor_lists: TensorListsT,
    lr: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    step: int,
    bias_correction: int,
    weight_decay: float,
    grad_averaging: int,
    mode: int,
    global_grad_norm: torch.Tensor,
    max_grad_norm: float,
    use_nvlamb_python: Optional[bool],
    found_inf: torch.Tensor,
    inv_scale: torch.Tensor,
): ...
