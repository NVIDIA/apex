from typing import Sequence, Optional, Union, List

import torch

from apex.transformer.enums import ModelType
from apex.transformer import parallel_state
from apex.transformer.pipeline_parallel import p2p_communication


def get_tensor_shapes(
    rank: int,
    model_type: ModelType,
    *,
    tensor_shape: Union[List[int], torch.Size],
    decoder_sequence_length: Optional[int],
) -> Sequence[Sequence[int]]:
    # Determine right tensor sizes (based on position of rank with respect to split
    # rank) and model size.
    # Send two tensors if model is T5 and rank is in decoder stage:
    #     first tensor is decoder (pre-transpose),
    #     second tensor is encoder (post-transpose).
    # If model is T5 and rank is at the boundary:
    #     send one tensor (post-transpose from encoder).
    # Otherwise, send one tensor (pre-transpose).
    assert len(tensor_shape) == 3, "`tensor_shape` should be [sequence_length, micro_batch_size, hidden_size]"
    sequence_length, micro_batch_size, hidden_size = tensor_shape
    tensor_shapes = []
    if model_type == ModelType.encoder_and_decoder:
        if decoder_sequence_length is None:
            raise ValueError("`decoder_sequence_length` is required for `ModelType.encoder_and_decoder`")
        if parallel_state.is_pipeline_stage_before_split(rank):
            # If next rank is after split, then need transpose for encoder_hidden_state.
            if parallel_state.is_pipeline_stage_before_split(rank + 1):
                tensor_shapes.append((sequence_length, micro_batch_size, hidden_size))
            else:
                tensor_shapes.append((micro_batch_size, sequence_length, hidden_size))
        else:
            tensor_shapes.append((decoder_sequence_length, micro_batch_size, hidden_size))
            tensor_shapes.append((micro_batch_size, sequence_length, hidden_size))
    else:
        tensor_shapes.append((sequence_length, micro_batch_size, hidden_size))
    return tensor_shapes


def recv_forward(tensor_shapes: List[Union[None, List[int]]],) -> List[Union[None, torch.Tensor]]:
    input_tensors = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensors.append(p2p_communication.recv_forward(tensor_shape))
    return input_tensors


def recv_backward(tensor_shapes: List[Union[None, List[int]]],) -> List[Union[None, torch.Tensor]]:
    output_tensor_grads = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grads.append(p2p_communication.recv_backward(tensor_shape))
    return output_tensor_grads


def send_forward(
    output_tensors: Union[torch.Tensor, List[Union[None, torch.Tensor]]], tensor_shapes: List[Union[None, List[int]]],
) -> None:
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_forward(output_tensor, tensor_shape)


def send_backward(
    input_tensor_grads: Union[torch.Tensor, List[Union[None, torch.Tensor]]],
    tensor_shapes: List[Union[None, List[int]]],
) -> None:
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_backward(input_tensor_grad, tensor_shape)


def send_forward_recv_backward(
    output_tensors: Union[torch.Tensor, List[Union[None, torch.Tensor]]], tensor_shapes: List[Union[None, List[int]]],
) -> List[Union[None, torch.Tensor]]:
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    output_tensor_grads = []
    for (output_tensor, tensor_shape) in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            output_tensor_grads.append(None)
            continue
        output_tensor_grad = p2p_communication.send_forward_recv_backward(output_tensor, tensor_shape)
        output_tensor_grads.append(output_tensor_grad)
    return output_tensor_grads


def send_backward_recv_forward(
    input_tensor_grads: Union[torch.Tensor, List[Union[None, torch.Tensor]]],
    tensor_shapes: List[Union[None, List[int]]],
) -> List[Union[None, torch.Tensor]]:
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    input_tensors = []
    for (input_tensor_grad, tensor_shape) in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            input_tensors.append(None)
            continue
        input_tensor = p2p_communication.send_backward_recv_forward(input_tensor_grad, tensor_shape)
        input_tensors.append(input_tensor)
    return input_tensors
