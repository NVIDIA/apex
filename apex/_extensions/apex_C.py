import torch


def flatten(tensors):
    tensors = list(tensors)
    if len(tensors) == 0:
        return torch.tensor([])
    return torch.cat([tensor.contiguous().view(-1) for tensor in tensors], dim=0)


def unflatten(flat, tensors):
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return outputs
