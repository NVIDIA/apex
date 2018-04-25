import torch

def get_norm_shape(p, dim):
    if dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return output_size
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return output_size
    return None

def pt_norm(p, dim):
    """Computes the norm over all dimensions except dim"""
    if dim is None:
        return p.norm()
    elif dim == 0:
        return p.contiguous().view(p.size(0), -1).norm(2,dim=1).view(*get_norm_shape(p, dim))
    elif dim == p.dim() - 1:
        return p.contiguous().view(-1, p.size(-1)).norm(2,dim=0).view(*get_norm_shape(p, dim))
    return pt_norm(p.transpose(0, dim), 0).transpose(0, dim)
