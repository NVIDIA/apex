import torch
from torch.nn.parameter import Parameter
from ..fp16_utils import Fused_Weight_Norm
import time

from .reparameterization import Reparameterization

def _norm(p, dim):
    """Computes the norm over all dimensions except dim"""
    if dim is None:
        return p.norm()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
    return _norm(p.transpose(0, dim), 0).transpose(0, dim)

HALF_TYPES = (torch.cuda.HalfTensor, torch.HalfTensor)

class WeightNorm(Reparameterization):
    r"""
    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by `name` (e.g. "weight") with two parameters: one specifying the magnitude
    (e.g. "weight_g") and one specifying the direction (e.g. "weight_v").
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    By default, with `dim=0`, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    `dim=None`.
    """
    def compute_weight(self, module=None, name=None):
        """
        Computes weight normalized weight value to assign value to module attribute
        with name `name`.
        Arguments:
            module (nn.Module): module with weight we'd like to reparameterize
        Returns:
            w (Tensor): Tensor object containing value of reparameterized weight
        """
        if module is None:
            module = self.module
        if name is None:
            name = self.name
        module, name = Reparameterization.get_module_and_name(module, name)
        g = getattr(module, name + '_g')
        v = getattr(module, name + '_v')

        fused_weight_norm = Fused_Weight_Norm.apply
        v = v.contiguous()
        w = fused_weight_norm(v, g, self.dim)

        return w

    def reparameterize(self, name, weight, dim):
        """
        Creates Parameters v and gto be used for weight normalization
        and creates names that for attributes for the module these Parameters
        will correspond to. The parameters will be registered according to the names
        provided.
        Arguments:
            module (nn.Module): module with weight we'd like to reparameterize
            name (str, optional): name of weight parameter
            dim (int, optional): dimension over which to compute parameterization
        Returns:
            names (list, str): names of Parameters to be used for reparameterization
            params (list, Parameter): Parameters to be used for reparameterization
        """
        names = [name + '_g', name + '_v']
        params = [Parameter(_norm(weight, dim).data), Parameter(weight.data)]
        return names, params
