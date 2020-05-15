from .weight_norm import WeightNorm
from .reparameterization import Reparameterization

def apply_weight_norm(module, name='', dim=0, hook_child=True):
    r"""
    Applies weight normalization to a parameter in the given module.
    If no parameter is provided, applies weight normalization to all
    parameters in model (except 1-d vectors and scalars).

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by `name` (e.g. "weight") with two parameters: one specifying the magnitude
    (e.g. "weight_g") and one specifying the direction (e.g. "weight_v").
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.

    By default, with `dim=0`, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    `dim=None`.

    See https://arxiv.org/abs/1602.07868

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm
        hook_child (boolean, optional): adds reparameterization hook to direct parent of the 
            parameters. If False, it's added to `module` instead. Default: True

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = apply_weight_norm(nn.Linear(20, 40), name='weight')
        Linear (20 -> 40)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])

    """
    return apply_reparameterization(module, reparameterization=WeightNorm, hook_child=hook_child,
                                    name=name, dim=dim)

def remove_weight_norm(module, name='', remove_all=False):
    """
    Removes the weight normalization reparameterization of a parameter from a module.
    If no parameter is supplied then all weight norm parameterizations are removed.
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
    Example:
        >>> m = apply_weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    return remove_reparameterization(module, reparameterization=WeightNorm,
                                    name=name, remove_all=remove_all)

def apply_reparameterization(module, reparameterization=None, name='', dim=0, hook_child=True):
    """
    Applies a given weight reparameterization (such as weight normalization) to
    a parameter in the given module. If no parameter is given, applies the reparameterization
    to all parameters in model (except 1-d vectors and scalars).

    Args:
        module (nn.Module): containing module
        reparameterization (Reparameterization): reparamaterization class to apply
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to perform reparameterization op
        hook_child (boolean, optional): adds reparameterization hook to direct parent of the 
            parameters. If False, it's added to `module` instead. Default: True

    Returns:
        The original module with the reparameterization hook

    Example::

        >>> m = apply_reparameterization(nn.Linear(20, 40), WeightNorm)
        Linear (20 -> 40)

    """
    assert reparameterization is not None
    if name != '':
        Reparameterization.apply(module, name, dim, reparameterization, hook_child)
    else:
        names = list(module.state_dict().keys())
        for name in names:
            apply_reparameterization(module, reparameterization, name, dim, hook_child)
    return module

def remove_reparameterization(module, reparameterization=Reparameterization,
                                name='', remove_all=False):
    """
    Removes the given reparameterization of a parameter from a module.
    If no parameter is supplied then all reparameterizations are removed.
    Args:
        module (nn.Module): containing module
        reparameterization (Reparameterization): reparamaterization class to apply
        name (str, optional): name of weight parameter
        remove_all (bool, optional): if True, remove all reparamaterizations of given type. Default: False
    Example:
        >>> m = apply_reparameterization(nn.Linear(20, 40),WeightNorm)
        >>> remove_reparameterization(m)
    """
    if name != '' or remove_all:
        to_remove = []
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, reparameterization) and (hook.name == name or remove_all):
                hook.remove(module)
                to_remove.append(k)
        if len(to_remove) > 0:
            for k in to_remove:
                del module._forward_pre_hooks[k]
            return module
        if not remove_all:
            raise ValueError("reparameterization of '{}' not found in {}"
                             .format(name, module))
    else:
        modules = [module]+[x for x in module.modules()]
        for m in modules:
            remove_reparameterization(m, reparameterization=reparameterization, remove_all=True)
        return module
