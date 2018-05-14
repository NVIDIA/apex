import torch
import torch.nn as nn
from torch.autograd import Variable
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class tofp16(nn.Module):
    """
    Model wrapper that implements::

        def forward(self, input):
            return input.half()
    """

    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


def BN_convert_float(module):
    '''
    Designed to work with network_to_half.
    BatchNorm layers need parameters in single precision.
    Find all layers and convert them back to float. This can't
    be done with built in .apply as that function will apply
    fn to all modules, parameters, and buffers. Thus we wouldn't
    be able to guard the float conversion based on the module type.
    '''
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module


def network_to_half(network):
    """
    Convert model to half precision in a batchnorm-safe way.
    """
    return nn.Sequential(tofp16(), BN_convert_float(network.half()))


def backwards_debug_hook(grad):
    raise RuntimeError("master_params recieved a gradient in the backward pass!")

def prep_param_lists(model, flat_master=False):
    """
    Creates a list of FP32 master parameters for a given model, as in 
    `Training Neural Networks with Mixed Precision:  Real Examples`_.

    Args:
        model (torch.nn.Module): Existing Pytorch model
        flat_master (bool, optional, default=False):  Flatten the master parameters into a single tensor, as a performance optimization.
    Returns:
        A tuple (``model_params``, ``master_params``). ``model_params`` is a list of the model's parameters for later use with :func:`model_grads_to_master_grads` and :func:`master_params_to_model_params`.  ``master_params`` is a list of FP32 master gradients.  If ``flat_master=True``, ``master_params`` will be a list with one element.

    Example::

        model_params, master_params = prep_param_lists(model)

    .. warning::
        Currently, if ``flat_master=True``, all the model's parameters must be the same type.  If the model has parameters of different types, use ``flat_master=False``, or use :class:`FP16_Optimizer`.

    .. _`Training Neural Networks with Mixed Precision:  Real Examples`:
        http://on-demand.gputechconf.com/gtc/2018/video/S81012/
    """
    model_params = [param for param in model.parameters() if param.requires_grad]

    if flat_master:
        # Give the user some more useful error messages
        try:
            # flatten_dense_tensors returns a contiguous flat array.
            # http://pytorch.org/docs/master/_modules/torch/_utils.html
            master_params = _flatten_dense_tensors([param.data for param in model_params]).float()
        except:
            print("Error in prep_param_lists:  model may contain a mixture of parameters "
                      "of different types.  Use flat_master=False, or use F16_Optimizer.")
            raise
        master_params = torch.nn.Parameter(master_params)
        master_params.requires_grad = True
        # master_params.register_hook(backwards_debug_hook)
        if master_params.grad is None:
            master_params.grad = master_params.new(*master_params.size())
        return model_params, [master_params]
    else:
        master_params = [param.clone().float().detach() for param in model_params]
        for param in master_params:
            param.requires_grad = True
        return model_params, master_params


def model_grads_to_master_grads(model_params, master_params, flat_master=False):
    """
    Copy model gradients to master gradients.  

    Args:
        model_params:  List of model parameters created by :func:`prep_param_lists`.
        master_params:  List of FP32 master parameters created by :func:`prep_param_lists`.  If ``master_params`` was created with ``flat_master=True``, ``flat_master=True`` should also be supplied to :func:`model_grads_to_master_grads`.
    """
    if flat_master:
        # The flattening may incur one more deep copy than is necessary.
        master_params[0].grad.data.copy_(
            _flatten_dense_tensors([p.grad.data for p in model_params]))
    else:
        for model, master in zip(model_params, master_params):
            if model.grad is not None:
                if master.grad is None:
                    master.grad = Variable(master.data.new(*master.data.size()))
                master.grad.data.copy_(model.grad.data)
            else:
                master.grad = None


def master_params_to_model_params(model_params, master_params, flat_master=False):
    """
    Copy master parameters to model parameters.

    Args:
        model_params:  List of model parameters created by :func:`prep_param_lists`.
        master_params:  List of FP32 master parameters created by :func:`prep_param_lists`.  If ``master_params`` was created with ``flat_master=True``, ``flat_master=True`` should also be supplied to :func:`master_params_to_model_params`.
    """
    if flat_master:
        for model, master in zip(model_params, 
                                 _unflatten_dense_tensors(master_params[0].data, model_params)):
            model.data.copy_(master)
    else:
        for model, master in zip(model_params, master_params):
            model.data.copy_(master.data)

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]
