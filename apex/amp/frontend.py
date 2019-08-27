import torch
from ._initialize import _initialize
from ._amp_state import _amp_state, warn_or_err, maybe_print
from collections import OrderedDict


class Properties(object):
    """
    This class has two purposes: to establish a set of default properties,
    and to route setting of these attributes through __setattr__ so that (in theory)
    they can be checked for consistency with other existing args.
    """
    def __init__(self):
        self.options = {
            "enabled" : False,
            "opt_level" : None,
            "cast_model_type" : None,
            "patch_torch_functions" : False,
            "keep_batchnorm_fp32" : None,
            "master_weights" : None,
            "loss_scale" : 1.0,
            # Reserved for future functionality
            # "fused_optimizer" : False,
            # "enable_ddp_interop" : False,
            }

    """
    This function allows updating several options at a time without routing through
    __setattr__ checks, to avoid "you can't get there from here" scenarios.
    Currently not intended to be exposed; users are expected to select an opt_level
    and apply consistent modifications.
    """
    def _update_options_dict(self, new_options):
        for k, v in new_options:
            if k in self.options:
                self.options[k] = v
            else:
                raise ValueError("Tried to set unexpected option {}".format(k))
    """
    The members of "options" are not direct attributes of self, so access attempts
    will roll down to __getattr__.  This borrows from the logic in torch.nn.Module.
    """
    def __getattr__(self, name):
        if "options" in self.__dict__:
            options =  self.__dict__["options"]
            if name in options:
                return options[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
        if "options" in self.__dict__:
            if name in self.options:
                # print("setting {} {}".format(name, value))
                if name == "cast_model_type":
                    if self.opt_level == "O1" and value is not None:
                        if value is not False:
                            if value is not torch.float32:
                                warn_or_err("O1 inserts casts around Torch functions rather than "
                                            "model weights, so with O1, the model weights themselves "
                                            "should remain FP32. If you wish to cast the model to a "
                                            "different type, use opt_level='O2' or 'O3'. " +
                                            "cast_model_type was {}".format(value))
                    self.options[name] = value
                elif name == "patch_torch_functions":
                    if self.opt_level != "O1" and value:
                        warn_or_err("Currently, patch_torch_functions=True should only be set by "
                                    "selecting opt_level='O1'.")
                    self.options[name] = value
                elif name == "keep_batchnorm_fp32":
                    if self.opt_level == "O1" and value is not None:
                        warn_or_err("With opt_level O1, batchnorm functions are automatically patched "
                                    "to run in FP32, so keep_batchnorm_fp32 should be None." +
                                    " keep_batchnorm_fp32 was {}".format(value))
                    if value == "False":
                        self.options[name] = False
                    elif value == "True":
                        self.options[name] = True
                    else:
                        assert (value is True or value is False or value is None),\
                            "keep_batchnorm_fp32 must be a boolean, the string 'True' or 'False', "\
                            "or None, found keep_batchnorm_fp32={}".format(value)
                        self.options[name] = value
                elif name == "master_weights":
                    if self.opt_level == "O1" and value is not None:
                        warn_or_err("It doesn't make sense to use master_weights with O1. "
                                    "With O1, your model weights themselves should be FP32.")
                    self.options[name] = value
                elif name == "loss_scale":
                    if value == "dynamic":
                        self.options[name] = value
                    else:
                        self.options[name] = float(value)
                else:
                    self.options[name] = value
        else:
            super(Properties, self).__setattr__(name, value)


""" O0-O3 are convenience wrappers to establish defaults for typically used mixed precision options. """

class O3:
    brief = "O3:  Pure FP16 training."
    more = "Calls .half() on your model, converting the entire model to FP16.\n"\
        "A casting operation is also inserted to cast incoming Tensors to FP16,\n"\
        "so you don't need to change your data pipeline.\n"\
        "This mode is useful for establishing a performance ceiling.\n"\
        "It's also possible training may 'just work' in this mode.\n"\
        "If not, try other optimization levels."

    def __call__(self, properties):
        properties.enabled = True
        properties.opt_level = "O3"
        properties.cast_model_type = torch.float16
        properties.patch_torch_functions = False
        properties.keep_batchnorm_fp32 = False
        properties.master_weights = False
        properties.loss_scale = 1.0
        # properties.fused_optimizer = False
        # properties.enable_ddp_interop = False
        return properties # modified in place so this isn't really necessary


class O2:
    brief = "O2:  FP16 training with FP32 batchnorm and FP32 master weights.\n"
    more = "Calls .half() on your model, converting the entire model (except for batchnorms)\n"\
        "to FP16.  Batchnorms are retained in FP32 for additional stability.\n"\
        "The forward pass is patched to cast incoming Tensors to FP16, so you don't need to change\n"\
        "your data pipeline.\n"\
        "O2 creates FP32 master weights outside the model and patches any optimizers to update\n"\
        "these master weights, then copy the master weights into the FP16 model weights.\n"\
        "Master weights can also improve convergence and stability."

    def __call__(self, properties):
        properties.enabled = True
        properties.opt_level = "O2"
        properties.cast_model_type = torch.float16
        properties.patch_torch_functions = False
        properties.keep_batchnorm_fp32 = True
        properties.master_weights = True
        properties.loss_scale = "dynamic"
        # properties.fused_optimizer = False
        # properties.enable_ddp_interop = False
        return properties # modified in place so this isn't really necessary


class O1:
    brief = "O1:  Insert automatic casts around Pytorch functions and Tensor methods.\n"
    more = "The type of your model's weights is not altered.  However, internally,\n"\
        "Pytorch functions are patched to cast any Tensor Core-friendly ops to FP16 for speed,\n"\
        "while operations that might benefit from the additional stability of FP32 are patched\n"\
        "to cast their inputs to fp32.\n"\
        "O1 is the safest way to try mixed precision training, and is recommended when\n"\
        "trying mixed precision training for the first time."

    def __call__(self, properties):
        properties.enabled = True
        properties.opt_level = "O1"
        properties.cast_model_type = None
        properties.patch_torch_functions = True
        properties.keep_batchnorm_fp32 = None
        properties.master_weights = None
        properties.loss_scale = "dynamic"
        # properties.fused_optimizer = False
        # properties.enable_ddp_interop = False
        return properties # modified in place so this isn't really necessary


class O0:
    brief = "O0:  Pure FP32 training.\n"
    more = "Your models are checked to make sure parameters are FP32, but otherwise the\n"\
        "types of weights and internal Pytorch operations are not altered.  This mode disables any\n"\
        "FP16 arithmetic, although other optimizations like DDP interop may still be requested.\n"

    def __call__(self, properties):
        properties.enabled = True
        properties.opt_level = "O0"
        properties.cast_model_type = torch.float32
        properties.patch_torch_functions = False
        properties.keep_batchnorm_fp32 = None
        properties.master_weights = False
        properties.loss_scale = 1.0
        # properties.fused_optimizer = False
        # properties.enable_ddp_interop = False
        return properties # modified in place so this isn't really necessary


opt_levels = {"O3": O3(),
              "O2": O2(),
              "O1": O1(),
              "O0": O0()}


# allow user to directly pass Properties struct as well?
def initialize(
    models,
    optimizers=None,
    enabled=True,
    opt_level="O1",
    cast_model_type=None,
    patch_torch_functions=None,
    keep_batchnorm_fp32=None,
    master_weights=None,
    loss_scale=None,
    cast_model_outputs=None,
    num_losses=1,
    verbosity=1,
    min_loss_scale=None,
    max_loss_scale=2.**24
    ):
    """
    Initialize your models, optimizers, and the Torch tensor and functional namespace according to the
    chosen ``opt_level`` and overridden properties, if any.

    ``amp.initialize`` should be called **after** you have finished
    constructing your model(s) and
    optimizer(s), but **before** you send your model through any DistributedDataParallel wrapper.
    See `Distributed training`_ in the Imagenet example.

    Currently, ``amp.initialize`` should only be called **once**,
    although it can process an arbitrary number of
    models and optimizers (see the corresponding `Advanced Amp Usage topic`_).
    If you think your use case requires ``amp.initialize`` to be called more than once,
    `let us know`_.

    Any property keyword argument that is not ``None`` will be interpreted as a manual override.

    To prevent having to rewrite anything else in your script, name the returned models/optimizers
    to replace the passed models/optimizers, as in the code sample below.

    Args:
        models (torch.nn.Module or list of torch.nn.Modules):  Models to modify/cast.
        optimizers (optional, torch.optim.Optimizer or list of torch.optim.Optimizers):  Optimizers to modify/cast.
            REQUIRED for training, optional for inference.
        enabled (bool, optional, default=True):  If False, renders all Amp calls no-ops, so your script
            should run as if Amp were not present.
        opt_level (str, optional, default="O1"):  Pure or mixed precision optimization level.  Accepted values are
            "O0", "O1", "O2", and "O3", explained in detail above.
        cast_model_type (``torch.dtype``, optional, default=None):  Optional property override, see
            above.
        patch_torch_functions (bool, optional, default=None):  Optional property override.
        keep_batchnorm_fp32 (bool or str, optional, default=None):  Optional property override.  If
            passed as a string, must be the string "True" or "False".
        master_weights (bool, optional, default=None):  Optional property override.
        loss_scale (float or str, optional, default=None):  Optional property override.  If passed as a string,
            must be a string representing a number, e.g., "128.0", or the string "dynamic".
        cast_model_outputs (torch.dtype, optional, default=None):  Option to ensure that the outputs
            of your model(s) are always cast to a particular type regardless of ``opt_level``.
        num_losses (int, optional, default=1):  Option to tell Amp in advance how many losses/backward
            passes you plan to use.  When used in conjunction with the ``loss_id`` argument to
            ``amp.scale_loss``, enables Amp to use a different loss scale per loss/backward pass,
            which can improve stability.  See "Multiple models/optimizers/losses"
            under `Advanced Amp Usage`_ for examples.  If ``num_losses`` is left to 1, Amp will still
            support multiple losses/backward passes, but use a single global loss scale
            for all of them.
        verbosity (int, default=1):  Set to 0 to suppress Amp-related output.
        min_loss_scale (float, default=None):  Sets a floor for the loss scale values that can be chosen by dynamic
            loss scaling.  The default value of None means that no floor is imposed.
            If dynamic loss scaling is not used, `min_loss_scale` is ignored.
        max_loss_scale (float, default=2.**24):  Sets a ceiling for the loss scale values that can be chosen by
            dynamic loss scaling.  If dynamic loss scaling is not used, `max_loss_scale` is ignored.

    Returns:
        Model(s) and optimizer(s) modified according to the ``opt_level``.
        If either the ``models`` or ``optimizers`` args were lists, the corresponding return value will
        also be a list.

    Permissible invocations::

        model, optim = amp.initialize(model, optim,...)
        model, [optim1, optim2] = amp.initialize(model, [optim1, optim2],...)
        [model1, model2], optim = amp.initialize([model1, model2], optim,...)
        [model1, model2], [optim1, optim2] = amp.initialize([model1, model2], [optim1, optim2],...)

        # This is not an exhaustive list of the cross product of options that are possible,
        # just a set of examples.
        model, optim = amp.initialize(model, optim, opt_level="O0")
        model, optim = amp.initialize(model, optim, opt_level="O0", loss_scale="dynamic"|128.0|"128.0")

        model, optim = amp.initialize(model, optim, opt_level="O1") # uses "loss_scale="dynamic" default
        model, optim = amp.initialize(model, optim, opt_level="O1", loss_scale=128.0|"128.0")

        model, optim = amp.initialize(model, optim, opt_level="O2") # uses "loss_scale="dynamic" default
        model, optim = amp.initialize(model, optim, opt_level="O2", loss_scale=128.0|"128.0")
        model, optim = amp.initialize(model, optim, opt_level="O2", keep_batchnorm_fp32=True|False|"True"|"False")

        model, optim = amp.initialize(model, optim, opt_level="O3") # uses loss_scale=1.0 default
        model, optim = amp.initialize(model, optim, opt_level="O3", loss_scale="dynamic"|128.0|"128.0")
        model, optim = amp.initialize(model, optim, opt_level="O3", keep_batchnorm_fp32=True|False|"True"|"False")

    The `Imagenet example`_ demonstrates live use of various opt_levels and overrides.

    .. _`Distributed training`:
        https://github.com/NVIDIA/apex/tree/master/examples/imagenet#distributed-training

    .. _`Imagenet example`:
        https://github.com/NVIDIA/apex/tree/master/examples/imagenet

    .. _`Advanced Amp Usage`:
        https://nvidia.github.io/apex/advanced.html

    .. _`Advanced Amp Usage topic`:
        https://nvidia.github.io/apex/advanced.html#multiple-models-optimizers-losses

    .. _`let us know`:
        https://github.com/NVIDIA/apex/issues
    """
    _amp_state.opt_properties = Properties()
    _amp_state.verbosity = verbosity

    if not enabled:
        if optimizers is None:
            return models
        else:
            return models, optimizers

    if not torch.backends.cudnn.enabled:
        raise RuntimeError(
            "Amp requires torch.backends.cudnn.enabled = True")

    if opt_level not in opt_levels:
        raise RuntimeError(
            "Unexpected optimization level {}. ".format(opt_level) +
            "Options are 'O0', 'O1', 'O2', 'O3'.  Note that in `O0`, `O1`, etc., the prefix O is the letter O, " +
            "not the number zero.")
    else:
        _amp_state.opt_properties = opt_levels[opt_level](_amp_state.opt_properties)
        maybe_print("Selected optimization level {}".format(opt_levels[opt_level].brief), True)
        maybe_print("Defaults for this optimization level are:", True)
        for k, v in _amp_state.opt_properties.options.items():
            maybe_print("{:22} : {}".format(k, v), True)

    _amp_state.min_loss_scale = min_loss_scale
    _amp_state.max_loss_scale = max_loss_scale

    maybe_print("Processing user overrides (additional kwargs that are not None)...", True)
    # I chose to have the keyword arguments listed directly in the argument list,
    # instead of **kwargs, so I can't use kwargs.items() here.
    if enabled is not None:
        _amp_state.opt_properties.enabled = enabled
    if opt_level is not None:
        _amp_state.opt_properties.opt_level = opt_level
    if cast_model_type is not None:
        _amp_state.opt_properties.cast_model_type = cast_model_type
    if patch_torch_functions is not None:
        _amp_state.opt_properties.patch_torch_functions = patch_torch_functions
    if keep_batchnorm_fp32 is not None:
        _amp_state.opt_properties.keep_batchnorm_fp32 = keep_batchnorm_fp32
    if master_weights is not None:
        _amp_state.opt_properties.master_weights = master_weights
    if loss_scale is not None:
        _amp_state.opt_properties.loss_scale = loss_scale

    maybe_print("After processing overrides, optimization options are:", True)
    for k, v in _amp_state.opt_properties.options.items():
        maybe_print("{:22} : {}".format(k, v), True)

    return _initialize(models, optimizers, _amp_state.opt_properties, num_losses, cast_model_outputs)


def state_dict(destination=None):
    if destination is None:
        destination = OrderedDict()

    for idx, loss_scaler in enumerate(_amp_state.loss_scalers):
        destination['loss_scaler%d' % idx] = {
            'loss_scale': loss_scaler.loss_scale(),
            'unskipped': loss_scaler._unskipped,
        }
    return destination


def load_state_dict(state_dict):
    # Check if state_dict containes the same number of loss_scalers as current setup
    if len(state_dict) != len(_amp_state.loss_scalers):
        print('Warning: state_dict contains {} entries, while {} loss_scalers are used'.format(
            len(state_dict), len(_amp_state.loss_scalers)))

    state_dict = state_dict.copy()
    
    nb_loss_scalers = len(_amp_state.loss_scalers)
    unexpected_keys = []
    # Initialize idx outside, since unexpected_keys will increase it if enumerate is used
    idx = 0
    for key in state_dict:
        if 'loss_scaler' not in key:
            unexpected_keys.append(key)
        else:
            if idx > (nb_loss_scalers - 1):
                print('Skipping loss_scaler[{}], since num_losses was set to {}'.format(
                    idx, nb_loss_scalers))
                break
            _amp_state.loss_scalers[idx]._loss_scale = state_dict[key]['loss_scale']
            _amp_state.loss_scalers[idx]._unskipped = state_dict[key]['unskipped']
            idx += 1

    if len(unexpected_keys) > 0:
        raise RuntimeError(
            'Error(s) in loading state_dict. Unexpected key(s) in state_dict: {}. '.format(
                ', '.join('"{}"'.format(k) for k in unexpected_keys)))


# TODO:  is this necessary/useful?
# def check_option_consistency(enabled=True,
#                              opt_level=None,
#                              cast_model_type=None,
#                              patch_torch_functions=None,
#                              keep_batchnorm_fp32=None,
#                              master_weights=None,
#                              loss_scale=None,
#                              enable_ddp_interop=None,
#                              hard_override=False):
#     """
#     Utility function that enables users to quickly check if the option combination they intend
#     to use is permitted.  ``check_option_consistency`` does not require models or optimizers
#     to be constructed, and can be called at any point in the script.  ``check_option_consistency``
#     is totally self-contained; it does not set any amp global state or affect anything outside
#     of itself.
#     """
#
#     if not enabled:
#         return
#
#     if opt_level not in opt_levels:
#         raise RuntimeError("Unexpected optimization level.  Options are 'O0', 'O1', 'O2', 'O3'.")
#     else:
#         opt_properties = opt_levels[opt_level](Properties())
#         print("Selected optimization level {}", opt_levels[opt_level].brief)
#         print("Defaults for this optimization level are:")
#         for k, v in opt_properties.options:
#             print("{:22} : {}".format(k, v))
#
#     print("Processing user overrides (additional kwargs that are not None)...")
#     for k, v in kwargs:
#         if k not in _amp_state.opt_properties.options:
#             raise RuntimeError("Unexpected kwarg {}".format(k))
#         if v is not None:
#             setattr(opt_properties, k, v)
#
#     print("After processing overrides, optimization options are:")
#     for k, v in opt_properties.options:
#         print("{:22} : {}".format(k, v))
