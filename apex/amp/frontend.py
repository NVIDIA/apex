import torch
from .initialize import initialize


class Properties(object):
    """ 
    The purpose of this class is twofold:  to establish a set of default properties,
    and to route setting of these attributes through __setattr__ so that (in theory)
    they can be checked for consistency with other existing args.
    """
    def __init__(self):
        self.options = {
            "opt_level" : None,
            "cast_model_type" : None,
            "cast_torch_functions" : False,
            "cast_batchnorm" : None,
            "master_weights" : False,
            "loss_scale" : 1.0,
            "flatten_model_params" : False,
            "flatten_master_params" : False,
            "enable_ddp_interop" : False}

    """
    This function will allow updating several options at a time without routing through
    __setattr__ checks, to avoid "you can't get there from here" scenarios.
    """
    def update_options_dict(new_options):
        for k, v in new_options:
            if k in self.options:
                self.options[k] = v
            else:
                raise ValueError("Tried to set unexpected option {}".format(k))
    """
    The members of options are not direct attributes of self, so __getattr__ is ok.
    This borrows from the logic in torch.nn.Module. 
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
                print("setting {}".format(name))
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
        properties.opt_level = "O3",
        properties.cast_model_type = torch.float16
        properties.cast_torch_functions = False
        properties.cast_batchnorm = False
        properties.master_weights = False
        properties.loss_scale = 1.0
        properties.flatten_model_params = False
        properties.flatten_master_params = False
        properties.enable_ddp_interop = False
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
        properties.opt_level = "O2",
        properties.cast_model_type = torch.float16
        properties.cast_torch_functions = False
        properties.cast_batchnorm = torch.float32
        properties.master_weights = True
        properties.loss_scale = 128.0
        properties.flatten_model_params = False
        properties.flatten_master_params = False
        properties.enable_ddp_interop = False
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
        properties.opt_level = "O1",
        properties.cast_model_type = False
        properties.cast_torch_functions = True
        properties.cast_batchnorm = False
        properties.master_weights = False
        properties.loss_scale = "dynamic"
        properties.flatten_model_params = False
        properties.flatten_master_params = False
        properties.enable_ddp_interop = False
        return properties # modified in place so this isn't really necessary


class O0:
    brief = "O0:  Pure FP32 training.\n"
    more = "Your models are checked to make sure parameters are FP32, but otherwise the\n"\
        "types of weights and internal Pytorch operations are not altered.  This mode disables any\n"\
        "FP16 arithmetic, although other optimizations like parameter flattening and DDP interop\n"\
        "may still be requested.\n"

    def __call__(self, properties):
        properties.opt_level = "O0",
        properties.cast_model_type = torch.float32
        properties.cast_torch_functions = False
        properties.cast_batchnorm = False
        properties.master_weights = False
        properties.loss_scale = 1.0
        properties.flatten_model_params = False
        properties.flatten_master_params = False
        properties.enable_ddp_interop = False
        return properties # modified in place so this isn't really necessary


opt_levels = {"O3": O3(),
              "O2": O2(),
              "O1": O1(),
              "O0": O0()}

def check_params_fp32(model):
    for name, param in model.named_parameters():
        if param.type() != "torch.cuda.FloatTensor":
            print("Warning:  Found param {} with type {}, expected torch.cuda.FloatTensor.\n"
                  "When using amp.register, you do not need to call .half() on your model\n"
                  "before passing it, no matter what optimization level you choose.",
                  name, param.type())

    for name, param in model.named_buffers():
        if param.type() != "torch.cuda.FloatTensor":
            print("Warning:  Found buffer {} with type {}, expected torch.cuda.FloatTensor.\n"
                  "When using amp.register, you do not need to call .half() on your model\n"
                  "before passing it, no matter what optimization level you choose.",
                  name, param.type())


# allow user to directly pass Properties struct as well?
def register(enabled=False,
             optimizers=None,
             models=None,
             opt_level=None,
             cast_model_type=None,
             cast_torch_functions=None,
             cast_batchnorm=None,
             master_weights=None,
             loss_scale=None,
             flatten_model_params=None,
             flatten_master_params=None,
             enable_ddp_interop=None):

    if not enabled:
        return

    if opt_level not in opt_levels:
        raise RuntimeError("Unexpected optimization level.  Options are 'O0', 'O1', 'O2', 'O3'.")
    else:
        amp.opt_properties = opt_levels[opt_level](Properties())
        print("Selected optimization level {}", opt_levels[opt_level].brief)
        print("Defaults for this optimization level are:")
        for k, v in amp.opt_properties.options:
            print("{:20} : {}", k, v)

    for model in models:
        check_params_fp32(model)

    print("Processing user overrides (additional kwargs that are not None)...")
    for k, v in kwargs:
        if v is not None:
            setattr(amp.opt_properties, k, v)

    print("After processing overrides, optimization options are:")
    for k, v in amp.opt_properties.options:
        print("{:20} : {}", k, v)

    return initialize(optimizers, models)


def check_option_consistency(enabled=False,
                             opt_level=None,
                             cast_model_type=None,
                             cast_torch_functions=None,
                             cast_batchnorm=None,
                             master_weights=None,
                             loss_scale=None,
                             flatten_model_params=None,
                             flatten_master_params=None,
                             enable_ddp_interop=None):
    """
    Utility function that enables users to quickly check if the option combination they intend
    to use is permitted.  ``check_option_consistency`` does not require models or optimizers
    to be constructed, and can be called at any point in the script.  ``check_option_consistency``
    is totally self-contained; it does not set any amp global state or affect anything outside
    of itself.
    """

    if not enabled:
        return

    if opt_level not in opt_levels:
        raise RuntimeError("Unexpected optimization level.  Options are 'O0', 'O1', 'O2', 'O3'.")
    else:
        opt_properties = opt_levels[opt_level](Properties())
        print("Selected optimization level {}", opt_levels[opt_level].brief)
        print("Defaults for this optimization level are:")
        for k, v in opt_properties.options:
            print("{:20} : {}", k, v)

    print("Processing user overrides (additional kwargs that are not None)...")
    for k, v in kwargs:
        if v is not None:
            setattr(opt_properties, k, v)

    print("After processing overrides, optimization options are:")
    for k, v in opt_properties.options:
        print("{:20} : {}", k, v)
