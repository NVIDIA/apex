import types
import torch
from .sparse_masklib import create_mask

torchvision_imported=True
try:
    import torchvision
except ImportError:
    print("[ASP][Warning] torchvision cannot be imported.")
    torchvision_imported=False

def eligible_modules(model, whitelist_layer_types, allowed_layer_names, disallowed_layer_names):
    eligible_modules_list = []
    for name, mod in model.named_modules():
        if isinstance(mod, whitelist_layer_types) and name not in disallowed_layer_names:
            if allowed_layer_names is not None and name not in allowed_layer_names:
                continue
            eligible_modules_list.append((name, mod))
    return eligible_modules_list

class ASP:
    __model = None
    __verbosity = 0
    __optimizer = None
    __sparse_parameters = []
    __calculate_mask = None

    @classmethod
    def init_model_for_pruning(cls, model, mask_calculator="m4n2_1d",
             verbosity=3,
             whitelist=[torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d], 
             allowed_layer_names=None, disallowed_layer_names=[],
             allow_recompute_mask=False):
        """Call this method to modify your model to take advantage of sparse matrix multiplication.
        Note that this call alone only augments the model with additional buffers needed for sparse MMA,
        it does not enable use of sparse MMA. 

        If you are starting with a fresh model:

        model = ...
        ASP.init_model_for_pruning(model, mask_calculator, ...)
        if (training) ASP.init_optimizer_for_pruning(optimizer)
        ASP.compute_sparse_masks() // sparsity is off by default, call when youy want to enable it.

        If you are starting from a checkpoint:

        model = ...
        ASP.init_model_for_pruning(model, mask_calculator, ...)
        torch.load(...)
        if (training) ASP.init_optimizer_for_pruning(optimizer)

        Arguments:
          model                    The model
          mask_calculator          Either callable that computes mask given a tensor OR pattern string for sparse mask lib.
          verbosity                Integer controling verbosity level.
                                   0 -> Only errors.
                                   1 -> Errors and warnings.
                                   2 -> Errors, warnings and info.
                                   3 -> Errors, warnings, info and debug.
          whitelist                Module types approved for sparsity.
          allowed_layer_names      If not None, only layer names that appear in this list are considered for sparsity.
          disallowed_layer_names   If not [], only layer names that do not appear in this list are considered for sparsity.
          allow_recompute_mask     If True, stores pruned values so that dense weights can be restored.
                                   Pruned weights are stored in CPU memory, hence this option does not increase GPU memory usage.
          Support for allow_recompute_mask can be removed, it is not part of our recipe -- AKM. 
        """
        assert (cls.__model is None), "ASP has been initialized already."
        cls.__model = model
        cls.__verbosity = verbosity

        if isinstance(mask_calculator, str):
            def create_mask_from_pattern(param):
                return create_mask(param, mask_calculator).bool()
            cls.__calculate_mask = create_mask_from_pattern
        else:
            cls.__calculate_mask = mask_calculator #user defined function

        # function to extract variables that will be sparsified. 
        # idea is that you will add one of these functions for each module type that can be sparsified.
        if torchvision_imported:
            print("[ASP] torchvision is imported, can work with the MaskRCNN/KeypointRCNN from torchvision.")
            sparse_parameter_list = {torch.nn.Linear: ['weight'], torch.nn.Conv1d: ['weight'], torch.nn.Conv2d: ['weight'], torch.nn.Conv3d: ['weight'], torchvision.ops.misc.Conv2d: ['weight']}
        else:
            sparse_parameter_list = {torch.nn.Linear: ['weight'], torch.nn.Conv1d: ['weight'], torch.nn.Conv2d: ['weight'], torch.nn.Conv3d: ['weight']}
        for module_type in whitelist:
            assert (module_type in sparse_parameter_list), "Module %s :: Don't know how to sparsify module." % module.dtype()

        # find all sparse modules, extract sparse parameters and decorate
        def add_sparse_attributes(module_name, module):
            sparse_parameters = sparse_parameter_list[type(module)]
            for p_name, p in module.named_parameters():
                if p_name in sparse_parameters and p.requires_grad:
                    # check for NVIDIA's TC compatibility: we check along the horizontal direction
                    if p.dtype == torch.float32 and ((p.size()[0] % 8) != 0 or (p.size()[1] % 16) != 0): #User defines FP32 and APEX internally uses FP16 math
                        print("[ASP] Auto skipping pruning %s::%s of size=%s and type=%s for sparsity" % (module_name, p_name, str(p.size()), str(p.dtype)))
                        continue
                    if p.dtype == torch.float16 and ((p.size()[0] % 8) != 0 or (p.size()[1] % 16) != 0): #For Conv2d dim= K x CRS; we prune along C
                        print("[ASP] Auto skipping pruning %s::%s of size=%s and type=%s for sparsity" % (module_name, p_name, str(p.size()), str(p.dtype)))
                        continue

                    if cls.__verbosity >= 3:
                        print("[ASP] Sparsifying %s::%s of size=%s and type=%s for sparsity" % (module_name, p_name, str(p.size()), str(p.dtype)))
                    
                    mask = torch.ones_like(p).bool()
                    buffname = name.split(".")[-1] # buffer names cannot contain "."
                    module.register_buffer('__%s_mma_mask' % buffname, mask)
                    if allow_recompute_mask:
                        pruned = torch.zeros_like(p).cpu()
                        module.register_buffer('__%s_mma_pruned_p' % buffname, pruned)
                    else:
                        pruned = None
                    cls.__sparse_parameters.append((module_name, module, p_name, p, mask, pruned))

        for name, sparse_module in eligible_modules(model, tuple(whitelist), allowed_layer_names, disallowed_layer_names):
            add_sparse_attributes(name, sparse_module)

    @classmethod
    def init_optimizer_for_pruning(cls, optimizer):
        """Call this method to monkey patch optimizer step function so that masks can be applied to
        gradients and weights during training.
        You must call init_model_for_pruning(...) before calling init_optimizer_for_pruning(...)
        """
        assert (cls.__optimizer is None), "ASP has initialized optimizer already."
        assert (cls.__calculate_mask is not None), "Called ASP.init_optimizer_for_pruning before ASP.init_model_for_pruning."

        # store pointer to original optimizer step method
        cls.__optimizer = optimizer
        cls.__optimizer.__step = optimizer.step

        def __step(opt_self, *args, **kwargs):
            # prune gradients before step method
            with torch.no_grad():
                for module_name, module, p_name, p, mask, pruned in cls.__sparse_parameters:
                    p.grad.mul_(mask)
            # call original optimizer step method
            rval = opt_self.__step(*args, **kwargs)
            # prune parameters after step method
            with torch.no_grad():
                for module_name, module, p_name, p, mask, pruned in cls.__sparse_parameters:
                    p.mul_(mask)
            return rval
        cls.__optimizer.step = types.MethodType(__step, cls.__optimizer)

    @classmethod
    def compute_sparse_masks(cls):
        """Call this method to enable sparsity.
        If init(...) was called with allow_recompute_mask=False AND sparsity is disabled, pruned field can be None.
        """
        with torch.no_grad():
            for module_name, module, p_name, p, mask, pruned in cls.__sparse_parameters:
                if mask.sum() < mask.numel(): # when recalculating masks
                    # restore dense parameter if allow_recompute_mask is enabled
                    assert (pruned is not None), "Unable to restore dense parameter because allow_recompute_mask == False"
                    p.add_(pruned.cuda())

                mask.set_(cls.__calculate_mask(p))

                if pruned is not None: # stow away pruned weights to cpu
                    pruned.set_((p * (~mask)).cpu())

                p.mul_(mask) # in-place multiplication, so pruned weights are 0-values, hence checkpoint will have 0s for pruned weights
                if cls.__verbosity >= 2:
                    print("[ASP] Enabled %.2f%% sparsity for %s::%s of size=%s and type=%s" % (100.0*mask.sum()/mask.numel(), module_name, p_name, str(p.size()), str(p.dtype)))

    @classmethod
    def restore_pruned_weights(cls):
        """Call this method to disable sparsity and restore all weights.
        This will only work if init(...) was called with allow_recompute=True.
        """
        with torch.no_grad():
            for module_name, module, p_name, p, mask, pruned in cls.__sparse_parameters:
                if mask.sum() < mask.numel():
                    assert (pruned is not None), "Unable to restore dense parameter because allow_recompute_mask == False"
                    p.add_(pruned.cuda())
                    mask.fill_(1)
                    pruned.zero_()
                    if cls.__verbosity >= 2:
                        print("[ASP] Disabled sparsity for %s::%s (dense weights restored)" % (module_name, p_name))

    @classmethod
    def is_sparsity_enabled(cls):
        """Call this method to determine if sparsity is enabled in the model.
        The typical use case is right after checkpoint has been loaded.
        """
        total,sp100,sp50 = 0,0,0
        for module_name, module, p_name, p, mask, pruned in cls.__sparse_parameters:
            total += 1
            mask_sum = mask.sum()
            mask_numel = mask.numel()
            if mask_sum == mask_numel:
                sp100 += 1
            elif mask_sum*2 == mask_numel:
                sp50 += 1

        assert (total == sp100 or total == sp50), "Inconsistent model sparsity"
        if total == sp100:
            return False
        elif total == sp50:
            return True
    
    @classmethod
    def prune_trained_model(cls, model, optimizer):
        # add mask buffers to model (init_model_for_pruning), augment optimizer (init_optimizer_for_pruning) and compute masks (compute_sparse_masks)
        cls.init_model_for_pruning(model, mask_calculator="m4n2_1d", verbosity=2, whitelist=[torch.nn.Linear, torch.nn.Conv2d], allow_recompute_mask=False)
        cls.init_optimizer_for_pruning(optimizer)
        cls.compute_sparse_masks()

