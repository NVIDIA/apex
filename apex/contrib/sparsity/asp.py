import types
import torch
from .sparse_masklib import create_mask
from .permutation_lib import Permutation

torchvision_imported=True
try:
    import torchvision
except ImportError:
    print("[ASP][Warning] torchvision cannot be imported.")
    torchvision_imported=False

import json
import os
import string
import time

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
    __allow_permutation = True
    __all_parameters = []
    __save_permutation_graph = False
    __permutation_output_dir = ''

    @classmethod
    def init_model_for_pruning(cls, model, mask_calculator="m4n2_1d",
             verbosity=3,
             whitelist=[torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d], 
             allowed_layer_names=None, disallowed_layer_names=[],
             allow_recompute_mask=False, custom_layer_dict={},
             allow_permutation=True):
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
          custom_layer_dict        Dictionary of additional layer paremeters to sparsify. e.g. {CustomLinear: ['weight']}
          allow_permutation        If True, allow the input channel permutation to ease the influence of weight pruning.
          
          [Future] Support for allow_recompute_mask can be removed, it is not part of sparse inference recipe.
        """
        assert (cls.__model is None), "ASP has been initialized already."
        cls.__model = model
        cls.__verbosity = verbosity
        cls.__allow_permutation = allow_permutation

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
        if custom_layer_dict: # Update default list to include user supplied custom (layer type : parameter tensor), make sure this tensor type is something ASP knows how to prune
            sparse_parameter_list.update(custom_layer_dict)
            whitelist += list(custom_layer_dict.keys())

        for module_type in whitelist:
            assert (module_type in sparse_parameter_list), "Module %s :: Don't know how to sparsify module." % module.dtype()

        if allow_permutation:    # find all named modules, extract parameters and decorate, used for offline permutation in K dim
            for module_name, module in model.named_modules():
                module_type_str = str(type(module)).split("\'")[1]
                if module_type_str == 'torch.nn.modules.container.Sequential' or module_type_str.startswith('torchvision.models'):
                    # filter out the 'torch.nn.modules.container.Sequential' type and the whole model, like 'torchvision.models.vgg.VGG'
                    continue
                for p_name, p in module.named_parameters():
                    cls.__all_parameters.append((module_name, module, p_name, p))
                if module_type_str == 'torch.nn.modules.batchnorm.BatchNorm2d':
                # need to get the running_mean and running_var from model.state_dict(), as they are not the learnable parameters
                    module_mean_name = module_name + '.running_mean'
                    module_var_name = module_name + '.running_var'
                    for param_key in model.state_dict():
                        if module_mean_name == param_key or module_var_name == param_key:
                            cls.__all_parameters.append((module_name, module, param_key.split(".")[-1], model.state_dict()[param_key]))
            # add the __permutation_output_dir field to save the intermediate results for permutation
            cls.__permutation_output_dir = '.'
            # Set the corresponding params from ASP class to the Permutation class
            Permutation.set_permutation_params_from_asp(cls.__model, cls.__sparse_parameters, cls.__all_parameters)
            # Set the identical random seed for all GPUs to make sure the same results generated in permutation search
            Permutation.set_identical_seed()

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
                    buffname = p_name.split(".")[-1] # buffer names cannot contain "."
                    module.register_buffer('__%s_mma_mask' % buffname, mask)
                    if allow_recompute_mask:
                        pruned = torch.zeros_like(p).cpu()
                        module.register_buffer('__%s_mma_pruned_p' % buffname, pruned)
                    else:
                        pruned = None
                    cls.__sparse_parameters.append((module_name, module, p_name, p, mask, pruned))
                else:
                    if cls.__verbosity >= 3:
                        print("[ASP] Not sparsifying %s::%s of size=%s and type=%s" % (module_name, p_name, str(p.size()), str(p.dtype)))

        for name, sparse_module in eligible_modules(model, tuple(whitelist), allowed_layer_names, disallowed_layer_names):
            add_sparse_attributes(name, sparse_module)

    @classmethod
    def already_init_asp_model(cls):
        """Call this method to check whether ASP has been initialized already.
        """
        if cls.__model is None:
            if cls.__verbosity >= 3:
                print("[ASP] ASP has not been initialized.")
                return False
        else:
            if cls.__verbosity >= 3:
                print("[ASP] ASP has been initialized already.")
                return True

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
                    if p.grad is not None: #thx pjudd
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
            if cls.__allow_permutation:
                # Step 1: use the Torch.FX library to build the graph
                # Step 2: permutation search with the customized kernel
                # Notice: need to use the single GPU to build the Torch.FX graph
                # The simplest without user intervention:
                # A. try to import with the distributed mode of the original model
                # B. if meet the error, import with the none-distributed mode of the original model
                start_time_build_offline_permutation_graph = time.perf_counter()
                try:
                    offline_permutation_fx_graph, success_in_build_offline_permutation_graph = Permutation.build_offline_permutation_graph(cls.__model.module, dump_fx_graph=cls.__save_permutation_graph, save_dumped_fx_graph=os.path.join(cls.__permutation_output_dir, 'model_offline_permutation_graph.json'))
                    print("\n[compute_sparse_masks] build offline permutation graph on distributed model.")
                except AttributeError:
                    offline_permutation_fx_graph, success_in_build_offline_permutation_graph = Permutation.build_offline_permutation_graph(cls.__model, dump_fx_graph=cls.__save_permutation_graph, save_dumped_fx_graph=os.path.join(cls.__permutation_output_dir, 'model_offline_permutation_graph.json'))
                    print("\n[compute_sparse_masks] build offline permutation graph on none-distributed model.")
                duration_build_offline_permutation_graph = time.perf_counter() - start_time_build_offline_permutation_graph
                print("[compute_sparse_masks] Take {:.4f} seconds to finish build_offline_permutation_graph function.".format(duration_build_offline_permutation_graph))

                # Step 3: off-line permutation to avoid the runtime overhead in deployment
                if success_in_build_offline_permutation_graph:
                    start_time_apply_offline_permutation = time.perf_counter()
                    try:
                        Permutation.apply_offline_permutation(cls.__model.module, fx_graph=offline_permutation_fx_graph)
                        print("\n[compute_sparse_masks] apply offline permutation on distributed model.")
                    except AttributeError:
                        Permutation.apply_offline_permutation(cls.__model, fx_graph=offline_permutation_fx_graph)
                        print("\n[compute_sparse_masks] apply offline permutation on none-distributed model.")
                    duration_apply_offline_permutation = time.perf_counter() - start_time_apply_offline_permutation
                    print("[compute_sparse_masks] Take {:.4f} seconds to finish apply_offline_permutation function.\n".format(duration_apply_offline_permutation))
                else:
                    print("[compute_sparse_masks] skip applying offline permutation because there is no valid offline_permutation_fx_graph.")
                # Finally, permutation search and off-line permutation is done, give the model back to ASP to generate the normal structured sparse mask

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
                    print("[ASP] Enabled %.2f%% sparsity for %s::%s of size=%s and type=%s" % (100.0-100.0*mask.sum()/mask.numel(), module_name, p_name, str(p.size()), str(p.dtype)))

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

    @classmethod
    def set_permutation_saving_params(cls, allow_permutation=True, save_permutation_graph=False, permutation_output_dir='.'):
        """This function is used to set the permutation saving related parameters in ASP class and inside of the Permutation class."""
        print("\n[ASP][set_permutation_saving_param] Set permutation saving related parameters")
        print("\n[set_permutation_saving_param] Set permutation saving related parameters")
        cls.__allow_permutation = allow_permutation
        print("[set_permutation_saving_param]\t Allow permutation: {}".format(cls.__allow_permutation))
        cls.__save_permutation_graph = save_permutation_graph
        print("[set_permutation_saving_param]\t Save permutation graphs: {}".format(cls.__save_permutation_graph))
        cls.__permutation_output_dir = permutation_output_dir
        print("[set_permutation_saving_param]\t Permutation graphs saving dir: {}".format(cls.__permutation_output_dir))

        Permutation.set_permutation_saving_params(allow_permutation, save_permutation_graph, permutation_output_dir)

