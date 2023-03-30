import os
import torch
import json
import string
import time
import numpy as np
import sys
import builtins as __builtin__
import io
try:
    from .permutation_search_kernels import accelerated_search_for_good_permutation, sum_after_2_to_4
    print("[ASP][Info] permutation_search_kernels can be imported.")
except ImportError:
    print("[ASP][Warning] permutation_search_kernels cannot be imported.")
    print("[ASP][Warning] If you want to accelerate the permutation search process by GPU, please build APEX by following the instructions at https://github.com/NVIDIA/apex/blob/master/apex/contrib/sparsity/README.md")

def convert_fx_node_name(fx_node_name):
    """Standardize punctuation of a node's name: replace all '_' with '.'"""
    return fx_node_name.replace('_', '.')

def get_node_parent_children(fx_node):
    """Populate lists of all direct parents and children of a node"""
    # get node parent list, and convert node name to module name
    node_parent_name_converted = []
    if len(fx_node.all_input_nodes) > 0:
        node_parent = fx_node.all_input_nodes
        for item in node_parent:
            converted_item = convert_fx_node_name(item.name)
            node_parent_name_converted.append(converted_item)
    else:
        node_parent = []
    
    # get node children list, and convert node name to module name
    node_children_name_converted = []
    if len(list(fx_node.users.keys())) > 0:
        node_children = list(fx_node.users.keys())
        for item in node_children:
            converted_item = convert_fx_node_name(item.name)
            node_children_name_converted.append(converted_item)
    else:
        node_children = []

    return node_parent_name_converted, node_children_name_converted

def node_name_matches(node_name, module_name):
    """Check for a match between graph node name and stored module name, accounting for formatting and DDP training differences"""

    # process: remove all punctuation, everything to lower case
    def process(name):
        return ''.join(c for c in name if c not in string.punctuation).lower()

    processed_node_name = process(node_name)
    processed_module_name = process(module_name)

    # module names start with 'module.' in distributed data-parallel training, but fx graph node names don't; check for both
    distributed_node_name = 'module.' + node_name
    distributed_processed_node_name = 'module' + processed_node_name

    return (node_name == module_name) or (distributed_node_name == module_name) or (processed_node_name == processed_module_name) or (distributed_processed_node_name == processed_module_name)

def replicate_sequence(sequence, replications):
    """Replicate a permutation to apply it to an even multiple of channel counts"""
    replicated_sequence = []

    for rep in range(replications):
        offset = len(sequence) * rep
        for c in sequence:
            replicated_sequence.append(c+offset)

    return replicated_sequence

class Permutation:
    __model = None
    __sparse_parameters = []
    __allow_permutation = False
    __all_parameters = []
    __verbosity = 0                 ## 0: errors only, 1: also high-level details, warnings, 2: also intermediate steps, 3: everything
    __params_permuted_in_C = []
    __params_permuted_in_K = []
    __unpermuted_dims = []

    __save_permutation_graph = False
    __permutation_output_dir = ''
    __manual_seed = None
    __tcpstore_port = 2341
    
    # these module types may be the target of permutations (have potentially sparse weights or are attributes with no parents)
    __permutation_target_module_types = ['torch.nn.modules.conv.Conv1d',
                                         'torch.nn.modules.conv.Conv2d',
                                         'torch.nn.modules.linear.Linear',
                                         'torch.nn.modules.linear.LazyLinear',
                                         'torch.nn.modules.linear.NonDynamicallyQuantizableLinear',
                                         'torch.nn.modules.activation.MultiheadAttention',
                                         'get_attr']

    # these module types are not permuted, but must pass any permutation seen by a child's C or passed-thru K to the parents' K
    __simple_passthru_module_types = ['torch.nn.modules.activation.ReLU6',
                                      'torch.nn.modules.activation.ReLU',
                                      'torch.nn.modules.dropout.Dropout',
                                      'torch.nn.modules.dropout.Dropout1d',
                                      'torch.nn.modules.dropout.Dropout2d',
                                      'torch.nn.modules.dropout.Dropout3d',
                                      'torch.nn.modules.dropout.AlphaDropout',
                                      'torch.nn.modules.dropout.FeatureAlphaDropout',
                                      'torch.nn.modules.pooling.MaxPool2d',
                                      'torch.nn.modules.pooling.AdaptiveAvgPool2d',
                                      'torch.nn.modules.pooling.AvgPool2d',
                                      'torch.nn.modules.activation.Hardsigmoid',
                                      'torch.nn.modules.activation.Hardswish',
                                      'torch.nn.modules.activation.GELU',
                                      'torch.nn.modules.normalization.LocalResponseNorm',
                                      'torch.nn.modules.activation.Softmin',
                                      'torch.nn.modules.activation.Softmax',
                                      'torch.nn.modules.activation.Softmax2d',
                                      'torch.nn.modules.activation.LogSoftmax',
                                      'torch.nn.modules.activation.AdaptiveLogSoftmaxWithLoss',
                                      'torch.nn.modules.activation.SiLU',
                                      'torch.nn.modules.activation.Sigmoid',
                                      'concat',
                                      'torch.nn.modules.flatten.Flatten' # if it's a problem, it'll be handled via dimension mismatch check
                                      ]

    # these module types have parameters that must be permuted along K as well as need to pass the permutation thru to parents' K
    __permute_K_and_passthru_module_types = ['torch.nn.modules.batchnorm.BatchNorm2d',
                                             'torch.nn.modules.normalization.LayerNorm',
                                             'torch.nn.modules.instancenorm.InstanceNorm2d',
                                             'torch.nn.modules.batchnorm.SyncBatchNorm']

    # these module types cannot be permuted safely (today), and cause neighboring layers to have permutations disabled
    __disallow_permutations_module_types = ['torch.nn.modules.normalization.GroupNorm', # to handle: influence GCD of real children's sibling group
                                            'torch.nn.modules.linear.Bilinear',         # need to permute one input along in1_features and the other along in2_features
                                            'torch.nn.modules.activation.GLU',          # may work OOTB, but might need to explicitly handle dimsionality change
                                            ]

    @classmethod
    def set_identical_seed(cls, identical_seed=1):
        """Make all GPUs in DDP use the same seed to find identical permutations and not require syncing parameters later"""

        if cls.__verbosity > 0:
            print("[set_identical_seed] Set the identical seed: {:} for all GPUs to make sure the same results generated in permutation search".format(identical_seed))

        cls.__manual_seed = identical_seed
        cls.reset_seed()

    @classmethod
    def reset_seed(cls):
        """To find the same permutations no matter how many GPUs are used, we reset the seed before every search"""

        identical_seed = cls.__manual_seed
        assert identical_seed is not None, "Must call set_identical_seed() before it can be reset"

        torch.manual_seed(identical_seed)
        torch.cuda.manual_seed(identical_seed)
        import random
        np.random.seed(identical_seed)
        random.seed(identical_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @classmethod
    def set_tcpstore_port(cls, tcpstore_port):
        """Override the default port if it is in use in a distributed training session"""

        cls.__tcpstore_port = tcpstore_port
        if cls.__verbosity > 0:
            print(f"[set_tcpstore_port] TCPStore port set to {cls.__tcpstore_port} .")

    @classmethod
    def set_permutation_saving_params(cls, allow_permutation=False, save_permutation_graph=False, permutation_output_dir='.'):
        """This function is used to set the permutation saving related parameters."""

        cls.__allow_permutation = allow_permutation
        cls.__save_permutation_graph = save_permutation_graph
        cls.__permutation_output_dir = permutation_output_dir

        if cls.__verbosity > 0:
            print(f"[permutation_lib][set_permutation_saving_param] Set permutation saving related parameters\n\tAllow permutation: {cls.__alow_permutation}\n\tSave permutation graphs: {cls.__save_permutation_graph}\n\tPermutation graphs saving dir: {cls.__permutation_output_dir}")

    @classmethod
    def set_permutation_params_from_asp(cls, model, sparse_parameters, all_parameters, verbosity):
        """This function is used to set the permutation needed parameters from ASP class."""
        cls.__verbosity = verbosity

        if cls.__verbosity > 0:
            print("[set_permutation_params_from_asp] Set permutation needed parameters")
        cls.__model = model
        cls.__sparse_parameters = sparse_parameters
        cls.__all_parameters = all_parameters

        if cls.__verbosity > 1:
            sparse_param_names = [module_name+":"+p_name for (module_name, module, p_name, p, mask, pruned) in cls.__sparse_parameters]
            all_param_names = [module_name+":"+p_name for (module_name, module, p_name, p) in cls.__all_parameters]
            print(f"\tSparse parameter names: {sparse_param_names}\n\tAll parameter names: {all_param_names}")

        cls.__params_permuted_in_C = []
        cls.__params_permuted_in_K = []
        cls.__unpermuted_dims = []

    @classmethod
    def permute_model(cls, model, dump_fx_graph=False, save_dumped_fx_graph='./model_permutation_graph.json'):
        """Permute a model's weights in order to maintain more magnitude after enforcing the sparsity constraint."""

        if cls.__verbosity > 0:
            print("\n[permute_model] Permuting the model")

        # extract the output_dir, so all the intermediate fx_graph can be saved under that path
        extract_output_dir=os.path.split(save_dumped_fx_graph)[0]
        cls.__permutation_output_dir = extract_output_dir
        fx_graph, success_in_build_fx_graph = cls.build_fx_graph(model, dump_fx_graph=dump_fx_graph, save_dumped_fx_graph=save_dumped_fx_graph)

        if success_in_build_fx_graph:

            fx_graph_after_init_flags                   = cls.init_permutation_flags(fx_graph)
            fx_graph_after_find_real_parents            = cls.find_real_parents(fx_graph_after_init_flags)
            fx_graph_after_find_real_children           = cls.find_real_children(fx_graph_after_find_real_parents)
            fx_graph_after_making_groups                = cls.make_sibling_coparent_groups(fx_graph_after_find_real_children)
            fx_graph_after_fixup_concats                = cls.fixup_concats(fx_graph_after_making_groups)
            fx_graph_after_enforce_dimension_agreement  = cls.enforce_dimension_agreement(fx_graph_after_fixup_concats)
            fx_graph_after_propagate_flags              = cls.propagate_permutation_flags(fx_graph_after_enforce_dimension_agreement)

            start_time_search_for_good_permutation = time.perf_counter()
            fx_graph_after_find_permutations            = cls.find_permutations(fx_graph_after_propagate_flags)

            if torch.distributed.is_initialized():
                if cls.__verbosity > 0:
                    duration_search_for_good_permutation = time.perf_counter() - start_time_search_for_good_permutation
                    print(f"[permute_model] Rank {torch.distributed.get_rank()} completed search in {duration_search_for_good_permutation:.2f}s, waiting for others.", force=True)
                torch.distributed.barrier()

            duration_search_for_good_permutation = time.perf_counter() - start_time_search_for_good_permutation
            if cls.__verbosity > 0:
                print("\n[permute_model] Take {:.4f} seconds to finish search_for_good_permutation function.".format(duration_search_for_good_permutation))

            fx_graph_after_sync_permutations            = cls.sync_permutations(fx_graph_after_find_permutations)
            fx_graph_after_apply_permutations           = cls.apply_permutations(fx_graph_after_sync_permutations)
            cls.check_graph_for_unpermuted_nodes(fx_graph_after_apply_permutations)

            fx_graph = fx_graph_after_apply_permutations

        if cls.__save_permutation_graph:
            cls.save_graph_to_json(fx_graph, save_dumped_graph_path_with_name=os.path.join(cls.__permutation_output_dir, './model_graph_permutation_graph.json'))    # save the intermediate graph as JSON file for debugging

        return success_in_build_fx_graph


    @classmethod
    def get_permutation_stats(cls):
        """Return statistics for how many permutations were applied in various dimensions, used for testing"""

        return cls.__params_permuted_in_C, cls.__params_permuted_in_K, cls.__unpermuted_dims


    @classmethod
    def apply_permutation_in_C_dim(cls, node_name, permutation_sequence, dryrun):
        """This function is used to permutation for a node in C dim. (Only need to handle the weight of the node) """

        if cls.__verbosity > 1 and dryrun:
            print("[apply_permutation_in_C_dim] Permutation for node: \'{:}\' in C dim".format(node_name))

        if len(permutation_sequence) == 0:
            if cls.__verbosity >= 0:
                print(f"ERROR: [apply_permutation_in_C_dim] the permutation sequence for node {node_name} is empty, fail to apply permutation in C dim.")
            return False

        is_node_in_sparse_parameters = False
        success_permutation = False
        for module_name, module, p_name, p, mask, pruned in cls.__sparse_parameters:
            
            if node_name_matches(node_name, module_name):
                if cls.__verbosity > 2 and dryrun:
                    print("[apply_permutation_in_C_dim] find the node: \'{:}\' \'{:}\' in cls.__sparse_parameters, succeed to apply permutation in C dim.".format(node_name, p_name))
                is_node_in_sparse_parameters = True
                permutation_to_apply = permutation_sequence
                if p.shape[1] != len(permutation_sequence):               # assumed to be grouped convolutions or concatenated weights
                    if p.shape[1] % len(permutation_sequence) != 0:
                        return False
                    
                    permutation_to_apply = replicate_sequence(permutation_sequence, p.shape[1] // len(permutation_sequence))

                if not dryrun:
                    p.data.copy_(p[:, permutation_to_apply, ...])
                    cls.__params_permuted_in_C.append(node_name + "." + p_name)

                success_permutation = True
        if not is_node_in_sparse_parameters:
            # A special case: if the node itself not in sparse_module_names but one of its real_siblings in sparse_module_names, then the node will not do the permutation search, but it may need to apply the offline permutation in C dim according to the searched permutation sequence from its real_siblings in sparse_module_names
            try:
                for module_name_from_all_parameters, module_from_all_parameters, p_name_from_all_parameters, p_from_all_parameters in cls.__all_parameters:
                    
                    if node_name_matches(node_name, module_name_from_all_parameters) and p_name_from_all_parameters == "weight":
                        if cls.__verbosity > 3 and dryrun:
                            print("[apply_permutation_in_C_dim] cannot find the node: \'{:}\' \'{:}\' in cls.__sparse_parameters, but can find in cls.__all_parameters.".format(node_name, p_name_from_all_parameters))
                        permutation_to_apply = permutation_sequence
                        if p_from_all_parameters.shape[1] != len(permutation_sequence):               # assumed to be grouped convolutions
                            if p_from_all_parameters.shpae[1] % len(permutation_sequence) != 0:
                                return False
                            
                            permutation_to_apply = replicate_sequence(permutation_sequence, p_from_all_parameters.shape[1] // len(permutation_sequence))
                        
                        if not dryrun:
                            p_from_all_parameters.data.copy_(p_from_all_parameters[:, permutation_to_apply, ...])
                            cls.__params_permuted_in_C.append(node_name + "." + p_name_from_all_parameters)

                        success_permutation = True
                        if cls.__verbosity > 2 and dryrun:
                            print("[apply_permutation_in_C_dim] cannot find the node: \'{:}\' in cls.__sparse_parameters, after trying with cls.__all_parameters, succeed to apply permutation in C dim.".format(node_name))
            except:
                success_permutation = False
                if cls.__verbosity >= 0:
                    print("ERROR: [apply_permutation_in_C_dim] cannot find the node: \'{:}\' in cls.__sparse_parameters, after trying with cls.__all_parameters, still fail to apply permutation in C dim.".format(node_name))
        return success_permutation

    @classmethod
    def permute_attr(cls, node_name, permutation_sequence, fx_graph, dryrun):
        """ Permute a node's attributes. Somewhat hacky, assumes that we'll find exactly one dimension with a length matching the permutation's """

        assert 'attr' in fx_graph[node_name].keys()
        attr = fx_graph[node_name]['attr']
        if cls.__verbosity > 1:
            print(f"Found attribute {node_name} of shape {attr.shape}")
        found_perm = False
        for dim in range(len(attr.shape)):
            if attr.shape[dim] == len(permutation_sequence):
                if found_perm:
                    if cls.__verbosity > 0:
                        print(f"\tWARNING: {node_name} has already been permuted, but it's trying to happen again along another dimension {dim}.")

                    return False

                found_perm = True
                if cls.__verbosity > 1 and dryrun:
                    print(f"\tpermuting along dimension {dim}")

                if not dryrun:
                    # permute the dimension of interest to the front, permute within that dimension, then reset it
                    order = [c for c in range(len(attr.shape))]
                    order[0] = dim
                    order[dim] = 0
                    prmt = tuple(order)

                    temp_weight = torch.clone(attr)
                    temp_weight = torch.permute(temp_weight, prmt)
                    temp_weight.copy_(temp_weight[permutation_sequence, ...])
                    temp_weight = torch.permute(temp_weight, prmt)
                    attr.data.copy_(temp_weight)

                    cls.__params_permuted_in_K.append(node_name + "_" + str(dim))

        return found_perm


    @classmethod
    def apply_permutation_in_K_dim(cls, node_name, permutation_sequence, fx_graph, dryrun):
        """This function is used to permutation for a node in K dim. (Need to handle the weight/bias/running_mean/running_var of the node)"""

        if cls.__verbosity > 1:        
            print("[apply_permutation_in_K_dim] Permutation for node: \'{:}\' in K dim".format(node_name))

        if len(permutation_sequence) == 0:
            if cls.__verbosity >= 0:
                print("ERROR: [apply_permutation_in_K_dim] the permutation sequence is empty, fail to apply permutation in K dim.")
            return False

        # permute attribute nodes
        if 'attr' in fx_graph[node_name].keys():
            return cls.permute_attr(node_name, permutation_sequence, fx_graph, dryrun)

        # if we didn't store the attribute already, look in the modules' parameters
        is_node_in_all_parameters = False
        success_permutation = False

        for module_name, module, p_name, p in cls.__all_parameters:
            
            if node_name_matches(node_name, module_name):
                
                if cls.__verbosity > 1 and dryrun:
                    print("[apply_permutation_in_K_dim] find the node: \'{:}\' with \'{:}\' in cls.__all_parameters, may succeed to apply permutation in K dim.".format(node_name, p_name))
                is_node_in_all_parameters = True
                permutation_to_apply = permutation_sequence

                if p.shape[0] != len(permutation_sequence):               # assumed to be grouped convolutions
                    if cls.__verbosity > 2 and dryrun:
                        print(f"Mismatch in K dimension between found module {module_name} {p_name} for node {node_name}: permutation length {len(permutation_sequence)} but parameter shape in K {p.shape[0]}")

                    if p.shape[0] % len(permutation_sequence) != 0:
                        return False

                    permutation_to_apply = replicate_sequence(permutation_sequence, p.shape[0] // len(permutation_sequence))
                    
                    if cls.__verbosity > 1 and dryrun:
                        print("[apply_permutation_in_K_dim] the node: \'{:}\' with shape: \'{:}\' required replicating the permutation sequence with len \'{:}\' {:} times to succeed in applying the permutation in the K dimension.".format(node_name, p.shape, len(permutation_sequence), p.shape[0] // len(permutation_sequence)))
                else:
                    if cls.__verbosity > 1 and dryrun:
                        print("[apply_permutation_in_K_dim] the node: \'{:}\' with shape: \'{:}\', can match the size of permutation sequence with len: \'{:}\', succeed to apply permutation in K dim.".format(node_name, p.shape, len(permutation_sequence)))
                                
                if not dryrun:
                    p.data.copy_(p[permutation_to_apply, ...])
                    cls.__params_permuted_in_K.append(node_name + "." + p_name)

                success_permutation = True

        if not is_node_in_all_parameters:
            if cls.__verbosity >= 0:
                print("ERROR: [apply_permutation_in _K_dim] cannot find the node: \'{:}\' in cls.__all_parameters, fail to apply permutation in K dim.".format(node_name))
            success_permutation = False

        return success_permutation


    @classmethod
    def check_graph_for_unpermuted_nodes(cls, fx_graph):
        """Make sure that all permutable nodes/parameters were actually permuted and all GPUs agree"""

        for node_name in fx_graph.keys():
            node = fx_graph[node_name]

            if 'C_permutable' in node.keys() and node['C_permutable'] and not node['C_permuted']:
                sibling_group_id = node['sibling_group_id']
                if node['is_real'] and cls.__group_data['skipped_sibling_groups'][sibling_group_id] is None:
                    if cls.__verbosity >= 0:
                        print(f"{node_name} was C_permutable in a not skipped sibling group but was not permuted along C! {node}")
                    cls.__unpermuted_dims.append(node_name + "_C")

            if 'K_permutable' in node.keys() and node['K_permutable'] and not node['K_permuted']:
                coparent_group_id = node['coparent_group_id']
                if node['is_real'] and cls.__group_data['skipped_coparent_groups'][coparent_group_id] is None:
                    if cls.__verbosity >= 0:
                        print(f"{node_name} was K_permutable in a not skipped coparent group but was not permuted along K! {node}")
                    cls.__unpermuted_dims.append(node_name + "_K")

        if cls.__verbosity > 0:
            print(f"[check_graph_for_unpermuted_nodes] found nodes that missed permutations along {len(cls.__unpermuted_dims)} dimensions.")

        # make sure all GPUs agree
        if torch.distributed.is_initialized():
            cls.__unpermuted_dims = sorted(cls.__unpermuted_dims)
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            dist_store = torch.distributed.TCPStore("127.0.0.1", cls.__tcpstore_port, world_size, rank==0)
            torch.distributed.barrier()

            dist_store.set(str(rank), ','.join(cls.__unpermuted_dims))
            torch.distributed.barrier()

            if rank == 0:
                my_list = dist_store.get('0').decode()
                
                for peer in range(1, world_size):
                    peer_list = dist_store.get(str(peer)).decode()
                    assert my_list == peer_list, f"peer {peer} disagreed with rank 0's list of unpermuted nodes: \n{my_list}\n{peer_list}"
                    
    
    @classmethod
    def find_sparse_parameters_for_node(cls, node_name):
        """If the node has parameters that are in the trackd sparse parameter list, find them and reshape to a 2D tensor with channels last"""
        node_weight = None

        # check the sparse parameters
        for module_name, module, p_name, p, mask, pruned in cls.__sparse_parameters:
            
            if node_name_matches(node_name, module_name):
                node_weight = torch.zeros_like(p)
                node_weight.copy_(p)

        # if we found something, reshape to concatenate along the same dimension
        if node_weight is not None:
            # Need to handle the concat for layers with different R & S
            shape = node_weight.shape
            # 1d-tensor
            if len(shape) == 1:
                node_weight = node_weight.view(1, shape[0])
            # 2d-tensor (K, C)
            elif len(shape) == 2:
                node_weight = node_weight.view(shape[0], shape[1])
            # 3d-tensor (K, C, R)
            elif len(shape) == 3:
                node_weight = node_weight.permute(0,2,1).contiguous().view(shape[0]*shape[2], shape[1])
            # 4d-tensor (K, C, R, S)
            elif len(shape) == 4:
                # convs
                node_weight = node_weight.permute(2,3,0,1).contiguous().view(shape[2]*shape[3]*shape[0], shape[1])

        return node_weight

    @classmethod
    def find_permutation_for_matrix_group(cls, matrix_group):
        """Find a good permutation for some matrix (which may be concatenated matrices that require the same permutation)"""

        if cls.__verbosity > 1:
            print(f"Searching for a good permutation for this sibling group of shape {matrix_group.shape}")

        permutation_found = False
        num_channels = matrix_group.shape[1]
        group_permutation = [c for c in range(num_channels)]

        # automatic check for skipping the permutation search process
        original_magnitude = (torch.abs(matrix_group)).sum(dtype=torch.float64)
        pruned_magnitude = sum_after_2_to_4(matrix_group.cpu().detach().numpy())
        diff_ratio = abs(original_magnitude - pruned_magnitude)/original_magnitude
        epsilon = 1e-3

        if cls.__verbosity > 1:
            print("\n[search_for_good_permutation] Original element abs sum: {:}, Pruned element abs sum: {:}, Diff ratio: {:}".format(original_magnitude, pruned_magnitude, diff_ratio))

        start_time_accelerated_search_for_good_permutation = time.perf_counter()
        if diff_ratio < epsilon:
            if cls.__verbosity > 2:
                print("[search_for_good_permutation] Original element abs sum is almost same as the pruned element abs sum, further permutation search will not help, skipping!")
        
        else:
            if cls.__verbosity > 2:
                print("[search_for_good_permutation] Original element abs sum is different from the pruned element abs sum, further permutation search will help, continue with the permutation search!")

            # call the permutation search CUDA kernels as ASP extension.
            # users can provide prefer search strategy by providing a valid 'search_options' as a dictionary,
            # or users can implement their customized 'accelerated_search_for_good_permutation' function.
            search_options = {}
            # No.1 Strategy: Exhaustive Search
            search_options['strategy'] = 'exhaustive'
            search_options['stripe_group_size'] = 8
            search_options['escape_attempts'] = 100
            # No.2 Strategy: Progressive Channel Swap Search
            # search_options['strategy'] = 'progressive channel swap'
            # search_options['progressive_search_time_limit'] = 10
            # search_options['improvement_threshold'] = 1e-9

            # permutation search time is too long for matrix_group with large channel num
            # change from Exhaustive Search to Progressive Channel Swap Search based on input matrix_group size
            if num_channels > 2048:
                search_options = {}
                search_options['strategy'] = 'progressive channel swap'
                search_options['progressive_search_time_limit'] = 120
                search_options['improvement_threshold'] = 1e-9

            if cls.__verbosity > 1:
                print(f"[search_for_good_permutation] search options: {search_options}")

            group_permutation = accelerated_search_for_good_permutation(matrix_group, options=search_options, verbosity=cls.__verbosity)
            permutation_found = True

        if cls.__verbosity > 1:
            duration_accelerated_search_for_good_permutation = time.perf_counter() - start_time_accelerated_search_for_good_permutation
            permuted_magnitude = sum_after_2_to_4(matrix_group.cpu().detach().numpy()[:,group_permutation])
            print("[search_for_good_permutation] Take {:.4f} seconds to finish accelerated_search_for_good_permutation function and with final magnitude {:}.".format(duration_accelerated_search_for_good_permutation, permuted_magnitude))

        return group_permutation, permutation_found

    @classmethod
    def skip_sibling_group(cls, fx_graph, sibling_group_id, reason):
        """Keep track of sibling groups that do not have permutations applied"""

        # grab a parent to get the coparent group id
        sibling_group = cls.__group_data['sibling_groups'][sibling_group_id]
        a_sibling = list(sibling_group)[0]
        a_parent = fx_graph[a_sibling]['real_parents'][0]
        coparent_group_id = fx_graph[a_parent]['coparent_group_id']

        if cls.__verbosity > 1:
            print(f"Skipping permutations for Sibling Group {sibling_group_id} and Coparent Group {coparent_group_id}: {reason}")

        cls.__group_data['skipped_sibling_groups'][sibling_group_id] = reason
        cls.__group_data['skipped_coparent_groups'][coparent_group_id] = reason

    @classmethod
    def collect_sparse_weights(cls, fx_graph, sibling_group, sibling_group_C_param):
        """Gather all sparse weights for a sibling group (to serve as input to the permutation search)"""

        matrix_group = None

        for sibling in sibling_group:
            node_weight = cls.find_sparse_parameters_for_node(sibling)
            
            if node_weight is not None:
                # reshape due to siblings with grouped convolutions of different sizes
                assert node_weight.shape[1] % sibling_group_C_param == 0, f"sibling {sibling}'s weights' C={node_weight.shape[1]} must be even multiple of the sibling group's C parameter {sibling_group_C_param}"
                node_weight = torch.reshape(node_weight, (-1, sibling_group_C_param))

                if matrix_group is None:
                    matrix_group = node_weight
                else:
                    try:
                        matrix_group = torch.cat((matrix_group, node_weight), dim = 0) # concat the weights in the K dimension, keep the same C dimension

                    except:
                        if cls.__verbosity >= 0:
                            print("ERROR: [search_for_good_permutation][warning] cannot merge the weight for node: \'{:}\', with its weight shape: \'{:}\', the matrix_group shape: \'{:}\'.".format(sibling, node_weight.size(), matrix_group.size()))
                        continue
                if cls.__verbosity > 2:
                    print("[search_for_good_permutation] have merged the weight for node: \'{:}\', with its weight shape: \'{:}\', the matrix_group shape: \'{:}\'.".format(sibling, node_weight.size(), matrix_group.size()))
            else:
                if cls.__verbosity > 2:
                    print(f"[search_for_good_permutation] not adding dense weights for node {sibling} to the group")

        return matrix_group

    @classmethod
    def find_sibling_group_permutation(cls, fx_graph, sibling_group_id):
        """"Find a good permutation for some sibling group"""

        if cls.__verbosity > 1:
            print(f"Finding permutation for sibling group {sibling_group_id}")

        cls.reset_seed()

        sibling_group = cls.__group_data['sibling_groups'][sibling_group_id]
        sibling_group_C_param = int(cls.__group_data['sibling_group_C_params'][sibling_group_id])

        if sibling_group_C_param % 4 != 0 or sibling_group_C_param < 8:
            cls.skip_sibling_group(fx_graph, sibling_group_id, f"Useless C: {sibling_group_C_param}")
            return

        # collect *sparse* weights from all siblings, get the coparent group
        matrix_group = cls.collect_sparse_weights(fx_graph, sibling_group, sibling_group_C_param)

        # early-out if no siblings are sparse
        if matrix_group is None: 
            cls.skip_sibling_group(fx_graph, sibling_group_id, 'Dense')
            return

        # find a good permutation
        group_permutation, found = cls.find_permutation_for_matrix_group(matrix_group)

        # if no permutation was found, we didn't need it (input already sparse)
        if not found:
            cls.skip_sibling_group(fx_graph, sibling_group_id, 'Not needed')
            return

        if cls.__verbosity > 2:
            print(f"Permutation for sibling group {sibling_group_id}: {group_permutation}")

        cls.__group_data['sibling_group_permutations'][sibling_group_id] = group_permutation


    @classmethod
    def permute_sibling_group(cls, fx_graph, sibling_group_id, group_permutation):
        """Apply a permutation to some sibling group"""

        if cls.__verbosity > 1:
            print(f"Attempting to permute sibling group {sibling_group_id}")
        
        sibling_group = cls.__group_data['sibling_groups'][sibling_group_id]
        
        # apply the permutation in two steps: first, a dry run to find any issues.
        # if there were no issues, actually apply the permutation in the second step.
        success = True
        coparent_group_id = None
        for dryrun in [True, False]:
            # apply that permutation to the siblings' C dimension
            for sibling in sibling_group:
                assert fx_graph[sibling]['C_permutable'] and not fx_graph[sibling]['C_permuted']
                sibling_permuted = cls.apply_permutation_in_C_dim(sibling, group_permutation, dryrun)
                if dryrun:
                    success = success and sibling_permuted
                else:
                    assert sibling_permuted, "shouldn't fail permuting siblings after the dry run"
                    fx_graph[sibling]['C_permuted'] = sibling_permuted

                a_parent = fx_graph[sibling]['real_parents'][0]
                if coparent_group_id is None:
                    coparent_group_id = fx_graph[a_parent]['coparent_group_id']
                else:
                    assert coparent_group_id == fx_graph[a_parent]['coparent_group_id'], f"parent {a_parent} must belong to the same coparent group {coparent_group_id}, not {fx_graph[a_parent]['coparent_group_id']}"

            # grab the parents (and co-parents) and apply to their K dimension
            coparents = cls.__group_data['coparent_groups'][coparent_group_id]
            for coparent in coparents:
                assert fx_graph[coparent]['K_permutable'] and not fx_graph[coparent]['K_permuted']
                coparent_permuted = cls.apply_permutation_in_K_dim(coparent, group_permutation, fx_graph, dryrun)
                if dryrun:
                    success = success and coparent_permuted
                else:
                    assert coparent_permuted, "shouldn't fail permuting coparents after the dry run"
                    fx_graph[coparent]['K_permuted'] = coparent_permuted

                children_permuted = cls.apply_permutation_in_K_dim_to_children(fx_graph, coparent, group_permutation, dryrun)
                if dryrun:
                    success = success and children_permuted
                else:
                    assert children_permuted, "shouldn't fail permuting coparents' children after the dry run"

            if not success:
                cls.skip_sibling_group(fx_graph, sibling_group_id, "dryrun_failure")

                if cls.__verbosity > 0:
                    print(f"There was an issue permuting sibling group {sibling_group_id}, skipping it to preserve network quality.")

                break
            
                    
    @classmethod
    def apply_permutation_in_K_dim_to_children(cls, fx_graph, node_name, permutation, dryrun):
        """Apply a permutation along K to the children of some node"""

        success = True
        children = fx_graph[node_name]['children']
        if cls.__verbosity > 2 and dryrun:
            print(f"Applying a permutation in K to children of {node_name} : {children}")

        # apply the permutation along K to children as necessary
        for child in children:
            if 'is_real' in fx_graph[child].keys() and fx_graph[child]['is_real']:
                if cls.__verbosity > 3 and dryrun:
                    print(f"\tFound a real child {child}, not permuting it or its children along K")
            else:
                if 'module_type' not in fx_graph[child].keys() or fx_graph[child]['module_type'] == 'None':
                    if cls.__verbosity > 3 and dryrun:
                        print(f"\tPermuting children of non-module {child} along K")
                    success = success and cls.apply_permutation_in_K_dim_to_children(fx_graph, child, permutation, dryrun)
                elif not fx_graph[child]['C_permutable']:
                    if fx_graph[child]['K_permutable'] and not fx_graph[child]['K_permuted']:
                        if cls.__verbosity > 2 and dryrun:
                            print(f"\tPermuting {child} along K")
                        child_permuted = cls.apply_permutation_in_K_dim(child, permutation, fx_graph, dryrun)
                        success = success and child_permuted
                        if not dryrun:
                            fx_graph[child]['K_permuted'] = child_permuted
                        assert fx_graph[child]['K_passthru']

                    if fx_graph[child]['K_passthru']:
                        success = success and cls.apply_permutation_in_K_dim_to_children(fx_graph, child, permutation, dryrun)
                    else:
                        if cls.__verbosity >= 0:
                            print(f"\t!! ERROR {child} was a not real module that was not K_passthru")

        return success
    
    @classmethod
    def defer_prints(cls):
        """Collect prints from this rank in distributed mode to avoid interleaved output"""

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            cls.__new_stdout = io.StringIO(str(torch.distributed.get_rank()))
            cls.__builtin_print = __builtin__.print

            def deferred_print(*args, **kwargs):
                try:  # see if torchvision examples has suppressed other ranks with the force argument
                    cls.__builtin_print(*args, file=cls.__new_stdout, force=True, **kwargs)
                except:
                    cls.__builtin_print(*args, file=cls.__new_stdout, **kwargs)

            __builtin__.print = deferred_print


    @classmethod
    def resume_prints(cls):
        """Emit the collected outputs from this rank, resume immediate printing"""

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            output = cls.__new_stdout.getvalue()
            __builtin__.print = cls.__builtin_print

            try:
                print(output, force=True)
            except:
                print(output)

    @classmethod
    def find_permutations(cls, fx_graph):
        """Search for permutations for all sibling groups"""

        for sibling_group_id in cls.__group_data['sibling_groups'].keys():

            search_this_group = True
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()

                if sibling_group_id % world_size != rank:
                    search_this_group = False
            
            cls.__group_data['sibling_group_permutations'][sibling_group_id] = None
            if search_this_group:  

                cls.defer_prints()

                sibling_group = cls.__group_data['sibling_groups'][sibling_group_id]
                test_node_name = list(sibling_group)[0]
                if not fx_graph[test_node_name]['C_permutable']:
                    if cls.__verbosity > 1:
                        print(f"Skipping permutation for sibling group {sibling_group_id} since it does not allow permutations along C")

                else:
                    if cls.__verbosity > 1:
                        print(f"Sibling group {sibling_group_id} can permute along C, permuting it")
            
                    cls.find_sibling_group_permutation(fx_graph, sibling_group_id)

                cls.resume_prints()

        return fx_graph

    @classmethod
    def sync_permutations(cls, fx_graph):
        """If multiple GPUs were involved in finding permutations, make sure everyone's in sync"""

        if not torch.distributed.is_initialized():
            return fx_graph
        
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        dist_store = torch.distributed.TCPStore("127.0.0.1", cls.__tcpstore_port, world_size, rank==0)

        if cls.__verbosity > 0:
            print(f"Syncing permutations found among world size {world_size}")

        torch.distributed.barrier()
        for sibling_group_id in sorted(cls.__group_data['sibling_groups'].keys()):
            src_rank = sibling_group_id % world_size

            if src_rank == rank:
                to_send = cls.__group_data['sibling_group_permutations'].get(sibling_group_id, None)
                skip_reason = None
                if to_send is None:
                    skip_reason = cls.__group_data['skipped_sibling_groups'].get(sibling_group_id, None)
                    if skip_reason is None:
                        to_send = ''
                    else:
                        to_send = 'skip'
                else:
                    to_send = ','.join(str(c) for c in to_send)

                dist_store.set(str(sibling_group_id), to_send)
                if skip_reason is not None:
                    dist_store.set(f"skip {sibling_group_id}", skip_reason)
                
                if cls.__verbosity > 1:
                    print(f"{rank}: stored permutation for sibling group {sibling_group_id}", force=True)

        torch.distributed.barrier()
        for sibling_group_id in sorted(cls.__group_data['sibling_groups'].keys()):
            permutation = dist_store.get(str(sibling_group_id)).decode()

            if permutation == 'skip':
                permutation = None
                skip_reason = dist_store.get(f"skip {sibling_group_id}").decode()
                cls.skip_sibling_group(fx_graph, sibling_group_id, skip_reason)
            else:
                if len(permutation) == 0:
                    permutation = None
                else:
                    permutation = [int(c) for c in permutation.split(',')]

            cls.__group_data['sibling_group_permutations'][sibling_group_id] = permutation
            

            if cls.__verbosity > 1:
                print(f"Got permutation for sibling group {sibling_group_id}")

        torch.distributed.barrier()
        return fx_graph

    @classmethod
    def apply_permutations(cls, fx_graph):
        """Apply all the permutations that were found to the network appropriately"""

        for sibling_group_id in cls.__group_data['sibling_group_permutations'].keys():

            permutation = cls.__group_data['sibling_group_permutations'][sibling_group_id]

            if permutation is not None:
                cls.permute_sibling_group(fx_graph, sibling_group_id, permutation)

        return fx_graph

    @staticmethod
    def insert_MHA_out_proj(fx_graph, MHA_node, verbosity):
        """MHA nodes have a hidden out_proj node, so insert it and fix up neighboring nodes"""

        if verbosity > 1:
            print(f"Inserting MHA out_proj for node {MHA_node}")
        out_proj_node_name = MHA_node + ".out_proj"
        # insert the new node
        fx_graph[out_proj_node_name] = {}
        fx_graph[out_proj_node_name]['parents'] = [MHA_node]
        fx_graph[out_proj_node_name]['children'] = fx_graph[MHA_node]['children']
        fx_graph[MHA_node]['children'] = [out_proj_node_name]

        # set the new node's properties
        fx_graph[out_proj_node_name]['fx_op'] = 'call_module'
        fx_graph[out_proj_node_name]['module_type'] = 'torch.nn.modules.linear.Linear'
        fx_graph[out_proj_node_name]['groups_param'] = 'None'
        fx_graph[out_proj_node_name]['C_param'] = fx_graph[MHA_node]['C_param']
        fx_graph[out_proj_node_name]['K_param'] = fx_graph[MHA_node]['K_param']
        fx_graph[out_proj_node_name]['sibling_group_id'] = None
        fx_graph[out_proj_node_name]['coparent_group_id'] = None

        # set permutation flags
        fx_graph[out_proj_node_name]['C_permutable'] = False
        fx_graph[MHA_node]['K_permutable'] = False
        fx_graph[MHA_node]['C_permutable'] = True
        fx_graph[out_proj_node_name]['K_permutable'] = True
        fx_graph[out_proj_node_name]['K_passthru'] = False
        fx_graph[out_proj_node_name]['C_permuted'] = False
        fx_graph[out_proj_node_name]['K_permuted'] = False
        fx_graph[out_proj_node_name]['is_real'] = True

        if verbosity > 2:
            print(f"\tUpdated: {MHA_node}: {fx_graph[MHA_node]}")
            print(f"\tAdded: {out_proj_node_name}: {fx_graph[out_proj_node_name]}")

        # update any nodes that thought their parent was the MHA node
        for node in fx_graph.keys():
            parents = fx_graph[node]['parents']
            if node != out_proj_node_name and MHA_node in parents:
                parents.remove(MHA_node)
                parents.append(out_proj_node_name)
                fx_graph[node]['parents'] = parents
                if verbosity > 2:
                    print(f"\tUpdated parents of {node}: {fx_graph[node]}")

        return fx_graph

    @staticmethod
    def init_grouped_conv_permutation_flags(fx_graph, node_name, node_groups, verbosity):
        """Handle grouped convolutions to make dimensions match"""

        node_C = int(fx_graph.get(node_name).get('C_param'))
        node_K = int(fx_graph.get(node_name).get('K_param'))
        node_groups = int(node_groups)

        if verbosity > 2:
            print(f"\t{node_name} pre-divide C: {node_C}, K: {node_K}, G: {node_groups}")
        assert node_C % node_groups == 0
        node_C = int(node_C / node_groups)
        fx_graph[node_name]['C_param'] = str(node_C)
        if verbosity > 2:
            print(f"\t{node_name} post-divide C: {node_C}, K: {node_K}, G: {node_groups}")

        if node_C == 1:                         # G == C (C is pre-divided by G)
            if node_groups == node_K:                       # true depthwise, G == C == K (C will be pre-divided by G)
                fx_graph[node_name]['K_permutable'] = True
                fx_graph[node_name]['K_permuted'] = False
                fx_graph[node_name]['K_passthru'] = True
                fx_graph[node_name]['is_real'] = False
            #else:                                          # G != K, handling a permutation along K would be very tricky and not likely useful

        else:                                   # G != C
            if node_C > 4 and node_C % 4 == 0:              # permutations only help if there's more than one 2:4 pruning group
                fx_graph[node_name]['C_permutable'] = True
                fx_graph[node_name]['C_permuted'] = False


    @classmethod
    def init_permutation_flags(cls, fx_graph):
        """Set the permutation flags for each node based only on that node's module type and parameters"""

        if cls.__verbosity > 0:
            print("\n[init_permutation_flags] Initialize the permutation flags for each node according to module type and parameters")

        # initialize some graph-wide trackers
        cls.__group_data = {}
        cls.__group_data['next_sibling_group_id'] = 0
        cls.__group_data['next_coparent_group_id'] = 0
        cls.__group_data['sibling_groups'] = {}
        cls.__group_data['sibling_group_permutations'] = {}
        cls.__group_data['sibling_group_C_params'] = {}
        cls.__group_data['skipped_sibling_groups'] = {}
        cls.__group_data['coparent_groups'] = {}
        cls.__group_data['skipped_coparent_groups'] = {}

        # track MHA nodes
        MHA_nodes = []

        # initialize each node's details
        for node_name in fx_graph.keys():
            fx_node = fx_graph.get(node_name)
            node_module_type = fx_node.get('module_type')
            if cls.__verbosity > 1:
                if node_module_type == 'get_attr':
                    print(f"Initializing node {node_name} of type {node_module_type}")
                else:
                    print(f"Initializing node {node_name} of type {node_module_type}: {fx_node}")

            # default for all nodes: don't allow anything
            if node_module_type is not None:
                fx_graph[node_name]['C_permutable'] = False         # does this node have parameters that can be permuted in C    
                fx_graph[node_name]['K_permutable'] = False         # does this node have parameters that can be permuted in K
                fx_graph[node_name]['K_passthru'] = False           # does this node need to pass a K permutation to its parents
                fx_graph[node_name]['is_real'] = False
                fx_graph[node_name]['C_permuted'] = False
                fx_graph[node_name]['K_permuted'] = False
    
                # initialize sibling and coparent groups
                fx_graph[node_name]['sibling_group_id'] = None
                fx_graph[node_name]['coparent_group_id'] = None

                # update each node to be more permissive if supported
                if node_module_type in cls.__permutation_target_module_types:
                    fx_graph[node_name]['is_real'] = True
                    node_groups = fx_graph.get(node_name).get('groups_param')

                    if (node_groups in ['None', '1']):      # no groups, no constraints
                        fx_graph[node_name]['C_permutable'] = True
                        fx_graph[node_name]['K_permutable'] = True

                    else:                                   # handle groups
                        Permutation.init_grouped_conv_permutation_flags(fx_graph, node_name, node_groups, cls.__verbosity)

                elif node_module_type in cls.__permute_K_and_passthru_module_types:
                    fx_graph[node_name]['K_permutable'] = True
                    fx_graph[node_name]['K_passthru'] = True
                    fx_graph[node_name]['is_real'] = False

                elif node_module_type in cls.__simple_passthru_module_types:
                    fx_graph[node_name]['K_passthru'] = True
                    fx_graph[node_name]['is_real'] = False

                elif node_module_type in cls.__disallow_permutations_module_types:
                    fx_graph[node_name]['is_real'] = True
                    fx_graph[node_name]['C_param'] = 1
                    fx_graph[node_name]['K_param'] = 1
                    fx_graph[node_name]['groups_param'] = 1

                elif 'activation' in node_module_type:
                    if cls.__verbosity > 0:
                        print(f"WARNING: how should permutation flags be initialized for node {node_name} of module type {node_module_type}?  Found 'activation', assuming simple passthru behavior.")
                    fx_graph[node_name]['K_passthru'] = True
                    fx_graph[node_name]['is_real'] = False

                else:
                    if cls.__verbosity > 0:
                        print(f"WARNING: how should permutation flags be initialized for node {node_name} of module type {node_module_type}?  Defaulting to strict, disallowing permutations around it.")
                    # is_real coupled with disallowed C and K permutations will poison real parents and real children
                    fx_graph[node_name]['is_real'] = True
                    # dummy entries:
                    fx_graph[node_name]['C_param'] = 1
                    fx_graph[node_name]['K_param'] = 1
                    fx_graph[node_name]['groups_param'] = 1

                # MHA nodes only handle the in_proj, need to add out_proj nodes explicitly
                # keep track here so we can iterate directly and change fx_graph keys
                if node_module_type == 'torch.nn.modules.activation.MultiheadAttention':
                    MHA_nodes.append(node_name)

            if cls.__verbosity > 1:
                if node_module_type == 'get_attr':
                    print(f"\tInitialized node {node_name} of type {node_module_type}")
                else:
                    print(f"\tInitialized node {node_name} of type {node_module_type}: {fx_graph[node_name]}")

        for MHA_node in MHA_nodes:
            fx_graph = Permutation.insert_MHA_out_proj(fx_graph, MHA_node, cls.__verbosity)

        return fx_graph

    @staticmethod
    def collect_siblings(fx_graph, node_name, all_siblings):
        """Recursively build a set of some node's siblings in the graph"""

        # find all siblings of the requested node
        siblings = set()
        parents = fx_graph.get(node_name).get('real_parents')
        for parent in parents:
            children = fx_graph.get(parent).get('real_children')
            for child in children:
                siblings.add(child)

        # separate the new siblings, since we'll need to process them recursively
        new_siblings = siblings.difference(all_siblings)
        # update the final list with just the new elements
        all_siblings.update(new_siblings)

        for new_sibling in new_siblings:
            all_siblings = Permutation.collect_siblings(fx_graph, new_sibling, all_siblings)

        return all_siblings

    @staticmethod
    def propagate_sibling_group(fx_graph, all_siblings, verbosity):
        """Check a sibling group for ability to be permuted, disallow all siblings and coparents if there's an issue"""

        made_change = False
        allow_C = True
        for sibling in all_siblings:
            pre_check = allow_C
            allow_C = allow_C and fx_graph[sibling]['C_permutable']
            if allow_C != pre_check:
                if verbosity > 2:
                    if fx_graph[sibling]['module_type'] == 'get_attr':
                        print(f"\tnode {sibling} has poisoned the sibling group of {all_siblings}")
                    else:
                        print(f"\tnode {sibling} has poisoned the sibling group of {all_siblings}: {fx_graph[sibling]}")
                break

        if not allow_C:
            for sibling in all_siblings:
                made_change = made_change or fx_graph[sibling]['C_permutable']
                fx_graph[sibling]['C_permutable'] = False

                # only disable permutation along K for parents if this node cannot passthru, either
                if not fx_graph[sibling]['K_passthru']:
                    sibling_parents = fx_graph[sibling]['real_parents']
                    for sibling_parent in sibling_parents:
                        made_change = made_change or fx_graph[sibling_parent]['K_permutable'] or fx_graph[sibling_parent]['K_passthru']
                        fx_graph[sibling_parent]['K_permutable'] = False
                        fx_graph[sibling_parent]['K_passthru'] = False

        return made_change

    @staticmethod
    def collect_coparents(fx_graph, node_name, all_coparents):
        """Recursively build a set of all coparents of a particular node in the graph"""
        
        # find all coparents of the requested node
        coparents = set()
        children = fx_graph.get(node_name).get('real_children')
        for child in children:
            parents = fx_graph.get(child).get('real_parents')
            for parent in parents:
                coparents.add(parent)
                
                # coparents are used to restrict what nodes can be permuted along C, so we need to track if the current parents also pass their K permutations up
                if fx_graph[parent]['K_passthru']:
                    grandparents = fx_graph[parent]['real_parents']
                    for grandparent in grandparents:
                        coparents = coparents.union(Permutation.collect_coparents(fx_graph, grandparent, coparents))

        # separate the new coparents, since we'll need to process them recursively
        new_coparents = coparents.difference(all_coparents)
        # update the final list with just the new elements
        all_coparents.update(new_coparents)

        for new_coparent in new_coparents:
            all_coparents = Permutation.collect_coparents(fx_graph, new_coparent, all_coparents)

        return all_coparents

    @staticmethod
    def propagate_coparent_group(fx_graph, all_coparents, verbosity):
        """Check a coparent group for ability to be permuted, disallow all fellow coparents and children if there's an issue"""

        # see if all coparents agree that K can be permuted
        allow_K = True
        made_change = False
        for coparent in all_coparents:
            pre_check = allow_K
            allow_K = allow_K and (fx_graph[coparent]['K_permutable'] or fx_graph[coparent]['K_passthru'])
            if allow_K != pre_check:
                if verbosity > 2:
                    if fx_graph[coparent]['module_type'] == 'get_attr':
                        print(f"\tnode {coparent} has poisoned the coparent group of {all_coparents}")
                    else:
                        print(f"\tnode {coparent} has poisoned the coparent group of {all_coparents}: {fx_graph[coparent]}")
                break

        # if anyone says no, force everyone to 'no', keep track of updated state
        if not allow_K:
            for coparent in all_coparents:
                # all coparents can no longer be permuted along K
                if fx_graph[coparent]['K_permutable'] or fx_graph[coparent]['K_passthru']:
                    made_change = True

                    fx_graph[coparent]['K_permutable'] = False
                    fx_graph[coparent]['K_passthru'] = False

                # children of coparents can't be permuted along C
                coparent_children = fx_graph[coparent]['real_children']
                for coparent_child in coparent_children:
                    if fx_graph[coparent_child]['C_permutable']:
                        fx_graph[coparent_child]['C_permutable'] = False
                        made_change = True

        return made_change

    @classmethod
    def fixup_concats(cls, fx_graph):
        """concat operations/modules may concatenate along the channel dimension, which requires special handling (like grouped convs)"""

        if cls.__verbosity > 0:
            print("[fixup_concats]")

        for node_name in fx_graph.keys():
            fx_node = fx_graph[node_name]
            if fx_node.get('module_type') == 'concat':
                # get real parents, find GCD of their Ks
                node_real_parents = fx_node['real_parents']

                # some concats are at the front of networks (googlenet)
                if len(node_real_parents) == 0:
                    continue

                parents_K_params = []
                for parent in node_real_parents:
                    parent_K_param = int(fx_graph[parent]['K_param'])
                    parents_K_params.append(parent_K_param)
                    fx_graph[parent]['allow_K_mismatch'] = 'concat op'
   
                # if grouped convolutions make the input channels different among siblings different sizes,
                # restrict the permutation atom to the greatest common divisor so it can be tiled as needed for each sibling (and parent)
                if cls.__verbosity > 2:
                    print(f"\tfixing up concat node {node_name}, found parents' {node_real_parents} Ks: {parents_K_params}")

                children_GCD_param = str(np.gcd.reduce(parents_K_params))

                # set this to GCD of children's sibling group
                sibling_group_id = -1
                node_real_children = fx_node['real_children']
                for child in node_real_children:
                    sibling_group_id = fx_graph[child]['sibling_group_id']
                    fx_graph[child]['C_param'] = children_GCD_param
                    
                old_children_GCD = cls.__group_data['sibling_group_C_params'][sibling_group_id]
                cls.__group_data['sibling_group_C_params'][sibling_group_id] = children_GCD_param

                # fixup this node's dimensions
                # use the functionality of grouped convolutions
                fx_node['C_param'] = children_GCD_param
                fx_node['K_param'] = old_children_GCD
                fx_node['groups_param'] = str(int(old_children_GCD) // int(children_GCD_param))

                if cls.__verbosity > 2:
                    print(f"\tfixed up concat node {node_name}, found GCD of parents' {node_real_parents} K to be {children_GCD_param}, updated children's {node_real_children} C_params and sibling group {sibling_group_id} GCD")
                    print(f"\tthis node now: {fx_node}")

        return fx_graph

    @classmethod
    def enforce_dimension_agreement(cls, fx_graph):
        """Check all nodes' channel dimensions against parents and children to make sure they agree; e.g. flatten ops may change these dimensions"""

        if cls.__verbosity > 0:
            print("[enforce_dimension_agreement]")

        for node_name in fx_graph.keys():
            fx_node = fx_graph[node_name]
            if 'is_real' in fx_node.keys() and fx_node['is_real']:

                # enforce this node's input dimension matches its parents' output dimensions
                node_C = int(fx_node['C_param'])
                node_K = int(fx_node['K_param'])

                if fx_graph[node_name]['groups_param'] not in ['1', 'None']:
                    node_C = node_C * int(fx_node['groups_param'])

                node_real_parents = fx_node['real_parents']
                if len(node_real_parents) == 0:
                    if cls.__verbosity > 1:
                        print(f"\t{node_name} has no real parents, disabling permutations along C")
                    fx_graph[node_name]['C_permutable'] = False
                else:
                    for real_parent in node_real_parents:
                        parent_K = int(fx_graph[real_parent]['K_param'])
                        ignore_mismatch = fx_graph[real_parent].get('allow_K_mismatch')

                        if ignore_mismatch is not None:
                            if cls.__verbosity > 1:
                                print(f"\tIgnoring dimension mismatch between {node_name} (C={node_C}) and its parent {real_parent} (K={parent_K}) as requested: {ignore_mismatch}")

                        elif parent_K >= 0 and node_C != parent_K:
                            if cls.__verbosity > 1:
                                print(f"\tDimensions mismatch between {node_name} (C={node_C}) and its parent {real_parent} (K={parent_K}), disallowing the relevant permutations")

                            fx_graph[node_name]['C_permutable'] = False
                            fx_graph[real_parent]['K_permutable'] = False

                            if cls.__verbosity > 2:
                                print(f"\t{fx_graph[node_name]}\n\t{fx_graph[real_parent]}")

                if len(fx_graph[node_name]['real_children']) == 0:
                    if cls.__verbosity > 1:
                        print(f"\t{node_name} has no real children, disabling permutations along K")
                    fx_graph[node_name]['K_permutable'] = False

        return fx_graph

    @classmethod
    def make_sibling_coparent_groups(cls, fx_graph):
        """Traverse all real nodes in the graph and collect their siblings and coparents"""

        if cls.__verbosity > 0:
            print("[make_sibling_coparent_groups]")

        for node_name in fx_graph.keys():
            fx_node = fx_graph[node_name]
            
            if 'is_real' in fx_node.keys() and fx_node['is_real']:

                sibling_group_id = fx_node['sibling_group_id']
                if sibling_group_id is None: # need to make a new sibling group for this node
                    all_siblings = cls.collect_siblings(fx_graph, node_name, set([node_name]))
                    all_siblings = sorted(all_siblings) # deterministic order for DDP setups
                    sibling_group_id = cls.__group_data['next_sibling_group_id']
                    cls.__group_data['sibling_groups'][sibling_group_id] = all_siblings
                    cls.__group_data['next_sibling_group_id'] = sibling_group_id + 1

                    sibling_group_C_params = []
                    for sibling in all_siblings:
                        fx_graph[sibling]['sibling_group_id'] = sibling_group_id
                        sibling_C_param = int(fx_graph[sibling]['C_param'])
                        sibling_group_C_params.append(sibling_C_param)

                    # if grouped convolutions make the input channels different among siblings different sizes,
                    # restrict the permutation atom to the greatest common divisor so it can be tiled as needed for each sibling (and parent)
                    sibling_group_C_param = str(np.gcd.reduce(sibling_group_C_params))
                    cls.__group_data['sibling_group_C_params'][sibling_group_id] = sibling_group_C_param
                    cls.__group_data['skipped_sibling_groups'][sibling_group_id] = None

                    if cls.__verbosity > 1:
                        print(f"New sibling group {sibling_group_id} with GCD(C) of {sibling_group_C_param}: {all_siblings}")


                coparent_group_id = fx_node['coparent_group_id']
                if coparent_group_id is None:
                    all_coparents = cls.collect_coparents(fx_graph, node_name, set([node_name]))
                    coparent_group_id = cls.__group_data['next_coparent_group_id']
                    cls.__group_data['coparent_groups'][coparent_group_id] = all_coparents
                    cls.__group_data['next_coparent_group_id'] = coparent_group_id + 1
                    cls.__group_data['skipped_coparent_groups'][coparent_group_id] = None

                    for coparent in all_coparents:
                        fx_graph[coparent]['coparent_group_id'] = coparent_group_id

                    if cls.__verbosity > 1:
                        print(f"New coparent group {coparent_group_id}: {all_coparents}")
        return fx_graph

    @classmethod
    def propagate_permutation_flags(cls, fx_graph):
        """Disallow sibling groups from having different C_permutable flags and coparent groups from having different K_permutable flags within the groups"""
        
        made_change = True  # will we need to repeat this propagation?
        # TODO: just propagate to sibling groups and coparent groups directly, instead of iteratively to direct real_parents and siblings
        while made_change:
            made_change = False

            if cls.__verbosity > 0:
               print("Making a pass at propagating permutation flags")

            for node_name in fx_graph.keys():

                fx_node = fx_graph.get(node_name)
    
                node_parents = fx_graph.get(node_name).get('parents')
                node_real_parents = fx_graph.get(node_name).get('real_parents')
                node_children = fx_graph.get(node_name).get('children')
                node_real_children = fx_graph.get(node_name).get('real_children')

                # input layers can't be permuted along C without a runtime fixup, skip them
                if node_parents is None or ('x' in node_parents and 'C_permutable' in fx_graph[node_name].keys() and fx_graph[node_name]['C_permutable']):
                    if cls.__verbosity > 1:
                        print(f"{node_name} has no parents, or only an input, disabling permutations in C")
                    made_change = True
                    fx_graph[node_name]['C_permutable'] = False

                # output layers can't be permuted along K without a runtime fixup, skip them
                if node_children is None or ('output' in node_children and 'K_permutable' in fx_graph[node_name].keys() and fx_graph[node_name]['K_permutable']):
                    if cls.__verbosity > 1:
                        print(f"{node_name} has no children, or only an output, disabling permutations in K")
                    made_change = True
                    fx_graph[node_name]['K_permutable'] = False
                    fx_graph[node_name]['K_passthru'] = False

                if 'is_real' in fx_node.keys() and fx_node['is_real']:
                    # siblings must share C-flags; if one cannot be permuted along C, none can
                    sibling_group_id = fx_graph[node_name]['sibling_group_id']
                    all_siblings = cls.__group_data['sibling_groups'][sibling_group_id]
                    made_change = cls.propagate_sibling_group(fx_graph, all_siblings, cls.__verbosity) or made_change

                    # coparents must share K-flags; if one cannot be permuted along K, none can
                    coparent_group_id = fx_graph[node_name]['coparent_group_id']
                    all_coparents = cls.__group_data['coparent_groups'][coparent_group_id]
                    made_change = cls.propagate_coparent_group(fx_graph, all_coparents, cls.__verbosity) or made_change

        return fx_graph

 
    @classmethod
    def find_node_real_children(cls, fx_graph, node_name, found_children):
        """Collect the real children of some node"""

        if 'real_children' in fx_graph[node_name].keys():
            return found_children.union(fx_graph[node_name]['real_children'])

        children = fx_graph[node_name]['children']
        for child in children:
            if child in fx_graph.keys(): # not the output node
                if cls.__verbosity > 3:
                    print(f"\tchecking child {child} of node {node_name}")

                # if it's a real node, just add it
                if 'is_real' in fx_graph[child].keys() and fx_graph[child]['is_real']:
                    found_children.add(child)
                else:   # otherwise, search its children
                    found_children = cls.find_node_real_children(fx_graph, child, found_children)

        return found_children

    @classmethod
    def find_real_children(cls, fx_graph):
        """Collect the real children of all nodes in the graph"""

        if cls.__verbosity > 0:
            print("\n[find_real_children] Find the real children for each node according to the whole network graph built with Torch.FX")
        
        reversible_fx_graph_keys = list(fx_graph.keys())
        for node_name in reversed(reversible_fx_graph_keys):    # as the optimization, we need to find the real children from back to front, to use the already saved 'real_children'
            node_children = fx_graph.get(node_name).get('children')

            if cls.__verbosity > 2:
                print("[find_real_children] node_name: \'{:}\', children: {:}".format(node_name, node_children))

            real_children = cls.find_node_real_children(fx_graph, node_name, set())

            if cls.__verbosity > 1:
                print(f"[find_real_children] {node_name} has {len(real_children)} real children: {real_children}")

            fx_graph[node_name]['real_children'] = sorted(real_children)

        if cls.__save_permutation_graph:
            cls.save_graph_to_json(fx_graph, save_dumped_graph_path_with_name=os.path.join(cls.__permutation_output_dir, './model_graph_find_real_children.json'))    # save the intermediate graph as JSON file for debugging
        return fx_graph

    @classmethod
    def find_node_real_parents(cls, fx_graph, node_name, found_parents):
        """Collect the real parents of some node"""

        if 'real_parents' in fx_graph[node_name].keys():
            return found_parents.union(fx_graph[node_name]['real_parents'])

        parents = fx_graph[node_name]['parents']
        for parent in parents:
            if parent in fx_graph.keys(): # not the input node
                if cls.__verbosity > 3:
                    print(f"\tchecking parent {parent} of node {node_name}")

                # if it's a real node, just add it
                if 'is_real' in fx_graph[parent].keys() and fx_graph[parent]['is_real']:
                    found_parents.add(parent)
                else:   # otherwise, search its parents
                    found_parents = cls.find_node_real_parents(fx_graph, parent, found_parents)

        return found_parents


    @classmethod
    def find_real_parents(cls, fx_graph):
        """Collect the real parents of all nodes in the graph"""

        if cls.__verbosity > 0:
            print("\n[find_real_parents] Find the real parents for each node according to the whole network graph built with Torch.FX")

        for node_name in fx_graph.keys():
            node_real_parents_name = []
            node_real_parents_module_type = []

            real_parents = cls.find_node_real_parents(fx_graph, node_name, set())

            if cls.__verbosity > 1:
                print(f"[find_real_parents] {node_name} has {len(real_parents)} real parents: {real_parents}")

            fx_graph[node_name]['real_parents'] = sorted(real_parents)
            
        if cls.__save_permutation_graph:
            cls.save_graph_to_json(fx_graph, save_dumped_graph_path_with_name=os.path.join(cls.__permutation_output_dir, './model_graph_find_real_parent.json'))    # save the intermediate graph as JSON file for debugging
        return fx_graph

    @classmethod
    def build_fx_graph(cls, model, dump_fx_graph=False, save_dumped_fx_graph='./model_fx_graph.json'):
        """Build the whole network graph with Torch.FX."""

        network_fx_graph = {}
        success = True
        torch_version = str(torch.__version__)
        torch_version_major = int(torch_version.split('.')[0])
        torch_version_minor = int(torch_version.split('.')[1])
        try:
            torch_version_minimum = int(torch_version.split('.')[2])
        except ValueError:    # support the none standard version
            torch_version_minimum = torch_version.split('.')[2]
        if cls.__verbosity > 2:
            print("[build_fx_graph] The torch version is: {}, version major is: {}, version minor is: {}, version minimum is: {}".format(torch_version, torch_version_major, torch_version_minor, torch_version_minimum))

        if torch_version_major >= 2 or (torch_version_major >= 1 and torch_version_minor >= 8):
            if cls.__verbosity > 1:
                print("[build_fx_graph] The Torch.FX is supported.")
        else:    # Torch.FX is introduced in torch 1.8.0
            if cls.__verbosity >= 0:
                print("[build_fx_graph] The Torch.FX is not supported. So cannot build the Torch.FX graph.")
            success = False
            return network_fx_graph, success

        if cls.__verbosity > 2:
            print("\n[build_fx_graph] Print the model structure with pure PyTorch function")
            print(model)

        graph_module = cls.trace_and_print_raw_fx_graph(model, print_tabular=cls.__verbosity > 1) # needs "tabulate" library
        if graph_module is None:
            success = False
            return network_fx_graph, success

        if cls.__verbosity > 0:
            print("\n[build_fx_graph] Build the module name and type dictionary")

        module_name_type_dict = {}
        module_name_group_conv_dict = {}
        module_name_C_dict = {}
        module_name_K_dict = {}
        for name, mod in model.named_modules():
            if cls.__verbosity > 1:
                print("[build_fx_graph] module_name: {}, module type: {}".format(name, type(mod)))
            module_name_type_dict[name] = str(type(mod)).split("\'")[1]
            try:
                module_name_C_dict[name] = str(mod.in_channels)
            except:
                try:
                    module_name_C_dict[name] = str(mod.in_features)
                except:
                    try:
                        module_name_C_dict[name] = str(mod.embed_dim)
                    except:
                        module_name_C_dict[name] = 'None'
    
            try:
                module_name_K_dict[name] = str(mod.out_channels)
            except:
                try:
                    module_name_K_dict[name] = str(mod.out_features)
                except:
                    try:
                        module_name_K_dict[name] = str(mod.embed_dim)
                    except:
                        module_name_K_dict[name] = 'None'

            try:
                module_name_group_conv_dict[name] = str(mod.groups)
                if cls.__verbosity > 1:
                    print("[build_fx_graph] this module has \'group\' param with value: {}".format(mod.groups))
            except:
                module_name_group_conv_dict[name] = 'None'
                continue

        # keep track of children and parents for each layer (could be call_module or call_function)
        if cls.__verbosity > 0:
            print("\n[build_fx_graph] Print the children and parents relationship for each layer")
        network_fx_graph = {}
        for node in graph_module.graph.nodes:
            if node.op == 'placeholder':
                if cls.__verbosity > 2:
                    print("[build_fx_graph] This is the \'input\' node: {:}".format(node.target))
                continue
            elif node.op == 'get_attr':
                if cls.__verbosity > 2:
                    print("[build_fx_graph] This is the \'get_attr\' node: {:}".format(node.target))
                node_parent, node_children = get_node_parent_children(node)
                converted_node_name=convert_fx_node_name(node.target)

                network_fx_graph[converted_node_name] = {}
                network_fx_graph[converted_node_name]['parents'] = node_parent
                network_fx_graph[converted_node_name]['children'] = node_children
                network_fx_graph[converted_node_name]['module_type'] = 'get_attr'
                network_fx_graph[converted_node_name]['groups_param'] = 'None'

                # inspired by https://pytorch.org/docs/stable/fx.html
                def fetch_attr(target : str, mod):
                    target_atoms = target.split('.')
                    attr_itr = mod
                    for i, atom in enumerate(target_atoms):
                        if not hasattr(attr_itr, atom):
                            raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                        attr_itr = getattr(attr_itr, atom)
                    return attr_itr

                attr = fetch_attr(node.target, graph_module)
                network_fx_graph[converted_node_name]['C_param'] = 1
                network_fx_graph[converted_node_name]['K_param'] = -1
                network_fx_graph[converted_node_name]['attr'] = attr

            elif node.op == 'call_function':    # e.g. 'adaptive.avg.pool2d', 'add', 'cat', 'flatten', 'floordiv', 'getattr', 'getitem', 'hardsigmoid', 'mean', 'mul', 'relu', 'transpose'
                node_parent, node_children = get_node_parent_children(node)
                converted_node_name=convert_fx_node_name(node.name)
                if cls.__verbosity > 2:
                    print("[build_fx_graph] This is the \'call_function\' node: {:}, its parent list: {:}, its children list: {:}".format(converted_node_name, node_parent, node_children))
                network_fx_graph[converted_node_name] = {}
                network_fx_graph[converted_node_name]['parents'] = node_parent
                network_fx_graph[converted_node_name]['children'] = node_children
                network_fx_graph[converted_node_name]['fx_op'] = 'call_function'

                ### "convert" some ops to modules
                
                # concatenating along K can be handled by reducing the size of the childrens' C appropriately
                # see fixup_concats, if no dim arg, default is 0 (handled automatically)
                if node.target == torch.cat and len(node.args) > 1 and node.args[1] == 1:
                    network_fx_graph[converted_node_name]['fx_op'] = 'call_module'
                    network_fx_graph[converted_node_name]['module_type'] = 'concat'
                    network_fx_graph[converted_node_name]['groups_param'] = 'N/A'   # just need placeholders
                    network_fx_graph[converted_node_name]['C_param'] = 'N/A'
                    network_fx_graph[converted_node_name]['K_param'] = 'N/A'

            elif node.op == 'call_method':    # e.g. 'chunk', 'contiguous', 'mean', 'size', 'unsqueeze', 'view'
                node_parent, node_children = get_node_parent_children(node)
                converted_node_name=convert_fx_node_name(node.name)
                if cls.__verbosity > 2:
                    print("[build_fx_graph] This is the \'call_method\' node: {:}, its parent list: {:}, its children list: {:}".format(converted_node_name, node_parent, node_children))
                network_fx_graph[converted_node_name] = {}
                network_fx_graph[converted_node_name]['parents'] = node_parent
                network_fx_graph[converted_node_name]['children'] = node_children
                network_fx_graph[converted_node_name]['fx_op'] = 'call_method'
                continue

            elif node.op == 'call_module':
                node_parent, node_children = get_node_parent_children(node)
                converted_node_name=convert_fx_node_name(node.name)
                # check whether the converted_node_name is same as node.target, especially for ReLU case
                if converted_node_name != node.target:
                    if cls.__verbosity > 2:
                        print("[build_fx_graph][warning] The target name from Torch.FX is \'{:}\', the manually converted node name is \'{:}\', not the same one, choose the converted node name".format(node.target, converted_node_name))

                # assume the modules share the same target name have the same type, because converted_node_name may not be obtained by model.named_modules(), like some ReLU (defined in forward function)
                node_type = module_name_type_dict[node.target]
                if cls.__verbosity > 2:
                    print("[build_fx_graph] This is the \'call_module\' node: {:}, its parent list: {:}, its children list: {:}, its type: {:}".format(converted_node_name, node_parent, node_children, node_type))
                network_fx_graph[converted_node_name] = {}
                network_fx_graph[converted_node_name]['parents'] = node_parent
                network_fx_graph[converted_node_name]['children'] = node_children
                network_fx_graph[converted_node_name]['fx_op'] = 'call_module'
                network_fx_graph[converted_node_name]['module_type'] = node_type
                network_fx_graph[converted_node_name]['groups_param'] = module_name_group_conv_dict[node.target]
                network_fx_graph[converted_node_name]['C_param'] = module_name_C_dict[node.target]
                network_fx_graph[converted_node_name]['K_param'] = module_name_K_dict[node.target]


            elif node.op == 'output':
                if cls.__verbosity > 2:
                    print("[build_fx_graph] This is the \'output\' node: {:}".format(node.target))
                continue

        if dump_fx_graph:
            if cls.__verbosity > 0:
                print("\n[build_fx_graph] Dump the overall dict for children and parents relationship into JSON file")
            cls.save_graph_to_json(network_fx_graph, save_dumped_graph_path_with_name=save_dumped_fx_graph)

        return network_fx_graph, success

    @classmethod
    def trace_and_print_raw_fx_graph(cls, model, print_tabular=False, generate_python_code=False):
        """This function is used to find and print the intermediate representation (IR) - Graph representation with Torch.FX features."""

        from torch.fx import symbolic_trace
        import traceback

        # Symbolic tracing frontend - captures the semantics of the module
        try:
            symbolic_traced : torch.fx.GraphModule = symbolic_trace(model)
        except Exception as ex:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                if cls.__verbosity > 0:
                    print(ex)
                    print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
                    print("\n[print_raw_fx_graph] Meet the fatal fault when trying to symbolic trace the model with Torch.FX")
            return None

        # High-level intermediate representation (IR) - Graph representation
        if cls.__verbosity > 1:
            print("\n[print_raw_fx_graph] Print the intermediate representation (IR) with Torch.FX")
            print(symbolic_traced.graph)

        if print_tabular:
            print("\n[print_raw_fx_graph] Print the intermediate representation (IR) with Torch.FX in a table format")
            try:
                from tabulate import tabulate
                symbolic_traced.graph.print_tabular()
            except ImportError:
                if cls.__verbosity > 1:
                    print("[print_raw_fx_graph][Warning] \'print_tabular\' relies on the library `tabulate`; run `pip install tabulate` to install it.")
            except AttributeError:    # to avoid the AttributeError: 'Graph' object has no attribute 'print_tabular'
                if cls.__verbosity > 1:
                    print("[print_raw_fx_graph][Warning] \'print_tabular\' function is not supported in current Torch version. Skip!")

        # Code generation - valid Python code
        if generate_python_code:
            print("\n[print_raw_fx_graph] Create valid Python code matching the IR/Graph's semantics with Torch.FX")
            print(symbolic_traced.code)

        return symbolic_traced

    @classmethod
    def save_graph_to_json(cls, graph, save_dumped_graph_path_with_name='./model_fx_graph.json'):
        """This function is used to save the graph into JSON file for inspection."""

        # use dumps to transfer the dict to JSON string
        json_graph_str = json.dumps(graph)
        with open(save_dumped_graph_path_with_name, 'w', encoding='utf-8') as dumped_graph_file:
            dumped_graph_file.write(json_graph_str)  # write the transferred JSON string into JSON file
