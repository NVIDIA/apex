import os
import torch
import json
import string
import time
try:
    from .permutation_search_kernels import accelerated_search_for_good_permutation, sum_after_2_to_4
    print("[ASP][Info] permutation_search_kernels can be imported.")
except ImportError:
    print("[ASP][Warning] permutation_search_kernels cannot be imported.")
    print("[ASP][Warning] If you want to accelerate the permutation search process by GPU, please build APEX by following the instructions at https://github.com/NVIDIA/apex/blob/master/apex/contrib/sparsity/README.md")

def convert_fx_node_name(fx_node_name):
    converted_fx_node_name = fx_node_name
    converted_fx_node_name = converted_fx_node_name.replace('_', '.')
    return converted_fx_node_name

def get_node_parent_children(fx_node):
    # get node parent list, and convert node name to module name
    node_parent_name_converted = []
    if len(fx_node.all_input_nodes) > 0:
        node_parent = fx_node.all_input_nodes
        for item in node_parent:
            converted_item = convert_fx_node_name(item.name)
            node_parent_name_converted.append(converted_item)
    else:
        node_parent = list('None')
        node_parent_name_converted.append('None')
    # get node children list, and convert node name to module name
    node_children_name_converted = []
    if len(list(fx_node.users.keys())) > 0:
        node_children = list(fx_node.users.keys())
        for item in node_children:
            converted_item = convert_fx_node_name(item.name)
            node_children_name_converted.append(converted_item)
    else:
        node_children = list('None')
        node_children_name_converted.append('None')
    return node_parent_name_converted, node_children_name_converted


class Permutation:
    __model = None
    __sparse_parameters = []
    __allow_permutation = False
    __all_parameters = []
    __save_permutation_graph = False
    __permutation_output_dir = ''

    @classmethod
    def set_permutation_params_from_asp(cls, model, sparse_parameters, all_parameters):
        """This function is used to set the permutation needed parameters from ASP class."""
        print("\n[set_permutation_params_from_asp] Set permutation needed parameters")
        cls.__model = model
        cls.__sparse_parameters = sparse_parameters
        cls.__all_parameters = all_parameters

    @classmethod
    def set_identical_seed(cls, identical_seed=1):
        print("\n[set_identical_seed] Set the identical seed: {:} for all GPUs to make sure the same results generated in permutation search".format(identical_seed))
        torch.manual_seed(identical_seed)
        torch.cuda.manual_seed_all(identical_seed)
        import numpy as np
        import random
        np.random.seed(identical_seed)
        random.seed(identical_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @classmethod
    def set_permutation_saving_params(cls, allow_permutation=False, save_permutation_graph=False, permutation_output_dir='.'):
        """This function is used to set the permutation saving related parameters."""
        print("\n[permutation_lib][set_permutation_saving_param] Set permutation saving related parameters")
        cls.__allow_permutation = allow_permutation
        print("[set_permutation_saving_param]\t Allow permutation: {}".format(cls.__allow_permutation))
        cls.__save_permutation_graph = save_permutation_graph
        print("[set_permutation_saving_param]\t Save permutation graphs: {}".format(cls.__save_permutation_graph))
        cls.__permutation_output_dir = permutation_output_dir
        print("[set_permutation_saving_param]\t Permutation graphs saving dir: {}".format(cls.__permutation_output_dir))

    @classmethod
    def apply_offline_permutation(cls, model, fx_graph):
        """This function is used to offline permutation for each node according to the the whole network graph built with Torch.FX."""
        print("\n[apply_offline_permutation] Offline permutation for each node according to the the whole network graph built with Torch.FX")

        # Firstly, we should transfer the sparse mask to all-one dense mask
        cls.transfer_to_dense_mask()

        for node_name in fx_graph.keys():
            node_module_type = fx_graph.get(node_name).get('module_type')

            # check wheter the current layer can permute as plan, e.g., the flatten layer in VGG will change the shape and broke the permutation chain
            # only need to check the 'is_node_real_parents_K_permuted', because the 'is_node_real_parents_C_permuted' has no influence to the children
            node_real_parents = fx_graph.get(node_name).get('real_parents')
            is_node_real_parents_K_permuted = True
            if node_real_parents is not None:    # filter out the 'unique_siblings' item
                for real_parent_item in node_real_parents:
                    if fx_graph.get(real_parent_item).get('permutation_type') in ['K', 'KC']:
                        if fx_graph.get(real_parent_item).get('k_permuted') == 'False':
                            is_node_real_parents_K_permuted = False

            if fx_graph[node_name]['permutation_type'] == 'KC':    # intermediate Conv, FC
                C_permutation_sequence = cls.fetch_C_permutation_sequence_value(node_name, fx_graph)
                K_permutation_sequence = cls.fetch_K_permutation_sequence_value(node_name, fx_graph)
                print("\n[apply_offline_permutation] node_name: \'{:}\', node module type: \'{:}\', need to do offline permutation in K and C dims.".format(node_name, node_module_type))
                if is_node_real_parents_K_permuted == True:
                    fx_graph[node_name]['c_permuted'] = str(cls.apply_permutation_in_C_dim(node_name, C_permutation_sequence))
                    fx_graph[node_name]['k_permuted'] = str(cls.apply_permutation_in_K_dim(node_name, K_permutation_sequence))
                else:
                    print("[apply_offline_permutation][warning] node_name: \'{:}\', its real parents have trouble in permutation in K dim, so skip the offline permutation in C dim.".format(node_name, node_module_type))
                    fx_graph[node_name]['k_permuted'] = str(cls.apply_permutation_in_K_dim(node_name, K_permutation_sequence))
            elif fx_graph[node_name]['permutation_type'] == 'K':    # BN, first layer Conv/FC
                K_permutation_sequence = cls.fetch_K_permutation_sequence_value(node_name, fx_graph)
                print("\n[apply_offline_permutation] node_name: \'{:}\', node module type: \'{:}\', need to do offline permutation in K dim.".format(node_name, node_module_type))
                if is_node_real_parents_K_permuted == True:
                    fx_graph[node_name]['k_permuted'] = str(cls.apply_permutation_in_K_dim(node_name, K_permutation_sequence))
                else:    # for BN, if the previous Conv cannot do permutation in K dim, then no need to do permutation in K dim for this BN
                    print("[apply_offline_permutation][warning] node_name: \'{:}\', its real parents have trouble in permutation in K dim, so skip the offline permutation in K dim.".format(node_name, node_module_type))
            elif fx_graph[node_name]['permutation_type'] == 'C':    # last layer FC/Conv
                C_permutation_sequence = cls.fetch_C_permutation_sequence_value(node_name, fx_graph)
                print("\n[apply_offline_permutation] node_name: \'{:}\', node module type: \'{:}\', need to do offline permutation in C dim.".format(node_name, node_module_type))
                if is_node_real_parents_K_permuted == True:
                    fx_graph[node_name]['c_permuted'] = str(cls.apply_permutation_in_C_dim(node_name, C_permutation_sequence))
                else:
                    print("[apply_offline_permutation][warning] node_name: \'{:}\', its real parents have trouble in permutation in K dim, so skip the offline permutation in C dim.".format(node_name, node_module_type))

        if cls.__save_permutation_graph:
            cls.save_graph_to_json(fx_graph, save_dumped_graph_path_with_name=os.path.join(cls.__permutation_output_dir, './model_graph_apply_offline_permutation.json'))    # save the intermediate graph as JSON file for debugging
        return fx_graph

    @classmethod
    def transfer_to_dense_mask(cls):
        """Call this method to transfer the sparse mask to all-one dense mask."""
        with torch.no_grad():
            for module_name, module, p_name, p, mask, pruned in cls.__sparse_parameters:
                mask.fill_(1)

    @classmethod
    def fetch_C_permutation_sequence_value(cls, node_name, fx_graph):
        """This function is used to fetch the permutation sequence value in C dim from the unique_siblings record."""
        # C_permutation_sequence is the corresponding 'permutation_sequence' value stored in the fx_graph.get('unique_siblings') item which contains node_name
        unique_siblings_groups = fx_graph.get('unique_siblings').get('name')
        unique_siblings_groups_permutation_sequence = fx_graph.get('unique_siblings').get('permutation_sequence')
        item_index = 0
        fetched_C_permutation_sequence = []
        for item in unique_siblings_groups:
            if node_name in item:
                fetched_C_permutation_sequence = unique_siblings_groups_permutation_sequence[item_index]
            item_index = item_index + 1
        return fetched_C_permutation_sequence

    @classmethod
    def fetch_K_permutation_sequence_value(cls, node_name, fx_graph):
        """This function is used to fetch the permutation sequence value in K dim from the unique_siblings record."""
        # K_permutation_sequence is its real_children's corresponding 'permutation_sequence' value stored in the fx_graph.get('unique_siblings') item which contains real_children name
        # we have the assumption that all the real children are in one unique_sibling group, so should share the same permutation_sequence value
        unique_siblings_groups = fx_graph.get('unique_siblings').get('name')
        unique_siblings_groups_permutation_sequence = fx_graph.get('unique_siblings').get('permutation_sequence')
        node_real_children = fx_graph.get(node_name).get('real_children')
        fetched_K_permutation_sequence = []
        if len(node_real_children) > 0:
            node_representative_child = node_real_children[0]
            fetched_K_permutation_sequence = cls.fetch_C_permutation_sequence_value(node_representative_child, fx_graph)
        return fetched_K_permutation_sequence

    @classmethod
    def apply_permutation_in_C_dim(cls, node_name, permutation_sequence):
        """This function is used to permutation for a node in C dim. (Only need to handle the weight of the node) """
        print("[apply_permutation_in_C_dim] Permutation for node: \'{:}\' in C dim".format(node_name))
        if len(permutation_sequence) == 0:
            print("[apply_permutation_in_C_dim] the permutation sequence is empty, fail to apply permutation in C dim.")
            return False
        is_node_in_sparse_parameters = False
        success_permutation = False
        for module_name, module, p_name, p, mask, pruned in cls.__sparse_parameters:
            processed_module_name = ''.join(c for c in module_name if c not in string.punctuation).lower()
            processed_node_name = ''.join(c for c in node_name if c not in string.punctuation).lower()
            distributed_node_name = 'module.' + node_name
            processed_distributed_node_name = 'module.' + processed_node_name
            if (module_name == node_name) or (module_name == distributed_node_name) or (processed_module_name == processed_node_name) or (processed_module_name == processed_distributed_node_name):    # Inception-V3, module_name: Conv2d_2a_3x3.conv, node_name: conv2d.1a.3x3.conv
                print("[apply_permutation_in_C_dim] find the node: \'{:}\' in cls.__sparse_parameters, succeed to apply permutation in C dim.".format(node_name))
                is_node_in_sparse_parameters = True
                temp_weight = torch.zeros_like(p)
                temp_weight.copy_(p[:, permutation_sequence, ...])
                p.data.copy_(temp_weight)
                success_permutation = True
        if is_node_in_sparse_parameters == False:
            # A special case: if the node itself not in sparse_module_names but one of its real_siblings in sparse_module_names, then the node will not do the permutation search, but it may need to apply the offline permutation in C dim according to the searched permutation sequence from its real_siblings in sparse_module_names
            try:
                for module_name_from_all_parameters, module_from_all_parameters, p_name_from_all_parameters, p_from_all_parameters in cls.__all_parameters:
                    if ((node_name == module_name_from_all_parameters) or ('module.' + node_name == module_name_from_all_parameters)) and p_name_from_all_parameters == "weight":
                        print("[apply_permutation_in_C_dim] cannot find the node: \'{:}\' in cls.__sparse_parameters, but can find in cls.__all_parameters.".format(node_name))
                        temp_weight = torch.zeros_like(p_from_all_parameters)
                        temp_weight.copy_(p_from_all_parameters[:, permutation_sequence, ...])
                        p_from_all_parameters.data.copy_(temp_weight)
                        success_permutation = True
                        print("[apply_permutation_in_C_dim] cannot find the node: \'{:}\' in cls.__sparse_parameters, after trying with cls.__all_parameters, succeed to apply permutation in C dim.".format(node_name))
            except:
                success_permutation = False
                print("[apply_permutation_in_C_dim] cannot find the node: \'{:}\' in cls.__sparse_parameters, after trying with cls.__all_parameters, still fail to apply permutation in C dim.".format(node_name))
        return success_permutation

    @classmethod
    def apply_permutation_in_K_dim(cls, node_name, permutation_sequence):
        """This function is used to permutation for a node in K dim. (Need to handle the weight/bias/running_mean/running_var of the node)"""
        print("[apply_permutation_in_K_dim] Permutation for node: \'{:}\' in K dim".format(node_name))
        if len(permutation_sequence) == 0:
            print("[apply_permutation_in_K_dim] the permutation sequence is empty, fail to apply permutation in K dim.")
            return False
        is_node_in_all_parameters = False
        success_permutation = False
        for module_name, module, p_name, p in cls.__all_parameters:
            processed_module_name = ''.join(c for c in module_name if c not in string.punctuation).lower()
            processed_node_name = ''.join(c for c in node_name if c not in string.punctuation).lower()
            distributed_node_name = 'module.' + node_name
            processed_distributed_node_name = 'module.' + processed_node_name
            if (module_name == node_name) or (module_name == distributed_node_name) or (processed_module_name == processed_node_name) or (processed_module_name == processed_distributed_node_name):    # Inception-V3, module_name: Conv2d_2a_3x3.conv, node_name: conv2d.1a.3x3.conv
                print("[apply_permutation_in_K_dim] find the node: \'{:}\' with \'{:}\' in cls.__all_parameters, may succeed to apply permutation in K dim.".format(node_name, p_name))
                is_node_in_all_parameters = True
                temp_weight = torch.zeros_like(p)
                if p.shape[0] != len(permutation_sequence):
                    print("[apply_permutation_in_K_dim][warning] the node: \'{:}\' with shape: \'{:}\', cannot match the size of permutation sequence with len: \'{:}\', fail to apply permutation in K dim.".format(node_name, p.shape, len(permutation_sequence)))
                    success_permutation = False
                else:
                    print("[apply_permutation_in_K_dim] the node: \'{:}\' with shape: \'{:}\', can match the size of permutation sequence with len: \'{:}\', succeed to apply permutation in K dim.".format(node_name, p.shape, len(permutation_sequence)))
                    temp_weight.copy_(p[permutation_sequence, ...])
                    p.data.copy_(temp_weight)
                    success_permutation = True
        if is_node_in_all_parameters == False:
            print("[apply_permutation_in_K_dim] cannot find the node: \'{:}\' in cls.__all_parameters, fail to apply permutation in K dim.".format(node_name))
            success_permutation = False
        return success_permutation

    @classmethod
    def build_offline_permutation_graph(cls, model, dump_fx_graph=False, save_dumped_fx_graph='./model_offline_permutation_graph.json'):
        """This function is used to refine the whole network graph built with Torch.FX with some extra infomation needed for offline permutation."""
        print("\n[build_offline_permutation_graph] Further refine the model graph built by Torch.FX for offline permutation")
        # extract the output_dir, so all the intermediate fx_graph can be saved under that path
        extract_output_dir=os.path.split(save_dumped_fx_graph)[0]
        cls.__permutation_output_dir = extract_output_dir
        fx_graph, success_in_build_fx_graph = cls.build_fx_graph(model, dump_fx_graph=dump_fx_graph, save_dumped_fx_graph=save_dumped_fx_graph)
        if success_in_build_fx_graph:
            fx_graph_after_find_real_parents = cls.find_real_parents(fx_graph)
            fx_graph_after_find_real_children = cls.find_real_children(fx_graph_after_find_real_parents)
            fx_graph_after_find_real_siblings = cls.find_real_siblings(fx_graph_after_find_real_children)
            fx_graph_after_extract_all_unique_siblings = cls.extract_all_unique_siblings(fx_graph_after_find_real_siblings)
            fx_graph_after_init_permutation_flag = cls.init_permutation_flag(fx_graph_after_extract_all_unique_siblings)
            start_time_search_for_good_permutation = time.perf_counter()
            fx_graph_after_search_for_good_permutation = cls.search_for_good_permutation(fx_graph_after_init_permutation_flag)
            duration_search_for_good_permutation = time.perf_counter() - start_time_search_for_good_permutation
            print("\n[build_offline_permutation_graph] Take {:.4f} seconds to finish search_for_good_permutation function.".format(duration_search_for_good_permutation))
        else:
            fx_graph_after_search_for_good_permutation = {}
            return fx_graph_after_search_for_good_permutation, success_in_build_fx_graph

        # Please notice the apply_offline_permutation step cannot fold into the above search_for_good_permutation step.
        # Because the real_parent node needs to offline permutation in K direction according to the searched permutation sequence from its real_children.
        # However, when we search_for_good_permutation for the node, its real_children have not been handled by search_for_good_permutation.

        if cls.__save_permutation_graph:
            cls.save_graph_to_json(fx_graph_after_search_for_good_permutation, save_dumped_graph_path_with_name=os.path.join(cls.__permutation_output_dir, './model_graph_build_offline_permutation_graph.json'))    # save the intermediate graph as JSON file for debugging
        return fx_graph_after_search_for_good_permutation, success_in_build_fx_graph

    @classmethod
    def search_for_good_permutation(cls, fx_graph):
        """This function is used to:
        1. search for the good permutation sequence for each node weight, or each siblings_group weights by calling the permutation search kernels as ASP extension.
        2. add the searched permutation sequence for each node according to the whole network graph built with Torch.FX."""
        print("\n[search_for_good_permutation] Search for the good permutation sequence for each node according to the whole network graph built with Torch.FX")

        unique_siblings_groups = fx_graph.get('unique_siblings').get('name')
        unique_siblings_groups_module_type = fx_graph.get('unique_siblings').get('module_type')
        unique_siblings_groups_permutation_sequence = []
        item_index = 0
        for unique_siblings_group in unique_siblings_groups:    # loop through all unique siblings groups that must share a permutation sequence
            print("\n[search_for_good_permutation] this unique_siblings_group has {:} real siblings: \'{:}\', with module type: \'{:}\'.".format(len(unique_siblings_group), unique_siblings_group, unique_siblings_groups_module_type[item_index]))
            item_index = item_index + 1

            # concat the weight for layers in the same unique_siblings_group
            matrix_group = None
            for node_name in unique_siblings_group:
                node_module_type = fx_graph.get(node_name).get('module_type')
                print("[search_for_good_permutation] try to merge the weight for node: \'{:}\', with module type: \'{:}\'.".format(node_name, node_module_type))
                is_node_in_sparse_parameters = False
                node_weight = torch.zeros(0)
                for module_name, module, p_name, p, mask, pruned in cls.__sparse_parameters:
                    processed_module_name = ''.join(c for c in module_name if c not in string.punctuation).lower()
                    processed_node_name = ''.join(c for c in node_name if c not in string.punctuation).lower()
                    distributed_node_name = 'module.' + node_name
                    processed_distributed_node_name = 'module.' + processed_node_name
                    if (module_name == node_name) or (module_name == distributed_node_name) or (processed_module_name == processed_node_name) or (processed_module_name == processed_distributed_node_name):    # Inception-V3, module_name: Conv2d_2a_3x3.conv, node_name: conv2d.1a.3x3.conv
                        module_type_from_sparse_parameters = str(type(module))    # e.g. <class 'torch.nn.modules.conv.Conv2d'>
                        module_type_from_sparse_parameters = module_type_from_sparse_parameters[8:-2]
                        print("[search_for_good_permutation] find the node: \'{:}\' in cls.__sparse_parameters, module type match: \'{:}\'.".format(node_name, node_module_type==module_type_from_sparse_parameters))
                        is_node_in_sparse_parameters = True
                        node_weight = torch.zeros_like(p)
                        node_weight.copy_(p)
                        # Need to handle the concat for layers with different R & S
                        shape = node_weight.shape
                        # 1d-tensor
                        if len(shape) == 1:
                            node_weight = node_weight.view(1, shape[0])
                        # 2d-tensor (in, out)
                        elif len(shape) == 2:
                            node_weight = node_weight.view(shape[0], shape[1])
                        # 3d-tensor (batch, in, out)
                        elif len(shape) == 3:
                            node_weight = node_weight.view(shape[0]*shape[1], shape[2])
                        # 4d-tensor (in, out, h, w)
                        elif len(shape) == 4:
                            # convs
                            node_weight = node_weight.permute(2,3,0,1).contiguous().view(shape[2]*shape[3]*shape[0], shape[1])

                if is_node_in_sparse_parameters == False:
                    print("[search_for_good_permutation] cannot find the node: \'{:}\' in cls.__sparse_parameters, no need to merge its weight for permutation.".format(node_name))
                else:
                    if matrix_group == None:
                        matrix_group = node_weight
                    else:
                        try:
                            if matrix_group.dim() == node_weight.dim():
                                matrix_group = torch.cat((matrix_group, node_weight), dim=0)    # concat the weights in K dimension, and keep the same C dimension
                            else:    # e.g. when try to merge the Conv and FC layers
                                print("[search_for_good_permutation] matrix_group dim: {:} is not matched with node_weight dim: {:}.".format(matrix_group.dim(), node_weight.dim()))
                                print("[search_for_good_permutation] matrix_group shape: \'{:}\' is not matched with node_weight shape: \'{:}\'.".format(matrix_group.size(), node_weight.size()))
                                if matrix_group.dim() < node_weight.dim():
                                    while node_weight.dim() - matrix_group.dim() > 0:
                                        matrix_group = matrix_group.unsqueeze(matrix_group.dim())
                                else:
                                    while matrix_group.dim() - node_weight.dim() > 0:
                                        node_weight = node_weight.unsqueeze(node_weight.dim())
                                print("[search_for_good_permutation] matrix_group shape: \'{:}\' is now matched with node_weight shape: \'{:}\'.".format(matrix_group.size(), node_weight.size()))
                                matrix_group = torch.cat((matrix_group, node_weight), dim=0)    # concat the weights in K dimension, and keep the same C dimension
                        except:
                            print("[search_for_good_permutation][warning] cannot merge the weight for node: \'{:}\', with its weight shape: \'{:}\', the matrix_group shape: \'{:}\'.".format(node_name, node_weight.size(), matrix_group.size()))
                            continue
                    print("[search_for_good_permutation] have merged the weight for node: \'{:}\', with its weight shape: \'{:}\', the matrix_group shape: \'{:}\'.".format(node_name, node_weight.size(), matrix_group.size()))

            if matrix_group == None:    # cannot find the node: \'{:}\' in cls.__sparse_parameters
                input_channel_num = 0
                print("\n[search_for_good_permutation] init the all-zero list with length \'{:}\' for permutation search sequence of this unique_siblings_group.".format(input_channel_num))
                print("[search_for_good_permutation] no need to search the permutation_sequence for empty matrix_group.")
                permutation_sequence = [0 for n in range(input_channel_num)]
                unique_siblings_groups_permutation_sequence.append(permutation_sequence)
                continue
            else:
                input_channel_num = matrix_group.size()[1]
                print("\n[search_for_good_permutation] init the all-zero list with length \'{:}\' for permutation search sequence of this unique_siblings_group.".format(input_channel_num))
                permutation_sequence = [0 for n in range(input_channel_num)]

            # automatic check for skipping the permutation search process
            original_magnitude = (torch.abs(matrix_group)).sum(dtype=torch.float64)
            pruned_magnitude = sum_after_2_to_4(matrix_group.cpu().detach().numpy())
            diff_ratio = abs(original_magnitude - pruned_magnitude)/original_magnitude
            epsilon = 1e-3
            print("\n[search_for_good_permutation] Original element abs sum: {:}, Pruned element abs sum: {:}, Diff ratio: {:}".format(original_magnitude, pruned_magnitude, diff_ratio))
            if diff_ratio < epsilon:
                print("[search_for_good_permutation] Original element abs sum is almost same as the pruned element abs sum, further permutation search will not help, skipping!")
                print("[search_for_good_permutation] Change the all-zero permutation search sequence to a sequential permutation search sequence.")
                permutation_sequence = [n for n in range(input_channel_num)]
                unique_siblings_groups_permutation_sequence.append(permutation_sequence)
                continue
            else:
                print("[search_for_good_permutation] Original element abs sum is different from the pruned element abs sum, further permutation search will help, continue with the permutation search!")

            # call the permutation search CUDA kernels as ASP extension.
            # users can provide prefer search strategy by providing a valid 'search_options' as a dictionary,
            # or users can implement their customized 'accelerated_search_for_good_permutation' function.
            search_options = {}
            # No.1 Strategy: Exhaustive Search
            # search_options['strategy'] = 'exhaustive'
            # search_options['stripe_group_size'] = 8
            # search_options['escape_attempts'] = 100
            # No.2 Strategy: Progressive Channel Swap Search
            # search_options['strategy'] = 'progressive channel swap'
            # search_options['progressive_search_time_limit'] = 10
            # search_options['improvement_threshold'] = 1e-9
            # No.3 Strategy: User Defined Search
            # search_options['strategy'] = 'user defined'

            # permutation search time is too long for matrix_group with large channel num
            # change from Exhaustive Search to Progressive Channel Swap Search based on input matrix_group size
            if input_channel_num > 2048:
                search_options['strategy'] = 'progressive channel swap'
                search_options['progressive_search_time_limit'] = 120
                search_options['improvement_threshold'] = 1e-9
                print("[search_for_good_permutation] Change to Progressive Channel Swap Search with {} seconds limitation, because the {} is too large and will leading too long permutation search time with Exhaustive Search.".format(search_options['progressive_search_time_limit'], input_channel_num))

            start_time_accelerated_search_for_good_permutation = time.perf_counter()
            permutation_sequence = accelerated_search_for_good_permutation(matrix_group, options=search_options)
            duration_accelerated_search_for_good_permutation = time.perf_counter() - start_time_accelerated_search_for_good_permutation
            print("[search_for_good_permutation] Take {:.4f} seconds to finish accelerated_search_for_good_permutation function.".format(duration_accelerated_search_for_good_permutation))
            unique_siblings_groups_permutation_sequence.append(permutation_sequence)
        fx_graph['unique_siblings']['permutation_sequence'] = unique_siblings_groups_permutation_sequence

        if cls.__save_permutation_graph:
            cls.save_graph_to_json(fx_graph, save_dumped_graph_path_with_name=os.path.join(cls.__permutation_output_dir, './model_graph_search_for_good_permutation.json'))    # save the intermediate graph as JSON file for debugging
        return fx_graph

    @classmethod
    def init_permutation_flag(cls, fx_graph):
        """This function is used to init the permutation flag for each node according to the whole network graph built with Torch.FX."""
        print("\n[init_permutation_flag] Init the permutation flag for each node according to the whole network graph built with Torch.FX")
        sparse_module_names = []
        processed_sparse_module_names = []    # Inception-V3, module_name: Conv2d_2a_3x3.conv, node_name: conv2d.1a.3x3.conv
        for module_name, module, p_name, p, mask, pruned in cls.__sparse_parameters:
            sparse_module_names.append(module_name)
            processed_module_name = ''.join(c for c in module_name if c not in string.punctuation).lower()
            processed_sparse_module_names.append(processed_module_name)
        for node_name in fx_graph.keys():
            processed_node_name = ''.join(c for c in node_name if c not in string.punctuation).lower()
            distributed_node_name = 'module.' + node_name
            processed_distributed_node_name = 'module.' + processed_node_name
            node_module_type = fx_graph.get(node_name).get('module_type')
            if node_module_type in ['torch.nn.modules.conv.Conv2d', 'torch.nn.modules.linear.Linear']:
                node_parents = fx_graph.get(node_name).get('parents')
                node_children = fx_graph.get(node_name).get('children')
                node_real_parents = fx_graph.get(node_name).get('real_parents')
                node_real_children = fx_graph.get(node_name).get('real_children')
                node_groups_param = fx_graph.get(node_name).get('groups_param')
                is_node_real_children_in_sparse_parameters = False
                is_node_real_children_has_group_conv = False
                for real_child_item in node_real_children:
                    processed_real_child_item = ''.join(c for c in real_child_item if c not in string.punctuation).lower()
                    distributed_real_child_item = 'module.' + real_child_item
                    processed_distributed_real_child_item = 'module.' + processed_real_child_item
                    if (real_child_item in sparse_module_names) or (processed_real_child_item in processed_sparse_module_names) or (distributed_real_child_item in sparse_module_names) or (processed_distributed_real_child_item in processed_sparse_module_names):
                        is_node_real_children_in_sparse_parameters = True
                    if (fx_graph.get(real_child_item).get('groups_param') not in ['None', '1']):
                        is_node_real_children_has_group_conv = True
                is_node_real_parents_has_group_conv = False
                for real_parent_item in node_real_parents:
                    # notice: we assume the if one item of real_parents need to permute in C or K dim, then the corresponding flag should be set
                    # so for all items of real_parents, they may not share the same 'permutation_type' (e.g., one item is Group Conv, etc.)
                    # that's why we also need to judge the 'is_node_real_parents_has_group_conv'
                    if (fx_graph.get(real_parent_item).get('groups_param') not in ['None', '1']):
                        is_node_real_parents_has_group_conv = True
                # If the node itself is in sparse_module_names or one of its real_children in sparse_module_names, then it may need the offline permutation
                if ((node_name in sparse_module_names) or (processed_node_name in processed_sparse_module_names) or (distributed_node_name in sparse_module_names) or (processed_distributed_node_name in processed_sparse_module_names)) or (is_node_real_children_in_sparse_parameters == True):
                    if node_groups_param not in ['None', '1']:
                        # for Group Conv, disable the permutation in 'C' and 'K' dim
                        fx_graph[node_name]['permutation_type'] = 'None'
                    elif ('x' in node_parents) or ((node_name not in sparse_module_names) and (processed_node_name not in processed_sparse_module_names) and (distributed_node_name not in sparse_module_names) and (processed_distributed_node_name not in processed_sparse_module_names)):
                        # for the first (due to it is connected to 'x' node or itself is not in sparse_module_names) or not NVIDIA's TC compatiable Conv/FC, only permutate the K direction
                        if is_node_real_children_has_group_conv == False:
                            fx_graph[node_name]['permutation_type'] = 'K'
                            fx_graph[node_name]['k_permuted'] = 'False'
                        else:    # if node real_children contains Group Conv, disable the permutation for node in 'K' dim
                            fx_graph[node_name]['permutation_type'] = 'None'
                    elif ('output' in node_children) or (is_node_real_children_in_sparse_parameters == False):
                        # for the last (due to it is connected to 'output' node or to a node which is not in sparse_module_names) FC/Conv, only permutate the C direction
                        if is_node_real_parents_has_group_conv == False:
                            fx_graph[node_name]['permutation_type'] = 'C'
                            fx_graph[node_name]['c_permuted'] = 'False'
                        else:    # if node real_parents contains Group Conv, disable the permutation for node in 'C' dim
                            fx_graph[node_name]['permutation_type'] = 'None'
                    else:
                        if (is_node_real_parents_has_group_conv == False) and (is_node_real_children_has_group_conv == False):
                            fx_graph[node_name]['permutation_type'] = 'KC'
                            fx_graph[node_name]['k_permuted'] = 'False'
                            fx_graph[node_name]['c_permuted'] = 'False'
                        elif is_node_real_parents_has_group_conv == True:    # TODO: if node real_parents contains Group Conv, disable the permutation for node in 'C' dim
                            fx_graph[node_name]['permutation_type'] = 'K'
                            fx_graph[node_name]['k_permuted'] = 'False'
                        else:    # if node real_children contains Group Conv, disable the permutation for node in 'K' dim
                            fx_graph[node_name]['permutation_type'] = 'C'
                            fx_graph[node_name]['c_permuted'] = 'False'
                else:
                    fx_graph[node_name]['permutation_type'] = 'None'
            elif node_module_type in ['torch.nn.modules.batchnorm.BatchNorm2d']:
                node_real_parents = fx_graph.get(node_name).get('real_parents')
                is_node_real_parents_need_K_permutation = False
                is_node_real_parents_has_group_conv = False
                for real_parent_item in node_real_parents:
                    # notice: we assume the if one item of real_parents need to permute in K dim, then the corresponding flag should be set
                    # as in most of the cases, BN only follows one Conv, so it should be OK for now
                    if fx_graph.get(real_parent_item).get('permutation_type') in ['K', 'KC']:
                        is_node_real_parents_need_K_permutation = True
                    if (fx_graph.get(real_parent_item).get('groups_param') not in ['None', '1']):
                        is_node_real_parents_has_group_conv = True
                node_real_children = fx_graph.get(node_name).get('real_children')
                is_node_real_children_in_sparse_parameters = False
                for real_child_item in node_real_children:
                    processed_real_child_item = ''.join(c for c in real_child_item if c not in string.punctuation).lower()
                    distributed_real_child_item = 'module.' + real_child_item
                    processed_distributed_real_child_item = 'module.' + processed_real_child_item
                    if (real_child_item in sparse_module_names) or (processed_real_child_item in processed_sparse_module_names) or (distributed_real_child_item in sparse_module_names) or (processed_distributed_real_child_item in processed_sparse_module_names):
                        is_node_real_children_in_sparse_parameters = True
                # Firstly, we should make sure the BN is not in the last (due to it is connected to a FC/Conv node which is not in sparse_module_names), then:
                # If the real_parents of BN node are in sparse_module_names, then it may need the offline permutation
                # Or if the real_parents of BN node just needs to permute in K dim
                if (is_node_real_children_in_sparse_parameters == True) and (is_node_real_parents_need_K_permutation == True):
                    if (is_node_real_parents_has_group_conv == False) and (is_node_real_parents_need_K_permutation == True):
                        fx_graph[node_name]['permutation_type'] = 'K'
                        fx_graph[node_name]['k_permuted'] = 'False'
                    else:    # if node real_parents contains Group Conv or does not need permutation in 'K' dim, disable the permutation for node in 'K' dim
                        fx_graph[node_name]['permutation_type'] = 'None'
                else:
                    fx_graph[node_name]['permutation_type'] = 'None'
            else:
                fx_graph[node_name]['permutation_type'] = 'None'

        # A special case: if the node itself not in sparse_module_names but one of its real_siblings in sparse_module_names, then the node will not do the permutation search, but it may need to apply the offline permutation in C dim according to the searched permutation sequence from its real_siblings in sparse_module_names
        # We make it as the post-processing, because if we add this to the previous logic, will make it too complex
        # Post-processing Step No.1:
        print("\n[init_permutation_flag] Post-processing Step No.1.")
        node_change_permutation_due_to_siblings = []
        for node_name in fx_graph.keys():
            node_real_siblings = fx_graph.get(node_name).get('real_siblings')
            if node_real_siblings is not None:
                is_node_real_siblings_needs_C_permutation = False
                for real_sibling_item in node_real_siblings:
                    if fx_graph.get(real_sibling_item).get('permutation_type') in ['C', 'KC']:
                        is_node_real_siblings_needs_C_permutation = True
                if is_node_real_siblings_needs_C_permutation == True:
                    print("[init_permutation_flag] node_name: \'{:}\', one of its real siblings need do offline permutation in C dim.".format(node_name))
                    node_original_permutation_type = fx_graph.get(node_name).get('permutation_type')
                    if node_original_permutation_type in ['C', 'KC']:
                        print("[init_permutation_flag] node_name: \'{:}\', its original permutation: \'{:}\' already includes C dim, no need to do No.1 post-processing change.".format(node_name, node_original_permutation_type))
                    elif node_original_permutation_type == 'None':
                        fx_graph[node_name]['permutation_type'] = 'C'
                        print("[init_permutation_flag] node_name: \'{:}\', change its original permutation: \'{:}\' to new permutation: 'C'.".format(node_name, node_original_permutation_type))
                        node_change_permutation_due_to_siblings.append(node_name)
                    elif node_original_permutation_type == 'K':
                        fx_graph[node_name]['permutation_type'] = 'KC'
                        print("[init_permutation_flag] node_name: \'{:}\', change its original permutation: \'{:}\' to new permutation: 'KC'.".format(node_name, node_original_permutation_type))
                        node_change_permutation_due_to_siblings.append(node_name)
        # Post-processing Step No.2:
        print("\n[init_permutation_flag] Post-processing Step No.2.")
        for node_name in fx_graph.keys():
            node_real_children = fx_graph.get(node_name).get('real_children')
            node_module_type = fx_graph.get(node_name).get('module_type')
            if (node_real_children is not None) and (node_module_type in ['torch.nn.modules.conv.Conv2d', 'torch.nn.modules.linear.Linear', 'torch.nn.modules.batchnorm.BatchNorm2d']):
                is_node_real_children_has_node_change_permutation = False
                for real_child_item in node_real_children:
                    if real_child_item in node_change_permutation_due_to_siblings:
                        is_node_real_children_has_node_change_permutation = True
                if is_node_real_children_has_node_change_permutation == True:
                    print("[init_permutation_flag] node_name: \'{:}\', one of its real children has changed permutation due to its siblings.".format(node_name))
                    node_original_permutation_type = fx_graph.get(node_name).get('permutation_type')
                    if node_original_permutation_type in ['K', 'KC']:
                        print("[init_permutation_flag] node_name: \'{:}\', its original permutation: \'{:}\' already includes K dim, no need to do No.2 post-processing change.".format(node_name, node_original_permutation_type))
                    elif node_original_permutation_type == 'None':
                        fx_graph[node_name]['permutation_type'] = 'K'
                        print("[init_permutation_flag] node_name: \'{:}\', change its original permutation: \'{:}\' to new permutation: 'K'.".format(node_name, node_original_permutation_type))
                    elif node_original_permutation_type == 'C':
                        fx_graph[node_name]['permutation_type'] = 'KC'
                        print("[init_permutation_flag] node_name: \'{:}\', change its original permutation: \'{:}\' to new permutation: 'KC'.".format(node_name, node_original_permutation_type))

        if cls.__save_permutation_graph:
            cls.save_graph_to_json(fx_graph, save_dumped_graph_path_with_name=os.path.join(cls.__permutation_output_dir, './model_graph_init_permutation_flag.json'))    # save the intermediate graph as JSON file for debugging
        return fx_graph

    @classmethod
    def extract_all_unique_siblings(cls, fx_graph):
        """This function is used to extrat all unique siblings for the whole network graph built with Torch.FX."""
        print("\n[extract_all_unique_siblings] Extract all unique siblings for the whole network graph built with Torch.FX")
        all_unique_siblings_name = []
        all_unique_siblings_module_type = []
        for node_name in fx_graph.keys():
            fx_graph[node_name]['node_type'] = 'network_node'    # use the 'node_type' to divide the real nodes apart from the auxiliary info node, like 'unique_siblings' node
            node_module_type = fx_graph.get(node_name).get('module_type')
            node_real_siblings = fx_graph.get(node_name).get('real_siblings')
            node_real_siblings_module_type = fx_graph.get(node_name).get('real_siblings_module_type')
            if node_real_siblings == []:
                print("[extract_all_unique_siblings] node_name: \'{:}\', node module type: \'{:}\', has no real siblings.".format(node_name, node_module_type))
                # for the Conv/FC layers without real_siblings, then we should insert itself as an unique_siblings
                if node_module_type in ['torch.nn.modules.conv.Conv2d', 'torch.nn.modules.linear.Linear']:
                    # direct insert will change the real_siblings info for the node in the fx_graph
                    node_real_siblings_with_node_itself = node_real_siblings.copy()
                    node_real_siblings_with_node_itself.insert(0, node_name)
                    node_real_siblings_module_type_with_node_itself = node_real_siblings_module_type.copy()
                    node_real_siblings_module_type_with_node_itself.insert(0, node_module_type)
                    all_unique_siblings_name.append(node_real_siblings_with_node_itself)
                    all_unique_siblings_module_type.append(node_real_siblings_module_type_with_node_itself)
            else:
                print("[extract_all_unique_siblings] node_name: \'{:}\', node module type: \'{:}\', has {:} real siblings: \'{:}\'.".format(node_name, node_module_type, len(node_real_siblings), node_real_siblings))
                # for the two duplicated siblings lists, the node names included should be the same.
                # If the node name is already included in one of the unique_siblings_name list, which means the real_siblings of this node is duplicated with the unique_siblings_name list.
                # Otherwise, we should insert the [real_siblings + node_name] as a new unique_siblings_name list.
                has_include_siblings = False
                for unique_siblings_item in all_unique_siblings_name:
                    if node_name in unique_siblings_item:
                        has_include_siblings = True
                if has_include_siblings == False:
                    # direct insert will change the real_siblings info for the node in the fx_graph
                    node_real_siblings_with_node_itself = node_real_siblings.copy()
                    node_real_siblings_with_node_itself.insert(0, node_name)
                    node_real_siblings_module_type_with_node_itself = node_real_siblings_module_type.copy()
                    node_real_siblings_module_type_with_node_itself.insert(0, node_module_type)
                    all_unique_siblings_name.append(node_real_siblings_with_node_itself)
                    all_unique_siblings_module_type.append(node_real_siblings_module_type_with_node_itself)

        fx_graph['unique_siblings'] = {}
        fx_graph['unique_siblings']['name'] = all_unique_siblings_name
        fx_graph['unique_siblings']['module_type'] = all_unique_siblings_module_type
        fx_graph['unique_siblings']['node_type'] = 'auxiliary_info_node'

        if cls.__save_permutation_graph:
            cls.save_graph_to_json(fx_graph, save_dumped_graph_path_with_name=os.path.join(cls.__permutation_output_dir, './model_graph_extract_all_unique_siblings.json'))    # save the intermediate graph as JSON file for debugging
        return fx_graph

    @classmethod
    def find_real_siblings(cls, fx_graph):
        """This function is used to find all siblings for each node according to the whole network graph built with Torch.FX.
        we need to find siblings recursively, because siblings may have siblings via other parents we don't know about.
        """
        print("\n[find_real_siblings] Find all siblings for each node according to the whole network graph built with Torch.FX")
        for node_name in fx_graph.keys():
            node_real_siblings_name = []
            node_real_siblings_module_type = []
            node_real_parents = fx_graph.get(node_name).get('real_parents')
            node_module_type = fx_graph.get(node_name).get('module_type')
            if node_module_type not in ['torch.nn.modules.conv.Conv2d', 'torch.nn.modules.linear.Linear']:
                print("[find_real_siblings] node_name: \'{:}\', node module type: \'{:}\', has no real siblings.".format(node_name, node_module_type))
            else:
                print("[find_real_siblings] node_name: \'{:}\', node module type: \'{:}\', may have real siblings.".format(node_name, node_module_type))
                # sibling means the nodes share the same real parent
                for real_parent_item in node_real_parents:
                    for real_child_item in fx_graph.get(real_parent_item).get('real_children'):
                        if real_child_item != node_name:
                            sibling_module_type = fx_graph.get(real_child_item).get('module_type')
                            print("[find_real_siblings] node_name: \'{:}\', has one real sibling: \'{:}\', its real sibling module type: \'{:}\'.".format(node_name, real_child_item, sibling_module_type))
                            node_real_siblings_name.append(real_child_item)
                            node_real_siblings_module_type.append(sibling_module_type)

            # remove the duplicated real siblings
            exclusive_node_real_siblings_name = []
            exclusive_node_real_siblings_module_type = []
            item_index = 0
            duplicated_real_siblings = 0
            for item in node_real_siblings_name:
                if item not in exclusive_node_real_siblings_name:
                    exclusive_node_real_siblings_name.append(item)
                    exclusive_node_real_siblings_module_type.append(node_real_siblings_module_type[item_index])
                else:
                    duplicated_real_siblings = duplicated_real_siblings + 1
                item_index = item_index + 1
            if duplicated_real_siblings > 0:
                print("[find_real_siblings] node_name: \'{:}\', remove {:} duplicated real siblings.".format(node_name, duplicated_real_siblings))
            fx_graph[node_name]['real_siblings'] = exclusive_node_real_siblings_name
            fx_graph[node_name]['real_siblings_module_type'] = exclusive_node_real_siblings_module_type

        if cls.__save_permutation_graph:
            cls.save_graph_to_json(fx_graph, save_dumped_graph_path_with_name=os.path.join(cls.__permutation_output_dir, './model_graph_find_real_siblings.json'))    # save the intermediate graph as JSON file for debugging
        return fx_graph

    @classmethod
    def recursive_find_real_children(cls, node_name, fx_graph):
        """This function is used to recursively find the real children for each node according to the whole network graph built with Torch.FX.
        Used as the sub-function of find_real_children.
        """
        node_real_children_name = []
        node_real_children_module_type = []
        if node_name in fx_graph.keys():    # can be deleted, because node_name is already in the 'children' item in one node of the fx_graph
            node_children = fx_graph.get(node_name).get('children')
            node_module_type = fx_graph.get(node_name).get('module_type')
            has_visit_children_num = 0
            has_real_children_num = 0
            sub_node_need_recursive_search = []
            while has_visit_children_num < len(node_children):
                for child_name in node_children:
                    if child_name != 'output':    # 'output' node has no 'module_type'
                        child_module_type = fx_graph.get(child_name).get('module_type')
                        if child_module_type in ['torch.nn.modules.conv.Conv2d', 'torch.nn.modules.linear.Linear']:
                            print("[recursive_find_real_children] node_name: \'{:}\', has one real child: \'{:}\', its real child module type: \'{:}\'.".format(node_name, child_name, child_module_type))
                            node_real_children_name.append(child_name)
                            node_real_children_module_type.append(child_module_type)
                            has_real_children_num = has_real_children_num + 1
                        else:
                            print("[recursive_find_real_children] node_name: \'{:}\', its child: \'{:}\' with module type: \'{:}\', needs recursive search.".format(node_name, child_name, child_module_type))
                            sub_node_need_recursive_search.append(child_name)
                    else:
                        print("[recursive_find_real_children] node_name: \'{:}\', its child: \'{:}\' with no module type, is not its real child.".format(node_name, child_name))
                    has_visit_children_num = has_visit_children_num + 1
            if len(sub_node_need_recursive_search) > 0:
                for sub_node in sub_node_need_recursive_search:
                    if fx_graph.get(sub_node).get('real_children') == []:
                        sub_node_real_children_name, sub_node_real_children_module_type = cls.recursive_find_real_children(sub_node, fx_graph)
                    else:
                        # if the sub_node already find the 'real_children', no need to do recursive search
                        sub_node_real_children_name = fx_graph.get(sub_node).get('real_children')
                        sub_node_real_children_module_type = fx_graph.get(sub_node).get('real_children_module_type')
                    node_real_children_name.extend(sub_node_real_children_name)
                    node_real_children_module_type.extend(sub_node_real_children_module_type)
        return node_real_children_name, node_real_children_module_type

    @classmethod
    def find_real_children(cls, fx_graph):
        """This function is used to find the real children for each node according to the whole network graph built with Torch.FX.
        For example:
        The real children of Conv is the subsequent Conv/FC.
        The real children of BN or other no-need-permutataion layers is the subsequent Conv/FC.
        """
        print("\n[find_real_children] Find the real children for each node according to the whole network graph built with Torch.FX")
        from sys import version_info
        if version_info.major == 3 and version_info.minor >= 8:
            reversible_fx_graph_keys = fx_graph.keys()
        else:    # 'dict_keys' object is not reversible in previous of Python 3.8
            reversible_fx_graph_keys = list(fx_graph.keys())
        for node_name in reversed(reversible_fx_graph_keys):    # as the optimization, we need to find the real children from back to front, to use the already saved 'real_children'
            node_real_children_name = []
            node_real_children_module_type = []
            node_children = fx_graph.get(node_name).get('children')
            node_module_type = fx_graph.get(node_name).get('module_type')
            if node_module_type not in ['torch.nn.modules.conv.Conv2d', 'torch.nn.modules.linear.Linear']:
                print("\n[find_real_children] node_name: \'{:}\', node module type: \'{:}\', children num: {:}, recursive to find real children.".format(node_name, node_module_type, len(node_children)))
                node_real_children_name, node_real_children_module_type = cls.recursive_find_real_children(node_name, fx_graph)
            else:    # Quick method, but cannot get the real children for no-need-permutataion layers like BN
                print("\n[find_real_children] node_name: \'{:}\', node module type: \'{:}\', children num: {:}, can directly find real children.".format(node_name, node_module_type, len(node_children)))
                # if the node is in the 'real_parents' list of the other node, then the other node is the real children for this node
                for other_node_name in fx_graph.keys():
                    if (other_node_name != node_name) and (node_name in fx_graph.get(other_node_name).get('real_parents')):
                        child_module_type = fx_graph.get(other_node_name).get('module_type')
                        if child_module_type in ['torch.nn.modules.conv.Conv2d', 'torch.nn.modules.linear.Linear']:
                            print("[find_real_children] node_name: \'{:}\', has one real child: \'{:}\', its real child module type: \'{:}\'.".format(node_name, other_node_name, child_module_type))
                            node_real_children_name.append(other_node_name)
                            node_real_children_module_type.append(child_module_type)

            # remove the duplicated real children
            exclusive_node_real_children_name = []
            exclusive_node_real_children_module_type = []
            item_index = 0
            duplicated_real_children = 0
            for item in node_real_children_name:
                if item not in exclusive_node_real_children_name:
                    exclusive_node_real_children_name.append(item)
                    exclusive_node_real_children_module_type.append(node_real_children_module_type[item_index])
                else:
                    duplicated_real_children = duplicated_real_children + 1
                item_index = item_index + 1
            if duplicated_real_children > 0:
                print("[find_real_children] node_name: \'{:}\', remove {:} duplicated real children.".format(node_name, duplicated_real_children))
            fx_graph[node_name]['real_children'] = exclusive_node_real_children_name
            fx_graph[node_name]['real_children_module_type'] = exclusive_node_real_children_module_type

        if cls.__save_permutation_graph:
            cls.save_graph_to_json(fx_graph, save_dumped_graph_path_with_name=os.path.join(cls.__permutation_output_dir, './model_graph_find_real_children.json'))    # save the intermediate graph as JSON file for debugging
        return fx_graph

    @classmethod
    def find_real_parents(cls, fx_graph):
        """This function is used to find the real parents for each node according to the whole network graph built with Torch.FX.
        For example:
        The real parent of BN   is the previous Conv/FC.
        The real parent of Conv is the previous Conv/FC.
        """
        print("\n[find_real_parents] Find the real parents for each node according to the whole network graph built with Torch.FX")
        for node_name in fx_graph.keys():
            node_real_parents_name = []
            node_real_parents_module_type = []
            node_parents = fx_graph.get(node_name).get('parents')
            print("[find_real_parents] node_name: \'{:}\', parents num: {:}".format(node_name, len(node_parents)))

            has_visit_parent_num = 0
            while has_visit_parent_num < len(node_parents):
                for parent_name in node_parents:
                    if fx_graph.__contains__(parent_name):
                        parent_module_type = fx_graph.get(parent_name).get('module_type')
                        if parent_module_type in ['torch.nn.modules.conv.Conv2d', 'torch.nn.modules.linear.Linear']:
                            print("[find_real_parents] node_name: \'{:}\', has one real parent: \'{:}\', its real parent module type: \'{:}\'.".format(node_name, parent_name, parent_module_type))
                            node_real_parents_name.append(parent_name)
                            node_real_parents_module_type.append(parent_module_type)
                        else:
                            print("[find_real_parents] node_name: \'{:}\', has one/several real parent(s): \'{:}\', its real parent module type: \'{:}\'.".format(node_name, fx_graph[parent_name]['real_parents'], fx_graph[parent_name]['real_parents_module_type']))
                            for real_parent_item in fx_graph[parent_name]['real_parents']:
                                node_real_parents_name.append(real_parent_item)
                            for real_parent_module_type_item in fx_graph[parent_name]['real_parents_module_type']:
                                node_real_parents_module_type.append(real_parent_module_type_item)
                    else:
                        print("[find_real_parents] node_name: \'{:}\', has no real parent because this is the first node.".format(node_name))
                    has_visit_parent_num = has_visit_parent_num + 1

            # remove the duplicated real parents
            exclusive_node_real_parents_name = []
            exclusive_node_real_parents_module_type = []
            exclusive_node_real_parents_groups_param = []
            item_index = 0
            duplicated_real_parents = 0
            for item in node_real_parents_name:
                if item not in exclusive_node_real_parents_name:
                    exclusive_node_real_parents_name.append(item)
                    exclusive_node_real_parents_module_type.append(node_real_parents_module_type[item_index])
                    exclusive_node_real_parents_groups_param.append(fx_graph.get(item).get('groups_param'))
                else:
                    duplicated_real_parents = duplicated_real_parents + 1
                item_index = item_index + 1
            if duplicated_real_parents > 0:
                print("[find_real_parents] node_name: \'{:}\', remove {:} duplicated real parents.".format(node_name, duplicated_real_parents))
            fx_graph[node_name]['real_parents'] = exclusive_node_real_parents_name
            fx_graph[node_name]['real_parents_module_type'] = exclusive_node_real_parents_module_type
            fx_graph[node_name]['real_parents_groups_param'] = exclusive_node_real_parents_groups_param

        if cls.__save_permutation_graph:
            cls.save_graph_to_json(fx_graph, save_dumped_graph_path_with_name=os.path.join(cls.__permutation_output_dir, './model_graph_find_real_parent.json'))    # save the intermediate graph as JSON file for debugging
        return fx_graph

    @classmethod
    def build_fx_graph(cls, model, dump_fx_graph=False, save_dumped_fx_graph='./model_fx_graph.json'):
        """This function is used to build the whole network graph with Torch.FX features."""
        success = True
        torch_version = str(torch.__version__)
        torch_version_major = int(torch_version.split('.')[0])
        torch_version_minor = int(torch_version.split('.')[1])
        try:
            torch_version_minimum = int(torch_version.split('.')[2])
        except ValueError:    # support the none standard version
            torch_version_minimum = torch_version.split('.')[2]
        print("[build_fx_graph] The torch version is: {}, version major is: {}, version minor is: {}, version minimum is: {}".format(torch_version, torch_version_major, torch_version_minor, torch_version_minimum))
        if torch_version_major >= 1 and torch_version_minor >= 8:
            print("[build_fx_graph] The Torch.FX is supported.")
        else:    # Torch.FX is introduced in torch 1.8.0
            print("[build_fx_graph] The Torch.FX is not supported. So cannot build the Torch.FX graph.")
            success = False
            network_fx_graph = {}
            return network_fx_graph, success

        print("\n[build_fx_graph] Print the model structure with pure PyTorch function")
        print(model)

        print("\n[build_fx_graph] Build the module name and type dictionary")
        module_name_type_dict = {}
        module_name_group_conv_dict = {}
        for name, mod in model.named_modules():
            print("[build_fx_graph] module_name: {}, module type: {}".format(name, type(mod)))
            module_name_type_dict[name] = str(type(mod)).split("\'")[1]
            try:
                print("[build_fx_graph] this module has \'group\' param with value: {}".format(mod.groups))
                module_name_group_conv_dict[name] = str(mod.groups)
            except:
                module_name_group_conv_dict[name] = 'None'
                continue

        graph_module = cls.print_raw_fx_graph(model, print_tabular=True)

        # keep track of children and parents for each layer (could be call_module or call_function)
        print("\n[build_fx_graph] Print the children and parents relationship for each layer")
        network_fx_graph = {}
        for node in graph_module.graph.nodes:
            if node.op == 'placeholder':
                print("[build_fx_graph] This is the \'input\' node: {:}".format(node.target))
                continue
            elif node.op == 'get_attr':
                print("[build_fx_graph] This is the \'get_attr\' node: {:}".format(node.target))
                continue
            elif node.op == 'call_function':    # e.g. 'adaptive.avg.pool2d', 'add', 'cat', 'flatten', 'floordiv', 'getattr', 'getitem', 'hardsigmoid', 'mean', 'mul', 'relu', 'transpose'
                node_parent, node_children = get_node_parent_children(node)
                converted_node_name=convert_fx_node_name(node.name)
                print("[build_fx_graph] This is the \'call_function\' node: {:}, its parent list: {:}, its children list: {:}".format(converted_node_name, node_parent, node_children))
                network_fx_graph[converted_node_name] = {}
                network_fx_graph[converted_node_name]['parents'] = node_parent
                network_fx_graph[converted_node_name]['children'] = node_children
                network_fx_graph[converted_node_name]['fx_op'] = 'call_function'
            elif node.op == 'call_method':    # e.g. 'chunk', 'contiguous', 'mean', 'size', 'unsqueeze', 'view'
                node_parent, node_children = get_node_parent_children(node)
                converted_node_name=convert_fx_node_name(node.name)
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
                    print("[build_fx_graph][warning] The target name from Torch.FX is \'{:}\', the manually converted node name is \'{:}\', not the same one, choose the converted node name".format(node.target, converted_node_name))
                # assume the modules share the same target name have the same type, because converted_node_name may not be obtained by model.named_modules(), like some ReLU (defined in forward function)
                node_type = module_name_type_dict[node.target]
                print("[build_fx_graph] This is the \'call_module\' node: {:}, its parent list: {:}, its children list: {:}, its type: {:}".format(converted_node_name, node_parent, node_children, node_type))
                network_fx_graph[converted_node_name] = {}
                network_fx_graph[converted_node_name]['parents'] = node_parent
                network_fx_graph[converted_node_name]['children'] = node_children
                network_fx_graph[converted_node_name]['fx_op'] = 'call_module'
                network_fx_graph[converted_node_name]['module_type'] = node_type
                network_fx_graph[converted_node_name]['groups_param'] = module_name_group_conv_dict[node.target]
            elif node.op == 'output':
                print("[build_fx_graph] This is the \'output\' node: {:}".format(node.target))
                continue

        if dump_fx_graph:
            print("\n[build_fx_graph] Dump the overall dict for children and parents relationship into JSON file")
            cls.save_graph_to_json(network_fx_graph, save_dumped_graph_path_with_name=save_dumped_fx_graph)

        return network_fx_graph, success

    @classmethod
    def print_raw_fx_graph(cls, model, print_tabular=False, generate_python_code=False):
        """This function is used to print the intermediate representation (IR) - Graph representation with Torch.FX features."""
        from torch.fx import symbolic_trace
        # Symbolic tracing frontend - captures the semantics of the module
        try:
            symbolic_traced : torch.fx.GraphModule = symbolic_trace(model)
        except:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print("\n[print_raw_fx_graph] Meet the fatal fault when trying to symbolic trace the model with Torch.FX")
                raise
            exit(0)

        # High-level intermediate representation (IR) - Graph representation
        print("\n[print_raw_fx_graph] Print the intermediate representation (IR) with Torch.FX")
        print(symbolic_traced.graph)

        if print_tabular:
            print("\n[print_raw_fx_graph] Print the intermediate representation (IR) with Torch.FX in a table format")
            try:
                symbolic_traced.graph.print_tabular()
            except AttributeError:    # to avoid the AttributeError: 'Graph' object has no attribute 'print_tabular'
                print("[print_raw_fx_graph][Warning] \'print_tabular\' function is not supported in current Torch version. Skip!")

        # Code generation - valid Python code
        if generate_python_code:
            print("\n[print_raw_fx_graph] Create valid Python code matching the IR/Graph's semantics with Torch.FX")
            print(symbolic_traced.code)

        return symbolic_traced

    @classmethod
    def save_graph_to_json(cls, graph, save_dumped_graph_path_with_name='./model_fx_graph.json'):
        """This function is used to same the graph into JSON file."""
        # use dumps to transfer the dict to JSON string
        json_graph_str = json.dumps(graph)
        with open(save_dumped_graph_path_with_name, 'w', encoding='utf-8') as dumped_graph_file:
            dumped_graph_file.write(json_graph_str)  # write the transferred JSON string into JSON file
