import numpy as np
from .permutation_utilities import *
from .exhaustive_search import Exhaustive_Search

def accelerated_search_for_good_permutation(matrix_group, options=None, verbosity=0):
    """This function is used to call the permutation search CUDA kernels.
    users can provide prefer search strategy by providing a valid 'options' as a dictionary,
    or users can implement their customized 'accelerated_search_for_good_permutation' function.
    """
    input_matrix = matrix_group.cpu().detach().numpy()
    if verbosity > 1:
        print("\n[accelerated_search_for_good_permutation] input matrix shape: \'{:}\'.".format(input_matrix.shape))

    result = np.copy(input_matrix)
    # init a sequential permutation search sequence
    input_channel_num = matrix_group.size(1)
    permutation_sequence = [n for n in range(input_channel_num)]
    duration = 0.0

    if options == None:
        options = {}
    if 'strategy' not in options:    # right now, the default permutation search strategy is: 'exhaustive' search
        options['strategy'] = 'exhaustive'

    if verbosity > 1:
        print("[accelerated_search_for_good_permutation] the permutation strategy is: \'{:} search\'.".format(options['strategy']))

    # define sub options for each search strategy
    if options['strategy'] == 'exhaustive':
        # right now, the default options for 'exhaustive' search is: 'exhaustive,8,100'
        if 'stripe_group_size' not in options:
            options['stripe_group_size'] = 8
        if 'escape_attempts' not in options:
            options['escape_attempts'] = 100
    elif options['strategy'] == 'progressive channel swap':
        # just swaps meaningful channels, keeping the good swaps, until the search time limit expires.
        if 'progressive_search_time_limit' not in options:
            options['progressive_search_time_limit'] = 60
        if 'improvement_threshold' not in options:
            options['improvement_threshold'] = 1e-9

    # execute the requested strategy
    if options['strategy'] == 'exhaustive':
        result, duration, permutation_sequence = Exhaustive_Search(result, stripe_group_size=options['stripe_group_size'], escape_attempts=options['escape_attempts'])
    elif options['strategy'] == 'progressive channel swap':
        real_swap_num = 0
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < options['progressive_search_time_limit']:
            src = np.random.randint(result.shape[1])
            dst = np.random.randint(result.shape[1])
            src_group = int(src/4)
            dst_group = int(dst/4)
            if src_group == dst_group:    # channel swapping within a stripe does nothing
                continue
            new_sum, improvement = try_swap(result, dst, src)
            if improvement > options['improvement_threshold']:
                result[...,[src,dst]] = result[...,[dst,src]]
                permutation_sequence[src], permutation_sequence[dst] = permutation_sequence[dst], permutation_sequence[src]
                real_swap_num += 1
        duration = time.perf_counter() - start_time
        if verbosity > 1:
            print("\tFinally swap {} channel pairs until the search time limit expires.".format(real_swap_num))
    elif options['strategy'] == 'user defined':    # need to get the permutated matrix (result) by applying customized permutation search function
        if verbosity > 1:
            print("[accelerated_search_for_good_permutation] Use the user customized permutation search function!")
    else:
        if verbosity >= 0:
            print("[accelerated_search_for_good_permutation] Cannot find the implementation of the required strategy!")
    
    if verbosity > 1:
        print("[accelerated_search_for_good_permutation] Take {:.4f} seconds to search the permutation sequence.".format(duration))

    return permutation_sequence
