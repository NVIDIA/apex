import numpy as np
import time
import sys

# permutation-specifics
sys.path.append("../")
from permutation_search_kernels.permutation_utilities import *
from permutation_search_kernels.exhaustive_search import Exhaustive_Search
from permutation_search_kernels.channel_swap import Channel_Swap

# Arguments
import argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
       return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
       return False
    else:
       raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Test channel permutations')
parser.add_argument('--infile',                 default='random',       type=str,       help='input file or "random"')
parser.add_argument('--channels',               default=384,            type=int,       help='random input channel count (C)')
parser.add_argument('--filters',                default=96,             type=int,       help='random input filter count (K)')
parser.add_argument('--verbosity',              default=0,              type=int,       help='print status updates')
parser.add_argument('--seed',                   default=1,              type=int,       help='random seed')
parser.add_argument('--pretty_print',           default=True,           type=str2bool,  help='print the table for pretty viewing (as opposed to strict .csv)')
parser.add_argument('--unstructured',           default=0.0,            type=float,     help='perform unstructured pruning to a target sparsity before processing, emulate an unstructured sparse network. "-1" will find the minimum sparsity required to achieve a perfect permutation')
parser.add_argument('--gpu',                    default=True,           type=str2bool,  help='uses a gpu to accelerate the search if possible')
parser.add_argument('--check_permutation',      default=False,          type=str2bool,  help='check that the tracked permutation matches the recovered permutation')
parser.add_argument('--intermediate_steps',     default=0,              type=int,       help='find roughly evenly-spaced permutations in efficacy')
parser.add_argument('--print_permutation',      default=False,          type=str2bool,  help='print the final permutation found by each strategy')
parser.add_argument('strategies',               metavar='strategy',     type=str,       nargs='+', help='strategies to try')

## binary search for the minimum sparsity necessary to achieve a perfect permutation with some strategy
def find_minimum_sparsity(matrix, search_function, **kwargs):
    duration = 0
    min_sparsity = 50
    max_sparsity = 100
    sparsity = 75
    verbosity = 0
    if 'verbosity' in kwargs:
        verbosity = kwargs['verbosity']

    while min_sparsity < max_sparsity:
        if verbosity > 5:
            print(f"\tlooking now at {sparsity} (between {min_sparsity} and {max_sparsity})")

        # prepare unstructured sparse matrix, get row sparsity magnitude
        tmp_result = unstructured_prune(result, sparsity/100.0)
        local_unpruned_magnitude = np.sum(np.abs(tmp_result))
        local_unstructured_rows_magnitude = magnitude_after_pruning_rows(tmp_result, rate=0.5)

        # quick check to see if this sparsity is trivially too low
        if local_unstructured_rows_magnitude*1.0001 < local_unpruned_magnitude:
            if verbosity > 5:
                print(f"Skipping sparsity {sparsity} since there's no perfect permutation (unstructured mag {local_unpruned_magnitude} is larger than sparse rows {local_unstructured_rows_magnitude}).")
            min_sparsity = sparsity+1
            sparsity = int(min_sparsity + (max_sparsity - min_sparsity)/2.0)
            continue

        tmp_result, tmp_duration, found_permutation = search_function(tmp_result, **kwargs)
        duration += tmp_duration
        nonzeros = np.count_nonzero(tmp_result)
        tmp_result = apply_2_to_4(tmp_result)
        nonzeros_after_2to4 = np.count_nonzero(tmp_result)
        if nonzeros == nonzeros_after_2to4:  # found a winner, are we done?
            if verbosity > 3:
                print(f"Found an unstructured sparsity that we can turn into 2:4: {sparsity}")

            max_sparsity = sparsity
            if max_sparsity <= min_sparsity and verbosity > 0:
                print(f"Found the minimum unstructured sparsity that we can turn into 2:4: {sparsity}")
                break
        else:
            if verbosity > 5:
                print(f"Unstructured sparsity {sparsity} was insufficient to produce 2:4 sparsity")
            min_sparsity = sparsity+1
            if max_sparsity <= min_sparsity and verbosity > 0:
                print(f"Found the minimum unstructured sparsity that we can turn into 2:4: {max_sparsity}")
                sparsity = max_sparsity
                break

        sparsity = int(min_sparsity + (max_sparsity - min_sparsity)/2.0)
	
    return sparsity, duration


# Entry point
if __name__ == "__main__":
    args = parser.parse_args()
    verbosity = args.verbosity
    np.random.seed(seed=args.seed)
    use_gpu(initial_override=args.gpu)

    # get or create the input matrix    
    input_vals = np.random.rand(args.filters, args.channels)
    if args.infile != "random":
        if 'npy' in args.infile:
            input_vals = np.load(args.infile, 'r')
        shp = input_vals.shape
        shp_str = str(shp).replace(",","x")
        newshp_str = ''
        if len(shp) == 4:   # K,C,R,S -> RSK,C
            input_vals = np.transpose(input_vals,(2,3,0,1)).flatten().reshape((shp[2]*shp[3]*shp[0], shp[1]))
            newshp_str = str(input_vals.shape).replace(",","x")
        print(f"{args.infile},{shp_str},{newshp_str}")
        if input_vals.shape[1] % 4 != 0:
            print(f"Unfriendly shape {input_vals.shape}, not pruning.")
            sys.exit()

    # unstructured prune if requested
    if args.unstructured > 0.0:
        args.unstructured = min(args.unstructured, 1.0)
        input_vals = unstructured_prune(input_vals, args.unstructured)
        print(f"{args.infile} pruned to {args.unstructured*100.:>.1f} sparsity, shape is {input_vals.shape}")
    
    # calculate some early metrics
    sorted_magnitudes = np.sort(np.abs(input_vals), axis=None)
    unpruned_magnitude = np.sum(sorted_magnitudes)
    num_weights = sorted_magnitudes.size
    unstructured_magnitude = np.sum(sorted_magnitudes[int(num_weights/2):])
    unstructured_rows_magnitude = magnitude_after_pruning_rows(input_vals, rate=0.5)
    simple_2to4 = apply_2_to_4(np.copy(input_vals))
    simple_2to4_magnitude = sum_after_2_to_4(input_vals)
    tmp_time = time.perf_counter()
    simple_2to4_magnitude = sum_after_2_to_4(input_vals)
    default_duration = time.perf_counter() - tmp_time
    best_magnitude = unstructured_rows_magnitude

    best_lost_magnitude = unpruned_magnitude - best_magnitude
    base_lost_magnitude = unpruned_magnitude - simple_2to4_magnitude

    # prep results table
    final_metric = 'efficacy'
    if args.unstructured < 0.0:
        final_metric = 'min_sparsity'
    if args.pretty_print:
        print(f"{'strategy':<35},{'magnitude':>15},{final_metric:>15},{'duration':>15}")
        print(f"{'unpruned':<35},{unpruned_magnitude:>15.3f},{'-':^15},{'-':^15}")
        print(f"{'unstructured':<35},{unstructured_magnitude:>15.3f},{'-':^15},{'-':^15}")
        print(f"{'50% rows':<35},{unstructured_rows_magnitude:>15.3f},{'100.0':>15},{'-':^15}")
        print(f"{'default 2:4':<35},{simple_2to4_magnitude:>15.3f},{'0.0':>15},{default_duration:>15.3f}")
    else:
        print(f"strategy,magnitude,{final_metric},duration")
        print(f"unpruned,{unpruned_magnitude},-,-")
        print(f"unstructured,{unstructured_magnitude},-,-")
        print(f"50%_rows,{unstructured_rows_magnitude},100.0,-")
        print(f"2:4,{simple_2to4_magnitude},0.0,{default_duration}")
    

    # try the requested strategies
    for i,strategy in enumerate(args.strategies):
        result = np.copy(input_vals)
        np.random.seed(seed=args.seed)

        duration = 0.0
        min_sparsity = 0.0
        strat_split = strategy.split(",")
        found_permutation = None
    
        # optimize stripe groups
        if strat_split[0] == 'optimize_stripe_groups':
            stripe_group_size_in_cols = 8
            if len(strat_split) >= 2:
                stripe_group_size_in_cols = int(strat_split[1])
            escape_attempts = 100
            if len(strat_split) >= 3:
                escape_attempts = int(strat_split[2])

            if args.unstructured >= 0.0: # just perform the search on the current matrix
                result,duration,found_permutation = Exhaustive_Search(result, stripe_group_size=stripe_group_size_in_cols, escape_attempts=escape_attempts)
            else:                        # find the minimum sparsity needed to transparently transform the input
                min_sparsity,duration = find_minimum_sparsity(result, Exhaustive_Search, stripe_group_size=stripe_group_size_in_cols, escape_attempts=escape_attempts)
                result = unstructured_prune(result, min_sparsity/100.0)
    
        # channel swaps
        elif strat_split[0] == 'channel_swap':
            escape_attempts= 0
            if len(strat_split) >= 2:
                escape_attempts = int(strat_split[1])
               
            if args.unstructured >= 0.0: # just perform the search on the current matrix
                result,duration,found_permutation = Channel_Swap(result, escape_attempts=escape_attempts, verbosity=verbosity)
            else:                        # find the minimum sparsity needed to transparently transform the input
                min_sparsity,duration = find_minimum_sparsity(result, Channel_Swap, escape_attempts=escape_attempts, verbosity=verbosity)
                result = unstructured_prune(result, min_sparsity/100.0)

        # random permutations
        elif strat_split[0] == 'random':
            if args.unstructured < 0.0: # searching for minimum sparsity not supported for random permutations
                continue

            num_perms = 10
            if len(strat_split) >= 2 and int(strat_split[1]) >= 1:
                num_perms = int(strat_split[1])

            # try the seeds/permutations
            permutation = [c for c in range(result.shape[1])]
            best_sum = sum_after_2_to_4(result)
            best_perm = permutation.copy()
            start_time = time.perf_counter()
            for x in range(num_perms):
                permutation = np.random.permutation(permutation)
                cur_sum = sum_after_2_to_4(result[:,permutation])
                if cur_sum > best_sum:
                    best_sum = cur_sum
                    best_perm = permutation.copy()
                    if verbosity > 0:
                        print(f"\tnew best permutation {x} found with magnitude {best_sum:>15.3f}")
                elif verbosity > 5:
                    print(f"\tpermutation {x} magnitude too low: {cur_sum:>15.3f}")
            duration = time.perf_counter() - start_time
            result = result[:,best_perm]
            found_permutation = best_perm
                        
        else:
            print(f"Unknown strategy: {strategy}!")
            sys.exit()


        # report stats for this strategy
        cur_mag = sum_after_2_to_4(result)
        cur_eff = efficacy(best_lost_magnitude, base_lost_magnitude, unpruned_magnitude - cur_mag)*100.0
        final_metric = cur_eff
        if args.unstructured < 0.0:
            final_metric = min_sparsity
        perm_distance = ""

        error = None
        if args.check_permutation and found_permutation is not None:
            recovered_perm = find_permutation(result, input_vals)
        
            error = False
            for c in range(len(recovered_perm)):
                if recovered_perm[c] != found_permutation[c]:
                    if verbosity > 0:
                        print(f"tracked permutation at index {c} was {found_permutation[c]}, but the recovered permutation thought it was {recovered_perm[c]}")
                    error = True

        # if requested, generate permutations that divide the efficacy space into equal steps
        if args.intermediate_steps != 0:
            magnitude_targets = None
            if args.intermediate_steps != 0:
                ratios = [step/float(args.intermediate_steps+1) for step in range(1,args.intermediate_steps+1)]
                mag_diff = cur_mag - (unpruned_magnitude - base_lost_magnitude)
                magnitude_targets = [(unpruned_magnitude - base_lost_magnitude) + mag_diff * ratio for ratio in ratios]
            perm_distance, target_permutations = permutation_distance(found_permutation, [c for c in range(result.shape[1])], matrix=input_vals, magnitude_targets=magnitude_targets, debug=False, verbosity=verbosity)
            if target_permutations is not None:
                for target_permutation in target_permutations:
                    print(target_permutation)

        error_str = ""
        if error is not None:
            error_str = ",       correct"
            if error:
                error_str = ",      mismatch"

        if args.pretty_print:
            print(f"{strategy:35},{cur_mag:>15.3f},{final_metric:>15.1f},{duration:>15.3f}{error_str:>15}")
        else:
            strat_string = strategy.replace(",","_")
            print(f"{strat_string},{cur_mag},{final_metric},{duration}{error_str}")

        if args.print_permutation and found_permutation is not None:
            print(found_permutation)


