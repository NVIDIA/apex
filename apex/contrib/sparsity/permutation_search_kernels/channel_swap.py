from .permutation_utilities import *

################################################################################################################
# Greedy Channel Swaps - iterative, deterministic, can be parallelized
#   1. Build a map of the magnitude improvement of involved stripes for all pairs of channel swaps
#   2. Sort the map, march through by decreasing improvement, skipping entries whose stripes have been modified
#   3. Repeat until there's no entry with positive improvement (convergence)
################################################################################################################

## try swapping columns and tracking magnitude after pruning
def try_swap(matrix, dst, src):
    src_base = sum_after_2_to_4(matrix[...,int(src/4)*4:int(src/4)*4+4])
    dst_base = sum_after_2_to_4(matrix[...,int(dst/4)*4:int(dst/4)*4+4])

    # swap
    matrix[...,[src,dst]] = matrix[...,[dst,src]]

    # check the Nx4 slices of the swapped columns
    src_sum = sum_after_2_to_4(matrix[...,int(src/4)*4:int(src/4)*4+4])
    dst_sum = sum_after_2_to_4(matrix[...,int(dst/4)*4:int(dst/4)*4+4])

    # swap back
    matrix[...,[src,dst]] = matrix[...,[dst,src]]

    return src_sum + dst_sum, (src_sum + dst_sum) - (src_base + dst_base)


## convert stripe and a swap indices to columns
def stripes_and_swap_idx_to_columns(stripe0, stripe1, idx):
    i = 0
    for c0 in range(4):
        for c1 in range(4):
            if i == idx:
                return stripe0*4+c0, stripe1*4+c1
            i += 1
    return None

## convert columns to stripe and swap indices
def columns_to_stripes_and_swap_idx(col0, col1):
    stripe0 = int(col0/4)
    col0 %= 4
    stripe1 = int(col1/4)
    col1 %= 4

    idx = 0
    for c0 in range(4):
        for c1 in range(4):
            if c0 == col0 and c1 == col1:
                return stripe0, stripe1, idx
            idx += 1
    return None

## build a list of stripe pairs that need their benefits recomputed because one stripe was modified
def build_stripe_pairs(matrix, used_stripes):
    stripe_pairs = []
    total_stripes = int(matrix.shape[1]/4)

    used_stripes = np.sort(used_stripes)
    for stripe0 in range(total_stripes-1):
        for stripe1 in range(stripe0, total_stripes):
            if stripe0 in used_stripes or stripe1 in used_stripes:
                stripe_pairs.append([stripe0,stripe1])

    return np.asarray(stripe_pairs)

## compute the benefit of swapping each pair of columns in the matrix using the GPU
## only update stripes' columns that appear in used_stripes to avoid unnecessary computations
def compute_swap_map(matrix, used_stripes):
    do_gpu = use_gpu()
    assert(do_gpu)

    stripe_pairs = build_stripe_pairs(matrix, used_stripes).astype(np.uint32)
    matrix_view = matrix.astype(np.float32).flatten()
    stripe_pairs_view = stripe_pairs.flatten()
    output = np.zeros((len(stripe_pairs)*16), dtype=np.float32).flatten()
    result = permutation_search_cuda_kernels.build_swap_map(matrix_view, matrix.shape[0], matrix.shape[1], stripe_pairs_view, output)
    
    # translate the flat array from the GPU to a map
    pair_improvement_map = {}
    for i,pair in enumerate(stripe_pairs):
        for swap_idx in range(16):
            col0, col1 = stripes_and_swap_idx_to_columns(pair[0], pair[1], swap_idx)
            pair_improvement_map[(col0, col1)] = output[i*16+swap_idx]
    return pair_improvement_map

## build the full swap map
def build_swap_map(matrix, swap_map, swap_ids, used_stripes, verbosity):        
    improvements = None

    # if we have a GPU and built kernels, pre-compute the needed values
    do_gpu = use_gpu()
    if do_gpu:
        if len(swap_map) == 0:
            used_stripes = [s for s in range(int(matrix.shape[1]/4))]
        improvements = compute_swap_map(matrix, used_stripes)

    idx = 0
    updates = 0
    for src in range(matrix.shape[1]-1):             # parallelize these loops
        for dst in range(src+1, matrix.shape[1]):

            # swapping within a stripe does nothing
            if int(src/4) == int(dst/4):
                continue

            # if we touched this stripe last time, update it
            if (int(src/4) in used_stripes) or (int(dst/4) in used_stripes) or len(swap_map) <= idx:
                tmp_improvement = 0.0
                
                # use the pre-computed values from the GPU if possible, otherwise compute on the CPU
                if do_gpu:
                    tmp_improvement = improvements[(src,dst)]
                else:
                    tmp_mag, tmp_improvement = try_swap(matrix, src, dst)
                updates += 1
    
                if len(swap_map) <= idx:
                    swap_map.append(tmp_improvement)
                    swap_ids.append((src,dst))
                else:
                    swap_map[idx] = tmp_improvement
                    swap_ids[idx] = (src,dst)

            idx += 1
    
    if verbosity > 15:
        print(f"\tupdated {updates} map entries")  
    return swap_map, swap_ids

def use_swap_map(matrix, swap_map, swap_ids, threshold, used_escape_attempts, escape_attempts, permutation, verbosity):
    used_stripes = []
    swaps = 0
    improvement = 0.0

    # set the traversal order and threshold
    ix = np.flip(np.argsort(swap_map))  # small to large -> large to small
    threshold = min(max(swap_map[ix[0]] * threshold, 0.0001),1.0)
    
    # iterate through the potential swaps in benefit order
    for swap in range(len(ix)):
        swap_id = ix[swap]
        src = swap_ids[swap_id][0]
        dst = swap_ids[swap_id][1]

        # early-out of swaps that are below the threshold (don't be so greedy)
        if swap_map[ix[swap]] < threshold:
            # see if an arbitrary swap helps things if we've converged
            if len(used_stripes) == 0 and used_escape_attempts < escape_attempts:
                swap_id = np.random.randint(len(swap_ids))
                if verbosity > 15:
                    print(F"converged, attempt #{used_escape_attempts+1} to jiggle out, using index {swap_id} into the sorted list={ix[swap_id]}")
                swap_id =ix[swap_id]
                src = swap_ids[swap_id][0]
                dst = swap_ids[swap_id][1]
                used_escape_attempts += 1
            else:
                break

        # skip swaps that include a stripe we've already modified
        if int(src/4) in used_stripes or int(dst/4) in used_stripes:
            continue
        
        # we'll need to update these stripes later
        used_stripes.append(int(src/4))
        used_stripes.append(int(dst/4))

        # make the swap
        if verbosity > 20:
            print(F"\t{swap}\t{src},{dst}  {swap_map[swap_id]:.4f}")
        matrix[...,[src,dst]] = matrix[...,[dst,src]]
        permutation[src],permutation[dst] = permutation[dst],permutation[src]
        improvement += swap_map[swap_id]
        swaps += 1

    return matrix, swaps, swap_map, swap_ids, used_stripes, improvement, used_escape_attempts, permutation

def Channel_Swap(matrix, escape_attempts=0, verbosity=0, permutation=None):
    threshold = 0.00001
    used_escape_attempts = 0

    # initialize
    if permutation is None:
        permutation = [c for c in range(matrix.shape[1])]
    swap_map = []
    swap_ids = []
    used_stripes = []
    swap_count = 0
    iterations = 0
    agg_improvement = 0.
    cur_total_sum = sum_after_2_to_4(matrix)
    start_time = time.perf_counter()

    # do the work
    swapped = 1 # just start with nonzero value to fall into the loop
    while swapped > 0:
        swap_map, swap_ids = build_swap_map(matrix, swap_map, swap_ids, used_stripes, verbosity)
        matrix, swapped, swap_map, swap_ids, used_stripes, improvement, used_escape_attempts, permutation = use_swap_map(matrix, swap_map, swap_ids, threshold, used_escape_attempts, escape_attempts, permutation, verbosity)
        agg_improvement += improvement
    
        # keep track of statistics, print occasionally
        swap_count += swapped
        if verbosity > 10:
            iterations += 1
            cur_total_sum += agg_improvement
            duration = time.perf_counter() - start_time
            print(F"\t{iterations:8} {cur_total_sum:7.2f} {agg_improvement:7.2f} {swap_count:4} {agg_improvement/max(swap_count,1):5.2f} {duration:7.2f}")
            agg_improvement = 0.
            swap_count = 0

    # final status
    seconds = time.perf_counter() - start_time

    return matrix, seconds, permutation
