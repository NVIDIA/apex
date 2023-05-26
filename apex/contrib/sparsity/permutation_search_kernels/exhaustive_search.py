from .permutation_utilities import *

################################################################################################################
# Exhaustive
#   Try them all
#   - order of columns within a group doesn't matter
#   - order of groups doesn't matter
#   - we can eliminate effective duplicates by defining aunique combination to be a sorted list of sorted groups
################################################################################################################

####################################################################
# generate unique permutations
####################################################################

# check if adding a column index to a current permutation would keep it in canonical form
# assumes that perm is in canonical form already!
def is_canonical(perm, col):
    # if it's a new group
    if len(perm) % 4 == 0:
        # every column ID < col needs to be in the permutation already
        for val in range(col):
            if val not in perm:
                return False
        # this new group needs to be sorted w.r.t. the previous group
        return col > perm[-4]

    # not a new group, just check to see if it will still be sorted
    return col > perm[-1]


# recursive: build a unique permutation one column index at a time
def generate_unique_combinations(built_permutation, remaining_columns, full_permutation_list, group_width):

    # base case: nothing else to add
    if len(remaining_columns) == 0:
        full_permutation_list.append(np.copy(built_permutation))
        if len(full_permutation_list) % 1000000 == 0:
            print(f"{len(full_permutation_list)} unique permutations found so far")

    # still more choices to make, so add each remaining column in turn column if it keeps everything sorted
    else:
        for c in range(len(remaining_columns)):
            # to satisfy our immutables (values within groups are sorted, groups are globally sorted),
            # only add this column if either:
            #   it's starting a new group and is larger than the previous group's first entry
            #   OR
            #   it's larger than the last value in the built_permutation
            col_to_add = remaining_columns[c]

            if is_canonical(built_permutation, col_to_add):
                # add the column to the running permutation, remove it from remaining columns
                built_permutation.append(col_to_add)
                remaining_columns.pop(c)
                # recurse
                generate_unique_combinations(built_permutation, remaining_columns, full_permutation_list, group_width)
                # remove the most recent column and put it back on the remaining column list where we found it (sorted)
                remaining_columns.insert(c, built_permutation.pop(-1))

import pickle
import os.path
from os import path
master_unique_permutation_list = {}
def generate_all_unique_combinations(C, M, must_use_all_groups = False):
    global master_unique_permutation_list
    if len(master_unique_permutation_list) == 0 and path.exists("master_list.pkl"):
        with open("master_list.pkl","rb") as cache:
            master_unique_permutation_list = pickle.load(cache)

    if (C,M) not in master_unique_permutation_list:
        full_permutation_list = []
        generate_unique_combinations([0], [c for c in range(1,C)], full_permutation_list, M)
        master_unique_permutation_list[(C,M)] = full_permutation_list

        with open("master_list.pkl", "wb") as cache:
            pickle.dump(master_unique_permutation_list, cache)

    unique_permutations = master_unique_permutation_list[(C,M)]

    return unique_permutations

# analytical solution
import math
def predict_unique_combinations(C, M):
    assert(C%M==0)
    G = int(C/M)
    return int(int(math.factorial(C)) / (int(math.pow(math.factorial(M),G)) * math.factorial(G)))

#################################################################
# exhaustively try all unique permutations
#################################################################

# exhaustively search the entire matrix
def search_matrix(matrix, group_width):
    # give up quickly if we'd go on forever
    prediction = predict_unique_combinations(matrix.shape[1], group_width)
    best_permutation = [c for c in range(matrix.shape[1])]
    if prediction > 1e10:
        print(f"There are {prediction} unique combinations with {matrix.shape[1]} columns and a group width of {group_width}, not searching.")
        return matrix, prediction, best_permutation

    start_time = time.perf_counter()
    full_permutation_list = generate_all_unique_combinations(matrix.shape[1], group_width)

    # found them, now try them
    best_improvement = 0.0
    use_cuda = use_gpu()
    if use_cuda and matrix.shape[1] >= 8 and group_width == 4:  # CUDA path only works for a group width of 4
        best_improvement, best_permutation = try_permutations_on_matrix(matrix, full_permutation_list)
    else:
        base_sum = sum_after_2_to_4(matrix)
        for i in range(1,len(full_permutation_list)):
            permutation = full_permutation_list[i]
            permuted = matrix[:, permutation]
            cur_improvement = sum_after_2_to_4(permuted) - base_sum
    
            if (cur_improvement > best_improvement):
                best_improvement = cur_improvement
                best_permutation = permutation
    seconds = time.perf_counter() - start_time
    return matrix[:, best_permutation], seconds, best_permutation, best_improvement


#############
# Stripe group handling
#############

# gather stripes from a larger matrix into a single matrix
def collect_stripes(matrix, stripes, group_width):
    subset = np.zeros((matrix.shape[0], len(stripes)*group_width))
    for s,stripe in enumerate(stripes):
        subset[...,s*group_width:s*group_width+group_width] = matrix[...,stripe*group_width:stripe*group_width+group_width]
    return subset

# apply the stripe group permutation to the entire permutation
def apply_stripe_group_permutation(sgp, stripes, group_width, permutation):
    new_permutation = permutation.copy()
    for subset_idx in range(len(sgp)):
        dst_stripe_idx = stripes[int(subset_idx / group_width)]
        dst_col_idx = subset_idx % group_width

        subset_val = sgp[subset_idx]
        src_stripe_idx = stripes[int(subset_val / group_width)]
        src_col_idx = subset_val % group_width

        new_permutation[dst_stripe_idx*group_width + dst_col_idx] = permutation[src_stripe_idx*group_width + src_col_idx]

    return new_permutation

# generate all possible stripe groups
def generate_stripe_groups(num_stripes, window_size):
    stripe_array = [[c] for c in range(num_stripes)]

    next_stripe_array = []
    for w in range(1, window_size):
        for g in range(len(stripe_array)):
            start_c = stripe_array[g][w-1]+1
            group = stripe_array[g]
            for c in range(start_c, num_stripes):
                new_group = group.copy()
                new_group.append(c)
                next_stripe_array.append(new_group)
        stripe_array = next_stripe_array
        next_stripe_array = []

    return set(tuple(stripe_array[g]) for g in range(len(stripe_array)))

# It is not safe to just reset the stripe_set as None here.
# When calling the Exhaustive_Search in E2E search, the stripe_set will not be reset as None.
stripe_set = None
stripe_set_config = None
# build the stripe map
def build_stripe_map(matrix, group_width, window_size, stripe_map, stripe_ids, perm_map, used_stripes):
    global stripe_set, stripe_set_config

    window_size = int(window_size / group_width)

    if stripe_set is None or stripe_set_config is None or stripe_set_config != (group_width, window_size):
        num_stripes = int(matrix.shape[1] / group_width)
        assert(group_width * num_stripes == matrix.shape[1])
        stripe_set = generate_stripe_groups(num_stripes, window_size)
        stripe_set_config = (group_width, window_size)

    # step through each, update the stripe_map/stripe_ids if necessary
    updates = 0
    use_cuda = use_gpu()
    gpu_list = []
    gpu_groups = []
    for i,s in enumerate(stripe_set):
        sg = [] # build the group of stripes, check if any members changed
        need_update = i >= len(stripe_map)
        for stripe in s:
            sg.append(stripe)
            if stripe in used_stripes:
                need_update = True

        # pre-populate if we're building fresh
        if i >= len(stripe_map):
            stripe_ids.append(sg)
            stripe_map.append(0.)
            perm_map.append([c for c in range(group_width * window_size)])

        # update entries if needed (only stripe_map and perm_map)
        if need_update:
            updates += 1

            if not use_cuda:    # do the work here if using the CPU
                subset = collect_stripes(matrix, sg, group_width)
                sub_result, sub_duration, permutation, improvement = search_matrix(subset, group_width)
                stripe_map[i] = improvement
                perm_map[i] = permutation
            else:               # otherwise, just track the work needed to farm off to the GPU
                gpu_groups.append(sg)
                gpu_list.append(i)

    if use_cuda: # if using the GPU, perform the work
        matrix_view = np.copy(matrix).astype(np.float32).flatten()
        all_permutations = generate_all_unique_combinations(window_size*group_width, group_width)
        num_permutations = len(all_permutations)
        permutation_view = np.copy(np.asarray(all_permutations)).astype(np.uint32).flatten()
        stripe_groups_view = np.asarray(gpu_groups).astype(np.uint32).flatten()
        num_gpu_groups = len(gpu_list)
        gpu_improvement = np.zeros((num_gpu_groups), dtype=np.float32).flatten()
        gpu_permutation = np.zeros((num_gpu_groups), dtype=np.uint32).flatten()

        result = permutation_search_cuda_kernels.build_permute_map(matrix_view,
                                                              matrix.shape[0],
                                                              matrix.shape[1],
                                                              stripe_groups_view,
                                                              num_gpu_groups,
                                                              window_size,
                                                              permutation_view,
                                                              window_size * group_width,
                                                              gpu_improvement,
                                                              gpu_permutation)

        # put the data where python expects it
        for i in range(len(gpu_list)):
            stripe_map[gpu_list[i]] = gpu_improvement[i]
            perm_map[gpu_list[i]] = all_permutations[gpu_permutation[i]]

    return stripe_map, stripe_ids, perm_map


# start performing stripe checks
sm_perturbations = 0
sm_perturbation_limit = 0
def use_stripe_map(matrix, group_width, stripe_map, stripe_ids, perm_map, permutation):
    global sm_perturbations, sm_perturbation_limit
    used_stripes = []
    stripe_groups_optimized = 0
    improvement = 0.0

    # set the traversal order
    ix = np.flip(np.argsort(stripe_map)) # small to large --> large to small

    for i in range(len(ix)):
        stripe_group_id = ix[i]
        perm = perm_map[stripe_group_id].copy()

        if stripe_map[stripe_group_id] <= np.finfo(np.float16).tiny*5.:
            # perturbations
            if len(used_stripes) == 0 and sm_perturbations < sm_perturbation_limit:
                sm_perturbations += 1
                # use this permutation, but swap two channels from left/right halves to include two stripes, no matter the group size
                stripe_group_id = ix[np.random.randint(len(ix))]
                perm = perm_map[stripe_group_id].copy()
                # a little easier to escape from
                src = np.random.randint(int(len(perm)/2))
                dst = int(len(perm)/2) + np.random.randint(int(len(perm)/2))
                perm[src],perm[dst] = perm[dst],perm[src]
            else:
                break

        stripe_group = stripe_ids[stripe_group_id]

        # don't work on stripes we've already touched
        touched_stripe = False
        for stripe in stripe_group:
            if stripe in used_stripes:
                touched_stripe = True
        if touched_stripe:
            continue

        # apply the permutation we've already found to this stripe group
        subset = collect_stripes(matrix, stripe_group, group_width)
        sub_result = subset[...,perm]
        permutation = apply_stripe_group_permutation(perm, stripe_group, group_width, permutation)

        # scatter the results, track what changed
        for s,stripe in enumerate(stripe_group):
            # see if this group is in canonical form (entry 0 a multiple of 4, contiguous values))
            group = perm[s*group_width:s*group_width+group_width] # columns in this group of the used permutation
            changed = False
            if group[0] % 4 != 0:
                changed = True
            for c in range(1,group_width):
                if group[c] != group[c-1]+1:
                    changed = True
                    break
            # if it's not, then it changed
            if changed:
                used_stripes.append(stripe_group[s])

            matrix[...,stripe*group_width:stripe*group_width+group_width] = sub_result[...,s*group_width:s*group_width+group_width]

        improvement += stripe_map[stripe_group_id]
        stripe_groups_optimized += 1

    return matrix, stripe_groups_optimized, stripe_map, stripe_ids, used_stripes, improvement, permutation

# entry point for exhaustive searches - both the entire matrix, as well as stripe groups
def Exhaustive_Search(matrix, stripe_group_size=-1, escape_attempts=0, permutation=None):
    global sm_perturbation_limit, sm_perturbations
    sm_perturbations = 0
    sm_perturbation_limit = escape_attempts
    if permutation is None:
        permutation = [c for c in range(matrix.shape[1])]

    # It is much safer to reset the stripe_set as None in the entry point of Exhaustive_Search
    global stripe_set, stripe_set_config
    stripe_set = None
    stripe_set_config = None

    # only support N:4 for now
    group_width = 4

    result = np.copy(matrix)

    # if the matrix is too large for a window size of 12, subdivide, then fix up with a global optimization with a window size of 8
    if group_width==4 and stripe_group_size==12 and matrix.shape[1] > 512:
        stripe_split = int(matrix.shape[1]/2/group_width)
        col_split = stripe_split * group_width
        result[:,:col_split], durationL, permutation[:col_split] = Exhaustive_Search(result[:,:col_split], stripe_group_size=stripe_group_size, escape_attempts=escape_attempts, permutation=permutation[:col_split])
        result[:,col_split:], durationR, permutation[col_split:] = Exhaustive_Search(result[:,col_split:], stripe_group_size=stripe_group_size, escape_attempts=escape_attempts, permutation=permutation[col_split:])
        escape_attempts = max(escape_attempts, 100)*10
        result,duration,permutation = Exhaustive_Search(result, stripe_group_size=8, escape_attempts=escape_attempts, permutation=permutation)
        return result, durationL+durationR+duration, permutation

    # small enough to optimize the entire matrix at once
    if stripe_group_size != -1 and stripe_group_size < matrix.shape[1]:
        stripe_map = []
        stripe_ids = []
        perm_map = []
        used_stripes = []

        # in practice, this work will be cached ahead of time; doing it now.
        # (Reading the cached list from disk can take several seconds, which shouldn't be counted against the search, but amortized over every layer in a network)
        generate_all_unique_combinations(stripe_group_size, group_width)

        start_time = time.perf_counter()

        while True:
            #print("[Debug][Exhaustive_Search] Before entering the build_stripe_map function.")
            #print("[Debug][Exhaustive_Search] Now the stripe_set value is: {}".format(stripe_set))
            stripe_map, stripe_ids, perm_map = build_stripe_map(result, group_width, stripe_group_size, stripe_map, stripe_ids, perm_map, used_stripes)
            result, stripe_groups_optimized, stripe_map, stripe_ids, used_stripes, improvement, permutation = use_stripe_map(result, group_width, stripe_map, stripe_ids, perm_map, permutation)

            # converged?
            if len(used_stripes) == 0:
                break

        duration = time.perf_counter() - start_time

    else: # no sliding window, single iteration
        print(f"Matrix has {matrix.shape[1]} columns and the search window is only {stripe_group_size}: searching exhaustively")
        result, duration, permutation, improvement = search_matrix(matrix, group_width)

    return result, duration, permutation
