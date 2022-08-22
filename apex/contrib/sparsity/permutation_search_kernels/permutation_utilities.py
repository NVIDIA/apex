import numpy as np
import time
import subprocess
import math

gpus_tested = False
gpus_found = 0
kernels_found = True
try:
    import permutation_search_cuda as permutation_search_cuda_kernels
    print(f"Found permutation search CUDA kernels")
except ImportError:
    
    try:
        from . import permutation_search_cuda as permutation_search_cuda_kernels
        print(f"Found permutation search CUDA kernels for standalone testing")

    except ImportError:
        print(f"Could not find permutation search CUDA kernels, falling back to CPU path")
        kernels_found = False

def use_gpu(initial_override = True): 
    global gpus_tested, gpus_found, kernels_found
    if not gpus_tested:
        if not initial_override:
            gpus_tested = True
            return False

        try:
            gpus_found = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
            print(f"Found {gpus_found} gpus")
        except:
            gpus_found = 0
            print(f"Could not find nvidia-smi, please check your cuda installation")

        gpus_tested = True
    
    return gpus_found > 0 and kernels_found

##############################################################################################
# pruning utilities
##############################################################################################
## apply 2:4 to some matrix
def apply_2_to_4(matrix):
    for row in range(matrix.shape[0]):
        for col in range(0,matrix.shape[1],4):
            ix = np.argsort(np.abs(matrix[row,col:col+4]))
            matrix[row,col+ix[0]] = 0.0
            matrix[row,col+ix[1]] = 0.0
    return matrix

## find the sum of magnitudes if 2:4 were applied to a matrix
def sum_after_2_to_4(matrix):
    cur_sum = 0.0
    use_cuda = use_gpu()
    if not use_cuda:
        for row in range(matrix.shape[0]):
            for col in range(0,matrix.shape[1],4):
                ix = np.argsort(np.abs(matrix[row,col:col+4]))
                cur_sum += abs(matrix[row,col+ix[2]])
                cur_sum += abs(matrix[row,col+ix[3]])
    else:
        matrix = matrix.astype(np.float32)
        cuda_sum = np.zeros((1), dtype=np.float32)
        matrix_view = np.copy(matrix).flatten()
        sum_view = cuda_sum.flatten()
        blocks = max(int(matrix.shape[1]/4/2), 1)
        threads = min(max(math.ceil(matrix.shape[0]/4), 1), 1024)
        result = permutation_search_cuda_kernels.sum_after_2_to_4(matrix_view,
                                                             matrix.shape[0],
                                                             matrix.shape[1],
                                                             0,
                                                             matrix.shape[1],
                                                             blocks,
                                                             threads,
                                                             sum_view)
        cur_sum = sum_view[0]
    return cur_sum

# perform unstructured pruning on some matrix
def unstructured_prune(matrix, sparsity):
    shp = matrix.shape
    matrix = matrix.flatten()
    ix = np.argsort(matrix)
    ix = ix[:int(len(ix)*sparsity)]
    matrix[ix] = 0.0
    matrix = np.reshape(matrix, shp)
    return matrix

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

## magnitude improvement from the naive 2:4 matrix / how much was lost by naive 2:4 compared to the optimal
def efficacy(optimal_lost_magnitude, base_lost_magnitude, cur_lost_magnitude):
    if base_lost_magnitude == optimal_lost_magnitude:
        eff = 1.0
    else:
        eff = (base_lost_magnitude - cur_lost_magnitude) / (base_lost_magnitude - optimal_lost_magnitude)
    return eff

## find the magnitude if the rows of a matrix were pruned independently, without structure
def magnitude_after_pruning_rows(matrix, rate=0.5):
    magnitude = 0.
    cols = matrix.shape[1]
    for r in range(matrix.shape[0]):
        rowVals = matrix[r]
        rowVals = np.sort(np.abs(rowVals))
        magnitude += np.sum(rowVals[int(cols*rate):])

    return magnitude



##############################################################################################
# permutation utilities
##############################################################################################

## exhaustively search an entire matrix on the GPU
def try_permutations_on_matrix(matrix, permutations):
    use_cuda = use_gpu()
    assert(use_cuda) # caller should have checked
    matrix = np.copy(matrix)
    matrix = matrix.astype(np.float32)
    matrix_view = np.copy(matrix).flatten()
    permutations_view = np.copy(np.asarray(permutations)).astype(np.uint32).flatten()

    stripe_groups = np.asarray([[s for s in range(int(matrix.shape[1]/4))]]).astype(np.uint32)
    stripe_groups_view = stripe_groups.flatten()

    improvement = np.zeros((1), dtype=np.float32).flatten()
    permutation = np.zeros((1), dtype=np.uint32).flatten()

    result = permutation_search_cuda_kernels.check_permutations(matrix_view,
                                                           matrix.shape[0],
                                                           matrix.shape[1],
                                                           stripe_groups_view,
                                                           len(stripe_groups[0]),
                                                           len(stripe_groups),
                                                           permutations_view,
                                                           len(permutations),
                                                           improvement,
                                                           permutation)
    return improvement[0], permutations[permutation[0]]

## find the permutation needed to make matrix A look like matrix B
def find_permutation(A, B):
    permutation = []
    for col in range(A.shape[1]):
        Avals = A[...,col]
        for bcol in range(B.shape[1]):
            if np.all(Avals - B[...,bcol] == np.zeros(Avals.shape)):
                permutation.append(bcol)
                break
    return permutation


########################################
# reasonable method to find distance between permutations
# this is used to generate permutations "between" two other permutations to divide efficacy space
#######################################

## separate a flat permutation array into its groups, sort each group and the overall order to
## put the output into a canonical order: if two permutations have the same groups, they should appear identical
def make_grouped(A):
    groups = []
    for x in range(0,len(A),4):
        group = []
        for c in range(4):
            group.append(A[x+c])
        group = np.sort(group)

        groups.append(group)
    return groups

## given two permutations, find the groups they have in common
def common_groups(A, B):
    Ag = make_grouped(A)
    Bg = make_grouped(B)

    # convert to sets to take the intersection
    As = set(tuple(Ag[g]) for g in range(len(Ag)))
    Bs = set(tuple(Bg[g]) for g in range(len(Bg)))
    common = As.intersection(Bs)

    # flatten
    C = []
    for s in common:
        for v in s:
            C.append(v)

    # group
    return make_grouped(C)

## given two permutations, remove the groups that are common between them
def remove_common_groups(A, B):
    Ag = make_grouped(A)
    Bg = make_grouped(B)

    # convert to sets to take set difference
    As = set(tuple(Ag[g]) for g in range(len(Ag)))
    Bs = set(tuple(Bg[g]) for g in range(len(Bg)))
    Ad = As - Bs
    Bd = Bs - As

    # turn the differences back into flat arrays
    A = []
    for s in Ad:
        for v in s:
            A.append(v)
    B = []
    for s in Bd:
        for v in s:
            B.append(v)

    # group to put into canonical order, re-flatten
    A = make_grouped(A)
    B = make_grouped(B)
    A = [item for sublist in A for item in sublist]
    B = [item for sublist in B for item in sublist]

    return A,B

## given two permutations, find which elements in B need to go where to look like A
def group_differences(A, B):
    Ag = make_grouped(A)
    Bg = make_grouped(B)

    wrong_entries = []
    #for g,group in enumerate(Bg):
    for g in range(len(Bg)):
        group = Bg[g]
        for i in range(len(group)):
            val = group[i]
            if val not in Ag[g]:
                group_in_a = int(np.where(A == val)[0][0] / 4)
                wrong_entries.append((val, g, group_in_a))

    return wrong_entries

## (val, cur_group, desired_group) ==> dict[(cur_group, desired_group)] = [vals]
def dictify(wrong_entries):
    result = {}
    for entry in wrong_entries:
        key = (entry[1], entry[2])
        if key in result:
            result[key].append(entry[0])
        else:
            result[key] = [entry[0]]
    return result

## move groups of B to where they best match A's groups
def move_groups_to_match(B, A, debug=False):
    Ag = make_grouped(A)
    Bg = make_grouped(B)

    new_Bg = [[] for g in range(len(Ag))]
    wrong_entry_dict = dictify(group_differences(A, B))

    if debug:
        print(f"MGTM:\n\tAg: {Ag}\n\tBg: {Bg}\n\tWED: {wrong_entry_dict}")

    moved_groups = []

    keys_to_del = []
    # move triples to the right spot
    for k in wrong_entry_dict.keys():
        if k[0] in moved_groups:
            keys_to_del.append(k)
            continue

        if len(wrong_entry_dict[k]) == 3:
            new_Bg[k[1]] = Bg[k[0]]
            moved_groups.append(k[0])
            keys_to_del.append(k)
            if debug:
                print(f"MGTM: moved triple {wrong_entry_dict[k]} from group {k[0]} to group {k[1]}")

    for k in keys_to_del:
        del wrong_entry_dict[k]
    keys_to_del = []

    # move doubles
    for k in wrong_entry_dict.keys():
        # if we've already moved the group to which this key belongs, remove it
        if k[0] in moved_groups:
            keys_to_del.append(k)
            continue

        if len(wrong_entry_dict[k]) == 2:
            if len(new_Bg[k[1]]) == 0:  # move it to its requested destination if possible
                new_Bg[k[1]] = Bg[k[0]]
                keys_to_del.append(k)
                assert(k[0] not in moved_groups)
                moved_groups.append(k[0])
                if debug:
                    print(f"MGTM: moved double {wrong_entry_dict[k]} from group {k[0]} to its preferred group {k[1]}")
            elif len(new_Bg[k[0]]) == 0:                       # otherwise leave it where it is (if possible)
                new_Bg[k[0]] = Bg[k[0]]
                keys_to_del.append(k)
                assert(k[0] not in moved_groups)
                moved_groups.append(k[0])
                if debug:
                    print(f"MGTM: left double {wrong_entry_dict[k]} where it was in group {k[0]}")
    for k in keys_to_del:
        del wrong_entry_dict[k]
    keys_to_del = []

    # move singles
    # try to leave things where they are to prevent oscillating
    for k in wrong_entry_dict.keys():
        if k[0] in moved_groups:
            keys_to_del.append(k)
            continue

        if len(new_Bg[k[1]]) == 0: # requested destination
            new_Bg[k[1]] = Bg[k[0]]
            keys_to_del.append(k)
            assert(k[0] not in moved_groups)
            moved_groups.append(k[0])
            if debug:
                print(f"MGTM: moved single {wrong_entry_dict[k]} from group {k[0]} to its preferred group {k[1]}")

        elif len(new_Bg[k[0]]) == 0:
            new_Bg[k[0]] = Bg[k[0]]
            keys_to_del.append(k)
            assert(k[0] not in moved_groups)
            moved_groups.append(k[0])
            if debug:
                print(f"MGTM: left group {wrong_entry_dict[k]} where it was in group {k[0]}")

    for k in keys_to_del:
        del wrong_entry_dict[k]
    keys_to_del = []

    # put what's left where it'll fit
    for k in wrong_entry_dict.keys():
        if k[0] in moved_groups:
            keys_to_del.append(k)
            continue

        for dst in range(len(new_Bg)):
            if len(new_Bg[dst]) == 0:
                new_Bg[dst] = Bg[k[0]]
                keys_to_del.append(k)
                assert(k[0] not in moved_groups)
                moved_groups.append(k[0])
                if debug:
                    print(f"MGTM: put group {wrong_entry_dict[k]} where it found a spot in group {dst}")
                break

    for k in keys_to_del:
        del wrong_entry_dict[k]
    keys_to_del = []

    assert(len(wrong_entry_dict) == 0)
    Agsize = sum( [ len(group) for group in Ag] )
    Bgsize = sum( [ len(group) for group in new_Bg] )
    assert(Agsize == Bgsize)
    new_B = [item for sublist in new_Bg for item in sublist]
    return new_B

## swap two permutation entries and put the permutation into unique order
def swap_and_correct(permutation, src, tgt):
    permutation[src],permutation[tgt] = permutation[tgt],permutation[src]
    grouped = make_grouped(permutation)
    grouped = [item for sublist in grouped for item in sublist]
    return grouped

## make a swap that will move B in the direction of A
num_diffs = 0
def move_permutation_towards(B, A, debug=False):
    global num_diffs
    B = move_groups_to_match(B, A, debug)
    wrong_entries = group_differences(A, B)
    num_diffs = len(wrong_entries)

    # nothing to do, early out
    if len(wrong_entries) == 0:
        if debug:
            print("MPT: early out")
        return B

    if debug:
        print(f"MPT: checking {len(wrong_entries)} diffs: {wrong_entries}")

    # look for a group of three wrong entries that want to do the same thing
    entry_dict = dictify(wrong_entries)
    for k in entry_dict.keys():
        entry = entry_dict[k]
        if len(entry) == 3:
            if debug:
                print(f"MPT: found a triple swap at {k}: {entry_dict[k]}")
            (src, dst) = k
            # find the index of the one needed to complete the group
            # the value is the value in A[dst] that's not in B[src]
            # it's already in the destination group and may or may not need to move
            group_id = dst
            Ag = make_grouped(np.copy(A))
            Bg = make_grouped(np.copy(B))
            value = -1
            for c in range(4):
                if Ag[dst][c] not in Bg[src]:
                    value = Ag[dst][c]
                    if debug:
                        print(f"\tMPT: found the missing value {value} in A group {dst} offset {c}")
                    break
            assert(value != -1)

            # now find that value in B
            idx0 = np.where(B == value)[0][0]
            # find the index of the one this group doesn't need
            # it's a member of the group but not in the dict entry
            group_id = src
            for c in range(4):
                if B[group_id*4+c] not in entry_dict[k]:
                    if debug:
                        print(f"\tMPT: swapping {idx0} and {group_id*4+c}")
                    return swap_and_correct(B, idx0, group_id*4+c)

    # look for a group of two entries that are heading to the same place as another wrong entry
    victim_loner_pair = None
    for k in entry_dict.keys():
        entry = entry_dict[k]
        if len(entry) == 2:
            if debug:
                print(f"MPT: found a double swap at {k}: {entry_dict[k]}")
            (src, dst) = k
            # find a wrong entry whose dst is the same
            for k2 in entry_dict.keys():
                if k2 == k:
                    continue

                # k2 is a key whose value also belongs in stripe k2[1] (dst2)
                if dst == k2[1]:
                    if debug:
                        print(f"\tMPT: found a loner going in the same direction at {k2}: {entry_dict[k2][0]}")
                    # instead of moving these three to where they're headed, start merging them by moving the loner into the double

                    # look for a complement: something moving from src to src2
                    (src2, dst2) = k2
                    complement_key = (src, src2)
                    if complement_key in entry_dict:
                        complement = entry_dict[complement_key][0]
                        if debug:
                            print(f"\t\tMPT: found a complement to the loner:{complement}")
                        return swap_and_correct(B, np.where(B == entry_dict[k2][0])[0][0], np.where(B == complement)[0][0])
                    # didn't find a complement, choose one of the two in the src group that don't belong
                    elif victim_loner_pair is None:
                        for k3 in entry_dict.keys():
                            if k3 == k:
                                continue

                            if k3[0] == src: # found the victim
                                victim = entry_dict[k3][0]
                                if debug:
                                    print(f"\t\tMPT: found a victim for the double swap:{k3} -> {victim}")
                                victim_loner_pair = (victim, entry_dict[k2][0])
                                #return swap_and_correct(B, np.where(B == entry_dict[k2][0])[0][0], np.where(B == victim)[0][0])

    if victim_loner_pair is not None:
        if debug:
            print(f"\t\tMPT: couldn't find any complements for double swaps, so going with a loner to make a triple: {victim_loner_pair}")
        return swap_and_correct(B, np.where(B == victim_loner_pair[0])[0][0], np.where(B == victim_loner_pair[1])[0][0])

    # look for one swap that will correct two entries
    candidate_second = None
    for we in range(len(wrong_entries)):
        cur_entry = wrong_entries[we]
        #if debug:
        #    print(f"\tMPT: checking {cur_entry} for complement")
        for we2 in range(0,len(wrong_entries)):
            pos_swap = wrong_entries[we2]
            #if debug:
            #    print(f"\t\tMPT: is {pos_swap}?")
            if cur_entry[1] == pos_swap[2] and cur_entry[2] == pos_swap[1]:
                if debug:
                    print(f"\t\tfound complements: swapping {cur_entry} and {pos_swap}")
                return swap_and_correct(B, np.where(B == cur_entry[0])[0][0], np.where(B == pos_swap[0])[0][0])
            elif wrong_entries[0][2] == pos_swap[1]: # if pos_swap is currently where we[0] wants to go, keep it in mind
                candidate_second = pos_swap

    # fall back on picking the first one we come across
    assert(candidate_second is not None)
    if debug:
        print(f"No complement, swapping two entries: {wrong_entries[0]} {candidate_second}")
    return swap_and_correct(B, np.where(B == wrong_entries[0][0])[0][0], np.where(B == candidate_second[0])[0][0])

## find a shortest path from permutation A to B
def permutation_distance(A, B, matrix=None, magnitude_targets=None, debug=False, verbosity=0):
    global num_diffs
    swaps = 0
    debug = False

    swap_limit = int(math.pow(2,int(len(A)/4)-1))
    num_diffs = swap_limit
    common = []
    target_results = None
    if magnitude_targets is not None:
        assert matrix is not None
        cur_mag = sum_after_2_to_4(matrix[:,A])
        target_results = [(cur_mag, A) for i in range(len(magnitude_targets))]

    if verbosity > 0 and matrix is not None:
        print(f"swap {'0':>4} {sum_after_2_to_4(matrix[:, B]):>15.3f}")
        if verbosity > 5:
            print(f"swap {0:>4}, {make_grouped(A)} {make_grouped(B)}")

    while not np.all(np.array(A)-np.array(B) == np.zeros(np.array(A).shape)):
        cGroups = common_groups(A, B)
        for g in cGroups:
            common.append(g)
        A, B = remove_common_groups(A, B)
        if len(A) == 0:
            break

        B = move_permutation_towards(np.array(B), np.array(A), debug=debug)
        swaps += 1

        if matrix is not None:
            total_cur_permute = [c for c in B]

            for c in [item for sublist in common for item in sublist]:
                total_cur_permute.append(c)

            if verbosity > 0 or magnitude_targets is not None:
                cur_mag = sum_after_2_to_4(matrix[:,total_cur_permute])
                for i in range(len(target_results)):
                    result = target_results[i]
                    if abs(magnitude_targets[i] - result[0]) > abs(magnitude_targets[i] - cur_mag):
                        target_results[i] = (cur_mag, total_cur_permute)
                if verbosity > 0:
                    print(f"swap {swaps:>4} {cur_mag:>15.3f}")

        if verbosity > 5 or swaps > swap_limit:
            print(f"swap {swaps:>4}, {A} {B}, {num_diffs} diffs remain")

        # safety net
        if swaps > swap_limit+3:
            return swaps, target_results

    return swaps, target_results

