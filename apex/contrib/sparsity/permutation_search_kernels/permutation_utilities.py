import numpy as np
import time
import ctypes
import subprocess
import os
import math

gpus_tested = False
gpus_found = 0
kernels_found = True
try:
    import permutation_search_cuda as permutation_search_cuda_kernels
    print(f"Found permutation search CUDA kernels")
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
    #matrix = np.copy(matrix)
    cur_sum = 0.0
    use_cuda = use_gpu()
    if not use_cuda:
        start_time = time.perf_counter()
        for row in range(matrix.shape[0]):
            for col in range(0,matrix.shape[1],4):
                ix = np.argsort(np.abs(matrix[row,col:col+4]))
                cur_sum += abs(matrix[row,col+ix[2]])
                cur_sum += abs(matrix[row,col+ix[3]])
        np_elapsed = time.perf_counter() - start_time
    else:
        matrix = matrix.astype(np.float32)
        cuda_sum = np.zeros((1), dtype=np.float32)
        start_time = time.perf_counter()
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
        cuda_elapsed = time.perf_counter() - start_time
        #print(cuda_sum, cuda_elapsed, cur_sum, np_elapsed, np_elapsed/cuda_elapsed)
        cur_sum = sum_view[0]
    return cur_sum

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

##############################################################################################
# permutation utilities
##############################################################################################

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

