import numpy as np
import time
import ctypes
import subprocess
import os
import math

gpus_tested = False
gpus_found = 0
E = None
def use_gpu(initial_override = True): 
    global gpus_tested, gpus_found, E
    if not gpus_tested:
        if not initial_override:
            gpus_tested = True
            return False, None

        try:
            gpus_found = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
        except:
            gpus_found = 0

        gpus_tested = True
        
        if gpus_found > 0:
            # Method No.1
            #script_dir = os.path.dirname(__file__)
            ##print(script_dir)    # e.g. /opt/conda/lib/python3.8/site-packages/apex/contrib/sparsity/permutation_search_kernels
            #module_path = os.path.abspath(os.path.join(script_dir, '..', '..', '..', '..'))
            ##print(module_path)    # e.g. /opt/conda/lib/python3.8/site-packages
            #python_version_string = os.path.split(os.path.abspath(os.path.join(module_path, '..')))[1]
            ##print(python_version_string)    # e.g. python3.8
            #python_version_major_num = python_version_string.split('.')[0][-1]
            ##print(python_version_major_num)    # e.g. 3
            #python_version_minor_num = python_version_string.split('.')[1]
            ##print(python_version_minor_num)    # e.g. 8
            #cuda_kernel_name = 'permutation_search_cuda.cpython-{:}{:}-x86_64-linux-gnu.so'.format(python_version_major_num, python_version_minor_num)
            ##print(cuda_kernel_name)    # e.g. permutation_search_cuda.cpython-38-x86_64-linux-gnu.so
            #cuda_kernel_path = os.path.join(module_path, cuda_kernel_name)    # e.g. /opt/conda/lib/python3.8/site-packages/permutation_search_cuda.cpython-38-x86_64-linux-gnu.so

            # Method No.2, More robust
            import sysconfig
            lib_dir = sysconfig.get_config_var("LIBDEST")    # e.g. '/opt/conda/lib/python3.8'
            #lib_dir = sysconfig.get_config_var("BINLIBDEST")    # e.g. '/opt/conda/lib/python3.8'
            #print(lib_dir)
            ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')    # e.g. '.cpython-38-x86_64-linux-gnu.so'
            #print(ext_suffix)
            cuda_kernel_name = 'permutation_search_cuda' + ext_suffix
            #print(cuda_kernel_name)    # e.g. permutation_search_cuda.cpython-38-x86_64-linux-gnu.so
            cuda_kernel_path = os.path.join(lib_dir, 'site-packages', cuda_kernel_name)    # e.g. /opt/conda/lib/python3.8/site-packages/permutation_search_cuda.cpython-38-x86_64-linux-gnu.so

            print("[permutation_utilities] The permutation search CUDA kernel is from: {}".format(cuda_kernel_path))
            E = ctypes.cdll.LoadLibrary(cuda_kernel_path)

        print(f"Found {gpus_found} gpus and kernels in {E}")

    
    return gpus_found > 0 and E is not None, E

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
    use_cuda, E = use_gpu()
    if not use_cuda:
        start_time = time.perf_counter()
        for row in range(matrix.shape[0]):
            for col in range(0,matrix.shape[1],4):
                ix = np.argsort(np.abs(matrix[row,col:col+4]))
                cur_sum += abs(matrix[row,col+ix[2]])
                cur_sum += abs(matrix[row,col+ix[3]])
        np_elapsed = time.perf_counter() - start_time
    else:
        #E = ctypes.cdll.LoadLibrary("kernels/structured_sparsity.so")
        matrix = matrix.astype(np.float32)
        cuda_sum = np.zeros((1), dtype=np.float32)
        start_time = time.perf_counter()
        matrix_view = np.copy(matrix).flatten()
        blocks = max(int(matrix.shape[1]/4/2), 1)
        threads = min(max(math.ceil(matrix.shape[0]/4), 1), 1024)
        result = E._Z27run_subset_sum_after_2_to_4PfjjjjjjS_(ctypes.c_void_p(matrix_view.ctypes.data),
                                      ctypes.c_uint(matrix.shape[0]),
                                      ctypes.c_uint(matrix.shape[1]),
                                      ctypes.c_uint(0),
                                      ctypes.c_uint(matrix.shape[1]),
                                      ctypes.c_uint(blocks),
                                      ctypes.c_uint(threads),
                                      ctypes.c_void_p(cuda_sum.ctypes.data))
        cuda_elapsed = time.perf_counter() - start_time
        #print(cuda_sum, cuda_elapsed, cur_sum, np_elapsed, np_elapsed/cuda_elapsed)
        cur_sum = cuda_sum[0]
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

