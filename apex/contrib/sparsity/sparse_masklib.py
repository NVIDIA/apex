from math import floor
import sys
import torch
import numpy as np
import collections
from itertools import permutations, combinations


""" compute density (helper fn to compute % NNZs in a tensor) """
def fill(x):
    return float(x.nonzero().size(0))/torch.numel(x)

""" reshape matrix into m-dimensional vectors: (h,w) -> (hw/m, m) """
def reshape_1d(matrix, m):
    # If not a nice multiple of m, fill with zeroes.
    if matrix.shape[1] % m > 0:
        mat = torch.cuda.FloatTensor(matrix.shape[0], matrix.shape[1] + (m-matrix.shape[1]%m)).fill_(0)
        mat[:, :matrix.shape[1]] = matrix
        shape = mat.shape
        return mat.view(-1,m),shape
    else:
        return matrix.view(-1,m), matrix.shape

""" return all possible m:n patterns in a 1d vector """
valid_m4n2_1d_patterns = None
valid_m16n8_1d_patterns = None
valid_m32n8_1d_patterns = None
valid_m32n16_1d_patterns = None
def compute_valid_1d_patterns(m,n):
    # Early exit if patterns was already created.
    global valid_m4n2_1d_patterns
    global valid_m16n8_1d_patterns
    global valid_m32n8_1d_patterns
    global valid_m32n16_1d_patterns
    if m==4  and n==2 and valid_m4n2_1d_patterns  is not None: return valid_m4n2_1d_patterns
    if m==16  and n==8 and valid_m16n8_1d_patterns  is not None: return valid_m16n8_1d_patterns
    if m==32  and n==8 and valid_m32n8_1d_patterns  is not None: return valid_m32n8_1d_patterns
    if m==32  and n==16 and valid_m32n16_1d_patterns  is not None: return valid_m32n16_1d_patterns
    valid_patterns = []
    for i in list(combinations(range(0, m), n)):
        cur_pattern = np.zeros(m, dtype=np.int32)
        cur_pattern[list(i)] = 1
        valid_patterns.append(cur_pattern)
    valid_patterns = torch.Tensor(np.array(valid_patterns))
    # patterns = torch.zeros(m)
    # patterns[:n] = 1
    # valid_patterns = torch.Tensor(list(set(permutations(patterns.tolist()))))
    if m == 4  and n == 2: valid_m4n2_1d_patterns  = valid_patterns       
    if m == 16  and n == 8: valid_m16n8_1d_patterns  = valid_patterns       
    if m == 32  and n == 8: valid_m32n8_1d_patterns  = valid_patterns       
    if m == 32  and n == 16: valid_m32n16_1d_patterns  = valid_patterns       
    return valid_patterns

""" m:n 1d structured best """
def mn_1d_best(matrix, m, n):
    # Find all possible patterns.
    patterns = compute_valid_1d_patterns(m,n).cuda()
    # Find the best m:n pattern (sum of non-masked weights).
    mask = torch.cuda.IntTensor(matrix.shape).fill_(1).view(-1,m)
    mat,shape = reshape_1d(matrix,m)

    _dynamic, m = mat.shape
    factor = 256
    for start in range(0, _dynamic, factor):
        pmax = torch.argmax(torch.matmul(mat[start : start + factor].abs(), patterns.t()), dim=1)
        mask[start: start + factor] = patterns[pmax[:]]
    mask = mask.view(matrix.shape)

    return mask

""" m:n 1d structured pruning: greedy method to select mask """
def mn_1d_greedy(matrix, m, n):
    mat, shape = reshape_1d(matrix,m)
    mask = torch.cuda.IntTensor(matrix.shape).fill_(0).view(-1,m)

    values, indices = torch.abs(mat).topk(n, dim=1, largest=True)
    indexes = torch.arange(0, indices.shape[0], step=1, dtype=torch.long).view(-1, 1)

    mask[indexes, indices] = 1
    mask = mask.view(matrix.shape)

    return mask.cuda()


def m32n3_1d_best(mat, density):
    return mn_1d_best(mat, 32, 3)

def m32n4_1d_best(mat, density):
    return mn_1d_best(mat, 32, 4)

def m32n8_1d_best(mat, density):
    return mn_1d_best(mat, 32, 8)

def m32n16_1d_best(mat, density):
    return mn_1d_best(mat, 32, 16)

def m32n4_1d_greedy(mat, density):
    return mn_1d_greedy(mat, 32, 4)

def m32n16_1d_greedy(mat, density):
    return mn_1d_greedy(mat, 32, 16)

def m32n24_1d_best(mat, density):
    return mn_1d_best(mat, 32, 24)

def m16n8_1d_best(mat, density):
    return mn_1d_best(mat, 16, 8)

def m16n4_1d_best(mat, density):
    return mn_1d_best(mat, 16, 4)

def m8n4_1d_best(mat, density):
    return mn_1d_best(mat, 8, 4)

def m4n2_1d(mat, density):
    return mn_1d_best(mat, 4, 2)

def m4n2_1d_greedy(mat, density):
    return mn_1d_greedy(mat, 4, 2)

def unstructured(mat, density):
    mat_1d = mat.flatten()
    (m,) =  mat_1d.size()
    n = int(m *  density)

    mask = torch.cuda.IntTensor(mat_1d.shape).fill_(0)

    values, indices = torch.abs(mat_1d).topk(n, dim=0, largest=True)

    mask[indices] = 1;
    mask = mask.view(mat.shape)
    return mask

def unstructured_element_wise(mat, density):
    mat_1d = mat.flatten()
    (m,) =  mat_1d.size()
    n = int(m *  density)

    mask = torch.cuda.IntTensor(mat_1d.shape).fill_(0)

    values, indices = torch.abs(mat_1d).topk(n, dim=0, largest=True)

    mask[indices] = 1
    mask = mask.view(mat.shape)
    return mask

"""
  Below 2d-masking related code is targeted more for training (from scratch).
  2d-pruning of a weight tensor is done to accelerate DGRAD step during backprop
  phase of training algorithm. Acceleration comes from using SpMMA instructions in
  Tensor Cores of NVIDIA Ampere GPU Architecture 
  (note: this code does not do the acceleration, GPU kernels are required for this).
  1d pruning of weight tensor helps speed up FPROP step by pruning in 2:4 pattern
  along the horizontal (logical) direction.
  During DGRAD step, weight tensor is transposed. 2d pruning functions below, mask
  weight tensor such that their transposed versions are also 2:4 sparse along the
  horizontal (logical) direction. Thus, with 2d pruning, weight tensors are 
  2:4 sparse along row and column directions.
 """

""" m:n 2d structured pruning: greedy method to select mask """
def mn_2d_greedy(matrix, m, n):
    # Convert to numpy
    mat = matrix.cpu().detach().numpy()
    mask = np.ones(mat.shape, dtype=int)

    rowCount = int(mat.shape[0]/m) * m
    colCount = int(mat.shape[1]/m) * m
    for rowStartIdx in range(0, rowCount, m):
        rowEndIdx = rowStartIdx + m
        for colStartIdx in range(0, colCount, m):
            colEndIdx = colStartIdx + m
            matrixSub = np.absolute(np.squeeze(mat[rowStartIdx:rowEndIdx, colStartIdx:colEndIdx]))
            maskSub = np.squeeze(mask[rowStartIdx:rowEndIdx, colStartIdx:colEndIdx])
            maskSub.fill(0.0)
            matrixVecView = matrixSub.reshape(-1)
            maskVecView   = maskSub.reshape(-1)
            linearIdx = np.argsort(matrixVecView)
            matrixIdx = [(int(x/m), x % m) for x in linearIdx]
            rowCounter = collections.Counter()
            colCounter = collections.Counter()
            for currIdx in range(len(linearIdx) - 1, -1, -1):
                currMatrixEntry = matrixIdx[currIdx]
                if (rowCounter[currMatrixEntry[0]] == n) or (colCounter[currMatrixEntry[1]] == n):
                    continue
                #end if
                maskSub[currMatrixEntry[0], currMatrixEntry[1]] = 1.0
                rowCounter[currMatrixEntry[0]] += 1
                colCounter[currMatrixEntry[1]] += 1

    return torch.tensor(mask.cuda())

def m4n2_2d_greedy(mat, density):
    return mn_2d_greedy(mat, 4, 2)

""" return all possible m:n patterns in a mxn block. """
valid_m4n2_2d_patterns = None
def compute_valid_2d_patterns(m,n):
    # Early exit if patterns was already created.
    global valid_m4n2_2d_patterns
    if valid_m4n2_2d_patterns is not None: return valid_m4n2_2d_patterns

    patterns = torch.zeros(m)
    patterns[:n] = 1
    patterns = list(set(permutations(patterns.tolist())))
    patterns = patterns + patterns
    patterns = torch.Tensor(list(set(permutations(patterns,m))))

    valid = ((patterns.sum(dim=1) <= n).sum(dim=1) == m).nonzero().view(-1)
    valid_patterns = torch.Tensor(valid.shape[0],m,m)
    valid_patterns[:] = patterns[valid[:]]

    if m == 4  and n == 2: valid_m4n2_2d_patterns  = valid_patterns
    return valid_patterns

""" m:n 2d structured pruning: exhaustive method to select best mask """
def mn_2d_best(matrix, m, n):
    # Find all possible patterns.
    patterns = compute_valid_2d_patterns(m,n).cuda()

    # Find the best m:n pattern (sum of non-masked weights).
    mask = torch.cuda.IntTensor(matrix.shape).fill_(1)
    mat = reshape_2d(matrix,m,m).abs()
    pmax = torch.argmax(torch.matmul(mat,patterns.view(patterns.shape[0],m*m).t()), dim=2)

    # Copy best m:n patterns into mask.
    mat = mat.view(mat.shape[0]*mat.shape[1],-1)
    pmax = pmax.view(pmax.shape[0]*pmax.shape[1]).unsqueeze(1).expand(-1,mat.shape[1])
    patterns = patterns.view(patterns.shape[0],patterns.shape[1]*patterns.shape[2])
    mat = torch.gather(patterns,0,pmax)
    mat = reshape_2d_inv(mat.view(matrix.shape[0]//m,matrix.shape[1]//m,m,m))
    mask.copy_(mat.type(mask.type()))
    return mask

def m4n2_2d_best(mat, density):
    return mn_2d_best(mat, 4, 2)


def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return  torch.stack(list(tuple_of_tensors), dim=0)

def tensor_block_partition(matrix, m, n):
    if matrix.shape[0] % m > 0 or matrix.shape[1] % n > 0:
        print("matrix shape must be divisible by m and n, try to extend")
        m_pad = 0 if matrix.shape[0] % m == 0 else m - matrix.shape[0] % m
        n_pad = 0 if matrix.shape[1] % n == 0 else n - matrix.shape[1] % n
        mat = torch.nn.functional.pad(matrix, (0, n_pad, 0, m_pad))
        shape = mat.shape
        first_tile = tuple_of_tensors_to_tensor(torch.split(mat, m, 0))
        second_tile = tuple_of_tensors_to_tensor(torch.split(first_tile, n, 2))
        mat = second_tile
        return mat, shape
    else:
        first_tile = tuple_of_tensors_to_tensor(torch.split(matrix, m, 0))
        second_tile = tuple_of_tensors_to_tensor(torch.split(first_tile, n, 2))
        mat = second_tile
        return mat, matrix.shape

def unstructured_vector_wise(matrix, density, v):
    mat = matrix.view(-1, v)
    (m, v) =  mat.shape;
    n = int(m * density)

    mask = torch.cuda.IntTensor(mat.shape).fill_(0)
    mat_reduce = torch.sum(mat, dim=-1)
    values, indices = torch.abs(mat_reduce).topk(n, dim=0, largest=True)

    mask[indices, :] = 1;
    mask = mask.view(matrix.shape)
    return mask

def unstructured_v4(matrix, density):
    return unstructured_vector_wise(matrix, density, 4)

def unstructured_v32(matrix, density):
    return unstructured_vector_wise(matrix, density, 32)

def unstructured_v64(matrix, density):
    return unstructured_vector_wise(matrix, density, 64)

def mnv_vector_wise_greedy(matrix, m, n, v):
    '''
        m -> length
        n -> width
        v -> valid vector
    '''
    # split into tensor blocks
    raw_shape = matrix.shape
    # print("raw shape ", raw_shape)
    mat, pad_shape = tensor_block_partition(matrix, v, m)
    # print("extend shape ", pad_shape)
    mask = torch.cuda.IntTensor(mat.shape).fill_(0)
    mat_abs = torch.abs(mat)
    mat_reduce = torch.sum(mat_abs, dim=2)

    values, indices = torch.topk(mat_reduce, n, dim=2, largest=True)

    # todo: this can be optimize, currently is slow.
    for d0 in range(0, indices.shape[0]):
        for d1 in range(0, indices.shape[1]):
                mask[d0][d1][:, indices[d0][d1]] = 1
    # mask[0, 0, 0, indices] = 1;
    mask = torch.cat(tuple(mask), 2).view(pad_shape)
    mask = mask[0:raw_shape[0], 0:raw_shape[1]]
    return mask.cuda()

def m4n2v4_2d_greedy(mat, density):
    return mnv_vector_wise_greedy(mat, 4, 2, 4)

def m32n16v4_2d_greedy(mat, density):
    return mnv_vector_wise_greedy(mat, 32, 16, 4)

def m32n8v4_2d_greedy(mat, density):
    return mnv_vector_wise_greedy(mat, 32, 8, 4)

def m32n4v4_2d_greedy(mat, density):
    return mnv_vector_wise_greedy(mat, 32, 4, 4)

def m32n3v4_2d_greedy(mat, density):
    return mnv_vector_wise_greedy(mat, 32, 3, 4)

def m4n2v32_2d_greedy(mat, density):
    return mnv_vector_wise_greedy(mat, 4, 2, 32)

def m32n16v32_2d_greedy(mat, density):
    return mnv_vector_wise_greedy(mat, 32, 16, 32)

def m32n8v32_2d_greedy(mat, density):
    return mnv_vector_wise_greedy(mat, 32, 8, 32)

def m32n4v32_2d_greedy(mat, density):
    return mnv_vector_wise_greedy(mat, 32, 4, 32)

def m32n3v32_2d_greedy(mat, density):
    return mnv_vector_wise_greedy(mat, 32, 3, 32)

def m4n2v64_2d_greedy(mat, density):
    return mnv_vector_wise_greedy(mat, 4, 2, 64)

def m32n16v64_2d_greedy(mat, density):
    return mnv_vector_wise_greedy(mat, 32, 16, 64)

def m32n8v64_2d_greedy(mat, density):
    return mnv_vector_wise_greedy(mat, 32, 8, 64)

def m32n4v64_2d_greedy(mat, density):
    return mnv_vector_wise_greedy(mat, 32, 4, 64)

def m32n3v64_2d_greedy(mat, density):
    return mnv_vector_wise_greedy(mat, 32, 3, 64)

def unstructured_block_wise(matrix, density, bh, bw):
    # split into tensor blocks
    mat, shape = tensor_block_partition(matrix, bh, bw)
    (bm, bn, bh, bw) = mat.shape
    mask = torch.cuda.IntTensor(mat.shape).fill_(0)
    mat_abs = torch.abs(mat)
    mat_reduce = torch.sum(torch.sum(mat_abs, dim=-1), dim=-1)
    mat_reduce_recover = torch.stack(tuple(mat_reduce), dim=-1).view(-1)
    # n = int(bm * bn * density)
    n = int(bm * bn *  density)
    values, indices = torch.topk(mat_reduce_recover, n, dim=-1, largest=True)
    # todo: this can be optimize, currently is slow.
    for d0 in indices:
        mask[d0 // bn][d0 % bn][:][:] = 1
    # mask[0, 0, 0, indices] = 1;
    mask = torch.cat(tuple(mask), 2).view(matrix.shape)

    return mask.cuda()


def unstructured_b4(matrix, density):
    return unstructured_block_wise(matrix, density, 4, 4)


def mnb_block_wise_greedy(matrix, m, n, bh, bw):
    '''
        m -> length
        n -> width
        v -> valid vector
    '''
    # split into tensor blocks
    raw_shape = matrix.shape
    print("raw shape ", raw_shape)
    mat, pad_shape = tensor_block_partition(matrix, bh, bw * m)
    print("extend shape ", pad_shape)
    mask = torch.cuda.IntTensor(mat.shape).fill_(0)
    mat_abs = torch.abs(mat)
    mat_reduce = torch.sum(mat_abs, dim=2)
    mat_reduce_bw = tuple_of_tensors_to_tensor(torch.split(mat_reduce, bw, dim=-1))
    mat_reduce_bw_reduce = torch.sum(mat_reduce_bw, dim=-1)
    mat_reduce_bw_reduce_recover = torch.stack(tuple(mat_reduce_bw_reduce), dim=-1).view(mask.shape[0], mask.shape[1], m)
    # print(mat_reduce_bw_reduce)
    # print(third_tile)    
    # print(mask.shape)
    values, indices = torch.topk(mat_reduce_bw_reduce_recover, n, dim=2, largest=True)
    # todo: this can be optimize, currently is slow.
    for d0 in range(0, indices.shape[0]):
        for d1 in range(0, indices.shape[1]):
                for _bw in range(0, bw):
                    mask[d0][d1][:, indices[d0][d1]*bw+_bw] = 1
    # mask[0, 0, 0, indices] = 1;
    mask = torch.cat(tuple(mask), 2).view(pad_shape)
    mask = mask[0:raw_shape[0], 0:raw_shape[1]]
    return mask.cuda()

def m4n2b4_2d_greedy(mat, density):
    return mnb_block_wise_greedy(mat, 4, 2, 4, 4)

def m32n3b4_2d_greedy(mat, density):
    return mnb_block_wise_greedy(mat, 32, 3, 4, 4)

def m32n4b4_2d_greedy(mat, density):
    return mnb_block_wise_greedy(mat, 32, 4, 4, 4)

def m32n8b4_2d_greedy(mat, density):
    return mnb_block_wise_greedy(mat, 32, 8, 4, 4)

def m32n16b4_2d_greedy(mat, density):
    return mnb_block_wise_greedy(mat, 32, 16, 4, 4)

""" returns a sparse mask """
def create_mask(tensor, pattern="m4n2_1d", density=0.5):
    # Reshape tensor and mask.
    shape = tensor.shape
    ttype = tensor.type()
    t = tensor.float().contiguous()

    # 1d-tensor
    if len(shape) == 1:
        t = t.view(1, shape[0])
        func = getattr(sys.modules[__name__], pattern, None)
        mask = func(t, density)
        return mask.view(shape).type(ttype)
    # 2d-tensor (K, C)
    elif len(shape) == 2:
        # linear
        t = t.view(shape[0], shape[1])
        func = getattr(sys.modules[__name__], pattern, None)
        mask = func(t, density)
        return mask.view(shape).type(ttype)
    # 3d-tensor (K, C, R)
    elif len(shape) == 3:
        # 1d convs
        t = t.permute(0,2,1).contiguous().view(shape[0]*shape[2], shape[1])
        func = getattr(sys.modules[__name__], pattern, None)
        mask = func(t, density)
        mask = mask.view(shape[0], shape[2], shape[1]).permute(0,2,1).contiguous()     
        return mask.view(shape).type(ttype)
    # 4d-tensor (K, C, R, S)
    elif len(shape) == 4:
        """
        # transformers (bmm)
        t = t.view(shape[0]*shape[1]*shape[2], shape[3])
        func = getattr(sys.modules[__name__], pattern, None)
        mask = func(t, density)
        return mask.view(shape).type(ttype)
        """
        # 2d convs
        t = t.permute(2,3,0,1).contiguous().view(shape[2]*shape[3]*shape[0], shape[1])
        func = getattr(sys.modules[__name__], pattern, None)
        mask = func(t, density)
        mask = mask.view(shape[2], shape[3], shape[0], shape[1]).permute(2,3,0,1).contiguous()      
        return mask.view(shape).type(ttype)

