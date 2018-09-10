import torch

HALF = 'torch.cuda.HalfTensor'
FLOAT = 'torch.cuda.FloatTensor'

DTYPES = [torch.half, torch.float]

ALWAYS_HALF = {torch.float: HALF,
               torch.half: HALF}
ALWAYS_FLOAT = {torch.float: FLOAT,
                torch.half: FLOAT}
MATCH_INPUT = {torch.float: FLOAT,
               torch.half: HALF}

def common_init(test_case):
    test_case.h = 64
    test_case.b = 16
    test_case.c = 16
    test_case.k = 3
    test_case.t = 10
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
