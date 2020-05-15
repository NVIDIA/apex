import torch

HALF = 'torch.cuda.HalfTensor'
FLOAT = 'torch.cuda.FloatTensor'
BFLOAT16 = 'torch.cuda.BFloat16Tensor'

DTYPES = [torch.half, torch.float]

DTYPES2 = [torch.bfloat16, torch.float]

ALWAYS_HALF = {torch.float: HALF,
               torch.half: HALF}
ALWAYS_BFLOAT16 = {torch.bfloat16: BFLOAT16,
                   torch.float: BFLOAT16}
ALWAYS_FLOAT = {torch.float: FLOAT,
                torch.half: FLOAT}
MATCH_INPUT = {torch.float: FLOAT,
               torch.half: HALF,
               torch.bfloat16: BFLOAT16}

def common_init(test_case):
    test_case.h = 64
    test_case.b = 16
    test_case.c = 16
    test_case.k = 3
    test_case.t = 10
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
