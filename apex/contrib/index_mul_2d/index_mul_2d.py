import torch

import fused_index_mul_2d

class IndexMul2d_(torch.autograd.Function):
    '''
    Currently only support index in dimension 0 with a 2-dimension tensor.
    The shape of indexed in1 must be same with in2. Now this kernel does not support broadcast.
    The datatype must be float32 or float16.
    '''
    @staticmethod
    def forward(ctx, in1: torch.Tensor, in2: torch.Tensor, idx1: torch.Tensor) -> torch.Tensor:
        assert in2.size(0) == idx1.size(0)
        if ((in1.dtype != torch.float32 and in1.dtype != torch.half) or in2.dtype != in1.dtype):
            raise RuntimeError("input1'dtype and input2's dtype must be fp32 or fp16. And input type must be same")
        if (in1.dim() != 2 or in2.dim() != 2):
            raise RuntimeError("in1 and in2 must be 2-dimension tensor.")
        if (idx1.dim() != 1):
            raise RuntimeError("idx1 must be 1-dimension tensor.")

        if not in1.is_contiguous():
            in1 = in1.contiguous()
        if not in2.is_contiguous():
            in2 = in2.contiguous()
        if not idx1.is_contiguous():
            idx1 = idx1.contiguous()

        assert in1.is_contiguous()
        assert in2.is_contiguous()
        assert idx1.is_contiguous()

        out = torch.empty_like(in2)

        if (in1.dtype == torch.float32):
            fused_index_mul_2d.float_forward(
                out,
                in1,
                in2,
                idx1)
        elif (in1.dtype == torch.half):
            fused_index_mul_2d.half_forward(
                out,
                in1,
                in2,
                idx1)

        ctx.for_backwards = (in1, in2, idx1)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        
        in1, in2, idx1 = ctx.for_backwards
       
        grad_in1, grad_in2 = index_mul_2d_backward(in1, in2, idx1, grad_out)

        return grad_in1, grad_in2, None


class IndexMul2dBackward_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, in1: torch.Tensor, in2: torch.Tensor, idx1: torch.Tensor,
                grad_out: torch.Tensor) -> torch.Tensor:
        if not in1.is_contiguous():
            in1 = in1.contiguous()
        if not in2.is_contiguous():
            in2 = in2.contiguous()
        if not idx1.is_contiguous():
            idx1 = idx1.contiguous()
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        assert in1.is_contiguous()
        assert in2.is_contiguous()
        assert idx1.is_contiguous()
        assert grad_out.is_contiguous()

        grad_in1 = torch.zeros_like(in1)
        grad_in2 = torch.empty_like(in2)

        if (in1.dtype == torch.float32):
            fused_index_mul_2d.float_backward(
                grad_in1,
                grad_in2,
                grad_out,
                in1,
                in2,
                idx1)
        elif (in1.dtype == torch.half):
            fused_index_mul_2d.half_backward(
                grad_in1,
                grad_in2,
                grad_out,
                in1,
                in2,
                idx1)            
            
        ctx.for_backwards = (in1, in2, idx1, grad_out)
        return grad_in1, grad_in2

    @staticmethod
    def backward(ctx, grad_grad_in1, grad_grad_in2):
        if not grad_grad_in1.is_contiguous():
            grad_grad_in1 = grad_grad_in1.contiguous()
        if not grad_grad_in2.is_contiguous():
            grad_grad_in2 = grad_grad_in2.contiguous()
        
        assert grad_grad_in1.is_contiguous()
        assert grad_grad_in2.is_contiguous()

        in1, in2, idx1, grad_out = ctx.for_backwards

        grad_in1 = torch.zeros_like(in1)
        grad_in2 = torch.empty_like(in2)
        grad_grad_out = torch.empty_like(grad_out)

        if (in1.dtype == torch.float32):
            fused_index_mul_2d.float_backward_backward(
                grad_grad_out,
                grad_in1,
                grad_in2,
                grad_out,
                grad_grad_in1,
                grad_grad_in2,
                in1,
                in2,
                idx1)
        elif (in1.dtype == torch.half):
            fused_index_mul_2d.half_backward_backward(
                grad_grad_out,
                grad_in1,
                grad_in2,
                grad_out,
                grad_grad_in1,
                grad_grad_in2,
                in1,
                in2,
                idx1)            

        return grad_in1, grad_in2, None, grad_grad_out

index_mul_2d = IndexMul2d_.apply
index_mul_2d_backward = IndexMul2dBackward_.apply

