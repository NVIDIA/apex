import torch
from torch.autograd import Variable
from torch.autograd.function import Function, once_differentiable
import apex._C

def check_contig_cuda(tensors, names):
    for tensor, name in zip(tensors, names):
        if not tensor.is_contiguous():
            raise RuntimeError(name+" with size {} is not contiguous"
                               .format(tensor.size()))
        if not tensor.is_cuda:
            raise RuntimeError(name+".is_cuda = False."
                               "Currently, only cuda tensors are supported.")

class Fused_Weight_Norm(Function):
    """
    Implements weight norm along a tensor's slowest dimension using fused kernel launches for
    the forward and backward pass.
    Accepts fp32 or fp16 input; the output type will match the input type.
    Within the kernels, all calculations are performed in fp32 for numerical stability, regardless
    of input/output precision.
    """

    @staticmethod
    def forward(ctx, input, g, dim=0):
        """
        :attr:`input` is assumed to be contiguous.
        :attr:`input` may be either float or half precision. 
        The precision of :attr:`output` will match the precision of :attr:`input`.
        A float copy of the L2 norm across each slow dimension
        is also created and saved for the backward pass.
        """
        # torch.cuda.nvtx.range_push("FusedNorm.forward, input.size() = {}"
        #                            .format(input.size()))

        check_contig_cuda((input,g),("input","g"))

        """
        This is ok, new() treats a torch.Size object properly.
        No need to unpack with an asterisk via new(*input.size()).
        """
        output = input.new(input.size()).contiguous()

        """
        For output with size (slow, faster, faster, ...fastest), we may want
        norms with size (slow, 1, 1, ...1), so that if you want retrieve norms 
        and apply the same normalizing factors to another Tensor "t" with the 
        same size as output, "t/norms" will broadcast each element of norms 
        across the corresponding slowest dim of t.
        """
        if dim == 0:
            norm_size = (output.size(0),) + (1,)*(output.dim() - 1)
        elif dim == output.dim() - 1:
            norm_size = (1,)*(output.dim() - 1) + (output.size(-1),)
        else:
            raise RuntimeError("Currently, Fused_Weight_Norm only supports first or last dimension.")

        norms = torch.cuda.FloatTensor(*norm_size).contiguous()
        """
        Beware:  If you call the following:
        norms = torch.cuda.FloatTensor(norm_size).contiguous()
        the constructor sees a tuple:
        FloatTensor( (output_size(0),1,1,...) )
        and creates a 1D tensor with values from the tuple:
        [output_size(0),1,1,...].
        """

        apex._C.weight_norm_fwd(output, norms, input, g, dim)
        ctx.save_for_backward(input, g)

        # save_for_backward can only save input or output tensors,
        # use ctx state to save the norms and dimension:
        ctx.norms = norms
        ctx.dim = dim

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """
        :attr:`grad_output` is assumed to be contiguous.
        :attr:`grad_output` may be either float or half precision.
        The precision of :attr:`grad_input` will match the precision of :attr:`grad_output`.
        """
        check_contig_cuda((grad_output), ("grad_output"))

        savedInput, savedg = ctx.saved_tensors
        savedNorms = ctx.norms

        # better safe than sorry
        grad_output_contig = grad_output.contiguous()

        grad_input = grad_output_contig.new(grad_output.size()).contiguous()
        grad_g = savedg.new(savedg.size()).contiguous()

        apex._C.weight_norm_bwd(grad_input, 
                                grad_g,
                                grad_output_contig, 
                                savedInput, 
                                savedg,
                                savedNorms,
                                ctx.dim)

        return grad_input, grad_g, None
