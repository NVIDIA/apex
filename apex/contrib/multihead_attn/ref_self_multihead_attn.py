import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd.variable  import Variable

class SelfAttentionLinears(torch.autograd.Function) :
    @staticmethod
    def forward(ctx, input, weights) :
        ctx.save_for_backward(input, weights)

	# Addmm mat1 is (n x m), mat2 is (m x p), and output is (n x p)
        qkv = torch.mm(input.view(input.size(0) * input.size(1), input.size(2)), weights.transpose(0,1))
        qkv=qkv.view(input.size(0), input.size(1), weights.size(0))
        return qkv.detach()

    @staticmethod
    def backward(ctx, qkv_grad) :
        input,weights = ctx.saved_tensors
       
        # Dgrad 
        input_grad = torch.addmm(qkv_grad.view(qkv_grad.size(0)*qkv_grad.size(1), qkv_grad.size(2)), qkv_grad.view(qkv_grad.size(0) * qkv_grad.size(1), qkv_grad.size(2)), weights, beta=0.0, alpha=1.0)
        input_grad = q_grad.view(qkv_grad.size(0)*qkv_grad.size(1), wights.size(1))
       
        # Wgrad 
        qkv_grad = qkv_grad.view(qkv_grad.size(0)*qkv_grad.size(1), qkv_grad.size(2)).transpose(0,1)
        weights_grad = torch.addmm(weights, qkv_grad, input, beta=0.0, alpha=1.0)
        return input_grad, weights_grad

self_attn_linears = SelfAttentionLinears.apply

class StridedBmm1Func(torch.autograd.Function) :
    @staticmethod
    def forward(ctx, input1, input2, scale) :
        s = Variable(torch.tensor([scale]))
        ctx.save_for_backward(input1, input2, s)

        #print("INPUT1", input1.size(), input1.stride())
        #print("INPUT2", input2.size(), input2.stride())
        output = torch.empty((input1.size(0),input1.size(1),input2.size(2)), dtype=input1.dtype, device=torch.device('cuda'))
        
        output = torch.baddbmm(output, input1, input2, out=output, beta=0.0, alpha=s[0])
        return output.detach()

    @staticmethod
    def backward(ctx, grad_output) :
        input1,input2,s = ctx.saved_tensors
        #print("INPUT1", input1.size(), input1.stride())
        #print("INPUT2", input2.size(), input2.stride())
        grad_input1 = torch.empty_like(input1)
        grad_input2 = torch.empty_like(input2)
        #print("GINPUT1", grad_input1.size(), grad_input1.stride())
        #print("GINPUT2", grad_input2.size(), grad_input2.stride())

        # Dgrad1
        grad_input1 = torch.baddbmm(grad_input1, grad_output, input2.transpose(1,2), out=grad_input1, beta=0.0, alpha=s[0])
        # Dgrad2
        grad_input2 = torch.baddbmm(grad_input2, grad_output.transpose(1,2), input1, out=grad_input2, beta=0.0, alpha=s[0])

        return grad_input1,grad_input2,None
        #return None,None,None,None,None

strided_bmm1 = StridedBmm1Func.apply

class StridedBmm2Func(torch.autograd.Function) :
     @staticmethod
     def forward(ctx, input1, input2) :
         ctx.save_for_backward(input1, input2)
         output = torch.empty((input1.size(1), input1.size(0), input2.size(2)), dtype=input1.dtype, device=torch.device('cuda')).transpose(1,0)
         
         output = torch.bmm(input1, input2, out=output)
         return output.detach()

     @staticmethod
     def backward(ctx, grad_output) :
         input1,input2 = ctx.saved_tensors
         grad_input1 = torch.empty_like(input1)
         grad_input2 = torch.empty_like(input2)

         # Dgrad1
         grad_input1 = torch.bmm(grad_output, input2.transpose(1,2), out=grad_input1)
         # Dgrad2
         grad_input2 = torch.bmm(input1.transpose(1,2), grad_output, out=grad_input2)
         return grad_input1,grad_input2

strided_bmm2 = StridedBmm2Func.apply

class FillPadding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, mask, num_heads):
        assert len(inputs.size()) == 3, "Tensor is not 3 dims"
        bsz     = int(inputs.size(0) / num_heads)
        tgt_len = inputs.size(1)
        src_len = inputs.size(2)
        inputs  = inputs.view(bsz, num_heads, tgt_len, src_len)
        inputs  = inputs + mask.unsqueeze(1).unsqueeze(2)
        inputs  = inputs.view(bsz * num_heads, tgt_len, src_len)
        return inputs.detach()

    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None

fill_padding_func = FillPadding.apply

class RefSelfMultiheadAttn(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """
    def __init__(self, embed_dim, num_heads, dropout=0., softmax_type='default', bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.softmax_type = softmax_type
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5
        self._mask = None
        self.in_proj_weight = Parameter(torch.Tensor(3*embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3*embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, is_training=True, mask_future_timesteps=False,
                key_padding_mask=None, incremental_state=None,
                need_weights=False, static_kv=False):

        seq_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        # self-attention
        qkv = self_attn_linears(query, self.in_proj_weight)

	# Slice out q,k,v from one big Input Linear outuput
        qkv = qkv.view(seq_len, bsz*self.num_heads, 3, self.head_dim)
        q   = qkv[:,:,0,:]
        k   = qkv[:,:,1,:]
        v   = qkv[:,:,2,:]

        attn_weights = strided_bmm1(q.transpose(0,1), k.transpose(0,1).transpose(1, 2), self.scaling)
        assert list(attn_weights.size()) == [bsz * self.num_heads, seq_len, seq_len]

        # only apply masking at training time (when incremental state is None)
	# Time Mask
        if mask_future_timesteps and incremental_state is None:
            assert query.size() == key.size(), \
                'mask_future_timesteps only applies to self-attention'
            attn_weights += self.buffered_mask(attn_weights).unsqueeze(0)

        # Padding Mask for Inputs
        if key_padding_mask is not None:
            attn_weights = fill_padding_func(attn_weights, key_padding_mask, self.num_heads)
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        attn = strided_bmm2(attn_weights, v.transpose(0,1))
        assert list(attn.size()) == [bsz * self.num_heads, seq_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(seq_len, bsz, embed_dim)

        attn = self.out_proj(attn)

        return attn, None
