import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd.variable  import Variable

class PySelfAttnFunc(torch.autograd.Function) :
    @staticmethod
    def forward(ctx, use_time_mask, is_training, heads, scale, inputs, input_weights, output_weights, pad_mask, dropout_prob) :
        heads_t        = Variable(torch.tensor([heads]))
        scale_t        = Variable(torch.tensor([scale]))
        dropout_prob_t = Variable(torch.tensor([dropout_prob]))
        null_tensor    = torch.tensor([])
        use_mask       = (pad_mask is not None)
        head_dim       = inputs.size(2) // heads
       
        # Input Linear GEMM
        input_lin_results = torch.mm(inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2)), input_weights.transpose(0,1))
        input_lin_results=input_lin_results.view(inputs.size(0), inputs.size(1), input_weights.size(0))
	
        # Slice out q,k,v from one big Input Linear outuput (should only impact meta data, no copies!)
        input_lin_results = input_lin_results.view(inputs.size(0), inputs.size(1)*heads, 3, head_dim)
        queries = input_lin_results[:,:,0,:]
        keys    = input_lin_results[:,:,1,:]
        values  = input_lin_results[:,:,2,:]
       
        # Matmul1 Batched GEMMs
        matmul1_results = torch.empty((queries.size(1),queries.size(0),keys.size(2)), dtype=queries.dtype, device=torch.device('cuda'))
        matmul1_results = torch.baddbmm(matmul1_results, queries.transpose(0,1), keys.transpose(0,1).transpose(1,2), out=matmul1_results, beta=0.0, alpha=scale_t[0])
        softmax_results = F.softmax(matmul1_results, dim=-1)

        # Dropout
        if is_training :
            dropout_results,dropout_mask = torch._fused_dropout(softmax_results, p=dropout_prob_t[0])
        else :
            dropout_results = softmax_results
            dropout_mask    = null_tensor
       
        # Matmul2 Batched GEMMs
        matmul2_results = torch.empty((dropout_results.size(1), dropout_results.size(0), values.size(2)), dtype=dropout_results.dtype, device=torch.device('cuda')).transpose(1,0)
        matmul2_results = torch.bmm(dropout_results, values.transpose(0,1), out=matmul2_results)
        matmul2_results = matmul2_results.transpose(0, 1).contiguous().view(inputs.size(0), inputs.size(1), inputs.size(2))
       
        # Output Linear GEMM
        outputs = torch.mm(inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2)), output_weights.transpose(0,1))
        outputs = outputs.view(inputs.size(0), inputs.size(1), output_weights.size(0))

        ctx.save_for_backward(heads_t,                                  \
                              scale_t,                                  \
                              matmul2_results,                          \
                              dropout_results,                          \
                              softmax_results,                          \
                              input_lin_results,                        \
                              inputs,                                   \
                              input_weights,                            \
                              output_weights,                           \
                              dropout_mask,                             \
                              dropout_prob_t)
        
        return outputs.detach()
    
    @staticmethod
    def backward(ctx, output_grads) :
        heads_t,                                                        \
        scale_t,                                                        \
        matmul2_results,                                                \
        dropout_results,                                                \
        softmax_results,                                                \
        input_lin_results,                                              \
        inputs,                                                         \
        input_weights,                                                  \
        output_weights,                                                 \
        dropout_mask,                                                   \
        dropout_prob_t          = ctx.saved_tensors
        
        head_dim                = inputs.size(2) // heads_t[0]
        
        # Slice out q,k,v from one big Input Linear outuput (should only impact meta data, no copies!)
        input_lin_results       = input_lin_results.view(inputs.size(0), inputs.size(1)*heads_t[0], 3, head_dim)
        queries                 = input_lin_results[:,:,0,:]
        keys                    = input_lin_results[:,:,1,:]
        values                  = input_lin_results[:,:,2,:]
        
        # Slice out q,k,v from one big set of gradients entering the input linear's bprop  (should only impact meta data, no copies!)
        input_lin_results_grads = torch.empty_like(input_lin_results)
        queries_grads           = input_lin_results_grads[:,:,0,:]
        keys_grads              = input_lin_results_grads[:,:,1,:]
        values_grads            = input_lin_results_grads[:,:,2,:]
        
        # Output Linear GEMM - DGRAD
        output_lin_grads = torch.mm(output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)), output_weights)
        output_lin_grads = output_lin_grads.view(output_grads.size(0), output_grads.size(1), output_weights.size(1))
        # Output Linear GEMM - WGRAD
        output_weight_grads = torch.mm(output_grads.view(output_grads.size(0) * output_grads.size(1), output_grads.size(2)).transpose(0,1), 
                                       matmul2_results.view(matmul2_results.size(0) * matmul2_results.size(1), matmul2_results.size(2)))
        
        output_lin_grads = output_lin_grads.view(inputs.size(0), inputs.size(1)*heads_t[0], head_dim).transpose(0,1)

        # Matmul2 - DGRAD1
        matmul2_dgrad1 = torch.bmm(output_lin_grads, values.transpose(0,1).transpose(1,2))
        # Matmul2 - DGRAD2
        values_grads   = torch.bmm(dropout_results.transpose(1,2), output_lin_grads, out=values_grads.transpose(0,1))

        # Mask and Scaling for Dropout
        dropout_grads = torch._masked_scale(matmul2_dgrad1, dropout_mask, dropout_prob_t[0])

        # Softmax Grad
        softmax_grads = torch._softmax_backward_data(dropout_grads, softmax_results, -1, softmax_results)

        # Matmul1 - DGRAD1
        queries_grads = torch.baddbmm(queries_grads.transpose(0,1), softmax_grads, keys.transpose(0,1), 
                                      out=queries_grads.transpose(0,1), beta=0.0, alpha=scale_t[0])
        # Matmul1 - DGRAD2
        keys_grads    = torch.baddbmm(keys_grads.transpose(0,1), softmax_grads.transpose(1,2), queries.transpose(0,1), 
                                      out=keys_grads.transpose(0,1), beta=0.0, alpha=scale_t[0])

        # Input Linear GEMM - DGRAD
        input_lin_results_grads = input_lin_results_grads.view(inputs.size(0)*inputs.size(1), heads_t[0]*3*head_dim)
        input_grads = torch.mm(input_lin_results_grads, input_weights)
        input_grads = input_grads.view(inputs.size(0), inputs.size(1), inputs.size(2))
        # Input Linear GEMM - WGRAD
        input_weight_grads = torch.mm(input_lin_results_grads.transpose(0,1), inputs.view(inputs.size(0)*inputs.size(1), inputs.size(2)))

        return None, None, None, None, input_grads, input_weight_grads, output_weight_grads, None, None

py_self_attn_func = PySelfAttnFunc.apply

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

class PySelfMultiheadAttn(nn.Module):
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
        self.in_proj_weight  = Parameter(torch.Tensor(3*embed_dim, embed_dim))
        self.out_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3*embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.reset_parameters()
        
        self.attn_func = py_self_attn_func

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
    
    def forward(self, query, key, value, is_training=True, mask_future_timesteps=False,
                key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        if incremental_state is not None :
            outputs = self.attn_func(mask_future_timesteps, is_training, self.num_heads, self.scaling, query, self.in_proj_weight, self.out_proj_weight, None, self.dropout)
        else :
            outputs = self.attn_func(mask_future_timesteps, is_training, self.num_heads, self.scaling, query, self.in_proj_weight, self.out_proj_weight, key_padding_mask, self.dropout)

        return outputs,None
